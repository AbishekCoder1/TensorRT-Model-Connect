// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-ENG-CPP-03
// Architecture:   ARCH-MOD-001
// Unit Design:    UD-TRT-CORE-01
// Intent:         Engine lifecycle validation against a fake backend-neutral
//                 module with known tensor names.
// Preconditions:  No TensorRT SDK required.
// Postconditions: has_io_tensor returns correct bool, has_all_required_tensors
//                 validates base+layer tensors.
// =============================================================================

#include "runtime/core/trt_engine_lifecycle.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

class FakeModule final : public trtf::ITrtModule {
  public:
    explicit FakeModule(std::vector<std::string> tensor_names) {
        for (auto& name : tensor_names)
            names_.insert(std::move(name));
    }

    trtf::TensorMap forward(const trtf::TensorMap&) override { return {}; }
    trtf::DeviceTensorMap forward_device(const trtf::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtf::DeviceTensorMap&) override {}
    void forward_async(const trtf::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtf::TensorInfo> input_info() const override { return {}; }
    std::vector<trtf::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override { return names_.count(name) != 0; }
    bool has_output(const std::string& name) const override { return names_.count(name) != 0; }
    trtf::DType tensor_dtype(const std::string&) const override { return trtf::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtf::ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    std::unordered_set<std::string> names_;
};

std::vector<std::string> make_required_tensor_names(int32_t num_layers,
                                                    bool include_position_input) {
    std::vector<std::string> names;
    names.emplace_back("token_id");
    names.emplace_back("attention_mask");
    names.emplace_back("logits");
    if (include_position_input)
        names.emplace_back("position_id");
    for (int32_t i = 0; i < num_layers; ++i) {
        names.emplace_back(trtf::layer_tensor_name("cache_k", i));
        names.emplace_back(trtf::layer_tensor_name("cache_v", i));
        names.emplace_back(trtf::layer_tensor_name("present_k", i));
        names.emplace_back(trtf::layer_tensor_name("present_v", i));
    }
    return names;
}

void remove_name(std::vector<std::string>& names, const std::string& target) {
    names.erase(std::remove(names.begin(), names.end(), target), names.end());
}

trtf::DecoderStepEngine make_decoder_step_engine(FakeModule& module, int32_t num_layers,
                                                 bool requires_position_input) {
    trtf::DecoderStepEngine engine;
    engine.module = &module;
    engine.num_layers = num_layers;
    engine.requires_position_input = requires_position_input;

    engine.cache_k_input_names.reserve(static_cast<std::size_t>(num_layers));
    engine.cache_v_input_names.reserve(static_cast<std::size_t>(num_layers));
    engine.present_k_output_names.reserve(static_cast<std::size_t>(num_layers));
    engine.present_v_output_names.reserve(static_cast<std::size_t>(num_layers));
    for (int32_t i = 0; i < num_layers; ++i) {
        engine.cache_k_input_names.push_back(trtf::layer_tensor_name("cache_k", i));
        engine.cache_v_input_names.push_back(trtf::layer_tensor_name("cache_v", i));
        engine.present_k_output_names.push_back(trtf::layer_tensor_name("present_k", i));
        engine.present_v_output_names.push_back(trtf::layer_tensor_name("present_v", i));
    }
    return engine;
}

void test_has_io_tensor_true_false() {
    FakeModule module({"token_id", "attention_mask"});
    check(trtf::has_io_tensor(module, "token_id"), "has_io_tensor returns true for existing name");
    check(!trtf::has_io_tensor(module, "logits"), "has_io_tensor returns false for missing name");
    check(!trtf::has_io_tensor(module, ""), "has_io_tensor returns false for empty name");
}

void test_has_all_required_tensors_false_when_missing_base_tensors() {
    std::vector<std::string> names = make_required_tensor_names(/*num_layers=*/2, true);
    remove_name(names, "attention_mask");
    FakeModule module(std::move(names));
    trtf::DecoderStepEngine engine = make_decoder_step_engine(module, /*num_layers=*/2, true);
    check(!trtf::has_all_required_tensors(engine),
          "has_all_required_tensors fails when base tensor is missing");
}

void test_has_all_required_tensors_true_with_all_base_and_layer_tensors() {
    FakeModule module(make_required_tensor_names(/*num_layers=*/2,
                                                 /*include_position_input=*/true));
    trtf::DecoderStepEngine engine = make_decoder_step_engine(module, /*num_layers=*/2, true);
    check(trtf::has_all_required_tensors(engine),
          "has_all_required_tensors passes when all tensors are present");
}

void test_has_all_required_tensors_requires_position_input_branch() {
    FakeModule no_position_module(make_required_tensor_names(
        /*num_layers=*/1, /*include_position_input=*/false));
    trtf::DecoderStepEngine no_position_required =
        make_decoder_step_engine(no_position_module, /*num_layers=*/1, false);
    check(trtf::has_all_required_tensors(no_position_required),
          "requires_position_input=false does not require position tensor");

    FakeModule missing_position_module(make_required_tensor_names(
        /*num_layers=*/1, /*include_position_input=*/false));
    trtf::DecoderStepEngine position_required_missing =
        make_decoder_step_engine(missing_position_module, /*num_layers=*/1, true);
    check(!trtf::has_all_required_tensors(position_required_missing),
          "requires_position_input=true fails without position tensor");
}

void test_has_all_required_tensors_missing_per_layer_tensor_fails() {
    std::vector<std::string> names = make_required_tensor_names(/*num_layers=*/3, true);
    remove_name(names, trtf::layer_tensor_name("present_v", 1));
    FakeModule module(std::move(names));
    trtf::DecoderStepEngine engine = make_decoder_step_engine(module, /*num_layers=*/3, true);
    check(!trtf::has_all_required_tensors(engine),
          "has_all_required_tensors fails when a per-layer tensor is missing");
}

} // namespace

int main() {
    test_has_io_tensor_true_false();
    test_has_all_required_tensors_false_when_missing_base_tensors();
    test_has_all_required_tensors_true_with_all_base_and_layer_tensors();
    test_has_all_required_tensors_requires_position_input_branch();
    test_has_all_required_tensors_missing_per_layer_tensor_fails();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All trt_engine_lifecycle fake-engine tests passed.\n";
    return 0;
}
