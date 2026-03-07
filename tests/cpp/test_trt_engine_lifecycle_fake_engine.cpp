#include "runtime/trt/core/trt_engine_lifecycle.h"

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

#if TRTF_HAS_TRT

#if NV_TENSORRT_MAJOR >= 10

class FakeCudaEngineImpl final : public nvinfer1::apiv::VCudaEngine {
public:
    explicit FakeCudaEngineImpl(std::vector<std::optional<std::string>> io_tensor_names)
        : mIOTensorNames(std::move(io_tensor_names))
    {
    }

    nvinfer1::ICudaEngine* getPImpl() noexcept override
    {
        return nullptr;
    }

    int32_t getTensorRTVersion() const noexcept
    {
        return NV_TENSORRT_VERSION;
    }

    int32_t getNbLayers() const noexcept override
    {
        return 0;
    }

    nvinfer1::IHostMemory* serialize() const noexcept override
    {
        return nullptr;
    }

    nvinfer1::IExecutionContext* createExecutionContext(
        nvinfer1::ExecutionContextAllocationStrategy) noexcept override
    {
        return nullptr;
    }

    nvinfer1::IExecutionContext* createExecutionContextWithoutDeviceMemory() noexcept override
    {
        return nullptr;
    }

    std::size_t getDeviceMemorySize() const noexcept override
    {
        return 0;
    }

    bool isRefittable() const noexcept override
    {
        return false;
    }

    char const* getName() const noexcept override
    {
        return "fake_engine";
    }

    int32_t getNbOptimizationProfiles() const noexcept override
    {
        return 1;
    }

    int32_t const* getProfileTensorValues(
        char const*, int32_t, nvinfer1::OptProfileSelector) const noexcept override
    {
        return nullptr;
    }

    nvinfer1::EngineCapability getEngineCapability() const noexcept override
    {
        return static_cast<nvinfer1::EngineCapability>(0);
    }

    void setErrorRecorder(nvinfer1::IErrorRecorder* recorder) noexcept override
    {
        mErrorRecorder = recorder;
    }

    nvinfer1::IErrorRecorder* getErrorRecorder() const noexcept override
    {
        return mErrorRecorder;
    }

    bool hasImplicitBatchDimension() const noexcept override
    {
        return false;
    }

    nvinfer1::TacticSources getTacticSources() const noexcept override
    {
        return static_cast<nvinfer1::TacticSources>(0);
    }

    nvinfer1::ProfilingVerbosity getProfilingVerbosity() const noexcept override
    {
        return static_cast<nvinfer1::ProfilingVerbosity>(0);
    }

    nvinfer1::IEngineInspector* createEngineInspector() const noexcept override
    {
        return nullptr;
    }

    nvinfer1::Dims getTensorShape(char const*) const noexcept override
    {
        return nvinfer1::Dims{};
    }

    nvinfer1::DataType getTensorDataType(char const*) const noexcept override
    {
        return static_cast<nvinfer1::DataType>(0);
    }

    nvinfer1::TensorLocation getTensorLocation(char const*) const noexcept override
    {
        return static_cast<nvinfer1::TensorLocation>(0);
    }

    bool isShapeInferenceIO(char const*) const noexcept override
    {
        return false;
    }

    nvinfer1::TensorIOMode getTensorIOMode(char const*) const noexcept override
    {
        return static_cast<nvinfer1::TensorIOMode>(0);
    }

    int32_t getTensorBytesPerComponent(char const*) const noexcept override
    {
        return 4;
    }

    int32_t getTensorComponentsPerElement(char const*) const noexcept override
    {
        return 1;
    }

    nvinfer1::TensorFormat getTensorFormat(char const*) const noexcept override
    {
        return static_cast<nvinfer1::TensorFormat>(0);
    }

    char const* getTensorFormatDesc(char const*) const noexcept override
    {
        return "";
    }

    int32_t getTensorVectorizedDim(char const*) const noexcept override
    {
        return -1;
    }

    nvinfer1::Dims getProfileShape(
        char const*, int32_t, nvinfer1::OptProfileSelector) const noexcept override
    {
        return nvinfer1::Dims{};
    }

    int32_t getNbIOTensors() const noexcept override
    {
        return static_cast<int32_t>(mIOTensorNames.size());
    }

    char const* getIOTensorName(int32_t index) const noexcept override
    {
        if (index < 0 || static_cast<std::size_t>(index) >= mIOTensorNames.size())
        {
            return nullptr;
        }
        const std::optional<std::string>& maybe_name =
            mIOTensorNames[static_cast<std::size_t>(index)];
        if (!maybe_name.has_value())
        {
            return nullptr;
        }
        return maybe_name->c_str();
    }

    nvinfer1::HardwareCompatibilityLevel getHardwareCompatibilityLevel() const noexcept override
    {
        return static_cast<nvinfer1::HardwareCompatibilityLevel>(0);
    }

    int32_t getNbAuxStreams() const noexcept override
    {
        return 0;
    }

    int32_t getTensorBytesPerComponentV2(char const*, int32_t) const noexcept override
    {
        return 4;
    }

    int32_t getTensorComponentsPerElementV2(char const*, int32_t) const noexcept override
    {
        return 1;
    }

    nvinfer1::TensorFormat getTensorFormatV2(char const*, int32_t) const noexcept override
    {
        return static_cast<nvinfer1::TensorFormat>(0);
    }

    char const* getTensorFormatDescV2(char const*, int32_t) const noexcept override
    {
        return "";
    }

    int32_t getTensorVectorizedDimV2(char const*, int32_t) const noexcept override
    {
        return -1;
    }

    nvinfer1::ISerializationConfig* createSerializationConfig() noexcept override
    {
        return nullptr;
    }

    nvinfer1::IHostMemory* serializeWithConfig(nvinfer1::ISerializationConfig&) const noexcept override
    {
        return nullptr;
    }

    std::size_t getDeviceMemorySizeForProfile(int32_t) const noexcept override
    {
        return 0;
    }

    nvinfer1::IRefitter* createRefitter(nvinfer1::ILogger&) noexcept override
    {
        return nullptr;
    }

    bool setWeightStreamingBudget(int64_t) noexcept override
    {
        return false;
    }

    int64_t getWeightStreamingBudget() const noexcept override
    {
        return 0;
    }

    int64_t getMinimumWeightStreamingBudget() const noexcept override
    {
        return 0;
    }

    int64_t getStreamableWeightsSize() const noexcept override
    {
        return 0;
    }

    bool isDebugTensor(char const*) const noexcept override
    {
        return false;
    }

    bool setWeightStreamingBudgetV2(int64_t) noexcept override
    {
        return false;
    }

    int64_t getWeightStreamingBudgetV2() const noexcept override
    {
        return 0;
    }

    int64_t getWeightStreamingAutomaticBudget() const noexcept override
    {
        return 0;
    }

    int64_t getWeightStreamingScratchMemorySize() const noexcept override
    {
        return 0;
    }

    int64_t getDeviceMemorySizeV2() const noexcept override
    {
        return 0;
    }

    int64_t getDeviceMemorySizeForProfileV2(int32_t) const noexcept override
    {
        return 0;
    }

    int64_t const* getProfileTensorValuesV2(
        char const*, int32_t, nvinfer1::OptProfileSelector) const noexcept override
    {
        return nullptr;
    }

    nvinfer1::IExecutionContext* createExecutionContextWithRuntimeConfig(
        nvinfer1::IRuntimeConfig*) noexcept override
    {
        return nullptr;
    }

    nvinfer1::IRuntimeConfig* createRuntimeConfig() noexcept override
    {
        return nullptr;
    }

    int64_t getEngineStat(nvinfer1::EngineStat) const noexcept override
    {
        return 0;
    }

    char const* getAliasedInputTensor(char const*) const noexcept override
    {
        return nullptr;
    }

private:
    std::vector<std::optional<std::string>> mIOTensorNames;
    nvinfer1::IErrorRecorder* mErrorRecorder{nullptr};
};

class FakeCudaEngine final : public nvinfer1::ICudaEngine {
public:
    explicit FakeCudaEngine(std::vector<std::optional<std::string>> io_tensor_names)
        : mImplStorage(std::move(io_tensor_names))
    {
        mImpl = &mImplStorage;
    }

private:
    FakeCudaEngineImpl mImplStorage;
};

std::vector<std::optional<std::string>> make_required_tensor_names(
    int32_t num_layers, bool include_position_input)
{
    std::vector<std::optional<std::string>> names;
    names.emplace_back("token_id");
    names.emplace_back("attention_mask");
    names.emplace_back("logits");
    if (include_position_input)
    {
        names.emplace_back("position_id");
    }
    for (int32_t i = 0; i < num_layers; ++i)
    {
        names.emplace_back(trtf::layer_tensor_name("cache_k", i));
        names.emplace_back(trtf::layer_tensor_name("cache_v", i));
        names.emplace_back(trtf::layer_tensor_name("present_k", i));
        names.emplace_back(trtf::layer_tensor_name("present_v", i));
    }
    return names;
}

void remove_name(std::vector<std::optional<std::string>>& names, const std::string& target)
{
    names.erase(std::remove_if(names.begin(), names.end(),
                    [&](const std::optional<std::string>& value) {
                        return value.has_value() && *value == target;
                    }),
        names.end());
}

trtf::DecoderStepEngine make_decoder_step_engine(
    std::vector<std::optional<std::string>> io_tensor_names, int32_t num_layers, bool requires_position_input)
{
    trtf::DecoderStepEngine engine;
    engine.engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(new FakeCudaEngine(std::move(io_tensor_names)));
    engine.num_layers = num_layers;
    engine.requires_position_input = requires_position_input;

    engine.cache_k_input_names.reserve(static_cast<std::size_t>(num_layers));
    engine.cache_v_input_names.reserve(static_cast<std::size_t>(num_layers));
    engine.present_k_output_names.reserve(static_cast<std::size_t>(num_layers));
    engine.present_v_output_names.reserve(static_cast<std::size_t>(num_layers));
    for (int32_t i = 0; i < num_layers; ++i)
    {
        engine.cache_k_input_names.push_back(trtf::layer_tensor_name("cache_k", i));
        engine.cache_v_input_names.push_back(trtf::layer_tensor_name("cache_v", i));
        engine.present_k_output_names.push_back(trtf::layer_tensor_name("present_k", i));
        engine.present_v_output_names.push_back(trtf::layer_tensor_name("present_v", i));
    }
    return engine;
}

void test_has_io_tensor_true_false_and_null_name()
{
    FakeCudaEngine fake_engine({std::nullopt, std::string("token_id"), std::string("attention_mask")});
    check(trtf::has_io_tensor(fake_engine, "token_id"), "has_io_tensor returns true for existing name");
    check(!trtf::has_io_tensor(fake_engine, "logits"), "has_io_tensor returns false for missing name");
    check(!trtf::has_io_tensor(fake_engine, ""), "has_io_tensor ignores null tensor names");
}

void test_has_all_required_tensors_false_when_missing_base_tensors()
{
    std::vector<std::optional<std::string>> names = make_required_tensor_names(/*num_layers=*/2, true);
    remove_name(names, "attention_mask");
    trtf::DecoderStepEngine engine = make_decoder_step_engine(std::move(names), /*num_layers=*/2, true);
    check(!trtf::has_all_required_tensors(engine), "has_all_required_tensors fails when base tensor is missing");
}

void test_has_all_required_tensors_true_with_all_base_and_layer_tensors()
{
    trtf::DecoderStepEngine engine = make_decoder_step_engine(
        make_required_tensor_names(/*num_layers=*/2, /*include_position_input=*/true), /*num_layers=*/2, true);
    check(trtf::has_all_required_tensors(engine), "has_all_required_tensors passes when all tensors are present");
}

void test_has_all_required_tensors_requires_position_input_branch()
{
    trtf::DecoderStepEngine no_position_required = make_decoder_step_engine(
        make_required_tensor_names(/*num_layers=*/1, /*include_position_input=*/false), /*num_layers=*/1, false);
    check(trtf::has_all_required_tensors(no_position_required),
        "requires_position_input=false does not require position tensor");

    trtf::DecoderStepEngine position_required_missing = make_decoder_step_engine(
        make_required_tensor_names(/*num_layers=*/1, /*include_position_input=*/false), /*num_layers=*/1, true);
    check(!trtf::has_all_required_tensors(position_required_missing),
        "requires_position_input=true fails without position tensor");
}

void test_has_all_required_tensors_missing_per_layer_tensor_fails()
{
    std::vector<std::optional<std::string>> names = make_required_tensor_names(/*num_layers=*/3, true);
    remove_name(names, trtf::layer_tensor_name("present_v", 1));
    trtf::DecoderStepEngine engine = make_decoder_step_engine(std::move(names), /*num_layers=*/3, true);
    check(!trtf::has_all_required_tensors(engine), "has_all_required_tensors fails when a per-layer tensor is missing");
}

#endif // NV_TENSORRT_MAJOR >= 10

#endif // TRTF_HAS_TRT

} // namespace

int main()
{
#if TRTF_HAS_TRT
#if NV_TENSORRT_MAJOR >= 10
    test_has_io_tensor_true_false_and_null_name();
    test_has_all_required_tensors_false_when_missing_base_tensors();
    test_has_all_required_tensors_true_with_all_base_and_layer_tensors();
    test_has_all_required_tensors_requires_position_input_branch();
    test_has_all_required_tensors_missing_per_layer_tensor_fails();
#else
    std::cerr << "test_trt_engine_lifecycle_fake_engine: skipping, fake ICudaEngine is only implemented for "
                 "NV_TENSORRT_MAJOR >= 10.\n";
#endif
#else
    std::cerr << "test_trt_engine_lifecycle_fake_engine: TRTF_HAS_TRT=0, skipping TRT-dependent tests.\n";
#endif

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All trt_engine_lifecycle fake-engine tests passed.\n";
    return 0;
}
