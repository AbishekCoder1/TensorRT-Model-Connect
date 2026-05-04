// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-KVC-CPP-02
// Architecture:   ARCH-KVC-001
// Unit Design:    UD-KVC-02
// Intent:         DeviceResources construction, ok(), and run_decoder_step_device
//                 end-to-end execution with a minimal mock TRT engine
// Preconditions:  TRT + CUDA GPU available
// Postconditions: DeviceResources allocates GPU memory correctly; ok()=true;
//                 run_decoder_step_device executes one step and returns logits
// =============================================================================

// test_device_resources.cpp — Unit tests for DeviceResources and
//   run_decoder_step_device in device_kv_cache.cpp
//
// Purpose:
//   Validates DeviceResources construction (GPU allocation) and the
//   run_decoder_step_device function using a tiny in-process TRT engine.
//   Complements test_device_kv_cache.cpp which covers DeviceKvCache but
//   leaves DeviceResources and run_decoder_step_device uncovered.
//
// Approach:
//   Builds a minimal TRT engine with:
//     Inputs : token_id [1] int32, attention_mask [4] float32
//     Outputs: logits [4] float32 (constant [0.1, 0.2, 0.9, 0.3])
//   Sets num_layers=0 so no KV-cache tensors are required.
//   Constructs DeviceKvCache + DeviceResources, then calls
//   run_decoder_step_device to exercise the full step path.
//
// Dependencies:
//   - runtime/core/device_kv_cache.h
//   - runtime/core/trt_engine_lifecycle.h
//   - runtime/core/trt_common.h
//   TRT + CUDA required.

#include "runtime/core/device_kv_cache.h"
#include "runtime/backend/trt_logger.h"
#include "runtime/backend/trt_module_impl.h"
#include "runtime/core/cuda_common.h"
#include "runtime/core/trt_common.h"
#include "runtime/core/trt_engine_lifecycle.h"

#include <NvInfer.h>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static trtf::TrtLogger g_logger;

// ---------------------------------------------------------------------------
// Build a minimal TRT engine suitable for DeviceResources testing.
//
// Engine spec (no KV-cache tensors; num_layers=0):
//   Inputs : token_id [1] int32
//             attention_mask [4] float32
//   Outputs: logits [4] float32 — constant [0.1, 0.2, 0.9, 0.3], argmax=2
// ---------------------------------------------------------------------------
static trtf::TrtUniquePtr<nvinfer1::ICudaEngine> build_engine_for_resources() {
    auto builder = trtf::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    if (!builder)
        return nullptr;

    auto network = trtf::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = trtf::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    // Inputs (must be present for binding to succeed)
    auto* tok_inp =
        network->addInput("token_id", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {1}});
    auto* mask_inp =
        network->addInput("attention_mask", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {4}});

    // Constant logits output: argmax = index 2
    float const_logits[4] = {0.1f, 0.2f, 0.9f, 0.3f};
    auto* const_w = network->addConstant(
        nvinfer1::Dims{1, {4}}, nvinfer1::Weights{nvinfer1::DataType::kFLOAT, const_logits, 4});
    if (!const_w)
        return nullptr;

    auto* logits_out = const_w->getOutput(0);
    logits_out->setName("logits");
    network->markOutput(*logits_out);

    // Mark inputs as "used" via identity (prevents TRT from pruning them)
    auto* id_tok = network->addIdentity(*tok_inp);
    id_tok->getOutput(0)->setName("_tok_unused");
    auto* id_mask = network->addIdentity(*mask_inp);
    id_mask->getOutput(0)->setName("_mask_unused");

    auto plan = trtf::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan)
        return nullptr;

    auto runtime = trtf::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
}

struct StepFixture {
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> engine;
    trtf::CudaStream stream;
    std::unique_ptr<trtf::TrtModuleImpl> module;
    trtf::DecoderStepEngine step;
};

// Build a populated DecoderStepEngine (no cache layers) from the mock engine.
static std::unique_ptr<StepFixture> make_step_fixture() {
    auto engine = build_engine_for_resources();
    if (!engine) {
        std::cerr << "WARNING: Could not build mock engine, skipping\n";
        return nullptr;
    }

    auto fixture = std::make_unique<StepFixture>();
    fixture->engine = std::move(engine);
    if (!fixture->stream.ok()) {
        std::cerr << "WARNING: CUDA stream creation failed, skipping\n";
        return nullptr;
    }
    auto* context = fixture->engine->createExecutionContext();
    if (!context) {
        std::cerr << "WARNING: createExecutionContext failed, skipping\n";
        return nullptr;
    }
    fixture->module = std::make_unique<trtf::TrtModuleImpl>(
        fixture->engine.get(), context, fixture->stream.get(), /*profile_idx=*/0);
    if (!fixture->module->ok()) {
        std::cerr << "WARNING: TrtModuleImpl creation failed, skipping\n";
        return nullptr;
    }

    auto& eng = fixture->step;
    eng.module = fixture->module.get();
    eng.num_layers = 0; // No KV-cache tensors required
    eng.vocab_size = 4;
    eng.hidden_size = 0;
    eng.cache_state_size = 0;
    eng.attention_mask_size = 4;
    eng.max_cache_length = 4;
    eng.requires_position_input = false;
    // token_input_name / mask_input_name / logits_output_name use defaults
    return fixture;
}

// ---------------------------------------------------------------------------
// Intention: DeviceResources constructs successfully and ok() returns true.
// Preconditions:  Real TRT engine built in-process; CUDA GPU available
// Postconditions: ok()==true; mandatory device buffers are allocated
// ---------------------------------------------------------------------------
static bool test_device_resources_construction() {
    auto fixture = make_step_fixture();
    if (!fixture)
        return true; // not a test failure — just unavailable TRT

    trtf::DeviceResources res(fixture->step);
    if (!res.ok()) {
        std::cerr << "device_resources_construction: ok() returned false\n";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Intention: run_decoder_step_device executes one step and returns logits.
// Preconditions:  Real TRT engine with constant logits output; CUDA GPU
// Postconditions: logits has vocab_size=4 elements; argmax of output is 2;
//                 error string is empty on success
// ---------------------------------------------------------------------------
static bool test_run_decoder_step_device_basic() {
    auto fixture = make_step_fixture();
    if (!fixture)
        return true;
    auto& eng = fixture->step;

    trtf::DeviceKvCache cache(eng);
    if (!cache.ok()) {
        std::cerr << "run_decoder_step_device_basic: cache.ok() = false\n";
        return false;
    }

    trtf::DeviceResources res(eng);
    if (!res.ok()) {
        std::cerr << "run_decoder_step_device_basic: res.ok() = false\n";
        return false;
    }

    std::vector<float> logits;
    std::string error;

    const bool ok = trtf::run_decoder_step_device(eng, cache, res,
                                                  /*token_id=*/1, logits, error,
                                                  /*input_embed_host=*/nullptr,
                                                  /*embed_dim=*/0,
                                                  /*use_input_embed=*/0.0F,
                                                  /*deepstack_embeds_host=*/{},
                                                  /*deepstack_active=*/0.0F,
                                                  /*input_embed_device_ready=*/false,
                                                  /*skip_logits_d2h=*/false,
                                                  /*skip_sync=*/false,
                                                  /*skip_bind=*/false);

    if (!ok) {
        std::cerr << "run_decoder_step_device_basic: step failed: " << error << '\n';
        return false;
    }
    if (logits.size() != 4) {
        std::cerr << "run_decoder_step_device_basic: logits.size()=" << logits.size()
                  << " expected 4\n";
        return false;
    }

    // Find argmax: expect index 2 (logit 0.9 is largest)
    int32_t argmax = 0;
    float best = logits[0];
    for (int32_t i = 1; i < 4; ++i) {
        if (logits[static_cast<std::size_t>(i)] > best) {
            best = logits[static_cast<std::size_t>(i)];
            argmax = i;
        }
    }
    if (argmax != 2) {
        std::cerr << "run_decoder_step_device_basic: argmax=" << argmax << " expected 2\n";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Intention: run_decoder_step_device with skip_bind=true (cache unchanged)
//            succeeds on a second step without re-binding tensor addresses.
// Preconditions:  Same engine as above; cache updated by first step
// Postconditions: second step also returns valid logits
// ---------------------------------------------------------------------------
static bool test_run_decoder_step_device_skip_bind() {
    auto fixture = make_step_fixture();
    if (!fixture)
        return true;
    auto& eng = fixture->step;

    trtf::DeviceKvCache cache(eng);
    trtf::DeviceResources res(eng);
    if (!res.ok())
        return false;

    std::vector<float> logits;
    std::string error;

    // First step: bind tensor addresses
    if (!trtf::run_decoder_step_device(eng, cache, res, 1, logits, error, nullptr, 0, 0.0F, {},
                                       0.0F, false, false, false, /*skip_bind=*/false)) {
        std::cerr << "step_skip_bind: step 1 failed: " << error << '\n';
        return false;
    }

    // Second step: skip rebind (addresses haven't changed)
    if (!trtf::run_decoder_step_device(eng, cache, res, 1, logits, error, nullptr, 0, 0.0F, {},
                                       0.0F, false, false, false, /*skip_bind=*/true)) {
        std::cerr << "step_skip_bind: step 2 (skip_bind) failed: " << error << '\n';
        return false;
    }

    return logits.size() == 4;
}

// ---------------------------------------------------------------------------
// Intention: run_decoder_step_device with skip_logits_d2h=true skips
//            the device-to-host logits transfer and leaves logits empty.
// Preconditions:  Same engine; skip_logits_d2h=true
// Postconditions: logits vector is empty (no D2H copy); step returns true
// ---------------------------------------------------------------------------
static bool test_run_decoder_step_device_skip_d2h() {
    auto fixture = make_step_fixture();
    if (!fixture)
        return true;
    auto& eng = fixture->step;

    trtf::DeviceKvCache cache(eng);
    trtf::DeviceResources res(eng);
    if (!res.ok())
        return false;

    std::vector<float> logits;
    std::string error;

    const bool ok = trtf::run_decoder_step_device(eng, cache, res, 1, logits, error, nullptr, 0,
                                                  0.0F, {}, 0.0F, false,
                                                  /*skip_logits_d2h=*/true,
                                                  /*skip_sync=*/false,
                                                  /*skip_bind=*/false);

    if (!ok) {
        std::cerr << "step_skip_d2h: step failed: " << error << '\n';
        return false;
    }
    // With skip_logits_d2h, logits should remain empty (no D2H)
    return logits.empty();
}

int main() {
    bool all_passed = true;
    std::cout << "test_device_resources:" << std::endl;

    const auto run = [&](const char* name, bool (*fn)()) {
        const bool ok = fn();
        std::cout << "  " << name << ": " << (ok ? "PASS" : "FAIL") << '\n';
        all_passed &= ok;
    };

    run("device_resources_construction", test_device_resources_construction);
    run("run_decoder_step_device_basic", test_run_decoder_step_device_basic);
    run("run_decoder_step_device_skip_bind", test_run_decoder_step_device_skip_bind);
    run("run_decoder_step_device_skip_d2h", test_run_decoder_step_device_skip_d2h);

    if (all_passed) {
        std::cout << "test_device_resources passed" << std::endl;
        return 0;
    }
    std::cerr << "test_device_resources FAILED" << std::endl;
    return 1;
}
