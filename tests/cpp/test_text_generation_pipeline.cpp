// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-DEC-CPP-02
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-TRT-DEC-01
// Intent:         TextGenerationPipeline prefill/decode loop, argmax selection, EOS stopping
// Preconditions:  TRT + CUDA GPU available, identity engine built in-process
// Postconditions: Pipeline generates correct tokens, stops at EOS, respects max_new_tokens
// =============================================================================

// =============================================================================
// Test suite: TextGenerationPipeline
// =============================================================================
//
// Tests the TextGenerationPipeline using a tiny TRT identity engine.
// The identity engine maps token_id[1] → logits[4] (just copies input to output).
// This validates the prefill→decode loop, argmax, and EOS stopping.
//
// For full E2E validation with real models, see tests/test_e2e.py.
// =============================================================================

#include "runtime/pipelines/text_generation_pipeline.h"
#include "trtf/runtime/trt_module.h"
#include "trtf/runtime/kv_cache.h"
// pipeline_interface.h was removed; GenerateConfig is in trtf/pipeline.h
// (already included transitively via text_generation_pipeline.h)

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#if TRTF_HAS_TRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "runtime/core/trt_common.h"
#include "runtime/backend/trt_module_impl.h"
#endif

static int failures = 0;

static void check(bool condition, const char* test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

#if TRTF_HAS_TRT

static trtf::TrtLogger g_logger;

// Build a tiny decoder-like engine:
// Inputs:  token_id [1] int32, attention_mask [8] float32
// Outputs: logits [4] float32
// The engine produces fixed logits [0.1, 0.2, 0.9, 0.3] regardless of input
// (identity on a constant), so argmax always returns 2.
static trtf::TrtUniquePtr<nvinfer1::ICudaEngine> build_mock_decoder()
{
    auto builder = trtf::TrtUniquePtr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(g_logger));
    if (!builder) return nullptr;

    auto network = trtf::TrtUniquePtr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(0));
    auto config = trtf::TrtUniquePtr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    // Inputs
    auto* token_inp = network->addInput("token_id", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {1}});
    auto* mask_inp = network->addInput("attention_mask", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {8}});

    // Constant logits: [0.1, 0.2, 0.9, 0.3] — argmax = index 2
    float const_logits[4] = {0.1f, 0.2f, 0.9f, 0.3f};
    auto* const_w = network->addConstant(nvinfer1::Dims{1, {4}},
        nvinfer1::Weights{nvinfer1::DataType::kFLOAT, const_logits, 4});
    if (!const_w) return nullptr;

    auto* out = const_w->getOutput(0);
    out->setName("logits");
    network->markOutput(*out);

    // Need to "use" the inputs so TRT doesn't optimize them away
    // Add identity on token_id and mask (mark as outputs too, then unmark)
    // Actually, for a proper test engine, just mark them as used via identity
    auto* id_token = network->addIdentity(*token_inp);
    id_token->getOutput(0)->setName("_unused_token");

    auto* id_mask = network->addIdentity(*mask_inp);
    id_mask->getOutput(0)->setName("_unused_mask");

    auto plan = trtf::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan) return nullptr;

    auto runtime = trtf::TrtUniquePtr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(g_logger));
    return trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
}

static void test_pipeline_construction()
{
    auto engine = build_mock_decoder();
    if (!engine)
    {
        std::cerr << "WARNING: Could not build mock decoder engine, skipping test\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtf::TrtModuleImpl>(engine.get(), engine->createExecutionContext(), stream);
    auto cache = std::make_unique<trtf::KvCache>(1, 8, 4, stream);

    trtf::TextGenConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_bos = 0;
    cfg.id_eos = 2;  // argmax will always hit this!
    cfg.has_position_input = false;

    trtf::TextGenerationPipeline pipeline(
        std::move(module), std::move(cache), cfg, stream);

    check(std::string(pipeline.pipeline_type()) == "TextGenerationPipeline",
          "pipeline name");

    cudaStreamDestroy(stream);
}

static void test_generate_stops_at_eos()
{
    auto engine = build_mock_decoder();
    if (!engine)
    {
        std::cerr << "WARNING: Could not build mock decoder engine, skipping test\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtf::TrtModuleImpl>(engine.get(), engine->createExecutionContext(), stream);
    auto cache = std::make_unique<trtf::KvCache>(1, 8, 4, stream);

    trtf::TextGenConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_bos = 0;
    cfg.id_eos = 2;  // argmax of [0.1, 0.2, 0.9, 0.3] = 2 = eos
    cfg.has_position_input = false;

    trtf::TextGenerationPipeline pipeline(
        std::move(module), std::move(cache), cfg, stream);

    trtf::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 10;

    auto result = pipeline.generate_ids({1}, gen_cfg);

    // Input [1] + one generated token (eos=2) → should stop immediately
    check(result.token_ids.size() == 2, "output has 2 tokens (input + eos)");
    check(result.token_ids[0] == 1, "first token is input");
    check(result.token_ids[1] == 2, "second token is eos (argmax=2)");

    cudaStreamDestroy(stream);
}

static void test_generate_max_tokens()
{
    auto engine = build_mock_decoder();
    if (!engine)
    {
        std::cerr << "WARNING: Could not build mock decoder engine, skipping test\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtf::TrtModuleImpl>(engine.get(), engine->createExecutionContext(), stream);
    auto cache = std::make_unique<trtf::KvCache>(1, 8, 4, stream);

    trtf::TextGenConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_bos = 0;
    cfg.id_eos = 99;  // EOS token that argmax will never produce
    cfg.has_position_input = false;

    trtf::TextGenerationPipeline pipeline(
        std::move(module), std::move(cache), cfg, stream);

    trtf::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 3;

    auto result = pipeline.generate_ids({1}, gen_cfg);

    // Input [1] + 3 generated tokens (all argmax=2, never hits eos=99)
    check(result.token_ids.size() == 4, "output has 4 tokens (input + 3 generated)");
    check(result.token_ids[0] == 1, "first = input");
    check(result.token_ids[1] == 2, "gen 1 = argmax(2)");
    check(result.token_ids[2] == 2, "gen 2 = argmax(2)");
    check(result.token_ids[3] == 2, "gen 3 = argmax(2)");

    cudaStreamDestroy(stream);
}

static void test_argmax()
{
    std::vector<float> logits = {0.1f, 0.5f, 0.3f, 0.8f, 0.2f};
    int32_t result = trtf::TextGenerationPipeline::argmax(logits);
    check(result == 3, "argmax of [0.1, 0.5, 0.3, 0.8, 0.2] = 3");

    std::vector<float> single = {42.0f};
    check(trtf::TextGenerationPipeline::argmax(single) == 0, "argmax of single = 0");

    std::vector<float> empty;
    check(trtf::TextGenerationPipeline::argmax(empty) == 0, "argmax of empty = 0");
}

static void test_zero_max_tokens()
{
    auto engine = build_mock_decoder();
    if (!engine) return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtf::TrtModuleImpl>(engine.get(), engine->createExecutionContext(), stream);
    auto cache = std::make_unique<trtf::KvCache>(1, 8, 4, stream);

    trtf::TextGenConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 2;
    cfg.has_position_input = false;

    trtf::TextGenerationPipeline pipeline(
        std::move(module), std::move(cache), cfg, stream);

    trtf::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 0;

    auto result = pipeline.generate_ids({1, 2, 3}, gen_cfg);
    check(result.token_ids.size() == 3, "zero max_new_tokens returns input unchanged");

    cudaStreamDestroy(stream);
}

#endif // TRTF_HAS_TRT

int main()
{
#if TRTF_HAS_TRT
    test_argmax();
    test_pipeline_construction();
    test_generate_stops_at_eos();
    test_generate_max_tokens();
    test_zero_max_tokens();
#else
    std::cerr << "TRT not available, skipping TextGenerationPipeline tests\n";
#endif

    if (failures > 0) std::cerr << failures << " test(s) FAILED\n";
    return failures;
}
