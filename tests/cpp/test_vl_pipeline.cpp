// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-VL-CPP-01
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-FAC-01
// Intent:         VLPipeline text-only generation with mock engines
// Preconditions:  TRT + CUDA GPU available
// Postconditions: Pipeline generates text tokens correctly in text-only mode
// =============================================================================

// =============================================================================
// Test suite: VLPipeline — vision-language generation
// =============================================================================

#include "runtime/pipelines/vl_pipeline.h"
#include "trtf/runtime/trt_module.h"
#include "trtf/runtime/kv_cache.h"

#include <cstdint>
#include <iostream>
#include <vector>

#if TRTF_HAS_TRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "runtime/trt/core/trt_common.h"
#endif

static int failures = 0;
static void check(bool c, const char* n) { if (!c) { std::cerr << "FAIL: " << n << '\n'; ++failures; } }

#if TRTF_HAS_TRT

static trtf::TrtLogger g_logger;

static trtf::TrtUniquePtr<nvinfer1::ICudaEngine> build_mock_decoder()
{
    auto b = trtf::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtf::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtf::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* tok = n->addInput("token_id", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {1}});
    auto* mask = n->addInput("attention_mask", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {8}});

    float cl[4] = {0.1f, 0.2f, 0.9f, 0.3f};
    auto* cst = n->addConstant(nvinfer1::Dims{1, {4}},
        nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cl, 4});
    cst->getOutput(0)->setName("logits");
    n->markOutput(*cst->getOutput(0));

    n->addIdentity(*tok)->getOutput(0)->setName("_t");
    n->addIdentity(*mask)->getOutput(0)->setName("_m");

    auto plan = trtf::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan) return nullptr;
    auto rt = trtf::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(rt->deserializeCudaEngine(plan->data(), plan->size()));
}

static void test_vl_text_only()
{
    auto engine = build_mock_decoder();
    if (!engine) { std::cerr << "SKIP\n"; return; }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto decoder = std::make_unique<trtf::TrtModule>(engine.get(), stream);
    auto cache = std::make_unique<trtf::KvCache>(1, 8, 4, stream);

    trtf::VLConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 2;
    cfg.has_position_input = false;

    // No vision encoder (text-only mode)
    trtf::VLPreprocessConfig vl_pp;
    trtf::VLPipeline pipeline(std::move(decoder), nullptr, std::move(cache), cfg, vl_pp, stream);
    check(std::string(pipeline.pipeline_type()) == "VLPipeline", "vl name");

    trtf::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 5;
    auto result = pipeline.generate_ids({1}, gen_cfg);

    // argmax=2=eos → stops after 1 generated token
    check(result.token_ids.size() == 2, "text-only: input + eos");
    check(result.token_ids[1] == 2, "text-only: eos generated");

    cudaStreamDestroy(stream);
}

static void test_vl_text_only_max_tokens()
{
    auto engine = build_mock_decoder();
    if (!engine) { std::cerr << "SKIP\n"; return; }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto decoder = std::make_unique<trtf::TrtModule>(engine.get(), stream);
    auto cache = std::make_unique<trtf::KvCache>(1, 8, 4, stream);

    trtf::VLConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 99;
    cfg.has_position_input = false;

    trtf::VLPreprocessConfig vl_pp;
    trtf::VLPipeline pipeline(std::move(decoder), nullptr, std::move(cache), cfg, vl_pp, stream);

    trtf::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 3;
    auto result = pipeline.generate_ids({0, 1}, gen_cfg);

    // 2 input + 3 generated = 5 total
    check(result.token_ids.size() == 5, "max tokens: 2 input + 3 gen");

    cudaStreamDestroy(stream);
}

#endif

int main()
{
#if TRTF_HAS_TRT
    test_vl_text_only();
    test_vl_text_only_max_tokens();
#else
    std::cerr << "TRT not available, skipping\n";
#endif
    if (failures > 0) std::cerr << failures << " FAILED\n";
    return failures;
}
