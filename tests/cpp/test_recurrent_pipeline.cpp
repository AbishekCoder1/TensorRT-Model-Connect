// =============================================================================
// Test suite: RecurrentPipeline (Mamba, RWKV, Hybrid)
// =============================================================================
//
// Tests the RecurrentPipeline with mock engines and both RecurrentStateManager
// (for pure SSM) and HybridStateManager (for attention+SSM).
// =============================================================================

#include "runtime/pipelines/recurrent_pipeline.h"
#include "trtf/runtime/trt_module.h"
#include "trtf/runtime/kv_cache.h"
#include "trtf/runtime/recurrent_state.h"
// pipeline_interface.h was removed; GenerateConfig is in trtf/pipeline.h
// (already included transitively via recurrent_pipeline.h)

#include <cstdint>
#include <iostream>
#include <vector>

#if TRTF_HAS_TRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "runtime/trt/core/trt_common.h"
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

// Mock decoder: token_id[1] → logits[4] = constant [0.1, 0.2, 0.9, 0.3]
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

    auto* inp = network->addInput("token_id", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {1}});
    float const_logits[4] = {0.1f, 0.2f, 0.9f, 0.3f};
    auto* cst = network->addConstant(nvinfer1::Dims{1, {4}},
        nvinfer1::Weights{nvinfer1::DataType::kFLOAT, const_logits, 4});
    cst->getOutput(0)->setName("logits");
    network->markOutput(*cst->getOutput(0));

    // Use input so it's not optimized away
    auto* id = network->addIdentity(*inp);
    id->getOutput(0)->setName("_unused");

    auto plan = trtf::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan) return nullptr;
    auto rt = trtf::TrtUniquePtr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(g_logger));
    return trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

static void test_mamba_pipeline()
{
    auto engine = build_mock_decoder();
    if (!engine) { std::cerr << "SKIP: can't build engine\n"; return; }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtf::TrtModule>(engine.get(), stream);

    // Mamba: 2 state specs, 1 layer
    std::vector<trtf::RecurrentState::TensorSpec> specs = {
        {"conv_state", {12}}, {"ssm_state", {32}},
    };
    auto rs = std::make_unique<trtf::RecurrentState>(1, specs, stream);
    auto mgr = std::make_unique<trtf::RecurrentStateManager>(std::move(rs));

    trtf::RecurrentGenConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 2;  // argmax=2=eos

    trtf::RecurrentPipeline pipeline(std::move(module), std::move(mgr), cfg, stream, "MambaPipeline");
    check(std::string(pipeline.pipeline_type()) == "MambaPipeline", "mamba name");

    trtf::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 5;
    auto result = pipeline.generate_ids({1}, gen_cfg);

    // argmax=2=eos → stops after 1 generated token
    check(result.token_ids.size() == 2, "mamba: input + 1 generated");
    check(result.token_ids[1] == 2, "mamba: generated token = 2 (eos)");

    cudaStreamDestroy(stream);
}

static void test_rwkv_pipeline()
{
    auto engine = build_mock_decoder();
    if (!engine) { std::cerr << "SKIP: can't build engine\n"; return; }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtf::TrtModule>(engine.get(), stream);

    // RWKV: 5 state specs, 2 layers
    std::vector<trtf::RecurrentState::TensorSpec> specs = {
        {"attn_state", {8}}, {"ff_state", {8}}, {"num_state", {8}},
        {"den_state", {8}}, {"max_state", {8}},
    };
    auto rs = std::make_unique<trtf::RecurrentState>(2, specs, stream);
    auto mgr = std::make_unique<trtf::RecurrentStateManager>(std::move(rs));

    trtf::RecurrentGenConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 99;  // never hit

    trtf::RecurrentPipeline pipeline(std::move(module), std::move(mgr), cfg, stream, "RwkvPipeline");
    check(std::string(pipeline.pipeline_type()) == "RwkvPipeline", "rwkv name");

    trtf::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 3;
    auto result = pipeline.generate_ids({0}, gen_cfg);

    check(result.token_ids.size() == 4, "rwkv: input + 3 generated");
    check(result.token_ids[1] == 2, "rwkv: all gen tokens = 2");
    check(result.token_ids[3] == 2, "rwkv: last gen = 2");

    cudaStreamDestroy(stream);
}

static void test_hybrid_pipeline()
{
    auto engine = build_mock_decoder();
    if (!engine) { std::cerr << "SKIP: can't build engine\n"; return; }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Build mock engine with mask input too
    auto builder = trtf::TrtUniquePtr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(g_logger));
    auto network = trtf::TrtUniquePtr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(0));
    auto bconfig = trtf::TrtUniquePtr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    bconfig->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* tok = network->addInput("token_id", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {1}});
    auto* pos = network->addInput("position_id", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {1}});
    auto* mask = network->addInput("attention_mask", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {4}});

    float cl[4] = {0.1f, 0.2f, 0.9f, 0.3f};
    auto* c = network->addConstant(nvinfer1::Dims{1, {4}},
        nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cl, 4});
    c->getOutput(0)->setName("logits");
    network->markOutput(*c->getOutput(0));

    network->addIdentity(*tok)->getOutput(0)->setName("_t");
    network->addIdentity(*pos)->getOutput(0)->setName("_p");
    network->addIdentity(*mask)->getOutput(0)->setName("_m");

    auto plan = trtf::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *bconfig));
    if (!plan) { cudaStreamDestroy(stream); return; }
    auto rt = trtf::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    auto hybrid_engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
    if (!hybrid_engine) { cudaStreamDestroy(stream); return; }

    auto module = std::make_unique<trtf::TrtModule>(hybrid_engine.get(), stream);
    auto kv = std::make_unique<trtf::KvCache>(1, 4, 2, stream);
    std::vector<trtf::RecurrentState::TensorSpec> specs = {{"ssm", {4}}};
    auto ssm = std::make_unique<trtf::RecurrentState>(1, specs, stream);
    auto mgr = std::make_unique<trtf::HybridStateManager>(std::move(kv), std::move(ssm));

    trtf::RecurrentGenConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 2;
    cfg.has_position_input = true;

    trtf::RecurrentPipeline pipeline(std::move(module), std::move(mgr), cfg, stream, "HybridPipeline");
    check(std::string(pipeline.pipeline_type()) == "HybridPipeline", "hybrid name");

    trtf::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 5;
    auto result = pipeline.generate_ids({0}, gen_cfg);

    check(result.token_ids.size() == 2, "hybrid: input + eos");
    check(result.token_ids[1] == 2, "hybrid: eos generated");

    cudaStreamDestroy(stream);
}

static void test_argmax_recurrent()
{
    std::vector<float> v = {-1.0f, 5.0f, 3.0f};
    check(trtf::RecurrentPipeline::argmax(v) == 1, "argmax = 1");
}

#endif

int main()
{
#if TRTF_HAS_TRT
    test_argmax_recurrent();
    test_mamba_pipeline();
    test_rwkv_pipeline();
    test_hybrid_pipeline();
#else
    std::cerr << "TRT not available, skipping\n";
#endif
    if (failures > 0) std::cerr << failures << " FAILED\n";
    return failures;
}
