// =============================================================================
// Test suite: EncoderPipeline + SegmentPipeline + SamPipeline
// =============================================================================

#include "runtime/pipelines/encoder_pipeline.h"
#include "trtf/runtime/trt_module.h"

#include <cstdint>
#include <iostream>
#include <string>
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

// Mock: input_ids[4] int32 -> output[8] float (constant)
static trtf::TrtUniquePtr<nvinfer1::ICudaEngine> build_encoder_engine()
{
    auto b = trtf::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtf::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtf::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* ids = n->addInput("input_ids", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {4}});
    auto* mask = n->addInput("attention_mask", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {4}});

    float cv[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    auto* cst = n->addConstant(nvinfer1::Dims{1, {8}},
        nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cv, 8});
    cst->getOutput(0)->setName("output_embeddings");
    n->markOutput(*cst->getOutput(0));

    n->addIdentity(*ids)->getOutput(0)->setName("_i");
    n->addIdentity(*mask)->getOutput(0)->setName("_m");

    auto plan = trtf::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan) return nullptr;
    auto rt = trtf::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(rt->deserializeCudaEngine(plan->data(), plan->size()));
}

// Mock: pixel_values[3*4*4] -> output[16] float (constant)
static trtf::TrtUniquePtr<nvinfer1::ICudaEngine> build_segment_engine()
{
    auto b = trtf::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtf::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtf::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* pv = n->addInput("pixel_values", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{3, {3, 4, 4}});

    float cv[16];
    for (int i = 0; i < 16; ++i) cv[i] = static_cast<float>(i);
    auto* cst = n->addConstant(nvinfer1::Dims{1, {16}},
        nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cv, 16});
    cst->getOutput(0)->setName("output_mask");
    n->markOutput(*cst->getOutput(0));

    n->addIdentity(*pv)->getOutput(0)->setName("_pv");

    auto plan = trtf::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan) return nullptr;
    auto rt = trtf::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(rt->deserializeCudaEngine(plan->data(), plan->size()));
}

static void test_encoder_pipeline()
{
    auto engine = build_encoder_engine();
    if (!engine) { std::cerr << "SKIP encoder\n"; return; }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtf::TrtModule>(engine.get(), stream);
    trtf::EncoderPipeline pipeline(std::move(module), "embedding");

    check(std::string(pipeline.pipeline_type()) == "EncoderPipeline", "encoder name");

    auto result = pipeline.encode_ids({1, 2, 3, 0});
    check(result.dim == 8, "encoder output dim = 8");
    check(result.data.size() == 8, "embedding has 8 floats");

    cudaStreamDestroy(stream);
}

static void test_segment_pipeline()
{
    auto engine = build_segment_engine();
    if (!engine) { std::cerr << "SKIP segment\n"; return; }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtf::TrtModule>(engine.get(), stream);
    trtf::SegmentPipeline pipeline(std::move(module));

    check(std::string(pipeline.pipeline_type()) == "SegmentPipeline", "segment name");

    float img[3 * 4 * 4] = {0};
    auto result = pipeline.segment(img, 4, 4);
    check(!result.mask.empty(), "segment output has values");

    cudaStreamDestroy(stream);
}

static void test_sam_pipeline()
{
    auto enc_engine = build_segment_engine();
    auto dec_engine = build_segment_engine();
    if (!enc_engine || !dec_engine) { std::cerr << "SKIP sam\n"; return; }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto enc_mod = std::make_unique<trtf::TrtModule>(enc_engine.get(), stream);
    auto dec_mod = std::make_unique<trtf::TrtModule>(dec_engine.get(), stream);
    trtf::SamPipeline pipeline(std::move(enc_mod), std::move(dec_mod));

    check(std::string(pipeline.pipeline_type()) == "SamPipeline", "sam name");

    float img[3 * 4 * 4] = {0};
    auto result = pipeline.segment(img, 4, 4);
    check(!result.mask.empty(), "sam produces output");

    cudaStreamDestroy(stream);
}

#endif

int main()
{
#if TRTF_HAS_TRT
    test_encoder_pipeline();
    test_segment_pipeline();
    test_sam_pipeline();
#else
    std::cerr << "TRT not available, skipping\n";
#endif
    if (failures > 0) std::cerr << failures << " FAILED\n";
    return failures;
}
