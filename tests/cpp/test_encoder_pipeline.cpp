// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-SEG-CPP-01
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-FAC-01
// Intent:         EncoderPipeline, SegmentPipeline, and SamPipeline construction,
//                 type checks, embed/encode/rerank paths, int32 mask branch,
//                 constructor validation, argmax_class_map coverage,
//                 no-tokenizer throws, score-named output, 4D segmentation,
//                 and single-output mask branch in find_segmentation_output
// Preconditions:  TRT headers and CUDA available
// Postconditions: Pipelines construct with mock engines and expose correct interfaces;
//                 embed/encode/rerank methods return non-empty results;
//                 invalid inputs are rejected with std::exception;
//                 score output name, 4D logits shape, and size==1 output branch covered
// =============================================================================

// =============================================================================
// Test suite: EncoderPipeline + SegmentPipeline + SamPipeline
// =============================================================================

#include "runtime/pipelines/encoder_pipeline.h"
#include "runtime/pipelines/sam_pipeline.h"
#include "runtime/pipelines/segment_pipeline.h"
#include "trtf/runtime/trt_module.h"
#include "trtf/tokenizer.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "runtime/backend/trt_module_impl.h"
#include "runtime/core/trt_common.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

static int failures = 0;
static void check(bool c, const char* n) {
    if (!c) {
        std::cerr << "FAIL: " << n << '\n';
        ++failures;
    }
}


static trtf::TrtLogger g_logger;

// ---------------------------------------------------------------------------
// Inline FixedTokenizer — encodes any string as {1,2,3,4}
// ---------------------------------------------------------------------------
class FixedTokenizer : public trtf::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string&) const override { return {1, 2, 3, 4}; }
    std::string decode(const std::vector<int32_t>&) const override { return "test"; }
    int32_t id_for_token(std::string_view) const override { return 0; }
    std::string token_for_id(int32_t) const override { return ""; }
};

// ---------------------------------------------------------------------------
// Engine builders
// ---------------------------------------------------------------------------

// Mock: input_ids[4] int32 + attention_mask[4] float32 -> output_embeddings[8] float (flat)
static trtf::TrtUniquePtr<nvinfer1::ICudaEngine> build_encoder_engine() {
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
    if (!plan)
        return nullptr;
    auto rt = trtf::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

// Mock: input_ids[4] int32 + attention_mask[4] float32 -> output_hidden[4,2] float (2D)
// infer_output_hidden_dim returns 2 (last axis of [4,2])
static trtf::TrtUniquePtr<nvinfer1::ICudaEngine> build_encoder_engine_2d() {
    auto b = trtf::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtf::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtf::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* ids = n->addInput("input_ids", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {4}});
    auto* mask = n->addInput("attention_mask", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {4}});

    // Output shape [4, 2] so infer_output_hidden_dim returns 2
    float cv[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    auto* cst = n->addConstant(nvinfer1::Dims{2, {4, 2}},
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cv, 8});
    cst->getOutput(0)->setName("output_hidden");
    n->markOutput(*cst->getOutput(0));

    n->addIdentity(*ids)->getOutput(0)->setName("_i");
    n->addIdentity(*mask)->getOutput(0)->setName("_m");

    auto plan = trtf::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan)
        return nullptr;
    auto rt = trtf::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

// Mock: input_ids[4] int32 + attention_mask[4] int32 -> output_hidden[8] float
// Used to cover the int32 mask branch in encode_ids()
static trtf::TrtUniquePtr<nvinfer1::ICudaEngine> build_encoder_engine_int32_mask() {
    auto b = trtf::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtf::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtf::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    // INT32 attention_mask to exercise engine_mask_is_int32() == true branch
    auto* ids = n->addInput("input_ids", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {4}});
    auto* mask = n->addInput("attention_mask", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {4}});

    float cv[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    auto* cst = n->addConstant(nvinfer1::Dims{1, {8}},
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cv, 8});
    cst->getOutput(0)->setName("output_hidden");
    n->markOutput(*cst->getOutput(0));

    n->addIdentity(*ids)->getOutput(0)->setName("_i");
    n->addIdentity(*mask)->getOutput(0)->setName("_m");

    auto plan = trtf::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan)
        return nullptr;
    auto rt = trtf::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

// Mock: pixel_values[3,4,4] float -> output_mask[1,16] float (flat output, no class dim)
static trtf::TrtUniquePtr<nvinfer1::ICudaEngine> build_segment_engine() {
    auto b = trtf::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtf::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtf::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* pv =
        n->addInput("pixel_values", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{3, {3, 4, 4}});

    float cv[16];
    for (int i = 0; i < 16; ++i)
        cv[i] = static_cast<float>(i);
    auto* cst = n->addConstant(nvinfer1::Dims{1, {16}},
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cv, 16});
    cst->getOutput(0)->setName("output_mask");
    n->markOutput(*cst->getOutput(0));

    n->addIdentity(*pv)->getOutput(0)->setName("_pv");

    auto plan = trtf::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan)
        return nullptr;
    auto rt = trtf::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

// Mock: input_ids[4] int32 (no attention_mask) -> score[1] float
// Covers: engine_mask_is_int32() return false (line 52) and name.find("score") (line 164)
static trtf::TrtUniquePtr<nvinfer1::ICudaEngine> build_encoder_engine_score_output() {
    auto b = trtf::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtf::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtf::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    // Only input_ids — NO attention_mask — so engine_mask_is_int32 returns false (line 52)
    auto* ids = n->addInput("input_ids", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {4}});

    float cv[1] = {0.5f};
    auto* cst = n->addConstant(nvinfer1::Dims{1, {1}},
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cv, 1});
    cst->getOutput(0)->setName("score"); // name.find("score") covers line 164
    n->markOutput(*cst->getOutput(0));

    n->addIdentity(*ids)->getOutput(0)->setName("_i");

    auto plan = trtf::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan)
        return nullptr;
    auto rt = trtf::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

// Mock: pixel_values[3,4,4] float -> mask[1] float (single output not named logits/output)
// Covers: find_segmentation_output() outputs.size()==1 branch (line 238)
static trtf::TrtUniquePtr<nvinfer1::ICudaEngine> build_segment_engine_mask_output() {
    auto b = trtf::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtf::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtf::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* pv =
        n->addInput("pixel_values", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{3, {3, 4, 4}});

    float cv[1] = {1.0f};
    auto* cst = n->addConstant(nvinfer1::Dims{1, {1}},
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cv, 1});
    cst->getOutput(0)->setName("mask"); // NOT "logits" or "output" — triggers size()==1 branch
    n->markOutput(*cst->getOutput(0));

    n->addIdentity(*pv)->getOutput(0)->setName("_pv");

    auto plan = trtf::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan)
        return nullptr;
    auto rt = trtf::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

// Mock: pixel_values[3,4,4] float -> logits[1,2,4,4] float (4D output)
// Covers: parse_segmentation_shape shape.size()==4 branch (lines 214-216)
static trtf::TrtUniquePtr<nvinfer1::ICudaEngine> build_segment_engine_4d() {
    auto b = trtf::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtf::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtf::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* pv =
        n->addInput("pixel_values", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{3, {3, 4, 4}});

    // 1*2*4*4 = 32 values — shape [1,2,4,4] triggers the 4D branch
    float cv[32];
    for (int i = 0; i < 32; ++i)
        cv[i] = static_cast<float>(i % 2);
    auto* cst = n->addConstant(nvinfer1::Dims{4, {1, 2, 4, 4}},
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cv, 32});
    cst->getOutput(0)->setName("logits");
    n->markOutput(*cst->getOutput(0));

    n->addIdentity(*pv)->getOutput(0)->setName("_pv");

    auto plan = trtf::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan)
        return nullptr;
    auto rt = trtf::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

// Mock: pixel_values[3,4,4] float -> logits[2,4,4] float (2 classes, 4x4 spatial)
// parse_segmentation_shape succeeds -> argmax_class_map is called
static trtf::TrtUniquePtr<nvinfer1::ICudaEngine> build_segment_engine_2d() {
    auto b = trtf::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtf::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtf::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* pv =
        n->addInput("pixel_values", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{3, {3, 4, 4}});

    // 2 * 4 * 4 = 32 values, alternating 0.0/1.0
    float cv[32];
    for (int i = 0; i < 32; ++i)
        cv[i] = static_cast<float>(i % 2);
    auto* cst = n->addConstant(nvinfer1::Dims{3, {2, 4, 4}},
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cv, 32});
    cst->getOutput(0)->setName("logits");
    n->markOutput(*cst->getOutput(0));

    n->addIdentity(*pv)->getOutput(0)->setName("_pv");

    auto plan = trtf::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan)
        return nullptr;
    auto rt = trtf::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_encoder_pipeline() {
    auto engine = build_encoder_engine();
    if (!engine) {
        std::cerr << "SKIP encoder\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtf::TrtModuleImpl>(engine.get(),
                                                        engine->createExecutionContext(), stream);
    trtf::EncoderPipeline pipeline(std::move(module), "embedding");

    check(std::string(pipeline.pipeline_type()) == "EncoderPipeline", "encoder name");

    auto result = pipeline.encode_ids({1, 2, 3, 0});
    check(result.dim == 8, "encoder output dim = 8");
    check(result.data.size() == 8, "embedding has 8 floats");

    cudaStreamDestroy(stream);
}

static void test_encoder_embed_mode() {
    auto engine = build_encoder_engine_2d();
    if (!engine) {
        std::cerr << "SKIP encoder_embed\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtf::TrtModuleImpl>(engine.get(),
                                                        engine->createExecutionContext(), stream);
    auto tokenizer = std::make_shared<FixedTokenizer>();
    trtf::EncoderPipeline pipeline(std::move(module), "embedding", tokenizer);

    // embed() calls tokenizer->encode() -> {1,2,3,4}, then encode_ids, then
    // mean_pool_and_normalize(data, 4, 2) since mode_=="embedding" and raw.dim(8) >= 4*2
    auto result = pipeline.embed("hello");
    check(!result.data.empty(), "embed: result has data");
    check(result.dim == 2, "embed: hidden dim = 2 after mean-pool");

    cudaStreamDestroy(stream);
}

static void test_encoder_encode_mode() {
    auto engine = build_encoder_engine_2d();
    if (!engine) {
        std::cerr << "SKIP encoder_encode\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtf::TrtModuleImpl>(engine.get(),
                                                        engine->createExecutionContext(), stream);
    auto tokenizer = std::make_shared<FixedTokenizer>();
    trtf::EncoderPipeline pipeline(std::move(module), "encode", tokenizer);

    // encode() extracts CLS token: first hidden_dim(=2) values from raw output
    auto result = pipeline.encode("hello");
    check(result.dim == 2, "encode: CLS dim = 2");
    check(result.data.size() == 2, "encode: data size = 2");

    cudaStreamDestroy(stream);
}

static void test_encoder_rerank() {
    auto engine = build_encoder_engine_2d();
    if (!engine) {
        std::cerr << "SKIP encoder_rerank\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtf::TrtModuleImpl>(engine.get(),
                                                        engine->createExecutionContext(), stream);
    auto tokenizer = std::make_shared<FixedTokenizer>();
    trtf::EncoderPipeline pipeline(std::move(module), "rerank", tokenizer);

    // rerank() concatenates query + " [SEP] " + document, encodes, returns data[0]
    float score = pipeline.rerank("query", "doc");
    // Score is a float (from engine constant output = 0.1f at index 0)
    check(score >= -1e6f && score <= 1e6f, "rerank: returns a finite float");

    cudaStreamDestroy(stream);
}

static void test_encoder_int32_mask() {
    auto engine = build_encoder_engine_int32_mask();
    if (!engine) {
        std::cerr << "SKIP encoder_int32_mask\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtf::TrtModuleImpl>(engine.get(),
                                                        engine->createExecutionContext(), stream);
    auto tokenizer = std::make_shared<FixedTokenizer>();
    // mode="embedding" with int32 mask covers engine_mask_is_int32() == true path
    trtf::EncoderPipeline pipeline(std::move(module), "embedding", tokenizer);

    auto result = pipeline.embed("hello");
    check(!result.data.empty(), "int32_mask: result has data");

    cudaStreamDestroy(stream);
}

static void test_encoder_validates() {
    // Null encoder -> constructor throws std::exception
    bool threw = false;
    try {
        trtf::EncoderPipeline pipeline(nullptr, "embedding");
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "encoder: null encoder throws");
}

static void test_segment_pipeline() {
    auto engine = build_segment_engine();
    if (!engine) {
        std::cerr << "SKIP segment\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtf::TrtModuleImpl>(engine.get(),
                                                        engine->createExecutionContext(), stream);
    trtf::SegmentPipeline pipeline(std::move(module));

    check(std::string(pipeline.pipeline_type()) == "SegmentPipeline", "segment name");

    float img[3 * 4 * 4] = {0};
    auto result = pipeline.segment(img, 4, 4);
    check(!result.mask.empty(), "segment output has values");

    cudaStreamDestroy(stream);
}

static void test_segment_with_class_output() {
    // Engine with logits[2,4,4] -> parse_segmentation_shape succeeds -> argmax_class_map called
    auto engine = build_segment_engine_2d();
    if (!engine) {
        std::cerr << "SKIP segment_2d\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtf::TrtModuleImpl>(engine.get(),
                                                        engine->createExecutionContext(), stream);
    trtf::SegmentPipeline pipeline(std::move(module));

    float img[3 * 4 * 4] = {0};
    auto result = pipeline.segment(img, 4, 4);
    // Output shape [2,4,4]: num_classes=2, out_h=4, out_w=4 -> mask has 16 entries
    check(result.mask.size() == 16, "segment 2d: mask has 16 entries (4x4)");
    check(result.height == 4, "segment 2d: height = 4");
    check(result.width == 4, "segment 2d: width = 4");

    cudaStreamDestroy(stream);
}

static void test_segment_validates() {
    // Null model -> constructor throws
    bool threw = false;
    try {
        trtf::SegmentPipeline pipeline(nullptr);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "segment: null model throws");
}

static void test_sam_pipeline() {
    auto enc_engine = build_segment_engine();
    auto dec_engine = build_segment_engine();
    if (!enc_engine || !dec_engine) {
        std::cerr << "SKIP sam\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto enc_mod = std::make_unique<trtf::TrtModuleImpl>(
        enc_engine.get(), enc_engine->createExecutionContext(), stream);
    auto dec_mod = std::make_unique<trtf::TrtModuleImpl>(
        dec_engine.get(), dec_engine->createExecutionContext(), stream);
    trtf::SamPipeline pipeline(std::move(enc_mod), std::move(dec_mod));

    check(std::string(pipeline.pipeline_type()) == "SamPipeline", "sam name");

    float img[3 * 4 * 4] = {0};
    auto result = pipeline.segment(img, 4, 4);
    check(!result.mask.empty(), "sam produces output");

    cudaStreamDestroy(stream);
}

static void test_sam_validates() {
    // Null encoder -> constructor throws
    bool threw = false;
    try {
        trtf::SamPipeline pipeline(nullptr, nullptr);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "sam: null encoder throws");
}

static void test_encoder_score_output() {
    // Engine with output "score" and no attention_mask input covers:
    //   engine_mask_is_int32() return false path (line 52)
    //   name.find("score") evaluation in encode_ids (line 164)
    auto engine = build_encoder_engine_score_output();
    if (!engine) {
        std::cerr << "SKIP encoder_score\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtf::TrtModuleImpl>(engine.get(),
                                                        engine->createExecutionContext(), stream);
    trtf::EncoderPipeline pipeline(std::move(module), "embedding");

    auto result = pipeline.encode_ids({1, 2, 3, 4});
    check(!result.data.empty(), "score output: data not empty");
    check(result.dim == 1, "score output: dim=1");

    cudaStreamDestroy(stream);
}

static void test_encoder_no_tokenizer_embed() {
    // embed() without tokenizer covers line 75 throw
    auto engine = build_encoder_engine();
    if (!engine) {
        std::cerr << "SKIP encoder_no_tok_embed\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtf::TrtModuleImpl>(engine.get(),
                                                        engine->createExecutionContext(), stream);
    trtf::EncoderPipeline pipeline(std::move(module), "embedding"); // no tokenizer

    bool threw = false;
    try {
        pipeline.embed("hello");
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "embed: no tokenizer throws");

    cudaStreamDestroy(stream);
}

static void test_encoder_no_tokenizer_encode() {
    // encode() without tokenizer covers line 98 throw
    auto engine = build_encoder_engine();
    if (!engine) {
        std::cerr << "SKIP encoder_no_tok_encode\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtf::TrtModuleImpl>(engine.get(),
                                                        engine->createExecutionContext(), stream);
    trtf::EncoderPipeline pipeline(std::move(module), "encode"); // no tokenizer

    bool threw = false;
    try {
        pipeline.encode("hello");
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "encode: no tokenizer throws");

    cudaStreamDestroy(stream);
}

static void test_encoder_no_tokenizer_rerank() {
    // rerank() without tokenizer covers line 117 throw
    auto engine = build_encoder_engine();
    if (!engine) {
        std::cerr << "SKIP encoder_no_tok_rerank\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtf::TrtModuleImpl>(engine.get(),
                                                        engine->createExecutionContext(), stream);
    trtf::EncoderPipeline pipeline(std::move(module), "rerank"); // no tokenizer

    bool threw = false;
    try {
        pipeline.rerank("q", "d");
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "rerank: no tokenizer throws");

    cudaStreamDestroy(stream);
}

static void test_segment_4d_output() {
    // 4D logits[1,2,4,4] covers parse_segmentation_shape shape.size()==4 branch (lines 214-216)
    auto engine = build_segment_engine_4d();
    if (!engine) {
        std::cerr << "SKIP segment_4d\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtf::TrtModuleImpl>(engine.get(),
                                                        engine->createExecutionContext(), stream);
    trtf::SegmentPipeline pipeline(std::move(module));

    float img[3 * 4 * 4] = {0};
    auto result = pipeline.segment(img, 4, 4);
    // shape [1,2,4,4]: num_classes=2, out_h=4, out_w=4 -> 16 mask entries
    check(result.mask.size() == 16, "segment 4d: mask has 16 entries");
    check(result.height == 4, "segment 4d: height = 4");
    check(result.width == 4, "segment 4d: width = 4");

    cudaStreamDestroy(stream);
}

static void test_segment_mask_named_output() {
    // Single output "mask" (not "logits"/"output") covers outputs.size()==1 branch (line 238)
    auto engine = build_segment_engine_mask_output();
    if (!engine) {
        std::cerr << "SKIP segment_mask\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtf::TrtModuleImpl>(engine.get(),
                                                        engine->createExecutionContext(), stream);
    trtf::SegmentPipeline pipeline(std::move(module));

    float img[3 * 4 * 4] = {0};
    auto result = pipeline.segment(img, 4, 4);
    // Single flat output "mask" -> parse_segmentation_shape fails (size=1) -> raw copy path
    check(result.mask.size() == 1, "segment mask: 1 value in mask");

    cudaStreamDestroy(stream);
}


int main() {
    test_encoder_pipeline();
    test_encoder_embed_mode();
    test_encoder_encode_mode();
    test_encoder_rerank();
    test_encoder_int32_mask();
    test_encoder_validates();
    test_encoder_score_output();
    test_encoder_no_tokenizer_embed();
    test_encoder_no_tokenizer_encode();
    test_encoder_no_tokenizer_rerank();
    test_segment_pipeline();
    test_segment_with_class_output();
    test_segment_validates();
    test_segment_4d_output();
    test_segment_mask_named_output();
    test_sam_pipeline();
    test_sam_validates();
    if (failures > 0)
        std::cerr << failures << " FAILED\n";
    return failures;
}
