#include "cabi/factories/factories_encoder.h"

#include "runtime/trt/encoder/embedding_backend.h"
#include "runtime/trt/encoder/encoder_backend.h"
#include "runtime/trt/multimodal/image_preprocessor.h"
#include "runtime/trt/encoder/reranking_backend.h"
#include "runtime/trt/multimodal/vision_engine.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtf {
namespace cabi {

#if TRTF_HAS_TRT

namespace {

constexpr std::size_t kBytesPerMb = 1024 * 1024;

bool has_data(const std::vector<char>* data)
{
    return data != nullptr && !data->empty();
}

void set_bundle_temp_dir_if_present(
    trtf::IPipeline* pipeline, std::string&& temp_dir)
{
    if (!temp_dir.empty())
    {
        detail::set_bundle_temp_dir(pipeline, std::move(temp_dir));
    }
}

trtf::TokenizerResult try_extract_tokenizer(
    const trtf::BundleSections& sections,
    const std::string& hf_python,
    const char* model_label)
{
    trtf::TokenizerResult tok = {nullptr, ""};
    try
    {
        tok = trtf::extract_tokenizer_from_bundle(
            sections, hf_python, /*add_special_tokens=*/true);
    }
    catch (const std::exception& e)
    {
        std::cerr << "[trtf] Warning: no tokenizer for " << model_label
                  << " (" << e.what() << ")" << std::endl;
    }
    return tok;
}

std::string section_to_string(const std::vector<char>* data)
{
    if (!has_data(data))
    {
        return {};
    }
    return {data->begin(), data->end()};
}

trtf::VLPreprocessConfig parse_vl_preprocess_from_sections(
    const trtf::BundleSections& sections)
{
    auto config_text_vl = section_to_string(sections.config_json_data);
    auto preproc_text = section_to_string(sections.preprocessor_config_data);
    return trtf::parse_vl_preprocess_config(config_text_vl, preproc_text);
}

template <typename EmbeddingBackendT>
void maybe_set_embedding_vision_engine(
    EmbeddingBackendT& emb_backend,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const trtf::BundleSections& sections,
    const trtf::FastPathModelConfig& fp_cfg,
    const std::string& bundle_path)
{
    if (!has_data(sections.vision_plan_data))
    {
        return;
    }

    std::cerr << "[trtf] Deserializing vision TRT engine for embedding ("
              << sections.vision_plan_data->size() / kBytesPerMb
              << " MB) ..." << std::endl;

    auto vision_trt_engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime_ptr->deserializeCudaEngine(
            sections.vision_plan_data->data(),
            sections.vision_plan_data->size()));
    if (!vision_trt_engine)
    {
        std::cerr << "[trtf] Warning: failed to deserialize vision engine for embedding: "
                  << bundle_path << std::endl;
        return;
    }

    auto vision_exec_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
        vision_trt_engine->createExecutionContext());
    if (!vision_exec_ctx)
    {
        return;
    }

    auto vision_step_engine = std::make_unique<trtf::VisionStepEngine>();
    vision_step_engine->engine = std::move(vision_trt_engine);
    vision_step_engine->context = std::move(vision_exec_ctx);
    vision_step_engine->pixel_input_name = "pixel_values";
    vision_step_engine->features_output_name = "vision_features";
    vision_step_engine->num_output_features = fp_cfg.num_image_pad_tokens;
    vision_step_engine->feature_dim = (fp_cfg.vision_output_dim > 0)
        ? fp_cfg.vision_output_dim
        : fp_cfg.hidden_size;

    auto vl_config = parse_vl_preprocess_from_sections(sections);
    emb_backend.set_vision_engine(
        std::move(vision_step_engine), std::move(vl_config));
    std::cerr << "[trtf] VL vision engine loaded for embedding" << std::endl;
}

} // namespace

trtf::IPipeline* create_encoder_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& /* bundle_path */)
{
    auto enc_backend = trtf::CreateEncoderBackend(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!enc_backend || !enc_backend->is_available())
    {
        throw std::runtime_error("Failed to create encoder backend from bundle engine");
    }

    auto tok = try_extract_tokenizer(sections, hf_python, "encoder");

    auto* pipeline = detail::create_encoder_pipeline_impl(
        model_id, std::move(tok.tokenizer), "trt_encoder");
    set_bundle_temp_dir_if_present(pipeline, std::move(tok.temp_dir));
    detail::set_encoder_backend(pipeline, std::move(enc_backend));

    std::cerr << "[trtf] Runtime ready (backend=trt_encoder, strategy=encoder_only)" << std::endl;
    return pipeline;
}

trtf::IPipeline* create_embedding_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    auto emb_backend = trtf::CreateEmbeddingBackend(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!emb_backend || !emb_backend->is_available())
    {
        throw std::runtime_error("Failed to create embedding backend from bundle engine");
    }

    maybe_set_embedding_vision_engine(
        *emb_backend, runtime_ptr, sections, fp_cfg, bundle_path);
    auto tok = try_extract_tokenizer(sections, hf_python, "embedding");

    auto* pipeline = detail::create_encoder_pipeline_impl(
        model_id, std::move(tok.tokenizer), "trt_embedding");
    set_bundle_temp_dir_if_present(pipeline, std::move(tok.temp_dir));
    detail::set_embedding_backend(pipeline, std::move(emb_backend));

    std::cerr << "[trtf] Runtime ready (backend=trt_embedding, strategy=embedding)" << std::endl;
    return pipeline;
}

trtf::IPipeline* create_reranking_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& /* bundle_path */)
{
    auto rerank_backend = trtf::CreateRerankingBackend(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!rerank_backend || !rerank_backend->is_available())
    {
        throw std::runtime_error("Failed to create reranking backend from bundle engine");
    }

    auto tok = try_extract_tokenizer(sections, hf_python, "reranking");

    auto* pipeline = detail::create_encoder_pipeline_impl(
        model_id, std::move(tok.tokenizer), "trt_reranking");
    set_bundle_temp_dir_if_present(pipeline, std::move(tok.temp_dir));
    detail::set_reranking_backend(pipeline, std::move(rerank_backend));

    std::cerr << "[trtf] Runtime ready (backend=trt_reranking, strategy=reranking)" << std::endl;
    return pipeline;
}

#endif // TRTF_HAS_TRT

} // namespace cabi
} // namespace trtf
