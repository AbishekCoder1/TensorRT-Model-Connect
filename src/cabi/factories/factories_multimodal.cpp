#include "cabi/factories/factories_multimodal.h"

#include "cabi/factories/factory_decls.h"
#include "cabi/factories/factories_audio.h"
#include "cabi/factories/factories_text.h"
#include "runtime/trt/multimodal/image_preprocessor.h"
#include "runtime/trt/multimodal/vision_engine.h"
#include "runtime/trt/multimodal/vl_backend.h"
#include "runtime/trt/audio/whisper_backend.h"

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

std::pair<trtf::TrtUniquePtr<nvinfer1::ICudaEngine>,
          trtf::TrtUniquePtr<nvinfer1::IExecutionContext>>
load_optional_whisper_encoder(
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const trtf::BundleSections& sections,
    const std::string& bundle_path)
{
    if (!has_data(sections.vision_plan_data))
    {
        return {};
    }

    std::cerr << "[trtf] Deserializing Whisper encoder TRT engine ("
              << sections.vision_plan_data->size() / kBytesPerMb
              << " MB) ..." << std::endl;
    auto encoder_engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime_ptr->deserializeCudaEngine(
            sections.vision_plan_data->data(),
            sections.vision_plan_data->size()));
    if (!encoder_engine)
    {
        throw std::runtime_error(
            "Failed to deserialize Whisper encoder engine: " + bundle_path);
    }

    auto encoder_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
        encoder_engine->createExecutionContext());
    if (!encoder_ctx)
    {
        throw std::runtime_error(
            "Failed to create Whisper encoder execution context");
    }
    return {std::move(encoder_engine), std::move(encoder_ctx)};
}

trtf::WhisperConfig make_whisper_config(
    const trtf::FastPathModelConfig& fp_cfg)
{
    trtf::WhisperConfig whisper_cfg;
    whisper_cfg.num_mel_bins = fp_cfg.num_mel_bins;
    whisper_cfg.max_source_positions = fp_cfg.max_source_positions;
    whisper_cfg.max_target_positions = fp_cfg.max_target_positions;
    whisper_cfg.encoder_layers = fp_cfg.encoder_layers;
    whisper_cfg.decoder_layers = fp_cfg.decoder_layers;
    whisper_cfg.mel_length = fp_cfg.mel_length;
    whisper_cfg.decoder_start_token_ids = fp_cfg.decoder_start_token_ids;
    if (fp_cfg.eot_token_id >= 0)
    {
        whisper_cfg.eot_token_id = fp_cfg.eot_token_id;
    }
    return whisper_cfg;
}

std::string section_to_string(const std::vector<char>* data)
{
    if (!has_data(data))
    {
        return {};
    }
    return {data->begin(), data->end()};
}

std::unique_ptr<trtf::VisionStepEngine> load_optional_vl_vision_engine(
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const trtf::BundleSections& sections,
    const trtf::FastPathModelConfig& fp_cfg,
    const std::string& bundle_path)
{
    if (!has_data(sections.vision_plan_data))
    {
        return nullptr;
    }

    std::cerr << "[trtf] Deserializing vision TRT engine from bundle ("
              << sections.vision_plan_data->size() / kBytesPerMb
              << " MB) ..." << std::endl;
    auto vision_trt_engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime_ptr->deserializeCudaEngine(
            sections.vision_plan_data->data(),
            sections.vision_plan_data->size()));
    if (!vision_trt_engine)
    {
        throw std::runtime_error(
            "Failed to deserialize vision engine from bundle: " + bundle_path);
    }

    auto vision_exec_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
        vision_trt_engine->createExecutionContext());
    if (!vision_exec_ctx)
    {
        throw std::runtime_error("Failed to create vision execution context");
    }

    auto vision_step_engine = std::make_unique<trtf::VisionStepEngine>();
    vision_step_engine->engine = std::move(vision_trt_engine);
    vision_step_engine->context = std::move(vision_exec_ctx);
    vision_step_engine->num_output_features = fp_cfg.num_image_pad_tokens;
    vision_step_engine->feature_dim = (fp_cfg.vision_output_dim > 0)
        ? fp_cfg.vision_output_dim
        : fp_cfg.hidden_size;

    std::cerr << "[trtf] Vision engine deserialized (features="
              << vision_step_engine->num_output_features
              << ", dim=" << vision_step_engine->feature_dim
              << ")" << std::endl;
    return vision_step_engine;
}

trtf::VLPreprocessConfig parse_vl_preprocess_from_bundle(
    const trtf::BundleSections& sections)
{
    auto config_text_vl = section_to_string(sections.config_json_data);
    auto preproc_text = section_to_string(sections.preprocessor_config_data);
    return trtf::parse_vl_preprocess_config(config_text_vl, preproc_text);
}

} // namespace

trtf::IPipeline* create_whisper_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    auto decoder_engine = trtf::make_decoder_engine(std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!trtf::has_all_required_tensors(*decoder_engine))
    {
        throw std::runtime_error("Bundle engine missing required decoder tensors: " + bundle_path);
    }

    auto [encoder_engine, encoder_ctx] = load_optional_whisper_encoder(
        runtime_ptr, sections, bundle_path);

    auto tok = trtf::extract_tokenizer_from_bundle(sections, hf_python);
    auto backend = trtf::CreateWhisperBackend(
        std::move(decoder_engine), std::move(encoder_engine), std::move(encoder_ctx),
        make_whisper_config(fp_cfg), fp_cfg);
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create Whisper TRT backend from bundle engine");
    }

    auto mel_fb = trtf::load_mel_filterbank(sections);
    if (!mel_fb.data.empty())
    {
        std::cerr << "[trtf] Loaded mel filterbank from bundle: "
                  << mel_fb.n_freq_bins << "x" << mel_fb.n_mel_bins << std::endl;
    }

    std::cerr << "[trtf] Runtime ready (backend=trt_whisper, strategy=speech_to_text)" << std::endl;

    auto* pipeline = detail::create_pipeline_impl(
        model_id, std::move(tok.tokenizer), nullptr, "trt_whisper");
    set_bundle_temp_dir_if_present(pipeline, std::move(tok.temp_dir));
    detail::set_whisper_backend(pipeline, std::move(backend), std::move(mel_fb), fp_cfg);
    detail::set_hf_python(pipeline, hf_python);
    return pipeline;
}

trtf::IPipeline* create_vl_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    auto decoder_engine = trtf::make_decoder_engine(std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!trtf::has_all_required_tensors(*decoder_engine))
    {
        throw std::runtime_error("Bundle engine missing required tensors: " + bundle_path);
    }

    auto vision_step_engine = load_optional_vl_vision_engine(
        runtime_ptr, sections, fp_cfg, bundle_path);
    auto vl_preproc = parse_vl_preprocess_from_bundle(sections);

    auto tok = trtf::extract_tokenizer_from_bundle(
        sections, hf_python, fp_cfg.tokenizer_add_special_tokens);
    auto backend = trtf::CreateVLBackendFromEngines(
        std::move(decoder_engine), std::move(vision_step_engine), fp_cfg, std::move(vl_preproc));
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create VL TRT backend from bundle engine");
    }

    std::cerr << "[trtf] Runtime ready (backend=trt_vl, strategy=vision_language)" << std::endl;
    return detail::create_text_pipeline_impl(
        model_id, std::move(tok), std::move(backend), "trt_vl");
}

namespace detail {

trtf::IPipeline* create_whisper_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    return trtf::cabi::create_whisper_pipeline(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg, sections,
        runtime_ptr, model_id, hf_python, bundle_path);
}

trtf::IPipeline* create_vl_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    return trtf::cabi::create_vl_pipeline(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg, sections,
        runtime_ptr, model_id, hf_python, bundle_path);
}

} // namespace detail

#endif // TRTF_HAS_TRT

} // namespace cabi
} // namespace trtf
