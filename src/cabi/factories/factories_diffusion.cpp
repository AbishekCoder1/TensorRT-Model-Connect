#include "cabi/factories/factories_diffusion.h"

#include "runtime/trt/diffusion/diffusion_backend.h"
#include "runtime/trt/diffusion/z_image_diffusion_backend.h"

#include <iostream>
#include <stdexcept>
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

std::vector<trtf::DiffusionEngine> load_text_encoder_engines(
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr)
{
    std::vector<trtf::DiffusionEngine> text_encoders;
    for (std::size_t i = 0; i < sections.text_encoder_plans.size(); ++i)
    {
        const auto* plan = sections.text_encoder_plans[i];
        if (!has_data(plan))
        {
            continue;
        }

        std::cerr << "[trtf] Deserializing text encoder " << i
                  << " (" << plan->size() / kBytesPerMb << " MB) ..."
                  << std::endl;
        trtf::DiffusionEngine te;
        te.name = "text_encoder_" + std::to_string(i);
        te.engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime_ptr->deserializeCudaEngine(plan->data(), plan->size()));
        if (!te.engine)
        {
            throw std::runtime_error(
                "Failed to deserialize text encoder " + std::to_string(i));
        }

        te.context = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
            te.engine->createExecutionContext());
        if (!te.context)
        {
            throw std::runtime_error(
                "Failed to create text encoder context " + std::to_string(i));
        }
        text_encoders.push_back(std::move(te));
    }
    return text_encoders;
}

trtf::DiffusionEngine load_required_diffusion_engine(
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::vector<char>* plan_data,
    const char* engine_name,
    const char* log_label,
    const std::string& missing_error,
    const std::string& deserialize_error)
{
    if (!has_data(plan_data))
    {
        throw std::runtime_error(missing_error);
    }

    std::cerr << "[trtf] Deserializing " << log_label << " ("
              << plan_data->size() / kBytesPerMb << " MB) ..."
              << std::endl;
    trtf::DiffusionEngine engine;
    engine.name = engine_name;
    engine.engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime_ptr->deserializeCudaEngine(
            plan_data->data(), plan_data->size()));
    if (!engine.engine)
    {
        throw std::runtime_error(deserialize_error);
    }
    engine.context = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
        engine.engine->createExecutionContext());
    return engine;
}

template <typename BackendT>
void maybe_load_preprocessor_weights(
    BackendT& backend, const trtf::BundleSections& sections)
{
    if (!has_data(sections.preprocessor_weights_data))
    {
        return;
    }

    auto pp_weights = trtf::parse_preprocessor_weights(
        *sections.preprocessor_weights_data);
    backend.set_preprocessor_weights(std::move(pp_weights));

    auto* z_image = dynamic_cast<trtf::ZImageDiffusionBackend*>(&backend);
    if (z_image != nullptr)
    {
        z_image->load_z_image_preprocessor_weights(
            *sections.preprocessor_weights_data);
    }
}

template <typename BackendT>
void maybe_load_clip_tokenizer(
    BackendT& backend,
    const trtf::BundleSections& sections,
    const std::string& hf_python)
{
    if (!has_data(sections.clip_vocab_json_data))
    {
        return;
    }

    try
    {
        auto clip_tok = trtf::extract_clip_tokenizer_from_bundle(
            sections, hf_python);
        backend.set_clip_tokenizer(std::move(clip_tok.tokenizer));
        std::cerr << "[trtf] CLIP tokenizer loaded for dual-tokenizer model"
                  << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[trtf] Warning: CLIP tokenizer extraction failed ("
                  << e.what() << ")" << std::endl;
    }
}

trtf::TokenizerResult try_extract_diffusion_tokenizer(
    const trtf::BundleSections& sections, const std::string& hf_python)
{
    trtf::TokenizerResult tok = {nullptr, ""};
    try
    {
        tok = trtf::extract_tokenizer_from_bundle(
            sections, hf_python, /*add_special_tokens=*/true);
    }
    catch (const std::exception& e)
    {
        std::cerr << "[trtf] Warning: no tokenizer in bundle ("
                  << e.what() << ")" << std::endl;
    }
    return tok;
}

template <typename BackendT>
trtf::IPipeline* create_diffusion_pipeline_with_tokenizer(
    const std::string& model_id,
    trtf::TokenizerResult&& tok,
    BackendT&& backend)
{
    if (!tok.tokenizer)
    {
        return detail::create_pipeline_impl(
            model_id, nullptr, std::move(backend), "trt_diffusion");
    }

    auto* pipeline = detail::create_pipeline_impl(
        model_id, std::move(tok.tokenizer), std::move(backend),
        "trt_diffusion");
    detail::set_bundle_temp_dir(pipeline, std::move(tok.temp_dir));
    return pipeline;
}

} // namespace

trtf::IPipeline* create_diffusion_pipeline(
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    std::cerr << "[trtf] Creating diffusion pipeline ..." << std::endl;

    auto text_encoders = load_text_encoder_engines(sections, runtime_ptr);
    auto denoiser = load_required_diffusion_engine(
        runtime_ptr,
        sections.denoiser_plan_data,
        "denoiser",
        "denoiser",
        "Bundle has no denoiser_plan section: " + bundle_path,
        "Failed to deserialize denoiser engine");
    auto vae_decoder = load_required_diffusion_engine(
        runtime_ptr,
        sections.vae_decoder_plan_data,
        "vae_decoder",
        "VAE decoder",
        "Bundle has no vae_decoder_plan section: " + bundle_path,
        "Failed to deserialize VAE decoder engine");

    auto backend = trtf::CreateDiffusionBackend(
        std::move(text_encoders), std::move(denoiser), std::move(vae_decoder), fp_cfg);
    if (!backend || !backend->is_available())
        throw std::runtime_error("Failed to create diffusion backend");

    maybe_load_preprocessor_weights(*backend, sections);

    // Set paths for VAE subprocess and bundle info
    backend->set_hf_python(hf_python);
    backend->set_bundle_path(bundle_path);

    maybe_load_clip_tokenizer(*backend, sections, hf_python);
    auto tok = try_extract_diffusion_tokenizer(sections, hf_python);

    std::cerr << "[trtf] Runtime ready (backend=trt_diffusion, strategy=diffusion)" << std::endl;
    return create_diffusion_pipeline_with_tokenizer(
        model_id, std::move(tok), std::move(backend));
}

#endif // TRTF_HAS_TRT

} // namespace cabi
} // namespace trtf
