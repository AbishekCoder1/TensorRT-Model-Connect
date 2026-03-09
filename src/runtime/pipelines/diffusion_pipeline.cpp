#include "runtime/pipelines/diffusion_pipeline.h"

#if TRTF_HAS_TRT

#include "runtime/trt/diffusion/diffusion_backend.h"
#include "trtf/tokenizer.h"

#include <stdexcept>

namespace trtf {

namespace {

// Convert old-style VideoResult -> public ImageResult.
ImageResult video_to_image(const VideoResult& vr, int32_t default_h, int32_t default_w)
{
    ImageResult out;
    out.pixels = vr.frames;
    out.height = (vr.height > 0) ? vr.height : default_h;
    out.width = (vr.width > 0) ? vr.width : default_w;
    out.channels = 3;
    out.num_frames = (vr.num_frames > 0) ? vr.num_frames : 1;
    return out;
}

// Run diffusion generation: tokenize prompt, call backend generate_video.
ImageResult run_diffusion(
    IDiffusionBackend& backend,
    ITokenizer* tokenizer,
    const std::string& prompt,
    const GenerateConfig& cfg,
    int32_t default_h,
    int32_t default_w)
{
    // Prepare and tokenize the prompt
    std::string prepared = backend.prepare_prompt(prompt);
    backend.set_prompt(prepared);

    std::vector<int32_t> input_ids;
    if (tokenizer)
        input_ids = tokenizer->encode(prepared);

    int32_t num_steps = (cfg.num_steps > 0) ? cfg.num_steps : -1;
    float guidance = (cfg.guidance_scale >= 0.0f) ? cfg.guidance_scale : -1.0f;

    auto vr = backend.generate_video(input_ids, num_steps, guidance);
    return video_to_image(vr, default_h, default_w);
}

} // namespace

// ─── FluxPipeline ───

FluxPipeline::FluxPipeline(
    std::unique_ptr<IDiffusionBackend> backend,
    std::shared_ptr<ITokenizer> tokenizer,
    int32_t image_height,
    int32_t image_width,
    std::string model_id_str)
    : backend_(std::move(backend))
    , tokenizer_(std::move(tokenizer))
    , image_height_(image_height)
    , image_width_(image_width)
    , model_id_(std::move(model_id_str))
{
    if (!backend_ || !backend_->is_available())
        throw std::runtime_error("FluxPipeline: invalid diffusion backend");
}

FluxPipeline::~FluxPipeline() = default;

ImageResult FluxPipeline::generate_image(
    const std::string& prompt, const GenerateConfig& cfg)
{
    return run_diffusion(*backend_, tokenizer_.get(), prompt, cfg,
                         image_height_, image_width_);
}

// ─── WanPipeline ───

WanPipeline::WanPipeline(
    std::unique_ptr<IDiffusionBackend> backend,
    std::shared_ptr<ITokenizer> tokenizer,
    int32_t image_height,
    int32_t image_width,
    std::string model_id_str)
    : backend_(std::move(backend))
    , tokenizer_(std::move(tokenizer))
    , image_height_(image_height)
    , image_width_(image_width)
    , model_id_(std::move(model_id_str))
{
    if (!backend_ || !backend_->is_available())
        throw std::runtime_error("WanPipeline: invalid diffusion backend");
}

WanPipeline::~WanPipeline() = default;

ImageResult WanPipeline::generate_image(
    const std::string& prompt, const GenerateConfig& cfg)
{
    return run_diffusion(*backend_, tokenizer_.get(), prompt, cfg,
                         image_height_, image_width_);
}

// ─── ZImagePipeline ───

ZImagePipeline::ZImagePipeline(
    std::unique_ptr<IDiffusionBackend> backend,
    std::shared_ptr<ITokenizer> tokenizer,
    int32_t image_height,
    int32_t image_width,
    std::string model_id_str)
    : backend_(std::move(backend))
    , tokenizer_(std::move(tokenizer))
    , image_height_(image_height)
    , image_width_(image_width)
    , model_id_(std::move(model_id_str))
{
    if (!backend_ || !backend_->is_available())
        throw std::runtime_error("ZImagePipeline: invalid diffusion backend");
}

ZImagePipeline::~ZImagePipeline() = default;

ImageResult ZImagePipeline::generate_image(
    const std::string& prompt, const GenerateConfig& cfg)
{
    return run_diffusion(*backend_, tokenizer_.get(), prompt, cfg,
                         image_height_, image_width_);
}

} // namespace trtf

#endif // TRTF_HAS_TRT
