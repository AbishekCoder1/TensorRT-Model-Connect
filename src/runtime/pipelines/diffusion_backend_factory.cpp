// Creates a diffusion IPipeline from bundle sections.
// Includes old diffusion backend headers and pipeline headers.

#include "runtime/pipelines/diffusion_backend_factory.h"
#include "runtime/pipelines/diffusion_pipeline.h"

#if TRTF_HAS_TRT

#include "runtime/trt/core/trt_common.h"
#include "runtime/trt/diffusion/diffusion_backend.h"
#include "runtime/trt/diffusion/wan_diffusion_backend.h"
#include "runtime/trt/diffusion/flux_diffusion_backend.h"
#include "runtime/trt/diffusion/z_image_diffusion_backend.h"
#include "trtf/tokenizer.h"

#include <stdexcept>
#include <string>

namespace trtf {

namespace {

struct RawEngine {
    TrtUniquePtr<nvinfer1::ICudaEngine> engine;
    TrtUniquePtr<nvinfer1::IExecutionContext> context;
};

RawEngine deser(const std::vector<char>* plan, const char* label)
{
    if (!plan || plan->empty())
        throw std::runtime_error(std::string("Bundle missing ") + label);
    auto rt = create_trt_runtime();
    if (!rt) throw std::runtime_error(std::string("TRT runtime failed: ") + label);
    RawEngine re;
    re.engine.reset(rt->deserializeCudaEngine(plan->data(), plan->size()));
    if (!re.engine) throw std::runtime_error(std::string("Deserialize failed: ") + label);
    re.context.reset(re.engine->createExecutionContext());
    if (!re.context) throw std::runtime_error(std::string("Context failed: ") + label);
    return re;
}

bool detect_add_special(const BundleSections& s)
{
    if (!s.config_json_data) return true;
    std::string t(s.config_json_data->begin(), s.config_json_data->end());
    auto p = t.find("\"tokenizer_add_special_tokens\"");
    if (p == std::string::npos) return true;
    auto v = t.find(':', p);
    if (v == std::string::npos) return true;
    return t.substr(v + 1, 20).find("false") == std::string::npos;
}

std::shared_ptr<ITokenizer> make_tok(const BundleSections& s, const std::string& hf)
{
    if (hf.empty()) return nullptr;
    try {
        auto r = extract_tokenizer_from_bundle(s, hf, detect_add_special(s));
        if (r.tokenizer) return std::move(r.tokenizer);
    } catch (...) {}
    return nullptr;
}

DiffusionConfig make_config(const FastPathModelConfig& cfg)
{
    DiffusionConfig dc;
    dc.scheduler = cfg.scheduler.empty() ? "flow_match_euler" : cfg.scheduler;
    dc.num_inference_steps = cfg.num_inference_steps;
    dc.guidance_scale = cfg.guidance_scale;
    dc.flow_shift = cfg.flow_shift;
    dc.use_dynamic_shifting = cfg.use_dynamic_shifting;
    dc.base_shift = cfg.base_shift;
    dc.max_shift = cfg.max_shift;
    dc.video_height = cfg.video_height;
    dc.video_width = cfg.video_width;
    dc.video_num_frames = cfg.video_num_frames;
    dc.z_dim = cfg.z_dim;
    dc.scale_factor_temporal = cfg.scale_factor_temporal;
    dc.scale_factor_spatial = cfg.scale_factor_spatial;
    dc.dit_dim = cfg.dit_dim;
    dc.dit_num_heads = cfg.dit_num_heads;
    dc.freq_dim = cfg.freq_dim;
    dc.text_seq_len = cfg.text_seq_len;
    dc.text_encoder_dim = cfg.text_encoder_dim;
    dc.num_vae_caches = cfg.num_vae_caches;
    dc.latents_mean = cfg.latents_mean;
    dc.latents_std = cfg.latents_std;
    dc.patch_size = cfg.patch_size;
    dc.axes_dims_rope = cfg.axes_dims_rope;
    dc.rope_theta = cfg.rope_theta;
    dc.vae_model_id = cfg.vae_model_id;
    dc.guidance_embeds = cfg.guidance_embeds;
    dc.use_rope = cfg.use_rope;
    dc.vae_scaling_factor = cfg.vae_scaling_factor;
    dc.diffusion_backend_type = cfg.diffusion_backend_type;
    return dc;
}

bool is_flux(const std::string& bt)
{ return bt == "flux_2d" || bt.find("flux") != std::string::npos; }

bool is_zimage(const std::string& bt)
{ return bt == "z_image_2d" || bt.find("z_image") != std::string::npos; }

DiffusionEngine load_denoiser(const BundleSections& sections)
{
    auto raw = deser(sections.denoiser_plan_data, "denoiser_plan");
    DiffusionEngine eng;
    eng.engine = std::move(raw.engine);
    eng.context = std::move(raw.context);
    eng.name = "denoiser";
    return eng;
}

DiffusionEngine load_vae(const BundleSections& sections)
{
    auto raw = deser(sections.vae_decoder_plan_data, "vae_decoder_plan");
    DiffusionEngine eng;
    eng.engine = std::move(raw.engine);
    eng.context = std::move(raw.context);
    eng.name = "vae_decoder";
    return eng;
}

std::vector<DiffusionEngine> load_text_encoders(const BundleSections& sections)
{
    std::vector<DiffusionEngine> te;
    for (std::size_t i = 0; i < sections.text_encoder_plans.size(); ++i) {
        std::string label = "text_encoder_" + std::to_string(i);
        auto raw = deser(sections.text_encoder_plans[i], label.c_str());
        DiffusionEngine de;
        de.engine = std::move(raw.engine);
        de.context = std::move(raw.context);
        de.name = label;
        te.push_back(std::move(de));
    }
    if (te.empty() && sections.plan_data && !sections.plan_data->empty()) {
        auto raw = deser(sections.plan_data, "text_encoder_0");
        DiffusionEngine de;
        de.engine = std::move(raw.engine);
        de.context = std::move(raw.context);
        de.name = "text_encoder_0";
        te.push_back(std::move(de));
    }
    return te;
}

} // namespace

std::unique_ptr<IPipeline> make_diffusion_pipeline_from_bundle(
    const BundleSections& sections,
    const FastPathModelConfig& cfg,
    const std::string& bundle_path,
    const std::string& hf_python,
    const std::string& model_id)
{
    auto den_eng = load_denoiser(sections);
    auto vae_eng = load_vae(sections);
    auto te = load_text_encoders(sections);
    auto dc = make_config(cfg);

    PreprocessorWeights pw;
    if (sections.preprocessor_weights_data && !sections.preprocessor_weights_data->empty())
        pw = parse_preprocessor_weights(*sections.preprocessor_weights_data);

    auto tok = make_tok(sections, hf_python);

    if (is_flux(cfg.diffusion_backend_type)) {
        auto fb = std::make_unique<FluxDiffusionBackend>(
            std::move(te), std::move(den_eng), std::move(vae_eng), dc);
        fb->set_preprocessor_weights(std::move(pw));
        fb->set_hf_python(hf_python);
        fb->set_bundle_path(bundle_path);
        try {
            auto ct = extract_clip_tokenizer_from_bundle(sections, hf_python);
            if (ct.tokenizer) fb->set_clip_tokenizer(std::move(ct.tokenizer));
        } catch (...) {}
        return std::make_unique<FluxPipeline>(
            std::move(fb), std::move(tok), cfg.video_height, cfg.video_width, model_id);
    }

    if (is_zimage(cfg.diffusion_backend_type)) {
        auto zb = std::make_unique<ZImageDiffusionBackend>(
            std::move(te), std::move(den_eng), std::move(vae_eng), dc);
        zb->set_preprocessor_weights(std::move(pw));
        if (sections.preprocessor_weights_data && !sections.preprocessor_weights_data->empty())
            zb->load_z_image_preprocessor_weights(*sections.preprocessor_weights_data);
        zb->set_hf_python(hf_python);
        zb->set_bundle_path(bundle_path);
        return std::make_unique<ZImagePipeline>(
            std::move(zb), std::move(tok), cfg.video_height, cfg.video_width, model_id);
    }

    auto wb = std::make_unique<WanDiffusionBackend>(
        std::move(te), std::move(den_eng), std::move(vae_eng), dc);
    wb->set_preprocessor_weights(std::move(pw));
    wb->set_hf_python(hf_python);
    wb->set_bundle_path(bundle_path);
    return std::make_unique<WanPipeline>(
        std::move(wb), std::move(tok), cfg.video_height, cfg.video_width, model_id);
}

} // namespace trtf

#endif // TRTF_HAS_TRT
