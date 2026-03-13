#include "trtf/runtime/pipeline_factory.h"
#include "trtf/tokenizer.h"
#include "trtf/runtime/trt_module.h"
#include "trtf/runtime/kv_cache.h"
#include "trtf/runtime/recurrent_state.h"
#include "trtf/runtime/scheduler.h"
#include "runtime/pipelines/text_generation_pipeline.h"
#include "runtime/pipelines/recurrent_pipeline.h"
#include "runtime/pipelines/vl_pipeline.h"
#include "runtime/pipelines/encoder_pipeline.h"
#include "runtime/pipelines/diffusion_pipeline.h"
#include "runtime/pipelines/audio_pipeline.h"

#include "bundle/bundle_format.h"
#include "cabi/config/fast_path_config.h"
#include "cabi/bundle/bundle_helpers.h"
#include "runtime/trt/multimodal/image_preprocessor.h"
#include "runtime/trt/diffusion/diffusion_types.h"
#include "runtime/trt/diffusion/diffusion_preprocessor_weights_helpers.h"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

#include <algorithm>
#include <cstring>

#if TRTF_HAS_TRT
#include "runtime/trt/core/trt_common.h"
#include "runtime/trt/audio/audio_configs.h"
#include "runtime/trt/audio/mel_spectrogram.h"
#include "trtf/runtime/trt/audio/subprocess_runner.h"
#endif

namespace trtf {

namespace {

enum class StrategyFamily { kText, kEncoder, kVision, kAudio, kDiffusion, kUnknown };

StrategyFamily resolve_family(const std::string& strategy)
{
    static const std::unordered_map<std::string, StrategyFamily> kMap = {
        {"decoder_kv_cache", StrategyFamily::kText},
        {"decoder_moe", StrategyFamily::kText},
        {"ssm_recurrent", StrategyFamily::kText},
        {"rwkv_recurrent", StrategyFamily::kText},
        {"hybrid_mamba_attention", StrategyFamily::kText},
        {"encoder_only", StrategyFamily::kEncoder},
        {"embedding", StrategyFamily::kEncoder},
        {"reranking", StrategyFamily::kEncoder},
        {"neural_operator", StrategyFamily::kEncoder},
        {"vision_language", StrategyFamily::kVision},
        {"segmentation", StrategyFamily::kVision},
        {"prompted_segmentation", StrategyFamily::kVision},
        {"object_detection", StrategyFamily::kVision},
        {"diffusion", StrategyFamily::kDiffusion},
        {"speech_to_text", StrategyFamily::kAudio},
        {"text_to_audio", StrategyFamily::kAudio},
        {"speech_to_speech", StrategyFamily::kAudio},
        {"omni_multimodal", StrategyFamily::kAudio},
    };
    auto it = kMap.find(strategy);
    return it != kMap.end() ? it->second : StrategyFamily::kUnknown;
}

#if TRTF_HAS_TRT

// ─── Shared helpers ───

bool detect_add_special_tokens(const BundleSections& sections)
{
    if (!sections.config_json_data) return true;
    std::string cfg_text(sections.config_json_data->begin(), sections.config_json_data->end());
    auto pos = cfg_text.find("\"tokenizer_add_special_tokens\"");
    if (pos == std::string::npos) return true;
    auto val_pos = cfg_text.find(':', pos);
    if (val_pos == std::string::npos) return true;
    auto rest = cfg_text.substr(val_pos + 1, 20);
    return rest.find("false") == std::string::npos;
}

std::shared_ptr<ITokenizer> create_tokenizer_from_bundle(
    const BundleSections& sections, const std::string& hf_python)
{
    if (hf_python.empty()) return nullptr;
    bool add_special = detect_add_special_tokens(sections);
    try {
        auto result = extract_tokenizer_from_bundle(sections, hf_python, add_special);
        if (result.tokenizer) return std::move(result.tokenizer);
    } catch (...) {}
    return nullptr;
}

struct LoadedModule {
    std::unique_ptr<TrtModule> module;
    std::shared_ptr<CudaStream> stream;
};

LoadedModule load_trt_module_from_plan(
    const std::vector<char>* plan, const char* label,
    std::shared_ptr<CudaStream> shared_stream = nullptr)
{
    if (!plan || plan->empty())
        throw std::runtime_error(std::string("Bundle missing ") + label);
    auto trt_runtime = create_trt_runtime();
    if (!trt_runtime)
        throw std::runtime_error(std::string("Failed to create TRT runtime for ") + label);
    auto engine = TrtUniquePtr<nvinfer1::ICudaEngine>(
        trt_runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine)
        throw std::runtime_error(std::string("Failed to deserialize ") + label);
    auto stream = shared_stream ? shared_stream : std::make_shared<CudaStream>();
    if (!stream->ok())
        throw std::runtime_error("Failed to create CUDA stream");
    LoadedModule result;
    result.stream = stream;
    result.module = std::make_unique<TrtModule>(engine.get(), stream->get());
    if (!result.module->ok())
        throw std::runtime_error(std::string("Failed to create TrtModule for ") + label);
    nvinfer1::ICudaEngine* raw_engine = engine.release();
    result.module->keep_alive(std::shared_ptr<nvinfer1::ICudaEngine>(
        raw_engine, [](nvinfer1::ICudaEngine* p) { delete p; }));
    result.module->keep_alive(stream);
    return result;
}

LoadedModule try_load_trt_module_from_plan(
    const std::vector<char>* plan, const char* label,
    std::shared_ptr<CudaStream> shared_stream)
{
    if (!plan || plan->empty()) return LoadedModule{};
    try { return load_trt_module_from_plan(plan, label, shared_stream); }
    catch (...) {
        std::cerr << "[trtf] WARNING: failed to load optional engine: " << label << std::endl;
        return LoadedModule{};
    }
}

LoadedModule load_trt_module(const BundleSections& sections)
{ return load_trt_module_from_plan(sections.plan_data, "engine_plan"); }

int32_t compute_kv_dim(const FastPathModelConfig& cfg)
{
    if (cfg.attention_size > 0) return cfg.attention_size;
    int32_t hd = (cfg.head_dim > 0) ? cfg.head_dim
        : ((cfg.num_heads > 0) ? cfg.hidden_size / cfg.num_heads : 128);
    return cfg.num_heads * hd;
}

std::unique_ptr<KvCache> make_coarse_kv_cache(
    const FastPathModelConfig& cfg, cudaStream_t stream)
{
    int32_t hidden = (cfg.coarse_hidden_size > 0) ? cfg.coarse_hidden_size : cfg.hidden_size;
    int32_t layers = (cfg.coarse_num_layers > 0) ? cfg.coarse_num_layers : cfg.num_layers;
    int32_t heads  = (cfg.coarse_num_heads > 0)  ? cfg.coarse_num_heads  : cfg.num_heads;
    int32_t hd     = (heads > 0) ? hidden / heads : 128;
    int32_t max_cache = (cfg.coarse_max_cache_length > 0)
        ? cfg.coarse_max_cache_length : cfg.max_cache_length;
    return std::make_unique<KvCache>(layers, max_cache, heads * hd, stream);
}

RecurrentGenConfig make_recurrent_gen_config(const FastPathModelConfig& cfg)
{
    RecurrentGenConfig rgc;
    rgc.vocab_size = cfg.vocab_size;
    rgc.id_bos = cfg.id_bos;
    rgc.id_eos = cfg.id_eos;
    return rgc;
}

// ─── Text strategy ───

std::unique_ptr<IPipeline> create_decoder_pipeline(
    LoadedModule& loaded, const FastPathModelConfig& cfg,
    std::shared_ptr<ITokenizer> tokenizer, const std::string& model_id)
{
    cudaStream_t stream = loaded.stream->get();
    int32_t kv_dim = compute_kv_dim(cfg);
    auto cache = std::make_unique<KvCache>(cfg.num_layers, cfg.max_cache_length, kv_dim, stream);
    if (!cache->ok()) throw std::runtime_error("Failed to create KvCache");
    TextGenConfig tgc;
    tgc.vocab_size = cfg.vocab_size;
    tgc.id_bos = cfg.id_bos;
    tgc.id_eos = cfg.id_eos;
    tgc.has_position_input = loaded.module->has_input("position_id");
    return std::make_unique<TextGenerationPipeline>(
        std::move(loaded.module), std::move(cache), tgc, stream,
        std::move(tokenizer), model_id);
}

std::unique_ptr<IPipeline> create_recurrent_pipeline(
    LoadedModule& loaded, const FastPathModelConfig& cfg,
    std::vector<RecurrentState::TensorSpec> specs, const char* name,
    std::shared_ptr<ITokenizer> tokenizer, const std::string& model_id)
{
    cudaStream_t stream = loaded.stream->get();
    auto state = std::make_unique<RecurrentState>(cfg.num_layers, specs, stream);
    auto mgr = std::make_unique<RecurrentStateManager>(std::move(state));
    auto rgc = make_recurrent_gen_config(cfg);
    return std::make_unique<RecurrentPipeline>(
        std::move(loaded.module), std::move(mgr), rgc, stream, name,
        std::move(tokenizer), model_id);
}

std::unique_ptr<IPipeline> create_hybrid_pipeline(
    LoadedModule& loaded, const FastPathModelConfig& cfg,
    std::shared_ptr<ITokenizer> tokenizer, const std::string& model_id)
{
    cudaStream_t stream = loaded.stream->get();
    int32_t kv_dim = compute_kv_dim(cfg);

    auto cache = std::make_unique<KvCache>(
        cfg.num_attention_layers, cfg.max_cache_length, kv_dim, stream);
    if (!cache->ok())
        throw std::runtime_error("Failed to create KvCache for hybrid model");

    int32_t effective_conv_dim = (cfg.conv_dim > 0) ? cfg.conv_dim : cfg.d_inner;
    int64_t conv_elems = static_cast<int64_t>(effective_conv_dim) * cfg.mamba_d_conv;
    int64_t ssm_elems = static_cast<int64_t>(cfg.mamba_nheads)
        * std::max(cfg.mamba_head_dim, 1) * cfg.mamba_d_state;

    auto ssm = std::make_unique<RecurrentState>(cfg.num_mamba_layers,
        std::vector<RecurrentState::TensorSpec>{
            {"conv_state", {conv_elems}, "present_conv"},
            {"ssm_state", {ssm_elems}, "present_ssm"}},
        stream);

    auto mgr = std::make_unique<HybridStateManager>(std::move(cache), std::move(ssm));
    auto rgc = make_recurrent_gen_config(cfg);
    rgc.has_position_input = loaded.module->has_input("position_id");

    return std::make_unique<RecurrentPipeline>(
        std::move(loaded.module), std::move(mgr), rgc, stream,
        "HybridPipeline", std::move(tokenizer), model_id);
}

std::unique_ptr<IPipeline> create_text_pipeline(
    const BundleFile& bundle, const BundleSections& sections,
    const FastPathModelConfig& cfg, const std::string& strategy,
    const std::string&, const std::string& hf_python)
{
    auto loaded = load_trt_module(sections);
    auto tokenizer = create_tokenizer_from_bundle(sections, hf_python);
    if (strategy == "decoder_kv_cache" || strategy == "decoder_moe")
        return create_decoder_pipeline(loaded, cfg, std::move(tokenizer), bundle.info.model_id);
    if (strategy == "ssm_recurrent")
        return create_recurrent_pipeline(loaded, cfg,
            {{"conv_state", {cfg.d_inner * cfg.conv_kernel}, "present_conv"},
             {"ssm_state", {cfg.state_size * cfg.d_inner}, "present_ssm"}},
            "MambaPipeline", std::move(tokenizer), bundle.info.model_id);
    if (strategy == "rwkv_recurrent")
        return create_recurrent_pipeline(loaded, cfg,
            {{"attn_state", {cfg.hidden_size}, "present_attn"},
             {"ff_state", {cfg.hidden_size}, "present_ff"},
             {"num_state", {cfg.hidden_size}, "present_num"},
             {"den_state", {cfg.hidden_size}, "present_den"},
             {"max_state", {cfg.hidden_size}, "present_max"}},
            "RwkvPipeline", std::move(tokenizer), bundle.info.model_id);
    if (strategy == "hybrid_mamba_attention")
        return create_hybrid_pipeline(loaded, cfg, std::move(tokenizer), bundle.info.model_id);
    throw std::runtime_error("Unsupported text strategy: " + strategy);
}

// ─── Encoder strategy ───

std::unique_ptr<IPipeline> create_encoder_pipeline(
    const BundleFile& bundle, const BundleSections& sections,
    const FastPathModelConfig&, const std::string& strategy,
    const std::string&, const std::string& hf_python)
{
    auto loaded = load_trt_module(sections);
    if (strategy == "segmentation")
        return std::make_unique<SegmentPipeline>(std::move(loaded.module), bundle.info.model_id);
    if (strategy == "prompted_segmentation") {
        auto decoder = try_load_trt_module_from_plan(
            sections.vision_plan_data, "vision_plan (SAM mask_decoder)", loaded.stream);
        if (decoder.module && decoder.module->ok())
            return std::make_unique<SamPipeline>(
                std::move(loaded.module), std::move(decoder.module), bundle.info.model_id);
        return std::make_unique<SegmentPipeline>(std::move(loaded.module), bundle.info.model_id);
    }
    auto tokenizer = create_tokenizer_from_bundle(sections, hf_python);
    return std::make_unique<EncoderPipeline>(
        std::move(loaded.module), strategy, std::move(tokenizer), bundle.info.model_id);
}

// ─── Vision strategy ───

VLPreprocessConfig build_vl_preprocess_config(const BundleSections& sections)
{
    std::string config_text, preproc_text;
    if (sections.config_json_data && !sections.config_json_data->empty())
        config_text.assign(sections.config_json_data->begin(), sections.config_json_data->end());
    if (sections.preprocessor_config_data && !sections.preprocessor_config_data->empty())
        preproc_text.assign(sections.preprocessor_config_data->begin(), sections.preprocessor_config_data->end());
    return parse_vl_preprocess_config(config_text, preproc_text);
}

std::unique_ptr<IPipeline> create_vision_pipeline(
    const BundleFile& bundle, const BundleSections& sections,
    const FastPathModelConfig& cfg, const std::string& strategy,
    const std::string& bundle_path, const std::string& hf_python)
{
    if (strategy == "segmentation" || strategy == "prompted_segmentation")
        return create_encoder_pipeline(bundle, sections, cfg, strategy, bundle_path, hf_python);
    auto loaded = load_trt_module(sections);
    cudaStream_t stream = loaded.stream->get();
    int32_t kv_dim = compute_kv_dim(cfg);
    auto cache = std::make_unique<KvCache>(cfg.num_layers, cfg.max_cache_length, kv_dim, stream);
    auto tokenizer = create_tokenizer_from_bundle(sections, hf_python);
    VLConfig vlc;
    vlc.vocab_size = cfg.vocab_size; vlc.id_bos = cfg.id_bos; vlc.id_eos = cfg.id_eos;
    vlc.image_token_id = cfg.image_token_id; vlc.vision_output_dim = cfg.vision_output_dim;
    vlc.has_position_input = loaded.module->has_input("position_id");
    std::unique_ptr<TrtModule> vision_module;
    auto vision_loaded = try_load_trt_module_from_plan(
        sections.vision_plan_data, "vision_engine_plan", loaded.stream);
    if (vision_loaded.module && vision_loaded.module->ok()) {
        vision_module = std::move(vision_loaded.module);
        std::cerr << "[trtf] Vision encoder loaded" << std::endl;
    } else if (cfg.has_vision_engine) {
        std::cerr << "[trtf] WARNING: Bundle declares vision engine but deserialization failed" << std::endl;
    }
    auto vl_preprocess = build_vl_preprocess_config(sections);
    return std::make_unique<VLPipeline>(
        std::move(loaded.module), std::move(vision_module), std::move(cache),
        vlc, vl_preprocess, stream, std::move(tokenizer), bundle.info.model_id);
}

// ─── Diffusion: TrtModule-based pipelines ───

DiffusionConfig make_diffusion_config(const FastPathModelConfig& cfg)
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

bool is_flux_type(const std::string& bt)
{ return bt == "flux_2d" || bt.find("flux") != std::string::npos; }

bool is_zimage_type(const std::string& bt)
{ return bt == "z_image_2d" || bt.find("z_image") != std::string::npos; }

// Shared diffusion resources loaded once, then dispatched to per-model factory.
struct DiffusionParts {
    LoadedModule denoiser;
    LoadedModule vae;
    std::vector<LoadedModule> text_encoders;
    DiffusionConfig config;
    PreprocessorWeights weights;
    std::shared_ptr<ITokenizer> tokenizer;
};

DiffusionParts load_diffusion_parts(
    const BundleSections& sections,
    const FastPathModelConfig& cfg,
    const std::string& hf_python)
{
    DiffusionParts parts;

    auto shared_stream = std::make_shared<CudaStream>();
    if (!shared_stream->ok())
        throw std::runtime_error("Failed to create CUDA stream for diffusion");

    parts.denoiser = load_trt_module_from_plan(
        sections.denoiser_plan_data, "denoiser_plan", shared_stream);
    parts.vae = load_trt_module_from_plan(
        sections.vae_decoder_plan_data, "vae_decoder_plan", shared_stream);

    for (std::size_t i = 0; i < sections.text_encoder_plans.size(); ++i) {
        std::string label = "text_encoder_" + std::to_string(i);
        parts.text_encoders.push_back(load_trt_module_from_plan(
            sections.text_encoder_plans[i], label.c_str(), shared_stream));
    }
    if (parts.text_encoders.empty() && sections.plan_data && !sections.plan_data->empty()) {
        parts.text_encoders.push_back(load_trt_module_from_plan(
            sections.plan_data, "text_encoder_0", shared_stream));
    }

    parts.config = make_diffusion_config(cfg);

    if (sections.preprocessor_weights_data && !sections.preprocessor_weights_data->empty())
        parts.weights = parse_preprocessor_weights(*sections.preprocessor_weights_data);

    parts.tokenizer = create_tokenizer_from_bundle(sections, hf_python);
    return parts;
}

std::unique_ptr<IPipeline> create_flux_pipeline(
    DiffusionParts& parts,
    const BundleSections& sections,
    const std::string& hf_python,
    const std::string& model_id)
{
    std::vector<std::unique_ptr<TrtModule>> te_modules;
    for (auto& te : parts.text_encoders)
        te_modules.push_back(std::move(te.module));

    // Try to load CLIP tokenizer
    std::unique_ptr<ITokenizer> clip_tok;
    try {
        auto ct = extract_clip_tokenizer_from_bundle(sections, hf_python);
        if (ct.tokenizer)
            clip_tok = std::move(ct.tokenizer);
    } catch (...) {}

    return std::make_unique<FluxPipeline>(
        std::move(te_modules),
        std::move(parts.denoiser.module),
        std::move(parts.vae.module),
        std::move(parts.config),
        std::move(parts.weights),
        std::move(parts.tokenizer),
        std::move(clip_tok),
        model_id);
}

ZImagePreprocessorWeights parse_zimage_preprocessor_weights(
    const std::vector<char>& data)
{
    ZImagePreprocessorWeights w;
    std::string index_json;
    const char* blob = nullptr;
    std::size_t blob_size = 0;
    if (!diffusion::extract_preprocessor_index(data, index_json, blob, blob_size))
        return w;

    diffusion::load_preprocessor_floats(index_json, blob, blob_size,
        "t_embedder.mlp.0.weight", w.t_embedder_mlp_0_weight);
    diffusion::load_preprocessor_floats(index_json, blob, blob_size,
        "t_embedder.mlp.0.bias", w.t_embedder_mlp_0_bias);
    diffusion::load_preprocessor_floats(index_json, blob, blob_size,
        "t_embedder.mlp.2.weight", w.t_embedder_mlp_2_weight);
    diffusion::load_preprocessor_floats(index_json, blob, blob_size,
        "t_embedder.mlp.2.bias", w.t_embedder_mlp_2_bias);
    diffusion::load_preprocessor_floats(index_json, blob, blob_size,
        "cap_embedder.proj.weight", w.cap_proj_weight);
    diffusion::load_preprocessor_floats(index_json, blob, blob_size,
        "cap_embedder.proj.bias", w.cap_proj_bias);
    diffusion::load_preprocessor_floats(index_json, blob, blob_size,
        "cap_embedder.norm.weight", w.cap_norm_weight);
    diffusion::load_preprocessor_floats(index_json, blob, blob_size,
        "cap_pad_token", w.cap_pad_token);
    diffusion::load_preprocessor_floats(index_json, blob, blob_size,
        "x_embedder.weight", w.x_embed_weight);
    diffusion::load_preprocessor_floats(index_json, blob, blob_size,
        "x_embedder.bias", w.x_embed_bias);

    // Derive dimensions
    if (!w.cap_proj_bias.empty())
        w.dit_dim = static_cast<int32_t>(w.cap_proj_bias.size());
    if (!w.cap_norm_weight.empty())
        w.cap_dim = static_cast<int32_t>(w.cap_norm_weight.size());
    if (!w.t_embedder_mlp_0_bias.empty())
        w.freq_dim = static_cast<int32_t>(w.t_embedder_mlp_0_bias.size());

    w.valid = !w.x_embed_weight.empty()
           && !w.t_embedder_mlp_0_weight.empty()
           && !w.cap_proj_weight.empty()
           && !w.cap_norm_weight.empty();

    std::cerr << "[z-image] Preprocessor weights: "
              << (w.valid ? "OK" : "INCOMPLETE")
              << " (dit_dim=" << w.dit_dim
              << ", cap_dim=" << w.cap_dim << ")\n";
    return w;
}

std::unique_ptr<IPipeline> create_zimage_pipeline(
    DiffusionParts& parts,
    const BundleSections& sections,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    std::unique_ptr<TrtModule> te_module;
    if (!parts.text_encoders.empty())
        te_module = std::move(parts.text_encoders[0].module);

    ZImagePreprocessorWeights z_pw;
    if (sections.preprocessor_weights_data && !sections.preprocessor_weights_data->empty())
        z_pw = parse_zimage_preprocessor_weights(*sections.preprocessor_weights_data);

    return std::make_unique<ZImagePipeline>(
        std::move(te_module),
        std::move(parts.denoiser.module),
        std::move(parts.vae.module),
        std::move(parts.config),
        std::move(parts.weights),
        std::move(z_pw),
        std::move(parts.tokenizer),
        model_id,
        hf_python,
        bundle_path);
}

std::unique_ptr<IPipeline> create_wan_t2v_pipeline(
    DiffusionParts& parts,
    const std::string& model_id)
{
    std::unique_ptr<TrtModule> te_module;
    if (!parts.text_encoders.empty())
        te_module = std::move(parts.text_encoders[0].module);

    return std::make_unique<WanPipeline>(
        std::move(te_module),
        std::move(parts.denoiser.module),
        std::move(parts.vae.module),
        std::move(parts.config),
        std::move(parts.weights),
        std::move(parts.tokenizer),
        model_id);
}

std::unique_ptr<IPipeline> create_diffusion_pipeline(
    const BundleFile& bundle, const BundleSections& sections,
    const FastPathModelConfig& cfg, const std::string&,
    const std::string& bundle_path, const std::string& hf_python)
{
    auto parts = load_diffusion_parts(sections, cfg, hf_python);

    if (is_flux_type(cfg.diffusion_backend_type))
        return create_flux_pipeline(parts, sections, hf_python, bundle.info.model_id);

    if (is_zimage_type(cfg.diffusion_backend_type))
        return create_zimage_pipeline(parts, sections, bundle.info.model_id, hf_python, bundle_path);

    return create_wan_t2v_pipeline(parts, bundle.info.model_id);
}

// ─── Audio: Omni ───

std::unique_ptr<IPipeline> create_omni_pipeline(
    const BundleSections& sections, const FastPathModelConfig& cfg,
    const std::string& hf_python, const std::string& model_id)
{
    // Thinker (MoE decoder) — main engine plan
    auto thinker_loaded = load_trt_module_from_plan(sections.plan_data, "omni thinker");
    cudaStream_t stream = thinker_loaded.stream->get();
    int32_t kv_dim = compute_kv_dim(cfg);
    auto thinker_cache = std::make_unique<KvCache>(
        cfg.num_layers, cfg.max_cache_length, kv_dim, stream);
    if (!thinker_cache->ok())
        throw std::runtime_error("OmniPipeline: failed to create thinker KvCache");

    // Talker (optional)
    std::unique_ptr<TrtModule> talker_module;
    std::unique_ptr<KvCache> talker_cache;
    auto talker_loaded = try_load_trt_module_from_plan(
        sections.talker_engine_plan_data, "talker", thinker_loaded.stream);
    if (talker_loaded.module && talker_loaded.module->ok())
    {
        talker_module = std::move(talker_loaded.module);
        int32_t talker_kv_dim = cfg.omni_talker_hidden_size;
        int32_t talker_cache_len = cfg.omni_talker_max_cache_length;
        int32_t talker_layers = cfg.omni_talker_num_layers > 0
            ? cfg.omni_talker_num_layers : cfg.num_layers;
        talker_cache = std::make_unique<KvCache>(
            talker_layers, talker_cache_len, talker_kv_dim, stream);
    }

    // Code2Wav (optional)
    std::unique_ptr<TrtModule> code2wav_module;
    auto code2wav_loaded = try_load_trt_module_from_plan(
        sections.code2wav_engine_plan_data, "code2wav", thinker_loaded.stream);
    if (code2wav_loaded.module && code2wav_loaded.module->ok())
        code2wav_module = std::move(code2wav_loaded.module);

    // Build OmniConfig
    OmniConfig omni_cfg;
    omni_cfg.sample_rate = cfg.omni_sample_rate;
    omni_cfg.thinker_hidden_size = cfg.hidden_size;
    omni_cfg.thinker_num_layers = cfg.num_layers;
    omni_cfg.thinker_num_heads = cfg.num_heads;
    omni_cfg.num_experts = cfg.omni_num_experts;
    omni_cfg.num_experts_per_tok = cfg.omni_num_experts_per_tok;
    omni_cfg.talker_hidden_size = cfg.omni_talker_hidden_size;
    omni_cfg.talker_num_layers = cfg.omni_talker_num_layers;
    omni_cfg.talker_n_codebooks = cfg.omni_n_codebooks;
    omni_cfg.talker_codebook_size = cfg.omni_codebook_size;

    auto tokenizer = create_tokenizer_from_bundle(sections, hf_python);

    return std::make_unique<OmniPipeline>(
        std::move(thinker_loaded.module),
        std::move(thinker_cache),
        std::move(talker_module),
        std::move(talker_cache),
        std::move(code2wav_module),
        std::move(omni_cfg),
        stream,
        std::move(tokenizer),
        model_id);
}

// ─── Audio helpers ───

std::vector<float> section_to_floats(const std::vector<char>* sec)
{
    if (!sec || sec->empty()) return {};
    auto n = sec->size() / sizeof(float);
    std::vector<float> out(n);
    std::memcpy(out.data(), sec->data(), n * sizeof(float));
    return out;
}

std::vector<int32_t> section_to_int32s(const std::vector<char>* sec)
{
    if (!sec || sec->empty()) return {};
    auto n = sec->size() / sizeof(int32_t);
    std::vector<int32_t> out(n);
    std::memcpy(out.data(), sec->data(), n * sizeof(int32_t));
    return out;
}

bool has_section_data(const std::vector<char>* d)
{
    return d && !d->empty();
}

std::shared_ptr<ITokenizer> make_ipa_tok(const BundleSections& s)
{
    if (!has_section_data(s.magpie_ipa_phoneme_dict_data) ||
        !has_section_data(s.magpie_ipa_vocab_data))
    {
        throw std::runtime_error(
            "Bundle missing IPA tokenizer sections (magpie_ipa_phoneme_dict, "
            "magpie_ipa_vocab). Rebuild the bundle with the latest trtf-build.");
    }
    return CreateIpaTokenizer(
        s.magpie_ipa_phoneme_dict_data->data(), s.magpie_ipa_phoneme_dict_data->size(),
        has_section_data(s.magpie_ipa_heteronyms_data) ? s.magpie_ipa_heteronyms_data->data() : nullptr,
        has_section_data(s.magpie_ipa_heteronyms_data) ? s.magpie_ipa_heteronyms_data->size() : 0,
        s.magpie_ipa_vocab_data->data(), s.magpie_ipa_vocab_data->size(),
        has_section_data(s.magpie_ipa_config_data) ? s.magpie_ipa_config_data->data() : nullptr,
        has_section_data(s.magpie_ipa_config_data) ? s.magpie_ipa_config_data->size() : 0);
}

MagpieTTSConfig build_magpie_config(const FastPathModelConfig& cfg)
{
    MagpieTTSConfig magpie_cfg;
    magpie_cfg.sample_rate = cfg.audio_sample_rate;
    magpie_cfg.hidden_size = cfg.magpie_hidden_size > 0
        ? cfg.magpie_hidden_size : cfg.hidden_size;
    magpie_cfg.num_codebooks = cfg.magpie_num_codebooks;
    magpie_cfg.codebook_size = cfg.magpie_codebook_size;
    magpie_cfg.frames_per_second = cfg.magpie_fps;
    magpie_cfg.num_speakers = cfg.magpie_num_speakers;
    magpie_cfg.encoder_layers = cfg.magpie_encoder_layers;
    magpie_cfg.decoder_layers = cfg.magpie_decoder_layers;
    magpie_cfg.text_vocab_size = cfg.magpie_text_vocab_size;
    magpie_cfg.max_source_positions = cfg.magpie_max_source_positions;
    magpie_cfg.xa_n_heads = cfg.magpie_xa_n_heads;
    magpie_cfg.xa_d_head = cfg.magpie_xa_d_head;
    magpie_cfg.temperature = cfg.magpie_temperature;
    magpie_cfg.cfg_scale = cfg.magpie_cfg_scale;
    magpie_cfg.finished_limit_with_eot = cfg.magpie_finished_limit_with_eot;
    return magpie_cfg;
}

int32_t compute_kv_dim_kv_heads(const FastPathModelConfig& cfg, int32_t default_dim)
{
    if (cfg.attention_size > 0) return cfg.attention_size;
    if (cfg.num_kv_heads > 0 && cfg.head_dim > 0) return cfg.num_kv_heads * cfg.head_dim;
    return default_dim;
}

void allocate_cross_kv_buffers(
    int32_t num_layers, std::size_t buf_size,
    std::vector<CudaBuffer>& cross_k, std::vector<CudaBuffer>& cross_v)
{
    cross_k.reserve(static_cast<std::size_t>(num_layers));
    cross_v.reserve(static_cast<std::size_t>(num_layers));
    for (int32_t i = 0; i < num_layers; ++i)
    {
        cross_k.emplace_back(buf_size);
        cross_v.emplace_back(buf_size);
    }
}

std::unique_ptr<TrtModule> extract_optional_module(
    const std::vector<char>* plan, const char* label,
    std::shared_ptr<CudaStream> shared_stream)
{
    auto loaded = try_load_trt_module_from_plan(plan, label, shared_stream);
    if (loaded.module && loaded.module->ok())
        return std::move(loaded.module);
    return nullptr;
}

std::vector<std::unique_ptr<TrtModule>> load_depth_engines(
    const BundleSections& sections,
    std::shared_ptr<CudaStream> shared_stream)
{
    std::vector<std::unique_ptr<TrtModule>> depth_engines;
    if (!sections.depth_engine_plans.empty())
    {
        for (std::size_t i = 0; i < sections.depth_engine_plans.size(); ++i)
        {
            auto m = extract_optional_module(
                sections.depth_engine_plans[i],
                ("speech depth_" + std::to_string(i)).c_str(),
                shared_stream);
            if (m) depth_engines.push_back(std::move(m));
        }
    }
    if (depth_engines.empty())
    {
        auto m = extract_optional_module(
            sections.depth_engine_plan_data, "speech depth", shared_stream);
        if (m) depth_engines.push_back(std::move(m));
    }
    return depth_engines;
}

SpeechConfig build_speech_config_from_bundle(
    const BundleSections& sections,
    const FastPathModelConfig& cfg,
    const std::string& hf_python)
{
    SpeechConfig sc;
    sc.sample_rate = cfg.audio_sample_rate;
    sc.temporal_hidden_size = cfg.hidden_size;
    sc.temporal_num_layers = cfg.num_layers;
    sc.num_codebooks = cfg.codec_n_codebooks;
    sc.codebook_size = cfg.codebook_size;
    sc.frame_rate = cfg.speech_frame_rate;
    sc.depth_num_layers = cfg.fine_num_layers;
    sc.depth_hidden_size = cfg.fine_hidden_size;
    sc.depth_num_heads = cfg.fine_num_heads;
    sc.depth_num_kv_heads = cfg.speech_depth_num_kv_heads;
    sc.delays = cfg.speech_delays;
    sc.text_initial_token_id = cfg.speech_text_initial_token_id;
    sc.audio_initial_token_id = cfg.speech_audio_initial_token_id;
    sc.text_padding_id = cfg.speech_text_padding_id;
    sc.depth_temperature = cfg.speech_depth_temperature;
    sc.depth_top_k = cfg.speech_depth_top_k;
    sc.text_eos_token_id = cfg.id_eos;
    sc.system_prompt = cfg.speech_system_prompt;
    sc.text_prompt_ids = cfg.speech_text_prompt_ids;
    sc.hf_python = hf_python;
    sc.audio_embeddings = section_to_floats(sections.audio_embeddings_data);
    sc.temporal_text_embedding = section_to_floats(sections.temporal_text_embedding_data);
    sc.depth_text_embedding = section_to_floats(sections.depth_text_embedding_data);
    sc.depth_audio_embeddings = section_to_floats(sections.depth_audio_embeddings_data);
    sc.depth_projection = section_to_floats(sections.depth_projection_data);
    return sc;
}

int32_t safe_embed_dim(const std::vector<float>& data, int32_t divisor)
{
    return (divisor > 0 && !data.empty())
        ? static_cast<int32_t>(data.size()) / divisor : 0;
}

void infer_speech_vocab_sizes(SpeechConfig& sc, const FastPathModelConfig& cfg)
{
    const int32_t h = cfg.hidden_size;
    const int32_t dh = cfg.fine_hidden_size;
    sc.audio_vocab_size = safe_embed_dim(sc.audio_embeddings, cfg.codec_n_codebooks * h);
    sc.temporal_text_vocab = safe_embed_dim(sc.temporal_text_embedding, h);
    sc.depth_text_vocab = safe_embed_dim(sc.depth_text_embedding, dh);
    sc.num_depformer_emb = safe_embed_dim(sc.depth_audio_embeddings,
        sc.audio_vocab_size * dh);
    sc.temporal_hidden_for_proj = (!sc.depth_projection.empty() && h > 0) ? h : 0;
}

FastPathModelConfig make_depth_engine_config(const FastPathModelConfig& cfg)
{
    FastPathModelConfig dc = cfg;
    dc.num_layers = cfg.speech_depth_num_layers > 0 ? cfg.speech_depth_num_layers : cfg.fine_num_layers;
    dc.hidden_size = cfg.speech_depth_hidden_size > 0 ? cfg.speech_depth_hidden_size : cfg.fine_hidden_size;
    dc.num_heads = cfg.speech_depth_num_heads > 0 ? cfg.speech_depth_num_heads : cfg.fine_num_heads;
    dc.num_kv_heads = cfg.speech_depth_num_kv_heads > 0
        ? cfg.speech_depth_num_kv_heads : dc.num_heads;
    dc.vocab_size = cfg.speech_codebook_size;
    dc.head_dim = dc.hidden_size / std::max(dc.num_heads, 1);
    dc.attention_size = dc.num_heads * dc.head_dim;
    dc.max_cache_length = cfg.speech_num_codebooks + 2;
    return dc;
}

// ─── Whisper: TrtModule(encoder) + TrtModule(decoder) + KvCache ───

std::unique_ptr<IPipeline> create_whisper_pipeline(
    const BundleSections& sections, const FastPathModelConfig& cfg,
    const std::string& hf_python, const std::string& model_id)
{
    // Load encoder (stored as vision_engine_plan in Whisper bundles)
    const auto* enc_plan = sections.vision_plan_data;
    if (!enc_plan || enc_plan->empty()) enc_plan = sections.coarse_engine_plan_data;
    auto enc_loaded = load_trt_module_from_plan(enc_plan, "whisper encoder");

    // Load decoder (main engine_plan)
    auto dec_loaded = load_trt_module_from_plan(
        sections.plan_data, "whisper decoder", enc_loaded.stream);

    // Build WhisperConfig
    int32_t dl = (cfg.decoder_layers > 0) ? cfg.decoder_layers : cfg.num_layers;
    WhisperConfig wc;
    wc.num_mel_bins = cfg.num_mel_bins;
    wc.max_source_positions = cfg.max_source_positions;
    wc.max_target_positions = cfg.max_target_positions;
    wc.encoder_layers = cfg.encoder_layers;
    wc.decoder_layers = dl;
    wc.eot_token_id = (cfg.eot_token_id >= 0) ? cfg.eot_token_id : cfg.id_eos;
    wc.mel_length = cfg.mel_length;
    wc.decoder_start_token_ids = cfg.decoder_start_token_ids;

    // Create KvCache for decoder self-attention
    cudaStream_t stream = dec_loaded.stream->get();
    int32_t kv_dim = compute_kv_dim(cfg);
    int32_t max_cache = cfg.max_cache_length;
    auto cache = std::make_unique<KvCache>(dl, max_cache, kv_dim, stream);
    if (!cache->ok()) throw std::runtime_error("Failed to create KvCache for Whisper decoder");

    // Load mel filterbank + tokenizer
    auto mel_fb = load_mel_filterbank(sections);
    auto tok = create_tokenizer_from_bundle(sections, hf_python);

    return std::make_unique<WhisperPipeline>(
        std::move(enc_loaded.module), std::move(dec_loaded.module),
        std::move(cache), std::move(wc), cfg.hidden_size, dl,
        std::move(mel_fb),
        cfg.mel_n_fft, cfg.mel_hop_length, cfg.mel_chunk_length, cfg.mel_sampling_rate,
        stream, std::move(tok), model_id);
}

// ─── Bark: TrtModule(semantic) + TrtModule(coarse) + optional TrtModule(codec/fine) ───

std::unique_ptr<IPipeline> create_bark_pipeline(
    const BundleSections& sections, const FastPathModelConfig& cfg,
    const std::string& hf_python, const std::string& model_id)
{
    // Load semantic engine (main plan)
    auto sem_loaded = load_trt_module_from_plan(sections.plan_data, "bark semantic");

    // Load coarse engine
    auto coarse_loaded = load_trt_module_from_plan(
        sections.coarse_engine_plan_data, "bark coarse", sem_loaded.stream);

    cudaStream_t stream = sem_loaded.stream->get();

    // Build BarkConfig
    BarkConfig bark_cfg;
    bark_cfg.sample_rate = cfg.audio_sample_rate;
    bark_cfg.hidden_size = cfg.hidden_size;
    bark_cfg.semantic_input_vocab = cfg.semantic_input_vocab;
    bark_cfg.semantic_output_vocab = cfg.vocab_size;
    bark_cfg.text_encoding_offset = cfg.text_encoding_offset;
    bark_cfg.text_pad_token = cfg.text_pad_token;
    bark_cfg.semantic_pad_token = cfg.semantic_pad_token;
    bark_cfg.semantic_infer_token = cfg.semantic_infer_token;
    bark_cfg.semantic_vocab_size = cfg.semantic_vocab_size;
    bark_cfg.coarse_input_vocab = cfg.coarse_input_vocab;
    bark_cfg.coarse_semantic_pad_token = cfg.coarse_semantic_pad_token;
    bark_cfg.coarse_infer_token = cfg.coarse_infer_token;
    bark_cfg.n_coarse_codebooks = cfg.n_coarse_codebooks;
    bark_cfg.codebook_size = cfg.codebook_size;
    bark_cfg.codec_seq_length = cfg.codec_seq_length;
    bark_cfg.codec_upsample_factor = cfg.codec_upsample_factor;
    bark_cfg.codec_n_codebooks = cfg.codec_n_codebooks;
    bark_cfg.fine_hidden_size = cfg.fine_hidden_size;
    bark_cfg.fine_n_lm_heads = cfg.fine_n_lm_heads;
    bark_cfg.fine_codebook_size = cfg.fine_codebook_size;
    bark_cfg.fine_seq_length = cfg.fine_seq_length;

    // Create KvCaches for semantic and coarse stages
    int32_t sem_kv_dim = compute_kv_dim(cfg);
    auto sem_cache = std::make_unique<KvCache>(
        cfg.num_layers, cfg.max_cache_length, sem_kv_dim, stream);

    // Coarse engine may have different dimensions — resolve with semantic fallbacks
    auto coarse_cache = make_coarse_kv_cache(cfg, stream);

    // Load embeddings
    auto sem_embed = section_to_floats(sections.semantic_embed_data);
    auto coarse_embed = section_to_floats(sections.coarse_embed_data);

    // Bark uses BertTokenizer WITHOUT special tokens ([CLS]/[SEP]).
    // HF's BarkProcessor calls encode(text, add_special_tokens=False).
    // The bundle's tokenizer_add_special_tokens field is unreliable here —
    // always use false to match the HF Bark pipeline.
    std::shared_ptr<ITokenizer> bark_tokenizer;
    if (!hf_python.empty()) {
        try {
            auto tok_result = extract_tokenizer_from_bundle(
                sections, hf_python, /*add_special_tokens=*/false);
            if (tok_result.tokenizer)
                bark_tokenizer = std::move(tok_result.tokenizer);
        } catch (...) {}
    }

    auto pipeline = std::make_unique<BarkPipeline>(
        std::move(sem_loaded.module), std::move(coarse_loaded.module),
        std::move(sem_cache), std::move(coarse_cache),
        std::move(sem_embed), std::move(coarse_embed),
        std::move(bark_cfg), stream,
        std::move(bark_tokenizer), model_id);

    // Optional codec engine
    auto codec_loaded = try_load_trt_module_from_plan(
        sections.codec_engine_plan_data, "bark codec", sem_loaded.stream);
    if (codec_loaded.module && codec_loaded.module->ok())
        pipeline->set_codec_module(std::move(codec_loaded.module));

    // Optional fine engine
    auto fine_loaded = try_load_trt_module_from_plan(
        sections.fine_engine_plan_data, "bark fine", sem_loaded.stream);
    if (fine_loaded.module && fine_loaded.module->ok())
    {
        pipeline->set_fine_module(std::move(fine_loaded.module));
        auto fe = section_to_floats(sections.fine_embed_data);
        auto fp = section_to_floats(sections.fine_position_embed_data);
        if (!fe.empty()) pipeline->set_fine_embeddings(std::move(fe), std::move(fp));
    }

    return pipeline;
}

// ─── Magpie: TrtModule(encoder) + TrtModule(decoder) + KvCache + IPA tokenizer ───

std::unique_ptr<IPipeline> create_magpie_pipeline(
    const BundleSections& sections, const FastPathModelConfig& cfg,
    const std::string& model_id)
{
    auto shared_stream = std::make_shared<CudaStream>();

    auto enc_loaded = load_trt_module_from_plan(
        sections.vision_plan_data, "magpie encoder", shared_stream);
    auto dec_loaded = load_trt_module_from_plan(
        sections.plan_data, "magpie decoder", shared_stream);

    cudaStream_t stream = shared_stream->get();

    auto magpie_cfg = build_magpie_config(cfg);
    int32_t kv_dim = (cfg.attention_size > 0) ? cfg.attention_size
        : ((cfg.num_heads > 0 && cfg.head_dim > 0) ? cfg.num_heads * cfg.head_dim
           : magpie_cfg.hidden_size);

    auto decoder_cache = std::make_unique<KvCache>(
        magpie_cfg.decoder_layers, cfg.max_cache_length, kv_dim, stream);
    if (!decoder_cache->ok())
        throw std::runtime_error("MagpiePipeline: failed to create decoder KvCache");

    std::unique_ptr<KvCache> decoder_cache_uncond;
    if (magpie_cfg.cfg_scale > 1.0F)
    {
        decoder_cache_uncond = std::make_unique<KvCache>(
            magpie_cfg.decoder_layers, cfg.max_cache_length, kv_dim, stream);
    }

    const std::size_t enc_buf_size = static_cast<std::size_t>(magpie_cfg.max_source_positions) *
        static_cast<std::size_t>(magpie_cfg.hidden_size) * sizeof(float);

    std::vector<CudaBuffer> cross_k, cross_v;
    allocate_cross_kv_buffers(magpie_cfg.decoder_layers, enc_buf_size, cross_k, cross_v);

    std::vector<CudaBuffer> cross_k_uncond, cross_v_uncond;
    if (magpie_cfg.cfg_scale > 1.0F)
        allocate_cross_kv_buffers(magpie_cfg.decoder_layers, enc_buf_size,
                                   cross_k_uncond, cross_v_uncond);

    CudaBuffer encoder_output(enc_buf_size);
    CudaBuffer encoder_output_uncond(magpie_cfg.cfg_scale > 1.0F ? enc_buf_size : 0);

    auto codec_module = extract_optional_module(
        sections.codec_engine_plan_data, "magpie codec", shared_stream);

    auto tok = make_ipa_tok(sections);

    return std::make_unique<MagpiePipeline>(
        std::move(enc_loaded.module),
        std::move(dec_loaded.module),
        std::move(decoder_cache),
        std::move(codec_module),
        std::move(decoder_cache_uncond),
        std::move(cross_k),
        std::move(cross_v),
        std::move(cross_k_uncond),
        std::move(cross_v_uncond),
        std::move(encoder_output),
        std::move(encoder_output_uncond),
        section_to_floats(sections.magpie_audio_embed_data),
        section_to_floats(sections.magpie_text_embed_data),
        section_to_floats(sections.magpie_context_embed_data),
        section_to_int32s(sections.magpie_context_lengths_data),
        std::move(magpie_cfg),
        stream,
        std::move(tok),
        model_id);
}

// ─── Speech: TrtModule(temporal) + TrtModule(depth[]) + KvCache + mimi enc/dec ───

std::unique_ptr<IPipeline> create_speech_pipeline(
    const BundleSections& sections, const FastPathModelConfig& cfg,
    const std::string& hf_python, const std::string& model_id)
{
    auto shared_stream = std::make_shared<CudaStream>();
    cudaStream_t stream = shared_stream->get();

    auto speech_cfg = build_speech_config_from_bundle(sections, cfg, hf_python);
    infer_speech_vocab_sizes(speech_cfg, cfg);

    auto temporal_loaded = load_trt_module_from_plan(
        sections.plan_data, "speech temporal", shared_stream);

    int32_t temporal_kv_dim = compute_kv_dim_kv_heads(cfg, cfg.hidden_size);

    auto temporal_cache = std::make_unique<KvCache>(
        cfg.num_layers, cfg.max_cache_length, temporal_kv_dim, stream);
    if (!temporal_cache->ok())
        throw std::runtime_error("SpeechPipeline: failed to create temporal KvCache");

    auto depth_engines = load_depth_engines(sections, shared_stream);

    const auto depth_cfg = make_depth_engine_config(cfg);
    int32_t depth_kv_dim = compute_kv_dim_kv_heads(depth_cfg, depth_cfg.hidden_size);

    auto depth_cache = std::make_unique<KvCache>(
        depth_cfg.num_layers, depth_cfg.max_cache_length, depth_kv_dim, stream);
    if (!depth_cache->ok())
        throw std::runtime_error("SpeechPipeline: failed to create depth KvCache");

    auto mimi_encoder = extract_optional_module(
        sections.mimi_encoder_plan_data, "speech mimi_encoder", shared_stream);
    auto mimi_decoder = extract_optional_module(
        sections.mimi_decoder_plan_data, "speech mimi_decoder", shared_stream);

    return std::make_unique<SpeechPipeline>(
        std::move(mimi_encoder),
        std::move(temporal_loaded.module),
        std::move(temporal_cache),
        std::move(depth_engines),
        std::move(depth_cache),
        std::move(mimi_decoder),
        std::move(speech_cfg),
        stream,
        nullptr,  // subprocess_runner: default
        model_id);
}

// ─── Audio strategy dispatch ───

std::unique_ptr<IPipeline> create_audio_pipeline(
    const BundleFile& bundle, const BundleSections& sections,
    const FastPathModelConfig& cfg, const std::string& strategy,
    const std::string&, const std::string& hf_python)
{
    if (strategy == "speech_to_text")
        return create_whisper_pipeline(sections, cfg, hf_python, bundle.info.model_id);
    if (strategy == "text_to_audio") {
        if (cfg.is_magpie_tts)
            return create_magpie_pipeline(sections, cfg, bundle.info.model_id);
        return create_bark_pipeline(sections, cfg, hf_python, bundle.info.model_id);
    }
    if (strategy == "speech_to_speech")
        return create_speech_pipeline(sections, cfg, hf_python, bundle.info.model_id);
    if (strategy == "omni_multimodal")
        return create_omni_pipeline(sections, cfg, hf_python, bundle.info.model_id);
    throw std::runtime_error("Unsupported audio strategy: " + strategy + " (bundle: " + bundle.info.model_id + ")");
}

// ─── Config parsing + dispatch ───

FastPathModelConfig parse_bundle_config(const BundleSections& sections, const BundleInfo& info)
{
    if (sections.config_json_data && !sections.config_json_data->empty()) {
        std::string config_text(sections.config_json_data->begin(), sections.config_json_data->end());
        return parse_fast_path_config(config_text, info.max_cache_length);
    }
    FastPathModelConfig cfg;
    cfg.vocab_size = info.vocab_size; cfg.hidden_size = info.hidden_size;
    cfg.num_layers = info.num_layers; cfg.num_heads = info.num_attention_heads;
    cfg.num_kv_heads = info.num_key_value_heads;
    cfg.head_dim = cfg.hidden_size / std::max(cfg.num_heads, 1);
    cfg.max_cache_length = info.max_cache_length;
    return cfg;
}

std::unique_ptr<IPipeline> dispatch_pipeline(
    StrategyFamily family, const BundleFile& bundle, const BundleSections& sections,
    const FastPathModelConfig& cfg, const std::string& strategy,
    const std::string& bundle_path, const std::string& hf_python)
{
    switch (family) {
    case StrategyFamily::kText:
        return create_text_pipeline(bundle, sections, cfg, strategy, bundle_path, hf_python);
    case StrategyFamily::kEncoder:
        return create_encoder_pipeline(bundle, sections, cfg, strategy, bundle_path, hf_python);
    case StrategyFamily::kVision:
        return create_vision_pipeline(bundle, sections, cfg, strategy, bundle_path, hf_python);
    case StrategyFamily::kDiffusion:
        return create_diffusion_pipeline(bundle, sections, cfg, strategy, bundle_path, hf_python);
    case StrategyFamily::kAudio:
        return create_audio_pipeline(bundle, sections, cfg, strategy, bundle_path, hf_python);
    default:
        throw std::runtime_error("Unknown runtime_strategy: " + strategy);
    }
}

#endif // TRTF_HAS_TRT

} // namespace

std::unique_ptr<IPipeline> PipelineFactory::from_bundle(
    const std::string& bundle_path, const std::string& hf_python)
{
#if TRTF_HAS_TRT
    BundleFile bundle = ReadBundleFile(bundle_path);
    if (bundle.sections.empty())
        throw std::runtime_error("Failed to read bundle: " + bundle_path);
    auto sections = find_bundle_sections(bundle);
    auto cfg = parse_bundle_config(sections, bundle.info);
    if (cfg.runtime_strategy.empty()) cfg.runtime_strategy = "decoder_kv_cache";
    auto pipeline = dispatch_pipeline(
        resolve_family(cfg.runtime_strategy),
        bundle, sections, cfg, cfg.runtime_strategy, bundle_path, hf_python);
    std::cerr << "[trtf] Pipeline loaded (strategy=" << cfg.runtime_strategy
              << ", backend=trt_new_runtime)" << std::endl;
    return pipeline;
#else
    (void) bundle_path; (void) hf_python;
    throw std::runtime_error("Bundle loading requires TRT support (compile with TRT)");
#endif
}

std::unique_ptr<IPipeline> load(const std::string& bundle_path, const std::string& hf_python)
{
    return PipelineFactory::from_bundle(bundle_path, hf_python);
}

} // namespace trtf
