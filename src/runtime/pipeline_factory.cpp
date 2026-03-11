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

#if TRTF_HAS_TRT
#include "runtime/trt/core/trt_common.h"
#include "runtime/trt/audio/audio_configs.h"
#include "runtime/pipelines/audio_backend_factory.h"
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

// ─── Audio: Omni uses TrtModule; Magpie/Speech delegate to old backends ───

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

std::unique_ptr<IPipeline> create_audio_pipeline(
    const BundleFile& bundle, const BundleSections& sections,
    const FastPathModelConfig& cfg, const std::string& strategy,
    const std::string&, const std::string& hf_python)
{
    if (strategy == "speech_to_text")
        return make_whisper_pipeline_from_bundle(sections, cfg, hf_python, bundle.info.model_id);
    if (strategy == "text_to_audio") {
        if (cfg.is_magpie_tts)
            return make_magpie_pipeline_from_bundle(sections, cfg, hf_python, bundle.info.model_id);
        return make_bark_pipeline_from_bundle(sections, cfg, hf_python, bundle.info.model_id);
    }
    if (strategy == "speech_to_speech")
        return make_speech_pipeline_from_bundle(sections, cfg, hf_python, bundle.info.model_id);
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
