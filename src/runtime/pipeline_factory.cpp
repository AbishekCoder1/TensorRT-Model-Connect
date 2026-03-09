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

#include <iostream>
#include <stdexcept>
#include <unordered_map>

#if TRTF_HAS_TRT
#include "runtime/trt/core/trt_common.h"
#include "runtime/pipelines/audio_backend_factory.h"
#include "runtime/pipelines/diffusion_backend_factory.h"
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

// ─── Diffusion: delegate to old DiffusionBackendBase subclasses ───

std::unique_ptr<IPipeline> create_diffusion_pipeline(
    const BundleFile& bundle, const BundleSections& sections,
    const FastPathModelConfig& cfg, const std::string&,
    const std::string& bundle_path, const std::string& hf_python)
{
    return make_diffusion_pipeline_from_bundle(sections, cfg, bundle_path, hf_python, bundle.info.model_id);
}

// ─── Audio: delegate to backend factory ───

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
        return make_omni_pipeline_from_bundle(sections, cfg, hf_python, bundle.info.model_id);
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
