// Creates audio IPipeline instances from bundle sections.
// Whisper and Bark use old-style backends; Magpie and Speech use TrtModule + KvCache.

#include "runtime/pipelines/audio_backend_factory.h"
#include "runtime/pipelines/audio_pipeline.h"

#if TRTF_HAS_TRT

#include "runtime/trt/core/trt_common.h"
#include "runtime/trt/audio/whisper_backend.h"
#include "runtime/trt/audio/bark_backend.h"
#include "runtime/trt/audio/audio_configs.h"
#include "runtime/trt/audio/mel_spectrogram.h"
#include "trtf/runtime/trt/audio/subprocess_runner.h"
#include "trtf/runtime/trt_module.h"
#include "trtf/runtime/kv_cache.h"
#include "trtf/tokenizer.h"

#include <algorithm>
#include <cstring>
#include <iostream>
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

RawEngine try_deser(const std::vector<char>* plan, const char* label)
{
    if (!plan || plan->empty()) return {};
    try { return deser(plan, label); }
    catch (...) { return {}; }
}

std::vector<float> to_floats(const std::vector<char>* sec)
{
    if (!sec || sec->empty()) return {};
    auto n = sec->size() / sizeof(float);
    std::vector<float> out(n);
    std::memcpy(out.data(), sec->data(), n * sizeof(float));
    return out;
}

std::vector<int32_t> to_int32s(const std::vector<char>* sec)
{
    if (!sec || sec->empty()) return {};
    auto n = sec->size() / sizeof(int32_t);
    std::vector<int32_t> out(n);
    std::memcpy(out.data(), sec->data(), n * sizeof(int32_t));
    return out;
}

bool has_data(const std::vector<char>* d)
{
    return d && !d->empty();
}

void set_decoder_vocab_from_logits(DecoderStepEngine& engine)
{
    auto logits_shape = engine.engine->getTensorShape("logits");
    int32_t actual_vocab = -1;
    if (logits_shape.nbDims >= 2)
        actual_vocab = logits_shape.d[logits_shape.nbDims - 1];
    else if (logits_shape.nbDims == 1)
        actual_vocab = logits_shape.d[0];
    if (actual_vocab > 0)
        engine.vocab_size = actual_vocab;
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
    auto r = extract_tokenizer_from_bundle(s, hf, detect_add_special(s));
    if (r.tokenizer) return std::move(r.tokenizer);
    return nullptr;
}

std::shared_ptr<ITokenizer> make_ipa_tok(const BundleSections& s)
{
    if (!has_data(s.magpie_ipa_phoneme_dict_data) || !has_data(s.magpie_ipa_vocab_data))
    {
        throw std::runtime_error(
            "Bundle missing IPA tokenizer sections (magpie_ipa_phoneme_dict, "
            "magpie_ipa_vocab). Rebuild the bundle with the latest trtf-build.");
    }
    return CreateIpaTokenizer(
        s.magpie_ipa_phoneme_dict_data->data(), s.magpie_ipa_phoneme_dict_data->size(),
        has_data(s.magpie_ipa_heteronyms_data) ? s.magpie_ipa_heteronyms_data->data() : nullptr,
        has_data(s.magpie_ipa_heteronyms_data) ? s.magpie_ipa_heteronyms_data->size() : 0,
        s.magpie_ipa_vocab_data->data(), s.magpie_ipa_vocab_data->size(),
        has_data(s.magpie_ipa_config_data) ? s.magpie_ipa_config_data->data() : nullptr,
        has_data(s.magpie_ipa_config_data) ? s.magpie_ipa_config_data->size() : 0);
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

    sc.audio_embeddings = to_floats(sections.audio_embeddings_data);
    sc.temporal_text_embedding = to_floats(sections.temporal_text_embedding_data);
    sc.depth_text_embedding = to_floats(sections.depth_text_embedding_data);
    sc.depth_audio_embeddings = to_floats(sections.depth_audio_embeddings_data);
    sc.depth_projection = to_floats(sections.depth_projection_data);
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

// NOTE: attach_depth_engines() and attach_mimi_engines() were removed during
// the TrtModule migration (Phase 2). The old SpeechToSpeechBackend is still
// compiled for backward compatibility but the pipeline no longer uses it.
// These helpers will be fully removed in Phase 3 cleanup.

} // namespace

std::unique_ptr<IPipeline> make_whisper_pipeline_from_bundle(
    const BundleSections& sections, const FastPathModelConfig& cfg,
    const std::string& hf_python, const std::string& model_id)
{
    // Whisper bundles: engine_plan = decoder (token_id, position_id, cache_k/v,
    // cross_k/v), vision_engine_plan = encoder (mel_features -> encoder_output).
    // The decoder is the "main" engine built by build_engine(), the encoder is
    // built by build_vision_engine() and stored as vision_engine_plan.
    const auto* enc_plan = sections.vision_plan_data;
    if (!enc_plan || enc_plan->empty()) enc_plan = sections.coarse_engine_plan_data;
    auto enc = deser(enc_plan, "whisper encoder");
    auto dec = deser(sections.plan_data, "whisper decoder");
    auto de = make_decoder_engine(std::move(dec.engine), std::move(dec.context), cfg);
    int32_t dl = (cfg.decoder_layers > 0) ? cfg.decoder_layers : cfg.num_layers;
    de->num_layers = dl;
    WhisperConfig wc;
    wc.num_mel_bins = cfg.num_mel_bins;
    wc.max_source_positions = cfg.max_source_positions;
    wc.max_target_positions = cfg.max_target_positions;
    wc.encoder_layers = cfg.encoder_layers;
    wc.decoder_layers = dl;
    wc.eot_token_id = (cfg.eot_token_id >= 0) ? cfg.eot_token_id : cfg.id_eos;
    wc.mel_length = cfg.mel_length;
    wc.decoder_start_token_ids = cfg.decoder_start_token_ids;
    auto be = CreateWhisperBackend(
        std::move(de), std::move(enc.engine), std::move(enc.context), wc, cfg);
    if (!be) throw std::runtime_error("WhisperBackend creation failed");
    auto mel_fb = load_mel_filterbank(sections);
    auto tok = make_tok(sections, hf_python);
    return std::make_unique<WhisperPipeline>(
        std::move(be), std::move(mel_fb),
        cfg.mel_n_fft, cfg.mel_hop_length, cfg.mel_chunk_length, cfg.mel_sampling_rate,
        std::move(tok), model_id);
}

std::unique_ptr<IPipeline> make_bark_pipeline_from_bundle(
    const BundleSections& sections, const FastPathModelConfig& cfg,
    const std::string& hf_python, const std::string& model_id)
{
    auto sem = deser(sections.plan_data, "bark semantic");
    auto se = make_decoder_engine(std::move(sem.engine), std::move(sem.context), cfg);
    set_decoder_vocab_from_logits(*se);
    auto coarse = deser(sections.coarse_engine_plan_data, "bark coarse");
    FastPathModelConfig cc = cfg;
    if (cfg.coarse_hidden_size > 0) cc.hidden_size = cfg.coarse_hidden_size;
    if (cfg.coarse_num_layers > 0) cc.num_layers = cfg.coarse_num_layers;
    if (cfg.coarse_num_heads > 0) {
        cc.num_heads = cfg.coarse_num_heads;
        cc.num_kv_heads = cfg.coarse_num_heads;
    }
    cc.vocab_size = cfg.coarse_input_vocab;
    cc.head_dim = cc.hidden_size / std::max(cc.num_heads, 1);
    cc.attention_size = cc.num_heads * cc.head_dim;
    cc.max_cache_length = cfg.coarse_max_cache_length;
    auto ce = make_decoder_engine(std::move(coarse.engine), std::move(coarse.context), cc);
    auto be = CreateBarkBackend(std::move(se), std::move(ce),
        to_floats(sections.semantic_embed_data), to_floats(sections.coarse_embed_data), cfg);
    auto codec = try_deser(sections.codec_engine_plan_data, "bark codec");
    if (codec.engine) be->set_codec_engine(std::move(codec.engine), std::move(codec.context));
    auto fine = try_deser(sections.fine_engine_plan_data, "bark fine");
    if (fine.engine) {
        be->set_fine_engine(std::move(fine.engine), std::move(fine.context));
        auto fe = to_floats(sections.fine_embed_data);
        auto fp = to_floats(sections.fine_position_embed_data);
        if (!fe.empty()) be->set_fine_embeddings(std::move(fe), std::move(fp));
    }
    // Bark tokenizer must use add_special_tokens=false — bark prepends its
    // own special tokens (text_encoding_offset, semantic_infer_token) in the
    // semantic stage; HF BOS/EOS tokens would corrupt the input sequence.
    std::shared_ptr<ITokenizer> tok;
    if (!hf_python.empty()) {
        auto r = extract_tokenizer_from_bundle(sections, hf_python, /*add_special_tokens=*/false);
        if (r.tokenizer) tok = std::move(r.tokenizer);
    }
    return std::make_unique<BarkPipeline>(std::move(be), std::move(tok), model_id);
}

namespace {

struct MagpieLoadedModule {
    std::unique_ptr<TrtModule> module;
    std::shared_ptr<CudaStream> stream;
};

MagpieLoadedModule load_magpie_trt_module(
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
    MagpieLoadedModule result;
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

MagpieLoadedModule try_load_magpie_trt_module(
    const std::vector<char>* plan, const char* label,
    std::shared_ptr<CudaStream> shared_stream)
{
    if (!plan || plan->empty()) return MagpieLoadedModule{};
    try { return load_magpie_trt_module(plan, label, shared_stream); }
    catch (...) {
        std::cerr << "[trtf] WARNING: failed to load optional engine: " << label << std::endl;
        return MagpieLoadedModule{};
    }
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

int32_t compute_kv_dim(const FastPathModelConfig& cfg, int32_t default_dim)
{
    if (cfg.attention_size > 0) return cfg.attention_size;
    if (cfg.num_heads > 0 && cfg.head_dim > 0) return cfg.num_heads * cfg.head_dim;
    return default_dim;
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
    auto loaded = try_load_magpie_trt_module(plan, label, shared_stream);
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

} // anonymous namespace

std::unique_ptr<IPipeline> make_magpie_pipeline_from_bundle(
    const BundleSections& sections, const FastPathModelConfig& cfg,
    const std::string& /*hf_python*/, const std::string& model_id)
{
    auto shared_stream = std::make_shared<CudaStream>();

    auto enc_loaded = load_magpie_trt_module(
        sections.vision_plan_data, "magpie encoder", shared_stream);
    auto dec_loaded = load_magpie_trt_module(
        sections.plan_data, "magpie decoder", shared_stream);

    cudaStream_t stream = shared_stream->get();

    auto magpie_cfg = build_magpie_config(cfg);
    int32_t kv_dim = compute_kv_dim(cfg, magpie_cfg.hidden_size);

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
        to_floats(sections.magpie_audio_embed_data),
        to_floats(sections.magpie_text_embed_data),
        to_floats(sections.magpie_context_embed_data),
        to_int32s(sections.magpie_context_lengths_data),
        std::move(magpie_cfg),
        stream,
        std::move(tok),
        model_id);
}

std::unique_ptr<IPipeline> make_speech_pipeline_from_bundle(
    const BundleSections& sections, const FastPathModelConfig& cfg,
    const std::string& hf_python, const std::string& model_id)
{
    auto shared_stream = std::make_shared<CudaStream>();
    cudaStream_t stream = shared_stream->get();

    auto speech_cfg = build_speech_config_from_bundle(sections, cfg, hf_python);
    infer_speech_vocab_sizes(speech_cfg, cfg);

    auto temporal_loaded = load_magpie_trt_module(
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

} // namespace trtf

#endif // TRTF_HAS_TRT
