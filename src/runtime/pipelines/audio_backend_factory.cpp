// Creates audio IPipeline instances from bundle sections.
// Includes old backend headers (which define LegacyAudioResult) and
// pipeline headers (which define AudioResult) without conflict.

#include "runtime/pipelines/audio_backend_factory.h"
#include "runtime/pipelines/audio_pipeline.h"

#if TRTF_HAS_TRT

#include "runtime/trt/core/trt_common.h"
#include "runtime/trt/audio/whisper_backend.h"
#include "runtime/trt/audio/bark_backend.h"
#include "runtime/trt/audio/magpie_tts_backend.h"
#include "runtime/trt/audio/speech_backend.h"
#include "runtime/trt/audio/omni_backend.h"
#include "runtime/trt/audio/mel_spectrogram.h"
#include "trtf/tokenizer.h"

#include <algorithm>
#include <cstring>
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

void attach_depth_engines(
    SpeechToSpeechBackend& be, const BundleSections& sections,
    const FastPathModelConfig& depth_cfg)
{
    for (std::size_t i = 0; i < sections.depth_engine_plans.size(); ++i) {
        auto d = try_deser(sections.depth_engine_plans[i],
            ("depth_" + std::to_string(i)).c_str());
        if (d.engine) {
            auto de = make_decoder_engine(std::move(d.engine), std::move(d.context), depth_cfg);
            be.set_depth_engine(static_cast<int32_t>(i), std::move(de));
        }
    }
    if (sections.depth_engine_plans.empty()) {
        auto d = try_deser(sections.depth_engine_plan_data, "depth");
        if (d.engine) {
            auto de = make_decoder_engine(std::move(d.engine), std::move(d.context), depth_cfg);
            be.set_depth_engine(std::move(de));
        }
    }
}

void attach_mimi_engines(SpeechToSpeechBackend& be, const BundleSections& sections)
{
    auto me = try_deser(sections.mimi_encoder_plan_data, "mimi_enc");
    if (me.engine) be.set_mimi_encoder(std::move(me.engine), std::move(me.context));
    auto md = try_deser(sections.mimi_decoder_plan_data, "mimi_dec");
    if (md.engine) be.set_mimi_decoder(std::move(md.engine), std::move(md.context));
}

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

std::unique_ptr<IPipeline> make_magpie_pipeline_from_bundle(
    const BundleSections& sections, const FastPathModelConfig& cfg,
    const std::string& hf_python, const std::string& model_id)
{
    // MagpieTTS bundles store the text encoder as vision_engine_plan and the
    // autoregressive decoder as the main engine_plan.
    auto enc = deser(sections.vision_plan_data, "magpie encoder");
    auto dec = deser(sections.plan_data, "magpie decoder");
    auto de = make_decoder_engine(std::move(dec.engine), std::move(dec.context), cfg);
    set_decoder_vocab_from_logits(*de);
    auto be = CreateMagpieTTSBackend(std::move(de), std::move(enc.engine), std::move(enc.context),
        to_floats(sections.magpie_audio_embed_data), to_floats(sections.magpie_text_embed_data),
        to_floats(sections.magpie_context_embed_data), to_int32s(sections.magpie_context_lengths_data), cfg);
    auto codec = try_deser(sections.codec_engine_plan_data, "magpie codec");
    if (codec.engine) be->set_codec_engine(std::move(codec.engine), std::move(codec.context));
    auto tok = make_ipa_tok(sections);
    return std::make_unique<MagpiePipeline>(std::move(be), std::move(tok), model_id);
}

std::unique_ptr<IPipeline> make_speech_pipeline_from_bundle(
    const BundleSections& sections, const FastPathModelConfig& cfg,
    const std::string& hf_python, const std::string& model_id)
{
    auto te = deser(sections.plan_data, "temporal");
    auto teng = make_decoder_engine(std::move(te.engine), std::move(te.context), cfg);

    auto speech_cfg = build_speech_config_from_bundle(sections, cfg, hf_python);
    infer_speech_vocab_sizes(speech_cfg, cfg);

    auto be = std::make_unique<SpeechToSpeechBackend>(
        std::move(teng), std::move(speech_cfg));

    attach_depth_engines(*be, sections, make_depth_engine_config(cfg));
    attach_mimi_engines(*be, sections);
    return std::make_unique<SpeechPipeline>(std::move(be), model_id);
}

std::unique_ptr<IPipeline> make_omni_pipeline_from_bundle(
    const BundleSections& sections, const FastPathModelConfig& cfg,
    const std::string& hf_python, const std::string& model_id)
{
    auto th = deser(sections.plan_data, "omni thinker");
    auto teng = make_decoder_engine(std::move(th.engine), std::move(th.context), cfg);
    auto be = CreateOmniBackend(std::move(teng), cfg);
    auto ae = try_deser(sections.audio_encoder_plan_data, "audio_enc");
    if (ae.engine) be->set_audio_encoder(std::move(ae.engine), std::move(ae.context));
    auto tk = try_deser(sections.talker_engine_plan_data, "talker");
    if (tk.engine) {
        auto te = make_decoder_engine(std::move(tk.engine), std::move(tk.context), cfg);
        be->set_talker_engine(std::move(te));
    }
    auto cw = try_deser(sections.code2wav_engine_plan_data, "code2wav");
    if (cw.engine) be->set_code2wav_engine(std::move(cw.engine), std::move(cw.context));
    auto tok = make_tok(sections, hf_python);
    return std::make_unique<OmniPipeline>(std::move(be), std::move(tok), model_id);
}

} // namespace trtf

#endif // TRTF_HAS_TRT
