#include "runtime/pipelines/audio_pipeline.h"

#if TRTF_HAS_TRT

#include "runtime/trt/audio/whisper_backend.h"
#include "runtime/trt/audio/bark_backend.h"
#include "runtime/trt/audio/magpie_tts_backend.h"
#include "runtime/trt/audio/speech_backend.h"
#include "runtime/trt/audio/audio_configs.h"
#include "runtime/trt/audio/omni_audio_plan.h"
#include "runtime/trt/audio/mel_spectrogram.h"
#include "cabi/bundle/bundle_helpers.h"
#include "utils/wav_reader.h"
#include "trtf/tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace trtf {

// ─── WhisperPipeline ───

WhisperPipeline::WhisperPipeline(
    std::unique_ptr<WhisperBackend> backend,
    MelFilterbank mel_fb,
    int32_t mel_n_fft,
    int32_t mel_hop_length,
    int32_t mel_chunk_length,
    int32_t mel_sampling_rate,
    std::shared_ptr<ITokenizer> tokenizer,
    std::string model_id_str)
    : backend_(std::move(backend))
    , mel_fb_(std::make_unique<MelFilterbank>(std::move(mel_fb)))
    , tokenizer_(std::move(tokenizer))
    , mel_n_fft_(mel_n_fft)
    , mel_hop_length_(mel_hop_length)
    , mel_chunk_length_(mel_chunk_length)
    , mel_sampling_rate_(mel_sampling_rate)
    , model_id_(std::move(model_id_str))
{
    if (!backend_ || !backend_->is_available())
        throw std::runtime_error("WhisperPipeline: invalid WhisperBackend");
}

WhisperPipeline::~WhisperPipeline() = default;

TextResult WhisperPipeline::transcribe(
    const float* audio_data, int32_t num_samples, int32_t max_new_tokens,
    int32_t input_sample_rate)
{
    // Step 0: Resample if the input sample rate differs from the model's rate.
    // This handles e.g. 48 kHz WAV files when the model expects 16 kHz.
    const float* samples_ptr = audio_data;
    int32_t samples_count = num_samples;
    std::vector<float> resampled_buf;

    if (input_sample_rate > 0 && input_sample_rate != mel_sampling_rate_)
    {
        std::cerr << "[whisper] Resampling audio from " << input_sample_rate
                  << " Hz to " << mel_sampling_rate_ << " Hz" << std::endl;
        resampled_buf = resample_linear(
            audio_data, num_samples, input_sample_rate, mel_sampling_rate_);
        samples_ptr = resampled_buf.data();
        samples_count = static_cast<int32_t>(resampled_buf.size());
    }

    // Step 1: Extract mel spectrogram from raw audio
    MelResult mel;
    if (mel_fb_ && !mel_fb_->data.empty())
    {
        mel = extract_mel_spectrogram(
            samples_ptr, samples_count,
            mel_fb_->data.data(), mel_fb_->n_freq_bins, mel_fb_->n_mel_bins,
            mel_n_fft_, mel_hop_length_,
            mel_chunk_length_, mel_sampling_rate_);
    }

    if (mel.data.empty())
    {
        return TextResult{"[mel extraction failed]", {}};
    }

    // Step 2: Run WhisperBackend transcription
    auto result = backend_->transcribe(
        mel.data.data(), mel.n_mels, mel.n_frames, max_new_tokens);

    TextResult out;
    out.token_ids = std::move(result.output_ids);
    if (tokenizer_ && !out.token_ids.empty())
    {
        out.text = tokenizer_->decode(out.token_ids);
    }
    else
    {
        out.text = std::move(result.text);
    }
    return out;
}

// ─── BarkPipeline ───

BarkPipeline::BarkPipeline(
    std::unique_ptr<BarkBackend> backend,
    std::shared_ptr<ITokenizer> tokenizer,
    std::string model_id_str)
    : backend_(std::move(backend))
    , tokenizer_(std::move(tokenizer))
    , model_id_(std::move(model_id_str))
{
    if (!backend_ || !backend_->is_available())
        throw std::runtime_error("BarkPipeline: invalid BarkBackend");
}

BarkPipeline::~BarkPipeline() = default;

AudioResult BarkPipeline::generate_audio(
    const std::string& prompt, const GenerateConfig& cfg)
{
    // Tokenize the prompt
    std::vector<int32_t> input_ids;
    if (tokenizer_)
        input_ids = tokenizer_->encode(prompt);

    int32_t max_tokens = cfg.max_new_tokens > 0 ? cfg.max_new_tokens : 768;

    // Delegate to BarkBackend
    auto bark_result = backend_->generate_audio(input_ids, max_tokens);

    // Convert BarkBackend AudioResult -> pipeline AudioResult
    AudioResult out;
    out.samples = std::move(bark_result.waveform);
    out.num_samples = bark_result.num_samples;
    out.sample_rate = bark_result.sample_rate;
    return out;
}

// ─── MagpiePipeline ───

MagpiePipeline::MagpiePipeline(
    std::unique_ptr<MagpieTTSBackend> backend,
    std::shared_ptr<ITokenizer> tokenizer,
    std::string model_id_str)
    : backend_(std::move(backend))
    , tokenizer_(std::move(tokenizer))
    , model_id_(std::move(model_id_str))
{
    if (!backend_ || !backend_->is_available())
        throw std::runtime_error("MagpiePipeline: invalid MagpieTTSBackend");
}

MagpiePipeline::~MagpiePipeline() = default;

AudioResult MagpiePipeline::generate_audio(
    const std::string& prompt, const GenerateConfig& cfg)
{
    std::vector<int32_t> input_ids;
    if (tokenizer_)
        input_ids = tokenizer_->encode(prompt);

    int32_t max_frames = cfg.max_new_tokens > 0 ? cfg.max_new_tokens : 512;

    auto magpie_result = backend_->generate_audio(input_ids, max_frames);

    AudioResult out;
    out.samples = std::move(magpie_result.waveform);
    out.num_samples = magpie_result.num_samples;
    out.sample_rate = magpie_result.sample_rate;
    return out;
}

// ─── SpeechPipeline ───

SpeechPipeline::SpeechPipeline(
    std::unique_ptr<SpeechToSpeechBackend> backend,
    std::string model_id_str)
    : backend_(std::move(backend))
    , model_id_(std::move(model_id_str))
{
    if (!backend_ || !backend_->is_available())
        throw std::runtime_error("SpeechPipeline: invalid SpeechToSpeechBackend");
}

SpeechPipeline::~SpeechPipeline() = default;

AudioResult SpeechPipeline::speak(
    const float* audio_in, int32_t num_samples, const GenerateConfig& cfg,
    int32_t input_sample_rate)
{
    int32_t max_frames = cfg.max_new_tokens > 0 ? cfg.max_new_tokens : 375;

    // Resample if the input sample rate differs from the model's expected rate.
    // The backend's run_mimi_encode does not resample internally, so we must
    // do it here (matching master's SpeechToSpeechAudioService::prepare_input).
    const float* samples_ptr = audio_in;
    int32_t samples_count = num_samples;
    std::vector<float> resampled_buf;
    const int32_t target_rate = backend_->config().sample_rate;

    if (input_sample_rate > 0 && target_rate > 0
        && input_sample_rate != target_rate)
    {
        std::cerr << "[speech] Resampling audio from " << input_sample_rate
                  << " Hz to " << target_rate << " Hz" << std::endl;
        resampled_buf = resample_linear(
            audio_in, num_samples, input_sample_rate, target_rate);
        samples_ptr = resampled_buf.data();
        samples_count = static_cast<int32_t>(resampled_buf.size());
        // After resampling, the data is at target_rate — tell the backend
        // so the output-plan does not double-count the rate conversion.
        input_sample_rate = target_rate;
    }

    auto speech_result = backend_->process_audio(
        samples_ptr, samples_count, max_frames, input_sample_rate,
        cfg.tail_frames);

    AudioResult out;
    out.samples = std::move(speech_result.waveform);
    out.num_samples = speech_result.num_samples;
    out.sample_rate = speech_result.sample_rate;
    return out;
}

// ─── OmniPipeline (TrtModule-based) ───

OmniPipeline::OmniPipeline(
    std::unique_ptr<TrtModule> thinker,
    std::unique_ptr<KvCache> thinker_cache,
    std::unique_ptr<TrtModule> talker,
    std::unique_ptr<KvCache> talker_cache,
    std::unique_ptr<TrtModule> code2wav,
    OmniConfig config,
    cudaStream_t stream,
    std::shared_ptr<ITokenizer> tokenizer,
    std::string model_id_str)
    : thinker_(std::move(thinker))
    , thinker_cache_(std::move(thinker_cache))
    , talker_(std::move(talker))
    , talker_cache_(std::move(talker_cache))
    , code2wav_(std::move(code2wav))
    , config_(std::make_unique<OmniConfig>(std::move(config)))
    , stream_(stream)
    , tokenizer_(std::move(tokenizer))
    , model_id_(std::move(model_id_str))
{
    if (!thinker_ || !thinker_->ok())
        throw std::runtime_error("OmniPipeline: invalid thinker module");
    if (!thinker_cache_ || !thinker_cache_->ok())
        throw std::runtime_error("OmniPipeline: invalid thinker cache");
}

OmniPipeline::~OmniPipeline() = default;

// ─── Thinker step ───

void OmniPipeline::run_thinker_step(int32_t token_id, std::vector<float>& logits)
{
    std::vector<float> mask;
    thinker_cache_->build_attention_mask(mask);

    int32_t position = thinker_cache_->position();

    Tensor token_tensor;
    token_tensor.data = &token_id;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;

    Tensor position_tensor;
    position_tensor.data = &position;
    position_tensor.shape = {1};
    position_tensor.dtype = DType::kInt32;

    Tensor mask_tensor;
    mask_tensor.data = mask.data();
    mask_tensor.shape = {static_cast<int64_t>(mask.size())};
    mask_tensor.dtype = DType::kFloat32;

    TensorMap inputs;
    inputs["token_id"] = token_tensor;
    if (thinker_->has_input("position_id"))
        inputs["position_id"] = position_tensor;
    inputs["attention_mask"] = mask_tensor;

    TensorMap outputs = thinker_->forward(inputs);

    auto it = outputs.find("logits");
    if (it == outputs.end())
        throw std::runtime_error("OmniPipeline thinker: no 'logits' output");

    const auto& lt = it->second;
    auto n = lt.numel();
    logits.resize(static_cast<std::size_t>(n));
    std::memcpy(logits.data(), lt.data, n * sizeof(float));

    thinker_cache_->advance();
}

// ─── Talker embed step ───

void OmniPipeline::run_talker_embed_step(
    const float* embed_ptr, int32_t embed_size,
    std::vector<float>& logits)
{
    std::vector<float> mask;
    talker_cache_->build_attention_mask(mask);

    int32_t position = talker_cache_->position();
    float use_input_embed = 1.0F;

    // Copy embed to mutable buffer (Tensor requires non-const pointer)
    std::vector<float> embed_buf(embed_ptr, embed_ptr + embed_size);

    Tensor token_tensor;
    int32_t dummy_token = 0;
    token_tensor.data = &dummy_token;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;

    Tensor position_tensor;
    position_tensor.data = &position;
    position_tensor.shape = {1};
    position_tensor.dtype = DType::kInt32;

    Tensor mask_tensor;
    mask_tensor.data = mask.data();
    mask_tensor.shape = {static_cast<int64_t>(mask.size())};
    mask_tensor.dtype = DType::kFloat32;

    Tensor embed_tensor;
    embed_tensor.data = embed_buf.data();
    embed_tensor.shape = {static_cast<int64_t>(embed_size)};
    embed_tensor.dtype = DType::kFloat32;

    Tensor use_embed_tensor;
    use_embed_tensor.data = &use_input_embed;
    use_embed_tensor.shape = {1};
    use_embed_tensor.dtype = DType::kFloat32;

    TensorMap inputs;
    inputs["token_id"] = token_tensor;
    if (talker_->has_input("position_id"))
        inputs["position_id"] = position_tensor;
    inputs["attention_mask"] = mask_tensor;
    inputs["input_embed"] = embed_tensor;
    inputs["use_input_embed"] = use_embed_tensor;

    TensorMap outputs = talker_->forward(inputs);

    auto it = outputs.find("logits");
    if (it == outputs.end())
        throw std::runtime_error("OmniPipeline talker: no 'logits' output");

    const auto& lt = it->second;
    auto n = lt.numel();
    logits.resize(static_cast<std::size_t>(n));
    std::memcpy(logits.data(), lt.data, n * sizeof(float));

    talker_cache_->advance();
}

// ─── Stage 0: Thinker ───

static int32_t omni_argmax(const std::vector<float>& logits)
{
    if (logits.empty()) return 0;
    return static_cast<int32_t>(
        std::distance(logits.begin(),
                      std::max_element(logits.begin(), logits.end())));
}

std::vector<int32_t> OmniPipeline::run_thinker(
    const std::vector<int32_t>& input_ids,
    int32_t max_tokens,
    std::vector<float>& hidden_states_out)
{
    (void)hidden_states_out;

    thinker_cache_->reset();
    thinker_cache_->bind_to(*thinker_);

    std::vector<float> logits;

    // Prefill: all tokens except the last
    for (std::size_t i = 0; i + 1 < input_ids.size(); ++i)
        run_thinker_step(input_ids[i], logits);

    // Last input token
    if (!input_ids.empty())
        run_thinker_step(input_ids.back(), logits);

    // Decode loop
    std::vector<int32_t> output_ids;
    output_ids.reserve(static_cast<std::size_t>(max_tokens));

    for (int32_t step = 0; step < max_tokens; ++step)
    {
        if (logits.empty()) break;

        int32_t token = omni_argmax(logits);
        if (token == 0) break;  // EOS
        output_ids.push_back(token);

        run_thinker_step(token, logits);
    }

    std::cerr << "[trtf] Omni Thinker: generated " << output_ids.size()
              << " text tokens" << std::endl;

    return output_ids;
}

// ─── Stage 1: Talker ───

std::vector<int32_t> OmniPipeline::run_talker(
    const std::vector<float>& hidden_states,
    int32_t num_tokens)
{
    if (!talker_ || !talker_cache_)
    {
        std::cerr << "[trtf] Omni: no Talker engine" << std::endl;
        return {};
    }

    talker_cache_->reset();
    talker_cache_->bind_to(*talker_);

    const int32_t n_codebooks = config_->talker_n_codebooks;
    const int32_t codebook_size = config_->talker_codebook_size;
    const int32_t talker_hidden = config_->talker_hidden_size;

    const OmniTalkerDecodePlan decode_plan = make_omni_talker_decode_plan(
        n_codebooks, codebook_size, num_tokens);

    std::vector<int32_t> all_codes;
    all_codes.reserve(static_cast<std::size_t>(num_tokens) * n_codebooks);

    std::vector<float> logits;
    for (int32_t t = 0; t < num_tokens; ++t)
    {
        const float* embed_ptr =
            hidden_states.data() + static_cast<std::size_t>(t) * talker_hidden;
        run_talker_embed_step(embed_ptr, talker_hidden, logits);
        append_omni_talker_codes_from_logits(logits, decode_plan, all_codes);
    }

    std::cerr << "[trtf] Omni Talker: generated " << all_codes.size()
              << " codec tokens (" << num_tokens << " frames x "
              << n_codebooks << " codebooks)" << std::endl;

    return all_codes;
}

// ─── Stage 2: Code2Wav ───

std::vector<float> OmniPipeline::run_code2wav(
    const std::vector<int32_t>& codec_tokens,
    int32_t n_codebooks,
    int32_t n_frames)
{
    if (!code2wav_)
    {
        // Fallback: generate simple sine waveform from codec token values
        std::cerr << "[trtf] Omni: no Code2Wav engine, generating simple waveform"
                  << std::endl;
        const int32_t samples_per_frame = config_->sample_rate / 75;
        const int32_t total_samples = n_frames * samples_per_frame;
        std::vector<float> waveform(static_cast<std::size_t>(total_samples), 0.0F);
        for (int32_t f = 0; f < n_frames; ++f)
        {
            const float freq = 200.0F +
                static_cast<float>(codec_tokens[f * n_codebooks]) * 800.0F /
                static_cast<float>(config_->talker_codebook_size);
            const float amp = 0.3F;
            for (int32_t s = 0; s < samples_per_frame; ++s)
            {
                const auto idx = static_cast<std::size_t>(f) * samples_per_frame + s;
                const float t = static_cast<float>(s) /
                                static_cast<float>(config_->sample_rate);
                waveform[idx] = amp * std::sin(2.0F * 3.14159265F * freq * t);
            }
        }
        return waveform;
    }

    const int32_t max_frames = config_->code2wav_max_frames;
    const int32_t actual_frames = std::min(n_frames, max_frames);
    const int32_t upsample = config_->code2wav_upsample_factor;

    std::vector<int32_t> input_codes = build_omni_code2wav_input_codes(
        codec_tokens, n_codebooks, max_frames, actual_frames);

    Tensor codes_tensor;
    codes_tensor.data = input_codes.data();
    codes_tensor.shape = {static_cast<int64_t>(n_codebooks),
                          static_cast<int64_t>(max_frames)};
    codes_tensor.dtype = DType::kInt32;

    TensorMap inputs;
    inputs["codec_tokens"] = codes_tensor;

    TensorMap outputs = code2wav_->forward(inputs);

    auto it = outputs.find("waveform");
    if (it == outputs.end())
    {
        std::cerr << "[trtf] Omni Code2Wav: no 'waveform' output" << std::endl;
        return {};
    }

    const auto& wt = it->second;
    const auto total_out = wt.numel();
    const auto trimmed = static_cast<std::size_t>(actual_frames)
        * static_cast<std::size_t>(upsample);
    const auto copy_n = std::min(total_out, trimmed);

    std::vector<float> waveform(static_cast<std::size_t>(copy_n));
    std::memcpy(waveform.data(), wt.data, copy_n * sizeof(float));

    std::cerr << "[trtf] Omni Code2Wav: " << actual_frames << " frames -> "
              << waveform.size() << " samples" << std::endl;
    return waveform;
}

// ─── Full pipeline ───

AudioResult OmniPipeline::generate_audio(
    const std::string& prompt, const GenerateConfig& cfg)
{
    std::vector<int32_t> input_ids;
    if (tokenizer_)
        input_ids = tokenizer_->encode(prompt);

    int32_t max_tokens = cfg.max_new_tokens > 0 ? cfg.max_new_tokens : 768;

    AudioResult result;
    result.sample_rate = config_->sample_rate;

    std::cerr << "[trtf] Omni: starting pipeline with " << input_ids.size()
              << " input tokens" << std::endl;

    // Stage 0: Thinker generates text tokens (+ hidden states for Talker)
    std::vector<float> hidden_states;
    auto text_tokens = run_thinker(input_ids, max_tokens, hidden_states);
    if (text_tokens.empty())
    {
        std::cerr << "[trtf] Omni: Thinker produced no tokens" << std::endl;
        return result;
    }

    // Stage 1: Talker converts hidden states to RVQ codec tokens
    const OmniTalkerPlan talker_plan = make_omni_talker_plan(
        text_tokens.size(),
        hidden_states.size(),
        talker_ != nullptr);
    if (talker_plan.should_run_talker)
    {
        auto codec_tokens = run_talker(hidden_states, talker_plan.num_tokens);

        const OmniCodecPlan codec_plan = make_omni_codec_plan(
            *config_, codec_tokens.size());
        if (codec_plan.should_run_codec)
        {
            // Stage 2: Code2Wav synthesizes waveform
            auto waveform = run_code2wav(
                codec_tokens,
                codec_plan.n_codebooks,
                codec_plan.n_frames);
            if (!waveform.empty())
            {
                result.samples = std::move(waveform);
                result.num_samples = static_cast<int32_t>(result.samples.size());
            }
        }
    }

    std::cerr << "[trtf] Omni: generated " << result.num_samples << " samples ("
              << (result.num_samples > 0
                  ? static_cast<float>(result.num_samples) / result.sample_rate
                  : 0.0F)
              << "s @ " << result.sample_rate << " Hz)" << std::endl;

    return result;
}

} // namespace trtf

#endif // TRTF_HAS_TRT
