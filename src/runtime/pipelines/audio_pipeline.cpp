#include "runtime/pipelines/audio_pipeline.h"

#if TRTF_HAS_TRT

#include "runtime/trt/audio/whisper_backend.h"
#include "runtime/trt/audio/bark_backend.h"
#include "runtime/trt/audio/magpie_tts_backend.h"
#include "runtime/trt/audio/speech_backend.h"
#include "runtime/trt/audio/omni_backend.h"
#include "runtime/trt/audio/mel_spectrogram.h"
#include "cabi/bundle/bundle_helpers.h"
#include "utils/wav_reader.h"
#include "trtf/tokenizer.h"

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
        samples_ptr, samples_count, max_frames, input_sample_rate);

    AudioResult out;
    out.samples = std::move(speech_result.waveform);
    out.num_samples = speech_result.num_samples;
    out.sample_rate = speech_result.sample_rate;
    return out;
}

// ─── OmniPipeline ───

OmniPipeline::OmniPipeline(
    std::unique_ptr<OmniBackend> backend,
    std::shared_ptr<ITokenizer> tokenizer,
    std::string model_id_str)
    : backend_(std::move(backend))
    , tokenizer_(std::move(tokenizer))
    , model_id_(std::move(model_id_str))
{
    if (!backend_ || !backend_->is_available())
        throw std::runtime_error("OmniPipeline: invalid OmniBackend");
}

OmniPipeline::~OmniPipeline() = default;

AudioResult OmniPipeline::generate_audio(
    const std::string& prompt, const GenerateConfig& cfg)
{
    std::vector<int32_t> input_ids;
    if (tokenizer_)
        input_ids = tokenizer_->encode(prompt);

    int32_t max_tokens = cfg.max_new_tokens > 0 ? cfg.max_new_tokens : 768;

    auto omni_result = backend_->generate_audio(input_ids, max_tokens);

    AudioResult out;
    out.samples = std::move(omni_result.waveform);
    out.num_samples = omni_result.num_samples;
    out.sample_rate = omni_result.sample_rate;
    return out;
}

} // namespace trtf

#endif // TRTF_HAS_TRT
