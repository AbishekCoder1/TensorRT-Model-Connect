#include "runtime/services/audio/audio_runtime_services.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <utility>
#include <vector>

namespace trtf::runtime::services::audio {
namespace {

int32_t effective_limit(int32_t requested, int32_t fallback)
{
    return requested > 0 ? requested : fallback;
}

std::vector<int32_t> encode_text(trtf::ITokenizer* tokenizer, const std::string& prompt)
{
    if (tokenizer == nullptr || prompt.empty())
    {
        return {};
    }
    return tokenizer->encode(prompt);
}

std::unique_ptr<common::IAudioResampler> make_default_resampler(std::unique_ptr<common::IAudioResampler> resampler)
{
    if (resampler != nullptr)
    {
        return resampler;
    }
    return std::make_unique<common::DefaultAudioResampler>();
}

std::unique_ptr<common::IMelSpectrogramPort> make_default_mel_port(
    std::unique_ptr<common::IMelSpectrogramPort> mel_port)
{
    if (mel_port != nullptr)
    {
        return mel_port;
    }
    return std::make_unique<common::DefaultMelSpectrogramPort>();
}

AudioGenerationResult invalid_audio_request(const char* message)
{
    return AudioGenerationResult::Failure(RuntimeServiceStatus::kInvalidArgument, message);
}

SpeechSynthesisResult invalid_speech_request(const char* message)
{
    return SpeechSynthesisResult::Failure(RuntimeServiceStatus::kInvalidArgument, message);
}

} // namespace

BarkAudioService::BarkAudioService(
    std::unique_ptr<common::IAudioGenerationPort> backend,
    std::shared_ptr<trtf::ITokenizer> tokenizer,
    std::shared_ptr<common::ScopedTempDirOwner> tokenizer_temp_dir,
    int32_t default_max_tokens)
    : mTokenizerTempDir(std::move(tokenizer_temp_dir))
    , mBackend(std::move(backend))
    , mTokenizer(std::move(tokenizer))
    , mDefaultMaxTokens(default_max_tokens)
{
}

AudioGenerationResult BarkAudioService::generate_audio(const AudioGenerationRequest& request)
{
    if (mBackend == nullptr)
    {
        return invalid_audio_request("audio backend missing");
    }

    try
    {
        const auto input_ids = encode_text(mTokenizer.get(), request.prompt);
        return AudioGenerationResult::Success(
            mBackend->generate_audio(input_ids, effective_limit(request.max_tokens, mDefaultMaxTokens)));
    }
    catch (const std::exception& e)
    {
        return AudioGenerationResult::Failure(RuntimeServiceStatus::kRuntimeError, e.what());
    }
}

MagpieAudioService::MagpieAudioService(
    std::unique_ptr<common::IAudioGenerationPort> backend,
    std::shared_ptr<trtf::ITokenizer> tokenizer,
    std::shared_ptr<common::ScopedTempDirOwner> tokenizer_temp_dir,
    int32_t default_max_frames)
    : mTokenizerTempDir(std::move(tokenizer_temp_dir))
    , mBackend(std::move(backend))
    , mTokenizer(std::move(tokenizer))
    , mDefaultMaxFrames(default_max_frames)
{
}

AudioGenerationResult MagpieAudioService::generate_audio(const AudioGenerationRequest& request)
{
    if (mBackend == nullptr)
    {
        return invalid_audio_request("audio backend missing");
    }

    try
    {
        const auto input_ids = encode_text(mTokenizer.get(), request.prompt);
        return AudioGenerationResult::Success(
            mBackend->generate_audio(input_ids, effective_limit(request.max_tokens, mDefaultMaxFrames)));
    }
    catch (const std::exception& e)
    {
        return AudioGenerationResult::Failure(RuntimeServiceStatus::kRuntimeError, e.what());
    }
}

OmniAudioService::OmniAudioService(
    std::unique_ptr<common::IAudioGenerationPort> backend,
    std::shared_ptr<trtf::ITokenizer> tokenizer,
    std::shared_ptr<common::ScopedTempDirOwner> tokenizer_temp_dir,
    int32_t default_max_tokens)
    : mTokenizerTempDir(std::move(tokenizer_temp_dir))
    , mBackend(std::move(backend))
    , mTokenizer(std::move(tokenizer))
    , mDefaultMaxTokens(default_max_tokens)
{
}

AudioGenerationResult OmniAudioService::generate_audio(const AudioGenerationRequest& request)
{
    if (mBackend == nullptr)
    {
        return invalid_audio_request("audio backend missing");
    }

    try
    {
        const auto input_ids = encode_text(mTokenizer.get(), request.prompt);
        return AudioGenerationResult::Success(
            mBackend->generate_audio(input_ids, effective_limit(request.max_tokens, mDefaultMaxTokens)));
    }
    catch (const std::exception& e)
    {
        return AudioGenerationResult::Failure(RuntimeServiceStatus::kRuntimeError, e.what());
    }
}

WhisperTranscriptionService::WhisperTranscriptionService(
    std::unique_ptr<common::ITranscriptionPort> backend,
    std::shared_ptr<trtf::ITokenizer> tokenizer,
    std::shared_ptr<common::ScopedTempDirOwner> tokenizer_temp_dir,
    common::MelSpectrogramConfig mel_config,
    int32_t default_max_new_tokens,
    std::unique_ptr<common::IAudioResampler> resampler,
    std::unique_ptr<common::IMelSpectrogramPort> mel_port)
    : mTokenizerTempDir(std::move(tokenizer_temp_dir))
    , mBackend(std::move(backend))
    , mTokenizer(std::move(tokenizer))
    , mMelConfig(std::move(mel_config))
    , mDefaultMaxNewTokens(default_max_new_tokens)
    , mResampler(make_default_resampler(std::move(resampler)))
    , mMelPort(make_default_mel_port(std::move(mel_port)))
{
}

TranscriptionResult WhisperTranscriptionService::transcribe(const TranscriptionRequest& request)
{
    if (mBackend == nullptr || request.input.empty())
    {
        return TranscriptionResult::Failure(RuntimeServiceStatus::kInvalidArgument, "transcription backend or audio input missing");
    }
    if (mMelConfig.filterbank.empty())
    {
        return TranscriptionResult::Failure(
            RuntimeServiceStatus::kInvalidArgument,
            "Bundle missing mel_filterbank section; rebuild with latest trtf-build");
    }

    try
    {
        const auto& decoded_audio = request.input.decoded;
        const float* audio_ptr = decoded_audio.samples.data();
        int32_t audio_len = static_cast<int32_t>(decoded_audio.samples.size());
        std::vector<float> resampled;
        if (decoded_audio.sample_rate != mMelConfig.sample_rate && decoded_audio.sample_rate > 0)
        {
            resampled = mResampler->resample(audio_ptr, audio_len, decoded_audio.sample_rate, mMelConfig.sample_rate);
            audio_ptr = resampled.data();
            audio_len = static_cast<int32_t>(resampled.size());
        }

        const auto mel = mMelPort->extract(audio_ptr, audio_len, mMelConfig);
        const auto output = mBackend->transcribe(
            mel.data.data(),
            mel.n_mels,
            mel.n_frames,
            effective_limit(request.max_new_tokens, mDefaultMaxNewTokens));

        if (mTokenizer != nullptr && !output.output_ids.empty())
        {
            return TranscriptionResult::Success(mTokenizer->decode(output.output_ids));
        }
        return TranscriptionResult::Success(output.text);
    }
    catch (const std::exception& e)
    {
        return TranscriptionResult::Failure(RuntimeServiceStatus::kRuntimeError, e.what());
    }
}

SpeechToSpeechAudioService::SpeechToSpeechAudioService(
    std::unique_ptr<common::ISpeechSynthesisPort> backend,
    std::unique_ptr<common::IAudioResampler> resampler)
    : mBackend(std::move(backend))
    , mResampler(make_default_resampler(std::move(resampler)))
{
}

AudioGenerationResult SpeechToSpeechAudioService::generate_audio(const AudioGenerationRequest& request)
{
    (void) request;
    return AudioGenerationResult::Failure(
        RuntimeServiceStatus::kUnsupported, "speech-to-speech service does not support text-to-audio generation");
}

bool SpeechToSpeechAudioService::supports_speech() const
{
    return mBackend != nullptr;
}

SpeechSynthesisResult SpeechToSpeechAudioService::speak(const SpeechSynthesisRequest& request)
{
    if (mBackend == nullptr || request.input.empty())
    {
        return invalid_speech_request("speech backend or audio input missing");
    }

    try
    {
        auto prepared = prepare_input(request.input.decoded.samples, request.input.decoded.sample_rate);
        if (prepared.samples.empty())
        {
            return invalid_speech_request("prepared speech input is empty");
        }

        return SpeechSynthesisResult::Success(mBackend->process_audio(
            prepared.samples.data(),
            static_cast<int32_t>(prepared.samples.size()),
            effective_limit(request.max_output_frames, 375),
            prepared.sample_rate,
            std::max(0, request.tail_frames)));
    }
    catch (const std::exception& e)
    {
        return SpeechSynthesisResult::Failure(RuntimeServiceStatus::kRuntimeError, e.what());
    }
}

SpeechToSpeechAudioService::PreparedInput SpeechToSpeechAudioService::prepare_input(
    std::vector<float> samples,
    int32_t sample_rate) const
{
    const auto policy = mBackend != nullptr ? mBackend->input_policy() : common::SpeechInputPolicy{};
    const int32_t target_rate = policy.sample_rate;
    if (sample_rate != target_rate && sample_rate > 0 && target_rate > 0)
    {
        if (!policy.prefers_python_resampler)
        {
            samples = mResampler->resample(samples.data(), static_cast<int32_t>(samples.size()), sample_rate, target_rate);
            sample_rate = target_rate;
        }
    }
    return {std::move(samples), sample_rate};
}

} // namespace trtf::runtime::services::audio
