#pragma once

#include "trtf/runtime/contracts/services.h"
#include "trtf/tokenizer.h"
#include "runtime/services/common/runtime_service_ports.h"
#include "runtime/services/common/scoped_temp_dir.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtf::runtime::services::audio {

class BarkAudioService final : public trtf::runtime::IAudioService {
public:
    BarkAudioService(
        std::unique_ptr<common::IAudioGenerationPort> backend,
        std::shared_ptr<trtf::ITokenizer> tokenizer,
        std::shared_ptr<common::ScopedTempDirOwner> tokenizer_temp_dir,
        int32_t default_max_tokens);

    AudioGenerationResult generate_audio(const AudioGenerationRequest& request) override;

private:
    std::shared_ptr<common::ScopedTempDirOwner> mTokenizerTempDir;
    std::unique_ptr<common::IAudioGenerationPort> mBackend;
    std::shared_ptr<trtf::ITokenizer> mTokenizer;
    int32_t mDefaultMaxTokens{0};
};

class MagpieAudioService final : public trtf::runtime::IAudioService {
public:
    MagpieAudioService(
        std::unique_ptr<common::IAudioGenerationPort> backend,
        std::shared_ptr<trtf::ITokenizer> tokenizer,
        std::shared_ptr<common::ScopedTempDirOwner> tokenizer_temp_dir,
        int32_t default_max_frames);

    AudioGenerationResult generate_audio(const AudioGenerationRequest& request) override;

private:
    std::shared_ptr<common::ScopedTempDirOwner> mTokenizerTempDir;
    std::unique_ptr<common::IAudioGenerationPort> mBackend;
    std::shared_ptr<trtf::ITokenizer> mTokenizer;
    int32_t mDefaultMaxFrames{0};
};

class OmniAudioService final : public trtf::runtime::IAudioService {
public:
    OmniAudioService(
        std::unique_ptr<common::IAudioGenerationPort> backend,
        std::shared_ptr<trtf::ITokenizer> tokenizer,
        std::shared_ptr<common::ScopedTempDirOwner> tokenizer_temp_dir,
        int32_t default_max_tokens);

    AudioGenerationResult generate_audio(const AudioGenerationRequest& request) override;

private:
    std::shared_ptr<common::ScopedTempDirOwner> mTokenizerTempDir;
    std::unique_ptr<common::IAudioGenerationPort> mBackend;
    std::shared_ptr<trtf::ITokenizer> mTokenizer;
    int32_t mDefaultMaxTokens{0};
};

class WhisperTranscriptionService final : public trtf::runtime::ITranscriptionService {
public:
    WhisperTranscriptionService(
        std::unique_ptr<common::ITranscriptionPort> backend,
        std::shared_ptr<trtf::ITokenizer> tokenizer,
        std::shared_ptr<common::ScopedTempDirOwner> tokenizer_temp_dir,
        common::MelSpectrogramConfig mel_config,
        int32_t default_max_new_tokens,
        std::unique_ptr<common::IAudioResampler> resampler = {},
        std::unique_ptr<common::IMelSpectrogramPort> mel_port = {});

    TranscriptionResult transcribe(const TranscriptionRequest& request) override;

private:
    std::shared_ptr<common::ScopedTempDirOwner> mTokenizerTempDir;
    std::unique_ptr<common::ITranscriptionPort> mBackend;
    std::shared_ptr<trtf::ITokenizer> mTokenizer;
    common::MelSpectrogramConfig mMelConfig;
    int32_t mDefaultMaxNewTokens{224};
    std::unique_ptr<common::IAudioResampler> mResampler;
    std::unique_ptr<common::IMelSpectrogramPort> mMelPort;
};

class SpeechToSpeechAudioService final : public trtf::runtime::IAudioService {
public:
    explicit SpeechToSpeechAudioService(
        std::unique_ptr<common::ISpeechSynthesisPort> backend,
        std::unique_ptr<common::IAudioResampler> resampler = {});

    AudioGenerationResult generate_audio(const AudioGenerationRequest& request) override;
    bool supports_speech() const override;
    SpeechSynthesisResult speak(const SpeechSynthesisRequest& request) override;

private:
    struct PreparedInput {
        std::vector<float> samples;
        int32_t sample_rate{0};
    };

    PreparedInput prepare_input(std::vector<float> samples, int32_t sample_rate) const;

    std::unique_ptr<common::ISpeechSynthesisPort> mBackend;
    std::unique_ptr<common::IAudioResampler> mResampler;
};

} // namespace trtf::runtime::services::audio
