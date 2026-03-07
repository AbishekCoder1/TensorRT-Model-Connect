#include "runtime/services/audio/audio_runtime_services.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

trtf::runtime::adapters::io::DecodedAudio make_decoded_audio()
{
    trtf::runtime::adapters::io::DecodedAudio audio;
    audio.samples = {0.25F, 0.5F};
    audio.sample_rate = 8000;
    return audio;
}

class FakeTokenizer final : public trtf::ITokenizer {
public:
    std::vector<int32_t> encode(const std::string& text) const override
    {
        ++encode_calls;
        last_encoded_text = text;
        return next_encode;
    }

    std::string decode(const std::vector<int32_t>& ids) const override
    {
        ++decode_calls;
        last_decoded_ids = ids;
        return next_decode;
    }

    int32_t id_for_token(std::string_view token) const override
    {
        return static_cast<int32_t>(token.size());
    }

    std::string token_for_id(int32_t id) const override
    {
        return std::to_string(id);
    }

    mutable int encode_calls{0};
    mutable int decode_calls{0};
    mutable std::string last_encoded_text;
    mutable std::vector<int32_t> last_decoded_ids;
    std::vector<int32_t> next_encode{4, 5};
    std::string next_decode{"decoded-transcript"};
};

class FakeAudioPort final : public trtf::runtime::services::common::IAudioGenerationPort {
public:
    trtf::runtime::adapters::io::AudioArtifact generate_audio(
        const std::vector<int32_t>& input_ids,
        int32_t max_tokens) override
    {
        ++calls;
        last_input_ids = input_ids;
        last_max_tokens = max_tokens;
        return next_artifact;
    }

    int calls{0};
    int32_t last_max_tokens{0};
    std::vector<int32_t> last_input_ids;
    trtf::runtime::adapters::io::AudioArtifact next_artifact{{0.1F, 0.2F, 0.3F}, 24000, 3};
};

class FakeTranscriptionPort final : public trtf::runtime::services::common::ITranscriptionPort {
public:
    trtf::runtime::services::common::TranscriptionOutput transcribe(
        const float* mel_data,
        int32_t mel_bins,
        int32_t mel_length,
        int32_t max_new_tokens) override
    {
        ++calls;
        last_mel.assign(mel_data, mel_data + (mel_bins * mel_length));
        last_mel_bins = mel_bins;
        last_mel_length = mel_length;
        last_max_new_tokens = max_new_tokens;
        return next_output;
    }

    int calls{0};
    int32_t last_mel_bins{0};
    int32_t last_mel_length{0};
    int32_t last_max_new_tokens{0};
    std::vector<float> last_mel;
    trtf::runtime::services::common::TranscriptionOutput next_output{"plain-text", {10, 11}, 2};
};

class FakeResampler final : public trtf::runtime::services::common::IAudioResampler {
public:
    std::vector<float> resample(
        const float* samples,
        int32_t n_samples,
        int32_t source_rate,
        int32_t target_rate) const override
    {
        ++calls;
        last_samples.assign(samples, samples + n_samples);
        last_source_rate = source_rate;
        last_target_rate = target_rate;
        return next_output;
    }

    mutable int calls{0};
    mutable int32_t last_source_rate{0};
    mutable int32_t last_target_rate{0};
    mutable std::vector<float> last_samples;
    std::vector<float> next_output{1.0F, 2.0F, 3.0F, 4.0F};
};

class FakeMelPort final : public trtf::runtime::services::common::IMelSpectrogramPort {
public:
    trtf::MelResult extract(
        const float* samples,
        int32_t n_samples,
        const trtf::runtime::services::common::MelSpectrogramConfig& config) const override
    {
        ++calls;
        last_samples.assign(samples, samples + n_samples);
        last_config = config;
        return next_result;
    }

    mutable int calls{0};
    mutable std::vector<float> last_samples;
    mutable trtf::runtime::services::common::MelSpectrogramConfig last_config;
    trtf::MelResult next_result{{0.5F, 0.6F, 0.7F, 0.8F}, 2, 2};
};

class FakeSpeechPort final : public trtf::runtime::services::common::ISpeechSynthesisPort {
public:
    trtf::runtime::services::common::SpeechInputPolicy input_policy() const override
    {
        return policy;
    }

    trtf::runtime::adapters::io::AudioArtifact process_audio(
        const float* input_samples,
        int32_t num_input_samples,
        int32_t max_output_frames,
        int32_t input_sample_rate,
        int32_t tail_frames) override
    {
        ++calls;
        last_samples.assign(input_samples, input_samples + num_input_samples);
        last_max_output_frames = max_output_frames;
        last_input_sample_rate = input_sample_rate;
        last_tail_frames = tail_frames;
        return next_artifact;
    }

    mutable int calls{0};
    mutable int32_t last_max_output_frames{0};
    mutable int32_t last_input_sample_rate{0};
    mutable int32_t last_tail_frames{0};
    mutable std::vector<float> last_samples;
    trtf::runtime::services::common::SpeechInputPolicy policy{16000, false};
    trtf::runtime::adapters::io::AudioArtifact next_artifact{{0.9F, 1.0F}, 16000, 2};
};

void test_audio_services_encode_and_return_artifacts()
{
    auto tokenizer = std::make_shared<FakeTokenizer>();
    auto temp_dir = std::make_shared<trtf::runtime::services::common::ScopedTempDirOwner>();

    auto bark_port = std::make_unique<FakeAudioPort>();
    auto* bark_port_ptr = bark_port.get();
    trtf::runtime::services::audio::BarkAudioService bark(
        std::move(bark_port), tokenizer, temp_dir, 7);
    const auto bark_result = bark.generate_audio({"bark", 0});
    check(bark_result.ok(), "bark audio returns success result");
    check(bark_result.value.num_samples == 3, "bark audio returns artifact samples");
    check(bark_port_ptr->last_input_ids == std::vector<int32_t>({4, 5}), "bark audio encodes prompt");
    check(bark_port_ptr->last_max_tokens == 7, "bark audio uses default max tokens");

    auto magpie_port = std::make_unique<FakeAudioPort>();
    auto* magpie_port_ptr = magpie_port.get();
    trtf::runtime::services::audio::MagpieAudioService magpie(
        std::move(magpie_port), tokenizer, temp_dir, 11);
    const auto magpie_result = magpie.generate_audio({"magpie", 4});
    check(magpie_result.ok(), "magpie audio returns success result");
    check(magpie_port_ptr->last_max_tokens == 4, "magpie audio forwards requested max tokens");

    auto omni_port = std::make_unique<FakeAudioPort>();
    auto* omni_port_ptr = omni_port.get();
    trtf::runtime::services::audio::OmniAudioService omni(
        std::move(omni_port), tokenizer, temp_dir, 9);
    const auto omni_result = omni.generate_audio({"omni", 0});
    check(omni_result.ok(), "omni audio returns success result");
    check(omni_port_ptr->last_max_tokens == 9, "omni audio uses default max tokens");
}

void test_whisper_transcription_service_uses_decoded_audio_resampler_and_mel_port()
{
    auto tokenizer = std::make_shared<FakeTokenizer>();
    auto backend = std::make_unique<FakeTranscriptionPort>();
    auto* backend_ptr = backend.get();
    auto resampler = std::make_unique<FakeResampler>();
    auto* resampler_ptr = resampler.get();
    auto mel_port = std::make_unique<FakeMelPort>();
    auto* mel_port_ptr = mel_port.get();

    trtf::runtime::services::common::MelSpectrogramConfig mel_config;
    mel_config.filterbank = {1.0F, 2.0F};
    mel_config.n_freq_bins = 1;
    mel_config.n_mel_bins = 2;
    mel_config.n_fft = 4;
    mel_config.hop_length = 2;
    mel_config.chunk_length_s = 1;
    mel_config.sample_rate = 16000;

    trtf::runtime::services::audio::WhisperTranscriptionService service(
        std::move(backend),
        tokenizer,
        std::make_shared<trtf::runtime::services::common::ScopedTempDirOwner>(),
        mel_config,
        12,
        std::move(resampler),
        std::move(mel_port));

    const auto result = service.transcribe({make_decoded_audio(), 0});
    check(result.ok(), "whisper transcription returns success result");
    check(result.value == "decoded-transcript", "whisper transcription decodes backend ids");
    check(resampler_ptr->calls == 1, "whisper transcription resamples audio");
    check(resampler_ptr->last_source_rate == 8000 && resampler_ptr->last_target_rate == 16000,
        "whisper transcription uses expected resample rates");
    check(mel_port_ptr->calls == 1, "whisper transcription extracts mel spectrogram");
    check(backend_ptr->calls == 1, "whisper transcription calls backend");
    check(backend_ptr->last_max_new_tokens == 12, "whisper transcription uses default max_new_tokens");
}

void test_speech_to_speech_service_handles_speak_requests()
{
    auto speech_port = std::make_unique<FakeSpeechPort>();
    auto* speech_port_ptr = speech_port.get();
    auto resampler = std::make_unique<FakeResampler>();
    auto* resampler_ptr = resampler.get();

    trtf::runtime::services::audio::SpeechToSpeechAudioService service(
        std::move(speech_port), std::move(resampler));

    const auto unsupported = service.generate_audio({"ignored", 0});
    check(!unsupported.ok(), "speech-to-speech generate_audio is unsupported");
    check(service.supports_speech(), "speech-to-speech service advertises speech support");

    const auto result = service.speak({make_decoded_audio(), 0, 5});
    check(result.ok(), "speech-to-speech speak returns success result");
    check(result.value.num_samples == 2, "speech-to-speech returns output artifact");
    check(resampler_ptr->calls == 1, "speech-to-speech resamples input");
    check(speech_port_ptr->calls == 1, "speech-to-speech calls synthesis backend");
    check(speech_port_ptr->last_max_output_frames == 375, "speech-to-speech uses default max frames");
    check(speech_port_ptr->last_tail_frames == 5, "speech-to-speech forwards tail frames");
}

} // namespace

int main()
{
    test_audio_services_encode_and_return_artifacts();
    test_whisper_transcription_service_uses_decoded_audio_resampler_and_mel_port();
    test_speech_to_speech_service_handles_speak_requests();

    if (g_failures != 0)
    {
        std::cerr << g_failures << " test(s) failed\n";
        return 1;
    }
    return 0;
}
