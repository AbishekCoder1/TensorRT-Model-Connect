#pragma once

#include "trtf/backend.h"
#include "trtf/generation.h"
#include "trtf/runtime/adapters/io/audio_io_adapter.h"
#include "trtf/runtime/adapters/io/detection_io_adapter.h"
#include "trtf/runtime/adapters/io/image_io_adapter.h"
#include "trtf/runtime/adapters/io/media_io_adapter.h"
#include "runtime/trt/audio/mel_spectrogram.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtf {
class BarkBackend;
class DetectionBackend;
class IGenerationBackend;
class MagpieTTSBackend;
class NeuralOperatorBackend;
class OmniBackend;
class SamBackend;
class SegmentationBackend;
class SpeechToSpeechBackend;
class WhisperBackend;
}

namespace trtf::runtime::services::common {

struct TranscriptionOutput {
    std::string text;
    std::vector<int32_t> output_ids;
    int32_t num_tokens{0};
};

struct SpeechInputPolicy {
    int32_t sample_rate{0};
    bool prefers_python_resampler{false};
};

struct MelSpectrogramConfig {
    std::vector<float> filterbank;
    int32_t n_freq_bins{0};
    int32_t n_mel_bins{0};
    int32_t n_fft{0};
    int32_t hop_length{0};
    int32_t chunk_length_s{0};
    int32_t sample_rate{0};
};

struct NeuralOperatorOutput {
    std::vector<float> output;
    int32_t output_dim{0};
    int32_t out_channels{0};
    int32_t height{0};
    int32_t width{0};
};

class ITextGenerationBackendAdapter {
public:
    virtual ~ITextGenerationBackendAdapter() = default;

    virtual std::vector<int32_t> generate_text(
        const std::vector<int32_t>& input_ids,
        const trtf::GenerationConfig& config)
        = 0;

    virtual bool supports_vision() const { return false; }

    virtual bool prepare_image(
        const adapters::io::DecodedImage& image,
        std::vector<float>& image_features,
        int32_t& num_features,
        int32_t& feature_dim,
        std::string& error)
    {
        (void) image;
        (void) image_features;
        (void) num_features;
        (void) feature_dim;
        (void) error;
        return false;
    }

    virtual std::string format_vision_prompt(const std::string& prompt) const
    {
        return prompt;
    }

    virtual std::vector<int32_t> generate_with_image(
        const std::vector<int32_t>& input_ids,
        const float* image_features,
        int32_t num_features,
        int32_t feature_dim,
        const trtf::GenerationConfig& config)
    {
        (void) image_features;
        (void) num_features;
        (void) feature_dim;
        return generate_text(input_ids, config);
    }
};

class ITextGenerationPort {
public:
    virtual ~ITextGenerationPort() = default;

    virtual std::vector<int32_t> generate_text(
        const std::vector<int32_t>& input_ids,
        const trtf::GenerationConfig& config)
        = 0;

    virtual bool supports_vision() const { return false; }

    virtual bool prepare_image(
        const adapters::io::DecodedImage& image,
        std::vector<float>& image_features,
        int32_t& num_features,
        int32_t& feature_dim,
        std::string& error)
    {
        (void) image;
        (void) image_features;
        (void) num_features;
        (void) feature_dim;
        (void) error;
        return false;
    }

    virtual std::string format_vision_prompt(const std::string& prompt) const
    {
        return prompt;
    }

    virtual std::vector<int32_t> generate_with_image(
        const std::vector<int32_t>& input_ids,
        const float* image_features,
        int32_t num_features,
        int32_t feature_dim,
        const trtf::GenerationConfig& config)
    {
        (void) image_features;
        (void) num_features;
        (void) feature_dim;
        return generate_text(input_ids, config);
    }
};

class IAudioGenerationBackendAdapter {
public:
    virtual ~IAudioGenerationBackendAdapter() = default;

    virtual adapters::io::AudioArtifact generate_audio(
        const std::vector<int32_t>& input_ids,
        int32_t max_tokens)
        = 0;
};

class IAudioGenerationPort {
public:
    virtual ~IAudioGenerationPort() = default;

    virtual adapters::io::AudioArtifact generate_audio(
        const std::vector<int32_t>& input_ids,
        int32_t max_tokens)
        = 0;
};

class ITranscriptionBackendAdapter {
public:
    virtual ~ITranscriptionBackendAdapter() = default;

    virtual TranscriptionOutput transcribe(
        const float* mel_data,
        int32_t mel_bins,
        int32_t mel_length,
        int32_t max_new_tokens)
        = 0;
};

class ITranscriptionPort {
public:
    virtual ~ITranscriptionPort() = default;

    virtual TranscriptionOutput transcribe(
        const float* mel_data,
        int32_t mel_bins,
        int32_t mel_length,
        int32_t max_new_tokens)
        = 0;
};

class ISpeechSynthesisBackendAdapter {
public:
    virtual ~ISpeechSynthesisBackendAdapter() = default;

    virtual SpeechInputPolicy input_policy() const = 0;

    virtual adapters::io::AudioArtifact process_audio(
        const float* input_samples,
        int32_t num_input_samples,
        int32_t max_output_frames,
        int32_t input_sample_rate,
        int32_t tail_frames)
        = 0;
};

class ISpeechSynthesisPort {
public:
    virtual ~ISpeechSynthesisPort() = default;

    virtual SpeechInputPolicy input_policy() const = 0;

    virtual adapters::io::AudioArtifact process_audio(
        const float* input_samples,
        int32_t num_input_samples,
        int32_t max_output_frames,
        int32_t input_sample_rate,
        int32_t tail_frames)
        = 0;
};

class ISegmentationBackendAdapter {
public:
    virtual ~ISegmentationBackendAdapter() = default;

    virtual adapters::io::SegmentationArtifact segment_image(const adapters::io::DecodedImage& image) = 0;
};

class ISegmentationPort {
public:
    virtual ~ISegmentationPort() = default;

    virtual adapters::io::SegmentationArtifact segment_image(const adapters::io::DecodedImage& image) = 0;
};

class IPromptedSegmentationBackendAdapter {
public:
    virtual ~IPromptedSegmentationBackendAdapter() = default;

    virtual bool encode_image(const adapters::io::DecodedImage& image) = 0;
    virtual adapters::io::SamMaskArtifact segment_point(float point_x, float point_y, bool is_foreground) = 0;
};

class IPromptedSegmentationPort {
public:
    virtual ~IPromptedSegmentationPort() = default;

    virtual bool encode_image(const adapters::io::DecodedImage& image) = 0;
    virtual adapters::io::SamMaskArtifact segment_point(float point_x, float point_y, bool is_foreground) = 0;
};

class IDetectionBackendAdapter {
public:
    virtual ~IDetectionBackendAdapter() = default;

    virtual adapters::io::DetectionArtifact detect_image(const adapters::io::DecodedImage& image) = 0;
};

class IDetectionPort {
public:
    virtual ~IDetectionPort() = default;

    virtual adapters::io::DetectionArtifact detect_image(const adapters::io::DecodedImage& image) = 0;
};

class INeuralOperatorBackendAdapter {
public:
    virtual ~INeuralOperatorBackendAdapter() = default;

    virtual NeuralOperatorOutput solve(
        const float* branch_input,
        int32_t branch_len,
        const float* trunk_input,
        int32_t trunk_len)
        = 0;

    virtual NeuralOperatorOutput solve_field(const float* field_input, int32_t input_size) = 0;
};

class INeuralOperatorPort {
public:
    virtual ~INeuralOperatorPort() = default;

    virtual NeuralOperatorOutput solve(
        const float* branch_input,
        int32_t branch_len,
        const float* trunk_input,
        int32_t trunk_len)
        = 0;

    virtual NeuralOperatorOutput solve_field(const float* field_input, int32_t input_size) = 0;
};

class IAudioResampler {
public:
    virtual ~IAudioResampler() = default;

    virtual std::vector<float> resample(
        const float* samples,
        int32_t n_samples,
        int32_t source_rate,
        int32_t target_rate) const
        = 0;
};

class IMelSpectrogramPort {
public:
    virtual ~IMelSpectrogramPort() = default;

    virtual trtf::MelResult extract(
        const float* samples,
        int32_t n_samples,
        const MelSpectrogramConfig& config) const
        = 0;
};

class GenerationBackendPort final : public ITextGenerationPort {
public:
    explicit GenerationBackendPort(std::unique_ptr<trtf::IGenerationBackend> backend);
    explicit GenerationBackendPort(std::unique_ptr<ITextGenerationBackendAdapter> backend);

    std::vector<int32_t> generate_text(
        const std::vector<int32_t>& input_ids,
        const trtf::GenerationConfig& config) override;
    bool supports_vision() const override;
    bool prepare_image(
        const adapters::io::DecodedImage& image,
        std::vector<float>& image_features,
        int32_t& num_features,
        int32_t& feature_dim,
        std::string& error) override;
    std::string format_vision_prompt(const std::string& prompt) const override;
    std::vector<int32_t> generate_with_image(
        const std::vector<int32_t>& input_ids,
        const float* image_features,
        int32_t num_features,
        int32_t feature_dim,
        const trtf::GenerationConfig& config) override;

private:
    std::unique_ptr<ITextGenerationBackendAdapter> mBackend;
};

class OmniTextGenerationPort final : public ITextGenerationPort {
public:
    explicit OmniTextGenerationPort(std::shared_ptr<trtf::OmniBackend> backend);
    explicit OmniTextGenerationPort(std::unique_ptr<ITextGenerationBackendAdapter> backend);

    std::vector<int32_t> generate_text(
        const std::vector<int32_t>& input_ids,
        const trtf::GenerationConfig& config) override;

private:
    std::unique_ptr<ITextGenerationBackendAdapter> mBackend;
};

class BarkAudioGenerationPort final : public IAudioGenerationPort {
public:
    explicit BarkAudioGenerationPort(std::unique_ptr<trtf::BarkBackend> backend);
    explicit BarkAudioGenerationPort(std::unique_ptr<IAudioGenerationBackendAdapter> backend);
    ~BarkAudioGenerationPort() override;

    adapters::io::AudioArtifact generate_audio(
        const std::vector<int32_t>& input_ids,
        int32_t max_tokens) override;

private:
    std::unique_ptr<IAudioGenerationBackendAdapter> mBackend;
};

class MagpieAudioGenerationPort final : public IAudioGenerationPort {
public:
    explicit MagpieAudioGenerationPort(std::unique_ptr<trtf::MagpieTTSBackend> backend);
    explicit MagpieAudioGenerationPort(std::unique_ptr<IAudioGenerationBackendAdapter> backend);
    ~MagpieAudioGenerationPort() override;

    adapters::io::AudioArtifact generate_audio(
        const std::vector<int32_t>& input_ids,
        int32_t max_tokens) override;

private:
    std::unique_ptr<IAudioGenerationBackendAdapter> mBackend;
};

class OmniAudioGenerationPort final : public IAudioGenerationPort {
public:
    explicit OmniAudioGenerationPort(std::shared_ptr<trtf::OmniBackend> backend);
    explicit OmniAudioGenerationPort(std::unique_ptr<IAudioGenerationBackendAdapter> backend);

    adapters::io::AudioArtifact generate_audio(
        const std::vector<int32_t>& input_ids,
        int32_t max_tokens) override;

private:
    std::unique_ptr<IAudioGenerationBackendAdapter> mBackend;
};

class WhisperTranscriptionPort final : public ITranscriptionPort {
public:
    explicit WhisperTranscriptionPort(std::unique_ptr<trtf::WhisperBackend> backend);
    explicit WhisperTranscriptionPort(std::unique_ptr<ITranscriptionBackendAdapter> backend);
    ~WhisperTranscriptionPort() override;

    TranscriptionOutput transcribe(
        const float* mel_data,
        int32_t mel_bins,
        int32_t mel_length,
        int32_t max_new_tokens) override;

private:
    std::unique_ptr<ITranscriptionBackendAdapter> mBackend;
};

class SpeechSynthesisPort final : public ISpeechSynthesisPort {
public:
    explicit SpeechSynthesisPort(std::unique_ptr<trtf::SpeechToSpeechBackend> backend);
    explicit SpeechSynthesisPort(std::unique_ptr<ISpeechSynthesisBackendAdapter> backend);
    ~SpeechSynthesisPort() override;

    SpeechInputPolicy input_policy() const override;
    adapters::io::AudioArtifact process_audio(
        const float* input_samples,
        int32_t num_input_samples,
        int32_t max_output_frames,
        int32_t input_sample_rate,
        int32_t tail_frames) override;

private:
    std::unique_ptr<ISpeechSynthesisBackendAdapter> mBackend;
};

class SegmentationBackendPort final : public ISegmentationPort {
public:
    explicit SegmentationBackendPort(std::unique_ptr<trtf::SegmentationBackend> backend);
    explicit SegmentationBackendPort(std::unique_ptr<ISegmentationBackendAdapter> backend);
    ~SegmentationBackendPort() override;

    adapters::io::SegmentationArtifact segment_image(const adapters::io::DecodedImage& image) override;

private:
    std::unique_ptr<ISegmentationBackendAdapter> mBackend;
};

class SamBackendPort final : public IPromptedSegmentationPort {
public:
    explicit SamBackendPort(std::unique_ptr<trtf::SamBackend> backend);
    explicit SamBackendPort(std::unique_ptr<IPromptedSegmentationBackendAdapter> backend);
    ~SamBackendPort() override;

    bool encode_image(const adapters::io::DecodedImage& image) override;
    adapters::io::SamMaskArtifact segment_point(float point_x, float point_y, bool is_foreground) override;

private:
    std::unique_ptr<IPromptedSegmentationBackendAdapter> mBackend;
};

class DetectionBackendPort final : public IDetectionPort {
public:
    explicit DetectionBackendPort(std::unique_ptr<trtf::DetectionBackend> backend);
    explicit DetectionBackendPort(std::unique_ptr<IDetectionBackendAdapter> backend);
    ~DetectionBackendPort() override;

    adapters::io::DetectionArtifact detect_image(const adapters::io::DecodedImage& image) override;

private:
    std::unique_ptr<IDetectionBackendAdapter> mBackend;
};

class NeuralOperatorBackendPort final : public INeuralOperatorPort {
public:
    explicit NeuralOperatorBackendPort(std::unique_ptr<trtf::NeuralOperatorBackend> backend);
    explicit NeuralOperatorBackendPort(std::unique_ptr<INeuralOperatorBackendAdapter> backend);
    ~NeuralOperatorBackendPort() override;

    NeuralOperatorOutput solve(
        const float* branch_input,
        int32_t branch_len,
        const float* trunk_input,
        int32_t trunk_len) override;
    NeuralOperatorOutput solve_field(const float* field_input, int32_t input_size) override;

private:
    std::unique_ptr<INeuralOperatorBackendAdapter> mBackend;
};

class DefaultAudioResampler final : public IAudioResampler {
public:
    std::vector<float> resample(
        const float* samples,
        int32_t n_samples,
        int32_t source_rate,
        int32_t target_rate) const override;
};

class DefaultMelSpectrogramPort final : public IMelSpectrogramPort {
public:
    trtf::MelResult extract(
        const float* samples,
        int32_t n_samples,
        const MelSpectrogramConfig& config) const override;
};

} // namespace trtf::runtime::services::common
