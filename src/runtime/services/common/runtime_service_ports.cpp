#include "runtime/services/common/runtime_service_ports.h"

#include "runtime/trt/audio/bark_backend.h"
#include "runtime/trt/audio/magpie_tts_backend.h"
#include "runtime/trt/audio/omni_backend.h"
#include "runtime/trt/audio/speech_backend.h"
#include "runtime/trt/audio/whisper_backend.h"
#include "runtime/trt/multimodal/image_preprocessor.h"
#include "runtime/trt/multimodal/vl_backend.h"
#include "runtime/trt/perception/detection_backend.h"
#include "runtime/trt/perception/neural_operator_backend.h"
#include "runtime/trt/perception/sam_backend.h"
#include "runtime/trt/perception/segmentation_backend.h"
#include "utils/wav_reader.h"

#include <stdexcept>
#include <utility>

namespace trtf::runtime::services::common {
namespace {

adapters::io::AudioArtifact to_audio_artifact(const trtf::AudioResult& result)
{
    adapters::io::AudioArtifact artifact;
    artifact.waveform = result.waveform;
    artifact.sample_rate = result.sample_rate;
    artifact.num_samples = result.num_samples;
    return artifact;
}

adapters::io::SegmentationArtifact to_segmentation_artifact(const trtf::SegmentationResult& result)
{
    adapters::io::SegmentationArtifact artifact;
    artifact.class_map = result.class_map;
    artifact.width = result.width;
    artifact.height = result.height;
    return artifact;
}

adapters::io::SamMaskArtifact to_sam_mask_artifact(const trtf::SamResult& result)
{
    adapters::io::SamMaskArtifact artifact;
    artifact.masks = result.masks;
    artifact.iou_scores = result.iou_scores;
    artifact.num_masks = result.num_masks;
    artifact.mask_width = result.mask_width;
    artifact.mask_height = result.mask_height;
    return artifact;
}

adapters::io::DetectionArtifact to_detection_artifact(const trtf::DetectionResult& result)
{
    adapters::io::DetectionArtifact artifact;
    artifact.detections.reserve(result.detections.size());
    for (const auto& detection : result.detections)
    {
        artifact.detections.push_back({
            detection.class_id,
            detection.confidence,
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2,
        });
    }
    return artifact;
}

class GenerationBackendAdapter final : public ITextGenerationBackendAdapter {
public:
    explicit GenerationBackendAdapter(std::unique_ptr<trtf::IGenerationBackend> backend)
        : mBackend(std::move(backend))
        , mVisionBackend(mBackend.get())
    {
    }

    std::vector<int32_t> generate_text(
        const std::vector<int32_t>& input_ids,
        const trtf::GenerationConfig& config) override
    {
        if (mBackend == nullptr)
        {
            return {};
        }
        return mBackend->generate(input_ids, config);
    }

    bool supports_vision() const override
    {
        return mBackend != nullptr && mBackend->supports_vision();
    }

    bool prepare_image(
        const adapters::io::DecodedImage& image,
        std::vector<float>& image_features,
        int32_t& num_features,
        int32_t& feature_dim,
        std::string& error) override
    {
#if TRTF_HAS_TRT
        auto* vl = dynamic_cast<trtf::VLBackendFastPath*>(mVisionBackend);
        if (vl == nullptr)
        {
            return false;
        }
        return vl->prepare_image(image, image_features, num_features, feature_dim, error);
#else
        (void) image;
        (void) image_features;
        (void) num_features;
        (void) feature_dim;
        (void) error;
        return false;
#endif
    }

    std::string format_vision_prompt(const std::string& prompt) const override
    {
#if TRTF_HAS_TRT
        auto* vl = dynamic_cast<trtf::VLBackendFastPath*>(mVisionBackend);
        if (vl == nullptr)
        {
            return prompt;
        }
        return trtf::format_vl_prompt(prompt, vl->vl_config());
#else
        return prompt;
#endif
    }

    std::vector<int32_t> generate_with_image(
        const std::vector<int32_t>& input_ids,
        const float* image_features,
        int32_t num_features,
        int32_t feature_dim,
        const trtf::GenerationConfig& config) override
    {
#if TRTF_HAS_TRT
        auto* vl = dynamic_cast<trtf::VLBackendFastPath*>(mVisionBackend);
        if (vl == nullptr)
        {
            return generate_text(input_ids, config);
        }
        return vl->generate_vl(input_ids, image_features, num_features, feature_dim, {}, config);
#else
        (void) image_features;
        (void) num_features;
        (void) feature_dim;
        return generate_text(input_ids, config);
#endif
    }

private:
    std::unique_ptr<trtf::IGenerationBackend> mBackend;
    trtf::IGenerationBackend* mVisionBackend{nullptr};
};

class OmniTextGenerationBackendAdapter final : public ITextGenerationBackendAdapter {
public:
    explicit OmniTextGenerationBackendAdapter(std::shared_ptr<trtf::OmniBackend> backend)
        : mBackend(std::move(backend))
    {
    }

    std::vector<int32_t> generate_text(
        const std::vector<int32_t>& input_ids,
        const trtf::GenerationConfig& config) override
    {
        if (mBackend == nullptr)
        {
            return {};
        }
        return mBackend->generate_text(input_ids, static_cast<int32_t>(config.max_new_tokens));
    }

private:
    std::shared_ptr<trtf::OmniBackend> mBackend;
};

class BarkAudioGenerationBackendAdapter final : public IAudioGenerationBackendAdapter {
public:
    explicit BarkAudioGenerationBackendAdapter(std::unique_ptr<trtf::BarkBackend> backend)
        : mBackend(std::move(backend))
    {
    }

    adapters::io::AudioArtifact generate_audio(
        const std::vector<int32_t>& input_ids,
        int32_t max_tokens) override
    {
        if (mBackend == nullptr)
        {
            return {};
        }
        return to_audio_artifact(mBackend->generate_audio(input_ids, max_tokens));
    }

private:
    std::unique_ptr<trtf::BarkBackend> mBackend;
};

class MagpieAudioGenerationBackendAdapter final : public IAudioGenerationBackendAdapter {
public:
    explicit MagpieAudioGenerationBackendAdapter(std::unique_ptr<trtf::MagpieTTSBackend> backend)
        : mBackend(std::move(backend))
    {
    }

    adapters::io::AudioArtifact generate_audio(
        const std::vector<int32_t>& input_ids,
        int32_t max_tokens) override
    {
        if (mBackend == nullptr)
        {
            return {};
        }
        return to_audio_artifact(mBackend->generate_audio(input_ids, max_tokens));
    }

private:
    std::unique_ptr<trtf::MagpieTTSBackend> mBackend;
};

class OmniAudioGenerationBackendAdapter final : public IAudioGenerationBackendAdapter {
public:
    explicit OmniAudioGenerationBackendAdapter(std::shared_ptr<trtf::OmniBackend> backend)
        : mBackend(std::move(backend))
    {
    }

    adapters::io::AudioArtifact generate_audio(
        const std::vector<int32_t>& input_ids,
        int32_t max_tokens) override
    {
        if (mBackend == nullptr)
        {
            return {};
        }
        return to_audio_artifact(mBackend->generate_audio(input_ids, max_tokens));
    }

private:
    std::shared_ptr<trtf::OmniBackend> mBackend;
};

class WhisperTranscriptionBackendAdapter final : public ITranscriptionBackendAdapter {
public:
    explicit WhisperTranscriptionBackendAdapter(std::unique_ptr<trtf::WhisperBackend> backend)
        : mBackend(std::move(backend))
    {
    }

    TranscriptionOutput transcribe(
        const float* mel_data,
        int32_t mel_bins,
        int32_t mel_length,
        int32_t max_new_tokens) override
    {
        if (mBackend == nullptr)
        {
            return {};
        }
        const auto result = mBackend->transcribe(mel_data, mel_bins, mel_length, max_new_tokens);
        return {result.text, result.output_ids, result.num_tokens};
    }

private:
    std::unique_ptr<trtf::WhisperBackend> mBackend;
};

class SpeechSynthesisBackendAdapter final : public ISpeechSynthesisBackendAdapter {
public:
    explicit SpeechSynthesisBackendAdapter(std::unique_ptr<trtf::SpeechToSpeechBackend> backend)
        : mBackend(std::move(backend))
    {
    }

    SpeechInputPolicy input_policy() const override
    {
        if (mBackend == nullptr)
        {
            return {};
        }
        const auto& config = mBackend->config();
        return {config.sample_rate, !config.hf_python.empty()};
    }

    adapters::io::AudioArtifact process_audio(
        const float* input_samples,
        int32_t num_input_samples,
        int32_t max_output_frames,
        int32_t input_sample_rate,
        int32_t tail_frames) override
    {
        if (mBackend == nullptr)
        {
            return {};
        }
        return to_audio_artifact(
            mBackend->process_audio(input_samples, num_input_samples, max_output_frames, input_sample_rate, tail_frames));
    }

private:
    std::unique_ptr<trtf::SpeechToSpeechBackend> mBackend;
};

class SegmentationBackendAdapter final : public ISegmentationBackendAdapter {
public:
    explicit SegmentationBackendAdapter(std::unique_ptr<trtf::SegmentationBackend> backend)
        : mBackend(std::move(backend))
    {
    }

    adapters::io::SegmentationArtifact segment_image(const adapters::io::DecodedImage& image) override
    {
        if (mBackend == nullptr)
        {
            return {};
        }
        return to_segmentation_artifact(mBackend->segment_image(image));
    }

private:
    std::unique_ptr<trtf::SegmentationBackend> mBackend;
};

class SamBackendAdapter final : public IPromptedSegmentationBackendAdapter {
public:
    explicit SamBackendAdapter(std::unique_ptr<trtf::SamBackend> backend)
        : mBackend(std::move(backend))
    {
    }

    bool encode_image(const adapters::io::DecodedImage& image) override
    {
        return mBackend != nullptr && mBackend->encode_image(image);
    }

    adapters::io::SamMaskArtifact segment_point(float point_x, float point_y, bool is_foreground) override
    {
        if (mBackend == nullptr)
        {
            return {};
        }
        return to_sam_mask_artifact(mBackend->segment_point(point_x, point_y, is_foreground));
    }

private:
    std::unique_ptr<trtf::SamBackend> mBackend;
};

class DetectionBackendAdapter final : public IDetectionBackendAdapter {
public:
    explicit DetectionBackendAdapter(std::unique_ptr<trtf::DetectionBackend> backend)
        : mBackend(std::move(backend))
    {
    }

    adapters::io::DetectionArtifact detect_image(const adapters::io::DecodedImage& image) override
    {
        if (mBackend == nullptr)
        {
            return {};
        }
        return to_detection_artifact(mBackend->detect_image(image));
    }

private:
    std::unique_ptr<trtf::DetectionBackend> mBackend;
};

class NeuralOperatorBackendAdapter final : public INeuralOperatorBackendAdapter {
public:
    explicit NeuralOperatorBackendAdapter(std::unique_ptr<trtf::NeuralOperatorBackend> backend)
        : mBackend(std::move(backend))
    {
    }

    NeuralOperatorOutput solve(
        const float* branch_input,
        int32_t branch_len,
        const float* trunk_input,
        int32_t trunk_len) override
    {
        if (mBackend == nullptr)
        {
            return {};
        }
        const auto result = mBackend->solve(branch_input, branch_len, trunk_input, trunk_len);
        return {result.output, result.output_dim, result.out_channels, result.height, result.width};
    }

    NeuralOperatorOutput solve_field(const float* field_input, int32_t input_size) override
    {
        if (mBackend == nullptr)
        {
            return {};
        }
        const auto result = mBackend->solve_field(field_input, input_size);
        return {result.output, result.output_dim, result.out_channels, result.height, result.width};
    }

private:
    std::unique_ptr<trtf::NeuralOperatorBackend> mBackend;
};

} // namespace

GenerationBackendPort::GenerationBackendPort(std::unique_ptr<trtf::IGenerationBackend> backend)
    : GenerationBackendPort(std::make_unique<GenerationBackendAdapter>(std::move(backend)))
{
}

GenerationBackendPort::GenerationBackendPort(std::unique_ptr<ITextGenerationBackendAdapter> backend)
    : mBackend(std::move(backend))
{
}

std::vector<int32_t> GenerationBackendPort::generate_text(
    const std::vector<int32_t>& input_ids,
    const trtf::GenerationConfig& config)
{
    if (mBackend == nullptr)
    {
        return {};
    }
    return mBackend->generate_text(input_ids, config);
}

bool GenerationBackendPort::supports_vision() const
{
    return mBackend != nullptr && mBackend->supports_vision();
}

bool GenerationBackendPort::prepare_image(
    const adapters::io::DecodedImage& image,
    std::vector<float>& image_features,
    int32_t& num_features,
    int32_t& feature_dim,
    std::string& error)
{
    if (mBackend == nullptr)
    {
        return false;
    }
    return mBackend->prepare_image(image, image_features, num_features, feature_dim, error);
}

std::string GenerationBackendPort::format_vision_prompt(const std::string& prompt) const
{
    return mBackend == nullptr ? prompt : mBackend->format_vision_prompt(prompt);
}

std::vector<int32_t> GenerationBackendPort::generate_with_image(
    const std::vector<int32_t>& input_ids,
    const float* image_features,
    int32_t num_features,
    int32_t feature_dim,
    const trtf::GenerationConfig& config)
{
    if (mBackend == nullptr)
    {
        return {};
    }
    return mBackend->generate_with_image(input_ids, image_features, num_features, feature_dim, config);
}

OmniTextGenerationPort::OmniTextGenerationPort(std::shared_ptr<trtf::OmniBackend> backend)
    : OmniTextGenerationPort(std::make_unique<OmniTextGenerationBackendAdapter>(std::move(backend)))
{
}

OmniTextGenerationPort::OmniTextGenerationPort(std::unique_ptr<ITextGenerationBackendAdapter> backend)
    : mBackend(std::move(backend))
{
}

std::vector<int32_t> OmniTextGenerationPort::generate_text(
    const std::vector<int32_t>& input_ids,
    const trtf::GenerationConfig& config)
{
    if (mBackend == nullptr)
    {
        return {};
    }
    return mBackend->generate_text(input_ids, config);
}

BarkAudioGenerationPort::BarkAudioGenerationPort(std::unique_ptr<trtf::BarkBackend> backend)
    : BarkAudioGenerationPort(std::make_unique<BarkAudioGenerationBackendAdapter>(std::move(backend)))
{
}

BarkAudioGenerationPort::BarkAudioGenerationPort(std::unique_ptr<IAudioGenerationBackendAdapter> backend)
    : mBackend(std::move(backend))
{
}

BarkAudioGenerationPort::~BarkAudioGenerationPort() = default;

adapters::io::AudioArtifact BarkAudioGenerationPort::generate_audio(
    const std::vector<int32_t>& input_ids,
    int32_t max_tokens)
{
    if (mBackend == nullptr)
    {
        return {};
    }
    return mBackend->generate_audio(input_ids, max_tokens);
}

MagpieAudioGenerationPort::MagpieAudioGenerationPort(std::unique_ptr<trtf::MagpieTTSBackend> backend)
    : MagpieAudioGenerationPort(std::make_unique<MagpieAudioGenerationBackendAdapter>(std::move(backend)))
{
}

MagpieAudioGenerationPort::MagpieAudioGenerationPort(std::unique_ptr<IAudioGenerationBackendAdapter> backend)
    : mBackend(std::move(backend))
{
}

MagpieAudioGenerationPort::~MagpieAudioGenerationPort() = default;

adapters::io::AudioArtifact MagpieAudioGenerationPort::generate_audio(
    const std::vector<int32_t>& input_ids,
    int32_t max_tokens)
{
    if (mBackend == nullptr)
    {
        return {};
    }
    return mBackend->generate_audio(input_ids, max_tokens);
}

OmniAudioGenerationPort::OmniAudioGenerationPort(std::shared_ptr<trtf::OmniBackend> backend)
    : OmniAudioGenerationPort(std::make_unique<OmniAudioGenerationBackendAdapter>(std::move(backend)))
{
}

OmniAudioGenerationPort::OmniAudioGenerationPort(std::unique_ptr<IAudioGenerationBackendAdapter> backend)
    : mBackend(std::move(backend))
{
}

adapters::io::AudioArtifact OmniAudioGenerationPort::generate_audio(
    const std::vector<int32_t>& input_ids,
    int32_t max_tokens)
{
    if (mBackend == nullptr)
    {
        return {};
    }
    return mBackend->generate_audio(input_ids, max_tokens);
}

WhisperTranscriptionPort::WhisperTranscriptionPort(std::unique_ptr<trtf::WhisperBackend> backend)
    : WhisperTranscriptionPort(std::make_unique<WhisperTranscriptionBackendAdapter>(std::move(backend)))
{
}

WhisperTranscriptionPort::WhisperTranscriptionPort(std::unique_ptr<ITranscriptionBackendAdapter> backend)
    : mBackend(std::move(backend))
{
}

WhisperTranscriptionPort::~WhisperTranscriptionPort() = default;

TranscriptionOutput WhisperTranscriptionPort::transcribe(
    const float* mel_data,
    int32_t mel_bins,
    int32_t mel_length,
    int32_t max_new_tokens)
{
    if (mBackend == nullptr)
    {
        return {};
    }
    return mBackend->transcribe(mel_data, mel_bins, mel_length, max_new_tokens);
}

SpeechSynthesisPort::SpeechSynthesisPort(std::unique_ptr<trtf::SpeechToSpeechBackend> backend)
    : SpeechSynthesisPort(std::make_unique<SpeechSynthesisBackendAdapter>(std::move(backend)))
{
}

SpeechSynthesisPort::SpeechSynthesisPort(std::unique_ptr<ISpeechSynthesisBackendAdapter> backend)
    : mBackend(std::move(backend))
{
}

SpeechSynthesisPort::~SpeechSynthesisPort() = default;

SpeechInputPolicy SpeechSynthesisPort::input_policy() const
{
    if (mBackend == nullptr)
    {
        return {};
    }
    return mBackend->input_policy();
}

adapters::io::AudioArtifact SpeechSynthesisPort::process_audio(
    const float* input_samples,
    int32_t num_input_samples,
    int32_t max_output_frames,
    int32_t input_sample_rate,
    int32_t tail_frames)
{
    if (mBackend == nullptr)
    {
        return {};
    }
    return mBackend->process_audio(input_samples, num_input_samples, max_output_frames, input_sample_rate, tail_frames);
}

SegmentationBackendPort::SegmentationBackendPort(std::unique_ptr<trtf::SegmentationBackend> backend)
    : SegmentationBackendPort(std::make_unique<SegmentationBackendAdapter>(std::move(backend)))
{
}

SegmentationBackendPort::SegmentationBackendPort(std::unique_ptr<ISegmentationBackendAdapter> backend)
    : mBackend(std::move(backend))
{
}

SegmentationBackendPort::~SegmentationBackendPort() = default;

adapters::io::SegmentationArtifact SegmentationBackendPort::segment_image(const adapters::io::DecodedImage& image)
{
    if (mBackend == nullptr)
    {
        return {};
    }
    return mBackend->segment_image(image);
}

SamBackendPort::SamBackendPort(std::unique_ptr<trtf::SamBackend> backend)
    : SamBackendPort(std::make_unique<SamBackendAdapter>(std::move(backend)))
{
}

SamBackendPort::SamBackendPort(std::unique_ptr<IPromptedSegmentationBackendAdapter> backend)
    : mBackend(std::move(backend))
{
}

SamBackendPort::~SamBackendPort() = default;

bool SamBackendPort::encode_image(const adapters::io::DecodedImage& image)
{
    return mBackend != nullptr && mBackend->encode_image(image);
}

adapters::io::SamMaskArtifact SamBackendPort::segment_point(float point_x, float point_y, bool is_foreground)
{
    if (mBackend == nullptr)
    {
        return {};
    }
    return mBackend->segment_point(point_x, point_y, is_foreground);
}

DetectionBackendPort::DetectionBackendPort(std::unique_ptr<trtf::DetectionBackend> backend)
    : DetectionBackendPort(std::make_unique<DetectionBackendAdapter>(std::move(backend)))
{
}

DetectionBackendPort::DetectionBackendPort(std::unique_ptr<IDetectionBackendAdapter> backend)
    : mBackend(std::move(backend))
{
}

DetectionBackendPort::~DetectionBackendPort() = default;

adapters::io::DetectionArtifact DetectionBackendPort::detect_image(const adapters::io::DecodedImage& image)
{
    if (mBackend == nullptr)
    {
        return {};
    }
    return mBackend->detect_image(image);
}

NeuralOperatorBackendPort::NeuralOperatorBackendPort(std::unique_ptr<trtf::NeuralOperatorBackend> backend)
    : NeuralOperatorBackendPort(std::make_unique<NeuralOperatorBackendAdapter>(std::move(backend)))
{
}

NeuralOperatorBackendPort::NeuralOperatorBackendPort(std::unique_ptr<INeuralOperatorBackendAdapter> backend)
    : mBackend(std::move(backend))
{
}

NeuralOperatorBackendPort::~NeuralOperatorBackendPort() = default;

NeuralOperatorOutput NeuralOperatorBackendPort::solve(
    const float* branch_input,
    int32_t branch_len,
    const float* trunk_input,
    int32_t trunk_len)
{
    if (mBackend == nullptr)
    {
        return {};
    }
    return mBackend->solve(branch_input, branch_len, trunk_input, trunk_len);
}

NeuralOperatorOutput NeuralOperatorBackendPort::solve_field(const float* field_input, int32_t input_size)
{
    if (mBackend == nullptr)
    {
        return {};
    }
    return mBackend->solve_field(field_input, input_size);
}

std::vector<float> DefaultAudioResampler::resample(
    const float* samples,
    int32_t n_samples,
    int32_t source_rate,
    int32_t target_rate) const
{
    return trtf::resample_linear(samples, n_samples, source_rate, target_rate);
}

trtf::MelResult DefaultMelSpectrogramPort::extract(
    const float* samples,
    int32_t n_samples,
    const MelSpectrogramConfig& config) const
{
    return trtf::extract_mel_spectrogram(
        samples,
        n_samples,
        config.filterbank.data(),
        config.n_freq_bins,
        config.n_mel_bins,
        config.n_fft,
        config.hop_length,
        config.chunk_length_s,
        config.sample_rate);
}

} // namespace trtf::runtime::services::common
