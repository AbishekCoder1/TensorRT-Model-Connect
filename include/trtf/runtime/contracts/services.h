#pragma once

#include "trtf/runtime/adapters/io/audio_io_adapter.h"
#include "trtf/runtime/adapters/io/detection_io_adapter.h"
#include "trtf/runtime/adapters/io/image_io_adapter.h"
#include "trtf/runtime/adapters/io/media_io_adapter.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace trtf::runtime {

enum class RuntimeServiceStatus {
    kOk,
    kInvalidArgument,
    kUnsupported,
    kRuntimeError,
};

template <typename Payload>
struct ServiceResult {
    RuntimeServiceStatus status{RuntimeServiceStatus::kOk};
    Payload value{};
    std::string message;

    [[nodiscard]] bool ok() const
    {
        return status == RuntimeServiceStatus::kOk;
    }

    static ServiceResult Success(Payload value_in)
    {
        ServiceResult result;
        result.value = std::move(value_in);
        return result;
    }

    static ServiceResult Failure(RuntimeServiceStatus status_in, std::string message_in)
    {
        ServiceResult result;
        result.status = status_in;
        result.message = std::move(message_in);
        return result;
    }
};

struct ImageInput {
    adapters::io::DecodedImage decoded;

    [[nodiscard]] bool empty() const
    {
        return decoded.empty();
    }
};

struct AudioInput {
    adapters::io::DecodedAudio decoded;

    [[nodiscard]] bool empty() const
    {
        return decoded.empty();
    }
};

struct VideoGenerationRequest {
    std::string prompt;
    int32_t num_steps{-1};
    float guidance_scale{-1.0F};
};

using VideoGenerationResult = ServiceResult<adapters::io::VideoFrameArtifact>;

struct AudioGenerationRequest {
    std::string prompt;
    int32_t max_tokens{-1};
};

using AudioGenerationResult = ServiceResult<adapters::io::AudioArtifact>;

struct SpeechSynthesisRequest {
    AudioInput input;
    int32_t max_output_frames{-1};
    int32_t tail_frames{0};
};

using SpeechSynthesisResult = ServiceResult<adapters::io::AudioArtifact>;

struct TranscriptionRequest {
    AudioInput input;
    int32_t max_new_tokens{224};
};

using TranscriptionResult = ServiceResult<std::string>;

struct SegmentationRequest {
    ImageInput image;
};

using SegmentationResult = ServiceResult<adapters::io::SegmentationArtifact>;

struct PromptedSegmentationRequest {
    ImageInput image;
    float point_x{0.0F};
    float point_y{0.0F};
    bool is_foreground{true};
};

using PromptedSegmentationResult = ServiceResult<adapters::io::SamMaskArtifact>;

struct DetectionRequest {
    ImageInput image;
    float conf_threshold{-1.0F};
};

using DetectionResult = ServiceResult<adapters::io::DetectionArtifact>;

class ITextService {
public:
    virtual ~ITextService() = default;

    virtual const char* generate(const char* prompt, std::size_t max_new_tokens) = 0;

    virtual const char* generate(
        const char* prompt,
        const adapters::io::DecodedImage& image,
        std::size_t max_new_tokens)
    {
        (void) image;
        return generate(prompt, max_new_tokens);
    }

    virtual bool supports_vision() const { return false; }

    virtual bool supports_encoding() const { return false; }

    virtual const float* encode(const char* text, int32_t* out_seq_len, int32_t* out_hidden_size)
    {
        (void) text;
        (void) out_seq_len;
        (void) out_hidden_size;
        return nullptr;
    }
};

class IVideoService {
public:
    virtual ~IVideoService() = default;

    virtual VideoGenerationResult generate_video(const VideoGenerationRequest& request) = 0;
};

class IAudioService {
public:
    virtual ~IAudioService() = default;

    virtual AudioGenerationResult generate_audio(const AudioGenerationRequest& request) = 0;

    virtual bool supports_speech() const { return false; }

    virtual SpeechSynthesisResult speak(const SpeechSynthesisRequest& request)
    {
        (void) request;
        return SpeechSynthesisResult::Failure(RuntimeServiceStatus::kUnsupported, "speech not supported");
    }
};

class ITranscriptionService {
public:
    virtual ~ITranscriptionService() = default;

    virtual TranscriptionResult transcribe(const TranscriptionRequest& request) = 0;
};

class IEmbeddingService {
public:
    virtual ~IEmbeddingService() = default;

    virtual const float* embed(const char* text, int32_t* out_dim) = 0;

    virtual const float* embed_image(const adapters::io::DecodedImage& image, int32_t* out_dim)
    {
        (void) image;
        (void) out_dim;
        return nullptr;
    }

    virtual const float* embed_image_text(
        const char* text,
        const adapters::io::DecodedImage& image,
        int32_t* out_dim)
    {
        (void) text;
        (void) image;
        (void) out_dim;
        return nullptr;
    }
};

class IRerankService {
public:
    virtual ~IRerankService() = default;

    virtual float rerank(const char* query, const char* document) = 0;
};

class ISegmentationService {
public:
    virtual ~ISegmentationService() = default;

    virtual SegmentationResult segment(const SegmentationRequest& request) = 0;

    virtual bool supports_prompted() const { return false; }

    virtual PromptedSegmentationResult segment_prompt(const PromptedSegmentationRequest& request)
    {
        (void) request;
        return PromptedSegmentationResult::Failure(
            RuntimeServiceStatus::kUnsupported, "prompted segmentation not supported");
    }
};

class IDetectionService {
public:
    virtual ~IDetectionService() = default;

    virtual DetectionResult detect(const DetectionRequest& request) = 0;
};

class ISolveService {
public:
    virtual ~ISolveService() = default;

    virtual const float* solve(const float* branch_input, int32_t branch_len, const float* trunk_input,
        int32_t trunk_len, int32_t* out_dim)
        = 0;

    virtual const float* solve_field(
        const float* field_input, int32_t input_size, int32_t* out_channels, int32_t* out_h, int32_t* out_w)
    {
        (void) field_input;
        (void) input_size;
        (void) out_channels;
        (void) out_h;
        (void) out_w;
        return nullptr;
    }
};

struct PipelineServices {
    std::unique_ptr<ITextService> text;
    std::unique_ptr<IVideoService> video;
    std::unique_ptr<IAudioService> audio;
    std::unique_ptr<ITranscriptionService> transcription;
    std::unique_ptr<IEmbeddingService> embedding;
    std::unique_ptr<IRerankService> rerank;
    std::unique_ptr<ISegmentationService> segmentation;
    std::unique_ptr<IDetectionService> detection;
    std::unique_ptr<ISolveService> solve;

    [[nodiscard]] bool empty() const
    {
        return !text && !video && !audio && !transcription && !embedding && !rerank && !segmentation
            && !detection && !solve;
    }
};

} // namespace trtf::runtime
