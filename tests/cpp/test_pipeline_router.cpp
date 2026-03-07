#include "trtf/runtime/pipeline/router.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using trtf::runtime::AudioGenerationRequest;
using trtf::runtime::AudioGenerationResult;
using trtf::runtime::DetectionRequest;
using trtf::runtime::DetectionResult;
using trtf::runtime::IAudioService;
using trtf::runtime::IDetectionService;
using trtf::runtime::IEmbeddingService;
using trtf::runtime::IRerankService;
using trtf::runtime::ISegmentationService;
using trtf::runtime::ISolveService;
using trtf::runtime::ITextService;
using trtf::runtime::ITranscriptionService;
using trtf::runtime::IVideoService;
using trtf::runtime::PipelineRouter;
using trtf::runtime::PipelineRouterDefaults;
using trtf::runtime::PipelineRouterIoAdapters;
using trtf::runtime::PipelineServices;
using trtf::runtime::PromptedSegmentationRequest;
using trtf::runtime::PromptedSegmentationResult;
using trtf::runtime::RuntimeServiceStatus;
using trtf::runtime::SegmentationRequest;
using trtf::runtime::SegmentationResult;
using trtf::runtime::SpeechSynthesisRequest;
using trtf::runtime::SpeechSynthesisResult;
using trtf::runtime::TranscriptionRequest;
using trtf::runtime::TranscriptionResult;
using trtf::runtime::VideoGenerationRequest;
using trtf::runtime::VideoGenerationResult;
using trtf::runtime::adapters::io::AudioLoadResult;
using trtf::runtime::adapters::io::DecodedAudio;
using trtf::runtime::adapters::io::DecodedImage;
using trtf::runtime::adapters::io::ImageLoadResult;

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

DecodedImage make_image()
{
    return {{255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255}, 2, 2, 3};
}

DecodedAudio make_audio()
{
    return {{0.1F, 0.2F, 0.3F, 0.4F}, 16000};
}

class FakeImageLoader final : public trtf::runtime::adapters::io::IImageInputLoader {
public:
    ImageLoadResult load(const char* path) const override
    {
        ++calls;
        last_path = path != nullptr ? path : "<null>";
        return next_result;
    }

    mutable int calls{0};
    mutable std::string last_path;
    ImageLoadResult next_result = ImageLoadResult::Success(make_image());
};

class FakeAudioLoader final : public trtf::runtime::adapters::io::IAudioInputLoader {
public:
    AudioLoadResult load(const char* path) const override
    {
        ++calls;
        last_path = path != nullptr ? path : "<null>";
        return next_result;
    }

    mutable int calls{0};
    mutable std::string last_path;
    AudioLoadResult next_result = AudioLoadResult::Success(make_audio());
};

class FakeTextService final : public ITextService {
public:
    const char* generate(const char* prompt, std::size_t max_new_tokens) override
    {
        ++generate_calls;
        last_prompt = prompt != nullptr ? prompt : "<null>";
        last_max_new_tokens = max_new_tokens;
        return "text-generate";
    }

    const char* generate(
        const char* prompt, const DecodedImage& image, std::size_t max_new_tokens) override
    {
        ++generate_with_image_calls;
        last_prompt = prompt != nullptr ? prompt : "<null>";
        last_image_width = image.width;
        last_image_height = image.height;
        last_max_new_tokens = max_new_tokens;
        return "text-generate-image";
    }

    bool supports_vision() const override { return vision; }
    bool supports_encoding() const override { return encoding; }

    const float* encode(const char* text, int32_t* out_seq_len, int32_t* out_hidden_size) override
    {
        ++encode_calls;
        last_prompt = text != nullptr ? text : "<null>";
        if (out_seq_len != nullptr)
        {
            *out_seq_len = 3;
        }
        if (out_hidden_size != nullptr)
        {
            *out_hidden_size = 2;
        }
        return encode_values.data();
    }

    bool vision{false};
    bool encoding{false};
    int generate_calls{0};
    int generate_with_image_calls{0};
    int encode_calls{0};
    std::size_t last_max_new_tokens{0};
    int32_t last_image_width{0};
    int32_t last_image_height{0};
    std::string last_prompt;
    std::array<float, 6> encode_values{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
};

class FakeVideoService final : public IVideoService {
public:
    VideoGenerationResult generate_video(const VideoGenerationRequest& request) override
    {
        ++calls;
        last_prompt = request.prompt;
        last_num_steps = request.num_steps;
        last_guidance_scale = request.guidance_scale;
        return VideoGenerationResult::Success(next_artifact);
    }

    int calls{0};
    int32_t last_num_steps{0};
    float last_guidance_scale{0.0F};
    std::string last_prompt;
    trtf::runtime::adapters::io::VideoFrameArtifact next_artifact{{0.1F, 0.2F, 0.3F}, 7, 1, 1};
};

class FakeVideoWriter final : public trtf::runtime::adapters::io::IVideoFrameArtifactWriter {
public:
    trtf::runtime::adapters::io::ImageWriteResult write_png_frames(
        const trtf::runtime::adapters::io::VideoFrameArtifact& artifact,
        const char* output_dir) const override
    {
        ++calls;
        last_output_dir = output_dir != nullptr ? output_dir : "<null>";
        last_artifact = artifact;
        return next_result;
    }

    mutable int calls{0};
    mutable std::string last_output_dir;
    mutable trtf::runtime::adapters::io::VideoFrameArtifact last_artifact;
    trtf::runtime::adapters::io::ImageWriteResult next_result{
        trtf::runtime::adapters::io::ImageIoStatus::kOk, 7, {}};
};

class FakeAudioService final : public IAudioService {
public:
    AudioGenerationResult generate_audio(const AudioGenerationRequest& request) override
    {
        ++generate_audio_calls;
        last_prompt = request.prompt;
        last_max_tokens = request.max_tokens;
        return AudioGenerationResult::Success(next_audio);
    }

    bool supports_speech() const override { return speech; }

    SpeechSynthesisResult speak(const SpeechSynthesisRequest& request) override
    {
        ++speak_calls;
        last_audio_sample_rate = request.input.decoded.sample_rate;
        last_audio_num_samples = static_cast<int32_t>(request.input.decoded.samples.size());
        last_max_output_frames = request.max_output_frames;
        last_tail_frames = request.tail_frames;
        if (!speech)
        {
            return SpeechSynthesisResult::Failure(RuntimeServiceStatus::kUnsupported, "speech disabled");
        }
        return SpeechSynthesisResult::Success(next_speech_audio);
    }

    bool speech{false};
    int generate_audio_calls{0};
    int speak_calls{0};
    int32_t last_max_tokens{0};
    int32_t last_max_output_frames{0};
    int32_t last_tail_frames{0};
    int32_t last_audio_sample_rate{0};
    int32_t last_audio_num_samples{0};
    std::string last_prompt;
    trtf::runtime::adapters::io::AudioArtifact next_audio{{0.1F, 0.2F, 0.3F}, 24000, 3};
    trtf::runtime::adapters::io::AudioArtifact next_speech_audio{{0.5F, 0.6F}, 16000, 2};
};

class FakeAudioWriter final : public trtf::runtime::adapters::io::IAudioArtifactWriter {
public:
    trtf::runtime::adapters::io::AudioWriteResult write_wav(
        const trtf::runtime::adapters::io::AudioArtifact& artifact,
        const char* output_path) const override
    {
        ++calls;
        last_output_path = output_path != nullptr ? output_path : "<null>";
        last_artifact = artifact;
        return next_result;
    }

    mutable int calls{0};
    mutable std::string last_output_path;
    mutable trtf::runtime::adapters::io::AudioArtifact last_artifact;
    trtf::runtime::adapters::io::AudioWriteResult next_result{
        trtf::runtime::adapters::io::AudioIoStatus::kOk, 3, {}};
};

class FakeTranscriptionService final : public ITranscriptionService {
public:
    TranscriptionResult transcribe(const TranscriptionRequest& request) override
    {
        ++calls;
        last_audio_sample_rate = request.input.decoded.sample_rate;
        last_audio_num_samples = static_cast<int32_t>(request.input.decoded.samples.size());
        last_max_new_tokens = request.max_new_tokens;
        return TranscriptionResult::Success(result);
    }

    int calls{0};
    int32_t last_max_new_tokens{0};
    int32_t last_audio_sample_rate{0};
    int32_t last_audio_num_samples{0};
    std::string result{"transcribed-output"};
};

class FakeEmbeddingService final : public IEmbeddingService {
public:
    const float* embed(const char* text, int32_t* out_dim) override
    {
        ++embed_calls;
        last_text = text != nullptr ? text : "<null>";
        if (out_dim != nullptr)
        {
            *out_dim = static_cast<int32_t>(text_values.size());
        }
        return text_values.data();
    }

    const float* embed_image(const DecodedImage& image, int32_t* out_dim) override
    {
        ++embed_image_calls;
        last_image_width = image.width;
        last_image_height = image.height;
        if (out_dim != nullptr)
        {
            *out_dim = static_cast<int32_t>(image_values.size());
        }
        return image_values.data();
    }

    const float* embed_image_text(const char* text, const DecodedImage& image, int32_t* out_dim) override
    {
        ++embed_image_text_calls;
        last_text = text != nullptr ? text : "<null>";
        last_image_width = image.width;
        last_image_height = image.height;
        if (out_dim != nullptr)
        {
            *out_dim = static_cast<int32_t>(image_text_values.size());
        }
        return image_text_values.data();
    }

    int embed_calls{0};
    int embed_image_calls{0};
    int embed_image_text_calls{0};
    int32_t last_image_width{0};
    int32_t last_image_height{0};
    std::string last_text;
    std::array<float, 3> text_values{0.1F, 0.2F, 0.3F};
    std::array<float, 2> image_values{1.1F, 1.2F};
    std::array<float, 4> image_text_values{2.1F, 2.2F, 2.3F, 2.4F};
};

class FakeRerankService final : public IRerankService {
public:
    float rerank(const char* query, const char* document) override
    {
        ++calls;
        last_query = query != nullptr ? query : "<null>";
        last_document = document != nullptr ? document : "<null>";
        return score;
    }

    int calls{0};
    float score{0.875F};
    std::string last_query;
    std::string last_document;
};

class FakeSegmentationService final : public ISegmentationService {
public:
    SegmentationResult segment(const SegmentationRequest& request) override
    {
        ++segment_calls;
        last_image_width = request.image.decoded.width;
        last_image_height = request.image.decoded.height;
        return SegmentationResult::Success(next_segmentation);
    }

    bool supports_prompted() const override
    {
        return prompted_segmentation;
    }

    PromptedSegmentationResult segment_prompt(const PromptedSegmentationRequest& request) override
    {
        ++segment_sam_calls;
        last_image_width = request.image.decoded.width;
        last_image_height = request.image.decoded.height;
        last_point_x = request.point_x;
        last_point_y = request.point_y;
        last_is_foreground = request.is_foreground;
        if (!prompted_segmentation)
        {
            return PromptedSegmentationResult::Failure(RuntimeServiceStatus::kUnsupported, "prompted disabled");
        }
        return PromptedSegmentationResult::Success(next_sam_masks);
    }

    bool prompted_segmentation{false};
    int segment_calls{0};
    int segment_sam_calls{0};
    float last_point_x{0.0F};
    float last_point_y{0.0F};
    bool last_is_foreground{false};
    int32_t last_image_width{0};
    int32_t last_image_height{0};
    trtf::runtime::adapters::io::SegmentationArtifact next_segmentation{{1, 2, 3, 4}, 2, 2};
    trtf::runtime::adapters::io::SamMaskArtifact next_sam_masks{{1.0F, 0.0F, 0.0F, 1.0F}, {0.9F}, 2, 2, 2};
};

class FakeSegmentationWriter final : public trtf::runtime::adapters::io::ISegmentationArtifactWriter {
public:
    trtf::runtime::adapters::io::ImageWriteResult write_png(
        const trtf::runtime::adapters::io::SegmentationArtifact& artifact,
        const char* output_path) const override
    {
        ++calls;
        last_output_path = output_path != nullptr ? output_path : "<null>";
        last_artifact = artifact;
        return next_result;
    }

    mutable int calls{0};
    mutable std::string last_output_path;
    mutable trtf::runtime::adapters::io::SegmentationArtifact last_artifact;
    trtf::runtime::adapters::io::ImageWriteResult next_result{
        trtf::runtime::adapters::io::ImageIoStatus::kOk, 1, {}};
};

class FakeSamWriter final : public trtf::runtime::adapters::io::ISamMaskArtifactWriter {
public:
    trtf::runtime::adapters::io::ImageWriteResult write_masks(
        const trtf::runtime::adapters::io::SamMaskArtifact& artifact,
        const char* output_dir) const override
    {
        ++calls;
        last_output_dir = output_dir != nullptr ? output_dir : "<null>";
        last_artifact = artifact;
        return next_result;
    }

    mutable int calls{0};
    mutable std::string last_output_dir;
    mutable trtf::runtime::adapters::io::SamMaskArtifact last_artifact;
    trtf::runtime::adapters::io::ImageWriteResult next_result{
        trtf::runtime::adapters::io::ImageIoStatus::kOk, 2, {}};
};

class FakeDetectionService final : public IDetectionService {
public:
    DetectionResult detect(const DetectionRequest& request) override
    {
        ++calls;
        last_image_width = request.image.decoded.width;
        last_image_height = request.image.decoded.height;
        last_conf_threshold = request.conf_threshold;
        return DetectionResult::Success(next_artifact);
    }

    int calls{0};
    float last_conf_threshold{0.0F};
    int32_t last_image_width{0};
    int32_t last_image_height{0};
    trtf::runtime::adapters::io::DetectionArtifact next_artifact{{{1, 0.95F, 0.0F, 1.0F, 2.0F, 3.0F}}};
};

class FakeDetectionWriter final : public trtf::runtime::adapters::io::IDetectionArtifactWriter {
public:
    trtf::runtime::adapters::io::DetectionWriteResult write_json(
        const trtf::runtime::adapters::io::DetectionArtifact& artifact,
        const char* output_path,
        float conf_threshold) const override
    {
        ++calls;
        last_output_path = output_path != nullptr ? output_path : "<null>";
        last_conf_threshold = conf_threshold;
        last_artifact = artifact;
        return next_result;
    }

    mutable int calls{0};
    mutable float last_conf_threshold{0.0F};
    mutable std::string last_output_path;
    mutable trtf::runtime::adapters::io::DetectionArtifact last_artifact;
    trtf::runtime::adapters::io::DetectionWriteResult next_result{
        trtf::runtime::adapters::io::DetectionIoStatus::kOk, 5, {}};
};

class FakeSolveService final : public ISolveService {
public:
    const float* solve(const float* branch_input, int32_t branch_len,
        const float* trunk_input, int32_t trunk_len, int32_t* out_dim) override
    {
        ++solve_calls;
        saw_branch = branch_input != nullptr;
        saw_trunk = trunk_input != nullptr;
        last_branch_len = branch_len;
        last_trunk_len = trunk_len;
        if (out_dim != nullptr)
        {
            *out_dim = static_cast<int32_t>(solve_values.size());
        }
        return solve_values.data();
    }

    const float* solve_field(const float* field_input, int32_t input_size,
        int32_t* out_channels, int32_t* out_h, int32_t* out_w) override
    {
        ++solve_field_calls;
        saw_field = field_input != nullptr;
        last_input_size = input_size;
        if (out_channels != nullptr)
        {
            *out_channels = 1;
        }
        if (out_h != nullptr)
        {
            *out_h = 2;
        }
        if (out_w != nullptr)
        {
            *out_w = 2;
        }
        return solve_field_values.data();
    }

    int solve_calls{0};
    int solve_field_calls{0};
    bool saw_branch{false};
    bool saw_trunk{false};
    bool saw_field{false};
    int32_t last_branch_len{0};
    int32_t last_trunk_len{0};
    int32_t last_input_size{0};
    std::array<float, 3> solve_values{9.0F, 8.0F, 7.0F};
    std::array<float, 4> solve_field_values{6.0F, 5.0F, 4.0F, 3.0F};
};

void test_unsupported_defaults()
{
    PipelineRouter router(PipelineServices{}, "model-empty", "backend-empty");

    check(std::strcmp(router.model_id(), "model-empty") == 0, "model_id preserved");
    check(std::strcmp(router.backend_name(), "backend-empty") == 0, "backend_name preserved");

    const char* generated = router.generate("hello", 8);
    check(generated != nullptr && std::strlen(generated) == 0, "generate fallback empty string");
    const char* generated_image = router.generate("hello", "image.png", 8);
    check(generated_image != nullptr && std::strlen(generated_image) == 0,
        "generate with image fallback empty string");

    check(!router.supports_vision(), "supports_vision false without text service");
    check(!router.supports_video(), "supports_video false without video service");
    check(router.generate_video("p", "/tmp/out", -1, -1.0F) == -1, "generate_video fallback -1");
    check(!router.supports_segmentation(), "supports_segmentation false");
    check(router.segment("img.png", "out.png") == -1, "segment fallback -1");
    check(!router.supports_encoding(), "supports_encoding false");
    check(router.encode("hello", nullptr, nullptr) == nullptr, "encode fallback nullptr");
    check(!router.supports_solve(), "supports_solve false");
    check(router.solve(nullptr, 0, nullptr, 0, nullptr) == nullptr, "solve fallback nullptr");
    check(router.solve_field(nullptr, 0, nullptr, nullptr, nullptr) == nullptr,
        "solve_field fallback nullptr");
    check(!router.supports_detection(), "supports_detection false");
    check(router.detect("img.png", "det.json", 0.5F) == -1, "detect fallback -1");
    check(!router.supports_transcription(), "supports_transcription false");
    check(router.transcribe("audio.wav", 32) == nullptr, "transcribe fallback nullptr");
    check(!router.supports_audio(), "supports_audio false");
    check(router.generate_audio("prompt", "out.wav", 16) == -1, "generate_audio fallback -1");
    check(!router.supports_embedding(), "supports_embedding false");
    check(router.embed("text", nullptr) == nullptr, "embed fallback nullptr");
    check(router.embed_image("img.png", nullptr) == nullptr, "embed_image fallback nullptr");
    check(router.embed_image_text("text", "img.png", nullptr) == nullptr,
        "embed_image_text fallback nullptr");
    check(!router.supports_reranking(), "supports_reranking false");
    check(router.rerank("q", "d") == 0.0F, "rerank fallback 0.0");
    check(!router.supports_prompted_segmentation(), "supports_prompted_segmentation false");
    check(router.segment_sam("img.png", "/tmp/out", 0.5F, 0.6F, true) == -1,
        "segment_sam fallback -1");
    check(!router.supports_speech(), "supports_speech false");
    check(router.speak("in.wav", "out.wav", 1, 2) == -1, "speak fallback -1");
}

void test_text_service_delegation_and_gates()
{
    PipelineServices services;
    auto text = std::make_unique<FakeTextService>();
    auto* text_ptr = text.get();
    services.text = std::move(text);

    PipelineRouterDefaults defaults;
    defaults.max_new_tokens = 42;
    auto image_loader = std::make_unique<FakeImageLoader>();
    auto* image_loader_ptr = image_loader.get();
    PipelineRouterIoAdapters io;
    io.image_loader = std::move(image_loader);
    PipelineRouter router(std::move(services), "text-model", "text-backend", defaults, std::move(io));

    const char* generated = router.generate("hello", 0);
    check(std::strcmp(generated, "text-generate") == 0, "generate delegates to text service");
    check(text_ptr->generate_calls == 1, "generate call count");
    check(text_ptr->last_max_new_tokens == 42, "generate resolves default max_new_tokens");

    router.generate("hello", "img.png", 0);
    check(text_ptr->generate_calls == 2, "generate(image) falls back when no vision support");
    check(text_ptr->generate_with_image_calls == 0, "generate_with_image not called without vision");
    check(!router.supports_vision(), "supports_vision false when text service says false");

    text_ptr->vision = true;
    check(router.supports_vision(), "supports_vision true when text service says true");
    const char* generated_image = router.generate("hello", "img.png", 0);
    check(std::strcmp(generated_image, "text-generate-image") == 0, "generate_with_image delegates");
    check(image_loader_ptr->calls == 1, "generate_with_image loads image");
    check(image_loader_ptr->last_path == "img.png", "generate_with_image forwards image path to loader");
    check(text_ptr->generate_with_image_calls == 1, "generate_with_image call count");
    check(text_ptr->last_image_width == 2 && text_ptr->last_image_height == 2,
        "generate_with_image forwards decoded image");
    check(text_ptr->last_max_new_tokens == 42, "generate_with_image resolves default max_new_tokens");

    check(!router.supports_encoding(), "supports_encoding false by default");
    text_ptr->encoding = true;
    check(router.supports_encoding(), "supports_encoding true when enabled by service");
    int32_t seq_len = 0;
    int32_t hidden_size = 0;
    const float* encoded = router.encode("encode me", &seq_len, &hidden_size);
    check(encoded == text_ptr->encode_values.data(), "encode delegates and returns service pointer");
    check(text_ptr->encode_calls == 1, "encode call count");
    check(seq_len == 3 && hidden_size == 2, "encode forwards output dims");
}

void test_video_service_delegation()
{
    PipelineServices services;
    auto video = std::make_unique<FakeVideoService>();
    auto* video_ptr = video.get();
    services.video = std::move(video);

    auto writer = std::make_unique<FakeVideoWriter>();
    auto* writer_ptr = writer.get();
    PipelineRouterIoAdapters io;
    io.video_frame_writer = std::move(writer);

    PipelineRouterDefaults defaults;
    defaults.video_num_steps = 12;
    defaults.video_guidance_scale = 7.5F;
    PipelineRouter router(std::move(services), "video-model", "video-backend", defaults, std::move(io));

    check(router.supports_video(), "supports_video true with service");
    check(router.generate_video("prompt", "/tmp/frames", -1, -1.0F) == 7,
        "generate_video writes frames count from artifact");
    check(video_ptr->calls == 1, "generate_video call count");
    check(video_ptr->last_num_steps == 12, "generate_video resolves default steps");
    check(video_ptr->last_guidance_scale == 7.5F, "generate_video resolves default guidance");
    check(writer_ptr->calls == 1, "video writer called");
    check(writer_ptr->last_output_dir == "/tmp/frames", "video writer forwards output dir");

    router.generate_video("prompt", "/tmp/frames", 3, 1.5F);
    check(video_ptr->calls == 2, "generate_video explicit call count");
    check(video_ptr->last_num_steps == 3, "generate_video forwards explicit steps");
    check(video_ptr->last_guidance_scale == 1.5F, "generate_video forwards explicit guidance");
}

void test_segmentation_service_delegation_and_prompted_gate()
{
    PipelineServices services;
    auto segmentation = std::make_unique<FakeSegmentationService>();
    auto* seg_ptr = segmentation.get();
    services.segmentation = std::move(segmentation);

    auto seg_writer = std::make_unique<FakeSegmentationWriter>();
    auto* seg_writer_ptr = seg_writer.get();
    auto sam_writer = std::make_unique<FakeSamWriter>();
    auto* sam_writer_ptr = sam_writer.get();
    auto image_loader = std::make_unique<FakeImageLoader>();
    auto* image_loader_ptr = image_loader.get();
    PipelineRouterIoAdapters io;
    io.image_loader = std::move(image_loader);
    io.segmentation_writer = std::move(seg_writer);
    io.sam_mask_writer = std::move(sam_writer);

    PipelineRouter router(std::move(services), "seg-model", "seg-backend", {}, std::move(io));

    check(router.supports_segmentation(), "supports_segmentation true with service");
    check(router.segment("img.png", "mask.png") == 0, "segment writes artifact and returns success");
    check(image_loader_ptr->calls == 1, "segment loads image");
    check(seg_ptr->segment_calls == 1, "segment call count");
    check(seg_ptr->last_image_width == 2 && seg_ptr->last_image_height == 2,
        "segment forwards decoded image");
    check(seg_writer_ptr->calls == 1, "segmentation writer called");
    check(seg_writer_ptr->last_output_path == "mask.png", "segmentation writer forwards output path");
    check(!router.supports_prompted_segmentation(), "prompted segmentation false by default");
    check(router.segment_sam("img.png", "/tmp/masks", 0.1F, 0.9F, true) == -1,
        "segment_sam gated when prompted unsupported");
    check(seg_ptr->segment_sam_calls == 0, "segment_sam not called when unsupported");

    seg_ptr->prompted_segmentation = true;
    check(router.supports_prompted_segmentation(), "prompted segmentation true when enabled");
    check(router.segment_sam("img.png", "/tmp/masks", 0.3F, 0.7F, false) == 2,
        "segment_sam writes masks and returns artifact count");
    check(image_loader_ptr->calls == 2, "segment_sam loads image");
    check(seg_ptr->segment_sam_calls == 1, "segment_sam call count");
    check(seg_ptr->last_point_x == 0.3F && seg_ptr->last_point_y == 0.7F,
        "segment_sam forwards point coordinates");
    check(!seg_ptr->last_is_foreground, "segment_sam forwards foreground flag");
    check(sam_writer_ptr->calls == 1, "sam writer called");
    check(sam_writer_ptr->last_output_dir == "/tmp/masks", "sam writer forwards output dir");
}

void test_audio_service_delegation_and_speech_gate()
{
    PipelineServices services;
    auto audio = std::make_unique<FakeAudioService>();
    auto* audio_ptr = audio.get();
    services.audio = std::move(audio);

    auto writer = std::make_unique<FakeAudioWriter>();
    auto* writer_ptr = writer.get();
    auto audio_loader = std::make_unique<FakeAudioLoader>();
    auto* audio_loader_ptr = audio_loader.get();
    PipelineRouterIoAdapters io;
    io.audio_loader = std::move(audio_loader);
    io.audio_writer = std::move(writer);

    PipelineRouterDefaults defaults;
    defaults.audio_max_tokens = 500;
    defaults.speech_max_output_frames = 375;
    defaults.speech_tail_frames = 9;
    PipelineRouter router(std::move(services), "audio-model", "audio-backend", defaults, std::move(io));

    check(router.supports_audio(), "supports_audio true with service");
    check(router.generate_audio("hello", "audio.wav", -1) == 3,
        "generate_audio writes artifact and returns sample count");
    check(audio_ptr->generate_audio_calls == 1, "generate_audio call count");
    check(audio_ptr->last_max_tokens == 500, "generate_audio resolves default max_tokens");
    check(writer_ptr->calls == 1, "audio writer called");
    check(writer_ptr->last_output_path == "audio.wav", "audio writer forwards output path");

    router.generate_audio("hello", "audio.wav", 44);
    check(audio_ptr->generate_audio_calls == 2, "generate_audio explicit call count");
    check(audio_ptr->last_max_tokens == 44, "generate_audio forwards explicit max_tokens");

    check(!router.supports_speech(), "supports_speech false when audio service disables it");
    check(router.speak("in.wav", "out.wav", 10, 2) == -1, "speak gated when service disables speech");
    check(audio_ptr->speak_calls == 0, "speak not delegated when disabled");

    audio_ptr->speech = true;
    check(router.supports_speech(), "supports_speech true when enabled");
    check(router.speak("in.wav", "out.wav", -1, -1) == 2,
        "speak writes artifact and returns sample count");
    check(audio_loader_ptr->calls == 1, "speak loads audio");
    check(audio_ptr->speak_calls == 1, "speak call count");
    check(audio_ptr->last_audio_sample_rate == 16000 && audio_ptr->last_audio_num_samples == 4,
        "speak forwards decoded audio");
    check(audio_ptr->last_max_output_frames == 375, "speak resolves default max_output_frames");
    check(audio_ptr->last_tail_frames == 9, "speak resolves default tail_frames");

    router.speak("in.wav", "out.wav", 25, 4);
    check(audio_ptr->speak_calls == 2, "speak explicit call count");
    check(audio_ptr->last_max_output_frames == 25, "speak forwards explicit max_output_frames");
    check(audio_ptr->last_tail_frames == 4, "speak forwards explicit tail_frames");
}

void test_remaining_capability_delegation()
{
    PipelineServices services;

    auto transcription = std::make_unique<FakeTranscriptionService>();
    auto* tr_ptr = transcription.get();
    services.transcription = std::move(transcription);

    auto embedding = std::make_unique<FakeEmbeddingService>();
    auto* emb_ptr = embedding.get();
    services.embedding = std::move(embedding);

    auto rerank = std::make_unique<FakeRerankService>();
    auto* rr_ptr = rerank.get();
    services.rerank = std::move(rerank);

    auto detection = std::make_unique<FakeDetectionService>();
    auto* det_ptr = detection.get();
    services.detection = std::move(detection);

    auto solve = std::make_unique<FakeSolveService>();
    auto* solve_ptr = solve.get();
    services.solve = std::move(solve);

    auto detection_writer = std::make_unique<FakeDetectionWriter>();
    auto* detection_writer_ptr = detection_writer.get();
    auto image_loader = std::make_unique<FakeImageLoader>();
    auto* image_loader_ptr = image_loader.get();
    auto audio_loader = std::make_unique<FakeAudioLoader>();
    auto* audio_loader_ptr = audio_loader.get();
    PipelineRouterIoAdapters io;
    io.image_loader = std::move(image_loader);
    io.audio_loader = std::move(audio_loader);
    io.detection_writer = std::move(detection_writer);

    PipelineRouterDefaults defaults;
    defaults.transcription_max_new_tokens = 333;
    defaults.detection_conf_threshold = 0.42F;
    PipelineRouter router(std::move(services), "multi-model", "multi-backend", defaults, std::move(io));

    check(router.supports_transcription(), "supports_transcription true");
    check(std::strcmp(router.transcribe("audio.wav", 0), tr_ptr->result.c_str()) == 0,
        "transcribe delegates");
    check(audio_loader_ptr->calls == 1, "transcribe loads audio");
    check(tr_ptr->calls == 1, "transcribe call count");
    check(tr_ptr->last_audio_sample_rate == 16000 && tr_ptr->last_audio_num_samples == 4,
        "transcribe forwards decoded audio");
    check(tr_ptr->last_max_new_tokens == 333, "transcribe resolves default max_new_tokens");

    router.transcribe("audio.wav", 77);
    check(tr_ptr->calls == 2, "transcribe explicit call count");
    check(tr_ptr->last_max_new_tokens == 77, "transcribe forwards explicit max_new_tokens");

    check(router.supports_detection(), "supports_detection true");
    check(router.detect("img.png", "det.json", -1.0F) == 5, "detect writes filtered detections");
    check(image_loader_ptr->calls == 1, "detect loads image");
    check(det_ptr->calls == 1, "detect call count");
    check(det_ptr->last_image_width == 2 && det_ptr->last_image_height == 2,
        "detect forwards decoded image");
    check(det_ptr->last_conf_threshold == 0.42F, "detect resolves default confidence threshold");
    check(detection_writer_ptr->last_conf_threshold == 0.42F, "detect writer uses resolved threshold");

    router.detect("img.png", "det.json", 0.9F);
    check(det_ptr->calls == 2, "detect explicit call count");
    check(det_ptr->last_conf_threshold == 0.9F, "detect forwards explicit confidence threshold");
    check(detection_writer_ptr->last_conf_threshold == 0.9F, "detect writer uses explicit threshold");

    check(router.supports_embedding(), "supports_embedding true");
    int32_t out_dim = 0;
    const float* embedded = router.embed("query", &out_dim);
    check(embedded == emb_ptr->text_values.data(), "embed delegates");
    check(out_dim == static_cast<int32_t>(emb_ptr->text_values.size()), "embed forwards out_dim");
    const float* embedded_image = router.embed_image("img.png", &out_dim);
    check(embedded_image == emb_ptr->image_values.data(), "embed_image delegates");
    check(emb_ptr->last_image_width == 2 && emb_ptr->last_image_height == 2,
        "embed_image forwards decoded image");
    const float* embedded_image_text = router.embed_image_text("caption", "img.png", &out_dim);
    check(embedded_image_text == emb_ptr->image_text_values.data(), "embed_image_text delegates");

    check(router.supports_reranking(), "supports_reranking true");
    check(router.rerank("query", "doc") == rr_ptr->score, "rerank delegates");
    check(rr_ptr->calls == 1, "rerank call count");

    check(router.supports_solve(), "supports_solve true");
    int32_t solve_dim = 0;
    float branch[2]{1.0F, 2.0F};
    float trunk[3]{3.0F, 4.0F, 5.0F};
    const float* solved = router.solve(branch, 2, trunk, 3, &solve_dim);
    check(solved == solve_ptr->solve_values.data(), "solve delegates");
    check(solve_ptr->solve_calls == 1, "solve call count");
    check(solve_ptr->saw_branch && solve_ptr->saw_trunk, "solve forwards input pointers");
    check(solve_ptr->last_branch_len == 2 && solve_ptr->last_trunk_len == 3, "solve forwards lengths");
    check(solve_dim == static_cast<int32_t>(solve_ptr->solve_values.size()), "solve forwards out_dim");

    int32_t out_channels = 0;
    int32_t out_h = 0;
    int32_t out_w = 0;
    float field[4]{0.0F, 1.0F, 2.0F, 3.0F};
    const float* solved_field = router.solve_field(field, 4, &out_channels, &out_h, &out_w);
    check(solved_field == solve_ptr->solve_field_values.data(), "solve_field delegates");
    check(solve_ptr->solve_field_calls == 1, "solve_field call count");
    check(solve_ptr->saw_field, "solve_field forwards input pointer");
    check(solve_ptr->last_input_size == 4, "solve_field forwards input size");
    check(out_channels == 1 && out_h == 2 && out_w == 2, "solve_field forwards output shape");
}

} // namespace

int main()
{
    test_unsupported_defaults();
    test_text_service_delegation_and_gates();
    test_video_service_delegation();
    test_segmentation_service_delegation_and_prompted_gate();
    test_audio_service_delegation_and_speech_gate();
    test_remaining_capability_delegation();

    if (g_failures > 0)
    {
        std::cerr << g_failures << " test(s) FAILED\n";
        return 1;
    }

    std::cerr << "All pipeline_router tests passed.\n";
    return 0;
}
