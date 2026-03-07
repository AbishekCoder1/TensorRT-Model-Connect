#pragma once

#include "trtf/pipeline.h"
#include "trtf/runtime/adapters/io/audio_io_adapter.h"
#include "trtf/runtime/adapters/io/detection_io_adapter.h"
#include "trtf/runtime/adapters/io/image_io_adapter.h"
#include "trtf/runtime/adapters/io/media_io_adapter.h"
#include "trtf/runtime/contracts/services.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace trtf::runtime {

struct PipelineRouterDefaults {
    std::size_t max_new_tokens{0};
    int32_t video_num_steps{-1};
    float video_guidance_scale{-1.0F};
    int32_t transcription_max_new_tokens{224};
    int32_t audio_max_tokens{-1};
    float detection_conf_threshold{-1.0F};
    int32_t speech_max_output_frames{-1};
    int32_t speech_tail_frames{0};
};

struct PipelineRouterIoAdapters {
    std::unique_ptr<adapters::io::IImageInputLoader> image_loader;
    std::unique_ptr<adapters::io::IAudioInputLoader> audio_loader;
    std::unique_ptr<adapters::io::IAudioArtifactWriter> audio_writer;
    std::unique_ptr<adapters::io::ISegmentationArtifactWriter> segmentation_writer;
    std::unique_ptr<adapters::io::ISamMaskArtifactWriter> sam_mask_writer;
    std::unique_ptr<adapters::io::IDetectionArtifactWriter> detection_writer;
    std::unique_ptr<adapters::io::IVideoFrameArtifactWriter> video_frame_writer;
};

class PipelineRouter final : public trtf::IPipeline {
public:
    using trtf::IPipeline::generate;

    PipelineRouter(PipelineServices services, std::string model_id, std::string backend_name,
        PipelineRouterDefaults defaults = {}, PipelineRouterIoAdapters io_adapters = {});

    const char* generate(const char* prompt, std::size_t max_new_tokens) override;
    const char* generate(const char* prompt, const char* image_path, std::size_t max_new_tokens) override;
    bool supports_vision() const override;
    bool supports_video() const override;
    int32_t generate_video(
        const char* prompt, const char* output_dir, int32_t num_steps, float guidance_scale) override;
    bool supports_segmentation() const override;
    int32_t segment(const char* image_path, const char* output_path) override;
    bool supports_encoding() const override;
    const float* encode(const char* text, int32_t* out_seq_len, int32_t* out_hidden_size) override;
    bool supports_solve() const override;
    const float* solve(const float* branch_input, int32_t branch_len, const float* trunk_input,
        int32_t trunk_len, int32_t* out_dim) override;
    const float* solve_field(
        const float* field_input, int32_t input_size, int32_t* out_channels, int32_t* out_h, int32_t* out_w)
        override;
    bool supports_detection() const override;
    int32_t detect(const char* image_path, const char* output_path, float conf_threshold) override;
    bool supports_transcription() const override;
    const char* transcribe(const char* audio_path, int32_t max_new_tokens) override;
    bool supports_audio() const override;
    int32_t generate_audio(const char* prompt, const char* output_path, int32_t max_tokens) override;
    bool supports_embedding() const override;
    const float* embed(const char* text, int32_t* out_dim) override;
    const float* embed_image(const char* image_path, int32_t* out_dim) override;
    const float* embed_image_text(const char* text, const char* image_path, int32_t* out_dim) override;
    bool supports_reranking() const override;
    float rerank(const char* query, const char* document) override;
    bool supports_prompted_segmentation() const override;
    int32_t segment_sam(
        const char* image_path, const char* output_dir, float point_x, float point_y, bool is_foreground)
        override;
    bool supports_speech() const override;
    int32_t speak(
        const char* audio_in, const char* audio_out, int32_t max_output_frames, int32_t tail_frames) override;
    const char* model_id() const override;
    const char* backend_name() const override;
    void set_default_max_new_tokens(std::size_t max_new_tokens);

private:
    std::size_t resolve_max_new_tokens(std::size_t requested) const;
    int32_t resolve_video_num_steps(int32_t requested) const;
    float resolve_video_guidance_scale(float requested) const;
    int32_t resolve_transcription_max_new_tokens(int32_t requested) const;
    int32_t resolve_audio_max_tokens(int32_t requested) const;
    float resolve_detection_conf_threshold(float requested) const;
    int32_t resolve_speech_max_output_frames(int32_t requested) const;
    int32_t resolve_speech_tail_frames(int32_t requested) const;
    const char* clear_and_return_empty_output();
    void log_service_failure(const char* operation, const std::string& message) const;

    PipelineServices mServices;
    std::string mModelId;
    std::string mBackendName;
    PipelineRouterDefaults mDefaults;
    std::unique_ptr<adapters::io::IImageInputLoader> mImageLoader;
    std::unique_ptr<adapters::io::IAudioInputLoader> mAudioLoader;
    std::unique_ptr<adapters::io::IAudioArtifactWriter> mAudioWriter;
    std::unique_ptr<adapters::io::ISegmentationArtifactWriter> mSegmentationWriter;
    std::unique_ptr<adapters::io::ISamMaskArtifactWriter> mSamMaskWriter;
    std::unique_ptr<adapters::io::IDetectionArtifactWriter> mDetectionWriter;
    std::unique_ptr<adapters::io::IVideoFrameArtifactWriter> mVideoFrameWriter;
    std::string mLastOutput;
};

} // namespace trtf::runtime
