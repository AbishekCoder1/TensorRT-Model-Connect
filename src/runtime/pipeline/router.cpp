#include "trtf/runtime/pipeline/router.h"

#include <algorithm>
#include <iostream>
#include <utility>

namespace trtf::runtime {
namespace {

std::unique_ptr<adapters::io::IImageInputLoader> make_image_loader(
    std::unique_ptr<adapters::io::IImageInputLoader> loader)
{
    if (loader != nullptr)
    {
        return loader;
    }
    return std::make_unique<adapters::io::FileImageInputLoader>();
}

std::unique_ptr<adapters::io::IAudioInputLoader> make_audio_loader(
    std::unique_ptr<adapters::io::IAudioInputLoader> loader)
{
    if (loader != nullptr)
    {
        return loader;
    }
    return std::make_unique<adapters::io::WavFileAudioInputLoader>();
}

std::unique_ptr<adapters::io::IAudioArtifactWriter> make_audio_writer(
    std::unique_ptr<adapters::io::IAudioArtifactWriter> writer)
{
    if (writer != nullptr)
    {
        return writer;
    }
    return std::make_unique<adapters::io::AudioWavFileWriter>();
}

std::unique_ptr<adapters::io::ISegmentationArtifactWriter> make_segmentation_writer(
    std::unique_ptr<adapters::io::ISegmentationArtifactWriter> writer)
{
    if (writer != nullptr)
    {
        return writer;
    }
    return std::make_unique<adapters::io::SegmentationPngFileWriter>();
}

std::unique_ptr<adapters::io::ISamMaskArtifactWriter> make_sam_writer(
    std::unique_ptr<adapters::io::ISamMaskArtifactWriter> writer)
{
    if (writer != nullptr)
    {
        return writer;
    }
    return std::make_unique<adapters::io::SamMaskDirectoryWriter>();
}

std::unique_ptr<adapters::io::IDetectionArtifactWriter> make_detection_writer(
    std::unique_ptr<adapters::io::IDetectionArtifactWriter> writer)
{
    if (writer != nullptr)
    {
        return writer;
    }
    return std::make_unique<adapters::io::DetectionJsonFileWriter>();
}

std::unique_ptr<adapters::io::IVideoFrameArtifactWriter> make_video_writer(
    std::unique_ptr<adapters::io::IVideoFrameArtifactWriter> writer)
{
    if (writer != nullptr)
    {
        return writer;
    }
    return std::make_unique<adapters::io::VideoFrameDirectoryWriter>();
}

} // namespace

PipelineRouter::PipelineRouter(PipelineServices services, std::string model_id, std::string backend_name,
    PipelineRouterDefaults defaults, PipelineRouterIoAdapters io_adapters)
    : mServices(std::move(services))
    , mModelId(std::move(model_id))
    , mBackendName(std::move(backend_name))
    , mDefaults(defaults)
    , mImageLoader(make_image_loader(std::move(io_adapters.image_loader)))
    , mAudioLoader(make_audio_loader(std::move(io_adapters.audio_loader)))
    , mAudioWriter(make_audio_writer(std::move(io_adapters.audio_writer)))
    , mSegmentationWriter(make_segmentation_writer(std::move(io_adapters.segmentation_writer)))
    , mSamMaskWriter(make_sam_writer(std::move(io_adapters.sam_mask_writer)))
    , mDetectionWriter(make_detection_writer(std::move(io_adapters.detection_writer)))
    , mVideoFrameWriter(make_video_writer(std::move(io_adapters.video_frame_writer)))
{
}

const char* PipelineRouter::generate(const char* prompt, std::size_t max_new_tokens)
{
    if (mServices.text == nullptr)
    {
        return clear_and_return_empty_output();
    }
    return mServices.text->generate(prompt, resolve_max_new_tokens(max_new_tokens));
}

const char* PipelineRouter::generate(
    const char* prompt, const char* image_path, std::size_t max_new_tokens)
{
    if (mServices.text == nullptr)
    {
        return clear_and_return_empty_output();
    }

    if (!supports_vision() || image_path == nullptr)
    {
        return generate(prompt, max_new_tokens);
    }

    const auto image_result = mImageLoader->load(image_path);
    if (!image_result.ok())
    {
        std::cerr << "[trtf] vision input load failed: " << image_result.message
                  << ", falling back to text-only" << std::endl;
        return generate(prompt, max_new_tokens);
    }

    return mServices.text->generate(prompt, image_result.value, resolve_max_new_tokens(max_new_tokens));
}

bool PipelineRouter::supports_vision() const
{
    return mServices.text != nullptr && mServices.text->supports_vision();
}

bool PipelineRouter::supports_video() const
{
    return mServices.video != nullptr;
}

int32_t PipelineRouter::generate_video(
    const char* prompt, const char* output_dir, int32_t num_steps, float guidance_scale)
{
    if (mServices.video == nullptr || prompt == nullptr)
    {
        return -1;
    }

    const auto result = mServices.video->generate_video(
        {prompt != nullptr ? prompt : "", resolve_video_num_steps(num_steps), resolve_video_guidance_scale(guidance_scale)});
    if (!result.ok())
    {
        log_service_failure("video generation", result.message);
        return -1;
    }

    const auto write_result = mVideoFrameWriter->write_png_frames(result.value, output_dir);
    if (!write_result.ok())
    {
        std::cerr << "[trtf] Video frame write error: " << write_result.message << std::endl;
        return -1;
    }
    return result.value.num_frames;
}

bool PipelineRouter::supports_segmentation() const
{
    return mServices.segmentation != nullptr;
}

int32_t PipelineRouter::segment(const char* image_path, const char* output_path)
{
    if (mServices.segmentation == nullptr)
    {
        return -1;
    }

    const auto image_result = mImageLoader->load(image_path);
    if (!image_result.ok())
    {
        std::cerr << "[trtf] Segmentation input load error: " << image_result.message << std::endl;
        return -1;
    }

    const auto result = mServices.segmentation->segment({std::move(image_result.value)});
    if (!result.ok())
    {
        log_service_failure("segmentation", result.message);
        return -1;
    }

    const auto write_result = mSegmentationWriter->write_png(result.value, output_path);
    if (!write_result.ok())
    {
        std::cerr << "[trtf] Segmentation write error: " << write_result.message << std::endl;
        return -1;
    }
    return 0;
}

bool PipelineRouter::supports_encoding() const
{
    return mServices.text != nullptr && mServices.text->supports_encoding();
}

const float* PipelineRouter::encode(const char* text, int32_t* out_seq_len, int32_t* out_hidden_size)
{
    if (!supports_encoding())
    {
        return nullptr;
    }
    return mServices.text->encode(text, out_seq_len, out_hidden_size);
}

bool PipelineRouter::supports_solve() const
{
    return mServices.solve != nullptr;
}

const float* PipelineRouter::solve(const float* branch_input, int32_t branch_len,
    const float* trunk_input, int32_t trunk_len, int32_t* out_dim)
{
    if (mServices.solve == nullptr)
    {
        return nullptr;
    }
    return mServices.solve->solve(branch_input, branch_len, trunk_input, trunk_len, out_dim);
}

const float* PipelineRouter::solve_field(
    const float* field_input, int32_t input_size, int32_t* out_channels, int32_t* out_h, int32_t* out_w)
{
    if (mServices.solve == nullptr)
    {
        return nullptr;
    }
    return mServices.solve->solve_field(field_input, input_size, out_channels, out_h, out_w);
}

bool PipelineRouter::supports_detection() const
{
    return mServices.detection != nullptr;
}

int32_t PipelineRouter::detect(const char* image_path, const char* output_path, float conf_threshold)
{
    if (mServices.detection == nullptr)
    {
        return -1;
    }

    const auto resolved_threshold = resolve_detection_conf_threshold(conf_threshold);
    const auto image_result = mImageLoader->load(image_path);
    if (!image_result.ok())
    {
        std::cerr << "[trtf] Detection input load error: " << image_result.message << std::endl;
        return -1;
    }

    const auto result = mServices.detection->detect({std::move(image_result.value), resolved_threshold});
    if (!result.ok())
    {
        log_service_failure("detection", result.message);
        return -1;
    }

    const auto write_result = mDetectionWriter->write_json(result.value, output_path, resolved_threshold);
    if (!write_result.ok())
    {
        std::cerr << "[trtf] Detection write error: " << write_result.message << std::endl;
        return -1;
    }
    return write_result.written;
}

bool PipelineRouter::supports_transcription() const
{
    return mServices.transcription != nullptr;
}

const char* PipelineRouter::transcribe(const char* audio_path, int32_t max_new_tokens)
{
    if (mServices.transcription == nullptr)
    {
        return nullptr;
    }

    const auto audio_result = mAudioLoader->load(audio_path);
    if (!audio_result.ok())
    {
        std::cerr << "[trtf] Transcription input load error: " << audio_result.message << std::endl;
        return nullptr;
    }

    const auto result = mServices.transcription->transcribe(
        {std::move(audio_result.value), resolve_transcription_max_new_tokens(max_new_tokens)});
    if (!result.ok())
    {
        log_service_failure("transcription", result.message);
        return nullptr;
    }

    mLastOutput = result.value;
    return mLastOutput.c_str();
}

bool PipelineRouter::supports_audio() const
{
    return mServices.audio != nullptr;
}

int32_t PipelineRouter::generate_audio(const char* prompt, const char* output_path, int32_t max_tokens)
{
    if (mServices.audio == nullptr || prompt == nullptr)
    {
        return -1;
    }

    const auto result = mServices.audio->generate_audio(
        {prompt != nullptr ? prompt : "", resolve_audio_max_tokens(max_tokens)});
    if (!result.ok())
    {
        log_service_failure("audio generation", result.message);
        return -1;
    }

    const auto write_result = mAudioWriter->write_wav(result.value, output_path);
    if (!write_result.ok())
    {
        std::cerr << "[trtf] Audio write error: " << write_result.message << std::endl;
        return -1;
    }
    return result.value.num_samples;
}

bool PipelineRouter::supports_embedding() const
{
    return mServices.embedding != nullptr;
}

const float* PipelineRouter::embed(const char* text, int32_t* out_dim)
{
    if (mServices.embedding == nullptr)
    {
        return nullptr;
    }
    return mServices.embedding->embed(text, out_dim);
}

const float* PipelineRouter::embed_image(const char* image_path, int32_t* out_dim)
{
    if (mServices.embedding == nullptr)
    {
        return nullptr;
    }

    const auto image_result = mImageLoader->load(image_path);
    if (!image_result.ok())
    {
        std::cerr << "[trtf] Embedding image load error: " << image_result.message << std::endl;
        return nullptr;
    }
    return mServices.embedding->embed_image(image_result.value, out_dim);
}

const float* PipelineRouter::embed_image_text(const char* text, const char* image_path, int32_t* out_dim)
{
    if (mServices.embedding == nullptr)
    {
        return nullptr;
    }

    const auto image_result = mImageLoader->load(image_path);
    if (!image_result.ok())
    {
        std::cerr << "[trtf] Embedding vision-text image load error: " << image_result.message << std::endl;
        return nullptr;
    }
    return mServices.embedding->embed_image_text(text, image_result.value, out_dim);
}

bool PipelineRouter::supports_reranking() const
{
    return mServices.rerank != nullptr;
}

float PipelineRouter::rerank(const char* query, const char* document)
{
    if (mServices.rerank == nullptr)
    {
        return 0.0F;
    }
    return mServices.rerank->rerank(query, document);
}

bool PipelineRouter::supports_prompted_segmentation() const
{
    return mServices.segmentation != nullptr && mServices.segmentation->supports_prompted();
}

int32_t PipelineRouter::segment_sam(
    const char* image_path, const char* output_dir, float point_x, float point_y, bool is_foreground)
{
    if (!supports_prompted_segmentation())
    {
        return -1;
    }

    const auto image_result = mImageLoader->load(image_path);
    if (!image_result.ok())
    {
        std::cerr << "[trtf] Prompted segmentation input load error: " << image_result.message << std::endl;
        return -1;
    }

    const auto result = mServices.segmentation->segment_prompt(
        {std::move(image_result.value), point_x, point_y, is_foreground});
    if (!result.ok())
    {
        log_service_failure("prompted segmentation", result.message);
        return -1;
    }

    const auto write_result = mSamMaskWriter->write_masks(result.value, output_dir);
    if (!write_result.ok())
    {
        std::cerr << "[trtf] SAM mask write error: " << write_result.message << std::endl;
        return -1;
    }
    return result.value.num_masks;
}

bool PipelineRouter::supports_speech() const
{
    return mServices.audio != nullptr && mServices.audio->supports_speech();
}

int32_t PipelineRouter::speak(
    const char* audio_in, const char* audio_out, int32_t max_output_frames, int32_t tail_frames)
{
    if (!supports_speech())
    {
        return -1;
    }

    const auto audio_result = mAudioLoader->load(audio_in);
    if (!audio_result.ok())
    {
        std::cerr << "[trtf] Speech input load error: " << audio_result.message << std::endl;
        return -1;
    }

    const auto result = mServices.audio->speak(
        {std::move(audio_result.value), resolve_speech_max_output_frames(max_output_frames),
            resolve_speech_tail_frames(tail_frames)});
    if (!result.ok())
    {
        log_service_failure("speech synthesis", result.message);
        return -1;
    }

    const auto write_result = mAudioWriter->write_wav(result.value, audio_out);
    if (!write_result.ok())
    {
        std::cerr << "[trtf] Speech write error: " << write_result.message << std::endl;
        return -1;
    }
    return result.value.num_samples;
}

const char* PipelineRouter::model_id() const
{
    return mModelId.c_str();
}

const char* PipelineRouter::backend_name() const
{
    return mBackendName.c_str();
}

void PipelineRouter::set_default_max_new_tokens(std::size_t max_new_tokens)
{
    if (max_new_tokens > 0)
    {
        mDefaults.max_new_tokens = max_new_tokens;
    }
}

std::size_t PipelineRouter::resolve_max_new_tokens(std::size_t requested) const
{
    return requested == 0 ? mDefaults.max_new_tokens : requested;
}

int32_t PipelineRouter::resolve_video_num_steps(int32_t requested) const
{
    return requested < 0 ? mDefaults.video_num_steps : requested;
}

float PipelineRouter::resolve_video_guidance_scale(float requested) const
{
    return requested < 0.0F ? mDefaults.video_guidance_scale : requested;
}

int32_t PipelineRouter::resolve_transcription_max_new_tokens(int32_t requested) const
{
    return requested > 0 ? requested : mDefaults.transcription_max_new_tokens;
}

int32_t PipelineRouter::resolve_audio_max_tokens(int32_t requested) const
{
    return requested < 0 ? mDefaults.audio_max_tokens : requested;
}

float PipelineRouter::resolve_detection_conf_threshold(float requested) const
{
    return requested < 0.0F ? mDefaults.detection_conf_threshold : requested;
}

int32_t PipelineRouter::resolve_speech_max_output_frames(int32_t requested) const
{
    return requested > 0 ? requested : mDefaults.speech_max_output_frames;
}

int32_t PipelineRouter::resolve_speech_tail_frames(int32_t requested) const
{
    if (requested >= 0)
    {
        return requested;
    }
    return std::max(mDefaults.speech_tail_frames, 0);
}

const char* PipelineRouter::clear_and_return_empty_output()
{
    mLastOutput.clear();
    return mLastOutput.c_str();
}

void PipelineRouter::log_service_failure(const char* operation, const std::string& message) const
{
    std::cerr << "[trtf] " << operation << " failed";
    if (!message.empty())
    {
        std::cerr << ": " << message;
    }
    std::cerr << std::endl;
}

} // namespace trtf::runtime
