#include "trtf/runtime/adapters/io/media_io_adapter.h"

#include "runtime/trt/multimodal/image_preprocessor.h"
#include "utils/wav_reader.h"

#include <exception>
#include <utility>

namespace trtf::runtime::adapters::io {

ImageLoadResult FileImageInputLoader::load(const char* path) const
{
    if (path == nullptr || path[0] == '\0')
    {
        return ImageLoadResult::Failure(MediaIoStatus::kInvalidArgument, "image input path missing");
    }

    try
    {
        auto image = trtf::decode_image_rgb(path);
        if (image.empty())
        {
            return ImageLoadResult::Failure(MediaIoStatus::kIoError, "failed to decode image input");
        }
        return ImageLoadResult::Success(std::move(image));
    }
    catch (const std::exception& e)
    {
        return ImageLoadResult::Failure(MediaIoStatus::kIoError, e.what());
    }
}

AudioLoadResult WavFileAudioInputLoader::load(const char* path) const
{
    if (path == nullptr || path[0] == '\0')
    {
        return AudioLoadResult::Failure(MediaIoStatus::kInvalidArgument, "audio input path missing");
    }

    try
    {
        auto wav = trtf::read_wav(path);
        if (wav.samples.empty() || wav.sample_rate <= 0)
        {
            return AudioLoadResult::Failure(MediaIoStatus::kIoError, "failed to decode audio input");
        }
        return AudioLoadResult::Success({std::move(wav.samples), wav.sample_rate});
    }
    catch (const std::exception& e)
    {
        return AudioLoadResult::Failure(MediaIoStatus::kIoError, e.what());
    }
}

} // namespace trtf::runtime::adapters::io
