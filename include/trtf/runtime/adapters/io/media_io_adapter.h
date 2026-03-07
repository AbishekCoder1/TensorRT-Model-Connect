#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace trtf::runtime::adapters::io {

enum class MediaIoStatus {
    kOk,
    kInvalidArgument,
    kIoError,
};

template <typename Payload>
struct MediaLoadResult {
    MediaIoStatus status{MediaIoStatus::kOk};
    Payload value{};
    std::string message;

    [[nodiscard]] bool ok() const
    {
        return status == MediaIoStatus::kOk;
    }

    static MediaLoadResult Success(Payload value_in)
    {
        MediaLoadResult result;
        result.value = std::move(value_in);
        return result;
    }

    static MediaLoadResult Failure(MediaIoStatus status_in, std::string message_in)
    {
        MediaLoadResult result;
        result.status = status_in;
        result.message = std::move(message_in);
        return result;
    }
};

struct DecodedImage {
    std::vector<unsigned char> pixels;
    int32_t width{0};
    int32_t height{0};
    int32_t channels{0};

    [[nodiscard]] bool empty() const
    {
        return pixels.empty() || width <= 0 || height <= 0 || channels <= 0;
    }
};

struct DecodedAudio {
    std::vector<float> samples;
    int32_t sample_rate{0};

    [[nodiscard]] bool empty() const
    {
        return samples.empty() || sample_rate <= 0;
    }
};

using ImageLoadResult = MediaLoadResult<DecodedImage>;
using AudioLoadResult = MediaLoadResult<DecodedAudio>;

class IImageInputLoader {
public:
    virtual ~IImageInputLoader() = default;

    virtual ImageLoadResult load(const char* path) const = 0;
};

class FileImageInputLoader final : public IImageInputLoader {
public:
    ImageLoadResult load(const char* path) const override;
};

class IAudioInputLoader {
public:
    virtual ~IAudioInputLoader() = default;

    virtual AudioLoadResult load(const char* path) const = 0;
};

class WavFileAudioInputLoader final : public IAudioInputLoader {
public:
    AudioLoadResult load(const char* path) const override;
};

} // namespace trtf::runtime::adapters::io
