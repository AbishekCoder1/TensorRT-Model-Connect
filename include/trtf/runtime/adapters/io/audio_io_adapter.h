#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtf::runtime::adapters::io {

enum class AudioIoStatus {
    kOk,
    kInvalidArgument,
    kIoError,
};

struct AudioArtifact {
    std::vector<float> waveform;
    int32_t sample_rate{0};
    int32_t num_samples{0};
};

struct AudioWriteResult {
    AudioIoStatus status{AudioIoStatus::kOk};
    int32_t written{0};
    std::string message;

    [[nodiscard]] bool ok() const
    {
        return status == AudioIoStatus::kOk;
    }
};

class IAudioArtifactWriter {
public:
    virtual ~IAudioArtifactWriter() = default;

    virtual AudioWriteResult write_wav(const AudioArtifact& artifact, const char* output_path) const = 0;
};

class AudioWavFileWriter final : public IAudioArtifactWriter {
public:
    AudioWriteResult write_wav(const AudioArtifact& artifact, const char* output_path) const override;
};

} // namespace trtf::runtime::adapters::io
