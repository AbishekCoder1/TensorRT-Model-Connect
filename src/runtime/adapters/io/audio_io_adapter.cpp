#include "trtf/runtime/adapters/io/audio_io_adapter.h"

#include "runtime/trt/audio/bark_backend.h"

namespace trtf::runtime::adapters::io {

AudioWriteResult AudioWavFileWriter::write_wav(
    const AudioArtifact& artifact,
    const char* output_path) const
{
    if (output_path == nullptr || output_path[0] == '\0')
    {
        return {AudioIoStatus::kInvalidArgument, 0, "Audio output path is required"};
    }
    if (artifact.num_samples <= 0 || artifact.sample_rate <= 0)
    {
        return {AudioIoStatus::kInvalidArgument, 0, "Audio artifact must contain samples and a positive sample rate"};
    }
    if (artifact.waveform.size() < static_cast<std::size_t>(artifact.num_samples))
    {
        return {AudioIoStatus::kInvalidArgument, 0, "Audio artifact size does not match sample metadata"};
    }
    if (!trtf::write_wav(output_path, artifact.waveform.data(), artifact.num_samples, artifact.sample_rate))
    {
        return {AudioIoStatus::kIoError, 0, std::string("Failed to write audio output: ") + output_path};
    }
    return {AudioIoStatus::kOk, artifact.num_samples, {}};
}

} // namespace trtf::runtime::adapters::io
