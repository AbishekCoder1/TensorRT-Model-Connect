#include "runtime/trt/audio/audio_types.h"

#include <fstream>

namespace trtf {

bool write_wav(const std::string& path, const float* samples,
               int32_t num_samples, int32_t sample_rate)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        return false;
    }

    const int32_t num_channels = 1;
    const int32_t bits_per_sample = 32;
    const int32_t byte_rate = sample_rate * num_channels * (bits_per_sample / 8);
    const int32_t block_align = num_channels * (bits_per_sample / 8);
    const int32_t data_size = num_samples * block_align;
    const int32_t chunk_size = 36 + data_size;

    // RIFF header (44 bytes)
    out.write("RIFF", 4);
    out.write(reinterpret_cast<const char*>(&chunk_size), 4);
    out.write("WAVE", 4);

    // fmt sub-chunk
    out.write("fmt ", 4);
    const int32_t fmt_size = 16;
    out.write(reinterpret_cast<const char*>(&fmt_size), 4);
    const int16_t audio_format = 3;  // IEEE float
    out.write(reinterpret_cast<const char*>(&audio_format), 2);
    const int16_t channels = static_cast<int16_t>(num_channels);
    out.write(reinterpret_cast<const char*>(&channels), 2);
    out.write(reinterpret_cast<const char*>(&sample_rate), 4);
    out.write(reinterpret_cast<const char*>(&byte_rate), 4);
    const int16_t ba = static_cast<int16_t>(block_align);
    out.write(reinterpret_cast<const char*>(&ba), 2);
    const int16_t bps = static_cast<int16_t>(bits_per_sample);
    out.write(reinterpret_cast<const char*>(&bps), 2);

    // data sub-chunk
    out.write("data", 4);
    out.write(reinterpret_cast<const char*>(&data_size), 4);
    out.write(reinterpret_cast<const char*>(samples),
              static_cast<std::streamsize>(data_size));

    return out.good();
}

} // namespace trtf
