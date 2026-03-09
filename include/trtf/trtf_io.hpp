#pragma once

// trtf_io.hpp — header-only file I/O utilities for trtf pipeline results.
//
// Usage:
//   #include <trtf/trtf_io.hpp>
//   auto pipe = trtf::load("model.trtfb");
//   auto img = pipe->generate_image("a cat");
//   trtf::io::save_png(img, "output.png");

#include "trtf/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtf::io {

// Write a WAV file from AudioResult.
inline void write_wav(const AudioResult& audio, const std::string& path)
{
    if (audio.samples.empty())
        throw std::runtime_error("write_wav: empty audio");

    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("write_wav: cannot open " + path);

    const int32_t num_samples = static_cast<int32_t>(audio.samples.size());
    const int32_t sample_rate = audio.sample_rate;
    const int16_t num_channels = 1;
    const int16_t bits_per_sample = 16;
    const int32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    const int16_t block_align = num_channels * bits_per_sample / 8;
    const int32_t data_size = num_samples * block_align;
    const int32_t chunk_size = 36 + data_size;

    // RIFF header
    f.write("RIFF", 4);
    f.write(reinterpret_cast<const char*>(&chunk_size), 4);
    f.write("WAVE", 4);

    // fmt chunk
    f.write("fmt ", 4);
    int32_t fmt_size = 16;
    int16_t audio_format = 1; // PCM
    f.write(reinterpret_cast<const char*>(&fmt_size), 4);
    f.write(reinterpret_cast<const char*>(&audio_format), 2);
    f.write(reinterpret_cast<const char*>(&num_channels), 2);
    f.write(reinterpret_cast<const char*>(&sample_rate), 4);
    f.write(reinterpret_cast<const char*>(&byte_rate), 4);
    f.write(reinterpret_cast<const char*>(&block_align), 2);
    f.write(reinterpret_cast<const char*>(&bits_per_sample), 2);

    // data chunk
    f.write("data", 4);
    f.write(reinterpret_cast<const char*>(&data_size), 4);

    // Convert float [-1,1] to int16
    for (float sample : audio.samples)
    {
        float clamped = std::max(-1.0f, std::min(1.0f, sample));
        auto val = static_cast<int16_t>(clamped * 32767.0f);
        f.write(reinterpret_cast<const char*>(&val), 2);
    }
}

// Read a WAV file into an AudioResult (mono float32).
inline AudioResult read_wav(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("read_wav: cannot open " + path);

    // Read RIFF header
    char riff[4];
    f.read(riff, 4);
    if (std::string(riff, 4) != "RIFF")
        throw std::runtime_error("read_wav: not a RIFF file");

    int32_t chunk_size = 0;
    f.read(reinterpret_cast<char*>(&chunk_size), 4);

    char wave[4];
    f.read(wave, 4);
    if (std::string(wave, 4) != "WAVE")
        throw std::runtime_error("read_wav: not a WAVE file");

    int32_t sample_rate = 0;
    int16_t num_channels = 0;
    int16_t bits_per_sample = 0;

    // Find fmt and data chunks
    std::vector<int16_t> raw_samples;
    while (f)
    {
        char id[4];
        if (!f.read(id, 4)) break;
        int32_t size = 0;
        f.read(reinterpret_cast<char*>(&size), 4);

        if (std::string(id, 4) == "fmt ")
        {
            int16_t format = 0;
            f.read(reinterpret_cast<char*>(&format), 2);
            f.read(reinterpret_cast<char*>(&num_channels), 2);
            f.read(reinterpret_cast<char*>(&sample_rate), 4);
            f.seekg(size - 8, std::ios::cur); // skip rest of fmt
        }
        else if (std::string(id, 4) == "data")
        {
            bits_per_sample = 16; // assume 16-bit PCM
            int32_t num = size / 2;
            raw_samples.resize(static_cast<std::size_t>(num));
            f.read(reinterpret_cast<char*>(raw_samples.data()), size);
        }
        else
        {
            f.seekg(size, std::ios::cur);
        }
    }

    // Convert to mono float32
    AudioResult result;
    result.sample_rate = sample_rate;
    if (num_channels <= 1)
    {
        result.samples.resize(raw_samples.size());
        for (std::size_t i = 0; i < raw_samples.size(); ++i)
            result.samples[i] = static_cast<float>(raw_samples[i]) / 32768.0f;
    }
    else
    {
        // Downmix to mono
        std::size_t frames = raw_samples.size() / static_cast<std::size_t>(num_channels);
        result.samples.resize(frames);
        for (std::size_t i = 0; i < frames; ++i)
        {
            float sum = 0.0f;
            for (int16_t ch = 0; ch < num_channels; ++ch)
                sum += static_cast<float>(raw_samples[i * num_channels + ch]);
            result.samples[i] = sum / (32768.0f * num_channels);
        }
    }
    result.num_samples = static_cast<int32_t>(result.samples.size());
    return result;
}

// Loaded image: float RGB pixels in HWC layout [height * width * 3], values in [0, 1].
struct LoadedImage {
    std::vector<float> pixels;  // [H * W * 3] float32 in [0, 1]
    int32_t height{0};
    int32_t width{0};

    bool empty() const { return pixels.empty(); }
};

// Load an image file (JPEG, PNG, BMP, etc.) and return float RGB pixels.
// Uses stb_image internally (linked via trtf_core).
// Throws on file-not-found; returns empty LoadedImage on decode failure.
LoadedImage read_image(const std::string& path);

// Legacy placeholder (prefer read_image).
inline std::vector<float> decode_image(const std::string& path, int& h, int& w)
{
    auto img = read_image(path);
    h = img.height;
    w = img.width;
    return std::move(img.pixels);
}

} // namespace trtf::io
