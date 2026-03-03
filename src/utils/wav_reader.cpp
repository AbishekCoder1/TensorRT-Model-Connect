#include "utils/wav_reader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace trtf {

WavData read_wav(const std::string& path)
{
    std::ifstream infile(path, std::ios::binary);
    if (!infile)
        throw std::runtime_error("Failed to open WAV file: " + path);

    std::vector<char> wav_bytes(
        (std::istreambuf_iterator<char>(infile)),
        std::istreambuf_iterator<char>());
    if (wav_bytes.size() < 44)
        throw std::runtime_error("WAV file too small: " + path);

    if (std::memcmp(wav_bytes.data(), "RIFF", 4) != 0 ||
        std::memcmp(wav_bytes.data() + 8, "WAVE", 4) != 0)
    {
        throw std::runtime_error("Invalid WAV container: " + path);
    }

    uint16_t fmt_tag = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    const char* raw_ptr = nullptr;
    std::size_t raw_size = 0;
    bool have_fmt = false;
    bool have_data = false;

    std::size_t pos = 12;
    while (pos + 8 <= wav_bytes.size())
    {
        const char* chunk = wav_bytes.data() + pos;
        const char* chunk_data = chunk + 8;
        uint32_t chunk_size = 0;
        std::memcpy(&chunk_size, chunk + 4, sizeof(uint32_t));
        if (pos + 8 + static_cast<std::size_t>(chunk_size) > wav_bytes.size())
            break;

        if (std::memcmp(chunk, "fmt ", 4) == 0)
        {
            if (chunk_size < 16)
                throw std::runtime_error("WAV fmt chunk too small");
            std::memcpy(&fmt_tag, chunk_data + 0, sizeof(uint16_t));
            std::memcpy(&channels, chunk_data + 2, sizeof(uint16_t));
            std::memcpy(&sample_rate, chunk_data + 4, sizeof(uint32_t));
            std::memcpy(&bits_per_sample, chunk_data + 14, sizeof(uint16_t));
            have_fmt = true;
        }
        else if (std::memcmp(chunk, "data", 4) == 0)
        {
            raw_ptr = chunk_data;
            raw_size = static_cast<std::size_t>(chunk_size);
            have_data = true;
        }

        pos += 8 + static_cast<std::size_t>(chunk_size);
        if ((chunk_size & 1U) != 0)
            ++pos;  // RIFF chunks are word-aligned
    }

    if (!have_fmt || !have_data || raw_ptr == nullptr || raw_size == 0)
        throw std::runtime_error("WAV missing fmt/data chunk: " + path);

    // Convert to float32 samples
    std::vector<float> samples;
    if (fmt_tag == 3 && bits_per_sample == 32)
    {
        // IEEE float32
        auto ns = static_cast<int32_t>(raw_size / sizeof(float));
        samples.resize(ns);
        std::memcpy(samples.data(), raw_ptr, ns * sizeof(float));
    }
    else if (fmt_tag == 1 && bits_per_sample == 16)
    {
        // PCM int16
        auto ns = static_cast<int32_t>(raw_size / sizeof(int16_t));
        samples.resize(ns);
        for (int32_t i = 0; i < ns; ++i)
        {
            int16_t pcm = 0;
            std::memcpy(&pcm, raw_ptr + i * sizeof(int16_t), sizeof(int16_t));
            samples[i] = static_cast<float>(pcm) / 32768.0F;
        }
    }
    else
    {
        throw std::runtime_error(
            "Unsupported WAV format: tag=" + std::to_string(fmt_tag) +
            " bits=" + std::to_string(bits_per_sample));
    }

    // Convert stereo to mono (average channels)
    if (channels == 2)
    {
        auto mono_len = static_cast<int32_t>(samples.size()) / 2;
        std::vector<float> mono(mono_len);
        for (int32_t i = 0; i < mono_len; ++i)
            mono[i] = (samples[2 * i] + samples[2 * i + 1]) * 0.5F;
        samples = std::move(mono);
    }
    else if (channels > 2)
    {
        // Take first channel
        auto mono_len = static_cast<int32_t>(samples.size()) / channels;
        std::vector<float> mono(mono_len);
        for (int32_t i = 0; i < mono_len; ++i)
            mono[i] = samples[i * channels];
        samples = std::move(mono);
    }

    WavData result;
    result.samples = std::move(samples);
    result.sample_rate = static_cast<int32_t>(sample_rate);
    return result;
}

std::vector<float> resample_linear(
    const float* samples, int32_t n_samples,
    int32_t source_rate, int32_t target_rate)
{
    if (source_rate == target_rate || n_samples <= 0)
        return std::vector<float>(samples, samples + n_samples);

    auto out_len = static_cast<int32_t>(
        static_cast<int64_t>(n_samples) * target_rate / source_rate);
    std::vector<float> resampled(out_len);

    // Use windowed-sinc interpolation for quality resampling.
    // This provides proper anti-aliasing for downsampling (e.g., 48kHz -> 16kHz).
    // Half-window size in input samples; more taps = better quality.
    const int32_t half_taps = 16;
    const double pi = 3.14159265358979323846;
    // Cutoff: min(1, target/source) to prevent aliasing when downsampling
    const double cutoff = std::min(1.0, static_cast<double>(target_rate) /
                                        static_cast<double>(source_rate));

    for (int32_t i = 0; i < out_len; ++i)
    {
        double src_pos = static_cast<double>(i) *
            static_cast<double>(source_rate) /
            static_cast<double>(target_rate);
        auto center = static_cast<int32_t>(std::floor(src_pos));
        double frac = src_pos - static_cast<double>(center);

        double acc = 0.0;
        double weight_sum = 0.0;
        const int32_t lo = std::max(0, center - half_taps + 1);
        const int32_t hi = std::min(n_samples - 1, center + half_taps);

        for (int32_t j = lo; j <= hi; ++j)
        {
            double d = static_cast<double>(j) - src_pos;
            // sinc(d * cutoff) * cutoff — the ideal low-pass filter
            double sinc_val;
            if (std::abs(d) < 1e-12)
                sinc_val = cutoff;
            else
                sinc_val = cutoff * std::sin(pi * d * cutoff) / (pi * d * cutoff);
            // Hann window over the tap range
            double win_pos = (d + static_cast<double>(half_taps)) /
                             (2.0 * static_cast<double>(half_taps));
            double window = 0.5 * (1.0 - std::cos(2.0 * pi * win_pos));
            double w = sinc_val * window;
            acc += static_cast<double>(samples[j]) * w;
            weight_sum += w;
        }

        resampled[i] = (weight_sum > 1e-12)
            ? static_cast<float>(acc / weight_sum)
            : 0.0F;
    }
    return resampled;
}

} // namespace trtf
