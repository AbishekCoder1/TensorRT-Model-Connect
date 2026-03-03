#include "runtime/trt/mel_spectrogram.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace trtf {

namespace {

// Periodic Hann window: w[n] = 0.5 * (1 - cos(2*pi*n / N))
// Matches np.hanning(N+1)[:-1] used by HF WhisperFeatureExtractor.
std::vector<float> make_hann_window(int32_t length)
{
    std::vector<float> window(length);
    const double pi2 = 2.0 * 3.14159265358979323846;
    for (int32_t i = 0; i < length; ++i)
    {
        window[i] = static_cast<float>(
            0.5 * (1.0 - std::cos(pi2 * static_cast<double>(i) /
                                   static_cast<double>(length))));
    }
    return window;
}

// Direct real-to-complex DFT for the first n_out bins.
// Computes X[k] = sum_{n=0}^{N-1} x[n] * e^{-j*2*pi*k*n/N} for k = 0..n_out-1.
// Uses double precision for twiddle factors to match numpy's FFT accuracy.
// Writes squared magnitude |X[k]|^2 directly into power_out.
void rfft_power_direct(const float* x, int32_t n, int32_t n_out, float* power_out)
{
    const double pi2 = 2.0 * 3.14159265358979323846;
    for (int32_t k = 0; k < n_out; ++k)
    {
        double re = 0.0, im = 0.0;
        const double w = pi2 * static_cast<double>(k) / static_cast<double>(n);
        for (int32_t t = 0; t < n; ++t)
        {
            const double angle = w * static_cast<double>(t);
            re += static_cast<double>(x[t]) * std::cos(angle);
            im -= static_cast<double>(x[t]) * std::sin(angle);
        }
        power_out[k] = static_cast<float>(re * re + im * im);
    }
}

} // anonymous namespace

MelResult extract_mel_spectrogram(
    const float* samples, int32_t n_samples,
    const float* mel_filters, int32_t n_freq_bins, int32_t n_mel_bins,
    int32_t n_fft, int32_t hop_length,
    int32_t chunk_length_s, int32_t sample_rate)
{
    // Step 1: Pad audio to chunk_length_s * sample_rate samples
    const int32_t audio_length = chunk_length_s * sample_rate;
    std::vector<float> audio_padded(audio_length, 0.0F);
    const int32_t copy_len = std::min(n_samples, audio_length);
    if (copy_len > 0)
        std::memcpy(audio_padded.data(), samples, copy_len * sizeof(float));

    // Step 1b: Center-pad with n_fft/2 zeros on each side (matches HF WhisperFeatureExtractor)
    const int32_t pad_size = n_fft / 2;
    const int32_t padded_length = pad_size + audio_length + pad_size;
    std::vector<float> padded(padded_length, 0.0F);
    std::memcpy(padded.data() + pad_size, audio_padded.data(),
                audio_length * sizeof(float));

    // Step 2: STFT — exact n_fft-point DFT (matches np.fft.rfft)
    const int32_t expected_freq_bins = n_fft / 2 + 1;
    if (n_freq_bins != expected_freq_bins)
        n_freq_bins = expected_freq_bins;

    // Number of frames after center padding
    const int32_t n_frames_raw = 1 + (padded_length - n_fft) / hop_length;

    auto window = make_hann_window(n_fft);

    // Power spectrum: [n_freq_bins, n_frames_raw]
    std::vector<float> power(
        static_cast<std::size_t>(n_freq_bins) * n_frames_raw, 0.0F);

    std::vector<float> windowed(n_fft);
    std::vector<float> frame_power(n_freq_bins);

    for (int32_t t = 0; t < n_frames_raw; ++t)
    {
        const int32_t start = t * hop_length;

        // Apply window
        for (int32_t i = 0; i < n_fft; ++i)
            windowed[i] = padded[start + i] * window[i];

        // Direct DFT: compute |X[k]|^2 for k = 0..n_freq_bins-1
        rfft_power_direct(windowed.data(), n_fft, n_freq_bins, frame_power.data());

        // Store in column-major layout [freq_bin, frame]
        for (int32_t f = 0; f < n_freq_bins; ++f)
        {
            power[static_cast<std::size_t>(f) * n_frames_raw + t] = frame_power[f];
        }
    }

    // Step 3: Mel projection
    // mel_filters layout: [n_freq_bins, n_mel_bins] (row = freq bin, col = mel bin)
    // mel_spec[m, t] = sum_f(power[f, t] * mel_filters[f, m])
    std::vector<float> mel_spec(
        static_cast<std::size_t>(n_mel_bins) * n_frames_raw, 0.0F);

    for (int32_t t = 0; t < n_frames_raw; ++t)
    {
        for (int32_t f = 0; f < n_freq_bins; ++f)
        {
            const float p = power[static_cast<std::size_t>(f) * n_frames_raw + t];
            if (p == 0.0F) continue;
            for (int32_t m = 0; m < n_mel_bins; ++m)
            {
                mel_spec[static_cast<std::size_t>(m) * n_frames_raw + t] +=
                    p * mel_filters[static_cast<std::size_t>(f) * n_mel_bins + m];
            }
        }
    }

    // Step 4: Log + normalize (matches HF WhisperFeatureExtractor)
    // log10(max(mel, 1e-10)), then max(result, global_max - 8.0), then (result + 4.0) / 4.0
    const auto total = static_cast<std::size_t>(n_mel_bins) * n_frames_raw;
    float global_max = -1e10F;
    for (std::size_t i = 0; i < total; ++i)
    {
        mel_spec[i] = std::log10(std::max(mel_spec[i], 1e-10F));
        if (mel_spec[i] > global_max)
            global_max = mel_spec[i];
    }

    const float floor = global_max - 8.0F;
    for (std::size_t i = 0; i < total; ++i)
    {
        mel_spec[i] = std::max(mel_spec[i], floor);
        mel_spec[i] = (mel_spec[i] + 4.0F) / 4.0F;
    }

    // Step 5: Trim last frame (HF convention: output has n_frames_raw - 1 frames)
    // But only if we have more than 1 frame
    int32_t n_frames_out = n_frames_raw;
    if (n_frames_raw > 1)
    {
        n_frames_out = n_frames_raw - 1;
        // Re-pack by removing last column from each mel row
        std::vector<float> trimmed(
            static_cast<std::size_t>(n_mel_bins) * n_frames_out);
        for (int32_t m = 0; m < n_mel_bins; ++m)
        {
            std::memcpy(
                trimmed.data() + static_cast<std::size_t>(m) * n_frames_out,
                mel_spec.data() + static_cast<std::size_t>(m) * n_frames_raw,
                n_frames_out * sizeof(float));
        }
        mel_spec = std::move(trimmed);
    }

    MelResult result;
    result.data = std::move(mel_spec);
    result.n_mels = n_mel_bins;
    result.n_frames = n_frames_out;
    return result;
}

} // namespace trtf
