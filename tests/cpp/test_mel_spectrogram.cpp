// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-08
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-01
// Intent:         Mel spectrogram extraction: output shape, energy concentration, normalization
// Preconditions:  Synthetic sine wave and filterbank data
// Postconditions: Spectrogram shape matches expected, energy concentrated in correct bins
// =============================================================================

// Test suite: Mel spectrogram extraction.
//
// Purpose:
//   Validates extract_mel_spectrogram() from runtime/trt/audio/mel_spectrogram.h.
//   Uses a synthetic sine wave and a known filterbank to verify output shape,
//   energy concentration, and normalization properties.
//
// Dependencies:
//   - runtime/trt/audio/mel_spectrogram.h: extract_mel_spectrogram
//   - No TRT, GPU, or CUDA required.

#include "runtime/trt/audio/mel_spectrogram.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

// Create a simple identity-ish filterbank for testing.
// Each mel bin m maps to frequency bin m (1:1), with value 1.0.
// Layout: [n_freq_bins, n_mel_bins] row-major
static std::vector<float> make_identity_filterbank(int32_t n_freq_bins, int32_t n_mel_bins)
{
    std::vector<float> fb(static_cast<std::size_t>(n_freq_bins) * n_mel_bins, 0.0F);
    const int32_t mapped = std::min(n_freq_bins, n_mel_bins);
    for (int32_t i = 0; i < mapped; ++i)
    {
        fb[static_cast<std::size_t>(i) * n_mel_bins + i] = 1.0F;
    }
    return fb;
}

// Create a triangular mel filterbank (more realistic).
// Each mel bin gets a triangular filter spanning a few frequency bins.
static std::vector<float> make_triangular_filterbank(int32_t n_freq_bins, int32_t n_mel_bins)
{
    std::vector<float> fb(static_cast<std::size_t>(n_freq_bins) * n_mel_bins, 0.0F);
    const float step = static_cast<float>(n_freq_bins) / static_cast<float>(n_mel_bins + 1);
    for (int32_t m = 0; m < n_mel_bins; ++m)
    {
        float center = (m + 1) * step;
        float left = center - step;
        float right = center + step;
        for (int32_t f = 0; f < n_freq_bins; ++f)
        {
            float ff = static_cast<float>(f);
            if (ff >= left && ff <= center)
            {
                fb[static_cast<std::size_t>(f) * n_mel_bins + m] =
                    (ff - left) / (center - left);
            }
            else if (ff > center && ff <= right)
            {
                fb[static_cast<std::size_t>(f) * n_mel_bins + m] =
                    (right - ff) / (right - center);
            }
        }
    }
    return fb;
}

int main()
{
    const int32_t sample_rate = 16000;
    const int32_t n_fft = 400;
    const int32_t hop_length = 160;
    const int32_t chunk_length_s = 30;
    const int32_t n_freq_bins = n_fft / 2 + 1;  // 201
    const int32_t n_mel_bins = 80;

    // Test 1: Output shape with identity filterbank and 1s of silence
    {
        const int32_t n_samples = sample_rate;  // 1 second
        std::vector<float> silence(n_samples, 0.0F);
        auto fb = make_identity_filterbank(n_freq_bins, n_mel_bins);

        auto mel = trtf::extract_mel_spectrogram(
            silence.data(), n_samples,
            fb.data(), n_freq_bins, n_mel_bins,
            n_fft, hop_length, chunk_length_s, sample_rate);

        // Expected: padded to 30s = 480000 samples, then center-padded with n_fft/2=200 on each side
        // center_padded_length = 200 + 480000 + 200 = 480400
        // n_frames_raw = 1 + (480400 - 400) / 160 = 1 + 3000 = 3001
        // n_frames_out = 3001 - 1 = 3000 (HF trims last frame)
        check(mel.n_mels == n_mel_bins, "shape_silence: n_mels");
        check(mel.n_frames == 3000, "shape_silence: n_frames == 3000");
        check(static_cast<int32_t>(mel.data.size()) == n_mel_bins * mel.n_frames,
              "shape_silence: data size matches");
    }

    // Test 2: 440Hz sine wave has energy in expected frequency bin
    {
        const int32_t n_samples = sample_rate;  // 1 second at 16kHz
        std::vector<float> sine(n_samples);
        const double pi2 = 2.0 * 3.14159265358979323846;
        for (int32_t i = 0; i < n_samples; ++i)
        {
            sine[i] = static_cast<float>(
                std::sin(pi2 * 440.0 * static_cast<double>(i) / sample_rate));
        }

        auto fb = make_identity_filterbank(n_freq_bins, n_mel_bins);
        auto mel = trtf::extract_mel_spectrogram(
            sine.data(), n_samples,
            fb.data(), n_freq_bins, n_mel_bins,
            n_fft, hop_length, chunk_length_s, sample_rate);

        check(mel.n_mels == n_mel_bins, "sine440: n_mels");
        check(mel.n_frames > 0, "sine440: n_frames > 0");

        // 440Hz should concentrate energy around bin 440 * n_fft / sample_rate = 440 * 400 / 16000 = 11
        // With identity filterbank, mel bin 11 should have more energy than mel bin 50
        // Check first few frames (where the sine wave is active)
        const int32_t target_bin = 11;
        const int32_t quiet_bin = 50;
        float energy_target = 0.0F;
        float energy_quiet = 0.0F;
        const int32_t check_frames = std::min(mel.n_frames, 50);
        for (int32_t t = 0; t < check_frames; ++t)
        {
            energy_target += mel.data[static_cast<std::size_t>(target_bin) * mel.n_frames + t];
            energy_quiet += mel.data[static_cast<std::size_t>(quiet_bin) * mel.n_frames + t];
        }
        check(energy_target > energy_quiet,
              "sine440: energy at 440Hz bin > energy at quiet bin");
    }

    // Test 3: Normalization produces values in reasonable range
    {
        const int32_t n_samples = sample_rate;
        std::vector<float> tone(n_samples);
        const double pi2 = 2.0 * 3.14159265358979323846;
        for (int32_t i = 0; i < n_samples; ++i)
        {
            tone[i] = 0.5F * static_cast<float>(
                std::sin(pi2 * 1000.0 * static_cast<double>(i) / sample_rate));
        }

        auto fb = make_triangular_filterbank(n_freq_bins, n_mel_bins);
        auto mel = trtf::extract_mel_spectrogram(
            tone.data(), n_samples,
            fb.data(), n_freq_bins, n_mel_bins,
            n_fft, hop_length, chunk_length_s, sample_rate);

        // After log + normalize: values should be roughly in [-1, 1] range
        // HF normalization: (log10(max(x, 1e-10)) + 4.0) / 4.0
        // For silence: log10(1e-10) = -10, then max(-10, max-8), then (-10+4)/4 = -1.5
        // For loud: log10(big) -> positive, (positive+4)/4 -> around 1-2
        float min_val = 1e10F, max_val = -1e10F;
        for (const auto v : mel.data)
        {
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }
        check(min_val > -3.0F, "normalize: min_val > -3");
        check(max_val < 3.0F, "normalize: max_val < 3");
        check(max_val > min_val, "normalize: max_val > min_val (dynamic range)");
    }

    // Test 4: Empty audio (0 samples) produces valid output
    {
        auto fb = make_identity_filterbank(n_freq_bins, n_mel_bins);
        auto mel = trtf::extract_mel_spectrogram(
            nullptr, 0,
            fb.data(), n_freq_bins, n_mel_bins,
            n_fft, hop_length, chunk_length_s, sample_rate);

        // Should still produce output (padded to 30s of silence)
        check(mel.n_mels == n_mel_bins, "empty: n_mels");
        check(mel.n_frames > 0, "empty: n_frames > 0");
    }

    // Test 5: Short audio (< n_fft) still works
    {
        std::vector<float> short_audio = {0.1F, 0.2F, 0.3F};
        auto fb = make_identity_filterbank(n_freq_bins, n_mel_bins);
        auto mel = trtf::extract_mel_spectrogram(
            short_audio.data(), 3,
            fb.data(), n_freq_bins, n_mel_bins,
            n_fft, hop_length, chunk_length_s, sample_rate);

        check(mel.n_mels == n_mel_bins, "short: n_mels");
        check(mel.n_frames > 0, "short: n_frames > 0");
    }

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
    }
    return failures;
}
