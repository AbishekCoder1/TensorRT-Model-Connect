// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-18
// Architecture:   ARCH-AUD-001
// Unit Design:    UD-AUD-01
// Intent:         write_wav: IEEE float32 WAV round-trip and error path
// Preconditions:  Writable temp directory
// Postconditions: Written WAV is readable and matches input; bad-path returns false
// =============================================================================

// test_audio_types.cpp — Unit tests for src/runtime/domains/audio/audio_types.cpp
//
// Purpose:
//   Validates write_wav() from audio_types.h: the function that serialises a
//   raw float32 sample array to a RIFF/WAVE file.  Tests use read_wav() from
//   utils/wav_reader.h to verify the written bytes round-trip correctly.
//
// Dependencies:
//   - runtime/domains/audio/audio_types.h : write_wav
//   - utils/wav_reader.h              : read_wav (for verification)
//   - test_helpers.h                  : TempDirGuard
//   No TRT, GPU, or CUDA required.

#include "runtime/domains/audio/audio_types.h"
#include "utils/wav_reader.h"
#include "test_helpers.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
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

// ---------------------------------------------------------------------------
// write_wav round-trip: write float32 samples, read back, verify
// ---------------------------------------------------------------------------

// Intention: write_wav writes a valid IEEE float32 WAV that read_wav can
//            decode, recovering the original sample values and sample rate.
// Preconditions:  temp dir exists and is writable
// Postconditions: decoded samples match input within float32 precision
static bool test_write_wav_roundtrip()
{
    trtmc_test::TempDirGuard dir;
    const auto path = (std::filesystem::path(dir.path()) / "out.wav").string();

    const std::vector<float> samples = {0.0F, 0.5F, 1.0F, -1.0F, -0.5F, 0.25F};
    const int32_t sample_rate = 22050;

    if (!trtmc::write_wav(path, samples.data(),
                         static_cast<int32_t>(samples.size()), sample_rate))
    {
        std::cerr << "write_wav_roundtrip: write_wav returned false\n";
        return false;
    }

    const auto wav = trtmc::read_wav(path);
    if (wav.sample_rate != sample_rate)
    {
        std::cerr << "write_wav_roundtrip: sample_rate mismatch "
                  << wav.sample_rate << " vs " << sample_rate << '\n';
        return false;
    }
    if (wav.samples.size() != samples.size())
    {
        std::cerr << "write_wav_roundtrip: sample count mismatch "
                  << wav.samples.size() << " vs " << samples.size() << '\n';
        return false;
    }
    for (std::size_t i = 0; i < samples.size(); ++i)
    {
        if (std::abs(wav.samples[i] - samples[i]) > 1e-6F)
        {
            std::cerr << "write_wav_roundtrip: sample[" << i << "] mismatch "
                      << wav.samples[i] << " vs " << samples[i] << '\n';
            return false;
        }
    }
    return true;
}

// Intention: write_wav with a different sample rate writes the correct rate
//            into the WAV header so read_wav recovers it.
// Preconditions:  temp dir writable
// Postconditions: returned sample_rate equals the written rate
static bool test_write_wav_sample_rate_preserved()
{
    trtmc_test::TempDirGuard dir;
    const auto path = (std::filesystem::path(dir.path()) / "rate.wav").string();

    const std::vector<float> samples = {0.1F, 0.2F, 0.3F};
    const int32_t sample_rate = 44100;

    if (!trtmc::write_wav(path, samples.data(),
                         static_cast<int32_t>(samples.size()), sample_rate))
    {
        return false;
    }

    const auto wav = trtmc::read_wav(path);
    return wav.sample_rate == sample_rate;
}

// Intention: write_wav with a single sample still produces a valid WAV.
// Preconditions:  temp dir writable
// Postconditions: decoded result has exactly one sample matching the input
static bool test_write_wav_single_sample()
{
    trtmc_test::TempDirGuard dir;
    const auto path = (std::filesystem::path(dir.path()) / "single.wav").string();

    const float sample = 0.75F;
    if (!trtmc::write_wav(path, &sample, 1, 16000))
    {
        return false;
    }

    const auto wav = trtmc::read_wav(path);
    return wav.samples.size() == 1 &&
           std::abs(wav.samples[0] - sample) < 1e-6F;
}

// Intention: write_wav returns false when the output path is not writable
//            (directory does not exist).
// Preconditions:  "/nonexistent/dir/" does not exist on the filesystem
// Postconditions: write_wav returns false without throwing
static bool test_write_wav_bad_path_returns_false()
{
    const std::vector<float> samples = {0.0F};
    return !trtmc::write_wav("/nonexistent/dir/out.wav",
                            samples.data(), 1, 16000);
}

int main()
{
    bool all_passed = true;
    std::cout << "test_audio_types:" << std::endl;

    const auto run = [&](const char* name, bool (*fn)()) {
        const bool ok = fn();
        std::cout << "  " << name << ": " << (ok ? "PASS" : "FAIL") << '\n';
        all_passed &= ok;
    };

    run("write_wav_roundtrip",             test_write_wav_roundtrip);
    run("write_wav_sample_rate_preserved", test_write_wav_sample_rate_preserved);
    run("write_wav_single_sample",         test_write_wav_single_sample);
    run("write_wav_bad_path_returns_false",test_write_wav_bad_path_returns_false);

    if (all_passed)
    {
        std::cout << "test_audio_types passed" << std::endl;
        return 0;
    }
    std::cerr << "test_audio_types FAILED" << std::endl;
    return 1;
}
