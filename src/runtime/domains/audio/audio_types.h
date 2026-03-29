#pragma once

// Shared types for old-style audio backends (Magpie, Speech, Omni).
// Extracted from bark_backend.h during Whisper/Bark TrtModule migration.

#include <cstdint>
#include <string>
#include <vector>

namespace trtf {

struct LegacyAudioResult {
    std::vector<float> waveform;  // [num_samples] float32
    int32_t sample_rate{24000};
    int32_t num_samples{0};
};

// Write a WAV file with 44-byte RIFF header + IEEE float32 PCM.
bool write_wav(const std::string& path, const float* samples,
               int32_t num_samples, int32_t sample_rate);

} // namespace trtf
