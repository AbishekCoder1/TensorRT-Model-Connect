#pragma once

#include "cabi/bundle/bundle_helpers.h"

#include <string>

namespace trtf::runtime::builders::audio {

enum class TextToAudioBundleKind {
    kBark,
    kMagpieTts,
};

void validate_text_to_audio_bundle_sections(
    TextToAudioBundleKind kind,
    const trtf::BundleSections& sections,
    const std::string& bundle_path);

} // namespace trtf::runtime::builders::audio
