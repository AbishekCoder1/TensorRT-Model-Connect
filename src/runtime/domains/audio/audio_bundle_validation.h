#pragma once

#include "bundle/bundle_format.h"
#include "bundle/bundle_view.h"

#include <string>

namespace trtf::runtime::builders::audio {

enum class TextToAudioBundleKind {
    kBark,
    kMagpieTts,
};

void validate_text_to_audio_bundle_sections(
    TextToAudioBundleKind kind,
    const trtf::BundleFile& bundle,
    const std::string& bundle_path);

} // namespace trtf::runtime::builders::audio
