#include "runtime/domains/audio/audio_bundle_validation.h"

#include <stdexcept>

namespace trtf::runtime::builders::audio {

namespace {

void require_section(const trtf::BundleFile& bundle, const std::string& name,
                     const std::string& bundle_path)
{
    auto* data = trtf::find_section(bundle, name);
    if (!data || data->empty())
        throw std::runtime_error(bundle_path + ": missing " + name + " section");
}

} // namespace

static void validate_bark(const trtf::BundleFile& bundle, const std::string& bundle_path)
{
    require_section(bundle, "semantic_embed", bundle_path);
    require_section(bundle, "coarse_embed", bundle_path);
    require_section(bundle, "coarse_engine_plan", bundle_path);
}

static void validate_magpie(const trtf::BundleFile& bundle, const std::string& bundle_path)
{
    require_section(bundle, "magpie_audio_embed", bundle_path);
    require_section(bundle, "magpie_text_embed", bundle_path);
    require_section(bundle, "magpie_context_embed", bundle_path);
    require_section(bundle, "magpie_ipa_phoneme_dict", bundle_path);
    require_section(bundle, "magpie_ipa_vocab", bundle_path);
}

void validate_text_to_audio_bundle_sections(
    TextToAudioBundleKind kind,
    const trtf::BundleFile& bundle,
    const std::string& bundle_path)
{
    switch (kind)
    {
    case TextToAudioBundleKind::kBark:
        validate_bark(bundle, bundle_path);
        break;
    case TextToAudioBundleKind::kMagpieTts:
        validate_magpie(bundle, bundle_path);
        break;
    }
}

} // namespace trtf::runtime::builders::audio
