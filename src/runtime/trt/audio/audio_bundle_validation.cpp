#include "runtime/trt/audio/audio_bundle_validation.h"

#include <stdexcept>

namespace trtf::runtime::builders::audio {

namespace {

void require_section(const std::vector<char>* data, const std::string& name,
                     const std::string& bundle_path)
{
    if (!data || data->empty())
        throw std::runtime_error(bundle_path + ": missing " + name + " section");
}

} // namespace

static void validate_bark(const trtf::BundleSections& sections, const std::string& bundle_path)
{
    require_section(sections.semantic_embed_data, "semantic_embed", bundle_path);
    require_section(sections.coarse_embed_data, "coarse_embed", bundle_path);
    require_section(sections.coarse_engine_plan_data, "coarse_engine_plan", bundle_path);
}

static void validate_magpie(const trtf::BundleSections& sections, const std::string& bundle_path)
{
    require_section(sections.magpie_audio_embed_data, "magpie_audio_embed", bundle_path);
    require_section(sections.magpie_text_embed_data, "magpie_text_embed", bundle_path);
    require_section(sections.magpie_context_embed_data, "magpie_context_embed", bundle_path);
    require_section(sections.magpie_ipa_phoneme_dict_data, "magpie_ipa_phoneme_dict", bundle_path);
    require_section(sections.magpie_ipa_vocab_data, "magpie_ipa_vocab", bundle_path);
}

void validate_text_to_audio_bundle_sections(
    TextToAudioBundleKind kind,
    const trtf::BundleSections& sections,
    const std::string& bundle_path)
{
    switch (kind)
    {
    case TextToAudioBundleKind::kBark:
        validate_bark(sections, bundle_path);
        break;
    case TextToAudioBundleKind::kMagpieTts:
        validate_magpie(sections, bundle_path);
        break;
    }
}

} // namespace trtf::runtime::builders::audio
