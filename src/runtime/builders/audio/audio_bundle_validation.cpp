#include "runtime/builders/audio/audio_bundle_validation.h"

#include <array>
#include <stdexcept>
#include <vector>

namespace trtf::runtime::builders::audio {
namespace {

using BundleSectionPtr = const std::vector<char>* trtf::BundleSections::*;

struct RequiredTextToAudioSectionRule {
    std::array<BundleSectionPtr, 2> sections;
    std::size_t section_count;
    const char* missing_error;
    bool append_bundle_path;
};

constexpr RequiredTextToAudioSectionRule kBarkRequiredSectionRules[] = {
    {{{&trtf::BundleSections::semantic_embed_data, nullptr}}, 1,
        "Bundle missing semantic_embed section: ", true},
    {{{&trtf::BundleSections::coarse_embed_data, nullptr}}, 1,
        "Bundle missing coarse_embed section: ", true},
    {{{&trtf::BundleSections::coarse_engine_plan_data, nullptr}}, 1,
        "Bundle missing coarse_engine section: ", true},
};

constexpr RequiredTextToAudioSectionRule kMagpieRequiredSectionRules[] = {
    {{{&trtf::BundleSections::vision_plan_data, nullptr}}, 1,
        "Bundle missing encoder engine (vision_engine_plan) for MagpieTTS: ", true},
    {{{&trtf::BundleSections::magpie_audio_embed_data, nullptr}}, 1,
        "Bundle missing magpie_audio_embed section: ", true},
    {{{&trtf::BundleSections::magpie_text_embed_data, nullptr}}, 1,
        "Bundle missing magpie_text_embed section: ", true},
    {{{&trtf::BundleSections::magpie_context_embed_data, nullptr}}, 1,
        "Bundle missing magpie_context_embed section: ", true},
    {{{&trtf::BundleSections::magpie_ipa_phoneme_dict_data,
       &trtf::BundleSections::magpie_ipa_vocab_data}},
        2,
        "Bundle missing IPA tokenizer sections (magpie_ipa_phoneme_dict, "
        "magpie_ipa_vocab). Rebuild the bundle with the latest trtf-build.",
        false},
};

bool has_data(const std::vector<char>* data)
{
    return data != nullptr && !data->empty();
}

bool required_text_to_audio_sections_present(
    const trtf::BundleSections& sections,
    const RequiredTextToAudioSectionRule& rule)
{
    for (std::size_t index = 0; index < rule.section_count; ++index)
    {
        if (!has_data(sections.*rule.sections[index]))
        {
            return false;
        }
    }
    return true;
}

[[noreturn]] void throw_missing_text_to_audio_section(
    const RequiredTextToAudioSectionRule& rule,
    const std::string& bundle_path)
{
    if (rule.append_bundle_path)
    {
        throw std::runtime_error(std::string(rule.missing_error) + bundle_path);
    }
    throw std::runtime_error(rule.missing_error);
}

const RequiredTextToAudioSectionRule* required_text_to_audio_section_rules(
    TextToAudioBundleKind kind,
    std::size_t& rule_count)
{
    switch (kind)
    {
    case TextToAudioBundleKind::kBark:
        rule_count = sizeof(kBarkRequiredSectionRules) / sizeof(kBarkRequiredSectionRules[0]);
        return kBarkRequiredSectionRules;
    case TextToAudioBundleKind::kMagpieTts:
        rule_count = sizeof(kMagpieRequiredSectionRules) / sizeof(kMagpieRequiredSectionRules[0]);
        return kMagpieRequiredSectionRules;
    }

    rule_count = 0;
    return nullptr;
}

} // namespace

void validate_text_to_audio_bundle_sections(
    TextToAudioBundleKind kind,
    const trtf::BundleSections& sections,
    const std::string& bundle_path)
{
    std::size_t rule_count = 0;
    const auto* rules = required_text_to_audio_section_rules(kind, rule_count);
    for (std::size_t index = 0; index < rule_count; ++index)
    {
        if (!required_text_to_audio_sections_present(sections, rules[index]))
        {
            throw_missing_text_to_audio_section(rules[index], bundle_path);
        }
    }
}

} // namespace trtf::runtime::builders::audio
