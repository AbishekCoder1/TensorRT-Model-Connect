#include "runtime/trt/audio/audio_bundle_validation.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

std::vector<char> bytes_from_text(const std::string& text)
{
    return std::vector<char>(text.begin(), text.end());
}

void test_bark_validation_requires_semantic_and_coarse_assets()
{
    trtf::BundleSections sections;
    sections.semantic_embed_data = nullptr;
    sections.coarse_embed_data = nullptr;
    sections.coarse_engine_plan_data = nullptr;

    try
    {
        trtf::runtime::builders::audio::validate_text_to_audio_bundle_sections(
            trtf::runtime::builders::audio::TextToAudioBundleKind::kBark,
            sections,
            "bark.trtfb");
        check(false, "bark validation rejects missing semantic/coarse sections");
    }
    catch (const std::runtime_error& error)
    {
        check(
            std::string(error.what()).find("semantic_embed") != std::string::npos,
            "bark validation reports semantic_embed");
    }
}

void test_magpie_validation_requires_ipa_tokenizer_sections()
{
    trtf::BundleSections sections;
    const auto vision = bytes_from_text("vision");
    const auto audio = bytes_from_text("audio");
    const auto text = bytes_from_text("text");
    const auto context = bytes_from_text("context");
    sections.vision_plan_data = &vision;
    sections.magpie_audio_embed_data = &audio;
    sections.magpie_text_embed_data = &text;
    sections.magpie_context_embed_data = &context;

    try
    {
        trtf::runtime::builders::audio::validate_text_to_audio_bundle_sections(
            trtf::runtime::builders::audio::TextToAudioBundleKind::kMagpieTts,
            sections,
            "magpie.trtfb");
        check(false, "magpie validation rejects missing IPA tokenizer sections");
    }
    catch (const std::runtime_error& error)
    {
        check(
            std::string(error.what()).find("magpie_ipa") != std::string::npos,
            "magpie validation reports IPA tokenizer sections");
    }
}

void test_magpie_validation_accepts_complete_required_sections()
{
    trtf::BundleSections sections;
    const auto vision = bytes_from_text("vision");
    const auto audio = bytes_from_text("audio");
    const auto text = bytes_from_text("text");
    const auto context = bytes_from_text("context");
    const auto phoneme_dict = bytes_from_text("dict");
    const auto vocab = bytes_from_text("vocab");

    sections.vision_plan_data = &vision;
    sections.magpie_audio_embed_data = &audio;
    sections.magpie_text_embed_data = &text;
    sections.magpie_context_embed_data = &context;
    sections.magpie_ipa_phoneme_dict_data = &phoneme_dict;
    sections.magpie_ipa_vocab_data = &vocab;

    try
    {
        trtf::runtime::builders::audio::validate_text_to_audio_bundle_sections(
            trtf::runtime::builders::audio::TextToAudioBundleKind::kMagpieTts,
            sections,
            "magpie.trtfb");
        check(true, "magpie validation accepts complete section set");
    }
    catch (const std::exception&)
    {
        check(false, "magpie validation accepts complete section set");
    }
}

} // namespace

int main()
{
    test_bark_validation_requires_semantic_and_coarse_assets();
    test_magpie_validation_requires_ipa_tokenizer_sections();
    test_magpie_validation_accepts_complete_required_sections();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }

    std::cerr << "All audio bundle validation tests passed.\n";
    return 0;
}
