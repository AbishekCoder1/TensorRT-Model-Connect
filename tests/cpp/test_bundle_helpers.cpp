// =============================================================================
// test_bundle_helpers.cpp — Unit tests for bundle_helpers functions
// =============================================================================
//
// Purpose:
//   Validates the find_bundle_sections() function from bundle_helpers.h/cpp,
//   which scans a BundleFile's sections and populates a BundleSections struct
//   with pointers by name. When TRTF_HAS_TRT is disabled, tests skip
//   gracefully (exit 0).
//
// Dependencies:
//   - cabi/bundle/bundle_helpers.h (find_bundle_sections, BundleSections)
//   - bundle/bundle_format.h (BundleFile, BundleSection)
//
// Environment:
//   CPU-only. No GPU, CUDA, or TRT runtime required.
//   The function itself only iterates sections and assigns pointers.
// =============================================================================

#include "cabi/bundle/bundle_helpers.h"
#include "bundle/bundle_format.h"

#include <iostream>
#include <string>
#include <vector>

#if TRTF_HAS_TRT

static int failures = 0;

static void check(bool condition, const char* test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static trtf::BundleFile make_test_bundle()
{
    trtf::BundleFile bundle;
    bundle.info.model_id = "test-model";

    // Add sections with known names and dummy data
    bundle.sections.push_back({"engine_plan", {'P', 'L', 'A', 'N'}});
    bundle.sections.push_back({"config.json", {'{', '}'}});
    bundle.sections.push_back({"tokenizer.json", {'T', 'O', 'K'}});
    bundle.sections.push_back({"tokenizer_config.json", {'T', 'C'}});
    bundle.sections.push_back({"vocab.json", {'V'}});
    bundle.sections.push_back({"merges.txt", {'M'}});
    bundle.sections.push_back({"special_tokens_map.json", {'S'}});
    bundle.sections.push_back({"tokenizer.model", {'T', 'M'}});
    bundle.sections.push_back({"vision_engine_plan", {'V', 'E'}});
    bundle.sections.push_back({"preprocessor_config.json", {'P', 'C'}});

    return bundle;
}

static void test_find_sections_engine_plan()
{
    const auto bundle = make_test_bundle();
    const auto sections = trtf::find_bundle_sections(bundle);
    check(sections.plan_data != nullptr, "find_sections engine_plan not null");
    check(sections.plan_data->size() == 4, "find_sections engine_plan size");
    check((*sections.plan_data)[0] == 'P', "find_sections engine_plan data[0]");
}

static void test_find_sections_config_json()
{
    const auto bundle = make_test_bundle();
    const auto sections = trtf::find_bundle_sections(bundle);
    check(sections.config_json_data != nullptr, "find_sections config.json not null");
    check(sections.config_json_data->size() == 2, "find_sections config.json size");
}

static void test_find_sections_tokenizer()
{
    const auto bundle = make_test_bundle();
    const auto sections = trtf::find_bundle_sections(bundle);
    check(sections.tokenizer_json_data != nullptr, "find_sections tokenizer.json not null");
    check(sections.tokenizer_config_data != nullptr, "find_sections tokenizer_config.json not null");
    check(sections.vocab_json_data != nullptr, "find_sections vocab.json not null");
    check(sections.merges_txt_data != nullptr, "find_sections merges.txt not null");
    check(sections.special_tokens_data != nullptr, "find_sections special_tokens not null");
    check(sections.tokenizer_model_data != nullptr, "find_sections tokenizer.model not null");
}

static void test_find_sections_vision()
{
    const auto bundle = make_test_bundle();
    const auto sections = trtf::find_bundle_sections(bundle);
    check(sections.vision_plan_data != nullptr, "find_sections vision_engine_plan not null");
    check(sections.preprocessor_config_data != nullptr, "find_sections preprocessor_config not null");
}

static void test_find_sections_empty_bundle()
{
    trtf::BundleFile empty;
    const auto sections = trtf::find_bundle_sections(empty);
    check(sections.plan_data == nullptr, "find_sections empty: plan null");
    check(sections.config_json_data == nullptr, "find_sections empty: config null");
    check(sections.tokenizer_json_data == nullptr, "find_sections empty: tokenizer null");
    check(sections.vision_plan_data == nullptr, "find_sections empty: vision null");
}

static void test_find_sections_unknown_ignored()
{
    trtf::BundleFile bundle;
    bundle.sections.push_back({"engine_plan", {'P'}});
    bundle.sections.push_back({"some_unknown_section", {'X'}});

    const auto sections = trtf::find_bundle_sections(bundle);
    check(sections.plan_data != nullptr, "find_sections unknown: plan not null");
    check(sections.config_json_data == nullptr, "find_sections unknown: config null");
}

static void test_find_sections_bark_sections()
{
    trtf::BundleFile bundle;
    bundle.sections.push_back({"coarse_engine_plan", {'C'}});
    bundle.sections.push_back({"fine_engine_plan", {'F'}});
    bundle.sections.push_back({"codec_engine_plan", {'K'}});
    bundle.sections.push_back({"semantic_embed", {'S', 'E'}});
    bundle.sections.push_back({"coarse_embed", {'C', 'E'}});
    bundle.sections.push_back({"fine_embed", {'F', 'E'}});
    bundle.sections.push_back({"fine_position_embed", {'F', 'P'}});

    const auto sections = trtf::find_bundle_sections(bundle);
    check(sections.coarse_engine_plan_data != nullptr, "find_sections bark: coarse not null");
    check(sections.fine_engine_plan_data != nullptr, "find_sections bark: fine not null");
    check(sections.codec_engine_plan_data != nullptr, "find_sections bark: codec not null");
    check(sections.semantic_embed_data != nullptr, "find_sections bark: semantic_embed not null");
    check(sections.coarse_embed_data != nullptr, "find_sections bark: coarse_embed not null");
    check(sections.fine_embed_data != nullptr, "find_sections bark: fine_embed not null");
    check(sections.fine_position_embed_data != nullptr, "find_sections bark: fine_pos_embed not null");
}

static void test_find_sections_diffusion()
{
    trtf::BundleFile bundle;
    bundle.sections.push_back({"text_encoder_0_plan", {'T', '0'}});
    bundle.sections.push_back({"text_encoder_1_plan", {'T', '1'}});
    bundle.sections.push_back({"denoiser_plan", {'D'}});
    bundle.sections.push_back({"vae_decoder_plan", {'V'}});
    bundle.sections.push_back({"preprocessor_weights", {'W'}});

    const auto sections = trtf::find_bundle_sections(bundle);
    check(sections.text_encoder_plans.size() == 2, "find_sections diffusion: 2 text encoders");
    check(sections.denoiser_plan_data != nullptr, "find_sections diffusion: denoiser not null");
    check(sections.vae_decoder_plan_data != nullptr, "find_sections diffusion: vae not null");
    check(sections.preprocessor_weights_data != nullptr, "find_sections diffusion: preproc weights not null");
}

static void test_find_sections_speech()
{
    trtf::BundleFile bundle;
    bundle.sections.push_back({"engine_plan", {'P'}});
    bundle.sections.push_back({"depth_engine_plan", {'D', 'E'}});
    bundle.sections.push_back({"mimi_encoder_plan", {'M', 'E'}});
    bundle.sections.push_back({"mimi_decoder_plan", {'M', 'D'}});

    const auto sections = trtf::find_bundle_sections(bundle);
    check(sections.plan_data != nullptr, "find_sections speech: engine_plan not null");
    check(sections.depth_engine_plan_data != nullptr, "find_sections speech: depth not null");
    check(sections.mimi_encoder_plan_data != nullptr, "find_sections speech: mimi_encoder not null");
    check(sections.mimi_decoder_plan_data != nullptr, "find_sections speech: mimi_decoder not null");
    check((*sections.depth_engine_plan_data)[0] == 'D', "find_sections speech: depth data[0]");
    check((*sections.mimi_encoder_plan_data)[0] == 'M', "find_sections speech: mimi_enc data[0]");
    check((*sections.mimi_decoder_plan_data)[0] == 'M', "find_sections speech: mimi_dec data[0]");
}

static void test_find_sections_omni()
{
    trtf::BundleFile bundle;
    bundle.sections.push_back({"engine_plan", {'P'}});
    bundle.sections.push_back({"audio_encoder_plan", {'A', 'E'}});
    bundle.sections.push_back({"talker_engine_plan", {'T', 'E'}});
    bundle.sections.push_back({"code2wav_engine_plan", {'C', 'W'}});
    bundle.sections.push_back({"vision_engine_plan", {'V', 'E'}});

    const auto sections = trtf::find_bundle_sections(bundle);
    check(sections.plan_data != nullptr, "find_sections omni: engine_plan not null");
    check(sections.audio_encoder_plan_data != nullptr, "find_sections omni: audio_encoder not null");
    check(sections.talker_engine_plan_data != nullptr, "find_sections omni: talker not null");
    check(sections.code2wav_engine_plan_data != nullptr, "find_sections omni: code2wav not null");
    check(sections.vision_plan_data != nullptr, "find_sections omni: vision not null");
    check((*sections.audio_encoder_plan_data)[0] == 'A', "find_sections omni: audio data[0]");
    check((*sections.talker_engine_plan_data)[0] == 'T', "find_sections omni: talker data[0]");
    check((*sections.code2wav_engine_plan_data)[0] == 'C', "find_sections omni: code2wav data[0]");
}

static void test_find_sections_magpie_ipa()
{
    trtf::BundleFile bundle;
    bundle.sections.push_back({"engine_plan", {'P'}});
    bundle.sections.push_back({"magpie_audio_embed", {'A', 'E'}});
    bundle.sections.push_back({"magpie_ipa_phoneme_dict", {'D', 'I', 'C', 'T'}});
    bundle.sections.push_back({"magpie_ipa_heteronyms", {'H', 'E', 'T'}});
    bundle.sections.push_back({"magpie_ipa_vocab", {'V', 'O', 'C'}});
    bundle.sections.push_back({"magpie_ipa_config", {'C', 'F', 'G'}});

    const auto sections = trtf::find_bundle_sections(bundle);
    check(sections.plan_data != nullptr, "find_sections magpie_ipa: engine_plan not null");
    check(sections.magpie_audio_embed_data != nullptr, "find_sections magpie_ipa: audio_embed not null");
    check(sections.magpie_ipa_phoneme_dict_data != nullptr, "find_sections magpie_ipa: phoneme_dict not null");
    check(sections.magpie_ipa_heteronyms_data != nullptr, "find_sections magpie_ipa: heteronyms not null");
    check(sections.magpie_ipa_vocab_data != nullptr, "find_sections magpie_ipa: vocab not null");
    check(sections.magpie_ipa_config_data != nullptr, "find_sections magpie_ipa: config not null");
    check(sections.magpie_ipa_phoneme_dict_data->size() == 4, "find_sections magpie_ipa: dict size");
    check(sections.magpie_ipa_heteronyms_data->size() == 3, "find_sections magpie_ipa: het size");
    check(sections.magpie_ipa_vocab_data->size() == 3, "find_sections magpie_ipa: vocab size");
    check(sections.magpie_ipa_config_data->size() == 3, "find_sections magpie_ipa: config size");
}

#endif // TRTF_HAS_TRT

int main()
{
#if TRTF_HAS_TRT
    test_find_sections_engine_plan();
    test_find_sections_config_json();
    test_find_sections_tokenizer();
    test_find_sections_vision();
    test_find_sections_empty_bundle();
    test_find_sections_unknown_ignored();
    test_find_sections_bark_sections();
    test_find_sections_diffusion();
    test_find_sections_speech();
    test_find_sections_omni();
    test_find_sections_magpie_ipa();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All bundle_helpers tests passed.\n";
#else
    std::cerr << "test_bundle_helpers: TRTF_HAS_TRT=0, skipping TRT-dependent tests.\n";
#endif
    return 0;
}
