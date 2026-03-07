// =============================================================================
// test_bundle_helpers.cpp — Unit tests for bundle_helpers functions
// =============================================================================
//
// Purpose:
//   Validates the find_bundle_sections() function from bundle_helpers.h/cpp,
//   which scans a BundleFile's sections and populates a BundleSections struct
//   with pointers by name. TRT-dependent coverage is skipped when
//   TRTF_HAS_TRT is disabled, while pure diffusion helper tests still run.
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
#include "runtime/trt/diffusion/diffusion_preprocessor_weights_helpers.h"
#include "runtime/trt/diffusion/diffusion_scheduler_helpers.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
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

static bool almost_equal(double lhs, double rhs, double epsilon = 1e-6)
{
    return std::fabs(lhs - rhs) <= epsilon;
}

static std::vector<char> make_preprocessor_weights_section(
    const std::string& index_json,
    const std::vector<float>& blob_values)
{
    const uint32_t index_len = static_cast<uint32_t>(index_json.size());
    const std::size_t blob_bytes = blob_values.size() * sizeof(float);
    std::vector<char> packed(4 + index_json.size() + blob_bytes, 0);
    std::memcpy(packed.data(), &index_len, sizeof(index_len));
    std::memcpy(packed.data() + 4, index_json.data(), index_json.size());
    if (!blob_values.empty()) {
        std::memcpy(packed.data() + 4 + index_json.size(), blob_values.data(), blob_bytes);
    }
    return packed;
}

#if TRTF_HAS_TRT

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

static void test_find_sections_depth_indexed_plans()
{
    trtf::BundleFile bundle;
    bundle.sections.push_back({"depth_engine_plan_-1", {'X'}});
    bundle.sections.push_back({"depth_engine_plan_2", {'D', '2'}});
    bundle.sections.push_back({"depth_engine_plan_0", {'D', '0'}});

    const auto sections = trtf::find_bundle_sections(bundle);
    check(sections.depth_engine_plans.size() == 3, "find_sections depth indexed: resize to max index");
    check(sections.depth_engine_plans[0] != nullptr, "find_sections depth indexed: idx0 present");
    check(sections.depth_engine_plans[1] == nullptr, "find_sections depth indexed: sparse idx1 null");
    check(sections.depth_engine_plans[2] != nullptr, "find_sections depth indexed: idx2 present");
    check((*sections.depth_engine_plans[0])[1] == '0', "find_sections depth indexed: idx0 data");
    check((*sections.depth_engine_plans[2])[1] == '2', "find_sections depth indexed: idx2 data");
}

static void test_load_mel_filterbank_absent_returns_empty()
{
    trtf::BundleSections sections;
    const auto fb = trtf::load_mel_filterbank(sections);
    check(fb.n_freq_bins == 0, "mel_filterbank absent: n_freq_bins=0");
    check(fb.n_mel_bins == 0, "mel_filterbank absent: n_mel_bins=0");
    check(fb.data.empty(), "mel_filterbank absent: data empty");
}

static void test_load_mel_filterbank_invalid_headers_return_empty()
{
    trtf::BundleSections sections;

    std::vector<char> too_small(3, 0);
    sections.mel_filterbank_data = &too_small;
    auto fb = trtf::load_mel_filterbank(sections);
    check(fb.data.empty(), "mel_filterbank invalid small payload");

    int32_t non_positive_header[2] = {0, 4};
    std::vector<char> invalid_dims(sizeof(non_positive_header), 0);
    std::memcpy(invalid_dims.data(), non_positive_header, sizeof(non_positive_header));
    sections.mel_filterbank_data = &invalid_dims;
    fb = trtf::load_mel_filterbank(sections);
    check(fb.n_freq_bins == 0, "mel_filterbank invalid dims reset freq");
    check(fb.n_mel_bins == 4, "mel_filterbank invalid dims keeps parsed mel header");

    int32_t short_payload_header[2] = {2, 3}; // needs 6 floats; provide none.
    std::vector<char> short_payload(sizeof(short_payload_header), 0);
    std::memcpy(short_payload.data(), short_payload_header, sizeof(short_payload_header));
    sections.mel_filterbank_data = &short_payload;
    fb = trtf::load_mel_filterbank(sections);
    check(fb.n_freq_bins == 0, "mel_filterbank short payload resets freq");
    check(fb.n_mel_bins == 0, "mel_filterbank short payload resets mel");
}

static void test_load_mel_filterbank_valid_payload()
{
    trtf::BundleSections sections;
    int32_t header[2] = {2, 3}; // 6 floats
    const float payload_values[6] = {0.1F, 0.2F, 0.3F, 1.0F, 2.0F, 3.0F};

    std::vector<char> packed(sizeof(header) + sizeof(payload_values), 0);
    std::memcpy(packed.data(), header, sizeof(header));
    std::memcpy(packed.data() + sizeof(header), payload_values, sizeof(payload_values));

    sections.mel_filterbank_data = &packed;
    const auto fb = trtf::load_mel_filterbank(sections);
    check(fb.n_freq_bins == 2, "mel_filterbank valid: n_freq_bins");
    check(fb.n_mel_bins == 3, "mel_filterbank valid: n_mel_bins");
    check(fb.data.size() == 6, "mel_filterbank valid: data size");
    check(fb.data[0] == 0.1F, "mel_filterbank valid: first value");
    check(fb.data[5] == 3.0F, "mel_filterbank valid: last value");
}

static void test_extract_tokenizer_from_bundle_missing_throws()
{
    trtf::BundleSections sections;
    bool threw = false;
    try
    {
        (void) trtf::extract_tokenizer_from_bundle(sections, "", false);
    }
    catch (const std::runtime_error& e)
    {
        threw = true;
        check(std::string(e.what()).find("Bundle has no tokenizer files") != std::string::npos,
            "extract_tokenizer missing: expected error message");
    }
    catch (...)
    {
        threw = true;
        check(false, "extract_tokenizer missing: unexpected exception type");
    }
    check(threw, "extract_tokenizer missing: throws runtime_error");
}

static void test_extract_clip_tokenizer_from_bundle_missing_throws()
{
    trtf::BundleSections sections;
    bool threw = false;
    try
    {
        (void) trtf::extract_clip_tokenizer_from_bundle(sections, "");
    }
    catch (const std::runtime_error& e)
    {
        threw = true;
        check(std::string(e.what()).find("Bundle has no CLIP tokenizer files") != std::string::npos,
            "extract_clip_tokenizer missing: expected error message");
    }
    catch (...)
    {
        threw = true;
        check(false, "extract_clip_tokenizer missing: unexpected exception type");
    }
    check(threw, "extract_clip_tokenizer missing: throws runtime_error");
}

static void test_make_decoder_engine_populates_metadata()
{
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine;
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx;

    trtf::FastPathModelConfig cfg{};
    cfg.vocab_size = 32000;
    cfg.hidden_size = 4096;
    cfg.attention_size = 4096;
    cfg.max_cache_length = 256;
    cfg.num_layers = 3;
    cfg.id_bos = 1;
    cfg.id_eos = 2;

    auto engine = trtf::make_decoder_engine(std::move(trt_engine), std::move(exec_ctx), cfg);
    check(engine != nullptr, "make_decoder_engine returns non-null");
    check(engine->vocab_size == 32000, "make_decoder_engine vocab size");
    check(engine->hidden_size == 4096, "make_decoder_engine hidden size");
    check(engine->cache_state_size == 4096, "make_decoder_engine cache state size");
    check(engine->attention_mask_size == 257, "make_decoder_engine attention mask size");
    check(engine->max_cache_length == 256, "make_decoder_engine max cache");
    check(engine->num_layers == 3, "make_decoder_engine num_layers");
    check(engine->requires_position_input, "make_decoder_engine requires_position_input");
    check(engine->id_bos == 1, "make_decoder_engine id_bos");
    check(engine->id_eos == 2, "make_decoder_engine id_eos");
    check(engine->cache_k_input_names.size() == 3, "make_decoder_engine cache_k names count");
    check(engine->cache_v_input_names.size() == 3, "make_decoder_engine cache_v names count");
    check(engine->present_k_output_names.size() == 3, "make_decoder_engine present_k names count");
    check(engine->present_v_output_names.size() == 3, "make_decoder_engine present_v names count");
    check(engine->cache_k_input_names[0] == "cache_k_0", "make_decoder_engine cache_k_0 name");
    check(engine->present_v_output_names[2] == "present_v_2", "make_decoder_engine present_v_2 name");
}

#endif // TRTF_HAS_TRT

static void test_diffusion_preprocessor_extract_index_success()
{
    const std::string index_json = "{\"meta\":1}";
    const auto packed = make_preprocessor_weights_section(index_json, {1.0F, 2.0F});

    std::string parsed_index_json;
    const char* blob = nullptr;
    std::size_t blob_size = 0;
    const bool ok = trtf::diffusion::extract_preprocessor_index(
        packed, parsed_index_json, blob, blob_size);

    check(ok, "diffusion preprocessor extract success");
    check(parsed_index_json == index_json, "diffusion preprocessor extract keeps index json");
    check(blob_size == 2 * sizeof(float), "diffusion preprocessor extract blob size");
    float first = 0.0F;
    std::memcpy(&first, blob, sizeof(first));
    check(first == 1.0F, "diffusion preprocessor extract first blob float");
}

static void test_diffusion_preprocessor_extract_index_failures()
{
    std::vector<char> too_small(3, 0);
    std::string index_json;
    const char* blob = nullptr;
    std::size_t blob_size = 0;
    check(!trtf::diffusion::extract_preprocessor_index(
              too_small, index_json, blob, blob_size),
        "diffusion preprocessor extract rejects too-small payload");

    std::vector<char> overflow(8, 0);
    uint32_t index_len = 100;
    std::memcpy(overflow.data(), &index_len, sizeof(index_len));
    check(!trtf::diffusion::extract_preprocessor_index(
              overflow, index_json, blob, blob_size),
        "diffusion preprocessor extract rejects length overflow");
}

static void test_diffusion_preprocessor_load_floats_success()
{
    const std::string index_json =
        "{\"patch_embedding.weight\":{\"shape\":[2, 2],\"offset\":0}}";
    const std::vector<float> blob_values = {1.0F, 2.0F, 3.0F, 4.0F};

    std::vector<float> dst;
    const bool ok = trtf::diffusion::load_preprocessor_floats(
        index_json,
        reinterpret_cast<const char*>(blob_values.data()),
        blob_values.size() * sizeof(float),
        "patch_embedding.weight",
        dst);

    check(ok, "diffusion preprocessor load floats success");
    check(dst.size() == 4, "diffusion preprocessor load floats count");
    check(dst[0] == 1.0F, "diffusion preprocessor load floats first");
    check(dst[3] == 4.0F, "diffusion preprocessor load floats last");
}

static void test_diffusion_preprocessor_load_floats_failures()
{
    const std::vector<float> blob_values = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<float> dst;

    const std::string invalid_shape_json =
        "{\"patch_embedding.weight\":{\"shape\":[2, bad],\"offset\":0}}";
    check(!trtf::diffusion::load_preprocessor_floats(
              invalid_shape_json,
              reinterpret_cast<const char*>(blob_values.data()),
              blob_values.size() * sizeof(float),
              "patch_embedding.weight",
              dst),
        "diffusion preprocessor load floats rejects invalid shape token");

    const std::string overflow_json =
        "{\"patch_embedding.weight\":{\"shape\":[2, 2],\"offset\":8}}";
    check(!trtf::diffusion::load_preprocessor_floats(
              overflow_json,
              reinterpret_cast<const char*>(blob_values.data()),
              blob_values.size() * sizeof(float),
              "patch_embedding.weight",
              dst),
        "diffusion preprocessor load floats rejects blob overflow");
}

static void test_diffusion_preprocessor_load_with_fallback()
{
    const std::string index_json =
        "{\"x_embedder.weight\":{\"shape\":[1, 2],\"offset\":0}}";
    const std::vector<float> blob_values = {7.0F, 9.0F};

    std::vector<float> dst;
    check(trtf::diffusion::load_with_fallback(
              index_json,
              reinterpret_cast<const char*>(blob_values.data()),
              blob_values.size() * sizeof(float),
              "patch_embedding.weight",
              "x_embedder.weight",
              dst),
        "diffusion preprocessor fallback uses secondary key");
    check(dst.size() == 2, "diffusion preprocessor fallback count");
    check(dst[0] == 7.0F && dst[1] == 9.0F, "diffusion preprocessor fallback values");

    dst.clear();
    check(!trtf::diffusion::load_with_fallback(
              index_json,
              reinterpret_cast<const char*>(blob_values.data()),
              blob_values.size() * sizeof(float),
              "patch_embedding.weight",
              "",
              dst),
        "diffusion preprocessor fallback rejects empty fallback key");
}

static void test_diffusion_scheduler_resolves_requested_steps()
{
    check(
        trtf::diffusion::resolve_requested_steps(-1, 30, true) == 30,
        "diffusion scheduler resolves negative steps to fallback");
    check(
        trtf::diffusion::resolve_requested_steps(0, 30, true) == 30,
        "diffusion scheduler resolves zero steps to fallback when requested");
    check(
        trtf::diffusion::resolve_requested_steps(0, 30, false) == 0,
        "diffusion scheduler keeps zero steps when backend allows it");
    check(
        trtf::diffusion::resolve_requested_steps(12, 30, true) == 12,
        "diffusion scheduler keeps positive requested steps");
    check(
        trtf::diffusion::resolve_requested_guidance(-1.0F, 7.5F) == 7.5F,
        "diffusion scheduler resolves negative guidance to fallback");
    check(
        trtf::diffusion::resolve_requested_guidance(4.0F, 7.5F) == 4.0F,
        "diffusion scheduler keeps requested guidance");
}

static void test_diffusion_scheduler_builds_standard_flow_match_plan()
{
    trtf::diffusion::FlowMatchEulerConfig config;
    config.num_train_timesteps = 1000;
    config.shift = 1.0F;

    const auto plan = trtf::diffusion::build_flow_match_euler_plan(4, config);
    check(plan.sigmas.size() == 5, "diffusion scheduler standard plan sigma count");
    check(plan.timesteps.size() == 4, "diffusion scheduler standard plan timestep count");
    check(almost_equal(plan.sigmas[0], 1.0), "diffusion scheduler standard plan sigma[0]");
    check(almost_equal(plan.sigmas[1], 0.667), "diffusion scheduler standard plan sigma[1]");
    check(almost_equal(plan.sigmas[2], 0.334), "diffusion scheduler standard plan sigma[2]");
    check(almost_equal(plan.sigmas[3], 0.001), "diffusion scheduler standard plan sigma[last]");
    check(almost_equal(plan.sigmas[4], 0.0), "diffusion scheduler standard plan terminal sigma");
    check(almost_equal(plan.timesteps[0], 1000.0), "diffusion scheduler standard plan timestep[0]");
    check(almost_equal(plan.timesteps[3], 1.0), "diffusion scheduler standard plan timestep[last]");
    check(!plan.used_dynamic_shifting, "diffusion scheduler standard plan is not dynamic");
}

static void test_diffusion_scheduler_builds_zero_sigma_min_plan()
{
    trtf::diffusion::FlowMatchEulerConfig config;
    config.num_train_timesteps = 1000;
    config.shift = 3.0F;
    config.use_zero_sigma_min = true;

    const auto plan = trtf::diffusion::build_flow_match_euler_plan(2, config);
    check(plan.sigmas.size() == 3, "diffusion scheduler zero-sigma plan sigma count");
    check(almost_equal(plan.sigmas[0], 1.0), "diffusion scheduler zero-sigma plan sigma[0]");
    check(almost_equal(plan.sigmas[1], 0.0), "diffusion scheduler zero-sigma plan sigma[last]");
    check(almost_equal(plan.timesteps[0], 1000.0), "diffusion scheduler zero-sigma plan timestep[0]");
    check(almost_equal(plan.timesteps[1], 0.0), "diffusion scheduler zero-sigma plan timestep[last]");
}

static void test_diffusion_scheduler_builds_dynamic_linear_mu_plan()
{
    trtf::diffusion::FlowMatchEulerConfig config;
    config.num_train_timesteps = 1000;
    config.use_dynamic_shifting = true;
    config.base_shift = 0.5F;
    config.max_shift = 1.15F;
    config.image_seq_len = 4096;

    const auto plan = trtf::diffusion::build_flow_match_euler_plan(4, config);
    const double expected_mu =
        (static_cast<double>(config.max_shift) -
         static_cast<double>(config.base_shift)) /
            (4096.0 - 256.0) *
            (static_cast<double>(config.image_seq_len) - 256.0) +
        static_cast<double>(config.base_shift);
    const double exp_mu = std::exp(expected_mu);
    const double expected_last_sigma = exp_mu / (exp_mu + (1.0 / 0.25 - 1.0));

    check(plan.used_dynamic_shifting, "diffusion scheduler dynamic plan marks dynamic mode");
    check(almost_equal(plan.dynamic_mu, expected_mu, 1e-6), "diffusion scheduler dynamic plan linear mu");
    check(almost_equal(plan.sigmas[0], 1.0), "diffusion scheduler dynamic plan sigma[0]");
    check(almost_equal(plan.sigmas[3], expected_last_sigma, 1e-6),
        "diffusion scheduler dynamic plan sigma[last]");
    check(almost_equal(plan.timesteps[3], expected_last_sigma * 1000.0, 1e-3),
        "diffusion scheduler dynamic plan timestep[last]");
}

static void test_diffusion_scheduler_builds_dynamic_empirical_mu_plan()
{
    trtf::diffusion::FlowMatchEulerConfig config;
    config.num_train_timesteps = 1000;
    config.use_dynamic_shifting = true;
    config.use_empirical_mu = true;
    config.image_seq_len = 4096;

    const auto plan = trtf::diffusion::build_flow_match_euler_plan(50, config);
    const double m_200 = 0.00016927 * 4096.0 + 0.45666666;
    const double m_10 = 8.73809524e-05 * 4096.0 + 1.89833333;
    const double slope = (m_200 - m_10) / 190.0;
    const double expected_mu = slope * 50.0 + (m_200 - 200.0 * slope);

    check(almost_equal(plan.dynamic_mu, expected_mu, 1e-6),
        "diffusion scheduler empirical mu uses interpolated branch");

    config.image_seq_len = 5000;
    const auto large_seq_plan = trtf::diffusion::build_flow_match_euler_plan(50, config);
    const double expected_large_mu = 0.00016927 * 5000.0 + 0.45666666;
    check(almost_equal(large_seq_plan.dynamic_mu, expected_large_mu, 1e-6),
        "diffusion scheduler empirical mu uses large-sequence branch");
}

static void test_diffusion_scheduler_zero_steps_yields_terminal_sigma_only()
{
    trtf::diffusion::FlowMatchEulerConfig config;
    const auto plan = trtf::diffusion::build_flow_match_euler_plan(0, config);
    check(plan.sigmas.size() == 1, "diffusion scheduler zero steps keeps terminal sigma");
    check(plan.timesteps.empty(), "diffusion scheduler zero steps keeps empty timesteps");
    check(almost_equal(plan.sigmas[0], 0.0), "diffusion scheduler zero steps terminal sigma is zero");
}

int main()
{
    test_diffusion_preprocessor_extract_index_success();
    test_diffusion_preprocessor_extract_index_failures();
    test_diffusion_preprocessor_load_floats_success();
    test_diffusion_preprocessor_load_floats_failures();
    test_diffusion_preprocessor_load_with_fallback();
    test_diffusion_scheduler_resolves_requested_steps();
    test_diffusion_scheduler_builds_standard_flow_match_plan();
    test_diffusion_scheduler_builds_zero_sigma_min_plan();
    test_diffusion_scheduler_builds_dynamic_linear_mu_plan();
    test_diffusion_scheduler_builds_dynamic_empirical_mu_plan();
    test_diffusion_scheduler_zero_steps_yields_terminal_sigma_only();

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
    test_find_sections_depth_indexed_plans();
    test_load_mel_filterbank_absent_returns_empty();
    test_load_mel_filterbank_invalid_headers_return_empty();
    test_load_mel_filterbank_valid_payload();
    test_extract_tokenizer_from_bundle_missing_throws();
    test_extract_clip_tokenizer_from_bundle_missing_throws();
    test_make_decoder_engine_populates_metadata();
#else
    std::cerr << "test_bundle_helpers: TRTF_HAS_TRT=0, skipping TRT-dependent tests.\n";
#endif

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All bundle_helpers tests passed.\n";
    return 0;
}
