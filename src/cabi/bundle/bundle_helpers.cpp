#include "cabi/bundle/bundle_helpers.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace trtf {

#if TRTF_HAS_TRT

namespace {

using SectionDataPtr = const std::vector<char>* BundleSections::*;

struct SectionMapping {
    const char* name;
    SectionDataPtr target;
};

constexpr SectionMapping kSectionMappings[] = {
    {"engine_plan", &BundleSections::plan_data},
    {"vision_engine_plan", &BundleSections::vision_plan_data},
    {"config.json", &BundleSections::config_json_data},
    {"preprocessor_config.json", &BundleSections::preprocessor_config_data},
    {"tokenizer.json", &BundleSections::tokenizer_json_data},
    {"tokenizer_config.json", &BundleSections::tokenizer_config_data},
    {"vocab.json", &BundleSections::vocab_json_data},
    {"merges.txt", &BundleSections::merges_txt_data},
    {"special_tokens_map.json", &BundleSections::special_tokens_data},
    {"tokenizer.model", &BundleSections::tokenizer_model_data},
    // Bark/multi-engine sections
    {"coarse_engine_plan", &BundleSections::coarse_engine_plan_data},
    {"fine_engine_plan", &BundleSections::fine_engine_plan_data},
    {"codec_engine_plan", &BundleSections::codec_engine_plan_data},
    {"semantic_embed", &BundleSections::semantic_embed_data},
    {"coarse_embed", &BundleSections::coarse_embed_data},
    {"fine_embed", &BundleSections::fine_embed_data},
    {"fine_position_embed", &BundleSections::fine_position_embed_data},
    // MagpieTTS sections
    {"magpie_audio_embed", &BundleSections::magpie_audio_embed_data},
    {"magpie_text_embed", &BundleSections::magpie_text_embed_data},
    {"magpie_context_embed", &BundleSections::magpie_context_embed_data},
    {"magpie_context_lengths", &BundleSections::magpie_context_lengths_data},
    // MagpieTTS native IPA tokenizer sections
    {"magpie_ipa_phoneme_dict", &BundleSections::magpie_ipa_phoneme_dict_data},
    {"magpie_ipa_heteronyms", &BundleSections::magpie_ipa_heteronyms_data},
    {"magpie_ipa_vocab", &BundleSections::magpie_ipa_vocab_data},
    {"magpie_ipa_config", &BundleSections::magpie_ipa_config_data},
    // Speech-to-speech (PersonaPlex/Moshi) sections
    {"depth_engine_plan", &BundleSections::depth_engine_plan_data},
    {"mimi_encoder_plan", &BundleSections::mimi_encoder_plan_data},
    {"mimi_decoder_plan", &BundleSections::mimi_decoder_plan_data},
    {"depth_projection", &BundleSections::depth_projection_data},
    {"audio_embeddings", &BundleSections::audio_embeddings_data},
    {"temporal_text_embedding", &BundleSections::temporal_text_embedding_data},
    {"depth_text_embedding", &BundleSections::depth_text_embedding_data},
    {"depth_audio_embeddings", &BundleSections::depth_audio_embeddings_data},
    // Omni multimodal sections
    {"audio_encoder_plan", &BundleSections::audio_encoder_plan_data},
    {"talker_engine_plan", &BundleSections::talker_engine_plan_data},
    {"code2wav_engine_plan", &BundleSections::code2wav_engine_plan_data},
    // Diffusion sections
    {"denoiser_plan", &BundleSections::denoiser_plan_data},
    {"vae_decoder_plan", &BundleSections::vae_decoder_plan_data},
    {"preprocessor_weights", &BundleSections::preprocessor_weights_data},
    // CLIP tokenizer sections (for FLUX dual-tokenizer)
    {"clip_tokenizer_config.json", &BundleSections::clip_tokenizer_config_data},
    {"clip_vocab.json", &BundleSections::clip_vocab_json_data},
    {"clip_merges.txt", &BundleSections::clip_merges_txt_data},
    {"clip_special_tokens_map.json", &BundleSections::clip_special_tokens_data},
    // Whisper mel filterbank
    {"mel_filterbank", &BundleSections::mel_filterbank_data},
};

constexpr const char* kDepthEnginePlanPrefix = "depth_engine_plan_";
constexpr std::size_t kDepthEnginePlanPrefixLen = sizeof("depth_engine_plan_") - 1;

bool assign_mapped_section(BundleSections& sections, const BundleSection& section)
{
    for (const auto& mapping : kSectionMappings)
    {
        if (section.name == mapping.name)
        {
            sections.*(mapping.target) = &section.data;
            return true;
        }
    }
    return false;
}

bool assign_depth_engine_plan_section(BundleSections& sections, const BundleSection& section)
{
    if (section.name.rfind(kDepthEnginePlanPrefix, 0) != 0)
        return false;

    auto idx_str = section.name.substr(kDepthEnginePlanPrefixLen);
    int idx = std::stoi(idx_str);
    if (idx < 0)
        return true;

    auto idx_size = static_cast<std::size_t>(idx);
    if (idx_size >= sections.depth_engine_plans.size())
        sections.depth_engine_plans.resize(idx_size + 1, nullptr);
    sections.depth_engine_plans[idx_size] = &section.data;
    return true;
}

bool is_text_encoder_plan_section(const std::string& section_name)
{
    return section_name.rfind("text_encoder_", 0) == 0 &&
        section_name.find("_plan") != std::string::npos;
}

bool has_non_empty_data(const std::vector<char>* data)
{
    return data != nullptr && !data->empty();
}

bool has_bundle_tokenizer_data(const BundleSections& sections)
{
    return has_non_empty_data(sections.tokenizer_json_data)
        || has_non_empty_data(sections.vocab_json_data)
        || has_non_empty_data(sections.tokenizer_model_data);
}

std::filesystem::path create_tokenizer_temp_dir()
{
    char temp_pattern[] = "/tmp/trtfb_tok_XXXXXX";
    char* created = mkdtemp(temp_pattern);
    if (created == nullptr)
    {
        throw std::runtime_error("Failed to create temp dir for bundle tokenizer");
    }
    return std::filesystem::path(created);
}

void write_optional_section_file(
    const std::filesystem::path& dir,
    const char* filename,
    const std::vector<char>* data)
{
    if (!has_non_empty_data(data))
    {
        return;
    }

    std::ofstream out(dir / filename, std::ios::binary | std::ios::trunc);
    if (!out)
    {
        return;
    }

    out.write(data->data(), static_cast<std::streamsize>(data->size()));
}

void write_bundle_tokenizer_files(
    const std::filesystem::path& temp_dir,
    const BundleSections& sections)
{
    write_optional_section_file(temp_dir, "tokenizer.json", sections.tokenizer_json_data);
    write_optional_section_file(temp_dir, "tokenizer_config.json", sections.tokenizer_config_data);
    write_optional_section_file(temp_dir, "vocab.json", sections.vocab_json_data);
    write_optional_section_file(temp_dir, "merges.txt", sections.merges_txt_data);
    write_optional_section_file(temp_dir, "special_tokens_map.json", sections.special_tokens_data);
    write_optional_section_file(temp_dir, "tokenizer.model", sections.tokenizer_model_data);
    write_optional_section_file(temp_dir, "preprocessor_config.json", sections.preprocessor_config_data);
}

} // namespace

BundleSections find_bundle_sections(const BundleFile& bundle)
{
    BundleSections s;
    for (const auto& section : bundle.sections)
    {
        if (assign_mapped_section(s, section))
            continue;
        if (assign_depth_engine_plan_section(s, section))
            continue;
        if (is_text_encoder_plan_section(section.name))
            s.text_encoder_plans.push_back(&section.data);
    }
    return s;
}

MelFilterbank load_mel_filterbank(const BundleSections& sections)
{
    MelFilterbank fb;
    if (sections.mel_filterbank_data == nullptr || sections.mel_filterbank_data->empty())
        return fb;

    const auto* data = sections.mel_filterbank_data;
    // Format: [n_freq_bins(int32), n_mel_bins(int32), float32 data...]
    if (data->size() < 2 * sizeof(int32_t))
        return fb;

    int32_t header[2] = {0, 0};
    std::memcpy(header, data->data(), sizeof(header));
    fb.n_freq_bins = header[0];
    fb.n_mel_bins = header[1];

    if (fb.n_freq_bins <= 0 || fb.n_mel_bins <= 0)
        return fb;

    const auto expected_data_size = static_cast<std::size_t>(fb.n_freq_bins) *
        static_cast<std::size_t>(fb.n_mel_bins) * sizeof(float);
    const auto payload_offset = 2 * sizeof(int32_t);
    if (data->size() < payload_offset + expected_data_size)
    {
        fb.n_freq_bins = 0;
        fb.n_mel_bins = 0;
        return fb;
    }

    fb.data.resize(static_cast<std::size_t>(fb.n_freq_bins) * fb.n_mel_bins);
    std::memcpy(fb.data.data(), data->data() + payload_offset,
                expected_data_size);
    return fb;
}

TokenizerResult extract_tokenizer_from_bundle(
    const BundleSections& sections,
    const std::string& hf_python,
    bool add_special_tokens)
{
    if (!has_bundle_tokenizer_data(sections))
    {
        throw std::runtime_error("Bundle has no tokenizer files");
    }

    const std::filesystem::path temp_dir = create_tokenizer_temp_dir();
    std::string temp_dir_str = temp_dir.string();
    write_bundle_tokenizer_files(temp_dir, sections);

    std::cerr << "[trtf] Initializing HF tokenizer from bundle ..." << std::endl;
    auto ttok0 = std::chrono::steady_clock::now();
    auto tokenizer = CreateHfPythonTokenizer(temp_dir_str, hf_python, add_special_tokens);
    auto ttok1 = std::chrono::steady_clock::now();
    std::cerr << "[trtf] Tokenizer ready ["
              << std::chrono::duration_cast<std::chrono::milliseconds>(ttok1 - ttok0).count()
              << " ms]" << std::endl;

    return TokenizerResult{std::move(tokenizer), std::move(temp_dir_str)};
}

TokenizerResult extract_clip_tokenizer_from_bundle(
    const BundleSections& sections,
    const std::string& hf_python)
{
    bool has_clip = (sections.clip_vocab_json_data != nullptr && !sections.clip_vocab_json_data->empty());
    if (!has_clip)
    {
        throw std::runtime_error("Bundle has no CLIP tokenizer files");
    }

    char temp_pattern[] = "/tmp/trtfb_clip_XXXXXX";
    char* created = mkdtemp(temp_pattern);
    if (created == nullptr)
    {
        throw std::runtime_error("Failed to create temp dir for CLIP tokenizer");
    }
    std::string temp_dir_str(created);
    const std::filesystem::path temp_dir(temp_dir_str);

    auto write_section = [&](const char* filename, const std::vector<char>* data) {
        if (data != nullptr && !data->empty())
        {
            std::ofstream out(temp_dir / filename, std::ios::binary | std::ios::trunc);
            if (out)
            {
                out.write(data->data(), static_cast<std::streamsize>(data->size()));
            }
        }
    };

    // Write CLIP tokenizer files with standard names (HfPythonTokenizer expects them)
    write_section("vocab.json", sections.clip_vocab_json_data);
    write_section("merges.txt", sections.clip_merges_txt_data);
    write_section("tokenizer_config.json", sections.clip_tokenizer_config_data);
    write_section("special_tokens_map.json", sections.clip_special_tokens_data);

    std::cerr << "[trtf] Initializing CLIP tokenizer from bundle ..." << std::endl;
    auto t0 = std::chrono::steady_clock::now();
    auto tokenizer = CreateHfPythonTokenizer(temp_dir_str, hf_python, /*add_special_tokens=*/true);
    auto t1 = std::chrono::steady_clock::now();
    std::cerr << "[trtf] CLIP tokenizer ready ["
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms]" << std::endl;

    return TokenizerResult{std::move(tokenizer), std::move(temp_dir_str)};
}

std::unique_ptr<DecoderStepEngine> make_decoder_engine(
    TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const FastPathModelConfig& cfg)
{
    auto engine = std::make_unique<DecoderStepEngine>();
    engine->engine = std::move(trt_engine);
    engine->context = std::move(exec_ctx);
    engine->vocab_size = cfg.vocab_size;
    engine->hidden_size = cfg.hidden_size;
    engine->cache_state_size = cfg.attention_size;
    engine->attention_mask_size = cfg.max_cache_length + 1;
    engine->max_cache_length = cfg.max_cache_length;
    engine->num_layers = cfg.num_layers;
    engine->requires_position_input = true;
    engine->id_bos = cfg.id_bos;
    engine->id_eos = cfg.id_eos;

    for (int32_t i = 0; i < cfg.num_layers; ++i)
    {
        engine->cache_k_input_names.push_back(layer_tensor_name("cache_k", i));
        engine->cache_v_input_names.push_back(layer_tensor_name("cache_v", i));
        engine->present_k_output_names.push_back(layer_tensor_name("present_k", i));
        engine->present_v_output_names.push_back(layer_tensor_name("present_v", i));
    }

    return engine;
}

#endif // TRTF_HAS_TRT

} // namespace trtf
