#include "cabi/bundle_helpers.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace trtf {

#if TRTF_HAS_TRT

BundleSections find_bundle_sections(const BundleFile& bundle)
{
    BundleSections s;
    for (const auto& section : bundle.sections)
    {
        if (section.name == "engine_plan") s.plan_data = &section.data;
        else if (section.name == "vision_engine_plan") s.vision_plan_data = &section.data;
        else if (section.name == "config.json") s.config_json_data = &section.data;
        else if (section.name == "preprocessor_config.json") s.preprocessor_config_data = &section.data;
        else if (section.name == "tokenizer.json") s.tokenizer_json_data = &section.data;
        else if (section.name == "tokenizer_config.json") s.tokenizer_config_data = &section.data;
        else if (section.name == "vocab.json") s.vocab_json_data = &section.data;
        else if (section.name == "merges.txt") s.merges_txt_data = &section.data;
        else if (section.name == "special_tokens_map.json") s.special_tokens_data = &section.data;
        else if (section.name == "tokenizer.model") s.tokenizer_model_data = &section.data;
        // Bark/multi-engine sections
        else if (section.name == "coarse_engine_plan") s.coarse_engine_plan_data = &section.data;
        else if (section.name == "fine_engine_plan") s.fine_engine_plan_data = &section.data;
        else if (section.name == "codec_engine_plan") s.codec_engine_plan_data = &section.data;
        else if (section.name == "semantic_embed") s.semantic_embed_data = &section.data;
        else if (section.name == "coarse_embed") s.coarse_embed_data = &section.data;
        else if (section.name == "fine_embed") s.fine_embed_data = &section.data;
        else if (section.name == "fine_position_embed") s.fine_position_embed_data = &section.data;
        // MagpieTTS sections
        else if (section.name == "magpie_audio_embed") s.magpie_audio_embed_data = &section.data;
        else if (section.name == "magpie_text_embed") s.magpie_text_embed_data = &section.data;
        else if (section.name == "magpie_context_embed") s.magpie_context_embed_data = &section.data;
        else if (section.name == "magpie_context_lengths") s.magpie_context_lengths_data = &section.data;
        // MagpieTTS native IPA tokenizer sections
        else if (section.name == "magpie_ipa_phoneme_dict") s.magpie_ipa_phoneme_dict_data = &section.data;
        else if (section.name == "magpie_ipa_heteronyms") s.magpie_ipa_heteronyms_data = &section.data;
        else if (section.name == "magpie_ipa_vocab") s.magpie_ipa_vocab_data = &section.data;
        else if (section.name == "magpie_ipa_config") s.magpie_ipa_config_data = &section.data;
        // Speech-to-speech (PersonaPlex/Moshi) sections
        else if (section.name == "depth_engine_plan") s.depth_engine_plan_data = &section.data;
        else if (section.name == "mimi_encoder_plan") s.mimi_encoder_plan_data = &section.data;
        else if (section.name == "mimi_decoder_plan") s.mimi_decoder_plan_data = &section.data;
        else if (section.name == "depth_projection") s.depth_projection_data = &section.data;
        else if (section.name == "audio_embeddings") s.audio_embeddings_data = &section.data;
        else if (section.name == "temporal_text_embedding") s.temporal_text_embedding_data = &section.data;
        else if (section.name == "depth_text_embedding") s.depth_text_embedding_data = &section.data;
        else if (section.name == "depth_audio_embeddings") s.depth_audio_embeddings_data = &section.data;
        else if (section.name.rfind("depth_engine_plan_", 0) == 0)
        {
            // depth_engine_plan_0, depth_engine_plan_1, ...
            // Parse codebook index from suffix
            auto idx_str = section.name.substr(18);  // strlen("depth_engine_plan_")
            int idx = std::stoi(idx_str);
            if (idx >= 0)
            {
                if (static_cast<std::size_t>(idx) >= s.depth_engine_plans.size())
                    s.depth_engine_plans.resize(static_cast<std::size_t>(idx) + 1, nullptr);
                s.depth_engine_plans[static_cast<std::size_t>(idx)] = &section.data;
            }
        }
        // Omni multimodal sections
        else if (section.name == "audio_encoder_plan") s.audio_encoder_plan_data = &section.data;
        else if (section.name == "talker_engine_plan") s.talker_engine_plan_data = &section.data;
        else if (section.name == "code2wav_engine_plan") s.code2wav_engine_plan_data = &section.data;
        // Diffusion sections
        else if (section.name == "denoiser_plan") s.denoiser_plan_data = &section.data;
        else if (section.name == "vae_decoder_plan") s.vae_decoder_plan_data = &section.data;
        else if (section.name == "preprocessor_weights") s.preprocessor_weights_data = &section.data;
        else if (section.name.rfind("text_encoder_", 0) == 0 &&
                 section.name.find("_plan") != std::string::npos)
        {
            // text_encoder_0_plan, text_encoder_1_plan, ...
            s.text_encoder_plans.push_back(&section.data);
        }
        // CLIP tokenizer sections (for FLUX dual-tokenizer)
        else if (section.name == "clip_tokenizer_config.json") s.clip_tokenizer_config_data = &section.data;
        else if (section.name == "clip_vocab.json") s.clip_vocab_json_data = &section.data;
        else if (section.name == "clip_merges.txt") s.clip_merges_txt_data = &section.data;
        else if (section.name == "clip_special_tokens_map.json") s.clip_special_tokens_data = &section.data;
        // Whisper mel filterbank
        else if (section.name == "mel_filterbank") s.mel_filterbank_data = &section.data;
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
    bool has_tok = (sections.tokenizer_json_data != nullptr && !sections.tokenizer_json_data->empty())
        || (sections.vocab_json_data != nullptr && !sections.vocab_json_data->empty())
        || (sections.tokenizer_model_data != nullptr && !sections.tokenizer_model_data->empty());

    if (!has_tok)
    {
        throw std::runtime_error("Bundle has no tokenizer files");
    }

    char temp_pattern[] = "/tmp/trtfb_tok_XXXXXX";
    char* created = mkdtemp(temp_pattern);
    if (created == nullptr)
    {
        throw std::runtime_error("Failed to create temp dir for bundle tokenizer");
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

    write_section("tokenizer.json", sections.tokenizer_json_data);
    write_section("tokenizer_config.json", sections.tokenizer_config_data);
    write_section("vocab.json", sections.vocab_json_data);
    write_section("merges.txt", sections.merges_txt_data);
    write_section("special_tokens_map.json", sections.special_tokens_data);
    write_section("tokenizer.model", sections.tokenizer_model_data);
    write_section("preprocessor_config.json", sections.preprocessor_config_data);

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
