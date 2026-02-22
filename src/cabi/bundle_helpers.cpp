#include "cabi/bundle_helpers.h"

#include <chrono>
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
    }
    return s;
}

TokenizerResult extract_tokenizer_from_bundle(
    const BundleSections& sections,
    const std::string& hf_python)
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
    auto tokenizer = CreateHfPythonTokenizer(temp_dir_str, hf_python);
    auto ttok1 = std::chrono::steady_clock::now();
    std::cerr << "[trtf] Tokenizer ready ["
              << std::chrono::duration_cast<std::chrono::milliseconds>(ttok1 - ttok0).count()
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
