#include "trtf/runtime/pipeline_plugin.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <initializer_list>

namespace trtf {

namespace {

// Extract the first non-zero int from a list of JSON key aliases.
int32_t first_nonzero_int(const std::string& text, std::initializer_list<const char*> keys)
{
    for (const char* key : keys)
    {
        int32_t v = extract_json_int(text, key, 0);
        if (v != 0) return v;
    }
    return 0;
}

void parse_model_dimensions(const std::string& config_text, BaseConfig& cfg)
{
    cfg.vocab_size = extract_json_int(config_text, "vocab_size", 0);
    cfg.hidden_size = first_nonzero_int(config_text,
        {"hidden_size", "n_embd", "d_model", "n_embed", "dim"});

    cfg.num_layers = std::max(first_nonzero_int(config_text,
        {"num_hidden_layers", "n_layer", "num_layers", "n_layers"}), 1);

    int32_t decoder_layers = extract_json_int(config_text, "decoder_layers", 0);
    if (decoder_layers > 0)
        cfg.num_layers = decoder_layers;

    cfg.num_heads = std::max(first_nonzero_int(config_text,
        {"num_attention_heads", "n_head", "attention_heads",
         "num_heads", "n_heads", "decoder_attention_heads"}), 1);

    cfg.num_kv_heads = std::max(
        extract_json_int(config_text, "num_key_value_heads", cfg.num_heads), 1);
    cfg.head_dim = extract_json_int(config_text, "head_dim",
        cfg.hidden_size / cfg.num_heads);
    cfg.attention_size = cfg.num_heads * cfg.head_dim;
}

void parse_cache_and_tokens(const std::string& config_text,
    int32_t max_cache_length_override, BaseConfig& cfg)
{
    if (max_cache_length_override > 0)
    {
        cfg.max_cache_length = max_cache_length_override;
    }
    else
    {
        cfg.max_cache_length = extract_json_int(
            config_text, "max_position_embeddings", 32);
        if (cfg.max_cache_length > 4096)
            cfg.max_cache_length = 4096;
    }
    cfg.id_bos = extract_json_int_or_first_array(config_text, "bos_token_id", -1);
    cfg.id_eos = extract_json_int_or_first_array(config_text, "eos_token_id", -1);
}

void parse_strategy_and_tokenizer_flags(const std::string& config_text, BaseConfig& cfg)
{
    cfg.runtime_strategy = extract_json_string(
        config_text, "runtime_strategy", "decoder_kv_cache");

    cfg.precision = extract_json_string(config_text, "precision", "fp32");

    int32_t raw = extract_json_int(config_text, "tokenizer_add_special_tokens", -1);
    if (raw >= 0)
    {
        cfg.tokenizer_add_special_tokens = (raw != 0);
        cfg.tokenizer_add_special_tokens_present = true;
    }
}

} // namespace

BaseConfig parse_base_config(const std::string& config_text, int32_t max_cache_length_override)
{
    BaseConfig cfg;
    parse_model_dimensions(config_text, cfg);
    parse_cache_and_tokens(config_text, max_cache_length_override, cfg);
    parse_strategy_and_tokenizer_flags(config_text, cfg);
    return cfg;
}

} // namespace trtf
