#include "cabi/fast_path_config.h"
#include "utils/json_helpers.h"

#include <algorithm>

namespace trtf {

FastPathModelConfig parse_fast_path_config(const std::string& config_text, int32_t max_cache_length_override)
{
    FastPathModelConfig cfg;

    cfg.vocab_size = extract_json_int(config_text, "vocab_size", 0);

    // Support GPT-2 aliases: n_embd -> hidden_size, n_layer -> num_hidden_layers,
    // n_head -> num_attention_heads, n_positions -> max_position_embeddings.
    cfg.hidden_size = extract_json_int(config_text, "hidden_size", 0);
    if (cfg.hidden_size == 0)
        cfg.hidden_size = extract_json_int(config_text, "n_embd", 0);

    int32_t raw_layers = extract_json_int(config_text, "num_hidden_layers", 0);
    if (raw_layers == 0)
        raw_layers = extract_json_int(config_text, "n_layer", 0);
    cfg.num_layers = std::max(raw_layers, 1);

    int32_t raw_heads = extract_json_int(config_text, "num_attention_heads", 0);
    if (raw_heads == 0)
        raw_heads = extract_json_int(config_text, "n_head", 0);
    cfg.num_heads = std::max(raw_heads, 1);

    cfg.num_kv_heads = std::max(extract_json_int(config_text, "num_key_value_heads", cfg.num_heads), 1);
    cfg.head_dim = extract_json_int(config_text, "head_dim", cfg.hidden_size / cfg.num_heads);
    cfg.attention_size = cfg.num_heads * cfg.head_dim;

    if (max_cache_length_override > 0)
    {
        cfg.max_cache_length = max_cache_length_override;
    }
    else
    {
        cfg.max_cache_length = extract_json_int(config_text, "max_position_embeddings", 32);
        if (cfg.max_cache_length > 4096)
        {
            cfg.max_cache_length = 4096;
        }
    }

    cfg.id_bos = extract_json_int_or_first_array(config_text, "bos_token_id", -1);
    cfg.id_eos = extract_json_int_or_first_array(config_text, "eos_token_id", -1);

    return cfg;
}

} // namespace trtf
