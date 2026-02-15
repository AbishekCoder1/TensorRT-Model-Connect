// Test: FastPathModelConfig parsing from config.json text.
// CPU-only tests verifying head_dim, attention_size, max_cache_length, eos/bos extraction.

#include "cabi/fast_path_config.h"

#include <cstdlib>
#include <iostream>
#include <string>

static int failures = 0;

static void check(bool condition, const char* test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static void test_head_dim_explicit_in_config()
{
    const std::string config = R"({
        "vocab_size": 32000,
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "max_position_embeddings": 4096
    })";

    const auto cfg = trtf::parse_fast_path_config(config, -1);
    check(cfg.head_dim == 128, "head_dim=128 from explicit config (not hidden/heads=64)");
    check(cfg.num_heads == 14, "num_heads=14");
    check(cfg.attention_size == 14 * 128, "attention_size = num_heads * head_dim = 1792");
}

static void test_head_dim_computed_fallback()
{
    const std::string config = R"({
        "vocab_size": 32000,
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "max_position_embeddings": 4096
    })";

    const auto cfg = trtf::parse_fast_path_config(config, -1);
    check(cfg.head_dim == 2048 / 16, "head_dim = hidden_size/num_heads = 128 (fallback)");
    check(cfg.attention_size == 16 * 128, "attention_size = 2048");
}

static void test_attention_size_with_gqa()
{
    const std::string config = R"({
        "vocab_size": 32000,
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "max_position_embeddings": 4096
    })";

    const auto cfg = trtf::parse_fast_path_config(config, -1);
    check(cfg.num_heads == 16, "num_heads=16");
    check(cfg.num_kv_heads == 4, "num_kv_heads=4 (GQA)");
    check(cfg.head_dim == 128, "head_dim=128");
    check(cfg.attention_size == 16 * 128, "attention_size = num_heads * head_dim = 2048");
}

static void test_max_cache_length_from_env()
{
    const std::string config = R"({
        "vocab_size": 32000,
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "max_position_embeddings": 131072
    })";

    // Override with 64 via the parameter (simulating TRTF_MAX_CACHE_LENGTH=64)
    const auto cfg = trtf::parse_fast_path_config(config, 64);
    check(cfg.max_cache_length == 64, "TRTF_MAX_CACHE_LENGTH=64 overrides config value");
}

static void test_max_cache_length_capped_at_4096()
{
    const std::string config = R"({
        "vocab_size": 32000,
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "max_position_embeddings": 131072
    })";

    // No override → should cap at 4096
    const auto cfg = trtf::parse_fast_path_config(config, -1);
    check(cfg.max_cache_length == 4096, "config says 131072 → capped to 4096");
}

static void test_eos_bos_from_array()
{
    const std::string config = R"({
        "vocab_size": 32000,
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "eos_token_id": [151645, 151643],
        "bos_token_id": [151644]
    })";

    const auto cfg = trtf::parse_fast_path_config(config, -1);
    check(cfg.id_eos == 151645, "eos_token_id from array → first element (151645)");
    check(cfg.id_bos == 151644, "bos_token_id from single-element array → 151644");
}

static void test_eos_bos_from_scalar()
{
    const std::string config = R"({
        "vocab_size": 32000,
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "eos_token_id": 2,
        "bos_token_id": 1
    })";

    const auto cfg = trtf::parse_fast_path_config(config, -1);
    check(cfg.id_eos == 2, "eos_token_id from scalar → 2");
    check(cfg.id_bos == 1, "bos_token_id from scalar → 1");
}

int main()
{
    test_head_dim_explicit_in_config();
    test_head_dim_computed_fallback();
    test_attention_size_with_gqa();
    test_max_cache_length_from_env();
    test_max_cache_length_capped_at_4096();
    test_eos_bos_from_array();
    test_eos_bos_from_scalar();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All fast path config tests passed.\n";
    return 0;
}
