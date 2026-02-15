// =============================================================================
// Test suite: FastPathModelConfig parsing from config.json
// =============================================================================
//
// Purpose:
//   Validates parse_fast_path_config() from cabi/fast_path_config.h — the
//   function that extracts model architecture parameters (head dimensions,
//   attention size, cache length, special token IDs) from a raw config.json
//   string. This parser is used by the zero-weight fast path (C ABI) to
//   determine engine shape parameters without loading the full DecoderModel.
//
// Dependencies:
//   - cabi/fast_path_config.h (parse_fast_path_config, FastPathModelConfig)
//
// Approach:
//   CPU-only, pure in-memory tests. Each test constructs a JSON string
//   representing a model's config.json, passes it to parse_fast_path_config(),
//   and asserts that the returned FastPathModelConfig struct contains the
//   correct derived values. Tests cover:
//   - Explicit vs. computed head_dim (fallback to hidden_size / num_heads)
//   - attention_size derivation with GQA (grouped query attention)
//   - max_cache_length override via parameter and default 4096 cap
//   - eos/bos token ID extraction from both JSON arrays and scalars
//
// Environment:
//   No disk I/O, no GPU, no TensorRT required.
// =============================================================================

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

// -----------------------------------------------------------------------------
// Intention:  Verify that when config.json contains an explicit "head_dim"
//             field, the parser uses it directly rather than computing
//             hidden_size / num_attention_heads.
// Setup:      A JSON config with hidden_size=896, num_attention_heads=14, and
//             head_dim=128. Note that 896/14=64 != 128, so the explicit value
//             must win.
// Mechanism:  Calls parse_fast_path_config with the JSON string and
//             max_cache_length_override=-1 (no override). Checks that head_dim
//             is 128, num_heads is 14, and attention_size equals
//             num_heads * head_dim = 1792.
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention:  Verify the fallback computation of head_dim when it is absent
//             from config.json. The parser should compute
//             head_dim = hidden_size / num_attention_heads.
// Setup:      A JSON config with hidden_size=2048 and num_attention_heads=16,
//             but no "head_dim" field.
// Mechanism:  Calls parse_fast_path_config and checks that head_dim equals
//             2048/16=128 and attention_size equals 16*128=2048.
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention:  Verify correct parsing of GQA (grouped query attention)
//             parameters and that attention_size is computed as
//             num_attention_heads * head_dim (not num_kv_heads * head_dim).
// Setup:      A JSON config with num_attention_heads=16, num_key_value_heads=4,
//             and explicit head_dim=128.
// Mechanism:  Calls parse_fast_path_config and checks num_heads, num_kv_heads,
//             head_dim, and attention_size. The key assertion is that
//             attention_size uses num_heads (16), not num_kv_heads (4).
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention:  Verify that the max_cache_length_override parameter (simulating
//             the TRTF_MAX_CACHE_LENGTH env var) overrides whatever the config
//             would normally produce.
// Setup:      A JSON config with max_position_embeddings=131072 (very large),
//             and max_cache_length_override=64.
// Mechanism:  Calls parse_fast_path_config with override=64 and asserts
//             max_cache_length is 64, not the config's 131072 or the default
//             cap of 4096.
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention:  Verify that max_cache_length is capped at 4096 when no override
//             is specified and the config's max_position_embeddings exceeds
//             the cap. This prevents allocating excessively large KV caches.
// Setup:      A JSON config with max_position_embeddings=131072 and no override
//             (max_cache_length_override=-1).
// Mechanism:  Calls parse_fast_path_config and asserts max_cache_length is
//             4096 (the built-in cap).
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention:  Verify that eos_token_id and bos_token_id are correctly parsed
//             when they appear as JSON arrays. The parser should extract the
//             first element of each array.
// Setup:      A JSON config with "eos_token_id": [151645, 151643] (multi-
//             element) and "bos_token_id": [151644] (single-element).
// Mechanism:  Calls parse_fast_path_config and asserts id_eos equals the first
//             element (151645) and id_bos equals the single element (151644).
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Intention:  Verify that eos_token_id and bos_token_id are correctly parsed
//             when they appear as plain JSON scalars (integers) rather than
//             arrays.
// Setup:      A JSON config with "eos_token_id": 2 and "bos_token_id": 1.
// Mechanism:  Calls parse_fast_path_config and asserts id_eos=2, id_bos=1.
// -----------------------------------------------------------------------------
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
