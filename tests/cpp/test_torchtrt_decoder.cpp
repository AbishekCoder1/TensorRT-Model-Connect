// =============================================================================
// test_torchtrt_decoder.cpp — Unit tests for torchtrt_decoder runtime strategy
// =============================================================================
//
// Purpose:
//   Validates the torchtrt_decoder strategy integration across the C++ runtime:
//   - BaseConfig parsing with runtime_strategy="torchtrt_decoder"
//   - Strategy dispatch via PipelineRegistry (torchtrt_decoder routes to DecoderPlugin)
//
// Dependencies:
//   - trtf/runtime/pipeline_plugin.h (parse_base_config, BaseConfig)
//
// Environment:
//   CPU-only for config parsing tests.
//
// Note: Engine tensor naming tests (make_decoder_engine) were removed when the
// codebase migrated from monolithic bundle_helpers to the plugin architecture.
// Tensor naming is now an internal concern of TextGenerationPipeline, validated
// by E2E tests rather than unit-level struct inspection.
// =============================================================================

#include "trtf/runtime/pipeline_plugin.h"

#include <iostream>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

// =============================================================================
// Config parsing tests — torchtrt_decoder strategy
// =============================================================================

// -----------------------------------------------------------------------------
// Intention: Verify that runtime_strategy="torchtrt_decoder" is parsed correctly
//            and all standard decoder fields are populated.
// Setup:     JSON config mimicking a Torch-TRT built Qwen3-0.6B bundle.
// Mechanism: parse_base_config, assert strategy and architecture fields.
// -----------------------------------------------------------------------------
static void test_torchtrt_decoder_config_basic() {
    const std::string config = R"({
        "vocab_size": 151936,
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "runtime_strategy": "torchtrt_decoder",
        "max_position_embeddings": 32768
    })";

    const auto cfg = trtf::parse_base_config(config, 256);
    check(cfg.runtime_strategy == "torchtrt_decoder", "torchtrt_decoder: runtime_strategy parsed");
    check(cfg.vocab_size == 151936, "torchtrt_decoder: vocab_size");
    check(cfg.hidden_size == 1024, "torchtrt_decoder: hidden_size");
    check(cfg.num_layers == 28, "torchtrt_decoder: num_layers");
    check(cfg.num_heads == 16, "torchtrt_decoder: num_heads");
    check(cfg.num_kv_heads == 2, "torchtrt_decoder: num_kv_heads");
    check(cfg.head_dim == 64, "torchtrt_decoder: head_dim");
    check(cfg.attention_size == 16 * 64, "torchtrt_decoder: attention_size = 1024");
    check(cfg.max_cache_length == 256, "torchtrt_decoder: max_cache_length (override)");
}

// -----------------------------------------------------------------------------
// Intention: Verify that torchtrt_decoder with GQA (num_kv_heads < num_heads)
//            computes attention_size from num_attention_heads * head_dim, not
//            num_kv_heads * head_dim. This matches the raw TRT cache format
//            where GQA heads are expanded.
// Setup:     Config with num_attention_heads=16, num_key_value_heads=2 (GQA 8:1).
// Mechanism: Assert attention_size = 16 * 64 = 1024, not 2 * 64 = 128.
// -----------------------------------------------------------------------------
static void test_torchtrt_decoder_gqa_attention_size() {
    const std::string config = R"({
        "vocab_size": 32000,
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "runtime_strategy": "torchtrt_decoder"
    })";

    const auto cfg = trtf::parse_base_config(config, 128);
    // attention_size must use num_heads (expanded), not num_kv_heads (compact)
    check(cfg.attention_size == 16 * 64,
          "torchtrt_decoder GQA: attention_size = num_heads * head_dim = 1024");
    check(cfg.attention_size != 2 * 64,
          "torchtrt_decoder GQA: attention_size != num_kv_heads * head_dim");
}

// -----------------------------------------------------------------------------
// Intention: Verify that torchtrt_decoder with MHA (num_kv_heads == num_heads)
//            works correctly — no GQA expansion needed.
// Setup:     Config with num_attention_heads=16, num_key_value_heads=16 (MHA).
// Mechanism: Assert attention_size = 16 * 128 = 2048.
// -----------------------------------------------------------------------------
static void test_torchtrt_decoder_mha() {
    const std::string config = R"({
        "vocab_size": 32000,
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 128,
        "runtime_strategy": "torchtrt_decoder"
    })";

    const auto cfg = trtf::parse_base_config(config, 512);
    check(cfg.num_heads == cfg.num_kv_heads, "torchtrt_decoder MHA: heads == kv_heads");
    check(cfg.attention_size == 16 * 128, "torchtrt_decoder MHA: attention_size = 2048");
}

// -----------------------------------------------------------------------------
// Intention: Verify tokenizer_add_special_tokens field is parsed from bundle
//            config. The Torch-TRT builder detects this at build time.
// Setup:     Config with tokenizer_add_special_tokens=1.
// Mechanism: Assert the field is parsed correctly.
// -----------------------------------------------------------------------------
static void test_torchtrt_decoder_tokenizer_config() {
    const std::string config = R"({
        "vocab_size": 32000,
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "runtime_strategy": "torchtrt_decoder",
        "tokenizer_add_special_tokens": 1,
        "bos_token_id": 151643,
        "eos_token_id": [151645, 151643]
    })";

    const auto cfg = trtf::parse_base_config(config, 128);
    check(cfg.tokenizer_add_special_tokens == true,
          "torchtrt_decoder: tokenizer_add_special_tokens = true");
    check(cfg.tokenizer_add_special_tokens_present == true,
          "torchtrt_decoder: tokenizer_add_special_tokens_present = true");
    check(cfg.id_bos == 151643, "torchtrt_decoder: bos_token_id");
    check(cfg.id_eos == 151645, "torchtrt_decoder: eos_token_id from array");
}

// -----------------------------------------------------------------------------
// Intention: Verify that max_cache_length respects the override, not the
//            config's max_position_embeddings, since Torch-TRT bundles
//            have max_cache_length baked in at build time.
// Setup:     Config with max_position_embeddings=32768, override=256.
// Mechanism: Assert max_cache_length = 256.
// -----------------------------------------------------------------------------
static void test_torchtrt_decoder_cache_length_override() {
    const std::string config = R"({
        "vocab_size": 32000,
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "max_position_embeddings": 32768,
        "runtime_strategy": "torchtrt_decoder"
    })";

    // Override = 256 (from bundle config's max_cache_length)
    const auto cfg = trtf::parse_base_config(config, 256);
    check(cfg.max_cache_length == 256, "torchtrt_decoder: cache_length override = 256");
}

// -----------------------------------------------------------------------------
// Intention: Verify that torchtrt_decoder config without max_cache_length
//            override still caps at 4096 (same as other decoder strategies).
// Setup:     Config with max_position_embeddings=131072, no override.
// Mechanism: Assert max_cache_length = 4096 (cap).
// -----------------------------------------------------------------------------
static void test_torchtrt_decoder_cache_length_cap() {
    const std::string config = R"({
        "vocab_size": 32000,
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "max_position_embeddings": 131072,
        "runtime_strategy": "torchtrt_decoder"
    })";

    const auto cfg = trtf::parse_base_config(config, -1);
    check(cfg.max_cache_length == 4096, "torchtrt_decoder: cache_length capped at 4096");
}

// Engine tensor naming tests were removed during the migration from monolithic
// bundle_helpers (make_decoder_engine) to the plugin architecture. The tensor
// naming convention (cache_kv_N / outputN for Torch-TRT, cache_k_N / present_k_N
// for raw TRT) is now an internal concern of TextGenerationPipeline, validated
// by E2E tests (test_e2e.py) rather than unit-level struct inspection.

int main() {
    // Config parsing tests (no TRT needed)
    test_torchtrt_decoder_config_basic();
    test_torchtrt_decoder_gqa_attention_size();
    test_torchtrt_decoder_mha();
    test_torchtrt_decoder_tokenizer_config();
    test_torchtrt_decoder_cache_length_override();
    test_torchtrt_decoder_cache_length_cap();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All torchtrt_decoder tests passed.\n";
    return 0;
}
