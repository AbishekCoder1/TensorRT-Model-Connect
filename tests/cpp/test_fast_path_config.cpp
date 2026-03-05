// =============================================================================
// Test suite: FastPathModelConfig parsing from config.json
// =============================================================================
//
// Purpose:
//   Validates parse_fast_path_config() from cabi/config/fast_path_config.h — the
//   function that extracts model architecture parameters (head dimensions,
//   attention size, cache length, special token IDs) from a raw config.json
//   string. This parser is used by the zero-weight fast path (C ABI) to
//   determine engine shape parameters without loading the full DecoderModel.
//
// Dependencies:
//   - cabi/config/fast_path_config.h (parse_fast_path_config, FastPathModelConfig)
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

#include "cabi/config/fast_path_config.h"

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

// -----------------------------------------------------------------------------
// Intention:  Verify that vision_language runtime_strategy correctly parses
//             VL-specific config fields.
// Setup:      A JSON config with runtime_strategy="vision_language" and
//             VL-specific fields (has_vision_engine, embed_input, image_token_id).
// Mechanism:  Calls parse_fast_path_config and checks VL fields are populated.
// -----------------------------------------------------------------------------
static void test_vision_language_config()
{
    const std::string config = R"({
        "vocab_size": 32000,
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "runtime_strategy": "vision_language",
        "has_vision_engine": 1,
        "embed_input": 1,
        "image_token_id": 151655,
        "vision_output_dim": 2048,
        "fixed_image_size": 448,
        "num_image_pad_tokens": 256,
        "vl_prompt_template": "test {image_pads} {prompt}",
        "image_token_str": "<|image_pad|>",
        "max_position_embeddings": 4096
    })";

    const auto cfg = trtf::parse_fast_path_config(config, -1);
    check(cfg.runtime_strategy == "vision_language", "runtime_strategy = vision_language");
    check(cfg.has_vision_engine == true, "has_vision_engine = true");
    check(cfg.embed_input == true, "embed_input = true");
    check(cfg.image_token_id == 151655, "image_token_id = 151655");
    check(cfg.vision_output_dim == 2048, "vision_output_dim = 2048");
    check(cfg.fixed_image_size == 448, "fixed_image_size = 448");
    check(cfg.num_image_pad_tokens == 256, "num_image_pad_tokens = 256");
    check(cfg.vl_prompt_template == "test {image_pads} {prompt}", "vl_prompt_template parsed");
    check(cfg.image_token_str == "<|image_pad|>", "image_token_str parsed");
}

// -----------------------------------------------------------------------------
// Intention:  Verify that non-VL models don't accidentally parse VL fields.
// Setup:      A standard decoder config (runtime_strategy="decoder_kv_cache").
// Mechanism:  Calls parse_fast_path_config and checks VL fields are at defaults.
// -----------------------------------------------------------------------------
static void test_non_vl_config_defaults()
{
    const std::string config = R"({
        "vocab_size": 32000,
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "max_position_embeddings": 4096
    })";

    const auto cfg = trtf::parse_fast_path_config(config, -1);
    check(cfg.runtime_strategy == "decoder_kv_cache", "default strategy = decoder_kv_cache");
    check(cfg.has_vision_engine == false, "has_vision_engine = false (default)");
    check(cfg.embed_input == false, "embed_input = false (default)");
    check(cfg.image_token_id == -1, "image_token_id = -1 (default)");
    check(cfg.vision_output_dim == 0, "vision_output_dim = 0 (default)");
}

static void test_encoder_only_config()
{
    const std::string config = R"({
        "model_type": "bert",
        "vocab_size": 30522,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "runtime_strategy": "encoder_only",
        "type_vocab_size": 2,
        "max_position_embeddings": 512
    })";

    const auto cfg = trtf::parse_fast_path_config(config, 128);
    check(cfg.runtime_strategy == "encoder_only", "encoder_only: runtime_strategy");
    check(cfg.vocab_size == 30522, "encoder_only: vocab_size");
    check(cfg.hidden_size == 768, "encoder_only: hidden_size");
    check(cfg.num_layers == 12, "encoder_only: num_layers");
    check(cfg.num_heads == 12, "encoder_only: num_heads");
    check(cfg.type_vocab_size == 2, "encoder_only: type_vocab_size");
    check(cfg.max_cache_length == 128, "encoder_only: max_cache_length (override)");
    check(cfg.head_dim == 64, "encoder_only: head_dim = 768/12 = 64");
}

// -----------------------------------------------------------------------------
// Intention:  Verify that omni_multimodal runtime_strategy correctly parses
//             all Omni-specific config fields.
// Setup:      A JSON config with runtime_strategy="omni_multimodal" and
//             Omni-specific fields (talker, audio encoder, codebook config).
// Mechanism:  Calls parse_fast_path_config and checks all omni fields.
// -----------------------------------------------------------------------------
static void test_omni_multimodal_config()
{
    const std::string config = R"({
        "vocab_size": 128000,
        "hidden_size": 2048,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "runtime_strategy": "omni_multimodal",
        "audio_sample_rate": 24000,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "omni_talker_hidden_size": 1024,
        "omni_talker_num_layers": 12,
        "omni_talker_max_cache_length": 512,
        "omni_n_codebooks": 8,
        "omni_codebook_size": 2048,
        "omni_audio_embed_dim": 1280,
        "omni_audio_num_mel": 128,
        "omni_audio_num_layers": 32,
        "omni_audio_num_frames": 1500,
        "has_vision_engine": 1,
        "embed_input": 1,
        "image_token_id": 151655,
        "vision_output_dim": 2048,
        "max_position_embeddings": 4096
    })";

    const auto cfg = trtf::parse_fast_path_config(config, 256);
    check(cfg.runtime_strategy == "omni_multimodal", "omni: runtime_strategy");
    check(cfg.omni_sample_rate == 24000, "omni: sample_rate");
    check(cfg.omni_num_experts == 8, "omni: num_experts");
    check(cfg.omni_num_experts_per_tok == 2, "omni: num_experts_per_tok");
    check(cfg.omni_talker_hidden_size == 1024, "omni: talker_hidden_size");
    check(cfg.omni_talker_num_layers == 12, "omni: talker_num_layers");
    check(cfg.omni_talker_max_cache_length == 512, "omni: talker_max_cache_length");
    check(cfg.omni_n_codebooks == 8, "omni: n_codebooks");
    check(cfg.omni_codebook_size == 2048, "omni: codebook_size");
    check(cfg.omni_audio_embed_dim == 1280, "omni: audio_embed_dim");
    check(cfg.omni_audio_num_mel == 128, "omni: audio_num_mel");
    check(cfg.omni_audio_num_layers == 32, "omni: audio_num_layers");
    check(cfg.omni_audio_num_frames == 1500, "omni: audio_num_frames");
    check(cfg.has_vision_engine == true, "omni: has_vision_engine");
    check(cfg.embed_input == true, "omni: embed_input");
    check(cfg.image_token_id == 151655, "omni: image_token_id");
    check(cfg.vision_output_dim == 2048, "omni: vision_output_dim");
}

// -----------------------------------------------------------------------------
// Intention:  Verify that speech_to_speech runtime_strategy correctly parses
//             PersonaPlex/Moshi speech-specific config fields.
// Setup:      A JSON config with runtime_strategy="speech_to_speech" and
//             speech-specific fields (sample_rate, codebooks, depth transformer).
// Mechanism:  Calls parse_fast_path_config and checks all speech fields.
// -----------------------------------------------------------------------------
static void test_speech_to_speech_config()
{
    const std::string config = R"({
        "vocab_size": 32000,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "runtime_strategy": "speech_to_speech",
        "sample_rate": 24000,
        "num_codebooks": 8,
        "codebook_size": 2048,
        "frame_rate": 12.5,
        "depth_hidden_size": 1024,
        "depth_num_layers": 6,
        "depth_num_attention_heads": 16,
        "depth_num_key_value_heads": 4,
        "eos_token_id": 2,
        "speech_depth_temperature": 0.0,
        "speech_depth_top_k": 0,
        "speech_system_prompt": "",
        "speech_text_prompt_ids": [],
        "max_position_embeddings": 4096
    })";

    const auto cfg = trtf::parse_fast_path_config(config, 512);
    check(cfg.runtime_strategy == "speech_to_speech", "speech: runtime_strategy");
    check(cfg.speech_sample_rate == 24000, "speech: sample_rate");
    check(cfg.speech_num_codebooks == 8, "speech: num_codebooks");
    check(cfg.speech_codebook_size == 2048, "speech: codebook_size");
    check(cfg.speech_depth_hidden_size == 1024, "speech: depth_hidden_size");
    check(cfg.speech_depth_num_layers == 6, "speech: depth_num_layers");
    check(cfg.speech_depth_num_heads == 16, "speech: depth_num_heads");
    check(cfg.speech_depth_num_kv_heads == 4, "speech: depth_num_kv_heads");
    check(cfg.id_eos == 2, "speech: eos_token_id");
    check(cfg.speech_depth_temperature == 0.0F, "speech: depth_temperature");
    check(cfg.speech_depth_top_k == 0, "speech: depth_top_k");
    check(cfg.speech_system_prompt.empty(), "speech: system_prompt");
    check(cfg.speech_text_prompt_ids.empty(), "speech: text_prompt_ids empty");
}

// -----------------------------------------------------------------------------
// Intention:  Verify that hybrid_mamba_attention runtime_strategy correctly
//             parses all hybrid Mamba-2 + Attention config fields.
// Setup:      A JSON config with runtime_strategy="hybrid_mamba_attention" and
//             hybrid-specific fields (layer counts, Mamba-2 dims, layer_types).
// Mechanism:  Calls parse_fast_path_config and checks all hybrid fields.
// -----------------------------------------------------------------------------
static void test_hybrid_mamba_attention_config()
{
    const std::string config = R"({
        "vocab_size": 64256,
        "hidden_size": 4096,
        "num_hidden_layers": 56,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "runtime_strategy": "hybrid_mamba_attention",
        "num_mamba_layers": 27,
        "num_attention_layers": 4,
        "d_inner": 8192,
        "mamba_d_state": 128,
        "mamba_d_conv": 4,
        "mamba_nheads": 64,
        "layer_types": ["mamba2", "mlp", "attention", "mamba2"],
        "max_position_embeddings": 4096,
        "bos_token_id": 0,
        "eos_token_id": 1
    })";

    const auto cfg = trtf::parse_fast_path_config(config, 256);
    check(cfg.runtime_strategy == "hybrid_mamba_attention", "hybrid: runtime_strategy");
    check(cfg.vocab_size == 64256, "hybrid: vocab_size");
    check(cfg.hidden_size == 4096, "hybrid: hidden_size");
    check(cfg.num_layers == 56, "hybrid: num_layers");
    check(cfg.num_heads == 32, "hybrid: num_heads");
    check(cfg.num_kv_heads == 8, "hybrid: num_kv_heads");
    check(cfg.num_mamba_layers == 27, "hybrid: num_mamba_layers");
    check(cfg.num_attention_layers == 4, "hybrid: num_attention_layers");
    check(cfg.d_inner == 8192, "hybrid: d_inner");
    check(cfg.mamba_d_state == 128, "hybrid: mamba_d_state");
    check(cfg.mamba_d_conv == 4, "hybrid: mamba_d_conv");
    check(cfg.mamba_nheads == 64, "hybrid: mamba_nheads");
    check(cfg.max_cache_length == 256, "hybrid: max_cache_length (override)");
    check(cfg.id_bos == 0, "hybrid: bos_token_id");
    check(cfg.id_eos == 1, "hybrid: eos_token_id");

    // Verify layer_types array was parsed
    check(cfg.layer_types.size() == 4, "hybrid: layer_types has 4 entries");
    if (cfg.layer_types.size() == 4)
    {
        check(cfg.layer_types[0] == "mamba2", "hybrid: layer_types[0] = mamba2");
        check(cfg.layer_types[1] == "mlp", "hybrid: layer_types[1] = mlp");
        check(cfg.layer_types[2] == "attention", "hybrid: layer_types[2] = attention");
        check(cfg.layer_types[3] == "mamba2", "hybrid: layer_types[3] = mamba2");
    }
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
    test_vision_language_config();
    test_non_vl_config_defaults();
    test_encoder_only_config();
    test_omni_multimodal_config();
    test_speech_to_speech_config();
    test_hybrid_mamba_attention_config();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All fast path config tests passed.\n";
    return 0;
}
