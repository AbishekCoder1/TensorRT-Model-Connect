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

#include <cmath>
#include <cstdlib>
#include <initializer_list>
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

static bool almost_equal(float lhs, float rhs, float eps = 1e-5F)
{
    return std::fabs(lhs - rhs) <= eps;
}

static void check_float_vector_eq(
    const std::vector<float>& actual,
    std::initializer_list<float> expected,
    const char* test_name)
{
    if (actual.size() != expected.size())
    {
        check(false, test_name);
        return;
    }

    std::size_t idx = 0;
    for (float exp : expected)
    {
        if (!almost_equal(actual[idx], exp))
        {
            check(false, test_name);
            return;
        }
        ++idx;
    }
    check(true, test_name);
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

static void test_decoder_alias_keys_and_decoder_layers_override()
{
    const std::string config = R"({
        "vocab_size": 50257,
        "n_embd": 1536,
        "n_layer": 24,
        "n_head": 24,
        "decoder_layers": 30
    })";

    const auto cfg = trtf::parse_fast_path_config(config, -1);
    check(cfg.hidden_size == 1536, "alias keys: hidden_size from n_embd");
    check(cfg.num_layers == 30, "alias keys: decoder_layers overrides n_layer");
    check(cfg.num_heads == 24, "alias keys: num_heads from n_head");
    check(cfg.num_kv_heads == 24, "alias keys: num_kv_heads defaults to num_heads");
    check(cfg.head_dim == 64, "alias keys: head_dim computed from hidden_size/num_heads");
    check(cfg.max_cache_length == 32, "alias keys: max_cache_length defaults to 32");
}

static void test_tokenizer_add_special_tokens_flag_parsed()
{
    const std::string disabled = R"({
        "hidden_size": 128,
        "num_attention_heads": 2,
        "runtime_strategy": "decoder_kv_cache",
        "tokenizer_add_special_tokens": 0
    })";
    const auto cfg_disabled = trtf::parse_fast_path_config(disabled, -1);
    check(cfg_disabled.tokenizer_add_special_tokens_present, "tokenizer flag: present when set to 0");
    check(!cfg_disabled.tokenizer_add_special_tokens, "tokenizer flag: 0 maps to false");

    const std::string enabled = R"({
        "hidden_size": 128,
        "num_attention_heads": 2,
        "runtime_strategy": "decoder_kv_cache",
        "tokenizer_add_special_tokens": 1
    })";
    const auto cfg_enabled = trtf::parse_fast_path_config(enabled, -1);
    check(cfg_enabled.tokenizer_add_special_tokens_present, "tokenizer flag: present when set to 1");
    check(cfg_enabled.tokenizer_add_special_tokens, "tokenizer flag: 1 maps to true");
}

static void test_ssm_recurrent_defaults_and_fallback()
{
    const std::string config = R"({
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "runtime_strategy": "ssm_recurrent",
        "state_size": 64,
        "conv_kernel": 7
    })";

    const auto cfg = trtf::parse_fast_path_config(config, -1);
    check(cfg.runtime_strategy == "ssm_recurrent", "ssm: runtime_strategy");
    check(cfg.d_inner == 2048, "ssm: d_inner falls back to hidden_size * 2");
    check(cfg.state_size == 64, "ssm: state_size parsed");
    check(cfg.conv_kernel == 7, "ssm: conv_kernel parsed");
}

static void test_speech_to_text_config()
{
    const std::string config = R"({
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "num_hidden_layers": 6,
        "runtime_strategy": "speech_to_text",
        "num_mel_bins": 128,
        "max_source_positions": 2048,
        "max_target_positions": 512,
        "encoder_layers": 12,
        "decoder_layers": 10,
        "mel_n_fft": 512,
        "mel_hop_length": 256,
        "mel_chunk_length": 40,
        "mel_sampling_rate": 22050,
        "mel_length": 3000,
        "eot_token_id": 50363,
        "decoder_start_token_ids": [50258, 50359]
    })";

    const auto cfg = trtf::parse_fast_path_config(config, -1);
    check(cfg.runtime_strategy == "speech_to_text", "speech_to_text: runtime_strategy");
    check(cfg.num_mel_bins == 128, "speech_to_text: num_mel_bins");
    check(cfg.max_source_positions == 2048, "speech_to_text: max_source_positions");
    check(cfg.max_target_positions == 512, "speech_to_text: max_target_positions");
    check(cfg.encoder_layers == 12, "speech_to_text: encoder_layers");
    check(cfg.decoder_layers == 10, "speech_to_text: decoder_layers");
    check(cfg.mel_n_fft == 512, "speech_to_text: mel_n_fft");
    check(cfg.mel_hop_length == 256, "speech_to_text: mel_hop_length");
    check(cfg.mel_chunk_length == 40, "speech_to_text: mel_chunk_length");
    check(cfg.mel_sampling_rate == 22050, "speech_to_text: mel_sampling_rate");
    check(cfg.mel_length == 3000, "speech_to_text: mel_length");
    check(cfg.eot_token_id == 50363, "speech_to_text: eot_token_id");
    check(cfg.decoder_start_token_ids.size() == 2, "speech_to_text: decoder_start_token_ids size");
    if (cfg.decoder_start_token_ids.size() == 2)
    {
        check(cfg.decoder_start_token_ids[0] == 50258, "speech_to_text: decoder_start_token_ids[0]");
        check(cfg.decoder_start_token_ids[1] == 50359, "speech_to_text: decoder_start_token_ids[1]");
    }
}

static void test_segmentation_config()
{
    const std::string config = R"({
        "hidden_size": 512,
        "num_attention_heads": 8,
        "runtime_strategy": "segmentation",
        "num_classes": 21,
        "input_image_h": 720,
        "input_image_w": 1280,
        "output_h": 180,
        "output_w": 320,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225]
    })";

    const auto cfg = trtf::parse_fast_path_config(config, -1);
    check(cfg.runtime_strategy == "segmentation", "segmentation: runtime_strategy");
    check(cfg.num_classes == 21, "segmentation: num_classes");
    check(cfg.input_image_h == 720, "segmentation: input_image_h");
    check(cfg.input_image_w == 1280, "segmentation: input_image_w");
    check(cfg.output_h == 180, "segmentation: output_h");
    check(cfg.output_w == 320, "segmentation: output_w");
    check_float_vector_eq(cfg.seg_image_mean, {0.485F, 0.456F, 0.406F}, "segmentation: image_mean");
    check_float_vector_eq(cfg.seg_image_std, {0.229F, 0.224F, 0.225F}, "segmentation: image_std");
}

static void test_prompted_segmentation_config()
{
    const std::string config = R"({
        "hidden_size": 512,
        "num_attention_heads": 8,
        "runtime_strategy": "prompted_segmentation",
        "input_image_h": 1024,
        "input_image_w": 1024,
        "sam_image_embedding_size": 64,
        "sam_decoder_hidden_size": 256,
        "sam_num_mask_outputs": 4,
        "sam_num_multimask_outputs": 3,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.25, 0.25, 0.25],
        "sam_point_embed_1": [0.1, 0.2],
        "sam_point_embed_0": [0.3, 0.4],
        "sam_not_a_point_embed": [0.5, 0.6],
        "sam_shared_image_pe": [0.7, 0.8]
    })";

    const auto cfg = trtf::parse_fast_path_config(config, -1);
    check(cfg.runtime_strategy == "prompted_segmentation", "sam: runtime_strategy");
    check(cfg.input_image_h == 1024, "sam: input_image_h");
    check(cfg.input_image_w == 1024, "sam: input_image_w");
    check(cfg.sam_image_embedding_size == 64, "sam: image_embedding_size");
    check(cfg.sam_decoder_hidden_size == 256, "sam: decoder_hidden_size");
    check(cfg.sam_num_mask_outputs == 4, "sam: num_mask_outputs");
    check(cfg.sam_num_multimask_outputs == 3, "sam: num_multimask_outputs");
    check_float_vector_eq(cfg.seg_image_mean, {0.5F, 0.5F, 0.5F}, "sam: image_mean");
    check_float_vector_eq(cfg.seg_image_std, {0.25F, 0.25F, 0.25F}, "sam: image_std");
    check_float_vector_eq(cfg.sam_point_embed_fg, {0.1F, 0.2F}, "sam: point_embed_fg");
    check_float_vector_eq(cfg.sam_point_embed_bg, {0.3F, 0.4F}, "sam: point_embed_bg");
    check_float_vector_eq(cfg.sam_not_a_point_embed, {0.5F, 0.6F}, "sam: not_a_point_embed");
    check_float_vector_eq(cfg.sam_shared_image_pe, {0.7F, 0.8F}, "sam: shared_image_pe");
}

static void test_text_to_audio_config_without_magpie()
{
    const std::string config = R"({
        "hidden_size": 768,
        "num_hidden_layers": 8,
        "num_attention_heads": 12,
        "runtime_strategy": "text_to_audio",
        "sample_rate": 24000,
        "semantic_vocab_size": 20000,
        "coarse_vocab_size": 2048,
        "fine_vocab_size": 4096,
        "n_coarse_codebooks": 4,
        "n_fine_codebooks": 10,
        "codec_seq_length": 1024,
        "fine_seq_length": 2048,
        "magpie_tts": 0
    })";

    const auto cfg = trtf::parse_fast_path_config(config, -1);
    check(cfg.runtime_strategy == "text_to_audio", "text_to_audio: runtime_strategy");
    check(cfg.audio_sample_rate == 24000, "text_to_audio: sample_rate");
    check(cfg.semantic_vocab_size == 20000, "text_to_audio: semantic_vocab_size");
    check(cfg.coarse_vocab_size == 2048, "text_to_audio: coarse_vocab_size");
    check(cfg.fine_vocab_size == 4096, "text_to_audio: fine_vocab_size");
    check(cfg.n_coarse_codebooks == 4, "text_to_audio: n_coarse_codebooks");
    check(cfg.n_fine_codebooks == 10, "text_to_audio: n_fine_codebooks");
    check(cfg.codec_seq_length == 1024, "text_to_audio: codec_seq_length");
    check(cfg.fine_seq_length == 2048, "text_to_audio: fine_seq_length");
    check(!cfg.is_magpie_tts, "text_to_audio: magpie_tts disabled");
}

static void test_text_to_audio_magpie_config()
{
    const std::string config = R"({
        "hidden_size": 1024,
        "num_hidden_layers": 12,
        "num_attention_heads": 16,
        "runtime_strategy": "text_to_audio",
        "sample_rate": 22050,
        "magpie_tts": 1,
        "magpie_num_codebooks": 16,
        "magpie_codebook_size": 4096,
        "magpie_fps": 25.0,
        "magpie_num_speakers": 32,
        "magpie_encoder_layers": 8,
        "magpie_decoder_layers": 14,
        "magpie_hidden_size": 1536,
        "magpie_text_vocab_size": 50000,
        "magpie_max_source_positions": 4096,
        "magpie_xa_n_heads": 2,
        "magpie_xa_d_head": 256,
        "magpie_temperature": 0.6,
        "magpie_cfg_scale": 2.5,
        "magpie_finished_limit_with_eot": 0
    })";

    const auto cfg = trtf::parse_fast_path_config(config, -1);
    check(cfg.is_magpie_tts, "magpie: enabled");
    check(cfg.magpie_num_codebooks == 16, "magpie: num_codebooks");
    check(cfg.magpie_codebook_size == 4096, "magpie: codebook_size");
    check(almost_equal(cfg.magpie_fps, 25.0F), "magpie: fps");
    check(cfg.magpie_num_speakers == 32, "magpie: num_speakers");
    check(cfg.magpie_encoder_layers == 8, "magpie: encoder_layers");
    check(cfg.magpie_decoder_layers == 14, "magpie: decoder_layers");
    check(cfg.magpie_hidden_size == 1536, "magpie: hidden_size");
    check(cfg.magpie_text_vocab_size == 50000, "magpie: text_vocab_size");
    check(cfg.magpie_max_source_positions == 4096, "magpie: max_source_positions");
    check(cfg.magpie_xa_n_heads == 2, "magpie: xa_n_heads");
    check(cfg.magpie_xa_d_head == 256, "magpie: xa_d_head");
    check(almost_equal(cfg.magpie_temperature, 0.6F), "magpie: temperature");
    check(almost_equal(cfg.magpie_cfg_scale, 2.5F), "magpie: cfg_scale");
    check(cfg.magpie_finished_limit_with_eot == 0, "magpie: finished_limit");
    check(cfg.audio_sample_rate == 22050, "magpie: sample_rate");
}

static void test_object_detection_config()
{
    const std::string config = R"({
        "hidden_size": 512,
        "num_attention_heads": 8,
        "runtime_strategy": "object_detection",
        "det_num_classes": 91,
        "det_input_h": 800,
        "det_input_w": 1333,
        "det_conf_threshold": 0.35,
        "det_nms_threshold": 0.55,
        "image_mean": [0.4, 0.5, 0.6],
        "image_std": [0.1, 0.2, 0.3]
    })";

    const auto cfg = trtf::parse_fast_path_config(config, -1);
    check(cfg.runtime_strategy == "object_detection", "object_detection: runtime_strategy");
    check(cfg.det_num_classes == 91, "object_detection: num_classes");
    check(cfg.det_input_h == 800, "object_detection: input_h");
    check(cfg.det_input_w == 1333, "object_detection: input_w");
    check(almost_equal(cfg.det_conf_threshold, 0.35F), "object_detection: conf_threshold");
    check(almost_equal(cfg.det_nms_threshold, 0.55F), "object_detection: nms_threshold");
    check_float_vector_eq(cfg.det_image_mean, {0.4F, 0.5F, 0.6F}, "object_detection: image_mean");
    check_float_vector_eq(cfg.det_image_std, {0.1F, 0.2F, 0.3F}, "object_detection: image_std");
}

static void test_neural_operator_config()
{
    const std::string config = R"({
        "hidden_size": 256,
        "num_attention_heads": 8,
        "runtime_strategy": "neural_operator",
        "operator_type": "fno",
        "num_sensors": 512,
        "spatial_dim": 3,
        "output_dim": 4,
        "in_channels": 5,
        "out_channels": 6,
        "hidden_channels": 128,
        "grid_h": 96,
        "grid_w": 160
    })";

    const auto cfg = trtf::parse_fast_path_config(config, -1);
    check(cfg.runtime_strategy == "neural_operator", "neural_operator: runtime_strategy");
    check(cfg.operator_type == "fno", "neural_operator: operator_type");
    check(cfg.num_sensors == 512, "neural_operator: num_sensors");
    check(cfg.spatial_dim == 3, "neural_operator: spatial_dim");
    check(cfg.output_dim == 4, "neural_operator: output_dim");
    check(cfg.fno_in_channels == 5, "neural_operator: in_channels");
    check(cfg.fno_out_channels == 6, "neural_operator: out_channels");
    check(cfg.fno_hidden_channels == 128, "neural_operator: hidden_channels");
    check(cfg.fno_grid_h == 96, "neural_operator: grid_h");
    check(cfg.fno_grid_w == 160, "neural_operator: grid_w");
}

static void test_embedding_config()
{
    const std::string config = R"({
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "runtime_strategy": "embedding",
        "embedding_dim": 1536
    })";

    const auto cfg = trtf::parse_fast_path_config(config, -1);
    check(cfg.runtime_strategy == "embedding", "embedding: runtime_strategy");
    check(cfg.embedding_dim == 1536, "embedding: embedding_dim");
}

static void test_reranking_config()
{
    const std::string config = R"({
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "runtime_strategy": "reranking",
        "is_reranker": 1
    })";

    const auto cfg = trtf::parse_fast_path_config(config, -1);
    check(cfg.runtime_strategy == "reranking", "reranking: runtime_strategy");
    check(cfg.is_reranker, "reranking: is_reranker");
}

static void test_speech_to_speech_prompt_ids_single_value()
{
    const std::string config = R"({
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "runtime_strategy": "speech_to_speech",
        "speech_text_prompt_ids": [32001],
        "delays": [1, 2, 3],
        "num_codebooks": 3,
        "codebook_size": 4096
    })";

    const auto cfg = trtf::parse_fast_path_config(config, -1);
    check(cfg.speech_text_prompt_ids.size() == 1, "speech_to_speech: prompt_ids parsed");
    if (cfg.speech_text_prompt_ids.size() == 1)
    {
        check(cfg.speech_text_prompt_ids[0] == 32001, "speech_to_speech: prompt_ids[0]");
    }
    check(cfg.speech_delays.size() == 3, "speech_to_speech: delays parsed");
    check(cfg.codec_n_codebooks == 3, "speech_to_speech: codec_n_codebooks mirrors num_codebooks");
    check(cfg.codebook_size == 4096, "speech_to_speech: codebook_size mirrored");
}

static void test_speech_to_speech_prompt_ids_malformed_array_is_ignored()
{
    // This intentionally malformed JSON exercises the parser's guard path.
    const std::string config = R"({
        "hidden_size": 512,
        "num_attention_heads": 8,
        "runtime_strategy": "speech_to_speech",
        "speech_text_prompt_ids": [32001
    })";

    const auto cfg = trtf::parse_fast_path_config(config, -1);
    check(cfg.runtime_strategy == "speech_to_speech", "speech_to_speech malformed: runtime_strategy");
    check(cfg.speech_text_prompt_ids.empty(), "speech_to_speech malformed: prompt_ids ignored");
}

static void test_diffusion_config()
{
    const std::string config = R"({
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "runtime_strategy": "diffusion",
        "num_text_encoders": 2,
        "scheduler": "flow_match_euler",
        "num_inference_steps": 30,
        "guidance_scale": 6.5,
        "flow_shift": 0.9,
        "use_dynamic_shifting": 1,
        "base_shift": 0.7,
        "max_shift": 1.3,
        "video_height": 720,
        "video_width": 1280,
        "video_num_frames": 97,
        "z_dim": 32,
        "scale_factor_temporal": 2,
        "scale_factor_spatial": 4,
        "dit_dim": 2048,
        "dit_num_heads": 16,
        "dit_num_layers": 40,
        "freq_dim": 320,
        "num_vae_caches": 8,
        "text_encoder_dim": 8192,
        "text_seq_len": 1024,
        "latents_mean": [0.1, 0.2, 0.3],
        "latents_std": [0.4, 0.5, 0.6],
        "patch_size": [2, 8, 8],
        "vae_model_id": "test-vae",
        "guidance_embeds": 1,
        "use_rope": 0,
        "vae_scaling_factor": 0.18215,
        "diffusion_backend_type": "z_image"
    })";

    const auto cfg = trtf::parse_fast_path_config(config, -1);
    check(cfg.runtime_strategy == "diffusion", "diffusion: runtime_strategy");
    check(cfg.num_text_encoders == 2, "diffusion: num_text_encoders");
    check(cfg.scheduler == "flow_match_euler", "diffusion: scheduler");
    check(cfg.num_inference_steps == 30, "diffusion: num_inference_steps");
    check(almost_equal(cfg.guidance_scale, 6.5F), "diffusion: guidance_scale");
    check(almost_equal(cfg.flow_shift, 0.9F), "diffusion: flow_shift");
    check(cfg.use_dynamic_shifting, "diffusion: use_dynamic_shifting");
    check(almost_equal(cfg.base_shift, 0.7F), "diffusion: base_shift");
    check(almost_equal(cfg.max_shift, 1.3F), "diffusion: max_shift");
    check(cfg.video_height == 720, "diffusion: video_height");
    check(cfg.video_width == 1280, "diffusion: video_width");
    check(cfg.video_num_frames == 97, "diffusion: video_num_frames");
    check(cfg.z_dim == 32, "diffusion: z_dim");
    check(cfg.scale_factor_temporal == 2, "diffusion: scale_factor_temporal");
    check(cfg.scale_factor_spatial == 4, "diffusion: scale_factor_spatial");
    check(cfg.dit_dim == 2048, "diffusion: dit_dim");
    check(cfg.dit_num_heads == 16, "diffusion: dit_num_heads");
    check(cfg.dit_num_layers == 40, "diffusion: dit_num_layers");
    check(cfg.freq_dim == 320, "diffusion: freq_dim");
    check(cfg.num_vae_caches == 8, "diffusion: num_vae_caches");
    check(cfg.text_encoder_dim == 8192, "diffusion: text_encoder_dim");
    check(cfg.text_seq_len == 1024, "diffusion: text_seq_len");
    check_float_vector_eq(cfg.latents_mean, {0.1F, 0.2F, 0.3F}, "diffusion: latents_mean");
    check_float_vector_eq(cfg.latents_std, {0.4F, 0.5F, 0.6F}, "diffusion: latents_std");
    check(cfg.patch_size.size() == 3, "diffusion: patch_size size");
    if (cfg.patch_size.size() == 3)
    {
        check(cfg.patch_size[0] == 2, "diffusion: patch_size[0]");
        check(cfg.patch_size[1] == 8, "diffusion: patch_size[1]");
        check(cfg.patch_size[2] == 8, "diffusion: patch_size[2]");
    }
    check(cfg.vae_model_id == "test-vae", "diffusion: vae_model_id");
    check(cfg.guidance_embeds, "diffusion: guidance_embeds");
    check(!cfg.use_rope, "diffusion: use_rope");
    check(almost_equal(cfg.vae_scaling_factor, 0.18215F), "diffusion: vae_scaling_factor");
    check(cfg.diffusion_backend_type == "z_image", "diffusion: backend_type");
}

static void test_unknown_strategy_skips_specialized_parsers()
{
    const std::string config = R"({
        "hidden_size": 512,
        "num_attention_heads": 8,
        "runtime_strategy": "custom_strategy",
        "num_classes": 999,
        "embedding_dim": 4096,
        "is_reranker": 1
    })";

    const auto cfg = trtf::parse_fast_path_config(config, -1);
    check(cfg.runtime_strategy == "custom_strategy", "unknown strategy: runtime_strategy preserved");
    check(cfg.num_classes == 150, "unknown strategy: segmentation parser not applied");
    check(cfg.embedding_dim == 0, "unknown strategy: embedding parser not applied");
    check(!cfg.is_reranker, "unknown strategy: reranking parser not applied");
}

int main()
{
    test_head_dim_explicit_in_config();
    test_head_dim_computed_fallback();
    test_attention_size_with_gqa();
    test_decoder_alias_keys_and_decoder_layers_override();
    test_tokenizer_add_special_tokens_flag_parsed();
    test_ssm_recurrent_defaults_and_fallback();
    test_max_cache_length_from_env();
    test_max_cache_length_capped_at_4096();
    test_eos_bos_from_array();
    test_eos_bos_from_scalar();
    test_speech_to_text_config();
    test_vision_language_config();
    test_non_vl_config_defaults();
    test_segmentation_config();
    test_prompted_segmentation_config();
    test_text_to_audio_config_without_magpie();
    test_text_to_audio_magpie_config();
    test_object_detection_config();
    test_neural_operator_config();
    test_encoder_only_config();
    test_embedding_config();
    test_reranking_config();
    test_omni_multimodal_config();
    test_speech_to_speech_prompt_ids_single_value();
    test_speech_to_speech_prompt_ids_malformed_array_is_ignored();
    test_speech_to_speech_config();
    test_hybrid_mamba_attention_config();
    test_diffusion_config();
    test_unknown_strategy_skips_specialized_parsers();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All fast path config tests passed.\n";
    return 0;
}
