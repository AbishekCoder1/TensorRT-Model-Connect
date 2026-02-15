// =============================================================================
// Test suite: TRT engine cache key determinism and sensitivity
// =============================================================================
//
// Purpose:
//   Validates BuildTrtEngineCacheKey from engine_cache.cpp — the function that
//   produces a content-addressable hash key for a fully-populated
//   TrtDecoderDefinition. The cache key must be deterministic (same inputs
//   always produce the same key) and sensitive (any change to model parameters,
//   weights, or extra fields must produce a different key). This ensures that
//   stale engine plans are never reused when the model definition changes.
//
// Dependencies:
//   - utils/trt/engine_cache.h     (BuildTrtEngineCacheKey, TrtEngineCacheKeyParams)
//   - model/trt_model_definition.h (TrtDecoderDefinition, TrtDecoderLayerDefinition)
//
// Approach:
//   CPU-only tests that construct synthetic TrtDecoderDefinition structs via
//   helper factories (make_base_definition, make_base_params), modify one field
//   at a time, and compare the resulting cache keys. No disk I/O, no temp dirs,
//   no GPU required. Tests cover core fields (hidden_size), extension maps
//   (extra_int_params, extra_float_params, extra_tensors), insertion-order
//   independence for maps, and per-layer extra tensors.
// =============================================================================

#include "utils/trt/engine_cache.h"
#include "model/trt_model_definition.h"

#include <iostream>
#include <string>

namespace {

// ---------------------------------------------------------------------------
// Factory: builds a minimal but complete TrtDecoderDefinition with small
// dimensions (vocab=100, hidden=16, attn=16, mlp=32, 2 heads, no layers).
// All weight vectors are filled with constant values so that tests can make
// targeted single-field modifications and observe key changes.
// ---------------------------------------------------------------------------
trtf::TrtDecoderDefinition make_base_definition()
{
    trtf::TrtDecoderDefinition def;
    def.vocab_size = 100;
    def.hidden_size = 16;
    def.attention_size = 16;
    def.mlp_size = 32;
    def.max_cache_length = 4;
    def.id_bos = 1;
    def.id_eos = 2;
    def.has_decoder_layers = false;
    def.rms_norm_eps = 1e-5F;
    def.num_attention_heads = 2;
    def.num_key_value_heads = 2;
    def.rope_theta = 10000.0F;
    def.embedding.assign(1600, 0.1F);
    def.w_out.assign(1600, 0.2F);
    def.b_out.assign(100, 0.0F);
    return def;
}

// ---------------------------------------------------------------------------
// Factory: builds a minimal TrtEngineCacheKeyParams with default values
// (no position input, 1 layer).
// ---------------------------------------------------------------------------
trtf::TrtEngineCacheKeyParams make_base_params()
{
    trtf::TrtEngineCacheKeyParams params;
    params.requires_position_input = false;
    params.num_layers = 1;
    return params;
}

// -----------------------------------------------------------------------------
// Intention:  Verify that BuildTrtEngineCacheKey is deterministic — calling it
//             twice with identical definition and params returns the same
//             non-empty key.
// Setup:      A single base definition and base params (no modifications).
// Mechanism:  Calls BuildTrtEngineCacheKey twice, asserts both keys are equal
//             and non-empty.
// -----------------------------------------------------------------------------
bool test_deterministic()
{
    const auto def = make_base_definition();
    const auto params = make_base_params();
    const std::string key1 = trtf::BuildTrtEngineCacheKey(def, params);
    const std::string key2 = trtf::BuildTrtEngineCacheKey(def, params);
    if (key1 != key2)
    {
        std::cerr << "deterministic: key1=" << key1 << " key2=" << key2 << std::endl;
        return false;
    }
    if (key1.empty())
    {
        std::cerr << "deterministic: key is empty" << std::endl;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Intention:  Verify that changing the hidden_size field in the definition
//             produces a different cache key, ensuring the hash is sensitive to
//             core architectural parameters.
// Setup:      Two definitions: one with hidden_size=16 (default), one with
//             hidden_size=32.
// Mechanism:  Computes cache keys for both and asserts they differ.
// -----------------------------------------------------------------------------
bool test_different_hidden_size()
{
    auto def1 = make_base_definition();
    auto def2 = make_base_definition();
    def2.hidden_size = 32;
    const auto params = make_base_params();
    const std::string key1 = trtf::BuildTrtEngineCacheKey(def1, params);
    const std::string key2 = trtf::BuildTrtEngineCacheKey(def2, params);
    if (key1 == key2)
    {
        std::cerr << "different_hidden: keys match when they shouldn't" << std::endl;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Intention:  Verify that adding an entry to extra_int_params changes the cache
//             key. This is critical for MoE and other extension architectures
//             that store additional integer configuration.
// Setup:      Two definitions: one without extra_int_params, one with
//             num_experts=8.
// Mechanism:  Computes cache keys for both and asserts they differ.
// -----------------------------------------------------------------------------
bool test_different_extra_int_params()
{
    auto def1 = make_base_definition();
    auto def2 = make_base_definition();
    def2.extra_int_params["num_experts"] = 8;
    const auto params = make_base_params();
    const std::string key1 = trtf::BuildTrtEngineCacheKey(def1, params);
    const std::string key2 = trtf::BuildTrtEngineCacheKey(def2, params);
    if (key1 == key2)
    {
        std::cerr << "different_extra_int: keys match when they shouldn't" << std::endl;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Intention:  Verify that adding an entry to extra_float_params changes the
//             cache key, ensuring floating-point extension parameters are
//             included in the hash.
// Setup:      Two definitions: one without extra_float_params, one with
//             moe_gate_threshold=0.5.
// Mechanism:  Computes cache keys for both and asserts they differ.
// -----------------------------------------------------------------------------
bool test_different_extra_float_params()
{
    auto def1 = make_base_definition();
    auto def2 = make_base_definition();
    def2.extra_float_params["moe_gate_threshold"] = 0.5F;
    const auto params = make_base_params();
    const std::string key1 = trtf::BuildTrtEngineCacheKey(def1, params);
    const std::string key2 = trtf::BuildTrtEngineCacheKey(def2, params);
    if (key1 == key2)
    {
        std::cerr << "different_extra_float: keys match when they shouldn't" << std::endl;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Intention:  Verify that adding entries to the model-level extra_tensors map
//             changes the cache key, ensuring that additional weight data (e.g.
//             router weights for MoE) is reflected in the hash.
// Setup:      Two definitions: one without extra_tensors, one with a 3-element
//             "router_weight" tensor.
// Mechanism:  Computes cache keys for both and asserts they differ.
// -----------------------------------------------------------------------------
bool test_different_extra_tensors()
{
    auto def1 = make_base_definition();
    auto def2 = make_base_definition();
    def2.extra_tensors["router_weight"] = {1.0F, 2.0F, 3.0F};
    const auto params = make_base_params();
    const std::string key1 = trtf::BuildTrtEngineCacheKey(def1, params);
    const std::string key2 = trtf::BuildTrtEngineCacheKey(def2, params);
    if (key1 == key2)
    {
        std::cerr << "different_extra_tensors: keys match when they shouldn't" << std::endl;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Intention:  Verify that the insertion order of entries in extra_int_params
//             does not affect the cache key. Since std::map is ordered by key,
//             different insertion orders should yield the same iteration order
//             and thus the same hash.
// Setup:      Two definitions with identical extra_int_params entries
//             (alpha=1, beta=2, gamma=3) inserted in different orders.
// Mechanism:  Computes cache keys for both and asserts they are equal.
// -----------------------------------------------------------------------------
bool test_extra_params_order_independence()
{
    auto def1 = make_base_definition();
    def1.extra_int_params["alpha"] = 1;
    def1.extra_int_params["beta"] = 2;
    def1.extra_int_params["gamma"] = 3;

    auto def2 = make_base_definition();
    def2.extra_int_params["gamma"] = 3;
    def2.extra_int_params["alpha"] = 1;
    def2.extra_int_params["beta"] = 2;

    const auto params = make_base_params();
    const std::string key1 = trtf::BuildTrtEngineCacheKey(def1, params);
    const std::string key2 = trtf::BuildTrtEngineCacheKey(def2, params);
    if (key1 != key2)
    {
        std::cerr << "order_independence: key1=" << key1 << " key2=" << key2 << std::endl;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Intention:  Verify that adding extra_tensors at the per-layer level (inside a
//             TrtDecoderLayerDefinition) changes the overall cache key, ensuring
//             per-layer extension weights are included in the hash.
// Setup:      Two definitions, each with one decoder layer populated with
//             standard weight vectors. The second definition has an additional
//             "router_weight" entry in the layer's extra_tensors map.
// Mechanism:  Computes cache keys for both and asserts they differ.
// -----------------------------------------------------------------------------
bool test_layer_extra_tensors_change_key()
{
    auto def1 = make_base_definition();
    def1.has_decoder_layers = true;
    trtf::TrtDecoderLayerDefinition layer;
    layer.input_norm.assign(16, 1.0F);
    layer.w_q.assign(256, 0.1F);
    layer.w_k.assign(256, 0.1F);
    layer.w_v.assign(256, 0.1F);
    layer.w_o.assign(256, 0.1F);
    layer.post_attn_norm.assign(16, 1.0F);
    layer.w_gate.assign(512, 0.1F);
    layer.w_up.assign(512, 0.1F);
    layer.w_down.assign(512, 0.1F);
    def1.decoder_layers.push_back(layer);

    auto def2 = def1;
    def2.decoder_layers[0].extra_tensors["router_weight"] = {1.0F, 2.0F};

    const auto params = make_base_params();
    const std::string key1 = trtf::BuildTrtEngineCacheKey(def1, params);
    const std::string key2 = trtf::BuildTrtEngineCacheKey(def2, params);
    if (key1 == key2)
    {
        std::cerr << "layer_extra: keys match when they shouldn't" << std::endl;
        return false;
    }
    return true;
}

} // namespace

int main()
{
    bool all_passed = true;
    std::cout << "test_engine_cache_key:" << std::endl;

    const auto run = [&](const char* name, bool (*fn)()) {
        const bool ok = fn();
        std::cout << "  " << name << ": " << (ok ? "PASS" : "FAIL") << std::endl;
        all_passed &= ok;
    };

    run("deterministic", test_deterministic);
    run("different_hidden_size", test_different_hidden_size);
    run("different_extra_int_params", test_different_extra_int_params);
    run("different_extra_float_params", test_different_extra_float_params);
    run("different_extra_tensors", test_different_extra_tensors);
    run("extra_params_order_independence", test_extra_params_order_independence);
    run("layer_extra_tensors_change_key", test_layer_extra_tensors_change_key);

    if (all_passed)
    {
        std::cout << "test_engine_cache_key passed" << std::endl;
        return 0;
    }
    std::cerr << "test_engine_cache_key FAILED" << std::endl;
    return 1;
}
