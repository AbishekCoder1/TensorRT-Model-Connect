// Unit tests for src/utils/trt/engine_cache.cpp (BuildTrtEngineCacheKey determinism).
// CPU-only — tests the hash function, not actual engine caching.

#include "utils/trt/engine_cache.h"
#include "model/trt_model_definition.h"

#include <iostream>
#include <string>

namespace {

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

trtf::TrtEngineCacheKeyParams make_base_params()
{
    trtf::TrtEngineCacheKeyParams params;
    params.requires_position_input = false;
    params.num_layers = 1;
    return params;
}

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
