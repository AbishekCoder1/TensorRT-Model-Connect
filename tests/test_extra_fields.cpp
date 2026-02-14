// Unit tests for Phase A extensibility: extra_tensors, extra_int_params, extra_float_params.
// Tests round-trip through StandardTrtModelDefinitionPopulator and find_extra_bindings.
// Requires mkdtemp for checkpoint fixture — runs in container, may fail in sandbox.

#include "trtf/model.h"
#include "model/trt_model_definition.h"
#include "model/standard_trt_model_definition_populator.h"
#include "runtime/trt/trt_engine_lifecycle.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool check_close(float actual, float expected, float atol, const char* label)
{
    if (std::abs(actual - expected) > atol)
    {
        std::cerr << label << ": actual=" << actual << " expected=" << expected << std::endl;
        return false;
    }
    return true;
}

bool test_extra_tensors_through_populator()
{
    // Create a minimal DecoderModel with extra_tensors in a decoder layer
    trtf::DecoderModel model;
    model.model_id = "test-extra";
    model.vocab.resize(10, "tok");
    model.max_cache_length = 4;
    model.has_checkpoint = true;
    model.checkpoint.hidden_size = 4;
    model.checkpoint.attention_size = 4;
    model.checkpoint.mlp_size = 8;
    model.checkpoint.has_decoder_layers = true;
    model.checkpoint.embedding.assign(40, 0.1F);
    model.checkpoint.w_out.assign(40, 0.2F);
    model.checkpoint.b_out.assign(10, 0.0F);
    model.checkpoint.final_norm.assign(4, 1.0F);
    model.architecture.num_attention_heads = 2;
    model.architecture.num_key_value_heads = 2;
    model.architecture.rms_norm_eps = 1e-5F;
    model.architecture.rope_theta = 10000.0F;

    trtf::DecoderLayerCheckpoint layer;
    layer.input_norm.assign(4, 1.0F);
    layer.w_q.assign(16, 0.1F);
    layer.w_k.assign(16, 0.1F);
    layer.w_v.assign(16, 0.1F);
    layer.w_o.assign(16, 0.1F);
    layer.post_attn_norm.assign(4, 1.0F);
    layer.w_gate.assign(32, 0.1F);
    layer.w_up.assign(32, 0.1F);
    layer.w_down.assign(32, 0.1F);
    layer.extra_tensors["router_weight"] = {1.0F, 2.0F, 3.0F};
    model.checkpoint.decoder_layers.push_back(layer);

    // Run through populator
    trtf::TrtDecoderDefinition definition;
    definition.vocab_size = 10;
    definition.hidden_size = 4;
    definition.attention_size = 4;
    definition.mlp_size = 8;
    definition.max_cache_length = 4;

    trtf::StandardTrtModelDefinitionPopulator populator;
    if (!populator.can_populate(model))
    {
        std::cerr << "extra_tensors: can_populate returned false" << std::endl;
        return false;
    }

    if (!populator.populate(definition, model))
    {
        std::cerr << "extra_tensors: populate returned false" << std::endl;
        return false;
    }

    if (definition.decoder_layers.size() != 1)
    {
        std::cerr << "extra_tensors: wrong layer count=" << definition.decoder_layers.size() << std::endl;
        return false;
    }

    const auto& extra = definition.decoder_layers[0].extra_tensors;
    auto it = extra.find("router_weight");
    if (it == extra.end())
    {
        std::cerr << "extra_tensors: router_weight not found in TRT definition" << std::endl;
        return false;
    }

    if (it->second.size() != 3)
    {
        std::cerr << "extra_tensors: wrong size=" << it->second.size() << std::endl;
        return false;
    }

    if (!check_close(it->second[0], 1.0F, 1e-6F, "router[0]") ||
        !check_close(it->second[1], 2.0F, 1e-6F, "router[1]") ||
        !check_close(it->second[2], 3.0F, 1e-6F, "router[2]"))
    {
        return false;
    }

    return true;
}

bool test_extra_int_params_in_architecture()
{
    trtf::DecoderArchitectureConfig arch;
    arch.extra_int_params["num_experts"] = 8;
    arch.extra_int_params["num_experts_per_tok"] = 2;

    if (arch.extra_int_params["num_experts"] != 8)
    {
        std::cerr << "extra_int: num_experts=" << arch.extra_int_params["num_experts"] << std::endl;
        return false;
    }
    if (arch.extra_int_params["num_experts_per_tok"] != 2)
    {
        std::cerr << "extra_int: num_experts_per_tok=" << arch.extra_int_params["num_experts_per_tok"] << std::endl;
        return false;
    }
    return true;
}

bool test_extra_float_params_in_architecture()
{
    trtf::DecoderArchitectureConfig arch;
    arch.extra_float_params["moe_gate_threshold"] = 0.5F;

    if (std::abs(arch.extra_float_params["moe_gate_threshold"] - 0.5F) > 1e-6F)
    {
        std::cerr << "extra_float: gate_threshold=" << arch.extra_float_params["moe_gate_threshold"] << std::endl;
        return false;
    }
    return true;
}

bool test_trt_definition_extra_params()
{
    trtf::TrtDecoderDefinition def;
    def.extra_int_params["num_experts"] = 8;
    def.extra_float_params["routing_scale"] = 1.5F;
    def.extra_tensors["global_router"] = {0.1F, 0.2F, 0.3F};

    if (def.extra_int_params.size() != 1 || def.extra_int_params["num_experts"] != 8)
    {
        std::cerr << "def_extra_int: wrong" << std::endl;
        return false;
    }
    if (def.extra_float_params.size() != 1)
    {
        std::cerr << "def_extra_float: wrong" << std::endl;
        return false;
    }
    if (def.extra_tensors.size() != 1 || def.extra_tensors["global_router"].size() != 3)
    {
        std::cerr << "def_extra_tensor: wrong" << std::endl;
        return false;
    }
    return true;
}

#if TRTF_HAS_TRT

bool test_find_extra_bindings()
{
    trtf::DecoderStepEngine engine;
    engine.extra_bindings = {
        {"mamba_ssm_state_0", "ssm_state_0", true, 64},
        {"mamba_ssm_state_1", "ssm_state_1", true, 64},
        {"mamba_conv_state_0", "conv_state_0", true, 32},
        {"output_ssm_state_0", "out_ssm_0", false, 64},
    };

    // Search input bindings with "mamba_ssm" prefix
    auto ssm_inputs = trtf::find_extra_bindings(engine, "mamba_ssm", true);
    if (ssm_inputs.size() != 2)
    {
        std::cerr << "find_extra: ssm_inputs size=" << ssm_inputs.size() << std::endl;
        return false;
    }

    // Search with non-matching prefix
    auto none = trtf::find_extra_bindings(engine, "nonexistent", true);
    if (!none.empty())
    {
        std::cerr << "find_extra: nonexistent not empty" << std::endl;
        return false;
    }

    // Search output bindings
    auto outputs = trtf::find_extra_bindings(engine, "output_ssm", false);
    if (outputs.size() != 1)
    {
        std::cerr << "find_extra: outputs size=" << outputs.size() << std::endl;
        return false;
    }

    // Verify pointer correctness
    if (outputs[0]->logical_name != "output_ssm_state_0")
    {
        std::cerr << "find_extra: output name=" << outputs[0]->logical_name << std::endl;
        return false;
    }

    return true;
}

#endif // TRTF_HAS_TRT

} // namespace

int main()
{
    bool all_passed = true;
    std::cout << "test_extra_fields:" << std::endl;

    const auto run = [&](const char* name, bool (*fn)()) {
        const bool ok = fn();
        std::cout << "  " << name << ": " << (ok ? "PASS" : "FAIL") << std::endl;
        all_passed &= ok;
    };

    run("extra_tensors_through_populator", test_extra_tensors_through_populator);
    run("extra_int_params_in_architecture", test_extra_int_params_in_architecture);
    run("extra_float_params_in_architecture", test_extra_float_params_in_architecture);
    run("trt_definition_extra_params", test_trt_definition_extra_params);
#if TRTF_HAS_TRT
    run("find_extra_bindings", test_find_extra_bindings);
#else
    std::cout << "  find_extra_bindings: SKIPPED (TRTF_HAS_TRT=0)" << std::endl;
#endif

    if (all_passed)
    {
        std::cout << "test_extra_fields passed" << std::endl;
        return 0;
    }
    std::cerr << "test_extra_fields FAILED" << std::endl;
    return 1;
}
