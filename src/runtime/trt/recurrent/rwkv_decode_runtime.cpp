#include "runtime/trt/recurrent/rwkv_decode_runtime.h"
#include "runtime/trt/core/trt_common.h"
#include "runtime/trt/recurrent/recurrent_step_contracts.h"
#include "runtime/trt/recurrent/recurrent_tensor_bindings.h"

#include <array>
#include <algorithm>
#include <string_view>

namespace trtf {

#if TRTF_HAS_TRT

namespace {

std::array<const std::string*, 10> rwkv_layer_tensor_names(
    const RwkvStepEngine& engine,
    std::size_t layer_idx)
{
    return {
        &engine.attn_state_input_names[layer_idx],
        &engine.ff_state_input_names[layer_idx],
        &engine.num_state_input_names[layer_idx],
        &engine.den_state_input_names[layer_idx],
        &engine.max_state_input_names[layer_idx],
        &engine.present_attn_output_names[layer_idx],
        &engine.present_ff_output_names[layer_idx],
        &engine.present_num_output_names[layer_idx],
        &engine.present_den_output_names[layer_idx],
        &engine.present_max_output_names[layer_idx]};
}

struct LayerInputBinding {
    const std::vector<std::string>* names{nullptr};
    const std::vector<std::vector<float>>* values{nullptr};
    const char* error{nullptr};
};

template <std::size_t N>
bool bind_rwkv_layer_inputs(
    const RwkvStepEngine& engine,
    CudaStream& stream,
    std::vector<std::unique_ptr<CudaBuffer>>& device_buffers,
    std::size_t state_bytes,
    const std::array<LayerInputBinding, N>& bindings,
    std::string& error)
{
    for (int32_t layer = 0; layer < engine.num_layers; ++layer)
    {
        const auto idx = static_cast<std::size_t>(layer);
        for (const auto& binding : bindings)
        {
            if (!bind_input_tensor(
                    engine,
                    stream,
                    device_buffers,
                    (*binding.names)[idx],
                    (*binding.values)[idx].data(),
                    state_bytes))
            {
                error = binding.error;
                return false;
            }
        }
    }
    return true;
}

struct LayerOutputBinding {
    const std::vector<std::string>* names{nullptr};
    std::vector<std::vector<float>>* values{nullptr};
    const char* error{nullptr};
};

template <std::size_t N>
bool bind_rwkv_layer_outputs(
    const RwkvStepEngine& engine,
    std::vector<std::unique_ptr<CudaBuffer>>& device_buffers,
    std::vector<PendingCopy>& output_copies,
    std::size_t state_bytes,
    const std::array<LayerOutputBinding, N>& bindings,
    std::string& error)
{
    for (int32_t layer = 0; layer < engine.num_layers; ++layer)
    {
        const auto idx = static_cast<std::size_t>(layer);
        for (const auto& binding : bindings)
        {
            if (!bind_output_tensor(
                    engine,
                    device_buffers,
                    output_copies,
                    (*binding.names)[idx],
                    (*binding.values)[idx].data(),
                    state_bytes))
            {
                error = binding.error;
                return false;
            }
        }
    }
    return true;
}

} // namespace

bool has_all_required_rwkv_tensors(const RwkvStepEngine& engine)
{
    const auto base_tensors = std::array<const std::string*, 2>{
        &engine.token_input_name,
        &engine.logits_output_name};
    if (!has_required_tensors(*engine.engine, base_tensors))
    {
        return false;
    }

    for (int32_t i = 0; i < engine.num_layers; ++i)
    {
        const auto idx = static_cast<std::size_t>(i);
        if (!has_required_tensors(*engine.engine, rwkv_layer_tensor_names(engine, idx)))
        {
            return false;
        }
    }
    return true;
}

bool run_rwkv_step(
    const RwkvStepEngine& engine,
    int32_t token_id,
    const std::vector<std::vector<float>>& attn_state_by_layer,
    const std::vector<std::vector<float>>& ff_state_by_layer,
    const std::vector<std::vector<float>>& num_state_by_layer,
    const std::vector<std::vector<float>>& den_state_by_layer,
    const std::vector<std::vector<float>>& max_state_by_layer,
    std::vector<float>& logits,
    std::vector<std::vector<float>>& present_attn_by_layer,
    std::vector<std::vector<float>>& present_ff_by_layer,
    std::vector<std::vector<float>>& present_num_by_layer,
    std::vector<std::vector<float>>& present_den_by_layer,
    std::vector<std::vector<float>>& present_max_by_layer,
    std::string& error)
{
    auto fail = [&error](std::string_view stage) {
        error = std::string(stage);
        return false;
    };

    const auto states = std::array<const std::vector<std::vector<float>>*, 5>{
        &attn_state_by_layer,
        &ff_state_by_layer,
        &num_state_by_layer,
        &den_state_by_layer,
        &max_state_by_layer};
    if (!validate_state_layer_count(states, engine.num_layers))
    {
        return fail("invalid state layer count");
    }

    // State tensor size: [1, hidden_size] flattened
    const auto state_elems = static_cast<std::size_t>(engine.hidden_size);

    const auto state_specs = std::array<StateTensorView, 5>{
        StateTensorView{&attn_state_by_layer, state_elems},
        StateTensorView{&ff_state_by_layer, state_elems},
        StateTensorView{&num_state_by_layer, state_elems},
        StateTensorView{&den_state_by_layer, state_elems},
        StateTensorView{&max_state_by_layer, state_elems}};
    if (!validate_state_tensor_sizes(state_specs, engine.num_layers))
    {
        return fail("invalid state tensor size");
    }

    initialize_rwkv_outputs(
        engine.num_layers,
        engine.vocab_size,
        state_elems,
        logits,
        present_attn_by_layer,
        present_ff_by_layer,
        present_num_by_layer,
        present_den_by_layer,
        present_max_by_layer);

    CudaStream stream;
    if (!stream.ok())
    {
        return fail("cudaStreamCreate failed");
    }

    std::vector<std::unique_ptr<CudaBuffer>> device_buffers;
    std::vector<PendingCopy> output_copies;

    const std::size_t state_bytes = state_elems * sizeof(float);
    const std::size_t logits_bytes = logits.size() * sizeof(float);

    // Bind token input
    if (!bind_input_tensor(engine, stream, device_buffers, engine.token_input_name, &token_id, sizeof(token_id)))
    {
        return fail("bind input token failed");
    }

    const auto state_input_bindings = std::array<LayerInputBinding, 5>{
        LayerInputBinding{&engine.attn_state_input_names, &attn_state_by_layer, "bind input attn_state failed"},
        LayerInputBinding{&engine.ff_state_input_names, &ff_state_by_layer, "bind input ff_state failed"},
        LayerInputBinding{&engine.num_state_input_names, &num_state_by_layer, "bind input num_state failed"},
        LayerInputBinding{&engine.den_state_input_names, &den_state_by_layer, "bind input den_state failed"},
        LayerInputBinding{&engine.max_state_input_names, &max_state_by_layer, "bind input max_state failed"}};
    if (!bind_rwkv_layer_inputs(
            engine,
            stream,
            device_buffers,
            state_bytes,
            state_input_bindings,
            error))
    {
        return false;
    }

    // Bind logits output
    if (!bind_output_tensor(engine, device_buffers, output_copies, engine.logits_output_name, logits.data(), logits_bytes))
    {
        return fail("bind output logits failed");
    }

    const auto state_output_bindings = std::array<LayerOutputBinding, 5>{
        LayerOutputBinding{&engine.present_attn_output_names, &present_attn_by_layer, "bind output present_attn failed"},
        LayerOutputBinding{&engine.present_ff_output_names, &present_ff_by_layer, "bind output present_ff failed"},
        LayerOutputBinding{&engine.present_num_output_names, &present_num_by_layer, "bind output present_num failed"},
        LayerOutputBinding{&engine.present_den_output_names, &present_den_by_layer, "bind output present_den failed"},
        LayerOutputBinding{&engine.present_max_output_names, &present_max_by_layer, "bind output present_max failed"}};
    if (!bind_rwkv_layer_outputs(
            engine,
            device_buffers,
            output_copies,
            state_bytes,
            state_output_bindings,
            error))
    {
        return false;
    }

    if (!execute_recurrent_step(engine, stream, output_copies, error))
    {
        return false;
    }

    return true;
}

#endif // TRTF_HAS_TRT

} // namespace trtf
