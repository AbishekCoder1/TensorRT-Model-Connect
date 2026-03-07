#include "runtime/trt/recurrent/mamba_decode_runtime.h"
#include "runtime/trt/core/trt_common.h"
#include "runtime/trt/recurrent/recurrent_step_contracts.h"
#include "runtime/trt/recurrent/recurrent_tensor_bindings.h"

#include <array>
#include <algorithm>
#include <string_view>

namespace trtf {

#if TRTF_HAS_TRT

namespace {

std::array<const std::string*, 4> mamba_layer_tensor_names(
    const MambaStepEngine& engine,
    std::size_t layer_idx)
{
    return {
        &engine.conv_state_input_names[layer_idx],
        &engine.ssm_state_input_names[layer_idx],
        &engine.present_conv_output_names[layer_idx],
        &engine.present_ssm_output_names[layer_idx]};
}

bool bind_mamba_state_inputs(
    const MambaStepEngine& engine,
    CudaStream& stream,
    std::vector<std::unique_ptr<CudaBuffer>>& device_buffers,
    const std::vector<std::vector<float>>& conv_state_by_layer,
    const std::vector<std::vector<float>>& ssm_state_by_layer,
    std::size_t conv_bytes,
    std::size_t ssm_bytes,
    std::string& error)
{
    for (int32_t layer = 0; layer < engine.num_layers; ++layer)
    {
        const auto idx = static_cast<std::size_t>(layer);
        if (!bind_input_tensor(
                engine,
                stream,
                device_buffers,
                engine.conv_state_input_names[idx],
                conv_state_by_layer[idx].data(),
                conv_bytes))
        {
            error = "bind input conv_state failed";
            return false;
        }
        if (!bind_input_tensor(
                engine,
                stream,
                device_buffers,
                engine.ssm_state_input_names[idx],
                ssm_state_by_layer[idx].data(),
                ssm_bytes))
        {
            error = "bind input ssm_state failed";
            return false;
        }
    }
    return true;
}

bool bind_mamba_state_outputs(
    const MambaStepEngine& engine,
    std::vector<std::unique_ptr<CudaBuffer>>& device_buffers,
    std::vector<PendingCopy>& output_copies,
    std::vector<std::vector<float>>& present_conv_by_layer,
    std::vector<std::vector<float>>& present_ssm_by_layer,
    std::size_t conv_bytes,
    std::size_t ssm_bytes,
    std::string& error)
{
    for (int32_t layer = 0; layer < engine.num_layers; ++layer)
    {
        const auto idx = static_cast<std::size_t>(layer);
        if (!bind_output_tensor(
                engine,
                device_buffers,
                output_copies,
                engine.present_conv_output_names[idx],
                present_conv_by_layer[idx].data(),
                conv_bytes))
        {
            error = "bind output present_conv failed";
            return false;
        }
        if (!bind_output_tensor(
                engine,
                device_buffers,
                output_copies,
                engine.present_ssm_output_names[idx],
                present_ssm_by_layer[idx].data(),
                ssm_bytes))
        {
            error = "bind output present_ssm failed";
            return false;
        }
    }
    return true;
}

} // namespace

bool has_all_required_mamba_tensors(const MambaStepEngine& engine)
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
        if (!has_required_tensors(*engine.engine, mamba_layer_tensor_names(engine, idx)))
        {
            return false;
        }
    }
    return true;
}

bool run_mamba_step(
    const MambaStepEngine& engine,
    int32_t token_id,
    const std::vector<std::vector<float>>& conv_state_by_layer,
    const std::vector<std::vector<float>>& ssm_state_by_layer,
    std::vector<float>& logits,
    std::vector<std::vector<float>>& present_conv_by_layer,
    std::vector<std::vector<float>>& present_ssm_by_layer,
    std::string& error)
{
    auto fail = [&error](std::string_view stage) {
        error = std::string(stage);
        return false;
    };

    const auto states = std::array<const std::vector<std::vector<float>>*, 2>{
        &conv_state_by_layer,
        &ssm_state_by_layer};
    if (!validate_state_layer_count(states, engine.num_layers))
    {
        return fail("invalid state layer count");
    }

    const auto conv_elems = static_cast<std::size_t>(engine.d_inner) * static_cast<std::size_t>(engine.conv_kernel);
    const auto ssm_elems = static_cast<std::size_t>(engine.d_inner) * static_cast<std::size_t>(engine.state_size);

    const auto state_specs = std::array<StateTensorView, 2>{
        StateTensorView{&conv_state_by_layer, conv_elems},
        StateTensorView{&ssm_state_by_layer, ssm_elems}};
    if (!validate_state_tensor_sizes(state_specs, engine.num_layers))
    {
        return fail("invalid state tensor size");
    }

    initialize_mamba_outputs(
        engine.num_layers,
        engine.vocab_size,
        conv_elems,
        ssm_elems,
        logits,
        present_conv_by_layer,
        present_ssm_by_layer);

    CudaStream stream;
    if (!stream.ok())
    {
        return fail("cudaStreamCreate failed");
    }

    std::vector<std::unique_ptr<CudaBuffer>> device_buffers;
    std::vector<PendingCopy> output_copies;

    const std::size_t conv_bytes = conv_elems * sizeof(float);
    const std::size_t ssm_bytes = ssm_elems * sizeof(float);
    const std::size_t logits_bytes = logits.size() * sizeof(float);

    // Bind token input
    if (!bind_input_tensor(engine, stream, device_buffers, engine.token_input_name, &token_id, sizeof(token_id)))
    {
        return fail("bind input token failed");
    }

    if (!bind_mamba_state_inputs(
            engine,
            stream,
            device_buffers,
            conv_state_by_layer,
            ssm_state_by_layer,
            conv_bytes,
            ssm_bytes,
            error))
    {
        return false;
    }

    // Bind logits output
    if (!bind_output_tensor(engine, device_buffers, output_copies, engine.logits_output_name, logits.data(), logits_bytes))
    {
        return fail("bind output logits failed");
    }

    if (!bind_mamba_state_outputs(
            engine,
            device_buffers,
            output_copies,
            present_conv_by_layer,
            present_ssm_by_layer,
            conv_bytes,
            ssm_bytes,
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
