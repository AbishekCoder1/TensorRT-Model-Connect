#include "runtime/trt/mamba_decode_runtime.h"
#include "runtime/trt/trt_common.h"

#include <algorithm>
#include <cstring>
#include <string_view>

namespace trtf {

#if TRTF_HAS_TRT

bool has_all_required_mamba_tensors(const MambaStepEngine& engine)
{
    if (!has_io_tensor(*engine.engine, engine.token_input_name)
        || !has_io_tensor(*engine.engine, engine.logits_output_name))
    {
        return false;
    }

    for (int32_t i = 0; i < engine.num_layers; ++i)
    {
        const auto idx = static_cast<std::size_t>(i);
        if (!has_io_tensor(*engine.engine, engine.conv_state_input_names[idx])
            || !has_io_tensor(*engine.engine, engine.ssm_state_input_names[idx])
            || !has_io_tensor(*engine.engine, engine.present_conv_output_names[idx])
            || !has_io_tensor(*engine.engine, engine.present_ssm_output_names[idx]))
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

    if (static_cast<int32_t>(conv_state_by_layer.size()) != engine.num_layers
        || static_cast<int32_t>(ssm_state_by_layer.size()) != engine.num_layers)
    {
        return fail("invalid state layer count");
    }

    const auto conv_elems = static_cast<std::size_t>(engine.d_inner) * static_cast<std::size_t>(engine.conv_kernel);
    const auto ssm_elems = static_cast<std::size_t>(engine.d_inner) * static_cast<std::size_t>(engine.state_size);

    for (int32_t i = 0; i < engine.num_layers; ++i)
    {
        const auto idx = static_cast<std::size_t>(i);
        if (conv_state_by_layer[idx].size() != conv_elems
            || ssm_state_by_layer[idx].size() != ssm_elems)
        {
            return fail("invalid state tensor size");
        }
    }

    logits.assign(static_cast<std::size_t>(engine.vocab_size), 0.0F);
    present_conv_by_layer.assign(
        static_cast<std::size_t>(engine.num_layers),
        std::vector<float>(conv_elems, 0.0F));
    present_ssm_by_layer.assign(
        static_cast<std::size_t>(engine.num_layers),
        std::vector<float>(ssm_elems, 0.0F));

    CudaStream stream;
    if (!stream.ok())
    {
        return fail("cudaStreamCreate failed");
    }

    struct PendingCopy {
        void* host_ptr{nullptr};
        void* device_ptr{nullptr};
        std::size_t bytes{0};
    };

    std::vector<std::unique_ptr<CudaBuffer>> device_buffers;
    std::vector<PendingCopy> output_copies;

    auto bind_input = [&](const std::string& name, const void* host_ptr, std::size_t bytes) -> bool {
        const auto location = engine.engine->getTensorLocation(name.c_str());
        if (location == nvinfer1::TensorLocation::kHOST)
        {
            return engine.context->setTensorAddress(name.c_str(), const_cast<void*>(host_ptr));
        }

        auto buffer = std::make_unique<CudaBuffer>(bytes);
        if (!buffer->ok())
        {
            return false;
        }
        if (cudaMemcpyAsync(buffer->data(), host_ptr, bytes, cudaMemcpyHostToDevice, stream.get()) != cudaSuccess)
        {
            return false;
        }

        void* device_ptr = buffer->data();
        if (!engine.context->setTensorAddress(name.c_str(), device_ptr))
        {
            return false;
        }

        device_buffers.push_back(std::move(buffer));
        return true;
    };

    auto bind_output = [&](const std::string& name, void* host_ptr, std::size_t bytes) -> bool {
        const auto location = engine.engine->getTensorLocation(name.c_str());
        if (location == nvinfer1::TensorLocation::kHOST)
        {
            return engine.context->setTensorAddress(name.c_str(), host_ptr);
        }

        auto buffer = std::make_unique<CudaBuffer>(bytes);
        if (!buffer->ok())
        {
            return false;
        }

        void* device_ptr = buffer->data();
        if (!engine.context->setTensorAddress(name.c_str(), device_ptr))
        {
            return false;
        }

        output_copies.push_back(PendingCopy{host_ptr, device_ptr, bytes});
        device_buffers.push_back(std::move(buffer));
        return true;
    };

    const std::size_t conv_bytes = conv_elems * sizeof(float);
    const std::size_t ssm_bytes = ssm_elems * sizeof(float);
    const std::size_t logits_bytes = logits.size() * sizeof(float);

    // Bind token input
    if (!bind_input(engine.token_input_name, &token_id, sizeof(token_id)))
    {
        return fail("bind input token failed");
    }

    // Bind conv/ssm state inputs
    for (int32_t layer = 0; layer < engine.num_layers; ++layer)
    {
        const auto idx = static_cast<std::size_t>(layer);
        if (!bind_input(engine.conv_state_input_names[idx], conv_state_by_layer[idx].data(), conv_bytes))
        {
            return fail("bind input conv_state failed");
        }
        if (!bind_input(engine.ssm_state_input_names[idx], ssm_state_by_layer[idx].data(), ssm_bytes))
        {
            return fail("bind input ssm_state failed");
        }
    }

    // Bind logits output
    if (!bind_output(engine.logits_output_name, logits.data(), logits_bytes))
    {
        return fail("bind output logits failed");
    }

    // Bind conv/ssm state outputs
    for (int32_t layer = 0; layer < engine.num_layers; ++layer)
    {
        const auto idx = static_cast<std::size_t>(layer);
        if (!bind_output(engine.present_conv_output_names[idx], present_conv_by_layer[idx].data(), conv_bytes))
        {
            return fail("bind output present_conv failed");
        }
        if (!bind_output(engine.present_ssm_output_names[idx], present_ssm_by_layer[idx].data(), ssm_bytes))
        {
            return fail("bind output present_ssm failed");
        }
    }

    if (!engine.context->enqueueV3(stream.get()))
    {
        return fail("enqueueV3 failed");
    }

    for (const PendingCopy& copy : output_copies)
    {
        if (cudaMemcpyAsync(copy.host_ptr, copy.device_ptr, copy.bytes, cudaMemcpyDeviceToHost, stream.get())
            != cudaSuccess)
        {
            return fail("cudaMemcpyAsync output failed");
        }
    }

    if (cudaStreamSynchronize(stream.get()) != cudaSuccess)
    {
        return fail("cudaStreamSynchronize failed");
    }

    return true;
}

#endif // TRTF_HAS_TRT

} // namespace trtf
