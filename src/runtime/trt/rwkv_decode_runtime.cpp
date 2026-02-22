#include "runtime/trt/rwkv_decode_runtime.h"
#include "runtime/trt/trt_common.h"

#include <algorithm>
#include <cstring>
#include <string_view>

namespace trtf {

#if TRTF_HAS_TRT

bool has_all_required_rwkv_tensors(const RwkvStepEngine& engine)
{
    if (!has_io_tensor(*engine.engine, engine.token_input_name)
        || !has_io_tensor(*engine.engine, engine.logits_output_name))
    {
        return false;
    }

    for (int32_t i = 0; i < engine.num_layers; ++i)
    {
        const auto idx = static_cast<std::size_t>(i);
        if (!has_io_tensor(*engine.engine, engine.attn_state_input_names[idx])
            || !has_io_tensor(*engine.engine, engine.ff_state_input_names[idx])
            || !has_io_tensor(*engine.engine, engine.num_state_input_names[idx])
            || !has_io_tensor(*engine.engine, engine.den_state_input_names[idx])
            || !has_io_tensor(*engine.engine, engine.max_state_input_names[idx])
            || !has_io_tensor(*engine.engine, engine.present_attn_output_names[idx])
            || !has_io_tensor(*engine.engine, engine.present_ff_output_names[idx])
            || !has_io_tensor(*engine.engine, engine.present_num_output_names[idx])
            || !has_io_tensor(*engine.engine, engine.present_den_output_names[idx])
            || !has_io_tensor(*engine.engine, engine.present_max_output_names[idx]))
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

    if (static_cast<int32_t>(attn_state_by_layer.size()) != engine.num_layers
        || static_cast<int32_t>(ff_state_by_layer.size()) != engine.num_layers
        || static_cast<int32_t>(num_state_by_layer.size()) != engine.num_layers
        || static_cast<int32_t>(den_state_by_layer.size()) != engine.num_layers
        || static_cast<int32_t>(max_state_by_layer.size()) != engine.num_layers)
    {
        return fail("invalid state layer count");
    }

    // State tensor size: [1, hidden_size] flattened
    const auto state_elems = static_cast<std::size_t>(engine.hidden_size);

    for (int32_t i = 0; i < engine.num_layers; ++i)
    {
        const auto idx = static_cast<std::size_t>(i);
        if (attn_state_by_layer[idx].size() != state_elems
            || ff_state_by_layer[idx].size() != state_elems
            || num_state_by_layer[idx].size() != state_elems
            || den_state_by_layer[idx].size() != state_elems
            || max_state_by_layer[idx].size() != state_elems)
        {
            return fail("invalid state tensor size");
        }
    }

    logits.assign(static_cast<std::size_t>(engine.vocab_size), 0.0F);

    const auto num_layers_sz = static_cast<std::size_t>(engine.num_layers);
    present_attn_by_layer.assign(num_layers_sz, std::vector<float>(state_elems, 0.0F));
    present_ff_by_layer.assign(num_layers_sz, std::vector<float>(state_elems, 0.0F));
    present_num_by_layer.assign(num_layers_sz, std::vector<float>(state_elems, 0.0F));
    present_den_by_layer.assign(num_layers_sz, std::vector<float>(state_elems, 0.0F));
    present_max_by_layer.assign(num_layers_sz, std::vector<float>(state_elems, 0.0F));

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

    const std::size_t state_bytes = state_elems * sizeof(float);
    const std::size_t logits_bytes = logits.size() * sizeof(float);

    // Bind token input
    if (!bind_input(engine.token_input_name, &token_id, sizeof(token_id)))
    {
        return fail("bind input token failed");
    }

    // Bind state inputs (5 per layer)
    for (int32_t layer = 0; layer < engine.num_layers; ++layer)
    {
        const auto idx = static_cast<std::size_t>(layer);
        if (!bind_input(engine.attn_state_input_names[idx], attn_state_by_layer[idx].data(), state_bytes))
        {
            return fail("bind input attn_state failed");
        }
        if (!bind_input(engine.ff_state_input_names[idx], ff_state_by_layer[idx].data(), state_bytes))
        {
            return fail("bind input ff_state failed");
        }
        if (!bind_input(engine.num_state_input_names[idx], num_state_by_layer[idx].data(), state_bytes))
        {
            return fail("bind input num_state failed");
        }
        if (!bind_input(engine.den_state_input_names[idx], den_state_by_layer[idx].data(), state_bytes))
        {
            return fail("bind input den_state failed");
        }
        if (!bind_input(engine.max_state_input_names[idx], max_state_by_layer[idx].data(), state_bytes))
        {
            return fail("bind input max_state failed");
        }
    }

    // Bind logits output
    if (!bind_output(engine.logits_output_name, logits.data(), logits_bytes))
    {
        return fail("bind output logits failed");
    }

    // Bind state outputs (5 per layer)
    for (int32_t layer = 0; layer < engine.num_layers; ++layer)
    {
        const auto idx = static_cast<std::size_t>(layer);
        if (!bind_output(engine.present_attn_output_names[idx], present_attn_by_layer[idx].data(), state_bytes))
        {
            return fail("bind output present_attn failed");
        }
        if (!bind_output(engine.present_ff_output_names[idx], present_ff_by_layer[idx].data(), state_bytes))
        {
            return fail("bind output present_ff failed");
        }
        if (!bind_output(engine.present_num_output_names[idx], present_num_by_layer[idx].data(), state_bytes))
        {
            return fail("bind output present_num failed");
        }
        if (!bind_output(engine.present_den_output_names[idx], present_den_by_layer[idx].data(), state_bytes))
        {
            return fail("bind output present_den failed");
        }
        if (!bind_output(engine.present_max_output_names[idx], present_max_by_layer[idx].data(), state_bytes))
        {
            return fail("bind output present_max failed");
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
