#include "runtime/trt/trt_decode_runtime.h"
#include "runtime/trt/trt_common.h"

#include <algorithm>
#include <cstring>
#include <string_view>

namespace trtf {

#if TRTF_HAS_TRT

int32_t select_argmax_token(const std::vector<float>& logits)
{
    if (logits.empty())
    {
        return 0;
    }
    const auto it = std::max_element(logits.begin(), logits.end());
    return static_cast<int32_t>(std::distance(logits.begin(), it));
}

std::vector<int32_t> select_topk_tokens(const std::vector<float>& logits, int32_t k)
{
    if (logits.empty() || k <= 0)
    {
        return {};
    }

    const int32_t capped = std::min(k, static_cast<int32_t>(logits.size()));
    std::vector<int32_t> indices(logits.size(), 0);
    for (std::size_t i = 0; i < indices.size(); ++i)
    {
        indices[i] = static_cast<int32_t>(i);
    }

    std::partial_sort(indices.begin(), indices.begin() + capped, indices.end(),
        [&](int32_t a, int32_t b) { return logits[static_cast<std::size_t>(a)] > logits[static_cast<std::size_t>(b)]; });
    indices.resize(static_cast<std::size_t>(capped));
    return indices;
}

std::vector<float> build_attention_mask(int32_t cache_length, int32_t max_cache_length, bool include_current_slot)
{
    const int32_t width = max_cache_length + (include_current_slot ? 1 : 0);
    if (width <= 0)
    {
        return {};
    }

    std::vector<float> mask(static_cast<std::size_t>(width), kMaskedScore);
    const int32_t valid = std::max(0, std::min(cache_length, max_cache_length));
    for (int32_t i = 0; i < valid; ++i)
    {
        mask[static_cast<std::size_t>(i)] = 0.0F;
    }

    if (include_current_slot)
    {
        mask.back() = 0.0F;
    }
    else if (valid <= 0)
    {
        mask[0] = 0.0F;
    }

    return mask;
}

void append_cache_state(
    std::vector<float>& cache, const std::vector<float>& state, int32_t hidden_size, int32_t max_cache_length,
    int32_t write_index)
{
    if (static_cast<int32_t>(state.size()) != hidden_size || max_cache_length <= 0)
    {
        return;
    }

    if (write_index < max_cache_length)
    {
        const std::size_t start = static_cast<std::size_t>(write_index) * static_cast<std::size_t>(hidden_size);
        std::copy(state.begin(), state.end(), cache.begin() + static_cast<std::ptrdiff_t>(start));
        return;
    }

    const std::size_t row_size = static_cast<std::size_t>(hidden_size);
    const std::size_t bytes_to_move
        = (static_cast<std::size_t>(max_cache_length - 1) * row_size) * sizeof(float);
    std::memmove(cache.data(), cache.data() + static_cast<std::ptrdiff_t>(row_size), bytes_to_move);

    const std::size_t tail = static_cast<std::size_t>(max_cache_length - 1) * row_size;
    std::copy(state.begin(), state.end(), cache.begin() + static_cast<std::ptrdiff_t>(tail));
}

bool run_decoder_step(const DecoderStepEngine& engine, int32_t token_id, int32_t position_id,
    const std::vector<std::vector<float>>& cache_k_by_layer, const std::vector<std::vector<float>>& cache_v_by_layer,
    const std::vector<float>& attention_mask, std::vector<float>& logits,
    std::vector<std::vector<float>>& present_k_by_layer, std::vector<std::vector<float>>& present_v_by_layer,
    std::string& error)
{
    auto fail = [&error](std::string_view stage) {
        error = std::string(stage);
        return false;
    };

    if (static_cast<int32_t>(cache_k_by_layer.size()) != engine.num_layers
        || static_cast<int32_t>(cache_v_by_layer.size()) != engine.num_layers
        || attention_mask.size() != static_cast<std::size_t>(engine.attention_mask_size))
    {
        return fail("invalid cache layer count");
    }

    const std::size_t expected_cache
        = static_cast<std::size_t>(engine.max_cache_length) * static_cast<std::size_t>(engine.cache_state_size);
    for (int32_t i = 0; i < engine.num_layers; ++i)
    {
        if (cache_k_by_layer[static_cast<std::size_t>(i)].size() != expected_cache
            || cache_v_by_layer[static_cast<std::size_t>(i)].size() != expected_cache)
        {
            return fail("invalid cache tensor size");
        }
    }

    logits.assign(static_cast<std::size_t>(engine.vocab_size), 0.0F);
    present_k_by_layer.assign(static_cast<std::size_t>(engine.num_layers),
        std::vector<float>(static_cast<std::size_t>(engine.cache_state_size), 0.0F));
    present_v_by_layer.assign(static_cast<std::size_t>(engine.num_layers),
        std::vector<float>(static_cast<std::size_t>(engine.cache_state_size), 0.0F));

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

    const std::size_t cache_bytes = expected_cache * sizeof(float);
    const std::size_t mask_bytes = attention_mask.size() * sizeof(float);
    const std::size_t logits_bytes = logits.size() * sizeof(float);
    const std::size_t state_bytes = static_cast<std::size_t>(engine.cache_state_size) * sizeof(float);

    if (!bind_input(engine.token_input_name, &token_id, sizeof(token_id)))
    {
        return fail("bind input token failed");
    }
    if (engine.requires_position_input
        && !bind_input(engine.position_input_name, &position_id, sizeof(position_id)))
    {
        return fail("bind input position_id failed");
    }
    if (!bind_input(engine.mask_input_name, attention_mask.data(), mask_bytes))
    {
        return fail("bind input attention_mask failed");
    }

    for (int32_t layer = 0; layer < engine.num_layers; ++layer)
    {
        const std::size_t idx = static_cast<std::size_t>(layer);
        if (!bind_input(engine.cache_k_input_names[idx], cache_k_by_layer[idx].data(), cache_bytes))
        {
            return fail("bind input cache_k failed");
        }
        if (!bind_input(engine.cache_v_input_names[idx], cache_v_by_layer[idx].data(), cache_bytes))
        {
            return fail("bind input cache_v failed");
        }
    }

    if (!bind_output(engine.logits_output_name, logits.data(), logits_bytes))
    {
        return fail("bind output logits failed");
    }

    for (int32_t layer = 0; layer < engine.num_layers; ++layer)
    {
        const std::size_t idx = static_cast<std::size_t>(layer);
        if (!bind_output(engine.present_k_output_names[idx], present_k_by_layer[idx].data(), state_bytes))
        {
            return fail("bind output present_k failed");
        }
        if (!bind_output(engine.present_v_output_names[idx], present_v_by_layer[idx].data(), state_bytes))
        {
            return fail("bind output present_v failed");
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
