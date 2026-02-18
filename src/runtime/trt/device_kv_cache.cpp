#include "runtime/trt/device_kv_cache.h"
#include "runtime/trt/trt_decode_runtime.h"

#include <algorithm>
#include <cstring>
#include <string_view>

namespace trtf {

#if TRTF_HAS_TRT

// ---------------------------------------------------------------------------
// DeviceKvCache
// ---------------------------------------------------------------------------

DeviceKvCache::DeviceKvCache(const DecoderStepEngine& engine)
    : mCacheStateSize(engine.cache_state_size)
    , mMaxCacheLength(engine.max_cache_length)
    , mNumLayers(engine.num_layers)
    , mIncludeCurrentSlot(engine.requires_position_input)
    , mPositionLimit(mIncludeCurrentSlot ? mMaxCacheLength : std::max(mMaxCacheLength - 1, 0))
{
    const std::size_t cache_bytes
        = static_cast<std::size_t>(mMaxCacheLength) * static_cast<std::size_t>(mCacheStateSize) * sizeof(float);

    mCacheK.reserve(static_cast<std::size_t>(mNumLayers));
    mCacheV.reserve(static_cast<std::size_t>(mNumLayers));
    for (int32_t i = 0; i < mNumLayers; ++i)
    {
        mCacheK.emplace_back(cache_bytes);
        mCacheV.emplace_back(cache_bytes);
    }

    // Zero-initialize (synchronous — only done once at construction)
    for (int32_t i = 0; i < mNumLayers; ++i)
    {
        const auto idx = static_cast<std::size_t>(i);
        if (mCacheK[idx].data() != nullptr)
        {
            cudaMemset(mCacheK[idx].data(), 0, cache_bytes);
        }
        if (mCacheV[idx].data() != nullptr)
        {
            cudaMemset(mCacheV[idx].data(), 0, cache_bytes);
        }
    }
}

void DeviceKvCache::prepare_step(int32_t& out_position_id, std::vector<float>& out_mask)
{
    out_position_id = std::min(mCacheLength, mPositionLimit);
    out_mask = build_attention_mask(mCacheLength, mMaxCacheLength, mIncludeCurrentSlot);
}

void DeviceKvCache::update_after_step(
    const std::vector<CudaBuffer>& present_k,
    const std::vector<CudaBuffer>& present_v,
    cudaStream_t stream)
{
    const std::size_t row_bytes = static_cast<std::size_t>(mCacheStateSize) * sizeof(float);

    auto copy_one = [&](CudaBuffer& cache_buf, const CudaBuffer& present_buf) {
        auto* cache_ptr = static_cast<char*>(cache_buf.data());
        const auto* present_ptr = present_buf.data();

        if (mCacheLength < mMaxCacheLength)
        {
            const std::size_t offset = static_cast<std::size_t>(mCacheLength) * row_bytes;
            cudaMemcpyAsync(cache_ptr + offset, present_ptr, row_bytes,
                cudaMemcpyDeviceToDevice, stream);
        }
        else
        {
            cudaMemcpyAsync(cache_ptr, cache_ptr + row_bytes,
                static_cast<std::size_t>(mMaxCacheLength - 1) * row_bytes,
                cudaMemcpyDeviceToDevice, stream);
            const std::size_t tail_offset = static_cast<std::size_t>(mMaxCacheLength - 1) * row_bytes;
            cudaMemcpyAsync(cache_ptr + tail_offset, present_ptr, row_bytes,
                cudaMemcpyDeviceToDevice, stream);
        }
    };

    for (int32_t layer = 0; layer < mNumLayers; ++layer)
    {
        const auto idx = static_cast<std::size_t>(layer);
        copy_one(mCacheK[idx], present_k[idx]);
        copy_one(mCacheV[idx], present_v[idx]);
    }

    mCacheLength = std::min(mCacheLength + 1, mMaxCacheLength);
}

void DeviceKvCache::reset(cudaStream_t stream)
{
    const std::size_t cache_bytes
        = static_cast<std::size_t>(mMaxCacheLength) * static_cast<std::size_t>(mCacheStateSize) * sizeof(float);

    for (int32_t i = 0; i < mNumLayers; ++i)
    {
        const auto idx = static_cast<std::size_t>(i);
        if (mCacheK[idx].data() != nullptr)
        {
            cudaMemsetAsync(mCacheK[idx].data(), 0, cache_bytes, stream);
        }
        if (mCacheV[idx].data() != nullptr)
        {
            cudaMemsetAsync(mCacheV[idx].data(), 0, cache_bytes, stream);
        }
    }
    mCacheLength = 0;
}

void* DeviceKvCache::cache_k_device_ptr(int32_t layer) const
{
    return mCacheK[static_cast<std::size_t>(layer)].data();
}

void* DeviceKvCache::cache_v_device_ptr(int32_t layer) const
{
    return mCacheV[static_cast<std::size_t>(layer)].data();
}

bool DeviceKvCache::ok() const
{
    for (const auto& buf : mCacheK)
    {
        if (!buf.ok()) return false;
    }
    for (const auto& buf : mCacheV)
    {
        if (!buf.ok()) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// DeviceResources
// ---------------------------------------------------------------------------

DeviceResources::DeviceResources(const DecoderStepEngine& engine)
    : d_token_id(sizeof(int32_t))
    , d_position_id(sizeof(int32_t))
    , d_mask(static_cast<std::size_t>(engine.attention_mask_size) * sizeof(float))
    , d_logits(static_cast<std::size_t>(engine.vocab_size) * sizeof(float))
    , d_input_embed(has_io_tensor(*engine.engine, "input_embed")
          ? static_cast<std::size_t>(std::max(engine.hidden_size, 1)) * sizeof(float)
          : 0)
    , d_use_input_embed(has_io_tensor(*engine.engine, "input_embed") ? sizeof(float) : 0)
{
    const std::size_t state_bytes = static_cast<std::size_t>(engine.cache_state_size) * sizeof(float);
    d_present_k.reserve(static_cast<std::size_t>(engine.num_layers));
    d_present_v.reserve(static_cast<std::size_t>(engine.num_layers));
    for (int32_t i = 0; i < engine.num_layers; ++i)
    {
        d_present_k.emplace_back(state_bytes);
        d_present_v.emplace_back(state_bytes);
    }
}

bool DeviceResources::ok() const
{
    if (!stream.ok() || !d_token_id.ok() || !d_position_id.ok()
        || !d_mask.ok() || !d_logits.ok())
    {
        return false;
    }
    if (d_input_embed.size() > 0 && !d_input_embed.ok()) return false;
    if (d_use_input_embed.size() > 0 && !d_use_input_embed.ok()) return false;
    for (const auto& buf : d_present_k)
    {
        if (!buf.ok()) return false;
    }
    for (const auto& buf : d_present_v)
    {
        if (!buf.ok()) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// run_decoder_step_device
// ---------------------------------------------------------------------------

bool run_decoder_step_device(
    const DecoderStepEngine& engine,
    DeviceKvCache& cache,
    DeviceResources& resources,
    int32_t token_id,
    std::vector<float>& logits,
    std::string& error,
    const float* input_embed_host,
    int32_t embed_dim,
    float use_input_embed)
{
    auto fail = [&error](std::string_view stage) {
        error = std::string(stage);
        return false;
    };

    cudaStream_t stream = resources.stream.get();

    // 1. Prepare step: compute position_id and mask on CPU
    int32_t position_id{};
    std::vector<float> mask;
    cache.prepare_step(position_id, mask);

    // 2. H2D: small inputs
    if (cudaMemcpyAsync(resources.d_token_id.data(), &token_id, sizeof(int32_t),
            cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        return fail("H2D token_id failed");
    }

    if (engine.requires_position_input)
    {
        if (cudaMemcpyAsync(resources.d_position_id.data(), &position_id, sizeof(int32_t),
                cudaMemcpyHostToDevice, stream) != cudaSuccess)
        {
            return fail("H2D position_id failed");
        }
    }

    const std::size_t mask_bytes = mask.size() * sizeof(float);
    if (cudaMemcpyAsync(resources.d_mask.data(), mask.data(), mask_bytes,
            cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        return fail("H2D mask failed");
    }

    // VL embed support
    if (has_io_tensor(*engine.engine, "input_embed"))
    {
        if (input_embed_host != nullptr && embed_dim > 0 && use_input_embed > 0.5F)
        {
            const std::size_t embed_bytes = static_cast<std::size_t>(embed_dim) * sizeof(float);
            if (cudaMemcpyAsync(resources.d_input_embed.data(), input_embed_host, embed_bytes,
                    cudaMemcpyHostToDevice, stream) != cudaSuccess)
            {
                return fail("H2D input_embed failed");
            }
        }
        else
        {
            cudaMemsetAsync(resources.d_input_embed.data(), 0, resources.d_input_embed.size(), stream);
            use_input_embed = 0.0F;
        }
        if (cudaMemcpyAsync(resources.d_use_input_embed.data(), &use_input_embed, sizeof(float),
                cudaMemcpyHostToDevice, stream) != cudaSuccess)
        {
            return fail("H2D use_input_embed failed");
        }
    }

    // 3. Bind tensor addresses
    if (!engine.context->setTensorAddress(engine.token_input_name.c_str(), resources.d_token_id.data()))
    {
        return fail("bind token_id failed");
    }
    if (engine.requires_position_input
        && !engine.context->setTensorAddress(engine.position_input_name.c_str(), resources.d_position_id.data()))
    {
        return fail("bind position_id failed");
    }
    if (!engine.context->setTensorAddress(engine.mask_input_name.c_str(), resources.d_mask.data()))
    {
        return fail("bind attention_mask failed");
    }
    if (!engine.context->setTensorAddress(engine.logits_output_name.c_str(), resources.d_logits.data()))
    {
        return fail("bind logits failed");
    }

    if (has_io_tensor(*engine.engine, "input_embed"))
    {
        if (!engine.context->setTensorAddress("input_embed", resources.d_input_embed.data()))
        {
            return fail("bind input_embed failed");
        }
        if (!engine.context->setTensorAddress("use_input_embed", resources.d_use_input_embed.data()))
        {
            return fail("bind use_input_embed failed");
        }
    }

    // Bind cache inputs (persistent device ptrs) and present outputs
    for (int32_t layer = 0; layer < engine.num_layers; ++layer)
    {
        const auto idx = static_cast<std::size_t>(layer);
        if (!engine.context->setTensorAddress(
                engine.cache_k_input_names[idx].c_str(), cache.cache_k_device_ptr(layer)))
        {
            return fail("bind cache_k failed");
        }
        if (!engine.context->setTensorAddress(
                engine.cache_v_input_names[idx].c_str(), cache.cache_v_device_ptr(layer)))
        {
            return fail("bind cache_v failed");
        }
        if (!engine.context->setTensorAddress(
                engine.present_k_output_names[idx].c_str(), resources.d_present_k[idx].data()))
        {
            return fail("bind present_k failed");
        }
        if (!engine.context->setTensorAddress(
                engine.present_v_output_names[idx].c_str(), resources.d_present_v[idx].data()))
        {
            return fail("bind present_v failed");
        }
    }

    // 4. Execute
    if (!engine.context->enqueueV3(stream))
    {
        return fail("enqueueV3 failed");
    }

    // 5. D2D cache update
    cache.update_after_step(resources.d_present_k, resources.d_present_v, stream);

    // 6. D2H logits
    logits.assign(static_cast<std::size_t>(engine.vocab_size), 0.0F);
    const std::size_t logits_bytes = logits.size() * sizeof(float);
    if (cudaMemcpyAsync(logits.data(), resources.d_logits.data(), logits_bytes,
            cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        return fail("D2H logits failed");
    }

    // 7. Sync
    if (cudaStreamSynchronize(stream) != cudaSuccess)
    {
        return fail("cudaStreamSynchronize failed");
    }

    return true;
}

#endif // TRTF_HAS_TRT

} // namespace trtf
