#include "runtime/trt/hybrid_backend.h"
#include "runtime/trt/trt_common.h"
#include "runtime/trt/trt_decode_runtime.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

bool has_all_required_hybrid_tensors(const HybridStepEngine& engine)
{
    if (!has_io_tensor(*engine.engine, engine.token_input_name)
        || !has_io_tensor(*engine.engine, engine.position_input_name)
        || !has_io_tensor(*engine.engine, engine.mask_input_name)
        || !has_io_tensor(*engine.engine, engine.logits_output_name))
    {
        return false;
    }

    for (int32_t i = 0; i < engine.num_mamba_layers; ++i)
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

    for (int32_t i = 0; i < engine.num_attention_layers; ++i)
    {
        const auto idx = static_cast<std::size_t>(i);
        if (!has_io_tensor(*engine.engine, engine.cache_k_input_names[idx])
            || !has_io_tensor(*engine.engine, engine.cache_v_input_names[idx])
            || !has_io_tensor(*engine.engine, engine.present_k_output_names[idx])
            || !has_io_tensor(*engine.engine, engine.present_v_output_names[idx]))
        {
            return false;
        }
    }
    return true;
}

namespace {

// Manages hybrid state: both Mamba-2 conv/ssm state and KV cache for attention layers.
class HybridStepState {
public:
    HybridStepState(const HybridStepEngine& engine)
        : mNumMamba(engine.num_mamba_layers)
        , mNumAttn(engine.num_attention_layers)
        , mDInner(engine.d_inner)
        , mDState(engine.d_state)
        , mDConv(engine.d_conv)
        , mNHeads(engine.nheads)
        , mHeadDim(engine.head_dim)
        , mConvDim(engine.conv_dim > 0 ? engine.conv_dim : engine.d_inner)
        , mAttentionSize(engine.attention_size)
        , mMaxCacheLength(engine.max_cache_length)
        , mPosition(0)
    {
        // Mamba state: conv operates on conv_dim channels, SSM has [nheads, headdim, d_state]
        const auto conv_elems = static_cast<std::size_t>(mConvDim) * static_cast<std::size_t>(mDConv);
        const auto ssm_elems = static_cast<std::size_t>(mNHeads)
                             * static_cast<std::size_t>(mHeadDim > 0 ? mHeadDim : 1)
                             * static_cast<std::size_t>(mDState);
        mConvState.assign(static_cast<std::size_t>(mNumMamba),
                          std::vector<float>(conv_elems, 0.0F));
        mSsmState.assign(static_cast<std::size_t>(mNumMamba),
                         std::vector<float>(ssm_elems, 0.0F));

        // KV cache
        const auto cache_elems = static_cast<std::size_t>(mMaxCacheLength) *
                                  static_cast<std::size_t>(mAttentionSize);
        mCacheK.assign(static_cast<std::size_t>(mNumAttn),
                       std::vector<float>(cache_elems, 0.0F));
        mCacheV.assign(static_cast<std::size_t>(mNumAttn),
                       std::vector<float>(cache_elems, 0.0F));
    }

    const std::vector<std::vector<float>>& conv_state() const { return mConvState; }
    const std::vector<std::vector<float>>& ssm_state() const { return mSsmState; }
    const std::vector<std::vector<float>>& cache_k() const { return mCacheK; }
    const std::vector<std::vector<float>>& cache_v() const { return mCacheV; }

    int32_t position() const { return mPosition; }
    int32_t max_cache_length() const { return mMaxCacheLength; }
    int32_t attention_size() const { return mAttentionSize; }

    void update_after_step(
        const std::vector<std::vector<float>>& present_conv,
        const std::vector<std::vector<float>>& present_ssm,
        const std::vector<std::vector<float>>& present_k,
        const std::vector<std::vector<float>>& present_v)
    {
        // Update Mamba state
        for (int32_t i = 0; i < mNumMamba; ++i)
        {
            const auto idx = static_cast<std::size_t>(i);
            mConvState[idx] = present_conv[idx];
            mSsmState[idx] = present_ssm[idx];
        }

        // Update KV cache: append present K/V at current position
        const int32_t write_pos = mPosition % mMaxCacheLength;
        const auto attn_sz = static_cast<std::size_t>(mAttentionSize);
        for (int32_t i = 0; i < mNumAttn; ++i)
        {
            const auto idx = static_cast<std::size_t>(i);
            const auto offset = static_cast<std::size_t>(write_pos) * attn_sz;
            std::memcpy(mCacheK[idx].data() + offset, present_k[idx].data(),
                        attn_sz * sizeof(float));
            std::memcpy(mCacheV[idx].data() + offset, present_v[idx].data(),
                        attn_sz * sizeof(float));
        }
        ++mPosition;
    }

    // Build attention mask: 0.0 for valid positions, -1e4 for invalid
    std::vector<float> build_attention_mask() const
    {
        const int32_t window = mMaxCacheLength + 1;
        std::vector<float> mask(static_cast<std::size_t>(window), kMaskedScore);

        // Tokens at positions 0..min(mPosition, mMaxCacheLength)-1 are valid
        const int32_t valid_count = std::min(mPosition, mMaxCacheLength);

        if (mPosition <= mMaxCacheLength)
        {
            // No overflow: valid positions are 0..mPosition-1 (in cache) + current (last slot)
            for (int32_t i = 0; i < valid_count; ++i)
            {
                mask[static_cast<std::size_t>(i)] = 0.0F;
            }
            // Current token occupies the last slot (appended after cache)
            mask[static_cast<std::size_t>(window - 1)] = 0.0F;
        }
        else
        {
            // Overflow: all cache slots are valid + current token
            for (int32_t i = 0; i < mMaxCacheLength; ++i)
            {
                mask[static_cast<std::size_t>(i)] = 0.0F;
            }
            mask[static_cast<std::size_t>(window - 1)] = 0.0F;
        }
        return mask;
    }

private:
    int32_t mNumMamba;
    int32_t mNumAttn;
    int32_t mDInner;
    int32_t mDState;
    int32_t mDConv;
    int32_t mNHeads;
    int32_t mHeadDim;
    int32_t mConvDim;
    int32_t mAttentionSize;
    int32_t mMaxCacheLength;
    int32_t mPosition;

    std::vector<std::vector<float>> mConvState;  // [mamba_layer][conv_dim * d_conv]
    std::vector<std::vector<float>> mSsmState;   // [mamba_layer][nheads * head_dim * d_state]
    std::vector<std::vector<float>> mCacheK;     // [attn_layer][max_cache * attn_size]
    std::vector<std::vector<float>> mCacheV;     // [attn_layer][max_cache * attn_size]
};


bool run_hybrid_step(
    const HybridStepEngine& engine,
    int32_t token_id,
    int32_t position_id,
    const std::vector<float>& attention_mask,
    const std::vector<std::vector<float>>& conv_state_by_layer,
    const std::vector<std::vector<float>>& ssm_state_by_layer,
    const std::vector<std::vector<float>>& cache_k_by_layer,
    const std::vector<std::vector<float>>& cache_v_by_layer,
    std::vector<float>& logits,
    std::vector<std::vector<float>>& present_conv,
    std::vector<std::vector<float>>& present_ssm,
    std::vector<std::vector<float>>& present_k,
    std::vector<std::vector<float>>& present_v,
    std::string& error)
{
    auto fail = [&error](std::string_view stage) {
        error = std::string(stage);
        return false;
    };

    const auto effective_conv_dim = engine.conv_dim > 0 ? engine.conv_dim : engine.d_inner;
    const auto conv_elems = static_cast<std::size_t>(effective_conv_dim) * static_cast<std::size_t>(engine.d_conv);
    const auto ssm_elems = static_cast<std::size_t>(engine.nheads)
                         * static_cast<std::size_t>(engine.head_dim > 0 ? engine.head_dim : 1)
                         * static_cast<std::size_t>(engine.d_state);
    const auto cache_elems = static_cast<std::size_t>(engine.max_cache_length) *
                              static_cast<std::size_t>(engine.attention_size);
    const auto attn_out_elems = static_cast<std::size_t>(engine.attention_size);

    logits.assign(static_cast<std::size_t>(engine.vocab_size), 0.0F);

    present_conv.assign(static_cast<std::size_t>(engine.num_mamba_layers),
                        std::vector<float>(conv_elems, 0.0F));
    present_ssm.assign(static_cast<std::size_t>(engine.num_mamba_layers),
                       std::vector<float>(ssm_elems, 0.0F));
    present_k.assign(static_cast<std::size_t>(engine.num_attention_layers),
                     std::vector<float>(attn_out_elems, 0.0F));
    present_v.assign(static_cast<std::size_t>(engine.num_attention_layers),
                     std::vector<float>(attn_out_elems, 0.0F));

    CudaStream stream;
    if (!stream.ok()) return fail("cudaStreamCreate failed");

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
        if (!buffer->ok()) return false;
        if (cudaMemcpyAsync(buffer->data(), host_ptr, bytes, cudaMemcpyHostToDevice, stream.get()) != cudaSuccess)
            return false;
        if (!engine.context->setTensorAddress(name.c_str(), buffer->data()))
            return false;
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
        if (!buffer->ok()) return false;
        if (!engine.context->setTensorAddress(name.c_str(), buffer->data()))
            return false;
        output_copies.push_back(PendingCopy{host_ptr, buffer->data(), bytes});
        device_buffers.push_back(std::move(buffer));
        return true;
    };

    // Bind inputs
    if (!bind_input(engine.token_input_name, &token_id, sizeof(token_id)))
        return fail("bind token failed");
    if (!bind_input(engine.position_input_name, &position_id, sizeof(position_id)))
        return fail("bind position failed");
    if (!bind_input(engine.mask_input_name, attention_mask.data(),
                    attention_mask.size() * sizeof(float)))
        return fail("bind mask failed");

    const std::size_t conv_bytes = conv_elems * sizeof(float);
    const std::size_t ssm_bytes = ssm_elems * sizeof(float);
    const std::size_t cache_bytes = cache_elems * sizeof(float);
    const std::size_t logits_bytes = logits.size() * sizeof(float);
    const std::size_t attn_out_bytes = attn_out_elems * sizeof(float);

    // Mamba state inputs
    for (int32_t i = 0; i < engine.num_mamba_layers; ++i)
    {
        const auto idx = static_cast<std::size_t>(i);
        if (!bind_input(engine.conv_state_input_names[idx],
                        conv_state_by_layer[idx].data(), conv_bytes))
            return fail("bind conv_state failed");
        if (!bind_input(engine.ssm_state_input_names[idx],
                        ssm_state_by_layer[idx].data(), ssm_bytes))
            return fail("bind ssm_state failed");
    }

    // KV cache inputs
    for (int32_t i = 0; i < engine.num_attention_layers; ++i)
    {
        const auto idx = static_cast<std::size_t>(i);
        if (!bind_input(engine.cache_k_input_names[idx],
                        cache_k_by_layer[idx].data(), cache_bytes))
            return fail("bind cache_k failed");
        if (!bind_input(engine.cache_v_input_names[idx],
                        cache_v_by_layer[idx].data(), cache_bytes))
            return fail("bind cache_v failed");
    }

    // Bind outputs
    if (!bind_output(engine.logits_output_name, logits.data(), logits_bytes))
        return fail("bind logits failed");

    for (int32_t i = 0; i < engine.num_mamba_layers; ++i)
    {
        const auto idx = static_cast<std::size_t>(i);
        if (!bind_output(engine.present_conv_output_names[idx],
                         present_conv[idx].data(), conv_bytes))
            return fail("bind present_conv failed");
        if (!bind_output(engine.present_ssm_output_names[idx],
                         present_ssm[idx].data(), ssm_bytes))
            return fail("bind present_ssm failed");
    }

    for (int32_t i = 0; i < engine.num_attention_layers; ++i)
    {
        const auto idx = static_cast<std::size_t>(i);
        if (!bind_output(engine.present_k_output_names[idx],
                         present_k[idx].data(), attn_out_bytes))
            return fail("bind present_k failed");
        if (!bind_output(engine.present_v_output_names[idx],
                         present_v[idx].data(), attn_out_bytes))
            return fail("bind present_v failed");
    }

    // Execute
    if (!engine.context->enqueueV3(stream.get()))
        return fail("enqueueV3 failed");

    for (const PendingCopy& copy : output_copies)
    {
        if (cudaMemcpyAsync(copy.host_ptr, copy.device_ptr, copy.bytes,
                            cudaMemcpyDeviceToHost, stream.get()) != cudaSuccess)
            return fail("cudaMemcpyAsync output failed");
    }

    if (cudaStreamSynchronize(stream.get()) != cudaSuccess)
        return fail("cudaStreamSynchronize failed");

    return true;
}


// Hybrid Mamba-2/Attention TRT backend.
class HybridBackendFastPath final : public IGenerationBackend {
public:
    explicit HybridBackendFastPath(std::unique_ptr<HybridStepEngine> engine)
        : mEngine(std::move(engine))
    {
    }

    bool is_available() const override { return static_cast<bool>(mEngine); }
    const char* name() const override { return "trt_hybrid"; }

    std::vector<int32_t> generate(const std::vector<int32_t>& input_ids, const GenerationConfig& config) override
    {
        if (!mEngine)
            throw std::runtime_error("Hybrid TRT backend not initialized");

        std::vector<int32_t> output = input_ids;
        if (config.max_new_tokens == 0) return output;

        HybridStepState state(*mEngine);
        std::vector<float> logits;
        std::vector<std::vector<float>> present_conv, present_ssm, present_k, present_v;

        // Prefill: process input tokens one by one
        if (input_ids.size() > 1)
        {
            for (std::size_t i = 0; i + 1 < input_ids.size(); ++i)
            {
                auto mask = state.build_attention_mask();
                std::string error;
                if (!run_hybrid_step(*mEngine, input_ids[i],
                        state.position(), mask,
                        state.conv_state(), state.ssm_state(),
                        state.cache_k(), state.cache_v(),
                        logits, present_conv, present_ssm, present_k, present_v,
                        error))
                {
                    throw std::runtime_error("hybrid prefill step failed: " + error);
                }
                state.update_after_step(present_conv, present_ssm, present_k, present_v);
            }
        }

        // Decode
        int32_t current_token = input_ids.empty() ? mEngine->id_bos : input_ids.back();
        for (std::size_t step = 0; step < config.max_new_tokens; ++step)
        {
            auto mask = state.build_attention_mask();
            std::string error;
            if (!run_hybrid_step(*mEngine, current_token,
                    state.position(), mask,
                    state.conv_state(), state.ssm_state(),
                    state.cache_k(), state.cache_v(),
                    logits, present_conv, present_ssm, present_k, present_v,
                    error))
            {
                throw std::runtime_error("hybrid decode step failed: " + error);
            }
            state.update_after_step(present_conv, present_ssm, present_k, present_v);
            const int32_t next_token = select_argmax_token(logits);
            output.push_back(next_token);
            current_token = next_token;
            if (next_token == mEngine->id_eos) break;
        }
        return output;
    }

private:
    std::unique_ptr<HybridStepEngine> mEngine;
};

} // namespace

std::unique_ptr<IGenerationBackend> CreateHybridBackendFromEngine(
    std::unique_ptr<HybridStepEngine> engine)
{
    return std::make_unique<HybridBackendFastPath>(std::move(engine));
}

#endif // TRTF_HAS_TRT

} // namespace trtf
