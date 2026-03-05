#include "runtime/trt/recurrent/hybrid_backend.h"
#include "runtime/trt/core/trt_common.h"
#include "runtime/trt/core/trt_decode_runtime.h"

#include <array>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

namespace {

template <std::size_t N>
bool has_required_tensors(
    const nvinfer1::ICudaEngine& trt_engine,
    const std::array<const std::string*, N>& names)
{
    for (const std::string* name : names)
    {
        if (!has_io_tensor(trt_engine, *name))
        {
            return false;
        }
    }
    return true;
}

std::array<const std::string*, 4> hybrid_mamba_layer_tensor_names(
    const HybridStepEngine& engine,
    std::size_t layer_idx)
{
    return {
        &engine.conv_state_input_names[layer_idx],
        &engine.ssm_state_input_names[layer_idx],
        &engine.present_conv_output_names[layer_idx],
        &engine.present_ssm_output_names[layer_idx]};
}

std::array<const std::string*, 4> hybrid_attention_layer_tensor_names(
    const HybridStepEngine& engine,
    std::size_t layer_idx)
{
    return {
        &engine.cache_k_input_names[layer_idx],
        &engine.cache_v_input_names[layer_idx],
        &engine.present_k_output_names[layer_idx],
        &engine.present_v_output_names[layer_idx]};
}

} // namespace

bool has_all_required_hybrid_tensors(const HybridStepEngine& engine)
{
    const auto base_tensors = std::array<const std::string*, 4>{
        &engine.token_input_name,
        &engine.position_input_name,
        &engine.mask_input_name,
        &engine.logits_output_name};
    if (!has_required_tensors(*engine.engine, base_tensors))
    {
        return false;
    }

    for (int32_t i = 0; i < engine.num_mamba_layers; ++i)
    {
        const auto idx = static_cast<std::size_t>(i);
        if (!has_required_tensors(*engine.engine, hybrid_mamba_layer_tensor_names(engine, idx)))
        {
            return false;
        }
    }

    for (int32_t i = 0; i < engine.num_attention_layers; ++i)
    {
        const auto idx = static_cast<std::size_t>(i);
        if (!has_required_tensors(*engine.engine, hybrid_attention_layer_tensor_names(engine, idx)))
        {
            return false;
        }
    }
    return true;
}

namespace {

struct PendingCopy {
    void* host_ptr{nullptr};
    void* device_ptr{nullptr};
    std::size_t bytes{0};
};

struct HybridTensorSizes {
    std::size_t conv_elems{0};
    std::size_t ssm_elems{0};
    std::size_t cache_elems{0};
    std::size_t attn_out_elems{0};
};

HybridTensorSizes compute_hybrid_tensor_sizes(const HybridStepEngine& engine)
{
    const auto effective_conv_dim = engine.conv_dim > 0 ? engine.conv_dim : engine.d_inner;
    const auto conv_elems = static_cast<std::size_t>(effective_conv_dim) * static_cast<std::size_t>(engine.d_conv);
    const auto ssm_elems = static_cast<std::size_t>(engine.nheads)
                         * static_cast<std::size_t>(engine.head_dim > 0 ? engine.head_dim : 1)
                         * static_cast<std::size_t>(engine.d_state);
    const auto cache_elems = static_cast<std::size_t>(engine.max_cache_length) *
                             static_cast<std::size_t>(engine.attention_size);
    return HybridTensorSizes{
        conv_elems,
        ssm_elems,
        cache_elems,
        static_cast<std::size_t>(engine.attention_size)};
}

void initialize_hybrid_outputs(
    const HybridStepEngine& engine,
    const HybridTensorSizes& sizes,
    std::vector<float>& logits,
    std::vector<std::vector<float>>& present_conv,
    std::vector<std::vector<float>>& present_ssm,
    std::vector<std::vector<float>>& present_k,
    std::vector<std::vector<float>>& present_v)
{
    logits.assign(static_cast<std::size_t>(engine.vocab_size), 0.0F);
    present_conv.assign(
        static_cast<std::size_t>(engine.num_mamba_layers),
        std::vector<float>(sizes.conv_elems, 0.0F));
    present_ssm.assign(
        static_cast<std::size_t>(engine.num_mamba_layers),
        std::vector<float>(sizes.ssm_elems, 0.0F));
    present_k.assign(
        static_cast<std::size_t>(engine.num_attention_layers),
        std::vector<float>(sizes.attn_out_elems, 0.0F));
    present_v.assign(
        static_cast<std::size_t>(engine.num_attention_layers),
        std::vector<float>(sizes.attn_out_elems, 0.0F));
}

bool bind_input_tensor(
    const HybridStepEngine& engine,
    CudaStream& stream,
    std::vector<std::unique_ptr<CudaBuffer>>& device_buffers,
    const std::string& name,
    const void* host_ptr,
    std::size_t bytes)
{
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
    if (!engine.context->setTensorAddress(name.c_str(), buffer->data()))
    {
        return false;
    }

    device_buffers.push_back(std::move(buffer));
    return true;
}

bool bind_output_tensor(
    const HybridStepEngine& engine,
    std::vector<std::unique_ptr<CudaBuffer>>& device_buffers,
    std::vector<PendingCopy>& output_copies,
    const std::string& name,
    void* host_ptr,
    std::size_t bytes)
{
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
    if (!engine.context->setTensorAddress(name.c_str(), buffer->data()))
    {
        return false;
    }

    output_copies.push_back(PendingCopy{host_ptr, buffer->data(), bytes});
    device_buffers.push_back(std::move(buffer));
    return true;
}

struct LayerInputBinding {
    const std::vector<std::string>* names{nullptr};
    const std::vector<std::vector<float>>* values{nullptr};
    std::size_t bytes{0};
    const char* error{nullptr};
};

template <std::size_t N>
bool bind_layer_inputs(
    const HybridStepEngine& engine,
    CudaStream& stream,
    std::vector<std::unique_ptr<CudaBuffer>>& device_buffers,
    int32_t layer_count,
    const std::array<LayerInputBinding, N>& bindings,
    std::string& error)
{
    for (int32_t layer = 0; layer < layer_count; ++layer)
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
                    binding.bytes))
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
    std::size_t bytes{0};
    const char* error{nullptr};
};

template <std::size_t N>
bool bind_layer_outputs(
    const HybridStepEngine& engine,
    std::vector<std::unique_ptr<CudaBuffer>>& device_buffers,
    std::vector<PendingCopy>& output_copies,
    int32_t layer_count,
    const std::array<LayerOutputBinding, N>& bindings,
    std::string& error)
{
    for (int32_t layer = 0; layer < layer_count; ++layer)
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
                    binding.bytes))
            {
                error = binding.error;
                return false;
            }
        }
    }
    return true;
}

bool copy_outputs_to_host(const std::vector<PendingCopy>& output_copies, CudaStream& stream)
{
    for (const PendingCopy& copy : output_copies)
    {
        if (cudaMemcpyAsync(
                copy.host_ptr,
                copy.device_ptr,
                copy.bytes,
                cudaMemcpyDeviceToHost,
                stream.get())
            != cudaSuccess)
        {
            return false;
        }
    }
    return true;
}

bool execute_hybrid_step(
    const HybridStepEngine& engine,
    CudaStream& stream,
    const std::vector<PendingCopy>& output_copies,
    std::string& error)
{
    if (!engine.context->enqueueV3(stream.get()))
    {
        error = "enqueueV3 failed";
        return false;
    }
    if (!copy_outputs_to_host(output_copies, stream))
    {
        error = "cudaMemcpyAsync output failed";
        return false;
    }
    if (cudaStreamSynchronize(stream.get()) != cudaSuccess)
    {
        error = "cudaStreamSynchronize failed";
        return false;
    }
    return true;
}

bool bind_hybrid_inputs(
    const HybridStepEngine& engine,
    CudaStream& stream,
    std::vector<std::unique_ptr<CudaBuffer>>& device_buffers,
    int32_t token_id,
    int32_t position_id,
    const std::vector<float>& attention_mask,
    const std::vector<std::vector<float>>& conv_state_by_layer,
    const std::vector<std::vector<float>>& ssm_state_by_layer,
    const std::vector<std::vector<float>>& cache_k_by_layer,
    const std::vector<std::vector<float>>& cache_v_by_layer,
    const HybridTensorSizes& sizes,
    std::string& error)
{
    if (!bind_input_tensor(engine, stream, device_buffers, engine.token_input_name, &token_id, sizeof(token_id)))
    {
        error = "bind token failed";
        return false;
    }
    if (!bind_input_tensor(
            engine,
            stream,
            device_buffers,
            engine.position_input_name,
            &position_id,
            sizeof(position_id)))
    {
        error = "bind position failed";
        return false;
    }
    if (!bind_input_tensor(
            engine,
            stream,
            device_buffers,
            engine.mask_input_name,
            attention_mask.data(),
            attention_mask.size() * sizeof(float)))
    {
        error = "bind mask failed";
        return false;
    }

    const auto mamba_bindings = std::array<LayerInputBinding, 2>{
        LayerInputBinding{
            &engine.conv_state_input_names,
            &conv_state_by_layer,
            sizes.conv_elems * sizeof(float),
            "bind conv_state failed"},
        LayerInputBinding{
            &engine.ssm_state_input_names,
            &ssm_state_by_layer,
            sizes.ssm_elems * sizeof(float),
            "bind ssm_state failed"}};
    if (!bind_layer_inputs(engine, stream, device_buffers, engine.num_mamba_layers, mamba_bindings, error))
    {
        return false;
    }

    const auto attention_bindings = std::array<LayerInputBinding, 2>{
        LayerInputBinding{
            &engine.cache_k_input_names,
            &cache_k_by_layer,
            sizes.cache_elems * sizeof(float),
            "bind cache_k failed"},
        LayerInputBinding{
            &engine.cache_v_input_names,
            &cache_v_by_layer,
            sizes.cache_elems * sizeof(float),
            "bind cache_v failed"}};
    return bind_layer_inputs(engine, stream, device_buffers, engine.num_attention_layers, attention_bindings, error);
}

bool bind_hybrid_outputs(
    const HybridStepEngine& engine,
    std::vector<std::unique_ptr<CudaBuffer>>& device_buffers,
    std::vector<PendingCopy>& output_copies,
    std::vector<float>& logits,
    std::vector<std::vector<float>>& present_conv,
    std::vector<std::vector<float>>& present_ssm,
    std::vector<std::vector<float>>& present_k,
    std::vector<std::vector<float>>& present_v,
    const HybridTensorSizes& sizes,
    std::string& error)
{
    if (!bind_output_tensor(
            engine,
            device_buffers,
            output_copies,
            engine.logits_output_name,
            logits.data(),
            logits.size() * sizeof(float)))
    {
        error = "bind logits failed";
        return false;
    }

    const auto mamba_bindings = std::array<LayerOutputBinding, 2>{
        LayerOutputBinding{
            &engine.present_conv_output_names,
            &present_conv,
            sizes.conv_elems * sizeof(float),
            "bind present_conv failed"},
        LayerOutputBinding{
            &engine.present_ssm_output_names,
            &present_ssm,
            sizes.ssm_elems * sizeof(float),
            "bind present_ssm failed"}};
    if (!bind_layer_outputs(engine, device_buffers, output_copies, engine.num_mamba_layers, mamba_bindings, error))
    {
        return false;
    }

    const auto attention_bindings = std::array<LayerOutputBinding, 2>{
        LayerOutputBinding{
            &engine.present_k_output_names,
            &present_k,
            sizes.attn_out_elems * sizeof(float),
            "bind present_k failed"},
        LayerOutputBinding{
            &engine.present_v_output_names,
            &present_v,
            sizes.attn_out_elems * sizeof(float),
            "bind present_v failed"}};
    return bind_layer_outputs(
        engine,
        device_buffers,
        output_copies,
        engine.num_attention_layers,
        attention_bindings,
        error);
}

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

    const HybridTensorSizes sizes = compute_hybrid_tensor_sizes(engine);
    initialize_hybrid_outputs(engine, sizes, logits, present_conv, present_ssm, present_k, present_v);

    CudaStream stream;
    if (!stream.ok())
    {
        return fail("cudaStreamCreate failed");
    }

    std::vector<std::unique_ptr<CudaBuffer>> device_buffers;
    std::vector<PendingCopy> output_copies;

    if (!bind_hybrid_inputs(
            engine,
            stream,
            device_buffers,
            token_id,
            position_id,
            attention_mask,
            conv_state_by_layer,
            ssm_state_by_layer,
            cache_k_by_layer,
            cache_v_by_layer,
            sizes,
            error))
    {
        return false;
    }

    if (!bind_hybrid_outputs(
            engine,
            device_buffers,
            output_copies,
            logits,
            present_conv,
            present_ssm,
            present_k,
            present_v,
            sizes,
            error))
    {
        return false;
    }

    if (!execute_hybrid_step(engine, stream, output_copies, error))
    {
        return false;
    }

    return true;
}

void run_hybrid_step_or_throw(
    const HybridStepEngine& engine,
    int32_t token_id,
    HybridStepState& state,
    std::vector<float>& logits,
    std::vector<std::vector<float>>& present_conv,
    std::vector<std::vector<float>>& present_ssm,
    std::vector<std::vector<float>>& present_k,
    std::vector<std::vector<float>>& present_v,
    const char* error_prefix)
{
    const std::vector<float> mask = state.build_attention_mask();
    std::string error;
    if (!run_hybrid_step(
            engine,
            token_id,
            state.position(),
            mask,
            state.conv_state(),
            state.ssm_state(),
            state.cache_k(),
            state.cache_v(),
            logits,
            present_conv,
            present_ssm,
            present_k,
            present_v,
            error))
    {
        throw std::runtime_error(std::string(error_prefix) + error);
    }
    state.update_after_step(present_conv, present_ssm, present_k, present_v);
}

void prefill_hybrid_or_throw(
    const HybridStepEngine& engine,
    const std::vector<int32_t>& input_ids,
    HybridStepState& state,
    std::vector<float>& logits,
    std::vector<std::vector<float>>& present_conv,
    std::vector<std::vector<float>>& present_ssm,
    std::vector<std::vector<float>>& present_k,
    std::vector<std::vector<float>>& present_v)
{
    if (input_ids.size() <= 1)
    {
        return;
    }

    const std::size_t last_idx = input_ids.size() - 1;
    for (std::size_t i = 0; i < last_idx; ++i)
    {
        run_hybrid_step_or_throw(
            engine,
            input_ids[i],
            state,
            logits,
            present_conv,
            present_ssm,
            present_k,
            present_v,
            "hybrid prefill step failed: ");
    }
}

void decode_hybrid_or_throw(
    const HybridStepEngine& engine,
    const std::vector<int32_t>& input_ids,
    const GenerationConfig& config,
    HybridStepState& state,
    std::vector<float>& logits,
    std::vector<std::vector<float>>& present_conv,
    std::vector<std::vector<float>>& present_ssm,
    std::vector<std::vector<float>>& present_k,
    std::vector<std::vector<float>>& present_v,
    std::vector<int32_t>& output)
{
    int32_t current_token = input_ids.empty() ? engine.id_bos : input_ids.back();
    for (std::size_t step = 0; step < config.max_new_tokens; ++step)
    {
        run_hybrid_step_or_throw(
            engine,
            current_token,
            state,
            logits,
            present_conv,
            present_ssm,
            present_k,
            present_v,
            "hybrid decode step failed: ");
        const int32_t next_token = select_argmax_token(logits);
        output.push_back(next_token);
        current_token = next_token;
        if (next_token == engine.id_eos)
        {
            break;
        }
    }
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
        {
            throw std::runtime_error("Hybrid TRT backend not initialized");
        }

        std::vector<int32_t> output = input_ids;
        if (config.max_new_tokens == 0)
        {
            return output;
        }

        HybridStepState state(*mEngine);
        std::vector<float> logits;
        std::vector<std::vector<float>> present_conv, present_ssm, present_k, present_v;
        prefill_hybrid_or_throw(
            *mEngine,
            input_ids,
            state,
            logits,
            present_conv,
            present_ssm,
            present_k,
            present_v);
        decode_hybrid_or_throw(
            *mEngine,
            input_ids,
            config,
            state,
            logits,
            present_conv,
            present_ssm,
            present_k,
            present_v,
            output);
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
