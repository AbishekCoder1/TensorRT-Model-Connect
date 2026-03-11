#include "runtime/trt/audio/magpie_tts_backend.h"
#include "runtime/trt/audio/magpie_codec_plan.h"
#include "runtime/trt/audio/magpie_decode_policy.h"
#include "runtime/trt/audio/magpie_decoder_plan.h"
#include "runtime/trt/audio/magpie_text_completion_policy.h"

#if TRTF_HAS_TRT

#include "runtime/trt/core/trt_decode_runtime.h"

#ifndef TRTF_HAS_CUDA_KERNELS
#define TRTF_HAS_CUDA_KERNELS 0
#endif

#if TRTF_HAS_CUDA_KERNELS
#include "runtime/trt/audio/magpie_kernels.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace {
using SteadyClock = std::chrono::steady_clock;
using TimePoint = SteadyClock::time_point;
inline double elapsed_ms(TimePoint start, TimePoint end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}
} // anonymous namespace

namespace trtf {

namespace {

void log_frame_preview(const std::vector<int32_t>& all_codes, int32_t num_cb)
{
    const int32_t gen_frames = static_cast<int32_t>(all_codes.size()) / std::max(num_cb, 1);
    for (int32_t f = 0; f < std::min(gen_frames, 10); ++f)
    {
        std::cerr << "[magpie-tts]   frame " << f << ": [";
        for (int32_t cb = 0; cb < num_cb; ++cb)
        {
            if (cb > 0)
            {
                std::cerr << ", ";
            }
            std::cerr << all_codes[static_cast<std::size_t>(f) * num_cb + cb];
        }
        std::cerr << "]" << std::endl;
    }

    if (gen_frames <= 15)
    {
        return;
    }

    std::cerr << "[magpie-tts]   ..." << std::endl;
    for (int32_t f = gen_frames - 5; f < gen_frames; ++f)
    {
        std::cerr << "[magpie-tts]   frame " << f << ": [";
        for (int32_t cb = 0; cb < num_cb; ++cb)
        {
            if (cb > 0)
            {
                std::cerr << ", ";
            }
            std::cerr << all_codes[static_cast<std::size_t>(f) * num_cb + cb];
        }
        std::cerr << "]" << std::endl;
    }
}

bool bind_magpie_codec_tensors(
    nvinfer1::IExecutionContext& codec_ctx,
    CudaBuffer& d_input,
    CudaBuffer& d_len,
    CudaBuffer& d_output)
{
    if (!codec_ctx.setTensorAddress("codec_tokens", d_input.data()))
    {
        std::cerr << "[magpie-tts] Failed to bind codec_tokens" << std::endl;
        return false;
    }
    if (!codec_ctx.setTensorAddress("input_len", d_len.data()))
    {
        std::cerr << "[magpie-tts] Failed to bind input_len" << std::endl;
        return false;
    }
    if (!codec_ctx.setTensorAddress("waveform", d_output.data()))
    {
        std::cerr << "[magpie-tts] Failed to bind waveform" << std::endl;
        return false;
    }
    return true;
}

std::vector<float> run_codec_engine(
    nvinfer1::ICudaEngine& codec_engine,
    nvinfer1::IExecutionContext& codec_ctx,
    const std::vector<int32_t>& codes,
    int32_t num_frames,
    int32_t num_cb)
{
    const auto codec_tokens_shape = codec_engine.getTensorShape("codec_tokens");
    const int32_t max_codec_frames = (codec_tokens_shape.nbDims >= 2)
        ? codec_tokens_shape.d[1] : num_frames;
    const auto plan = make_magpie_codec_plan(num_frames, num_cb, max_codec_frames);
    std::vector<int32_t> codec_input = build_magpie_codec_input(codes, num_cb, plan);

    CudaBuffer d_input(plan.input_bytes);
    CudaBuffer d_len(plan.len_bytes);
    CudaBuffer d_output(plan.output_bytes);
    CudaStream stream;
    if (!d_input.ok() || !d_len.ok() || !d_output.ok() || !stream.ok())
    {
        std::cerr << "[magpie-tts] Failed to allocate CUDA resources for codec" << std::endl;
        return {};
    }

    cudaMemcpyAsync(
        d_input.data(), codec_input.data(), plan.input_bytes, cudaMemcpyHostToDevice, stream.get());
    cudaMemcpyAsync(
        d_len.data(), &plan.input_len, plan.len_bytes, cudaMemcpyHostToDevice, stream.get());
    if (!bind_magpie_codec_tensors(codec_ctx, d_input, d_len, d_output))
    {
        return {};
    }
    if (!codec_ctx.enqueueV3(stream.get()))
    {
        std::cerr << "[magpie-tts] Codec TRT execution failed" << std::endl;
        return {};
    }

    std::vector<float> waveform(plan.valid_samples);
    cudaMemcpyAsync(
        waveform.data(),
        d_output.data(),
        plan.valid_samples * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream.get());
    cudaStreamSynchronize(stream.get());
    return waveform;
}

void maybe_enable_magpie_greedy(MagpieTTSConfig& cfg)
{
    const char* env = std::getenv("TRTF_MAGPIE_GREEDY");
    if (env != nullptr && std::string(env) == "1")
    {
        cfg.greedy = true;
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

MagpieTTSBackend::MagpieTTSBackend(
    std::unique_ptr<DecoderStepEngine> decoder_engine,
    TrtUniquePtr<nvinfer1::ICudaEngine> encoder_engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> encoder_context,
    std::vector<float> audio_embed,
    std::vector<float> text_embed,
    std::vector<float> context_embed,
    std::vector<int32_t> context_lengths,
    MagpieTTSConfig config)
    : mDecoderEngine(std::move(decoder_engine))
    , mEncoderEngine(std::move(encoder_engine))
    , mEncoderContext(std::move(encoder_context))
    , mEncoderOutput(static_cast<std::size_t>(config.max_source_positions) *
                     static_cast<std::size_t>(config.hidden_size) * sizeof(float))
    , mEncoderOutputUncond(config.cfg_scale > 1.0F
        ? static_cast<std::size_t>(config.max_source_positions) *
          static_cast<std::size_t>(config.hidden_size) * sizeof(float)
        : 0)
    , mAudioEmbed(std::move(audio_embed))
    , mTextEmbed(std::move(text_embed))
    , mContextEmbed(std::move(context_embed))
    , mContextLengths(std::move(context_lengths))
    , mDeviceCodes(static_cast<std::size_t>(config.num_codebooks) * sizeof(int32_t))
    , mDeviceFullArgmax(static_cast<std::size_t>(config.num_codebooks) * sizeof(int32_t))
    , mAudioEmbedDevice(mAudioEmbed.size() * sizeof(float))
    , mContextEmbedDevice(mContextEmbed.size() * sizeof(float))
    , mDevicePrevCodes(static_cast<std::size_t>(config.num_codebooks) * sizeof(int32_t))
    , mDeviceAllCodes(static_cast<std::size_t>(512) * config.num_codebooks * sizeof(int32_t))
    , mDeviceLogitsCond(0)
    , mDeviceLogitsUncond(0)
    , mDeviceCrossAttnWeights(0)
    , mDeviceCrossAttnWeightsScratch(0)
    , mConfig(config)
{
    if (mDecoderEngine)
    {
        mCache = std::make_unique<DeviceKvCache>(*mDecoderEngine);
        mResources = std::make_unique<DeviceResources>(*mDecoderEngine);

        // CFG: allocate second KV cache + resources when cfg_scale > 1
        if (config.cfg_scale > 1.0F)
        {
            mCacheUncond = std::make_unique<DeviceKvCache>(*mDecoderEngine);
            mResourcesUncond = std::make_unique<DeviceResources>(*mDecoderEngine);
        }
    }

    allocate_cross_kv_buffers();
    allocate_cfg_buffers_ctor();
    detect_cross_attn_output();
    upload_embed_tables_to_device();
}

void MagpieTTSBackend::allocate_cross_kv_buffers()
{
    // Allocate per-layer cross-attention K/V buffers (separate copies needed --
    // TRT may use input tensor memory as workspace during layer execution)
    const std::size_t enc_buf_size = mEncoderOutput.size();
    const int32_t dec_layers = mConfig.decoder_layers > 0
        ? mConfig.decoder_layers : (mDecoderEngine ? mDecoderEngine->num_layers : 1);
    mCrossK.reserve(static_cast<std::size_t>(dec_layers));
    mCrossV.reserve(static_cast<std::size_t>(dec_layers));
    for (int32_t i = 0; i < dec_layers; ++i)
    {
        mCrossK.emplace_back(enc_buf_size);
        mCrossV.emplace_back(enc_buf_size);
    }
}

void MagpieTTSBackend::allocate_cfg_buffers_ctor()
{
    if (mConfig.cfg_scale <= 1.0F)
    {
        return;
    }

    const std::size_t enc_buf_size = mEncoderOutput.size();
    const int32_t dec_layers = mConfig.decoder_layers > 0
        ? mConfig.decoder_layers : (mDecoderEngine ? mDecoderEngine->num_layers : 1);

    // CFG: allocate unconditional cross-KV buffers (filled at runtime from null-text encoder)
    mCrossKUncond.reserve(static_cast<std::size_t>(dec_layers));
    mCrossVUncond.reserve(static_cast<std::size_t>(dec_layers));
    for (int32_t i = 0; i < dec_layers; ++i)
    {
        mCrossKUncond.emplace_back(enc_buf_size);
        mCrossVUncond.emplace_back(enc_buf_size);
    }

    // Allocate device logit buffers for CFG blending
    const auto logits_bytes = static_cast<std::size_t>(mConfig.num_codebooks) *
        static_cast<std::size_t>(mConfig.codebook_size) * sizeof(float);
    mDeviceLogitsCond = CudaBuffer(logits_bytes);
    mDeviceLogitsUncond = CudaBuffer(logits_bytes);
}

void MagpieTTSBackend::detect_cross_attn_output()
{
    // Detect cross_attn_weights output tensor for text-completion tracking
    if (mDecoderEngine && mDecoderEngine->engine
        && has_io_tensor(*mDecoderEngine->engine, "cross_attn_weights"))
    {
        mHasCrossAttnOutput = true;
        const auto xattn_bytes = static_cast<std::size_t>(mConfig.max_source_positions) * sizeof(float);
        mDeviceCrossAttnWeights = CudaBuffer(xattn_bytes);
        // CFG: scratch buffer so uncond pass doesn't overwrite conditioned weights
        if (mConfig.cfg_scale > 1.0F)
        {
            mDeviceCrossAttnWeightsScratch = CudaBuffer(xattn_bytes);
        }
    }
}

void MagpieTTSBackend::upload_embed_tables_to_device()
{
    // Copy embedding tables to GPU for device-side lookup
    if (!mAudioEmbed.empty() && mAudioEmbedDevice.ok())
    {
        cudaMemcpy(mAudioEmbedDevice.data(), mAudioEmbed.data(),
                   mAudioEmbed.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
    if (!mContextEmbed.empty() && mContextEmbedDevice.ok())
    {
        cudaMemcpy(mContextEmbedDevice.data(), mContextEmbed.data(),
                   mContextEmbed.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
}

MagpieTTSBackend::~MagpieTTSBackend() = default;

bool MagpieTTSBackend::is_available() const
{
    return mDecoderEngine && mEncoderEngine && mCache && mResources
        && !mAudioEmbed.empty() && !mTextEmbed.empty();
}

void MagpieTTSBackend::set_codec_engine(
    TrtUniquePtr<nvinfer1::ICudaEngine> engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> context)
{
    mCodecEngine = std::move(engine);
    mCodecCtx = std::move(context);
}

// ---------------------------------------------------------------------------
// Embedding helpers (Bark pattern)
// ---------------------------------------------------------------------------

void MagpieTTSBackend::lookup_embed(const float* table, int32_t token_id,
                                     float* out) const
{
    const auto offset = static_cast<std::size_t>(token_id) *
                        static_cast<std::size_t>(mConfig.hidden_size);
    std::memcpy(out, table + offset,
                static_cast<std::size_t>(mConfig.hidden_size) * sizeof(float));
}

void MagpieTTSBackend::sum_embeds(const float* a, const float* b, float* out) const
{
    for (int32_t i = 0; i < mConfig.hidden_size; ++i)
    {
        out[i] = a[i] + b[i];
    }
}

// ---------------------------------------------------------------------------
// Top-k sampling with temperature (Bark pattern)
// ---------------------------------------------------------------------------

int32_t MagpieTTSBackend::sample_top_k(const float* logits, int32_t vocab_size,
                                         float temperature, int32_t top_k)
{
    // Greedy mode: return argmax
    if (mConfig.greedy)
    {
        int32_t best = 0;
        for (int32_t i = 1; i < vocab_size; ++i)
        {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }

    // 1. Find top-k indices
    top_k = std::min(top_k, vocab_size);
    std::vector<int32_t> indices(static_cast<std::size_t>(vocab_size));
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
        [logits](int32_t a, int32_t b) { return logits[a] > logits[b]; });

    // 2. Scale by temperature and softmax over top-k
    std::vector<float> probs(static_cast<std::size_t>(top_k));
    float max_logit = logits[indices[0]];
    float sum = 0.0F;
    for (int32_t i = 0; i < top_k; ++i)
    {
        probs[i] = std::exp((logits[indices[i]] - max_logit) / temperature);
        sum += probs[i];
    }
    for (int32_t i = 0; i < top_k; ++i)
    {
        probs[i] /= sum;
    }

    // 3. Sample from distribution
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    float r = dist(mRng);
    float cumulative = 0.0F;
    for (int32_t i = 0; i < top_k; ++i)
    {
        cumulative += probs[i];
        if (r < cumulative)
        {
            return indices[i];
        }
    }
    return indices[top_k - 1];
}

// ---------------------------------------------------------------------------
// run_encoder() -- text IDs + speaker/language -> encoder_output
// ---------------------------------------------------------------------------

void MagpieTTSBackend::run_encoder(
    const std::vector<int32_t>& text_ids,
    int32_t /*speaker_id*/, int32_t /*language_id*/)
{
    if (!mEncoderEngine || !mEncoderContext) return;

    const int32_t max_pos = mConfig.max_source_positions;

    // Pad or truncate input_ids to max_source_positions (encoder compiled for fixed shape)
    std::vector<int32_t> padded(static_cast<std::size_t>(max_pos), 0);
    const auto copy_len = std::min(static_cast<int32_t>(text_ids.size()), max_pos);
    if (copy_len > 0)
    {
        std::memcpy(padded.data(), text_ids.data(),
                     static_cast<std::size_t>(copy_len) * sizeof(int32_t));
    }

    const auto input_bytes = static_cast<std::size_t>(max_pos) * sizeof(int32_t);
    CudaBuffer d_input_ids(input_bytes);
    cudaMemcpy(d_input_ids.data(), padded.data(), input_bytes, cudaMemcpyHostToDevice);

    // Set up encoder bindings
    const int32_t num_tensors = mEncoderEngine->getNbIOTensors();
    for (int32_t i = 0; i < num_tensors; ++i)
    {
        const char* tensor_name = mEncoderEngine->getIOTensorName(i);
        const std::string name(tensor_name);
        if (name == "input_ids")
        {
            mEncoderContext->setTensorAddress(tensor_name, d_input_ids.data());
        }
        else if (name == "encoder_output")
        {
            mEncoderContext->setTensorAddress(tensor_name, mEncoderOutput.data());
        }
    }

    // Execute encoder
    CudaStream enc_stream;
    if (!mEncoderContext->enqueueV3(enc_stream.get()))
    {
        throw std::runtime_error("MagpieTTS encoder execution failed");
    }
    cudaStreamSynchronize(enc_stream.get());

    // Zero out encoder output for padded positions (beyond actual tokens).
    // The encoder produces non-zero junk for padding which leaks into
    // cross-attention and causes decoder divergence.
    if (copy_len < max_pos)
    {
        const auto hidden = mConfig.hidden_size;
        const auto zero_offset = static_cast<std::size_t>(copy_len) *
                                  static_cast<std::size_t>(hidden) * sizeof(float);
        const auto zero_bytes = static_cast<std::size_t>(max_pos - copy_len) *
                                 static_cast<std::size_t>(hidden) * sizeof(float);
        cudaMemset(static_cast<char*>(mEncoderOutput.data()) + zero_offset, 0, zero_bytes);
    }

    std::cerr << "[magpie-tts] Encoder: processed " << copy_len
              << " tokens (padded to " << max_pos << ")" << std::endl;
}

// ---------------------------------------------------------------------------
// compute_cross_kv() -- copy encoder output to per-layer cross K/V buffers.
// Each layer needs its own buffer because TRT may use input tensor memory
// as workspace during layer execution.
// ---------------------------------------------------------------------------

void MagpieTTSBackend::compute_cross_kv()
{
    const int32_t hidden = mConfig.hidden_size;
    const int32_t enc_seq = mConfig.max_source_positions;
    const std::size_t buf_size = static_cast<std::size_t>(enc_seq) *
                                 static_cast<std::size_t>(hidden) * sizeof(float);

    for (std::size_t i = 0; i < mCrossK.size(); ++i)
    {
        cudaMemcpy(mCrossK[i].data(), mEncoderOutput.data(),
                    buf_size, cudaMemcpyDeviceToDevice);
        cudaMemcpy(mCrossV[i].data(), mEncoderOutput.data(),
                    buf_size, cudaMemcpyDeviceToDevice);
    }
}

// ---------------------------------------------------------------------------
// bind_cross_kv() -- set cross_k/cross_v tensor addresses on decoder context
// ---------------------------------------------------------------------------

void MagpieTTSBackend::bind_cross_kv()
{
    const int32_t dec_layers = static_cast<int32_t>(mCrossK.size());
    for (int32_t i = 0; i < dec_layers; ++i)
    {
        const std::string cross_k_name = layer_tensor_name("cross_k", i);
        const std::string cross_v_name = layer_tensor_name("cross_v", i);
        mDecoderEngine->context->setTensorAddress(cross_k_name.c_str(), mCrossK[i].data());
        mDecoderEngine->context->setTensorAddress(cross_v_name.c_str(), mCrossV[i].data());
    }

    // Bind cross-attention weights output if available
    if (mHasCrossAttnOutput && mDeviceCrossAttnWeights.ok())
    {
        mDecoderEngine->context->setTensorAddress(
            "cross_attn_weights", mDeviceCrossAttnWeights.data());
    }
}

// ---------------------------------------------------------------------------
// compute_cross_kv_uncond() -- copy null-text encoder output to uncond cross-KV
// ---------------------------------------------------------------------------

void MagpieTTSBackend::compute_cross_kv_uncond()
{
    const int32_t hidden = mConfig.hidden_size;
    const int32_t enc_seq = mConfig.max_source_positions;
    const std::size_t buf_size = static_cast<std::size_t>(enc_seq) *
                                 static_cast<std::size_t>(hidden) * sizeof(float);

    for (std::size_t i = 0; i < mCrossKUncond.size(); ++i)
    {
        cudaMemcpy(mCrossKUncond[i].data(), mEncoderOutputUncond.data(),
                    buf_size, cudaMemcpyDeviceToDevice);
        cudaMemcpy(mCrossVUncond[i].data(), mEncoderOutputUncond.data(),
                    buf_size, cudaMemcpyDeviceToDevice);
    }
}

// ---------------------------------------------------------------------------
// bind_cross_kv_uncond() -- bind zeroed cross_k/cross_v on decoder context
// ---------------------------------------------------------------------------

void MagpieTTSBackend::bind_cross_kv_uncond()
{
    const int32_t dec_layers = static_cast<int32_t>(mCrossKUncond.size());
    for (int32_t i = 0; i < dec_layers; ++i)
    {
        const std::string cross_k_name = layer_tensor_name("cross_k", i);
        const std::string cross_v_name = layer_tensor_name("cross_v", i);
        mDecoderEngine->context->setTensorAddress(cross_k_name.c_str(), mCrossKUncond[i].data());
        mDecoderEngine->context->setTensorAddress(cross_v_name.c_str(), mCrossVUncond[i].data());
    }

    // Redirect cross_attn_weights to scratch buffer so uncond pass
    // doesn't overwrite the conditioned weights we need for tracking
    if (mHasCrossAttnOutput && mDeviceCrossAttnWeightsScratch.ok())
    {
        mDecoderEngine->context->setTensorAddress(
            "cross_attn_weights", mDeviceCrossAttnWeightsScratch.data());
    }
}

// ---------------------------------------------------------------------------
// Decoder loop state initialization
// ---------------------------------------------------------------------------

namespace {
bool check_gpu_kernels_available(
    [[maybe_unused]] const CudaBuffer& audio_embed,
    [[maybe_unused]] const CudaBuffer& codes,
    [[maybe_unused]] const CudaBuffer& full_argmax,
    [[maybe_unused]] const CudaBuffer& prev_codes)
{
#if TRTF_HAS_CUDA_KERNELS
    return audio_embed.ok() && codes.ok() && full_argmax.ok() && prev_codes.ok();
#else
    return false;
#endif
}

void upload_prev_codes_to_device(
    [[maybe_unused]] CudaBuffer& d_prev,
    [[maybe_unused]] const int32_t* host_codes,
    [[maybe_unused]] int32_t num_cb,
    [[maybe_unused]] bool use_gpu,
    [[maybe_unused]] bool use_gpu_greedy)
{
#if TRTF_HAS_CUDA_KERNELS
    if (use_gpu && !use_gpu_greedy)
    {
        cudaMemcpy(d_prev.data(), host_codes,
                   static_cast<std::size_t>(num_cb) * sizeof(int32_t),
                   cudaMemcpyHostToDevice);
    }
#endif
}
} // anonymous namespace

MagpieTTSBackend::DecoderLoopState MagpieTTSBackend::init_decoder_state() const
{
    DecoderLoopState s;
    const auto plan = make_magpie_decoder_plan(
        mConfig,
        static_cast<bool>(mCacheUncond),
        static_cast<bool>(mResourcesUncond),
        !mCrossKUncond.empty(),
        check_gpu_kernels_available(
            mAudioEmbedDevice, mDeviceCodes, mDeviceFullArgmax, mDevicePrevCodes),
        mHasCrossAttnOutput,
        mDeviceCrossAttnWeights.ok(),
        mTextLength);
    s.hidden = plan.hidden;
    s.num_cb = plan.num_cb;
    s.cb_size = plan.cb_size;
    s.total_logits = plan.total_logits;
    s.use_cfg = plan.use_cfg;
    s.use_gpu_kernels = plan.use_gpu_kernels;
    s.use_gpu_greedy = plan.use_gpu_greedy;
    s.finished_limit = plan.finished_limit;
    s.max_source_positions = plan.max_source_positions;
    s.use_cross_attn_tracking = plan.use_cross_attn_tracking;
    s.estimated_frames = plan.estimated_frames;
    s.text_consumed_threshold = plan.text_consumed_threshold;

    // Working buffers
    s.embed_buf.resize(static_cast<std::size_t>(s.hidden));
    s.cb_embed.resize(static_cast<std::size_t>(s.hidden));

    return s;
}

// ---------------------------------------------------------------------------
// Phase 1: Context prefill
// ---------------------------------------------------------------------------

bool MagpieTTSBackend::prefill_context_gpu(
    DecoderLoopState& state, DeviceKvCache& cache, DeviceResources& resources,
    int32_t ctx_frames, const char* label)
{
#if TRTF_HAS_CUDA_KERNELS
    const int32_t hidden = state.hidden;
    const std::size_t frame_bytes = static_cast<std::size_t>(hidden) * sizeof(float);
    for (int32_t pos = 0; pos < ctx_frames; ++pos)
    {
        const auto frame_offset = static_cast<std::size_t>(pos) * hidden * sizeof(float);
        const auto* src = static_cast<const char*>(mContextEmbedDevice.data()) + frame_offset;

        cudaMemcpyAsync(resources.d_input_embed.data(), src, frame_bytes,
                        cudaMemcpyDeviceToDevice, resources.stream.get());

        if (!run_decoder_step_device(*mDecoderEngine, cache, resources,
                0, state.logits, state.error,
                nullptr, hidden, 1.0F,
                {}, 0.0F,
                /*input_embed_device_ready=*/true,
                /*skip_logits_d2h=*/true,
                /*skip_sync=*/true,
                /*skip_bind=*/(pos > 0)))
        {
            std::cerr << "[magpie-tts] " << label << " step " << pos
                      << " failed: " << state.error << std::endl;
            return false;
        }
    }
    cudaStreamSynchronize(resources.stream.get());
    return true;
#else
    (void)state; (void)cache; (void)resources; (void)ctx_frames; (void)label;
    return false;
#endif
}

bool MagpieTTSBackend::prefill_context_cpu(
    DecoderLoopState& state, DeviceKvCache& cache, DeviceResources& resources,
    int32_t ctx_frames, const char* label)
{
    const int32_t hidden = state.hidden;
    const float* ctx_ptr = mContextEmbed.data();
    for (int32_t pos = 0; pos < ctx_frames; ++pos)
    {
        const float* frame_embed = ctx_ptr +
            static_cast<std::size_t>(pos) * hidden;

        if (!run_decoder_step_device(*mDecoderEngine, cache, resources,
                0, state.logits, state.error,
                frame_embed, hidden, 1.0F))
        {
            std::cerr << "[magpie-tts] " << label << " step " << pos
                      << " failed: " << state.error << std::endl;
            return false;
        }
    }
    return true;
}

bool MagpieTTSBackend::prefill_context_cfg(
    DecoderLoopState& state, int32_t ctx_frames)
{
    std::cerr << "[magpie-tts] CFG: prefilling unconditional cache ("
              << ctx_frames << " frames) ..." << std::endl;
    bind_cross_kv_uncond();
    auto& uncond_cache = *mCacheUncond;
    auto& uncond_res = *mResourcesUncond;

    bool ok = false;
#if TRTF_HAS_CUDA_KERNELS
    if (state.use_gpu_kernels && mContextEmbedDevice.ok())
    {
        ok = prefill_context_gpu(state, uncond_cache, uncond_res,
                                 ctx_frames, "CFG uncond context");
    }
    else
#endif
    {
        ok = prefill_context_cpu(state, uncond_cache, uncond_res,
                                 ctx_frames, "CFG uncond context");
    }

    // Rebind conditioned cross-KV for generation phase
    if (ok) bind_cross_kv();
    return ok;
}

int32_t MagpieTTSBackend::prefill_context(DecoderLoopState& state)
{
    if (mContextEmbed.empty() || mContextLengths.empty())
    {
        return 0;
    }

    auto& cache = *mCache;
    auto& resources = *mResources;
    const int32_t ctx_frames = mContextLengths[0];

    std::cerr << "[magpie-tts] Prefilling " << ctx_frames
              << " context frames ..." << std::endl;
    const auto t_prefill_start = SteadyClock::now();

    bool ok = false;
#if TRTF_HAS_CUDA_KERNELS
    if (state.use_gpu_kernels && mContextEmbedDevice.ok())
    {
        ok = prefill_context_gpu(state, cache, resources, ctx_frames, "Context");
    }
    else
#endif
    {
        ok = prefill_context_cpu(state, cache, resources, ctx_frames, "Context");
    }

    if (!ok) return -1;

    const auto t_prefill_end = SteadyClock::now();
    state.prof_prefill_ms = elapsed_ms(t_prefill_start, t_prefill_end);

    // CFG: prefill unconditional cache with same speaker context but zeroed cross-KV
    if (state.use_cfg)
    {
        if (!prefill_context_cfg(state, ctx_frames)) return -1;
    }

    return ctx_frames;
}

// ---------------------------------------------------------------------------
// CFG unconditional pass (GPU greedy path -- device-side blend)
// ---------------------------------------------------------------------------

bool MagpieTTSBackend::run_cfg_uncond_pass_gpu(DecoderLoopState& state,
                                                 int32_t frame)
{
#if TRTF_HAS_CUDA_KERNELS
    auto& resources = *mResources;
    auto& uncond_res = *mResourcesUncond;
    auto& uncond_cache = *mCacheUncond;

    // Copy conditioned logits to staging buffer
    cudaMemcpyAsync(mDeviceLogitsCond.data(), resources.d_logits.data(),
        static_cast<std::size_t>(state.total_logits) * sizeof(float),
        cudaMemcpyDeviceToDevice, resources.stream.get());

    // Run unconditional pass with same embedding
    cudaMemcpyAsync(uncond_res.d_input_embed.data(),
        resources.d_input_embed.data(),
        static_cast<std::size_t>(state.hidden) * sizeof(float),
        cudaMemcpyDeviceToDevice, uncond_res.stream.get());

    bind_cross_kv_uncond();
    if (!run_decoder_step_device(*mDecoderEngine, uncond_cache, uncond_res,
            0, state.logits, state.error,
            nullptr, state.hidden, 1.0F,
            {}, 0.0F,
            /*input_embed_device_ready=*/true,
            /*skip_logits_d2h=*/true,
            /*skip_sync=*/true,
            /*skip_bind=*/false))
    {
        cudaStreamSynchronize(resources.stream.get());
        std::cerr << "[magpie-tts] CFG uncond step " << frame
                  << " failed: " << state.error << std::endl;
        return false;
    }
    // Sync uncond stream before blending
    cudaStreamSynchronize(uncond_res.stream.get());

    // CFG interpolation: out = uncond + scale * (cond - uncond)
    magpie_cfg_interpolate_device(
        static_cast<const float*>(mDeviceLogitsCond.data()),
        static_cast<const float*>(uncond_res.d_logits.data()),
        static_cast<float*>(resources.d_logits.data()),
        mConfig.cfg_scale, state.total_logits,
        resources.stream.get());
    return true;
#else
    (void)state; (void)frame;
    return false;
#endif
}

// ---------------------------------------------------------------------------
// CFG unconditional pass (CPU path -- host-side blend into state.logits)
// ---------------------------------------------------------------------------

bool MagpieTTSBackend::run_cfg_uncond_pass_cpu(DecoderLoopState& state,
                                                 int32_t frame)
{
    std::vector<float> cond_logits = state.logits;
    std::vector<float> uncond_logits;
    std::string uncond_error;

    bind_cross_kv_uncond();
    auto& uncond_cache = *mCacheUncond;
    auto& uncond_res = *mResourcesUncond;

#if TRTF_HAS_CUDA_KERNELS
    if (state.use_gpu_kernels)
    {
        auto& resources = *mResources;
        cudaMemcpyAsync(uncond_res.d_input_embed.data(),
            resources.d_input_embed.data(),
            static_cast<std::size_t>(state.hidden) * sizeof(float),
            cudaMemcpyDeviceToDevice, uncond_res.stream.get());
        if (!run_decoder_step_device(*mDecoderEngine, uncond_cache, uncond_res,
                0, uncond_logits, uncond_error,
                nullptr, state.hidden, 1.0F,
                {}, 0.0F,
                /*input_embed_device_ready=*/true))
        {
            std::cerr << "[magpie-tts] CFG uncond step " << frame
                      << " failed: " << uncond_error << std::endl;
            return false;
        }
    }
    else
#endif
    {
        if (!run_decoder_step_device(*mDecoderEngine, uncond_cache, uncond_res,
                0, uncond_logits, uncond_error,
                state.embed_buf.data(), state.hidden, 1.0F))
        {
            std::cerr << "[magpie-tts] CFG uncond step " << frame
                      << " failed: " << uncond_error << std::endl;
            return false;
        }
    }

    // CPU-side CFG blend: logits = uncond + scale * (cond - uncond)
    const auto n = std::min(cond_logits.size(), uncond_logits.size());
    state.logits.resize(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        state.logits[i] = uncond_logits[i] + mConfig.cfg_scale *
            (cond_logits[i] - uncond_logits[i]);
    }

    // Rebind conditioned cross-KV for next frame
    bind_cross_kv();
    return true;
}

// ---------------------------------------------------------------------------
// Text-completion tracking via cross-attention weights
// ---------------------------------------------------------------------------

void MagpieTTSBackend::update_text_completion(DecoderLoopState& state,
                                               int32_t frame)
{
    if (state.use_cross_attn_tracking && !state.text_consumed)
    {
        std::vector<float> xattn(static_cast<std::size_t>(state.max_source_positions));
        cudaMemcpy(xattn.data(), mDeviceCrossAttnWeights.data(),
            static_cast<std::size_t>(state.max_source_positions) * sizeof(float),
            cudaMemcpyDeviceToHost);

        if (update_magpie_text_consumed_from_cross_attn(
                xattn.data(),
                state.max_source_positions,
                state.text_consumed_threshold,
                state.max_peak_pos,
                state.text_consumed))
        {
            std::cerr << "[magpie-tts] Text consumed at frame " << frame
                      << " (max_peak_pos=" << state.max_peak_pos
                      << ", threshold=" << state.text_consumed_threshold
                      << ", text_len=" << mTextLength << ")" << std::endl;
        }
    }

    // Heuristic fallback
    if (!state.use_cross_attn_tracking)
    {
        update_magpie_text_consumed_from_heuristic(
            state.estimated_frames,
            frame,
            state.text_consumed);
    }
}

// ---------------------------------------------------------------------------
// Check finished_limit_with_eot -- returns true if generation should stop
// ---------------------------------------------------------------------------

bool MagpieTTSBackend::check_finished_limit(DecoderLoopState& state,
                                              int32_t frame)
{
    if (advance_magpie_finished_limit(
            state.text_consumed,
            state.finished_limit,
            state.frames_past_text_consumed))
    {
        std::cerr << "[magpie-tts] finished_limit_with_eot: stopping at frame "
                  << frame << " (" << state.frames_past_text_consumed
                  << " frames past text consumed)" << std::endl;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Decoder profiling log
// ---------------------------------------------------------------------------

void MagpieTTSBackend::log_decoder_profiling(const DecoderLoopState& state,
                                              int32_t ctx_frames,
                                              int32_t gen_frames) const
{
    std::cerr << "\n[magpie-tts] --- Decoder Profiling Breakdown ---\n"
              << "[magpie-tts]   Context prefill:   " << state.prof_prefill_ms << " ms ("
              << ctx_frames << " frames, "
              << (ctx_frames > 0 ? state.prof_prefill_ms / ctx_frames : 0.0) << " ms/frame)\n"
              << "[magpie-tts]   Embed computation: " << state.prof_embed_ms << " ms ("
              << (gen_frames > 0 ? state.prof_embed_ms / gen_frames : 0.0) << " ms/frame)\n"
              << "[magpie-tts]   TRT decoder steps: " << state.prof_trt_step_ms << " ms ("
              << (gen_frames > 0 ? state.prof_trt_step_ms / gen_frames : 0.0) << " ms/frame)\n"
              << "[magpie-tts]   Sampling:          " << state.prof_sample_ms << " ms ("
              << (gen_frames > 0 ? state.prof_sample_ms / gen_frames : 0.0) << " ms/frame)\n"
              << "[magpie-tts]   Stop method:       "
              << (state.use_cross_attn_tracking ? "cross-attn tracking" : "heuristic (text_len*3)")
              << "\n"
              << "[magpie-tts] ---------------------------------\n";
}

// ---------------------------------------------------------------------------
// GPU greedy decode loop
// ---------------------------------------------------------------------------

bool MagpieTTSBackend::gpu_greedy_frame_step(
    DecoderLoopState& state, int32_t frame, CudaBuffer& d_eos_flag)
{
#if TRTF_HAS_CUDA_KERNELS
    constexpr int32_t EOS_TOKEN = 2017;
    constexpr int32_t AUDIO_RANGE = 2016;

    auto& engine = *mDecoderEngine;
    auto& cache = *mCache;
    auto& resources = *mResources;
    const int32_t num_cb = state.num_cb;
    const int32_t cb_size = state.cb_size;
    const int32_t hidden = state.hidden;

    // Embed
    const auto t_embed_start = SteadyClock::now();
    magpie_gather_average_embed_device(
        static_cast<const float*>(mAudioEmbedDevice.data()),
        static_cast<const int32_t*>(mDevicePrevCodes.data()),
        num_cb, cb_size, hidden,
        static_cast<float*>(resources.d_input_embed.data()),
        resources.stream.get());
    const auto t_embed_end = SteadyClock::now();
    state.prof_embed_ms += elapsed_ms(t_embed_start, t_embed_end);

    // Conditioned pass (rebind cross-KV when CFG is active or first frame)
    const auto t_step_start = SteadyClock::now();
    if (state.use_cfg || frame == 0) bind_cross_kv();
    if (!run_decoder_step_device(engine, cache, resources,
            0, state.logits, state.error,
            nullptr, hidden, 1.0F,
            {}, 0.0F,
            /*input_embed_device_ready=*/true,
            /*skip_logits_d2h=*/true,
            /*skip_sync=*/true,
            /*skip_bind=*/(!state.use_cfg && frame > 0)))
    {
        cudaStreamSynchronize(resources.stream.get());
        std::cerr << "[magpie-tts] Decode step " << frame << " failed: "
                  << state.error << std::endl;
        return false;
    }

    // CFG: unconditional pass + device-side blend
    if (state.use_cfg && !run_cfg_uncond_pass_gpu(state, frame))
    {
        return false;
    }
    const auto t_step_end = SteadyClock::now();
    state.prof_trt_step_ms += elapsed_ms(t_step_start, t_step_end);

    // Sample + scatter
    const auto t_sample_start = SteadyClock::now();
    magpie_greedy_sample_device(
        static_cast<const float*>(resources.d_logits.data()),
        num_cb, cb_size, AUDIO_RANGE,
        static_cast<int32_t*>(mDeviceCodes.data()),
        static_cast<int32_t*>(mDeviceFullArgmax.data()),
        resources.stream.get());

    magpie_scatter_codes_device(
        static_cast<const int32_t*>(mDeviceCodes.data()),
        static_cast<int32_t*>(mDeviceAllCodes.data()),
        static_cast<int32_t*>(mDevicePrevCodes.data()),
        static_cast<const int32_t*>(mDeviceFullArgmax.data()),
        static_cast<int32_t*>(d_eos_flag.data()),
        frame, num_cb, EOS_TOKEN,
        resources.stream.get());
    const auto t_sample_end = SteadyClock::now();
    state.prof_sample_ms += elapsed_ms(t_sample_start, t_sample_end);
    return true;
#else
    (void)state; (void)frame; (void)d_eos_flag;
    return false;
#endif
}

bool MagpieTTSBackend::gpu_check_stop_conditions(
    DecoderLoopState& state, int32_t frame, CudaBuffer& d_eos_flag,
    int32_t& h_eos_flag, int32_t& gen_frames_actual)
{
#if TRTF_HAS_CUDA_KERNELS
    constexpr int32_t REPEAT_STOP_THRESHOLD = 3;
    auto& resources = *mResources;
    const int32_t num_cb = state.num_cb;

    cudaMemcpyAsync(&h_eos_flag, d_eos_flag.data(), sizeof(int32_t),
                    cudaMemcpyDeviceToHost, resources.stream.get());

    bool repeated = false;
    if (frame >= REPEAT_STOP_THRESHOLD)
    {
        const int32_t check_start = frame + 1 - REPEAT_STOP_THRESHOLD;
        const std::size_t check_bytes = static_cast<std::size_t>(REPEAT_STOP_THRESHOLD)
            * num_cb * sizeof(int32_t);
        std::vector<int32_t> recent(static_cast<std::size_t>(REPEAT_STOP_THRESHOLD) * num_cb);
        cudaMemcpyAsync(recent.data(),
            static_cast<const int32_t*>(mDeviceAllCodes.data()) + check_start * num_cb,
            check_bytes, cudaMemcpyDeviceToHost, resources.stream.get());
        cudaStreamSynchronize(resources.stream.get());

        repeated = magpie_has_repeated_tail_frames(
            recent, num_cb, REPEAT_STOP_THRESHOLD);
        if (repeated)
        {
            gen_frames_actual = magpie_trimmed_frame_count_for_repetition(
                frame + 1, REPEAT_STOP_THRESHOLD);
            std::cerr << "[magpie-tts] Repetition detected at frame "
                      << frame << ", trimming to " << gen_frames_actual
                      << " frames" << std::endl;
        }
    }
    else
    {
        cudaStreamSynchronize(resources.stream.get());
    }

    return (h_eos_flag != 0 || repeated);
#else
    (void)state; (void)frame; (void)d_eos_flag;
    (void)h_eos_flag; (void)gen_frames_actual;
    return false;
#endif
}

void MagpieTTSBackend::gpu_update_text_completion(
    DecoderLoopState& state, int32_t frame)
{
    if (state.use_cross_attn_tracking)
    {
        auto& resources = *mResources;
        std::vector<float> xattn(static_cast<std::size_t>(state.max_source_positions));
        cudaMemcpyAsync(xattn.data(), mDeviceCrossAttnWeights.data(),
            static_cast<std::size_t>(state.max_source_positions) * sizeof(float),
            cudaMemcpyDeviceToHost, resources.stream.get());
        cudaStreamSynchronize(resources.stream.get());

        if (update_magpie_text_consumed_from_cross_attn(
                xattn.data(),
                state.max_source_positions,
                state.text_consumed_threshold,
                state.max_peak_pos,
                state.text_consumed))
        {
            std::cerr << "[magpie-tts] Text consumed at frame " << frame
                      << " (max_peak_pos=" << state.max_peak_pos
                      << ", threshold=" << state.text_consumed_threshold
                      << ", text_len=" << mTextLength << ")" << std::endl;
        }
    }

    if (!state.use_cross_attn_tracking)
    {
        update_magpie_text_consumed_from_heuristic(
            state.estimated_frames,
            frame,
            state.text_consumed);
    }
}

std::vector<int32_t> MagpieTTSBackend::run_gpu_greedy_loop(
    DecoderLoopState& state, int32_t max_frames)
{
#if TRTF_HAS_CUDA_KERNELS
    constexpr int32_t EOS_CHECK_INTERVAL = 16;
    constexpr int32_t MIN_FRAMES = 4;

    auto& resources = *mResources;
    const int32_t num_cb = state.num_cb;

    // Allocate device EOS flag (1 int32)
    CudaBuffer d_eos_flag(sizeof(int32_t));
    int32_t h_eos_flag = 0;
    cudaMemsetAsync(d_eos_flag.data(), 0, sizeof(int32_t), resources.stream.get());

    int32_t gen_frames_actual = 0;

    for (int32_t frame = 0; frame < max_frames; ++frame)
    {
        if (!gpu_greedy_frame_step(state, frame, d_eos_flag))
        {
            break;
        }

        gen_frames_actual = frame + 1;

        // Periodic checks (EOS, repetition, text-completion) every N frames
        const bool periodic = should_run_magpie_periodic_check(
            frame, MIN_FRAMES, EOS_CHECK_INTERVAL);
        if (periodic && gpu_check_stop_conditions(state, frame, d_eos_flag,
                                                  h_eos_flag, gen_frames_actual))
        {
            break;
        }
        if (periodic && !state.text_consumed)
        {
            gpu_update_text_completion(state, frame);
        }
        if (check_finished_limit(state, frame))
        {
            gen_frames_actual = frame + 1;
            break;
        }
    }

    // Final sync and bulk D2H of all accumulated codes
    cudaStreamSynchronize(resources.stream.get());
    const std::size_t total_codes_bytes = static_cast<std::size_t>(gen_frames_actual) *
        num_cb * sizeof(int32_t);
    std::vector<int32_t> all_codes(static_cast<std::size_t>(gen_frames_actual) * num_cb);
    cudaMemcpy(all_codes.data(), mDeviceAllCodes.data(), total_codes_bytes,
               cudaMemcpyDeviceToHost);
    return all_codes;
#else
    (void)state; (void)max_frames;
    return {};
#endif
}

// ---------------------------------------------------------------------------
// CPU / non-greedy decode loop
// ---------------------------------------------------------------------------

void MagpieTTSBackend::cpu_compute_frame_embed(
    DecoderLoopState& state, const std::vector<int32_t>& prev_codes)
{
    const int32_t num_cb = state.num_cb;
    const int32_t cb_size = state.cb_size;
    const int32_t hidden = state.hidden;

#if TRTF_HAS_CUDA_KERNELS
    if (state.use_gpu_kernels)
    {
        auto& resources = *mResources;
        magpie_gather_average_embed_device(
            static_cast<const float*>(mAudioEmbedDevice.data()),
            static_cast<const int32_t*>(mDevicePrevCodes.data()),
            num_cb, cb_size, hidden,
            static_cast<float*>(resources.d_input_embed.data()),
            resources.stream.get());
        return;
    }
#endif

    std::fill(state.embed_buf.begin(), state.embed_buf.end(), 0.0F);
    for (int32_t cb = 0; cb < num_cb; ++cb)
    {
        const float* table = mAudioEmbed.data() +
            static_cast<std::size_t>(cb) * cb_size * hidden;
        lookup_embed(table, prev_codes[cb], state.cb_embed.data());
        sum_embeds(state.embed_buf.data(), state.cb_embed.data(), state.embed_buf.data());
    }
    const float inv_cb = 1.0F / static_cast<float>(num_cb);
    for (int32_t i = 0; i < hidden; ++i)
    {
        state.embed_buf[i] *= inv_cb;
    }
}

bool MagpieTTSBackend::cpu_run_conditioned_step(
    DecoderLoopState& state, int32_t frame)
{
    auto& engine = *mDecoderEngine;
    auto& cache = *mCache;
    auto& resources = *mResources;
    const int32_t hidden = state.hidden;

    if (state.use_cfg || frame == 0) bind_cross_kv();

#if TRTF_HAS_CUDA_KERNELS
    if (state.use_gpu_kernels)
    {
        if (!run_decoder_step_device(engine, cache, resources,
                0, state.logits, state.error,
                nullptr, hidden, 1.0F,
                {}, 0.0F,
                /*input_embed_device_ready=*/true))
        {
            std::cerr << "[magpie-tts] Decode step " << frame << " failed: "
                      << state.error << std::endl;
            return false;
        }
        return true;
    }
#endif

    if (!run_decoder_step_device(engine, cache, resources,
            0, state.logits, state.error,
            state.embed_buf.data(), hidden, 1.0F))
    {
        std::cerr << "[magpie-tts] Decode step " << frame << " failed: "
                  << state.error << std::endl;
        return false;
    }
    return true;
}

bool MagpieTTSBackend::cpu_sample_frame_codes(
    DecoderLoopState& state,
    std::vector<int32_t>& frame_codes, bool& eos)
{
    const auto decoded = decode_magpie_frame_codes(
        state.logits,
        state.num_cb,
        state.cb_size,
        mConfig.greedy,
        mConfig.temperature,
        mConfig.top_k,
        [this](const float* cb_logits, int32_t vocab_size, float temperature, int32_t top_k)
        {
            return sample_top_k(cb_logits, vocab_size, temperature, top_k);
        });
    frame_codes = decoded.frame_codes;
    eos = decoded.eos;
    return true;
}

std::vector<int32_t> MagpieTTSBackend::run_cpu_sampling_loop(
    DecoderLoopState& state, int32_t max_frames)
{
    constexpr int32_t MIN_FRAMES = 4;
    constexpr int32_t REPEAT_STOP_THRESHOLD = 3;

    const int32_t num_cb = state.num_cb;

    std::vector<int32_t> all_codes;
    all_codes.reserve(static_cast<std::size_t>(max_frames) * num_cb);

    std::vector<int32_t> prev_codes(static_cast<std::size_t>(num_cb), kMagpieBosToken);

    for (int32_t frame = 0; frame < max_frames; ++frame)
    {
        // Embed computation
        const auto t_embed_start = SteadyClock::now();
        cpu_compute_frame_embed(state, prev_codes);
        const auto t_embed_end = SteadyClock::now();
        state.prof_embed_ms += elapsed_ms(t_embed_start, t_embed_end);

        // Conditioned decoder step
        const auto t_step_start = SteadyClock::now();
        if (!cpu_run_conditioned_step(state, frame)) break;

        // CFG: unconditional pass + blend
        if (state.use_cfg && !run_cfg_uncond_pass_cpu(state, frame)) break;
        const auto t_step_end = SteadyClock::now();
        state.prof_trt_step_ms += elapsed_ms(t_step_start, t_step_end);

        // Sample frame codes
        const auto t_sample_start = SteadyClock::now();
        std::vector<int32_t> frame_codes;
        bool eos = false;
        cpu_sample_frame_codes(state, frame_codes, eos);
        const auto t_sample_end = SteadyClock::now();
        state.prof_sample_ms += elapsed_ms(t_sample_start, t_sample_end);

        for (int32_t cb = 0; cb < num_cb; ++cb)
        {
            all_codes.push_back(frame_codes[cb]);
        }
        prev_codes = frame_codes;
        upload_prev_codes_to_device(mDevicePrevCodes, prev_codes.data(), num_cb,
                                    state.use_gpu_kernels, state.use_gpu_greedy);

        // Repetition-based early stopping
        if (magpie_has_repeated_tail_frames(
                all_codes, num_cb, REPEAT_STOP_THRESHOLD))
        {
            const int32_t trim_to = magpie_trimmed_frame_count_for_repetition(
                frame + 1, REPEAT_STOP_THRESHOLD);
            all_codes.resize(static_cast<std::size_t>(trim_to) * num_cb);
            std::cerr << "[magpie-tts] Repetition detected at frame "
                      << frame << ", trimming to " << trim_to
                      << " frames" << std::endl;
            break;
        }

        if (should_stop_magpie_on_eos(eos, frame, MIN_FRAMES)) break;

        update_text_completion(state, frame);

        if (check_finished_limit(state, frame)) break;
    }

    return all_codes;
}

// ---------------------------------------------------------------------------
// run_decoder() -- multi-codebook autoregressive decode (orchestrator)
// ---------------------------------------------------------------------------

std::vector<int32_t> MagpieTTSBackend::run_decoder(int32_t max_frames)
{
    if (!mDecoderEngine || !mCache || !mResources) return {};

    DecoderLoopState state = init_decoder_state();

    // Reset KV cache
    mCache->reset(mResources->stream.get());
    cudaStreamSynchronize(mResources->stream.get());

    // CFG: reset unconditional KV cache
    if (state.use_cfg)
    {
        mCacheUncond->reset(mResourcesUncond->stream.get());
        cudaStreamSynchronize(mResourcesUncond->stream.get());
    }

    // Bind cross-attention K/V
    bind_cross_kv();

    // Phase 1: Context prefill
    const int32_t ctx_frames = prefill_context(state);
    if (ctx_frames < 0) return {};

    // Phase 2: Autoregressive decode — upload BOS codes and dispatch to loop
    std::vector<int32_t> bos(static_cast<std::size_t>(state.num_cb), kMagpieBosToken);
    upload_prev_codes_to_device(mDevicePrevCodes, bos.data(), state.num_cb,
                                state.use_gpu_kernels, false);

    std::vector<int32_t> all_codes =
        (state.use_gpu_greedy && mDeviceAllCodes.ok())
        ? run_gpu_greedy_loop(state, max_frames)
        : run_cpu_sampling_loop(state, max_frames);

    const int32_t gen_frames = static_cast<int32_t>(all_codes.size()) /
        std::max(state.num_cb, 1);
    std::cerr << "[magpie-tts] Generated " << gen_frames
              << " frames (" << all_codes.size() << " codes)" << std::endl;

    log_decoder_profiling(state, ctx_frames, gen_frames);
    log_frame_preview(all_codes, state.num_cb);
    return all_codes;
}

// ---------------------------------------------------------------------------
// run_codec() -- codes -> waveform via codec engine
// ---------------------------------------------------------------------------

std::vector<float> MagpieTTSBackend::run_codec(
    const std::vector<int32_t>& codes, int32_t num_frames)
{
    const int32_t num_cb = mConfig.num_codebooks;

    if (num_frames <= 0) return {};

    // If no codec engine, generate silence as fallback
    if (!mCodecEngine || !mCodecCtx)
    {
        std::cerr << "[magpie-tts] No codec engine, generating silence" << std::endl;
        const int32_t samples_per_frame = mConfig.sample_rate /
            std::max(static_cast<int32_t>(mConfig.frames_per_second), 1);
        const auto total = static_cast<std::size_t>(num_frames) * samples_per_frame;
        return std::vector<float>(total, 0.0F);
    }
    std::vector<float> waveform = run_codec_engine(
        *mCodecEngine, *mCodecCtx, codes, num_frames, num_cb);
    if (waveform.empty())
    {
        return {};
    }
    std::cerr << "[magpie-tts] Codec: " << num_frames << " frames -> "
              << waveform.size() << " samples" << std::endl;
    return waveform;
}

// ---------------------------------------------------------------------------
// generate_audio() helpers
// ---------------------------------------------------------------------------

void MagpieTTSBackend::apply_env_overrides()
{
    maybe_enable_magpie_greedy(mConfig);

    const char* env_cfg = std::getenv("TRTF_MAGPIE_CFG_SCALE");
    if (env_cfg != nullptr)
    {
        float val = std::atof(env_cfg);
        if (val > 0.0F) mConfig.cfg_scale = val;
    }
    const char* env_limit = std::getenv("TRTF_MAGPIE_FINISHED_LIMIT");
    if (env_limit != nullptr)
    {
        int32_t val = std::atoi(env_limit);
        if (val >= 0) mConfig.finished_limit_with_eot = val;
    }
    const char* env_seed = std::getenv("TRTF_MAGPIE_SEED");
    if (env_seed != nullptr)
    {
        mRng.seed(static_cast<std::mt19937::result_type>(std::atol(env_seed)));
    }
}

void MagpieTTSBackend::ensure_cfg_resources()
{
    if (mConfig.cfg_scale <= 1.0F || mCacheUncond || !mDecoderEngine)
    {
        return;
    }

    mCacheUncond = std::make_unique<DeviceKvCache>(*mDecoderEngine);
    mResourcesUncond = std::make_unique<DeviceResources>(*mDecoderEngine);

    const int32_t dec_layers = mConfig.decoder_layers > 0
        ? mConfig.decoder_layers : (mDecoderEngine ? mDecoderEngine->num_layers : 1);
    const std::size_t enc_buf_size = static_cast<std::size_t>(mConfig.max_source_positions) *
        static_cast<std::size_t>(mConfig.hidden_size) * sizeof(float);
    mCrossKUncond.reserve(static_cast<std::size_t>(dec_layers));
    mCrossVUncond.reserve(static_cast<std::size_t>(dec_layers));
    for (int32_t i = 0; i < dec_layers; ++i)
    {
        mCrossKUncond.emplace_back(enc_buf_size);
        mCrossVUncond.emplace_back(enc_buf_size);
    }
    mEncoderOutputUncond = CudaBuffer(enc_buf_size);
    const auto logits_bytes = static_cast<std::size_t>(mConfig.num_codebooks) *
        static_cast<std::size_t>(mConfig.codebook_size) * sizeof(float);
    mDeviceLogitsCond = CudaBuffer(logits_bytes);
    mDeviceLogitsUncond = CudaBuffer(logits_bytes);

    // Allocate cross_attn_weights scratch if the engine has that output
    // and the scratch wasn't allocated at construction time (e.g. when
    // cfg_scale was overridden from 1.0 to >1.0 via TRTF_MAGPIE_CFG_SCALE).
    if (mHasCrossAttnOutput && !mDeviceCrossAttnWeightsScratch.ok())
    {
        const auto xattn_bytes = static_cast<std::size_t>(mConfig.max_source_positions) * sizeof(float);
        mDeviceCrossAttnWeightsScratch = CudaBuffer(xattn_bytes);
    }
}

void MagpieTTSBackend::run_cfg_encoder(const std::vector<int32_t>& text_ids)
{
    if (mConfig.cfg_scale <= 1.0F || !mEncoderOutputUncond.ok() || mCrossKUncond.empty())
    {
        return;
    }

    std::cerr << "[magpie-tts] CFG: encoding null text for unconditional path ..."
              << std::endl;

    const auto enc_bytes = mEncoderOutput.size();

    std::vector<int32_t> empty_ids;
    run_encoder(empty_ids, /*speaker_id=*/0, /*language_id=*/0);

    cudaMemcpy(mEncoderOutputUncond.data(), mEncoderOutput.data(),
               enc_bytes, cudaMemcpyDeviceToDevice);

    run_encoder(text_ids, /*speaker_id=*/0, /*language_id=*/0);

    compute_cross_kv_uncond();
}

void MagpieTTSBackend::log_pipeline_profiling(
    int32_t num_frames, int32_t num_samples,
    double ms_encoder, double ms_decoder,
    double ms_codec, double ms_total) const
{
    const double ms_per_frame = (num_frames > 0) ? ms_decoder / num_frames : 0.0;
    const double audio_duration = static_cast<double>(num_samples) / mConfig.sample_rate;
    const double rtf = (audio_duration > 0.0) ? (ms_total / 1000.0) / audio_duration : 0.0;

    std::cerr << "\n[magpie-tts] ===== PROFILING REPORT =====\n"
              << "[magpie-tts]   Encoder:        " << ms_encoder << " ms\n"
              << "[magpie-tts]   Cross-KV:       D2D copies (per-layer buffers)\n"
              << "[magpie-tts]   Decoder:        " << ms_decoder << " ms ("
              << num_frames << " frames, " << ms_per_frame << " ms/frame)\n"
              << "[magpie-tts]   Codec:          " << ms_codec << " ms\n"
              << "[magpie-tts]   Total pipeline: " << ms_total << " ms\n"
              << "[magpie-tts]   Audio duration: " << audio_duration << " s ("
              << num_samples << " samples @ " << mConfig.sample_rate << " Hz)\n"
              << "[magpie-tts]   RTF (real-time factor): " << rtf
              << " (< 1.0 = faster than real-time)\n"
              << "[magpie-tts]   CFG scale:      " << mConfig.cfg_scale
              << (mConfig.cfg_scale > 1.0F ? " (enabled, 2x decoder steps)" : " (disabled)")
              << "\n"
              << "[magpie-tts]   finished_limit: " << mConfig.finished_limit_with_eot
              << " (text_len=" << mTextLength << ", est_frames="
              << static_cast<int32_t>(static_cast<float>(mTextLength) * 3.0F) << ")\n"
              << "[magpie-tts] =============================\n" << std::endl;
}

// ---------------------------------------------------------------------------
// generate_audio() -- full pipeline orchestration
// ---------------------------------------------------------------------------

LegacyAudioResult MagpieTTSBackend::generate_audio(
    const std::vector<int32_t>& text_ids,
    int32_t max_frames)
{
    LegacyAudioResult result;
    result.sample_rate = mConfig.sample_rate;

    if (!is_available())
    {
        std::cerr << "[magpie-tts] Backend not fully initialized" << std::endl;
        return result;
    }

    apply_env_overrides();
    ensure_cfg_resources();

    mTextLength = static_cast<int32_t>(text_ids.size());

    std::cerr << "[magpie-tts] Starting pipeline with " << text_ids.size()
              << " text tokens, max_frames=" << max_frames
              << (mConfig.greedy ? " (greedy)" : "")
              << ", cfg_scale=" << mConfig.cfg_scale
              << ", finished_limit=" << mConfig.finished_limit_with_eot
              << std::endl;

    const auto t_pipeline_start = SteadyClock::now();

    // Stage 1: Encode text + speaker/language
    std::cerr << "[magpie-tts] Running encoder ..." << std::endl;
    const auto t_enc_start = SteadyClock::now();
    run_encoder(text_ids, /*speaker_id=*/0, /*language_id=*/0);
    const auto t_enc_end = SteadyClock::now();

    // Stage 2: Copy encoder output to per-layer cross-attention buffers
    compute_cross_kv();

    // Stage 2b (CFG): Run encoder with empty text for unconditional cross-KV
    run_cfg_encoder(text_ids);

    // Stage 3: Autoregressive decode -> multi-codebook codes
    std::cerr << "[magpie-tts] Running decoder ..." << std::endl;
    const auto t_dec_start = SteadyClock::now();
    auto codes = run_decoder(max_frames);
    const auto t_dec_end = SteadyClock::now();
    if (codes.empty())
    {
        std::cerr << "[magpie-tts] Decoder produced no codes" << std::endl;
        return result;
    }

    const int32_t num_frames = static_cast<int32_t>(codes.size()) / mConfig.num_codebooks;

    // Stage 4: Codec -> waveform
    std::cerr << "[magpie-tts] Running codec ..." << std::endl;
    const auto t_codec_start = SteadyClock::now();
    auto waveform = run_codec(codes, num_frames);
    const auto t_codec_end = SteadyClock::now();
    if (waveform.empty())
    {
        std::cerr << "[magpie-tts] Codec produced no audio" << std::endl;
        return result;
    }

    result.waveform = std::move(waveform);
    result.num_samples = static_cast<int32_t>(result.waveform.size());

    const auto t_pipeline_end = SteadyClock::now();

    log_pipeline_profiling(
        num_frames, result.num_samples,
        elapsed_ms(t_enc_start, t_enc_end),
        elapsed_ms(t_dec_start, t_dec_end),
        elapsed_ms(t_codec_start, t_codec_end),
        elapsed_ms(t_pipeline_start, t_pipeline_end));

    return result;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<MagpieTTSBackend> CreateMagpieTTSBackend(
    std::unique_ptr<DecoderStepEngine> decoder_engine,
    TrtUniquePtr<nvinfer1::ICudaEngine> encoder_engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> encoder_context,
    std::vector<float> audio_embed,
    std::vector<float> text_embed,
    std::vector<float> context_embed,
    std::vector<int32_t> context_lengths,
    const FastPathModelConfig& cfg)
{
    MagpieTTSConfig magpie_cfg;
    magpie_cfg.sample_rate = cfg.audio_sample_rate;
    magpie_cfg.hidden_size = cfg.magpie_hidden_size > 0
        ? cfg.magpie_hidden_size : cfg.hidden_size;
    magpie_cfg.num_codebooks = cfg.magpie_num_codebooks;
    magpie_cfg.codebook_size = cfg.magpie_codebook_size;
    magpie_cfg.frames_per_second = cfg.magpie_fps;
    magpie_cfg.num_speakers = cfg.magpie_num_speakers;
    magpie_cfg.encoder_layers = cfg.magpie_encoder_layers;
    magpie_cfg.decoder_layers = cfg.magpie_decoder_layers;
    magpie_cfg.text_vocab_size = cfg.magpie_text_vocab_size;
    magpie_cfg.max_source_positions = cfg.magpie_max_source_positions;
    magpie_cfg.xa_n_heads = cfg.magpie_xa_n_heads;
    magpie_cfg.xa_d_head = cfg.magpie_xa_d_head;
    magpie_cfg.cfg_scale = cfg.magpie_cfg_scale;
    magpie_cfg.finished_limit_with_eot = cfg.magpie_finished_limit_with_eot;

    auto backend = std::make_unique<MagpieTTSBackend>(
        std::move(decoder_engine), std::move(encoder_engine),
        std::move(encoder_context),
        std::move(audio_embed), std::move(text_embed),
        std::move(context_embed), std::move(context_lengths),
        std::move(magpie_cfg));

    if (!backend->is_available())
    {
        return nullptr;
    }
    return backend;
}

} // namespace trtf

#endif // TRTF_HAS_TRT
