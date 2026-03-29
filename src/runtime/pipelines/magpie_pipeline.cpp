#include "runtime/pipelines/magpie_pipeline.h"

#if TRTF_HAS_TRT

#include "runtime/domains/audio/audio_configs.h"
#include "runtime/domains/audio/magpie_codec_plan.h"
#include "runtime/domains/audio/magpie_decode_policy.h"
#include "runtime/domains/audio/magpie_decoder_plan.h"
#include "runtime/domains/audio/magpie_text_completion_policy.h"
#include "runtime/core/trt_engine_lifecycle.h"
#include "runtime/core/trt_decode_runtime.h"

#ifndef TRTF_HAS_CUDA_KERNELS
#define TRTF_HAS_CUDA_KERNELS 0
#endif

#if TRTF_HAS_CUDA_KERNELS
#include "runtime/domains/audio/magpie_kernels.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

namespace trtf {

namespace {
using SteadyClock = std::chrono::steady_clock;
using TimePoint = SteadyClock::time_point;
inline double elapsed_ms(TimePoint start, TimePoint end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void log_magpie_frame_preview(const std::vector<int32_t>& all_codes, int32_t num_cb)
{
    const int32_t gen_frames = static_cast<int32_t>(all_codes.size()) / std::max(num_cb, 1);
    for (int32_t f = 0; f < std::min(gen_frames, 10); ++f)
    {
        std::cerr << "[magpie-tts]   frame " << f << ": [";
        for (int32_t cb = 0; cb < num_cb; ++cb)
        {
            if (cb > 0) std::cerr << ", ";
            std::cerr << all_codes[static_cast<std::size_t>(f) * num_cb + cb];
        }
        std::cerr << "]" << std::endl;
    }
    if (gen_frames <= 15) return;
    std::cerr << "[magpie-tts]   ..." << std::endl;
    for (int32_t f = gen_frames - 5; f < gen_frames; ++f)
    {
        std::cerr << "[magpie-tts]   frame " << f << ": [";
        for (int32_t cb = 0; cb < num_cb; ++cb)
        {
            if (cb > 0) std::cerr << ", ";
            std::cerr << all_codes[static_cast<std::size_t>(f) * num_cb + cb];
        }
        std::cerr << "]" << std::endl;
    }
}

bool check_magpie_gpu_kernels_available(
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

void upload_magpie_prev_codes_to_device(
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

void maybe_enable_magpie_greedy(MagpieTTSConfig& cfg)
{
    const char* env = std::getenv("TRTF_MAGPIE_GREEDY");
    if (env != nullptr && std::string(env) == "1")
    {
        cfg.greedy = true;
    }
}

} // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════
// MagpiePipeline (TrtModule-based)
// ═══════════════════════════════════════════════════════════════════════════

MagpiePipeline::MagpiePipeline(
    std::unique_ptr<TrtModule> encoder,
    std::unique_ptr<TrtModule> decoder,
    std::unique_ptr<KvCache> decoder_cache,
    std::unique_ptr<TrtModule> codec,
    std::unique_ptr<KvCache> decoder_cache_uncond,
    std::vector<CudaBuffer> cross_k,
    std::vector<CudaBuffer> cross_v,
    std::vector<CudaBuffer> cross_k_uncond,
    std::vector<CudaBuffer> cross_v_uncond,
    CudaBuffer encoder_output,
    CudaBuffer encoder_output_uncond,
    std::vector<float> audio_embed,
    std::vector<float> text_embed,
    std::vector<float> context_embed,
    std::vector<int32_t> context_lengths,
    MagpieTTSConfig config,
    cudaStream_t stream,
    std::shared_ptr<ITokenizer> tokenizer,
    std::string model_id_str)
    : encoder_(std::move(encoder))
    , decoder_(std::move(decoder))
    , decoder_cache_(std::move(decoder_cache))
    , codec_(std::move(codec))
    , decoder_cache_uncond_(std::move(decoder_cache_uncond))
    , cross_k_(std::move(cross_k))
    , cross_v_(std::move(cross_v))
    , cross_k_uncond_(std::move(cross_k_uncond))
    , cross_v_uncond_(std::move(cross_v_uncond))
    , encoder_output_(std::move(encoder_output))
    , encoder_output_uncond_(std::move(encoder_output_uncond))
    , cross_attn_weights_(0)
    , cross_attn_weights_scratch_(0)
    , audio_embed_(std::move(audio_embed))
    , text_embed_(std::move(text_embed))
    , context_embed_(std::move(context_embed))
    , context_lengths_(std::move(context_lengths))
    , audio_embed_device_(0)
    , context_embed_device_(0)
    , device_codes_(static_cast<std::size_t>(config.num_codebooks) * sizeof(int32_t))
    , device_full_argmax_(static_cast<std::size_t>(config.num_codebooks) * sizeof(int32_t))
    , device_prev_codes_(static_cast<std::size_t>(config.num_codebooks) * sizeof(int32_t))
    , device_all_codes_(static_cast<std::size_t>(512) * config.num_codebooks * sizeof(int32_t))
    , device_logits_cond_(0)
    , device_logits_uncond_(0)
    , stream_(stream)
    , config_(config)
    , tokenizer_(std::move(tokenizer))
    , model_id_(std::move(model_id_str))
    , rng_(std::random_device{}())
{
    if (!decoder_ || !decoder_->ok())
        throw std::runtime_error("MagpiePipeline: invalid decoder module");
    if (!decoder_cache_ || !decoder_cache_->ok())
        throw std::runtime_error("MagpiePipeline: invalid decoder cache");
    if (!encoder_ || !encoder_->ok())
        throw std::runtime_error("MagpiePipeline: invalid encoder module");

    upload_embeddings_to_gpu();
    init_cross_attn_resources();
    init_cfg_logit_buffers();
}

MagpiePipeline::~MagpiePipeline() = default;

// The rest of the MagpiePipeline methods (lines 1421-2633 of original audio_pipeline.cpp)
// are included verbatim below.

void MagpiePipeline::upload_embeddings_to_gpu()
{
    audio_embed_device_ = CudaBuffer(audio_embed_.size() * sizeof(float));
    context_embed_device_ = CudaBuffer(context_embed_.size() * sizeof(float));
    if (!audio_embed_.empty() && audio_embed_device_.ok())
        cudaMemcpy(audio_embed_device_.data(), audio_embed_.data(),
                   audio_embed_.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (!context_embed_.empty() && context_embed_device_.ok())
        cudaMemcpy(context_embed_device_.data(), context_embed_.data(),
                   context_embed_.size() * sizeof(float), cudaMemcpyHostToDevice);
}

void MagpiePipeline::init_cross_attn_resources()
{
    if (!decoder_->has_output("cross_attn_weights"))
        return;
    has_cross_attn_output_ = true;
    const auto xattn_bytes = static_cast<std::size_t>(config_.max_source_positions) * sizeof(float);
    cross_attn_weights_ = CudaBuffer(xattn_bytes);
    if (config_.cfg_scale > 1.0F)
        cross_attn_weights_scratch_ = CudaBuffer(xattn_bytes);
}

void MagpiePipeline::init_cfg_logit_buffers()
{
    if (config_.cfg_scale <= 1.0F)
        return;
    const auto logits_bytes = static_cast<std::size_t>(config_.num_codebooks) *
        static_cast<std::size_t>(config_.codebook_size) * sizeof(float);
    device_logits_cond_ = CudaBuffer(logits_bytes);
    device_logits_uncond_ = CudaBuffer(logits_bytes);
}

void MagpiePipeline::lookup_embed(const float* table, int32_t token_id,
                                   float* out) const
{
    const auto offset = static_cast<std::size_t>(token_id) *
                        static_cast<std::size_t>(config_.hidden_size);
    std::memcpy(out, table + offset,
                static_cast<std::size_t>(config_.hidden_size) * sizeof(float));
}

void MagpiePipeline::sum_embeds(const float* a, const float* b, float* out) const
{
    for (int32_t i = 0; i < config_.hidden_size; ++i)
        out[i] = a[i] + b[i];
}

int32_t MagpiePipeline::sample_top_k(const float* logits, int32_t vocab_size,
                                       float temperature, int32_t top_k)
{
    if (config_.greedy)
    {
        int32_t best = 0;
        for (int32_t i = 1; i < vocab_size; ++i)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    top_k = std::min(top_k, vocab_size);
    std::vector<int32_t> indices(static_cast<std::size_t>(vocab_size));
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
        [logits](int32_t a, int32_t b) { return logits[a] > logits[b]; });

    std::vector<float> probs(static_cast<std::size_t>(top_k));
    float max_logit = logits[indices[0]];
    float sum = 0.0F;
    for (int32_t i = 0; i < top_k; ++i)
    {
        probs[i] = std::exp((logits[indices[i]] - max_logit) / temperature);
        sum += probs[i];
    }
    for (int32_t i = 0; i < top_k; ++i)
        probs[i] /= sum;

    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    float r = dist(rng_);
    float cumulative = 0.0F;
    for (int32_t i = 0; i < top_k; ++i)
    {
        cumulative += probs[i];
        if (r < cumulative) return indices[i];
    }
    return indices[top_k - 1];
}

void MagpiePipeline::run_encoder(const std::vector<int32_t>& text_ids)
{
    const int32_t max_pos = config_.max_source_positions;

    std::vector<int32_t> padded(static_cast<std::size_t>(max_pos), 0);
    const auto copy_len = std::min(static_cast<int32_t>(text_ids.size()), max_pos);
    if (copy_len > 0)
        std::memcpy(padded.data(), text_ids.data(),
                     static_cast<std::size_t>(copy_len) * sizeof(int32_t));

    Tensor input_ids_tensor;
    input_ids_tensor.data = padded.data();
    input_ids_tensor.shape = {static_cast<int64_t>(max_pos)};
    input_ids_tensor.dtype = DType::kInt32;

    TensorMap inputs;
    inputs["input_ids"] = input_ids_tensor;

    // Use forward_async + sync to keep output on device, then D2D copy
    encoder_->forward_async(inputs);
    encoder_->sync();

    // Copy encoder output from module's internal buffer to our persistent buffer
    void* enc_out_ptr = encoder_->device_ptr("encoder_output");
    if (enc_out_ptr)
    {
        const auto bytes = static_cast<std::size_t>(max_pos) *
            static_cast<std::size_t>(config_.hidden_size) * sizeof(float);
        cudaMemcpy(encoder_output_.data(), enc_out_ptr, bytes, cudaMemcpyDeviceToDevice);
    }

    // Zero out encoder output for padded positions
    if (copy_len < max_pos)
    {
        const auto hidden = config_.hidden_size;
        const auto zero_offset = static_cast<std::size_t>(copy_len) *
                                  static_cast<std::size_t>(hidden) * sizeof(float);
        const auto zero_bytes = static_cast<std::size_t>(max_pos - copy_len) *
                                 static_cast<std::size_t>(hidden) * sizeof(float);
        cudaMemset(static_cast<char*>(encoder_output_.data()) + zero_offset, 0, zero_bytes);
    }

    std::cerr << "[magpie-tts] Encoder: processed " << copy_len
              << " tokens (padded to " << max_pos << ")" << std::endl;
}

// ---------------------------------------------------------------------------
// Cross-KV management
// ---------------------------------------------------------------------------

void MagpiePipeline::compute_cross_kv()
{
    const int32_t hidden = config_.hidden_size;
    const int32_t enc_seq = config_.max_source_positions;
    const std::size_t buf_size = static_cast<std::size_t>(enc_seq) *
                                 static_cast<std::size_t>(hidden) * sizeof(float);
    for (std::size_t i = 0; i < cross_k_.size(); ++i)
    {
        cudaMemcpy(cross_k_[i].data(), encoder_output_.data(),
                    buf_size, cudaMemcpyDeviceToDevice);
        cudaMemcpy(cross_v_[i].data(), encoder_output_.data(),
                    buf_size, cudaMemcpyDeviceToDevice);
    }
}

void MagpiePipeline::bind_cross_kv()
{
    const int32_t dec_layers = static_cast<int32_t>(cross_k_.size());
    for (int32_t i = 0; i < dec_layers; ++i)
    {
        const std::string cross_k_name = layer_tensor_name("cross_k", i);
        const std::string cross_v_name = layer_tensor_name("cross_v", i);
        decoder_->bind_external(cross_k_name, cross_k_[i].data());
        decoder_->bind_external(cross_v_name, cross_v_[i].data());
    }

    if (has_cross_attn_output_ && cross_attn_weights_.ok())
        decoder_->bind_external("cross_attn_weights", cross_attn_weights_.data());
}

void MagpiePipeline::compute_cross_kv_uncond()
{
    const int32_t hidden = config_.hidden_size;
    const int32_t enc_seq = config_.max_source_positions;
    const std::size_t buf_size = static_cast<std::size_t>(enc_seq) *
                                 static_cast<std::size_t>(hidden) * sizeof(float);
    for (std::size_t i = 0; i < cross_k_uncond_.size(); ++i)
    {
        cudaMemcpy(cross_k_uncond_[i].data(), encoder_output_uncond_.data(),
                    buf_size, cudaMemcpyDeviceToDevice);
        cudaMemcpy(cross_v_uncond_[i].data(), encoder_output_uncond_.data(),
                    buf_size, cudaMemcpyDeviceToDevice);
    }
}

void MagpiePipeline::bind_cross_kv_uncond()
{
    const int32_t dec_layers = static_cast<int32_t>(cross_k_uncond_.size());
    for (int32_t i = 0; i < dec_layers; ++i)
    {
        const std::string cross_k_name = layer_tensor_name("cross_k", i);
        const std::string cross_v_name = layer_tensor_name("cross_v", i);
        decoder_->bind_external(cross_k_name, cross_k_uncond_[i].data());
        decoder_->bind_external(cross_v_name, cross_v_uncond_[i].data());
    }

    if (has_cross_attn_output_ && cross_attn_weights_scratch_.ok())
        decoder_->bind_external("cross_attn_weights", cross_attn_weights_scratch_.data());
}

// ---------------------------------------------------------------------------
// Decoder step via TrtModule
// ---------------------------------------------------------------------------

void MagpiePipeline::run_decoder_step(const float* embed, int32_t embed_size,
                                       std::vector<float>& logits_out)
{
    std::vector<float> mask;
    decoder_cache_->build_attention_mask(mask);
    int32_t position = decoder_cache_->position();
    int32_t dummy_token = 0;
    float use_input_embed = 1.0F;

    std::vector<float> embed_buf(embed, embed + embed_size);

    Tensor token_tensor;
    token_tensor.data = &dummy_token;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;

    Tensor position_tensor;
    position_tensor.data = &position;
    position_tensor.shape = {1};
    position_tensor.dtype = DType::kInt32;

    Tensor mask_tensor;
    mask_tensor.data = mask.data();
    mask_tensor.shape = {static_cast<int64_t>(mask.size())};
    mask_tensor.dtype = DType::kFloat32;

    Tensor embed_tensor;
    embed_tensor.data = embed_buf.data();
    embed_tensor.shape = {static_cast<int64_t>(embed_size)};
    embed_tensor.dtype = DType::kFloat32;

    Tensor use_embed_tensor;
    use_embed_tensor.data = &use_input_embed;
    use_embed_tensor.shape = {1};
    use_embed_tensor.dtype = DType::kFloat32;

    TensorMap inputs;
    inputs["token_id"] = token_tensor;
    if (decoder_->has_input("position_id"))
        inputs["position_id"] = position_tensor;
    inputs["attention_mask"] = mask_tensor;
    inputs["input_embed"] = embed_tensor;
    inputs["use_input_embed"] = use_embed_tensor;

    TensorMap outputs = decoder_->forward(inputs);

    auto it = outputs.find("logits");
    if (it != outputs.end())
    {
        const auto& lt = it->second;
        auto n = lt.numel();
        logits_out.resize(static_cast<std::size_t>(n));
        std::memcpy(logits_out.data(), lt.data, n * sizeof(float));
    }

    decoder_cache_->advance();
}

void MagpiePipeline::run_decoder_step_uncond(const float* embed, int32_t embed_size,
                                              std::vector<float>& logits_out)
{
    // Swap to unconditional cache + cross-KV
    decoder_cache_uncond_->bind_to(*decoder_);
    bind_cross_kv_uncond();

    std::vector<float> mask;
    decoder_cache_uncond_->build_attention_mask(mask);
    int32_t position = decoder_cache_uncond_->position();
    int32_t dummy_token = 0;
    float use_input_embed = 1.0F;

    std::vector<float> embed_buf(embed, embed + embed_size);

    Tensor token_tensor;
    token_tensor.data = &dummy_token;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;

    Tensor position_tensor;
    position_tensor.data = &position;
    position_tensor.shape = {1};
    position_tensor.dtype = DType::kInt32;

    Tensor mask_tensor;
    mask_tensor.data = mask.data();
    mask_tensor.shape = {static_cast<int64_t>(mask.size())};
    mask_tensor.dtype = DType::kFloat32;

    Tensor embed_tensor;
    embed_tensor.data = embed_buf.data();
    embed_tensor.shape = {static_cast<int64_t>(embed_size)};
    embed_tensor.dtype = DType::kFloat32;

    Tensor use_embed_tensor;
    use_embed_tensor.data = &use_input_embed;
    use_embed_tensor.shape = {1};
    use_embed_tensor.dtype = DType::kFloat32;

    TensorMap inputs;
    inputs["token_id"] = token_tensor;
    if (decoder_->has_input("position_id"))
        inputs["position_id"] = position_tensor;
    inputs["attention_mask"] = mask_tensor;
    inputs["input_embed"] = embed_tensor;
    inputs["use_input_embed"] = use_embed_tensor;

    TensorMap outputs = decoder_->forward(inputs);

    auto it = outputs.find("logits");
    if (it != outputs.end())
    {
        const auto& lt = it->second;
        auto n = lt.numel();
        logits_out.resize(static_cast<std::size_t>(n));
        std::memcpy(logits_out.data(), lt.data, n * sizeof(float));
    }

    decoder_cache_uncond_->advance();

    // Restore conditioned cache + cross-KV
    decoder_cache_->bind_to(*decoder_);
    bind_cross_kv();
}

// ---------------------------------------------------------------------------
// Decoder loop state initialization
// ---------------------------------------------------------------------------

MagpiePipeline::DecoderLoopState MagpiePipeline::init_decoder_state() const
{
    DecoderLoopState s;
    const auto plan = make_magpie_decoder_plan(
        config_,
        static_cast<bool>(decoder_cache_uncond_),
        static_cast<bool>(decoder_cache_uncond_),  // resources == cache in new runtime
        !cross_k_uncond_.empty(),
        check_magpie_gpu_kernels_available(
            audio_embed_device_, device_codes_, device_full_argmax_, device_prev_codes_),
        has_cross_attn_output_,
        cross_attn_weights_.ok(),
        text_length_);
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

    s.embed_buf.resize(static_cast<std::size_t>(s.hidden));
    s.cb_embed.resize(static_cast<std::size_t>(s.hidden));

    return s;
}

// ---------------------------------------------------------------------------
// Phase 1: Context prefill
// ---------------------------------------------------------------------------

int32_t MagpiePipeline::prefill_context(DecoderLoopState& state)
{
    if (context_embed_.empty() || context_lengths_.empty())
        return 0;

    const int32_t ctx_frames = context_lengths_[0];
    const int32_t hidden = state.hidden;

    std::cerr << "[magpie-tts] Prefilling " << ctx_frames
              << " context frames ..." << std::endl;
    const auto t_prefill_start = SteadyClock::now();

    // Conditioned cache prefill
    decoder_cache_->bind_to(*decoder_);
    bind_cross_kv();

    const float* ctx_ptr = context_embed_.data();
    for (int32_t pos = 0; pos < ctx_frames; ++pos)
    {
        const float* frame_embed = ctx_ptr + static_cast<std::size_t>(pos) * hidden;
        run_decoder_step(frame_embed, hidden, state.logits);
    }

    const auto t_prefill_end = SteadyClock::now();
    state.prof_prefill_ms = elapsed_ms(t_prefill_start, t_prefill_end);

    // CFG: prefill unconditional cache with same speaker context but uncond cross-KV
    if (state.use_cfg && decoder_cache_uncond_)
    {
        std::cerr << "[magpie-tts] CFG: prefilling unconditional cache ("
                  << ctx_frames << " frames) ..." << std::endl;

        decoder_cache_uncond_->bind_to(*decoder_);
        bind_cross_kv_uncond();

        for (int32_t pos = 0; pos < ctx_frames; ++pos)
        {
            const float* frame_embed = ctx_ptr + static_cast<std::size_t>(pos) * hidden;

            std::vector<float> dummy_logits;
            // Use the uncond cache step
            std::vector<float> mask;
            decoder_cache_uncond_->build_attention_mask(mask);
            int32_t position = decoder_cache_uncond_->position();
            int32_t dummy_token = 0;
            float use_input_embed = 1.0F;
            std::vector<float> embed_buf(frame_embed, frame_embed + hidden);

            Tensor token_tensor;
            token_tensor.data = &dummy_token;
            token_tensor.shape = {1};
            token_tensor.dtype = DType::kInt32;

            Tensor position_tensor;
            position_tensor.data = &position;
            position_tensor.shape = {1};
            position_tensor.dtype = DType::kInt32;

            Tensor mask_tensor;
            mask_tensor.data = mask.data();
            mask_tensor.shape = {static_cast<int64_t>(mask.size())};
            mask_tensor.dtype = DType::kFloat32;

            Tensor embed_tensor;
            embed_tensor.data = embed_buf.data();
            embed_tensor.shape = {static_cast<int64_t>(hidden)};
            embed_tensor.dtype = DType::kFloat32;

            Tensor use_embed_tensor;
            use_embed_tensor.data = &use_input_embed;
            use_embed_tensor.shape = {1};
            use_embed_tensor.dtype = DType::kFloat32;

            TensorMap inputs;
            inputs["token_id"] = token_tensor;
            if (decoder_->has_input("position_id"))
                inputs["position_id"] = position_tensor;
            inputs["attention_mask"] = mask_tensor;
            inputs["input_embed"] = embed_tensor;
            inputs["use_input_embed"] = use_embed_tensor;

            decoder_->forward(inputs);
            decoder_cache_uncond_->advance();
        }

        // Restore conditioned state
        decoder_cache_->bind_to(*decoder_);
        bind_cross_kv();
    }

    return ctx_frames;
}

// ---------------------------------------------------------------------------
// CFG unconditional passes
// ---------------------------------------------------------------------------

bool MagpiePipeline::run_cfg_uncond_pass_gpu(DecoderLoopState& state, int32_t frame)
{
#if TRTF_HAS_CUDA_KERNELS
    // Save conditioned logits
    void* cond_logits_ptr = decoder_->device_ptr("logits");
    cudaMemcpyAsync(device_logits_cond_.data(), cond_logits_ptr,
        static_cast<std::size_t>(state.total_logits) * sizeof(float),
        cudaMemcpyDeviceToDevice, stream_);

    // Copy embed from conditioned decoder's input_embed to reuse
    void* cond_embed_ptr = decoder_->device_ptr("input_embed");

    // Run unconditional pass
    decoder_cache_uncond_->bind_to(*decoder_);
    bind_cross_kv_uncond();

    // Copy embed
    void* uncond_embed_ptr = decoder_->device_ptr("input_embed");
    cudaMemcpyAsync(uncond_embed_ptr, cond_embed_ptr,
        static_cast<std::size_t>(state.hidden) * sizeof(float),
        cudaMemcpyDeviceToDevice, stream_);

    // Forward async
    decoder_->forward_device_async({});
    decoder_cache_uncond_->advance();

    void* uncond_logits_ptr = decoder_->device_ptr("logits");

    // CFG interpolation: out = uncond + scale * (cond - uncond)
    // Restore conditioned bindings first
    decoder_cache_->bind_to(*decoder_);
    bind_cross_kv();

    magpie_cfg_interpolate_device(
        static_cast<const float*>(device_logits_cond_.data()),
        static_cast<const float*>(uncond_logits_ptr),
        static_cast<float*>(decoder_->device_ptr("logits")),
        config_.cfg_scale, state.total_logits,
        stream_);
    return true;
#else
    (void)state; (void)frame;
    return false;
#endif
}

bool MagpiePipeline::run_cfg_uncond_pass_cpu(DecoderLoopState& state, int32_t frame)
{
    std::vector<float> cond_logits = state.logits;
    std::vector<float> uncond_logits;

    // Run unconditional pass using full step (which swaps cache/cross-KV)
    run_decoder_step_uncond(state.embed_buf.data(), state.hidden, uncond_logits);

    // CPU-side CFG blend: logits = uncond + scale * (cond - uncond)
    const auto n = std::min(cond_logits.size(), uncond_logits.size());
    state.logits.resize(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        state.logits[i] = uncond_logits[i] + config_.cfg_scale *
            (cond_logits[i] - uncond_logits[i]);
    }
    (void)frame;
    return true;
}

// ---------------------------------------------------------------------------
// Text-completion tracking
// ---------------------------------------------------------------------------

void MagpiePipeline::update_text_completion(DecoderLoopState& state, int32_t frame)
{
    if (state.use_cross_attn_tracking && !state.text_consumed)
    {
        std::vector<float> xattn(static_cast<std::size_t>(state.max_source_positions));
        cudaMemcpy(xattn.data(), cross_attn_weights_.data(),
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
                      << ", text_len=" << text_length_ << ")" << std::endl;
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

bool MagpiePipeline::check_finished_limit(DecoderLoopState& state, int32_t frame)
{
    if (!config_.enable_finished_limit_stop)
        return false;
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
// CPU frame embed computation
// ---------------------------------------------------------------------------

void MagpiePipeline::cpu_compute_frame_embed(
    DecoderLoopState& state, const std::vector<int32_t>& prev_codes)
{
    const int32_t num_cb = state.num_cb;
    const int32_t cb_size = state.cb_size;
    const int32_t hidden = state.hidden;

    std::fill(state.embed_buf.begin(), state.embed_buf.end(), 0.0F);
    for (int32_t cb = 0; cb < num_cb; ++cb)
    {
        const float* table = audio_embed_.data() +
            static_cast<std::size_t>(cb) * cb_size * hidden;
        lookup_embed(table, prev_codes[cb], state.cb_embed.data());
        sum_embeds(state.embed_buf.data(), state.cb_embed.data(), state.embed_buf.data());
    }
    const float inv_cb = 1.0F / static_cast<float>(num_cb);
    for (int32_t i = 0; i < hidden; ++i)
        state.embed_buf[i] *= inv_cb;
}

// ---------------------------------------------------------------------------
// GPU greedy loop
// ---------------------------------------------------------------------------

bool MagpiePipeline::gpu_greedy_frame_step(
    DecoderLoopState& state, int32_t frame, CudaBuffer& d_eos_flag)
{
#if TRTF_HAS_CUDA_KERNELS
    constexpr int32_t EOS_TOKEN = 2017;
    constexpr int32_t AUDIO_RANGE = 2016;

    const int32_t num_cb = state.num_cb;
    const int32_t cb_size = state.cb_size;
    const int32_t hidden = state.hidden;

    // Embed
    const auto t_embed_start = SteadyClock::now();
    void* embed_ptr = decoder_->device_ptr("input_embed");
    magpie_gather_average_embed_device(
        static_cast<const float*>(audio_embed_device_.data()),
        static_cast<const int32_t*>(device_prev_codes_.data()),
        num_cb, cb_size, hidden,
        static_cast<float*>(embed_ptr),
        stream_);
    const auto t_embed_end = SteadyClock::now();
    state.prof_embed_ms += elapsed_ms(t_embed_start, t_embed_end);

    // Build mask and position on host, upload
    const auto t_step_start = SteadyClock::now();
    if (state.use_cfg || frame == 0)
    {
        decoder_cache_->bind_to(*decoder_);
        bind_cross_kv();
    }

    // Use forward_async for GPU-resident path
    // Set use_input_embed = 1.0
    float use_embed_val = 1.0F;
    void* use_embed_ptr = decoder_->device_ptr("use_input_embed");
    cudaMemcpyAsync(use_embed_ptr, &use_embed_val, sizeof(float),
                    cudaMemcpyHostToDevice, stream_);

    // Build mask + position
    std::vector<float> mask;
    decoder_cache_->build_attention_mask(mask);
    int32_t position = decoder_cache_->position();

    void* mask_ptr = decoder_->device_ptr("attention_mask");
    cudaMemcpyAsync(mask_ptr, mask.data(), mask.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);

    if (decoder_->has_input("position_id"))
    {
        void* pos_ptr = decoder_->device_ptr("position_id");
        cudaMemcpyAsync(pos_ptr, &position, sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream_);
    }

    int32_t dummy_token = 0;
    void* token_ptr = decoder_->device_ptr("token_id");
    cudaMemcpyAsync(token_ptr, &dummy_token, sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream_);

    decoder_->forward_device_async({});
    decoder_cache_->advance();

    // CFG: unconditional pass + device-side blend
    if (state.use_cfg && !run_cfg_uncond_pass_gpu(state, frame))
        return false;

    const auto t_step_end = SteadyClock::now();
    state.prof_trt_step_ms += elapsed_ms(t_step_start, t_step_end);

    // Sample + scatter
    const auto t_sample_start = SteadyClock::now();
    void* logits_ptr = decoder_->device_ptr("logits");

    magpie_greedy_sample_device(
        static_cast<const float*>(logits_ptr),
        num_cb, cb_size, AUDIO_RANGE,
        static_cast<int32_t*>(device_codes_.data()),
        static_cast<int32_t*>(device_full_argmax_.data()),
        stream_);

    magpie_scatter_codes_device(
        static_cast<const int32_t*>(device_codes_.data()),
        static_cast<int32_t*>(device_all_codes_.data()),
        static_cast<int32_t*>(device_prev_codes_.data()),
        static_cast<const int32_t*>(device_full_argmax_.data()),
        static_cast<int32_t*>(d_eos_flag.data()),
        frame, num_cb, EOS_TOKEN,
        stream_);
    const auto t_sample_end = SteadyClock::now();
    state.prof_sample_ms += elapsed_ms(t_sample_start, t_sample_end);
    return true;
#else
    (void)state; (void)frame; (void)d_eos_flag;
    return false;
#endif
}

void MagpiePipeline::gpu_greedy_update_text_consumed(
    DecoderLoopState& state, int32_t frame)
{
#if TRTF_HAS_CUDA_KERNELS
    if (state.text_consumed)
        return;
    if (state.use_cross_attn_tracking)
    {
        std::vector<float> xattn(static_cast<std::size_t>(state.max_source_positions));
        cudaMemcpyAsync(xattn.data(), cross_attn_weights_.data(),
            static_cast<std::size_t>(state.max_source_positions) * sizeof(float),
            cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

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
                      << ", text_len=" << text_length_ << ")" << std::endl;
        }
        return;
    }
    update_magpie_text_consumed_from_heuristic(
        state.estimated_frames, frame, state.text_consumed);
#else
    (void)state; (void)frame;
#endif
}

std::vector<int32_t> MagpiePipeline::run_gpu_greedy_loop(
    DecoderLoopState& state, int32_t max_frames)
{
#if TRTF_HAS_CUDA_KERNELS
    constexpr int32_t EOS_CHECK_INTERVAL = 16;
    constexpr int32_t MIN_FRAMES = 4;

    const int32_t num_cb = state.num_cb;

    CudaBuffer d_eos_flag(sizeof(int32_t));
    int32_t h_eos_flag = 0;
    cudaMemsetAsync(d_eos_flag.data(), 0, sizeof(int32_t), stream_);

    int32_t gen_frames_actual = 0;

    for (int32_t frame = 0; frame < max_frames; ++frame)
    {
        if (!gpu_greedy_frame_step(state, frame, d_eos_flag))
            break;

        gen_frames_actual = frame + 1;

        const bool periodic = should_run_magpie_periodic_check(
            frame, MIN_FRAMES, EOS_CHECK_INTERVAL);
        if (periodic)
        {
            cudaMemcpyAsync(&h_eos_flag, d_eos_flag.data(), sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream_);
            cudaStreamSynchronize(stream_);
            if (h_eos_flag != 0) break;
            gpu_greedy_update_text_consumed(state, frame);
        }
        if (check_finished_limit(state, frame))
        {
            gen_frames_actual = frame + 1;
            break;
        }
    }

    cudaStreamSynchronize(stream_);
    const std::size_t total_codes_bytes = static_cast<std::size_t>(gen_frames_actual) *
        num_cb * sizeof(int32_t);
    std::vector<int32_t> all_codes(static_cast<std::size_t>(gen_frames_actual) * num_cb);
    cudaMemcpy(all_codes.data(), device_all_codes_.data(), total_codes_bytes,
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

std::vector<int32_t> MagpiePipeline::run_cpu_sampling_loop(
    DecoderLoopState& state, int32_t max_frames)
{
    constexpr int32_t MIN_FRAMES = 4;
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
        if (state.use_cfg || frame == 0)
        {
            decoder_cache_->bind_to(*decoder_);
            bind_cross_kv();
        }
        run_decoder_step(state.embed_buf.data(), state.hidden, state.logits);

        // CFG: unconditional pass + blend
        if (state.use_cfg && !run_cfg_uncond_pass_cpu(state, frame)) break;
        const auto t_step_end = SteadyClock::now();
        state.prof_trt_step_ms += elapsed_ms(t_step_start, t_step_end);

        // Sample frame codes
        const auto t_sample_start = SteadyClock::now();
        const auto decoded = decode_magpie_frame_codes(
            state.logits, state.num_cb, state.cb_size,
            config_.greedy, config_.temperature, config_.top_k,
            [this](const float* cb_logits, int32_t vocab_size, float temperature, int32_t top_k)
            {
                return sample_top_k(cb_logits, vocab_size, temperature, top_k);
            });
        const auto t_sample_end = SteadyClock::now();
        state.prof_sample_ms += elapsed_ms(t_sample_start, t_sample_end);

        if (should_stop_magpie_on_eos(decoded.eos, frame, MIN_FRAMES))
        {
            std::cerr << "[magpie-tts] EOS detected at frame " << frame
                      << ", dropping terminal frame" << std::endl;
            break;
        }

        for (int32_t cb = 0; cb < num_cb; ++cb)
            all_codes.push_back(decoded.frame_codes[cb]);
        prev_codes = decoded.frame_codes;
        upload_magpie_prev_codes_to_device(device_prev_codes_, prev_codes.data(), num_cb,
                                            state.use_gpu_kernels, state.use_gpu_greedy);

        update_text_completion(state, frame);
        if (check_finished_limit(state, frame)) break;
    }

    return all_codes;
}

// ---------------------------------------------------------------------------
// run_decoder() -- orchestrator
// ---------------------------------------------------------------------------

std::vector<int32_t> MagpiePipeline::run_decoder(int32_t max_frames)
{
    DecoderLoopState state = init_decoder_state();

    // Reset KV caches
    decoder_cache_->reset();
    if (state.use_cfg && decoder_cache_uncond_)
        decoder_cache_uncond_->reset();

    // Bind cross-attention K/V
    decoder_cache_->bind_to(*decoder_);
    bind_cross_kv();

    // Phase 1: Context prefill
    const int32_t ctx_frames = prefill_context(state);
    if (ctx_frames < 0) return {};

    // Phase 2: Autoregressive decode
    std::vector<int32_t> bos(static_cast<std::size_t>(state.num_cb), kMagpieBosToken);
    upload_magpie_prev_codes_to_device(device_prev_codes_, bos.data(), state.num_cb,
                                        state.use_gpu_kernels, false);

    std::vector<int32_t> all_codes =
        (state.use_gpu_greedy && device_all_codes_.ok())
        ? run_gpu_greedy_loop(state, max_frames)
        : run_cpu_sampling_loop(state, max_frames);

    const int32_t gen_frames = static_cast<int32_t>(all_codes.size()) /
        std::max(state.num_cb, 1);
    std::cerr << "[magpie-tts] Generated " << gen_frames
              << " frames (" << all_codes.size() << " codes)" << std::endl;

    log_decoder_profiling(state, ctx_frames, gen_frames);
    log_magpie_frame_preview(all_codes, state.num_cb);
    return all_codes;
}

// ---------------------------------------------------------------------------
// run_codec() -- codes -> waveform via TrtModule
// ---------------------------------------------------------------------------

std::vector<float> MagpiePipeline::run_codec(
    const std::vector<int32_t>& codes, int32_t num_frames)
{
    const int32_t num_cb = config_.num_codebooks;
    if (num_frames <= 0) return {};

    if (!codec_ || !codec_->ok())
    {
        std::cerr << "[magpie-tts] No codec engine, generating silence" << std::endl;
        const int32_t samples_per_frame = config_.sample_rate /
            std::max(static_cast<int32_t>(config_.frames_per_second), 1);
        const auto total = static_cast<std::size_t>(num_frames) * samples_per_frame;
        return std::vector<float>(total, 0.0F);
    }

    // Build codec input using the plan helper
    // We need to figure out max_codec_frames from the codec engine's codec_tokens shape
    int32_t max_codec_frames = num_frames;
    // Get codec_tokens shape from engine output info
    auto codec_inputs = codec_->input_info();
    for (const auto& ti : codec_inputs)
    {
        if (ti.name == "codec_tokens" && ti.shape.size() >= 2)
        {
            max_codec_frames = static_cast<int32_t>(ti.shape[1]);
            break;
        }
    }

    const auto plan = make_magpie_codec_plan(num_frames, num_cb, max_codec_frames);
    std::vector<int32_t> codec_input = build_magpie_codec_input(codes, num_cb, plan);

    Tensor codec_tokens_tensor;
    codec_tokens_tensor.data = codec_input.data();
    codec_tokens_tensor.shape = {static_cast<int64_t>(num_cb),
                                  static_cast<int64_t>(max_codec_frames)};
    codec_tokens_tensor.dtype = DType::kInt32;

    Tensor input_len_tensor;
    int32_t input_len = plan.input_len;
    input_len_tensor.data = &input_len;
    input_len_tensor.shape = {1};
    input_len_tensor.dtype = DType::kInt32;

    TensorMap inputs;
    inputs["codec_tokens"] = codec_tokens_tensor;
    inputs["input_len"] = input_len_tensor;

    TensorMap outputs = codec_->forward(inputs);

    auto it = outputs.find("waveform");
    if (it == outputs.end())
    {
        std::cerr << "[magpie-tts] Codec: no 'waveform' output" << std::endl;
        return {};
    }

    const auto& wt = it->second;
    const auto total_out = wt.numel();
    const auto trimmed = plan.valid_samples;
    const auto copy_n = std::min(static_cast<std::size_t>(total_out), trimmed);

    std::vector<float> waveform(copy_n);
    std::memcpy(waveform.data(), wt.data, copy_n * sizeof(float));

    std::cerr << "[magpie-tts] Codec: " << num_frames << " frames -> "
              << waveform.size() << " samples" << std::endl;
    return waveform;
}

// ---------------------------------------------------------------------------
// Profiling / logging
// ---------------------------------------------------------------------------

void MagpiePipeline::log_decoder_profiling(const DecoderLoopState& state,
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
              << "[magpie-tts]   Text tracking:     "
              << (state.use_cross_attn_tracking ? "cross-attn tracking" : "heuristic (text_len*3)")
              << "\n"
              << "[magpie-tts]   Stop guards:       "
              << (config_.enable_finished_limit_stop
                    ? "finished_limit"
                    : "none (EOS/max_frames only)")
              << "\n"
              << "[magpie-tts] ---------------------------------\n";
}

void MagpiePipeline::log_pipeline_profiling(
    int32_t num_frames, int32_t num_samples,
    double ms_encoder, double ms_decoder,
    double ms_codec, double ms_total) const
{
    const double ms_per_frame = (num_frames > 0) ? ms_decoder / num_frames : 0.0;
    const double audio_duration = static_cast<double>(num_samples) / config_.sample_rate;
    const double rtf = (audio_duration > 0.0) ? (ms_total / 1000.0) / audio_duration : 0.0;

    std::cerr << "\n[magpie-tts] ===== PROFILING REPORT =====\n"
              << "[magpie-tts]   Encoder:        " << ms_encoder << " ms\n"
              << "[magpie-tts]   Cross-KV:       D2D copies (per-layer buffers)\n"
              << "[magpie-tts]   Decoder:        " << ms_decoder << " ms ("
              << num_frames << " frames, " << ms_per_frame << " ms/frame)\n"
              << "[magpie-tts]   Codec:          " << ms_codec << " ms\n"
              << "[magpie-tts]   Total pipeline: " << ms_total << " ms\n"
              << "[magpie-tts]   Audio duration: " << audio_duration << " s ("
              << num_samples << " samples @ " << config_.sample_rate << " Hz)\n"
              << "[magpie-tts]   RTF (real-time factor): " << rtf
              << " (< 1.0 = faster than real-time)\n"
              << "[magpie-tts]   CFG scale:      " << config_.cfg_scale
              << (config_.cfg_scale > 1.0F ? " (enabled, 2x decoder steps)" : " (disabled)")
              << "\n"
              << "[magpie-tts]   finished_limit: "
              << (config_.enable_finished_limit_stop
                    ? std::to_string(config_.finished_limit_with_eot)
                    : std::string("disabled"))
              << " (text_len=" << text_length_ << ", est_frames="
              << static_cast<int32_t>(static_cast<float>(text_length_) * 3.0F) << ")\n"
              << "[magpie-tts] =============================\n" << std::endl;
}

// ---------------------------------------------------------------------------
// generate_audio() helpers
// ---------------------------------------------------------------------------

void MagpiePipeline::apply_env_overrides()
{
    maybe_enable_magpie_greedy(config_);

    const char* env_cfg = std::getenv("TRTF_MAGPIE_CFG_SCALE");
    if (env_cfg != nullptr)
    {
        float val = std::atof(env_cfg);
        if (val > 0.0F) config_.cfg_scale = val;
    }
    const char* env_temp = std::getenv("TRTF_MAGPIE_TEMPERATURE");
    if (env_temp != nullptr)
    {
        float val = std::atof(env_temp);
        if (val > 0.0F) config_.temperature = val;
    }
    const char* env_limit = std::getenv("TRTF_MAGPIE_FINISHED_LIMIT");
    if (env_limit != nullptr)
    {
        int32_t val = std::atoi(env_limit);
        if (val >= 0)
        {
            config_.finished_limit_with_eot = val;
            config_.enable_finished_limit_stop = (val > 0);
        }
    }
    const char* env_seed = std::getenv("TRTF_MAGPIE_SEED");
    if (env_seed != nullptr)
    {
        rng_.seed(static_cast<std::mt19937::result_type>(std::atol(env_seed)));
    }
}

void MagpiePipeline::ensure_cfg_resources()
{
    if (config_.cfg_scale <= 1.0F || decoder_cache_uncond_)
        return;

    // Lazy-allocate CFG resources if cfg_scale was bumped via env var
    const int32_t dec_layers = static_cast<int32_t>(cross_k_.size());
    const int32_t kv_dim = decoder_cache_->max_length() > 0
        ? static_cast<int32_t>(cross_k_[0].size() / (static_cast<std::size_t>(config_.max_source_positions) * sizeof(float)))
        : config_.hidden_size;

    // We can't easily construct a new KvCache without knowing kv_dim,
    // so just log a warning and skip CFG.
    std::cerr << "[magpie-tts] WARNING: CFG requested via env but no uncond cache allocated. "
              << "Rebuild with cfg_scale > 1 in bundle config." << std::endl;
    config_.cfg_scale = 1.0F;
    (void)dec_layers; (void)kv_dim;
}

void MagpiePipeline::run_cfg_encoder(const std::vector<int32_t>& text_ids)
{
    if (config_.cfg_scale <= 1.0F || !encoder_output_uncond_.ok() || cross_k_uncond_.empty())
        return;

    std::cerr << "[magpie-tts] CFG: encoding null text for unconditional path ..."
              << std::endl;

    const auto enc_bytes = encoder_output_.size();

    // Encode empty text
    std::vector<int32_t> empty_ids;
    run_encoder(empty_ids);

    // Save unconditional encoder output
    cudaMemcpy(encoder_output_uncond_.data(), encoder_output_.data(),
               enc_bytes, cudaMemcpyDeviceToDevice);

    // Re-encode actual text
    run_encoder(text_ids);

    compute_cross_kv_uncond();
}

// ---------------------------------------------------------------------------
// generate_audio() -- full pipeline orchestration
// ---------------------------------------------------------------------------

AudioResult MagpiePipeline::generate_audio(
    const std::string& prompt, const GenerateConfig& cfg)
{
    std::vector<int32_t> input_ids;
    if (tokenizer_)
        input_ids = tokenizer_->encode(prompt);

    int32_t max_frames = cfg.max_new_tokens > 0 ? cfg.max_new_tokens : 512;

    AudioResult result;
    result.sample_rate = config_.sample_rate;

    apply_env_overrides();
    ensure_cfg_resources();

    text_length_ = static_cast<int32_t>(input_ids.size());

    std::cerr << "[magpie-tts] Starting pipeline with " << input_ids.size()
              << " text tokens, max_frames=" << max_frames
              << (config_.greedy ? " (greedy)" : "")
              << ", cfg_scale=" << config_.cfg_scale
              << ", finished_limit="
              << (config_.enable_finished_limit_stop
                    ? std::to_string(config_.finished_limit_with_eot)
                    : std::string("disabled"))
              << std::endl;

    const auto t_pipeline_start = SteadyClock::now();

    // Stage 1: Encode text
    std::cerr << "[magpie-tts] Running encoder ..." << std::endl;
    const auto t_enc_start = SteadyClock::now();
    run_encoder(input_ids);
    const auto t_enc_end = SteadyClock::now();

    // Stage 2: Copy encoder output to per-layer cross-attention buffers
    compute_cross_kv();

    // Stage 2b (CFG): Run encoder with empty text for unconditional cross-KV
    run_cfg_encoder(input_ids);

    // Stage 3: Autoregressive decode
    std::cerr << "[magpie-tts] Running decoder ..." << std::endl;
    const auto t_dec_start = SteadyClock::now();
    auto codes = run_decoder(max_frames);
    const auto t_dec_end = SteadyClock::now();
    if (codes.empty())
    {
        std::cerr << "[magpie-tts] Decoder produced no codes" << std::endl;
        return result;
    }

    const int32_t num_frames = static_cast<int32_t>(codes.size()) / config_.num_codebooks;

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

    result.samples = std::move(waveform);
    result.num_samples = static_cast<int32_t>(result.samples.size());

    const auto t_pipeline_end = SteadyClock::now();

    log_pipeline_profiling(
        num_frames, result.num_samples,
        elapsed_ms(t_enc_start, t_enc_end),
        elapsed_ms(t_dec_start, t_dec_end),
        elapsed_ms(t_codec_start, t_codec_end),
        elapsed_ms(t_pipeline_start, t_pipeline_end));

    return result;
}

} // namespace trtf

#endif // TRTF_HAS_TRT
