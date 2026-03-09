#include "runtime/trt/audio/bark_backend.h"
#include "runtime/trt/audio/bark_generation_plan.h"

#if TRTF_HAS_TRT

#include "runtime/trt/core/trt_decode_runtime.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace trtf {

namespace {

constexpr int32_t kBarkMaxTextLen = 256;
constexpr float kNegInf = -1e9F;

void copy_embed_row(const float* table, int32_t token_id, int32_t hidden_size, float* out)
{
    const auto offset = static_cast<std::size_t>(token_id) *
        static_cast<std::size_t>(hidden_size);
    std::memcpy(out, table + offset, static_cast<std::size_t>(hidden_size) * sizeof(float));
}

void sum_embed_rows(const float* a, const float* b, int32_t hidden_size, float* out)
{
    for (int32_t i = 0; i < hidden_size; ++i)
    {
        out[i] = a[i] + b[i];
    }
}

int32_t semantic_text_token(
    const std::vector<int32_t>& text_ids,
    int32_t pos,
    int32_t copy_len,
    const BarkConfig& cfg)
{
    if (pos < copy_len && text_ids[pos] != 0)
    {
        return text_ids[pos] + cfg.text_encoding_offset;
    }
    return cfg.text_pad_token;
}

bool semantic_eos_threshold_hit(const std::vector<float>& logits, const BarkConfig& cfg)
{
    if (cfg.min_eos_p <= 0.0F)
    {
        return false;
    }

    float max_val = *std::max_element(
        logits.begin(), logits.begin() + cfg.semantic_pad_token + 1);
    float sum_exp = 0.0F;
    for (int32_t i = 0; i <= cfg.semantic_pad_token; ++i)
    {
        sum_exp += std::exp(logits[i] - max_val);
    }
    const float eos_p = std::exp(logits[cfg.semantic_pad_token] - max_val) /
        std::max(sum_exp, 1e-10F);
    return eos_p > cfg.min_eos_p;
}

void suppress_semantic_logits(std::vector<float>& logits, int32_t semantic_pad_token)
{
    for (int32_t i = semantic_pad_token + 1; i < static_cast<int32_t>(logits.size()); ++i)
    {
        logits[i] = kNegInf;
    }
}

void maybe_dump_tokens(const char* suffix, const std::vector<int32_t>& tokens)
{
    const char* dump_path = std::getenv("TRTF_BARK_DUMP");
    if (dump_path == nullptr)
    {
        return;
    }

    std::ofstream dump(std::string(dump_path) + suffix);
    for (int32_t token : tokens)
    {
        dump << token << "\n";
    }
}

void mask_coarse_logits_for_codebook(
    std::vector<float>& logits,
    int32_t codebook_idx,
    const BarkConfig& cfg)
{
    const int32_t cb_start = cfg.semantic_vocab_size + codebook_idx * cfg.codebook_size;
    const int32_t cb_end = cb_start + cfg.codebook_size;
    for (int32_t i = 0; i < static_cast<int32_t>(logits.size()); ++i)
    {
        if (i < cb_start || i >= cb_end)
        {
            logits[i] = kNegInf;
        }
    }
}

void prefill_coarse_window(
    DecoderStepEngine& coarse_engine,
    DeviceKvCache& cache,
    DeviceResources& resources,
    const std::vector<int32_t>& input_tokens,
    const float* coarse_embed,
    const BarkConfig& cfg,
    std::vector<float>& embed_buf,
    std::vector<float>& logits)
{
    std::string error;
    for (std::size_t i = 0; i + 1 < input_tokens.size(); ++i)
    {
        copy_embed_row(coarse_embed, input_tokens[i], cfg.hidden_size, embed_buf.data());
        if (!run_decoder_step_device(
                coarse_engine,
                cache,
                resources,
                0,
                logits,
                error,
                embed_buf.data(),
                cfg.hidden_size,
                1.0F))
        {
            throw std::runtime_error("Bark coarse prefill failed: " + error);
        }
    }

    copy_embed_row(coarse_embed, input_tokens.back(), cfg.hidden_size, embed_buf.data());
    if (!run_decoder_step_device(
            coarse_engine,
            cache,
            resources,
            0,
            logits,
            error,
            embed_buf.data(),
            cfg.hidden_size,
            1.0F))
    {
        throw std::runtime_error("Bark coarse last prefill failed: " + error);
    }
}

void prefill_semantic_context(
    DecoderStepEngine& semantic_engine,
    DeviceKvCache& cache,
    DeviceResources& resources,
    const std::vector<int32_t>& text_ids,
    const BarkConfig& cfg,
    const std::vector<float>& semantic_embed,
    std::vector<float>& embed_a,
    std::vector<float>& embed_b,
    std::vector<float>& embed_buf,
    std::vector<float>& logits)
{
    std::string error;
    const auto copy_len = std::min(static_cast<int32_t>(text_ids.size()), kBarkMaxTextLen);
    for (int32_t pos = 0; pos < kBarkMaxTextLen; ++pos)
    {
        const int32_t text_tok = semantic_text_token(text_ids, pos, copy_len, cfg);
        copy_embed_row(semantic_embed.data(), text_tok, cfg.hidden_size, embed_a.data());
        copy_embed_row(
            semantic_embed.data(), cfg.semantic_pad_token, cfg.hidden_size, embed_b.data());
        sum_embed_rows(embed_a.data(), embed_b.data(), cfg.hidden_size, embed_buf.data());

        if (!run_decoder_step_device(semantic_engine, cache, resources,
                0, logits, error,
                embed_buf.data(), cfg.hidden_size, 1.0F))
        {
            throw std::runtime_error("Bark semantic prefill failed: " + error);
        }
    }

    copy_embed_row(
        semantic_embed.data(), cfg.semantic_infer_token, cfg.hidden_size, embed_buf.data());
    if (!run_decoder_step_device(semantic_engine, cache, resources,
            0, logits, error,
            embed_buf.data(), cfg.hidden_size, 1.0F))
    {
        throw std::runtime_error("Bark semantic infer token failed: " + error);
    }
}

template <typename SampleFn>
bool sample_semantic_token_for_step(
    std::vector<float>& logits,
    const BarkConfig& cfg,
    SampleFn&& sample_fn,
    int32_t& token)
{
    if (semantic_eos_threshold_hit(logits, cfg))
    {
        return false;
    }
    suppress_semantic_logits(logits, cfg.semantic_pad_token);
    token = sample_fn(
        logits.data(),
        cfg.semantic_pad_token + 1,
        cfg.semantic_temperature,
        cfg.top_k);
    return token != cfg.semantic_pad_token;
}

void build_fine_input_embeddings(
    std::vector<float>& host_embeds,
    const std::vector<int32_t>& codes,
    int32_t cb_idx,
    int32_t n_frames,
    int32_t actual_frames,
    int32_t fine_hidden,
    int32_t fine_cb_size,
    const std::vector<float>& fine_embed,
    const std::vector<float>& fine_position_embed)
{
    std::fill(host_embeds.begin(), host_embeds.end(), 0.0F);
    for (int32_t frame = 0; frame < actual_frames; ++frame)
    {
        float* dst = host_embeds.data() + static_cast<std::size_t>(frame) * fine_hidden;
        for (int32_t cb = 0; cb <= cb_idx; ++cb)
        {
            const int32_t code = codes[static_cast<std::size_t>(cb) * n_frames + frame];
            const float* table = fine_embed.data() +
                static_cast<std::size_t>(cb) * fine_cb_size * fine_hidden;
            const float* row = table + static_cast<std::size_t>(code) * fine_hidden;
            for (int32_t h = 0; h < fine_hidden; ++h)
            {
                dst[h] += row[h];
            }
        }

        const float* pos_row = fine_position_embed.data() +
            static_cast<std::size_t>(frame) * fine_hidden;
        for (int32_t h = 0; h < fine_hidden; ++h)
        {
            dst[h] += pos_row[h];
        }
    }
}

bool bind_fine_outputs(
    nvinfer1::IExecutionContext& fine_ctx,
    const std::vector<CudaBuffer>& d_outputs,
    int32_t fine_n_lm_heads)
{
    for (int32_t head = 0; head < fine_n_lm_heads; ++head)
    {
        const std::string name = "logits_cb" + std::to_string(head + 1);
        if (!fine_ctx.setTensorAddress(name.c_str(), d_outputs[head].data()))
        {
            std::cerr << "[trtf] Bark fine: failed to bind " << name << std::endl;
            return false;
        }
    }
    return true;
}

std::vector<CudaBuffer> allocate_fine_output_buffers(
    int32_t fine_n_lm_heads,
    int32_t max_seq,
    int32_t fine_cb_size)
{
    std::vector<CudaBuffer> d_outputs;
    d_outputs.reserve(static_cast<std::size_t>(fine_n_lm_heads));
    for (int32_t i = 0; i < fine_n_lm_heads; ++i)
    {
        d_outputs.emplace_back(static_cast<std::size_t>(max_seq) * fine_cb_size * sizeof(float));
    }
    return d_outputs;
}

void update_fine_codes_from_logits(
    std::vector<int32_t>& codes,
    const std::vector<float>& host_logits,
    int32_t cb_idx,
    int32_t n_frames,
    int32_t actual_frames,
    int32_t fine_cb_size,
    int32_t codebook_size);

bool run_fine_codebook_step(
    nvinfer1::IExecutionContext& fine_ctx,
    CudaBuffer& d_input,
    const std::vector<CudaBuffer>& d_outputs,
    CudaStream& stream,
    const BarkConfig& cfg,
    int32_t cb_idx,
    int32_t n_frames,
    int32_t actual_frames,
    int32_t max_seq,
    int32_t fine_hidden,
    int32_t fine_cb_size,
    const std::vector<float>& fine_embed,
    const std::vector<float>& fine_position_embed,
    std::vector<float>& host_embeds,
    std::vector<float>& host_logits,
    std::vector<int32_t>& codes)
{
    build_fine_input_embeddings(
        host_embeds,
        codes,
        cb_idx,
        n_frames,
        actual_frames,
        fine_hidden,
        fine_cb_size,
        fine_embed,
        fine_position_embed);
    cudaMemcpyAsync(
        d_input.data(),
        host_embeds.data(),
        static_cast<std::size_t>(max_seq) * fine_hidden * sizeof(float),
        cudaMemcpyHostToDevice,
        stream.get());

    if (!fine_ctx.setTensorAddress("input_embeds", d_input.data()))
    {
        std::cerr << "[trtf] Bark fine: failed to bind input_embeds" << std::endl;
        return false;
    }
    if (!bind_fine_outputs(fine_ctx, d_outputs, cfg.fine_n_lm_heads))
    {
        return false;
    }
    if (!fine_ctx.enqueueV3(stream.get()))
    {
        std::cerr << "[trtf] Bark fine: TRT execution failed" << std::endl;
        return false;
    }

    const int32_t head_idx = cb_idx - 1;
    cudaMemcpyAsync(
        host_logits.data(),
        d_outputs[static_cast<std::size_t>(head_idx)].data(),
        static_cast<std::size_t>(max_seq) * fine_cb_size * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream.get());
    cudaStreamSynchronize(stream.get());
    update_fine_codes_from_logits(
        codes,
        host_logits,
        cb_idx,
        n_frames,
        actual_frames,
        fine_cb_size,
        cfg.codebook_size);
    return true;
}

void update_fine_codes_from_logits(
    std::vector<int32_t>& codes,
    const std::vector<float>& host_logits,
    int32_t cb_idx,
    int32_t n_frames,
    int32_t actual_frames,
    int32_t fine_cb_size,
    int32_t codebook_size)
{
    const int32_t valid_range = std::min(codebook_size, fine_cb_size);
    for (int32_t frame = 0; frame < actual_frames; ++frame)
    {
        const float* frame_logits = host_logits.data() +
            static_cast<std::size_t>(frame) * fine_cb_size;
        int32_t best = 0;
        for (int32_t i = 1; i < valid_range; ++i)
        {
            if (frame_logits[i] > frame_logits[best])
            {
                best = i;
            }
        }
        codes[static_cast<std::size_t>(cb_idx) * n_frames + frame] = best;
    }
}

std::vector<float> synthesize_simple_waveform(
    const std::vector<int32_t>& codes_flat,
    int32_t n_frames,
    const BarkConfig& cfg)
{
    const int32_t samples_per_frame = cfg.sample_rate / cfg.coarse_rate_hz;
    const int32_t total_samples = n_frames * samples_per_frame;
    std::vector<float> waveform(static_cast<std::size_t>(total_samples), 0.0F);

    for (int32_t f = 0; f < n_frames; ++f)
    {
        const float freq = 200.0F +
            static_cast<float>(codes_flat[f]) * 800.0F /
            static_cast<float>(cfg.codebook_size);
        const float amp = 0.3F;
        for (int32_t s = 0; s < samples_per_frame; ++s)
        {
            const auto idx = static_cast<std::size_t>(f) * samples_per_frame + s;
            const float t = static_cast<float>(s) / static_cast<float>(cfg.sample_rate);
            waveform[idx] = amp * std::sin(2.0F * 3.14159265F * freq * t);
        }
    }
    return waveform;
}

bool bind_codec_tensors(
    nvinfer1::IExecutionContext& codec_ctx,
    CudaBuffer& d_input,
    CudaBuffer& d_output)
{
    if (!codec_ctx.setTensorAddress("audio_codes", d_input.data()))
    {
        std::cerr << "[trtf] Bark codec: failed to bind audio_codes" << std::endl;
        return false;
    }
    if (!codec_ctx.setTensorAddress("waveform", d_output.data()))
    {
        std::cerr << "[trtf] Bark codec: failed to bind waveform" << std::endl;
        return false;
    }
    return true;
}

std::vector<float> run_codec_trt(
    nvinfer1::IExecutionContext& codec_ctx,
    const BarkConfig& cfg,
    const std::vector<int32_t>& codes_flat,
    int32_t n_frames,
    int32_t source_codebooks,
    const char* suffix)
{
    const int32_t n_cb = cfg.codec_n_codebooks;
    const int32_t max_T = cfg.codec_seq_length;
    const int32_t upsample = cfg.codec_upsample_factor;

    if (n_frames > max_T)
    {
        std::cerr << "[trtf] Bark codec: n_frames=" << n_frames
                  << " exceeds codec_seq_length=" << max_T
                  << ", truncating" << std::endl;
    }
    const int32_t actual_frames = std::min(n_frames, max_T);
    std::vector<int32_t> input_codes = make_bark_codec_input_codes(
        codes_flat, source_codebooks, n_frames, n_cb, max_T, actual_frames);

    const auto input_bytes = input_codes.size() * sizeof(int32_t);
    const auto output_elems = static_cast<std::size_t>(max_T) * upsample;
    const auto output_bytes = output_elems * sizeof(float);
    CudaBuffer d_input(input_bytes);
    CudaBuffer d_output(output_bytes);
    CudaStream stream;
    if (!d_input.ok() || !d_output.ok() || !stream.ok())
    {
        std::cerr << "[trtf] Bark codec: failed to allocate CUDA resources" << std::endl;
        return {};
    }

    cudaMemcpyAsync(
        d_input.data(), input_codes.data(), input_bytes, cudaMemcpyHostToDevice, stream.get());
    if (!bind_codec_tensors(codec_ctx, d_input, d_output))
    {
        return {};
    }
    if (!codec_ctx.enqueueV3(stream.get()))
    {
        std::cerr << "[trtf] Bark codec: TRT execution failed" << std::endl;
        return {};
    }

    std::vector<float> full_waveform(output_elems);
    cudaMemcpyAsync(
        full_waveform.data(), d_output.data(), output_bytes, cudaMemcpyDeviceToHost, stream.get());
    cudaStreamSynchronize(stream.get());

    const auto trimmed_samples = static_cast<std::size_t>(actual_frames) * upsample;
    std::vector<float> waveform(
        full_waveform.begin(),
        full_waveform.begin() + static_cast<std::ptrdiff_t>(trimmed_samples));

    std::cerr << "[trtf] Bark codec: TRT decode " << actual_frames << " frames -> "
              << waveform.size() << " samples" << suffix << std::endl;
    return waveform;
}

void maybe_enable_bark_greedy(BarkConfig& cfg)
{
    const char* env = std::getenv("TRTF_BARK_GREEDY");
    if (env != nullptr && std::string(env) == "1")
    {
        cfg.greedy = true;
    }
}

void maybe_seed_bark_rng(std::mt19937& rng)
{
    const char* env = std::getenv("TRTF_BARK_SEED");
    if (env == nullptr || *env == '\0')
    {
        return;
    }

    errno = 0;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(env, &end, 10);
    if (errno == 0 && end != env && *end == '\0')
    {
        const auto seed = static_cast<std::mt19937::result_type>(parsed);
        rng.seed(seed);
        std::cerr << "[trtf] Bark: sampler seed=" << seed << std::endl;
        return;
    }

    std::cerr << "[trtf] Bark: ignoring invalid TRTF_BARK_SEED='"
              << env << "'" << std::endl;
}

} // namespace

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

BarkBackend::BarkBackend(
    std::unique_ptr<DecoderStepEngine> semantic_engine,
    std::unique_ptr<DecoderStepEngine> coarse_engine,
    std::vector<float> semantic_embed,
    std::vector<float> coarse_embed,
    BarkConfig config)
    : mSemanticEngine(std::move(semantic_engine))
    , mCoarseEngine(std::move(coarse_engine))
    , mSemanticEmbed(std::move(semantic_embed))
    , mCoarseEmbed(std::move(coarse_embed))
    , mConfig(std::move(config))
{
}

BarkBackend::~BarkBackend() = default;

bool BarkBackend::is_available() const
{
    return mSemanticEngine != nullptr && mCoarseEngine != nullptr
        && !mSemanticEmbed.empty() && !mCoarseEmbed.empty();
}

void BarkBackend::set_codec_engine(
    TrtUniquePtr<nvinfer1::ICudaEngine> engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> context)
{
    mCodecEngine = std::move(engine);
    mCodecCtx = std::move(context);
}

void BarkBackend::set_fine_engine(
    TrtUniquePtr<nvinfer1::ICudaEngine> engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> context)
{
    mFineEngine = std::move(engine);
    mFineCtx = std::move(context);
}

void BarkBackend::set_fine_embeddings(
    std::vector<float> embed, std::vector<float> pos_embed)
{
    mFineEmbed = std::move(embed);
    mFinePositionEmbed = std::move(pos_embed);
}

// ---------------------------------------------------------------------------
// Embedding helpers
// ---------------------------------------------------------------------------

void BarkBackend::lookup_embed(const float* table, int32_t token_id,
                                float* out) const
{
    const auto offset = static_cast<std::size_t>(token_id) *
                        static_cast<std::size_t>(mConfig.hidden_size);
    std::memcpy(out, table + offset,
                static_cast<std::size_t>(mConfig.hidden_size) * sizeof(float));
}

void BarkBackend::sum_embeds(const float* a, const float* b, float* out) const
{
    for (int32_t i = 0; i < mConfig.hidden_size; ++i)
    {
        out[i] = a[i] + b[i];
    }
}

// ---------------------------------------------------------------------------
// Top-k sampling with temperature
// ---------------------------------------------------------------------------

int32_t BarkBackend::sample_top_k(const float* logits, int32_t vocab_size,
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
// Stage 1: Semantic (text tokens -> semantic audio tokens)
// ---------------------------------------------------------------------------

std::vector<int32_t> BarkBackend::run_semantic(
    const std::vector<int32_t>& text_ids,
    int32_t max_tokens)
{
    const auto& cfg = mConfig;

    DeviceKvCache cache(*mSemanticEngine);
    DeviceResources resources(*mSemanticEngine);
    if (!cache.ok() || !resources.ok())
    {
        throw std::runtime_error("Bark semantic: failed to allocate device resources");
    }

    std::vector<float> logits;
    std::vector<float> embed_buf(static_cast<std::size_t>(cfg.hidden_size));
    std::vector<float> embed_a(static_cast<std::size_t>(cfg.hidden_size));
    std::vector<float> embed_b(static_cast<std::size_t>(cfg.hidden_size));

    prefill_semantic_context(
        *mSemanticEngine,
        cache,
        resources,
        text_ids,
        cfg,
        mSemanticEmbed,
        embed_a,
        embed_b,
        embed_buf,
        logits);

    std::vector<int32_t> semantic_tokens;
    semantic_tokens.reserve(static_cast<std::size_t>(max_tokens));

    auto sampler = [this](const float* values, int32_t vocab_size, float temp, int32_t top_k) {
        return sample_top_k(values, vocab_size, temp, top_k);
    };
    std::string error;
    for (int32_t step = 0; step < max_tokens; ++step)
    {
        int32_t token = 0;
        if (!sample_semantic_token_for_step(logits, cfg, sampler, token))
        {
            break;
        }

        semantic_tokens.push_back(token);
        if (!run_decoder_step_device(*mSemanticEngine, cache, resources,
                token, logits, error,
                nullptr, 0, 0.0F))
        {
            throw std::runtime_error("Bark semantic decode failed: " + error);
        }
    }

    std::cerr << "[trtf] Bark semantic: generated " << semantic_tokens.size()
              << " tokens" << std::endl;
    maybe_dump_tokens(".sem_tokens", semantic_tokens);
    return semantic_tokens;
}

// ---------------------------------------------------------------------------
// Stage 2: Coarse (semantic tokens -> coarse acoustic codes)
// ---------------------------------------------------------------------------

std::vector<int32_t> BarkBackend::run_coarse(
    const std::vector<int32_t>& semantic_tokens)
{
    const auto& cfg = mConfig;
    const BarkCoarsePlan coarse_plan = make_bark_coarse_plan(semantic_tokens, cfg);
    const int32_t n_steps = coarse_plan.total_steps;

    if (n_steps == 0)
    {
        std::cerr << "[trtf] Bark coarse: no steps to generate" << std::endl;
        return {};
    }

    std::vector<int32_t> x_coarse;  // accumulated coarse output tokens
    x_coarse.reserve(static_cast<std::size_t>(n_steps));

    std::vector<float> logits;
    std::vector<float> embed_buf(static_cast<std::size_t>(cfg.hidden_size));
    std::string error;

    for (int32_t win = 0; win < coarse_plan.num_windows; ++win)
    {
        const BarkCoarseWindowPlan window_plan = make_bark_coarse_window_plan(
            coarse_plan,
            x_coarse,
            cfg);
        if (window_plan.generated_this_window <= 0)
        {
            break;
        }

        DeviceKvCache cache(*mCoarseEngine);
        DeviceResources resources(*mCoarseEngine);
        if (!cache.ok() || !resources.ok())
        {
            throw std::runtime_error("Bark coarse: failed to allocate device resources");
        }
        prefill_coarse_window(
            *mCoarseEngine,
            cache,
            resources,
            window_plan.input_tokens,
            mCoarseEmbed.data(),
            cfg,
            embed_buf,
            logits);

        for (int32_t step = 0; step < window_plan.generated_this_window; ++step)
        {
            const int32_t total_generated = window_plan.start_generated_count + step;
            const int32_t codebook_idx = bark_coarse_codebook_index(total_generated, cfg);
            mask_coarse_logits_for_codebook(logits, codebook_idx, cfg);

            const int32_t token = sample_top_k(logits.data(),
                static_cast<int32_t>(logits.size()),
                cfg.coarse_temperature, cfg.top_k);
            x_coarse.push_back(token);

            if (step + 1 < window_plan.generated_this_window)
            {
                copy_embed_row(mCoarseEmbed.data(), token, cfg.hidden_size, embed_buf.data());
                if (!run_decoder_step_device(*mCoarseEngine, cache, resources,
                        0, logits, error,
                        embed_buf.data(), cfg.hidden_size, 1.0F))
                {
                    throw std::runtime_error("Bark coarse decode failed: " + error);
                }
            }
        }
    }

    std::cerr << "[trtf] Bark coarse: generated " << x_coarse.size()
              << " tokens" << std::endl;
    maybe_dump_tokens(".coarse_tokens", x_coarse);
    return x_coarse;
}

// ---------------------------------------------------------------------------
// Stage 2.5: Fine (coarse codes -> 8 codebook codes)
// ---------------------------------------------------------------------------

std::vector<int32_t> BarkBackend::run_fine(const std::vector<int32_t>& coarse_tokens)
{
    const auto& cfg = mConfig;
    const BarkFinePlan plan = make_bark_fine_plan(
        cfg,
        coarse_tokens.size(),
        static_cast<bool>(mFineEngine),
        static_cast<bool>(mFineCtx));
    std::vector<int32_t> codes = initialize_bark_fine_codes(
        coarse_tokens,
        plan.n_frames,
        cfg);

    if (!plan.should_run_trt)
    {
        std::cerr << "[trtf] Bark fine: no TRT fine engine, "
                  << "codebooks 2-7 will be zero" << std::endl;
        return codes;
    }

    const int32_t fine_hidden = cfg.fine_hidden_size;
    const int32_t fine_cb_size = cfg.fine_codebook_size;
    const int32_t max_seq = cfg.fine_seq_length;
    CudaBuffer d_input(static_cast<std::size_t>(max_seq) * fine_hidden * sizeof(float));
    std::vector<CudaBuffer> d_outputs = allocate_fine_output_buffers(
        cfg.fine_n_lm_heads, max_seq, fine_cb_size);
    CudaStream stream;

    if (!d_input.ok() || !stream.ok())
    {
        std::cerr << "[trtf] Bark fine: failed to allocate CUDA resources" << std::endl;
        return codes;
    }

    std::vector<float> host_embeds(
        static_cast<std::size_t>(max_seq) * fine_hidden, 0.0F);
    std::vector<float> host_logits(
        static_cast<std::size_t>(max_seq) * fine_cb_size);

    for (int32_t cb_idx = plan.first_predicted_codebook;
         cb_idx < plan.last_predicted_codebook;
         ++cb_idx)
    {
        if (!run_fine_codebook_step(
                *mFineCtx,
                d_input,
                d_outputs,
                stream,
                cfg,
                cb_idx,
                plan.n_frames,
                plan.actual_frames,
                max_seq,
                fine_hidden,
                fine_cb_size,
                mFineEmbed,
                mFinePositionEmbed,
                host_embeds,
                host_logits,
                codes))
        {
            return codes;
        }
    }

    std::cerr << "[trtf] Bark fine: predicted codebooks 2-7 for "
              << plan.n_frames << " frames" << std::endl;

    return codes;
}

// ---------------------------------------------------------------------------
// Stage 3: Codec (coarse codes -> waveform)
// ---------------------------------------------------------------------------

std::vector<float> BarkBackend::run_codec(const std::vector<int32_t>& coarse_tokens)
{
    const auto& cfg = mConfig;
    const int32_t n_frames = static_cast<int32_t>(coarse_tokens.size()) /
        cfg.n_coarse_codebooks;

    if (n_frames == 0)
    {
        return {};
    }

    std::vector<int32_t> codes(
        static_cast<std::size_t>(cfg.n_coarse_codebooks) * n_frames, 0);
    for (int32_t t = 0; t < n_frames * cfg.n_coarse_codebooks; ++t)
    {
        const int32_t cb = t % cfg.n_coarse_codebooks;
        const int32_t frame = t / cfg.n_coarse_codebooks;
        int32_t raw_code = coarse_tokens[t] - cfg.semantic_vocab_size -
                           cb * cfg.codebook_size;
        raw_code = std::max(0, std::min(raw_code, cfg.codebook_size - 1));
        codes[cb * n_frames + frame] = raw_code;
    }

    if (!mCodecEngine || !mCodecCtx || cfg.codec_seq_length <= 0)
    {
        std::cerr << "[trtf] Bark codec: no TRT codec engine, "
                  << "generating simple waveform from codes" << std::endl;
        return synthesize_simple_waveform(codes, n_frames, cfg);
    }
    return run_codec_trt(
        *mCodecCtx, cfg, codes, n_frames, cfg.n_coarse_codebooks, "");
}

// ---------------------------------------------------------------------------
// Stage 3 overload: pre-computed codes [8 * n_frames] -> waveform via codec
// ---------------------------------------------------------------------------

std::vector<float> BarkBackend::run_codec(const std::vector<int32_t>& codes_flat,
                                           int32_t n_frames)
{
    const auto& cfg = mConfig;

    if (n_frames <= 0)
    {
        return {};
    }

    // If no codec engine, fall back to simple synthesis from codebook 0
    if (!mCodecEngine || !mCodecCtx || cfg.codec_seq_length <= 0)
    {
        std::cerr << "[trtf] Bark codec: no TRT codec engine, "
                  << "generating simple waveform from codes" << std::endl;
        return synthesize_simple_waveform(codes_flat, n_frames, cfg);
    }
    return run_codec_trt(
        *mCodecCtx, cfg, codes_flat, n_frames, 8, " (from fine codes)");
}

// ---------------------------------------------------------------------------
// Full pipeline: generate_audio
// ---------------------------------------------------------------------------

LegacyAudioResult BarkBackend::generate_audio(
    const std::vector<int32_t>& input_ids,
    int32_t max_semantic_tokens)
{
    LegacyAudioResult result;
    result.sample_rate = mConfig.sample_rate;

    if (!is_available())
    {
        std::cerr << "[trtf] Bark: backend not fully initialized" << std::endl;
        return result;
    }

    maybe_enable_bark_greedy(mConfig);
    maybe_seed_bark_rng(mRng);

    std::cerr << "[trtf] Bark: starting pipeline with " << input_ids.size()
              << " text tokens, max_semantic=" << max_semantic_tokens
              << (mConfig.greedy ? " (greedy)" : "") << std::endl;

    // Stage 1: Text -> Semantic tokens
    auto semantic_tokens = run_semantic(input_ids, max_semantic_tokens);
    if (semantic_tokens.empty())
    {
        std::cerr << "[trtf] Bark: semantic stage produced no tokens" << std::endl;
        return result;
    }

    // Stage 2: Semantic -> Coarse acoustic codes
    auto coarse_tokens = run_coarse(semantic_tokens);
    if (coarse_tokens.empty())
    {
        std::cerr << "[trtf] Bark: coarse stage produced no tokens" << std::endl;
        return result;
    }

    // Stage 2.5: Fine (coarse codes -> 8 codebook codes)
    auto fine_codes = run_fine(coarse_tokens);
    const BarkCodecPlan codec_plan = make_bark_codec_plan(
        fine_codes,
        static_cast<bool>(mFineEngine),
        coarse_tokens,
        mConfig.n_coarse_codebooks);

    std::vector<float> waveform = codec_plan.use_fine_codes
        ? run_codec(fine_codes, codec_plan.frame_count)
        : run_codec(coarse_tokens);
    if (waveform.empty())
    {
        std::cerr << "[trtf] Bark: codec produced no audio" << std::endl;
        return result;
    }

    result.waveform = std::move(waveform);
    result.num_samples = static_cast<int32_t>(result.waveform.size());
    std::cerr << "[trtf] Bark: generated " << result.num_samples << " samples ("
              << static_cast<float>(result.num_samples) / result.sample_rate
              << "s @ " << result.sample_rate << " Hz)" << std::endl;

    return result;
}

// ---------------------------------------------------------------------------
// WAV writer
// ---------------------------------------------------------------------------

bool write_wav(const std::string& path, const float* samples,
               int32_t num_samples, int32_t sample_rate)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        return false;
    }

    const int32_t num_channels = 1;
    const int32_t bits_per_sample = 32;
    const int32_t byte_rate = sample_rate * num_channels * (bits_per_sample / 8);
    const int32_t block_align = num_channels * (bits_per_sample / 8);
    const int32_t data_size = num_samples * block_align;
    const int32_t chunk_size = 36 + data_size;

    // RIFF header (44 bytes)
    out.write("RIFF", 4);
    out.write(reinterpret_cast<const char*>(&chunk_size), 4);
    out.write("WAVE", 4);

    // fmt sub-chunk
    out.write("fmt ", 4);
    const int32_t fmt_size = 16;
    out.write(reinterpret_cast<const char*>(&fmt_size), 4);
    const int16_t audio_format = 3;  // IEEE float
    out.write(reinterpret_cast<const char*>(&audio_format), 2);
    const int16_t channels = static_cast<int16_t>(num_channels);
    out.write(reinterpret_cast<const char*>(&channels), 2);
    out.write(reinterpret_cast<const char*>(&sample_rate), 4);
    out.write(reinterpret_cast<const char*>(&byte_rate), 4);
    const int16_t ba = static_cast<int16_t>(block_align);
    out.write(reinterpret_cast<const char*>(&ba), 2);
    const int16_t bps = static_cast<int16_t>(bits_per_sample);
    out.write(reinterpret_cast<const char*>(&bps), 2);

    // data sub-chunk
    out.write("data", 4);
    out.write(reinterpret_cast<const char*>(&data_size), 4);
    out.write(reinterpret_cast<const char*>(samples),
              static_cast<std::streamsize>(data_size));

    return out.good();
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<BarkBackend> CreateBarkBackend(
    std::unique_ptr<DecoderStepEngine> semantic_engine,
    std::unique_ptr<DecoderStepEngine> coarse_engine,
    std::vector<float> semantic_embed,
    std::vector<float> coarse_embed,
    const FastPathModelConfig& cfg)
{
    BarkConfig bark_cfg;
    bark_cfg.sample_rate = cfg.audio_sample_rate;
    bark_cfg.hidden_size = cfg.hidden_size;
    bark_cfg.semantic_input_vocab = cfg.semantic_input_vocab;
    bark_cfg.semantic_output_vocab = cfg.vocab_size;  // engine output vocab
    bark_cfg.text_encoding_offset = cfg.text_encoding_offset;
    bark_cfg.text_pad_token = cfg.text_pad_token;
    bark_cfg.semantic_pad_token = cfg.semantic_pad_token;
    bark_cfg.semantic_infer_token = cfg.semantic_infer_token;
    bark_cfg.semantic_vocab_size = cfg.semantic_vocab_size;
    bark_cfg.coarse_input_vocab = cfg.coarse_input_vocab;
    bark_cfg.coarse_semantic_pad_token = cfg.coarse_semantic_pad_token;
    bark_cfg.coarse_infer_token = cfg.coarse_infer_token;
    bark_cfg.n_coarse_codebooks = cfg.n_coarse_codebooks;
    bark_cfg.codebook_size = cfg.codebook_size;
    bark_cfg.codec_seq_length = cfg.codec_seq_length;
    bark_cfg.codec_upsample_factor = cfg.codec_upsample_factor;
    bark_cfg.codec_n_codebooks = cfg.codec_n_codebooks;
    bark_cfg.fine_hidden_size = cfg.fine_hidden_size;
    bark_cfg.fine_n_lm_heads = cfg.fine_n_lm_heads;
    bark_cfg.fine_codebook_size = cfg.fine_codebook_size;
    bark_cfg.fine_seq_length = cfg.fine_seq_length;

    return std::make_unique<BarkBackend>(
        std::move(semantic_engine), std::move(coarse_engine),
        std::move(semantic_embed), std::move(coarse_embed),
        std::move(bark_cfg));
}

} // namespace trtf

#endif // TRTF_HAS_TRT
