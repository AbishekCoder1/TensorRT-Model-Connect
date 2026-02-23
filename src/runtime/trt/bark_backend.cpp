#include "runtime/trt/bark_backend.h"

#if TRTF_HAS_TRT

#include "runtime/trt/trt_decode_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace trtf {

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

    // Build input sequence:
    //   text_ids padded to 256 (offset by text_encoding_offset) + semantic_history (pad tokens)
    // For no voice preset, semantic_history is all pad tokens.
    //
    // HF flow (BarkSemanticModel.generate):
    //   1. input_ids = input_ids + text_encoding_offset    (add 10048)
    //   2. input_ids.masked_fill(pad_mask, text_pad_token) (replace pad→129595)
    //   3. inputs_embeds = embed(input_ids) + embed(semantic_history)
    //
    // So real positions use embed[token + 10048] and pad positions use
    // embed[text_pad_token=129595]. The C++ must match this exactly —
    // using embed[0 + 10048] for pads corrupts the KV cache.
    constexpr int32_t kMaxTextLen = 256;

    const auto copy_len = std::min(static_cast<int32_t>(text_ids.size()), kMaxTextLen);

    // Prefill: embed = embed[text_tok] + embed[semantic_pad_token]
    std::string error;
    for (int32_t pos = 0; pos < kMaxTextLen; ++pos)
    {
        // Real token positions: embed[token + offset]
        // Pad positions: embed[text_pad_token]  (matching HF's masked_fill)
        int32_t text_tok = (pos < copy_len && text_ids[pos] != 0)
            ? (text_ids[pos] + cfg.text_encoding_offset)
            : cfg.text_pad_token;
        int32_t hist_tok = cfg.semantic_pad_token;

        lookup_embed(mSemanticEmbed.data(), text_tok, embed_a.data());
        lookup_embed(mSemanticEmbed.data(), hist_tok, embed_b.data());
        sum_embeds(embed_a.data(), embed_b.data(), embed_buf.data());

        if (!run_decoder_step_device(*mSemanticEngine, cache, resources,
                0, logits, error,
                embed_buf.data(), cfg.hidden_size, 1.0F))
        {
            throw std::runtime_error("Bark semantic prefill failed: " + error);
        }
    }

    // Feed the infer token
    lookup_embed(mSemanticEmbed.data(), cfg.semantic_infer_token, embed_buf.data());
    if (!run_decoder_step_device(*mSemanticEngine, cache, resources,
            0, logits, error,
            embed_buf.data(), cfg.hidden_size, 1.0F))
    {
        throw std::runtime_error("Bark semantic infer token failed: " + error);
    }

    // Autoregressive generation
    std::vector<int32_t> semantic_tokens;
    semantic_tokens.reserve(static_cast<std::size_t>(max_tokens));

    for (int32_t step = 0; step < max_tokens; ++step)
    {
        // min_eos_p check: compute P(EOS) from raw logits BEFORE suppression.
        // HF uses BarkEosPrioritizerLogitsProcessor which computes softmax
        // over the full output vocab and checks if P(EOS) > threshold.
        // If yes, force EOS (stop generation).
        // Note: HF bark-small has min_eos_p=None (disabled). When min_eos_p <= 0,
        // this check is skipped entirely.
        const float neg_inf = -1e9F;
        if (cfg.min_eos_p > 0.0F)
        {
            float max_val = *std::max_element(logits.begin(),
                logits.begin() + cfg.semantic_pad_token + 1);
            float sum_exp = 0.0F;
            for (int32_t i = 0; i <= cfg.semantic_pad_token; ++i)
            {
                sum_exp += std::exp(logits[i] - max_val);
            }
            float eos_p = std::exp(logits[cfg.semantic_pad_token] - max_val) /
                          std::max(sum_exp, 1e-10F);
            if (eos_p > cfg.min_eos_p)
            {
                break;
            }
        }

        // Suppress tokens [semantic_pad_token+1, output_vocab) matching HF exactly.
        // HF suppresses range(semantic_vocab_size, semantic_pad_token) which is
        // empty when semantic_vocab_size == semantic_pad_token (both 10000).
        // Then suppresses range(semantic_pad_token+1, output_vocab_size).
        // Token 10000 (EOS) is NOT suppressed — it can be sampled naturally.
        for (int32_t i = cfg.semantic_pad_token + 1;
             i < static_cast<int32_t>(logits.size()); ++i)
        {
            logits[i] = neg_inf;
        }

        // Sample from [0, semantic_pad_token+1) = [0, 10001)
        // This includes valid tokens [0,10000) and EOS token 10000
        int32_t token = sample_top_k(logits.data(),
            cfg.semantic_pad_token + 1,
            cfg.semantic_temperature, cfg.top_k);

        if (token == cfg.semantic_pad_token)
        {
            break;  // EOS sampled
        }

        semantic_tokens.push_back(token);

        // Feed sampled token back — use the engine's internal embedding
        // (use_input_embed=0.0) to match HF's autoregressive path, which
        // passes input_ids and lets the model do its own embedding lookup.
        if (!run_decoder_step_device(*mSemanticEngine, cache, resources,
                token, logits, error,
                nullptr, 0, 0.0F))
        {
            throw std::runtime_error("Bark semantic decode failed: " + error);
        }
    }

    std::cerr << "[trtf] Bark semantic: generated " << semantic_tokens.size()
              << " tokens" << std::endl;

    // Dump tokens for comparison (env TRTF_BARK_DUMP)
    {
        const char* dump_path = std::getenv("TRTF_BARK_DUMP");
        if (dump_path != nullptr)
        {
            std::ofstream dump(std::string(dump_path) + ".sem_tokens");
            for (auto t : semantic_tokens)
            {
                dump << t << "\n";
            }
        }
    }

    return semantic_tokens;
}

// ---------------------------------------------------------------------------
// Stage 2: Coarse (semantic tokens -> coarse acoustic codes)
// ---------------------------------------------------------------------------

std::vector<int32_t> BarkBackend::run_coarse(
    const std::vector<int32_t>& semantic_tokens)
{
    const auto& cfg = mConfig;

    // Replace semantic_pad_token with coarse_semantic_pad_token
    std::vector<int32_t> x_semantic;
    x_semantic.reserve(semantic_tokens.size());
    for (int32_t tok : semantic_tokens)
    {
        x_semantic.push_back(
            tok == cfg.semantic_pad_token ? cfg.coarse_semantic_pad_token : tok);
    }

    const int32_t sem_len = static_cast<int32_t>(x_semantic.size());
    const int32_t n_steps = std::max(
        static_cast<int32_t>(std::floor(
            static_cast<float>(sem_len) * cfg.coarse_rate_hz / cfg.semantic_rate_hz
        )) * cfg.n_coarse_codebooks - static_cast<int32_t>(0),
        0);

    if (n_steps == 0)
    {
        std::cerr << "[trtf] Bark coarse: no steps to generate" << std::endl;
        return {};
    }

    const int32_t n_window_steps = static_cast<int32_t>(
        std::ceil(static_cast<float>(n_steps) / cfg.sliding_window_len));

    std::vector<int32_t> x_coarse;  // accumulated coarse output tokens
    x_coarse.reserve(static_cast<std::size_t>(n_steps));

    std::vector<float> logits;
    std::vector<float> embed_buf(static_cast<std::size_t>(cfg.hidden_size));
    std::string error;

    for (int32_t win = 0; win < n_window_steps; ++win)
    {
        const int32_t gen_this_window = std::min(
            cfg.sliding_window_len,
            n_steps - static_cast<int32_t>(x_coarse.size()));
        if (gen_this_window <= 0) break;

        // 1. Build semantic context for this window (matching HF Bark logic).
        //    HF computes: semantic_idx = round(total_generated / semantic_to_coarse_ratio)
        //    Then takes x_semantic[max(0, semantic_idx - max_semantic_history):]
        //    and truncates to max_coarse_input_length, right-padding if needed.
        const int32_t total_generated = static_cast<int32_t>(x_coarse.size());
        const int32_t max_semantic_history = static_cast<int32_t>(std::floor(
            static_cast<float>(cfg.max_coarse_history) * cfg.semantic_rate_hz /
            cfg.coarse_rate_hz));
        const int32_t semantic_idx = static_cast<int32_t>(std::round(
            static_cast<float>(total_generated) * cfg.semantic_rate_hz /
            cfg.coarse_rate_hz));
        int32_t sem_start = std::max(0, semantic_idx - max_semantic_history);
        // Truncate to max_coarse_input_length tokens
        int32_t sem_context_len = std::min(
            sem_len - sem_start, cfg.max_coarse_input_length);

        // 2. Build input token sequence:
        //    [semantic_context] + [right_padding to max_coarse_input_length] + [infer_token] + [coarse_history]
        std::vector<int32_t> input_tokens;
        input_tokens.reserve(
            static_cast<std::size_t>(cfg.max_coarse_input_length) + 1 +
            static_cast<std::size_t>(cfg.max_coarse_history));

        // Semantic context (up to max_coarse_input_length tokens)
        for (int32_t i = sem_start; i < sem_start + sem_context_len; ++i)
        {
            input_tokens.push_back(x_semantic[i]);
        }
        // Right-pad to max_coarse_input_length (matching HF)
        for (int32_t i = sem_context_len; i < cfg.max_coarse_input_length; ++i)
        {
            input_tokens.push_back(cfg.coarse_semantic_pad_token);
        }

        // Append infer token
        input_tokens.push_back(cfg.coarse_infer_token);

        // Append coarse history (up to max_coarse_history)
        int32_t hist_start = std::max(
            0, static_cast<int32_t>(x_coarse.size()) - cfg.max_coarse_history);
        for (int32_t i = hist_start; i < static_cast<int32_t>(x_coarse.size()); ++i)
        {
            input_tokens.push_back(x_coarse[i]);
        }

        // 3. New KV cache for each window
        DeviceKvCache cache(*mCoarseEngine);
        DeviceResources resources(*mCoarseEngine);
        if (!cache.ok() || !resources.ok())
        {
            throw std::runtime_error("Bark coarse: failed to allocate device resources");
        }

        // 4. Prefill all input tokens
        for (std::size_t i = 0; i + 1 < input_tokens.size(); ++i)
        {
            lookup_embed(mCoarseEmbed.data(), input_tokens[i], embed_buf.data());
            if (!run_decoder_step_device(*mCoarseEngine, cache, resources,
                    0, logits, error,
                    embed_buf.data(), cfg.hidden_size, 1.0F))
            {
                throw std::runtime_error("Bark coarse prefill failed: " + error);
            }
        }

        // Feed last prefill token and get first logits
        lookup_embed(mCoarseEmbed.data(), input_tokens.back(), embed_buf.data());
        if (!run_decoder_step_device(*mCoarseEngine, cache, resources,
                0, logits, error,
                embed_buf.data(), cfg.hidden_size, 1.0F))
        {
            throw std::runtime_error("Bark coarse last prefill failed: " + error);
        }

        // 5. Generate sliding_window_len tokens
        const int32_t window_start_count = static_cast<int32_t>(x_coarse.size());
        for (int32_t step = 0; step < gen_this_window; ++step)
        {
            // Determine which codebook this step corresponds to
            const int32_t total_generated = window_start_count + step;
            const int32_t codebook_idx = total_generated % cfg.n_coarse_codebooks;

            // Mask logits to valid codebook range:
            // codebook 0: [cfg.semantic_vocab_size, cfg.semantic_vocab_size + cfg.codebook_size)
            // codebook 1: [cfg.semantic_vocab_size + cfg.codebook_size, cfg.semantic_vocab_size + 2*cfg.codebook_size)
            const int32_t cb_start = cfg.semantic_vocab_size +
                codebook_idx * cfg.codebook_size;
            const int32_t cb_end = cb_start + cfg.codebook_size;
            const float neg_inf = -1e9F;

            for (int32_t i = 0; i < static_cast<int32_t>(logits.size()); ++i)
            {
                if (i < cb_start || i >= cb_end)
                {
                    logits[i] = neg_inf;
                }
            }

            // Sample
            int32_t token = sample_top_k(logits.data(),
                static_cast<int32_t>(logits.size()),
                cfg.coarse_temperature, cfg.top_k);

            x_coarse.push_back(token);

            if (step + 1 < gen_this_window)
            {
                // Feed token embedding back for next step
                lookup_embed(mCoarseEmbed.data(), token, embed_buf.data());
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

    // Dump tokens for comparison
    {
        const char* dump_path = std::getenv("TRTF_BARK_DUMP");
        if (dump_path != nullptr)
        {
            std::ofstream dump(std::string(dump_path) + ".coarse_tokens");
            for (auto t : x_coarse)
            {
                dump << t << "\n";
            }
        }
    }

    return x_coarse;
}

// ---------------------------------------------------------------------------
// Stage 2.5: Fine (coarse codes -> 8 codebook codes)
// ---------------------------------------------------------------------------

std::vector<int32_t> BarkBackend::run_fine(const std::vector<int32_t>& coarse_tokens)
{
    const auto& cfg = mConfig;

    // De-interleave coarse tokens into codes [8, n_frames]
    const int32_t n_frames_raw = static_cast<int32_t>(coarse_tokens.size()) /
                                  cfg.n_coarse_codebooks;
    // Cap to fine_seq_length
    const int32_t n_frames = std::min(
        n_frames_raw,
        cfg.fine_seq_length > 0 ? cfg.fine_seq_length : n_frames_raw);

    // codes[cb * n_frames + frame] = raw code index [0, codebook_size)
    std::vector<int32_t> codes(static_cast<std::size_t>(8) * n_frames, 0);
    for (int32_t t = 0; t < n_frames * cfg.n_coarse_codebooks; ++t)
    {
        int32_t cb = t % cfg.n_coarse_codebooks;
        int32_t frame = t / cfg.n_coarse_codebooks;
        int32_t raw = coarse_tokens[t] - cfg.semantic_vocab_size -
                      cb * cfg.codebook_size;
        raw = std::max(0, std::min(raw, cfg.codebook_size - 1));
        codes[cb * n_frames + frame] = raw;
    }

    // If no fine engine, return codes as-is (codebooks 2-7 = 0)
    if (!mFineEngine || !mFineCtx || cfg.fine_seq_length <= 0)
    {
        std::cerr << "[trtf] Bark fine: no TRT fine engine, "
                  << "codebooks 2-7 will be zero" << std::endl;
        return codes;
    }

    const int32_t fine_hidden = cfg.fine_hidden_size;
    const int32_t fine_cb_size = cfg.fine_codebook_size;
    const int32_t max_seq = cfg.fine_seq_length;

    // Allocate device buffers
    // Input: [max_seq, fine_hidden] float32
    // Output: 7 logit tensors, each [max_seq, fine_cb_size] float32
    CudaBuffer d_input(static_cast<std::size_t>(max_seq) * fine_hidden * sizeof(float));
    std::vector<CudaBuffer> d_outputs;
    d_outputs.reserve(static_cast<std::size_t>(cfg.fine_n_lm_heads));
    for (int32_t i = 0; i < cfg.fine_n_lm_heads; ++i)
    {
        d_outputs.emplace_back(static_cast<std::size_t>(max_seq) * fine_cb_size * sizeof(float));
    }
    CudaStream stream;

    if (!d_input.ok() || !stream.ok())
    {
        std::cerr << "[trtf] Bark fine: failed to allocate CUDA resources" << std::endl;
        return codes;
    }

    // Host buffer for input embeddings
    std::vector<float> host_embeds(
        static_cast<std::size_t>(max_seq) * fine_hidden, 0.0F);

    // Host buffer for logits (read back one head at a time)
    std::vector<float> host_logits(
        static_cast<std::size_t>(max_seq) * fine_cb_size);

    const int32_t actual_frames = std::min(n_frames, max_seq);

    // For each codebook to predict (2..7):
    for (int32_t cb_idx = 2; cb_idx < 8; ++cb_idx)
    {
        // 1. Compute summed embeddings on host
        std::fill(host_embeds.begin(), host_embeds.end(), 0.0F);

        for (int32_t frame = 0; frame < actual_frames; ++frame)
        {
            float* dst = host_embeds.data() +
                static_cast<std::size_t>(frame) * fine_hidden;

            // Sum embeddings for codebooks 0..cb_idx
            for (int32_t cb = 0; cb <= cb_idx; ++cb)
            {
                int32_t code = codes[static_cast<std::size_t>(cb) * n_frames + frame];
                // Each embed table: [fine_cb_size, fine_hidden]
                // mFineEmbed layout: table0[fine_cb_size * fine_hidden] || table1[...] || ...
                const float* table = mFineEmbed.data() +
                    static_cast<std::size_t>(cb) * fine_cb_size * fine_hidden;
                const float* row = table +
                    static_cast<std::size_t>(code) * fine_hidden;
                for (int32_t h = 0; h < fine_hidden; ++h)
                {
                    dst[h] += row[h];
                }
            }

            // Add position embedding
            // mFinePositionEmbed: [max_position, fine_hidden]
            const float* pos_row = mFinePositionEmbed.data() +
                static_cast<std::size_t>(frame) * fine_hidden;
            for (int32_t h = 0; h < fine_hidden; ++h)
            {
                dst[h] += pos_row[h];
            }
        }

        // 2. Copy to device
        cudaMemcpyAsync(d_input.data(), host_embeds.data(),
                        static_cast<std::size_t>(max_seq) * fine_hidden * sizeof(float),
                        cudaMemcpyHostToDevice, stream.get());

        // 3. Bind tensors and execute
        if (!mFineCtx->setTensorAddress("input_embeds", d_input.data()))
        {
            std::cerr << "[trtf] Bark fine: failed to bind input_embeds" << std::endl;
            return codes;
        }
        for (int32_t head = 0; head < cfg.fine_n_lm_heads; ++head)
        {
            std::string name = "logits_cb" + std::to_string(head + 1);
            if (!mFineCtx->setTensorAddress(name.c_str(), d_outputs[head].data()))
            {
                std::cerr << "[trtf] Bark fine: failed to bind " << name << std::endl;
                return codes;
            }
        }

        if (!mFineCtx->enqueueV3(stream.get()))
        {
            std::cerr << "[trtf] Bark fine: TRT execution failed" << std::endl;
            return codes;
        }

        // 4. Read back logits for this codebook's head
        // head index = cb_idx - 1 (head 0 = codebook 1, head 1 = codebook 2, ...)
        int32_t head_idx = cb_idx - 1;
        cudaMemcpyAsync(host_logits.data(), d_outputs[head_idx].data(),
                        static_cast<std::size_t>(max_seq) * fine_cb_size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream.get());
        cudaStreamSynchronize(stream.get());

        // 5. Argmax for each frame, store in codes
        for (int32_t frame = 0; frame < actual_frames; ++frame)
        {
            const float* frame_logits = host_logits.data() +
                static_cast<std::size_t>(frame) * fine_cb_size;
            int32_t best = 0;
            // Only consider first codebook_size (1024) valid codes
            const int32_t valid_range = std::min(cfg.codebook_size, fine_cb_size);
            for (int32_t i = 1; i < valid_range; ++i)
            {
                if (frame_logits[i] > frame_logits[best]) best = i;
            }
            codes[static_cast<std::size_t>(cb_idx) * n_frames + frame] = best;
        }
    }

    std::cerr << "[trtf] Bark fine: predicted codebooks 2-7 for "
              << n_frames << " frames" << std::endl;

    return codes;
}

// ---------------------------------------------------------------------------
// Stage 3: Codec (coarse codes -> waveform)
// ---------------------------------------------------------------------------

std::vector<float> BarkBackend::run_codec(const std::vector<int32_t>& coarse_tokens)
{
    const auto& cfg = mConfig;

    // De-interleave coarse tokens into 2 codebooks:
    // Even indices -> codebook 0, odd indices -> codebook 1
    // Token values: subtract (semantic_vocab_size + codebook_idx * codebook_size)
    // to get raw code indices [0, codebook_size)
    const auto n_tokens = static_cast<int32_t>(coarse_tokens.size());
    const int32_t n_frames = n_tokens / cfg.n_coarse_codebooks;

    if (n_frames == 0)
    {
        return {};
    }

    // Build codes array [n_coarse_codebooks, n_frames] row-major
    // codebook i, frame j -> codes[i * n_frames + j]
    std::vector<int32_t> codes(
        static_cast<std::size_t>(cfg.n_coarse_codebooks) * n_frames, 0);

    for (int32_t t = 0; t < n_frames * cfg.n_coarse_codebooks; ++t)
    {
        const int32_t cb = t % cfg.n_coarse_codebooks;
        const int32_t frame = t / cfg.n_coarse_codebooks;
        // Coarse tokens are offset by semantic_vocab_size + cb * codebook_size
        int32_t raw_code = coarse_tokens[t] - cfg.semantic_vocab_size -
                           cb * cfg.codebook_size;
        raw_code = std::max(0, std::min(raw_code, cfg.codebook_size - 1));
        codes[cb * n_frames + frame] = raw_code;
    }

    // If no codec engine or codec not configured, fall back to simple synthesis
    if (!mCodecEngine || !mCodecCtx || cfg.codec_seq_length <= 0)
    {
        std::cerr << "[trtf] Bark codec: no TRT codec engine, "
                  << "generating simple waveform from codes" << std::endl;

        const int32_t samples_per_frame = cfg.sample_rate / cfg.coarse_rate_hz;
        const int32_t total_samples = n_frames * samples_per_frame;
        std::vector<float> waveform(static_cast<std::size_t>(total_samples), 0.0F);

        for (int32_t f = 0; f < n_frames; ++f)
        {
            const float freq = 200.0F +
                static_cast<float>(codes[f]) * 800.0F /
                static_cast<float>(cfg.codebook_size);
            const float amp = 0.3F;
            for (int32_t s = 0; s < samples_per_frame; ++s)
            {
                const auto idx = static_cast<std::size_t>(f) * samples_per_frame + s;
                const float t = static_cast<float>(s) /
                                static_cast<float>(cfg.sample_rate);
                waveform[idx] = amp * std::sin(2.0F * 3.14159265F * freq * t);
            }
        }

        return waveform;
    }

    // --- TRT codec (EnCodec decoder) execution ---
    const int32_t n_cb = cfg.codec_n_codebooks;     // 8
    const int32_t max_T = cfg.codec_seq_length;      // engine compiled max frames
    const int32_t upsample = cfg.codec_upsample_factor; // 320

    if (n_frames > max_T)
    {
        std::cerr << "[trtf] Bark codec: n_frames=" << n_frames
                  << " exceeds codec_seq_length=" << max_T
                  << ", truncating" << std::endl;
    }
    const int32_t actual_frames = std::min(n_frames, max_T);

    // Build input: [1, n_cb, max_T] int32, zero-padded.
    // We have codes[cb * n_frames + frame] for cb in [0, n_coarse_codebooks).
    // Remaining codebooks (2..7) are zero.
    const auto input_size = static_cast<std::size_t>(n_cb) * max_T;
    std::vector<int32_t> input_codes(input_size, 0);
    for (int32_t cb = 0; cb < std::min(cfg.n_coarse_codebooks, n_cb); ++cb)
    {
        for (int32_t f = 0; f < actual_frames; ++f)
        {
            input_codes[static_cast<std::size_t>(cb) * max_T + f] =
                codes[static_cast<std::size_t>(cb) * n_frames + f];
        }
    }

    // Allocate device memory
    const auto input_bytes = input_size * sizeof(int32_t);
    const auto output_elems = static_cast<std::size_t>(1) * 1 * max_T * upsample;
    const auto output_bytes = output_elems * sizeof(float);

    CudaBuffer d_input(input_bytes);
    CudaBuffer d_output(output_bytes);
    CudaStream stream;

    if (!d_input.ok() || !d_output.ok() || !stream.ok())
    {
        std::cerr << "[trtf] Bark codec: failed to allocate CUDA resources" << std::endl;
        return {};
    }

    // Copy input to device
    cudaMemcpyAsync(d_input.data(), input_codes.data(), input_bytes,
                    cudaMemcpyHostToDevice, stream.get());

    // Bind tensors to execution context
    if (!mCodecCtx->setTensorAddress("audio_codes", d_input.data()))
    {
        std::cerr << "[trtf] Bark codec: failed to bind audio_codes" << std::endl;
        return {};
    }
    if (!mCodecCtx->setTensorAddress("waveform", d_output.data()))
    {
        std::cerr << "[trtf] Bark codec: failed to bind waveform" << std::endl;
        return {};
    }

    // Execute
    if (!mCodecCtx->enqueueV3(stream.get()))
    {
        std::cerr << "[trtf] Bark codec: TRT execution failed" << std::endl;
        return {};
    }

    // Copy output back to host
    std::vector<float> full_waveform(output_elems);
    cudaMemcpyAsync(full_waveform.data(), d_output.data(), output_bytes,
                    cudaMemcpyDeviceToHost, stream.get());
    cudaStreamSynchronize(stream.get());

    // Trim to actual_frames * upsample_factor
    const auto trimmed_samples = static_cast<std::size_t>(actual_frames) * upsample;
    std::vector<float> waveform(full_waveform.begin(),
                                 full_waveform.begin() +
                                 static_cast<std::ptrdiff_t>(trimmed_samples));

    std::cerr << "[trtf] Bark codec: TRT decode " << actual_frames << " frames -> "
              << waveform.size() << " samples" << std::endl;

    return waveform;
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
                const float t = static_cast<float>(s) /
                                static_cast<float>(cfg.sample_rate);
                waveform[idx] = amp * std::sin(2.0F * 3.14159265F * freq * t);
            }
        }

        return waveform;
    }

    // --- TRT codec (EnCodec decoder) execution ---
    const int32_t n_cb = cfg.codec_n_codebooks;       // 8
    const int32_t max_T = cfg.codec_seq_length;
    const int32_t upsample = cfg.codec_upsample_factor;

    if (n_frames > max_T)
    {
        std::cerr << "[trtf] Bark codec: n_frames=" << n_frames
                  << " exceeds codec_seq_length=" << max_T
                  << ", truncating" << std::endl;
    }
    const int32_t actual_frames = std::min(n_frames, max_T);

    // Build input: [1, n_cb, max_T] int32, zero-padded.
    // codes_flat layout: codes_flat[cb * n_frames + frame]
    const auto input_size = static_cast<std::size_t>(n_cb) * max_T;
    std::vector<int32_t> input_codes(input_size, 0);
    for (int32_t cb = 0; cb < std::min(8, n_cb); ++cb)
    {
        for (int32_t f = 0; f < actual_frames; ++f)
        {
            input_codes[static_cast<std::size_t>(cb) * max_T + f] =
                codes_flat[static_cast<std::size_t>(cb) * n_frames + f];
        }
    }

    // Allocate device memory
    const auto input_bytes = input_size * sizeof(int32_t);
    const auto output_elems = static_cast<std::size_t>(1) * 1 * max_T * upsample;
    const auto output_bytes = output_elems * sizeof(float);

    CudaBuffer d_input_codec(input_bytes);
    CudaBuffer d_output_codec(output_bytes);
    CudaStream stream;

    if (!d_input_codec.ok() || !d_output_codec.ok() || !stream.ok())
    {
        std::cerr << "[trtf] Bark codec: failed to allocate CUDA resources" << std::endl;
        return {};
    }

    // Copy input to device
    cudaMemcpyAsync(d_input_codec.data(), input_codes.data(), input_bytes,
                    cudaMemcpyHostToDevice, stream.get());

    // Bind tensors to execution context
    if (!mCodecCtx->setTensorAddress("audio_codes", d_input_codec.data()))
    {
        std::cerr << "[trtf] Bark codec: failed to bind audio_codes" << std::endl;
        return {};
    }
    if (!mCodecCtx->setTensorAddress("waveform", d_output_codec.data()))
    {
        std::cerr << "[trtf] Bark codec: failed to bind waveform" << std::endl;
        return {};
    }

    // Execute
    if (!mCodecCtx->enqueueV3(stream.get()))
    {
        std::cerr << "[trtf] Bark codec: TRT execution failed" << std::endl;
        return {};
    }

    // Copy output back to host
    std::vector<float> full_waveform(output_elems);
    cudaMemcpyAsync(full_waveform.data(), d_output_codec.data(), output_bytes,
                    cudaMemcpyDeviceToHost, stream.get());
    cudaStreamSynchronize(stream.get());

    // Trim to actual_frames * upsample_factor
    const auto trimmed_samples = static_cast<std::size_t>(actual_frames) * upsample;
    std::vector<float> waveform(full_waveform.begin(),
                                 full_waveform.begin() +
                                 static_cast<std::ptrdiff_t>(trimmed_samples));

    std::cerr << "[trtf] Bark codec: TRT decode " << actual_frames << " frames -> "
              << waveform.size() << " samples (from fine codes)" << std::endl;

    return waveform;
}

// ---------------------------------------------------------------------------
// Full pipeline: generate_audio
// ---------------------------------------------------------------------------

AudioResult BarkBackend::generate_audio(
    const std::vector<int32_t>& input_ids,
    int32_t max_semantic_tokens)
{
    AudioResult result;
    result.sample_rate = mConfig.sample_rate;

    if (!is_available())
    {
        std::cerr << "[trtf] Bark: backend not fully initialized" << std::endl;
        return result;
    }

    // Check env for greedy mode (for testing)
    {
        const char* env = std::getenv("TRTF_BARK_GREEDY");
        if (env != nullptr && std::string(env) == "1")
        {
            mConfig.greedy = true;
        }
    }

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
    const int32_t n_fine_frames = static_cast<int32_t>(fine_codes.size()) / 8;

    // Stage 3: Fine codes -> Waveform via codec
    std::vector<float> waveform;
    if (n_fine_frames > 0 && mFineEngine)
    {
        waveform = run_codec(fine_codes, n_fine_frames);
    }
    else
    {
        // Fallback: use original coarse tokens path
        waveform = run_codec(coarse_tokens);
    }
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
