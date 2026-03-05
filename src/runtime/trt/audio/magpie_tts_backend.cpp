#include "runtime/trt/audio/magpie_tts_backend.h"

#if TRTF_HAS_TRT

#include "runtime/trt/core/trt_decode_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace trtf {

namespace {

constexpr int32_t kMagpieBosToken = 2016;
constexpr int32_t kMagpieEosToken = 2017;
constexpr int32_t kMagpieAudioRange = 2016;
constexpr int32_t kMagpieMinFrames = 4;
constexpr int32_t kFsqTotalCodes = 2016;

struct FrameDecodeResult {
    std::vector<int32_t> frame_codes;
    bool eos{false};
};

void copy_embed_row(const float* table, int32_t token_id, int32_t hidden, float* out)
{
    const auto offset = static_cast<std::size_t>(token_id) * static_cast<std::size_t>(hidden);
    std::memcpy(out, table + offset, static_cast<std::size_t>(hidden) * sizeof(float));
}

void build_frame_input_embed(
    const std::vector<int32_t>& prev_codes,
    const float* audio_embed,
    int32_t num_cb,
    int32_t cb_size,
    int32_t hidden,
    std::vector<float>& embed_buf,
    std::vector<float>& cb_embed)
{
    std::fill(embed_buf.begin(), embed_buf.end(), 0.0F);
    for (int32_t cb = 0; cb < num_cb; ++cb)
    {
        const float* table = audio_embed +
            static_cast<std::size_t>(cb) * cb_size * hidden;
        copy_embed_row(table, prev_codes[cb], hidden, cb_embed.data());
        for (int32_t i = 0; i < hidden; ++i)
        {
            embed_buf[i] += cb_embed[i];
        }
    }

    const float inv_cb = 1.0F / static_cast<float>(num_cb);
    for (float& value : embed_buf)
    {
        value *= inv_cb;
    }
}

int32_t argmax_index(const float* values, int32_t count)
{
    int32_t best_id = 0;
    float best_val = values[0];
    for (int32_t i = 1; i < count; ++i)
    {
        if (values[i] > best_val)
        {
            best_val = values[i];
            best_id = i;
        }
    }
    return best_id;
}

FrameDecodeResult decode_frame_codes(
    const std::vector<float>& logits,
    int32_t num_cb,
    int32_t cb_size,
    bool greedy,
    float temperature,
    int32_t top_k,
    const std::function<int32_t(const float*, int32_t, float, int32_t)>& sampler)
{
    FrameDecodeResult result;
    result.frame_codes.assign(static_cast<std::size_t>(num_cb), 0);

    for (int32_t cb = 0; cb < num_cb; ++cb)
    {
        const int32_t offset = cb * cb_size;
        if (offset + cb_size > static_cast<int32_t>(logits.size()))
        {
            continue;
        }

        const float* cb_logits = logits.data() + offset;
        if (argmax_index(cb_logits, cb_size) == kMagpieEosToken)
        {
            result.eos = true;
        }

        if (greedy)
        {
            result.frame_codes[cb] = argmax_index(cb_logits, kMagpieAudioRange);
            continue;
        }

        result.frame_codes[cb] = sampler(cb_logits, kMagpieAudioRange, temperature, top_k);
    }

    return result;
}

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

std::vector<int32_t> build_codec_input(
    const std::vector<int32_t>& codes,
    int32_t num_cb,
    int32_t max_codec_frames,
    int32_t padded_frames)
{
    std::vector<int32_t> codec_input(
        static_cast<std::size_t>(num_cb) * max_codec_frames, 0);
    for (int32_t f = 0; f < padded_frames; ++f)
    {
        for (int32_t cb = 0; cb < num_cb; ++cb)
        {
            const auto src_idx = static_cast<std::size_t>(f) * num_cb + cb;
            if (src_idx >= codes.size())
            {
                continue;
            }

            int32_t code = codes[src_idx];
            if (code >= kFsqTotalCodes)
            {
                code = 0;
            }
            codec_input[static_cast<std::size_t>(cb) * max_codec_frames + f] = code;
        }
    }
    return codec_input;
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
    const int32_t padded_frames = std::min(num_frames, max_codec_frames);
    std::vector<int32_t> codec_input = build_codec_input(
        codes, num_cb, max_codec_frames, padded_frames);

    const int32_t input_len_val = padded_frames;
    const auto output_elems = static_cast<std::size_t>(max_codec_frames) * 1024;
    const auto input_bytes = codec_input.size() * sizeof(int32_t);
    const auto len_bytes = sizeof(int32_t);
    const auto output_bytes = output_elems * sizeof(float);
    CudaBuffer d_input(input_bytes);
    CudaBuffer d_len(len_bytes);
    CudaBuffer d_output(output_bytes);
    CudaStream stream;
    if (!d_input.ok() || !d_len.ok() || !d_output.ok() || !stream.ok())
    {
        std::cerr << "[magpie-tts] Failed to allocate CUDA resources for codec" << std::endl;
        return {};
    }

    cudaMemcpyAsync(
        d_input.data(), codec_input.data(), input_bytes, cudaMemcpyHostToDevice, stream.get());
    cudaMemcpyAsync(
        d_len.data(), &input_len_val, len_bytes, cudaMemcpyHostToDevice, stream.get());
    if (!bind_magpie_codec_tensors(codec_ctx, d_input, d_len, d_output))
    {
        return {};
    }
    if (!codec_ctx.enqueueV3(stream.get()))
    {
        std::cerr << "[magpie-tts] Codec TRT execution failed" << std::endl;
        return {};
    }

    const auto valid_samples = static_cast<std::size_t>(padded_frames) * 1024;
    std::vector<float> waveform(valid_samples);
    cudaMemcpyAsync(
        waveform.data(),
        d_output.data(),
        valid_samples * sizeof(float),
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
    , mAudioEmbed(std::move(audio_embed))
    , mTextEmbed(std::move(text_embed))
    , mContextEmbed(std::move(context_embed))
    , mContextLengths(std::move(context_lengths))
    , mConfig(config)
{
    if (mDecoderEngine)
    {
        mCache = std::make_unique<DeviceKvCache>(*mDecoderEngine);
        mResources = std::make_unique<DeviceResources>(*mDecoderEngine);
    }

    // Cross-attention K/V: one buffer per decoder layer, same size as encoder output
    const std::size_t enc_buf_size = mEncoderOutput.size();
    const int32_t dec_layers = config.decoder_layers > 0
        ? config.decoder_layers : (mDecoderEngine ? mDecoderEngine->num_layers : 1);
    mCrossK.reserve(static_cast<std::size_t>(dec_layers));
    mCrossV.reserve(static_cast<std::size_t>(dec_layers));
    for (int32_t i = 0; i < dec_layers; ++i)
    {
        mCrossK.emplace_back(enc_buf_size);
        mCrossV.emplace_back(enc_buf_size);
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
// compute_cross_kv() -- copy encoder output to per-layer cross K/V buffers
// ---------------------------------------------------------------------------

void MagpieTTSBackend::compute_cross_kv()
{
    const int32_t enc_seq = mConfig.max_source_positions;
    const int32_t hidden = mConfig.hidden_size;
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
}

namespace {

bool prefill_magpie_context_frames(
    DecoderStepEngine& engine,
    DeviceKvCache& cache,
    DeviceResources& resources,
    const std::vector<float>& context_embed,
    const std::vector<int32_t>& context_lengths,
    int32_t hidden,
    std::vector<float>& logits)
{
    if (context_embed.empty() || context_lengths.empty())
    {
        return true;
    }

    const int32_t ctx_frames = context_lengths[0];
    const float* ctx_ptr = context_embed.data();
    std::cerr << "[magpie-tts] Prefilling " << ctx_frames
              << " context frames ..." << std::endl;

    std::string error;
    for (int32_t pos = 0; pos < ctx_frames; ++pos)
    {
        const float* frame_embed = ctx_ptr +
            static_cast<std::size_t>(pos) * hidden;
        if (!run_decoder_step_device(engine, cache, resources,
                0, logits, error,
                frame_embed, hidden, 1.0F))
        {
            std::cerr << "[magpie-tts] Context step " << pos
                      << " failed: " << error << std::endl;
            return false;
        }
    }
    return true;
}

bool run_magpie_decode_frame(
    DecoderStepEngine& engine,
    DeviceKvCache& cache,
    DeviceResources& resources,
    int32_t frame,
    int32_t hidden,
    int32_t num_cb,
    int32_t cb_size,
    const float* audio_embed,
    bool greedy,
    float temperature,
    int32_t top_k,
    const std::function<int32_t(const float*, int32_t, float, int32_t)>& sampler,
    std::vector<int32_t>& prev_codes,
    std::vector<float>& embed_buf,
    std::vector<float>& cb_embed,
    std::vector<float>& logits,
    std::vector<int32_t>& all_codes,
    std::string& error)
{
    build_frame_input_embed(
        prev_codes,
        audio_embed,
        num_cb,
        cb_size,
        hidden,
        embed_buf,
        cb_embed);
    if (!run_decoder_step_device(engine, cache, resources,
            0, logits, error,
            embed_buf.data(), hidden, 1.0F))
    {
        std::cerr << "[magpie-tts] Decode step " << frame << " failed: "
                  << error << std::endl;
        return false;
    }

    const FrameDecodeResult frame_result = decode_frame_codes(
        logits,
        num_cb,
        cb_size,
        greedy,
        temperature,
        top_k,
        sampler);

    all_codes.insert(
        all_codes.end(), frame_result.frame_codes.begin(), frame_result.frame_codes.end());
    prev_codes = frame_result.frame_codes;
    return !(frame_result.eos && frame >= kMagpieMinFrames);
}

} // namespace

// ---------------------------------------------------------------------------
// run_decoder() -- multi-codebook autoregressive decode
// ---------------------------------------------------------------------------

std::vector<int32_t> MagpieTTSBackend::run_decoder(int32_t max_frames)
{
    if (!mDecoderEngine || !mCache || !mResources) return {};

    auto& engine = *mDecoderEngine;
    auto& cache = *mCache;
    auto& resources = *mResources;
    const int32_t hidden = mConfig.hidden_size;
    const int32_t num_cb = mConfig.num_codebooks;
    const int32_t cb_size = mConfig.codebook_size;

    // Reset KV cache
    cache.reset(resources.stream.get());
    cudaStreamSynchronize(resources.stream.get());

    // Bind cross-attention K/V
    bind_cross_kv();

    std::vector<float> logits;
    std::vector<float> embed_buf(static_cast<std::size_t>(hidden));
    std::vector<float> cb_embed(static_cast<std::size_t>(hidden));

    if (!prefill_magpie_context_frames(
            engine,
            cache,
            resources,
            mContextEmbed,
            mContextLengths,
            hidden,
            logits))
    {
        return {};
    }

    std::vector<int32_t> all_codes;
    all_codes.reserve(static_cast<std::size_t>(max_frames) * num_cb);

    std::vector<int32_t> prev_codes(static_cast<std::size_t>(num_cb), kMagpieBosToken);
    auto sampler = [this](const float* values, int32_t vocab_size, float temp, int32_t top_k) {
        return sample_top_k(values, vocab_size, temp, top_k);
    };
    std::string error;

    for (int32_t frame = 0; frame < max_frames; ++frame)
    {
        if (!run_magpie_decode_frame(
                engine,
                cache,
                resources,
                frame,
                hidden,
                num_cb,
                cb_size,
                mAudioEmbed.data(),
                mConfig.greedy,
                mConfig.temperature,
                mConfig.top_k,
                sampler,
                prev_codes,
                embed_buf,
                cb_embed,
                logits,
                all_codes,
                error))
        {
            break;
        }
    }

    const int32_t gen_frames = static_cast<int32_t>(all_codes.size()) / std::max(num_cb, 1);
    std::cerr << "[magpie-tts] Generated " << gen_frames
              << " frames (" << all_codes.size() << " codes)" << std::endl;
    log_frame_preview(all_codes, num_cb);
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
// generate_audio() -- full pipeline orchestration
// ---------------------------------------------------------------------------

AudioResult MagpieTTSBackend::generate_audio(
    const std::vector<int32_t>& text_ids,
    int32_t max_frames)
{
    AudioResult result;
    result.sample_rate = mConfig.sample_rate;

    if (!is_available())
    {
        std::cerr << "[magpie-tts] Backend not fully initialized" << std::endl;
        return result;
    }

    maybe_enable_magpie_greedy(mConfig);

    std::cerr << "[magpie-tts] Starting pipeline with " << text_ids.size()
              << " text tokens, max_frames=" << max_frames
              << (mConfig.greedy ? " (greedy)" : "") << std::endl;

    // Stage 1: Encode text + speaker/language
    std::cerr << "[magpie-tts] Running encoder ..." << std::endl;
    run_encoder(text_ids, /*speaker_id=*/0, /*language_id=*/0);

    // Stage 2: Compute cross-attention K/V
    std::cerr << "[magpie-tts] Computing cross-attention K/V ..." << std::endl;
    compute_cross_kv();

    // Stage 3: Autoregressive decode -> multi-codebook codes
    std::cerr << "[magpie-tts] Running decoder ..." << std::endl;
    auto codes = run_decoder(max_frames);
    if (codes.empty())
    {
        std::cerr << "[magpie-tts] Decoder produced no codes" << std::endl;
        return result;
    }

    const int32_t num_frames = static_cast<int32_t>(codes.size()) / mConfig.num_codebooks;

    // Stage 4: Codec -> waveform
    std::cerr << "[magpie-tts] Running codec ..." << std::endl;
    auto waveform = run_codec(codes, num_frames);
    if (waveform.empty())
    {
        std::cerr << "[magpie-tts] Codec produced no audio" << std::endl;
        return result;
    }

    result.waveform = std::move(waveform);
    result.num_samples = static_cast<int32_t>(result.waveform.size());
    std::cerr << "[magpie-tts] Generated " << result.num_samples << " samples ("
              << static_cast<float>(result.num_samples) / result.sample_rate
              << "s @ " << result.sample_rate << " Hz)" << std::endl;

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
