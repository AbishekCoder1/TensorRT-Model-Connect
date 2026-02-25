#include "runtime/trt/speech_backend.h"

#if TRTF_HAS_TRT

#include "runtime/trt/trt_decode_runtime.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace trtf {

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

SpeechToSpeechBackend::SpeechToSpeechBackend(
    std::unique_ptr<DecoderStepEngine> temporal_engine,
    SpeechConfig config)
    : mTemporalEngine(std::move(temporal_engine))
    , mConfig(std::move(config))
{
    // Seed RNG from system clock (non-deterministic but good enough)
    auto now = std::chrono::high_resolution_clock::now();
    mRngState = static_cast<uint64_t>(now.time_since_epoch().count());
    if (mRngState == 0) mRngState = 0xDEADBEEF42ULL;  // avoid zero state
}

SpeechToSpeechBackend::~SpeechToSpeechBackend() = default;

bool SpeechToSpeechBackend::is_available() const
{
    return mTemporalEngine != nullptr;
}

void SpeechToSpeechBackend::set_depth_engine(
    std::unique_ptr<DecoderStepEngine> engine)
{
    mDepthEngine = std::move(engine);
}

void SpeechToSpeechBackend::set_depth_engine(
    int32_t codebook_idx,
    std::unique_ptr<DecoderStepEngine> engine)
{
    if (codebook_idx < 0) return;
    if (static_cast<std::size_t>(codebook_idx) >= mDepthEngines.size())
        mDepthEngines.resize(static_cast<std::size_t>(codebook_idx) + 1);
    mDepthEngines[static_cast<std::size_t>(codebook_idx)] = std::move(engine);
}

void SpeechToSpeechBackend::set_mimi_encoder(
    TrtUniquePtr<nvinfer1::ICudaEngine> engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> context)
{
    mMimiEncoderEngine = std::move(engine);
    mMimiEncoderCtx = std::move(context);
}

void SpeechToSpeechBackend::set_mimi_decoder(
    TrtUniquePtr<nvinfer1::ICudaEngine> engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> context)
{
    mMimiDecoderEngine = std::move(engine);
    mMimiDecoderCtx = std::move(context);
}

// ---------------------------------------------------------------------------
// Subprocess helper (used for runtime text tokenization fallback)
// ---------------------------------------------------------------------------

/// Run a subprocess: write input_data to stdin, read stdout/stderr.
/// Returns exit code. Populates out_stdout and out_stderr.
static int run_subprocess(
    const std::vector<std::string>& argv,
    const void* input_data, std::size_t input_size,
    std::vector<char>& out_stdout,
    std::string& out_stderr)
{
    // Build argv for execvp
    std::vector<const char*> c_argv;
    for (const auto& a : argv) c_argv.push_back(a.c_str());
    c_argv.push_back(nullptr);

    int stdin_pipe[2] = {-1, -1};
    int stdout_pipe[2] = {-1, -1};
    int stderr_pipe[2] = {-1, -1};

    if (pipe(stdin_pipe) != 0 || pipe(stdout_pipe) != 0 || pipe(stderr_pipe) != 0)
    {
        out_stderr = "pipe() failed";
        return -1;
    }

    pid_t pid = fork();
    if (pid < 0)
    {
        out_stderr = "fork() failed";
        return -1;
    }

    if (pid == 0)
    {
        // Child: redirect stdin/stdout/stderr
        dup2(stdin_pipe[0], STDIN_FILENO);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);
        close(stdin_pipe[0]); close(stdin_pipe[1]);
        close(stdout_pipe[0]); close(stdout_pipe[1]);
        close(stderr_pipe[0]); close(stderr_pipe[1]);

        execvp(c_argv[0], const_cast<char* const*>(c_argv.data()));
        _exit(127);
    }

    // Parent
    close(stdin_pipe[0]);
    close(stdout_pipe[1]);
    close(stderr_pipe[1]);

    // Write input to stdin
    if (input_data && input_size > 0)
    {
        const auto* p = static_cast<const char*>(input_data);
        std::size_t remaining = input_size;
        while (remaining > 0)
        {
            auto w = write(stdin_pipe[1], p, remaining);
            if (w <= 0) break;
            p += w;
            remaining -= static_cast<std::size_t>(w);
        }
    }
    close(stdin_pipe[1]);

    // Read stdout
    out_stdout.clear();
    char buf[65536];
    for (;;)
    {
        auto n = read(stdout_pipe[0], buf, sizeof(buf));
        if (n <= 0) break;
        out_stdout.insert(out_stdout.end(), buf, buf + n);
    }
    close(stdout_pipe[0]);

    // Read stderr
    out_stderr.clear();
    for (;;)
    {
        auto n = read(stderr_pipe[0], buf, sizeof(buf));
        if (n <= 0) break;
        out_stderr.append(buf, static_cast<std::size_t>(n));
    }
    close(stderr_pipe[0]);

    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

// ---------------------------------------------------------------------------
// Text Prompt Injection
// ---------------------------------------------------------------------------

void SpeechToSpeechBackend::run_text_prompt(
    DeviceKvCache& cache, DeviceResources& resources,
    CudaBuffer& d_hidden_state)
{
    const auto& cfg = mConfig;
    const int32_t hidden = mTemporalEngine->hidden_size;
    const bool has_temporal_text_emb = !cfg.temporal_text_embedding.empty()
                                       && cfg.temporal_text_vocab > 0;
    const bool has_input_embed = has_io_tensor(*mTemporalEngine->engine, "input_embed");

    if (!has_temporal_text_emb || !has_input_embed)
    {
        std::cerr << "[speech] Cannot inject text prompt: missing embeddings"
                  << std::endl;
        return;
    }

    std::vector<int32_t> text_tokens = cfg.text_prompt_ids;
    if (!text_tokens.empty())
    {
        std::cerr << "[speech] Injecting pre-tokenized text prompt ("
                  << text_tokens.size() << " tokens)" << std::endl;
    }
    else
    {
        if (cfg.system_prompt.empty() || cfg.hf_python.empty())
            return;

        // Fallback path: tokenize system prompt via Python subprocess.
        // Uses the tokenizer package available in the runtime environment.
        std::string cmd = cfg.hf_python + " -c \""
            "from transformers import AutoTokenizer; "
            "tok = AutoTokenizer.from_pretrained('kyutai/moshiko-pytorch-bf16'); "
            "ids = tok.encode('" + cfg.system_prompt + "', add_special_tokens=False); "
            "import sys; sys.stdout.buffer.write(b''.join(i.to_bytes(4, 'little') for i in ids))\"";

        std::vector<std::string> argv = {
            "/bin/sh", "-c", cmd
        };

        std::vector<char> stdout_data;
        std::string stderr_data;
        int rc = run_subprocess(argv, nullptr, 0, stdout_data, stderr_data);

        if (rc != 0 || stdout_data.empty())
        {
            std::cerr << "[speech] Text prompt tokenization failed (rc=" << rc
                      << "): " << stderr_data << std::endl;
            return;
        }

        auto num_tokens = stdout_data.size() / sizeof(int32_t);
        text_tokens.resize(num_tokens);
        std::memcpy(text_tokens.data(), stdout_data.data(),
                    num_tokens * sizeof(int32_t));

        std::cerr << "[speech] Injecting runtime-tokenized text prompt: \""
                  << cfg.system_prompt << "\" (" << num_tokens << " tokens)"
                  << std::endl;
    }

    // Run each text token through temporal with silence audio.
    // This primes the KV cache with the system instruction context.
    const int32_t audio_vocab = cfg.audio_vocab_size;
    const auto emb_stride_cb = static_cast<std::size_t>(audio_vocab) * hidden;
    const int32_t bos_token = cfg.codebook_size;  // silence/BOS token
    const int32_t num_cb = cfg.num_codebooks;

    std::vector<float> summed_embed(static_cast<std::size_t>(hidden));
    std::vector<float> logits;
    std::string error;
    auto num_tokens = text_tokens.size();

    for (std::size_t t = 0; t < num_tokens; ++t)
    {
        // Build embedding: text_emb(text_token) + sum of BOS audio embeddings
        std::fill(summed_embed.begin(), summed_embed.end(), 0.0F);

        // Text embedding
        int32_t ttok = std::max(0, std::min(text_tokens[t],
            cfg.temporal_text_vocab - 1));
        auto text_offset = static_cast<std::size_t>(ttok) * hidden;
        if (text_offset + hidden <= cfg.temporal_text_embedding.size())
        {
            const float* row = cfg.temporal_text_embedding.data() + text_offset;
            for (int32_t d = 0; d < hidden; ++d)
                summed_embed[d] += row[d];
        }

        // All audio codebooks set to BOS (silence)
        for (int32_t cb = 0; cb < num_cb; ++cb)
        {
            int32_t tok = std::max(0, std::min(bos_token, audio_vocab - 1));
            auto emb_offset = static_cast<std::size_t>(cb) * emb_stride_cb
                            + static_cast<std::size_t>(tok) * hidden;
            if (emb_offset + hidden <= cfg.audio_embeddings.size())
            {
                const float* row = cfg.audio_embeddings.data() + emb_offset;
                for (int32_t d = 0; d < hidden; ++d)
                    summed_embed[d] += row[d];
            }
        }

        // Run temporal step
        if (!run_decoder_step_device(*mTemporalEngine, cache, resources,
                0, logits, error,
                summed_embed.data(), hidden, 1.0F))
        {
            std::cerr << "[speech] Text prompt step " << t << " failed: "
                      << error << std::endl;
            return;
        }
    }

    std::cerr << "[speech] Text prompt injection complete (" << num_tokens
              << " temporal steps)" << std::endl;
}

// ---------------------------------------------------------------------------
// Stage 1: Mimi Encode (audio waveform -> codec tokens)
// ---------------------------------------------------------------------------

std::vector<int32_t> SpeechToSpeechBackend::run_mimi_encode(
    const float* samples, int32_t num_samples,
    int32_t input_sample_rate)
{
    mLastEncodeFrames = 0;
    mLastEncodeCodebooks = 0;

    if (!mMimiEncoderEngine || !mMimiEncoderCtx)
    {
        std::cerr << "[speech] No Mimi TRT encoder available" << std::endl;
        return {};
    }

    // Query the actual TRT engine I/O shapes
    auto in_dims = mMimiEncoderEngine->getTensorShape("audio_input");
    auto out_dims = mMimiEncoderEngine->getTensorShape("codec_tokens");

    // Encoder was built for a fixed input size (e.g. [1, 1, 12000])
    int32_t engine_input_samples = in_dims.d[in_dims.nbDims - 1];

    // Verify input matches engine expectation
    if (num_samples != engine_input_samples)
    {
        std::cerr << "[speech] WARNING: input samples " << num_samples
                  << " != engine expects " << engine_input_samples
                  << ", using engine size" << std::endl;
    }

    // Use the engine's expected input size for the buffer
    const auto input_elems = static_cast<std::size_t>(engine_input_samples);
    const auto input_bytes = input_elems * sizeof(float);

    // Output shape from TRT engine: [num_codebooks, num_frames]
    int32_t enc_codebooks = out_dims.d[0];
    int32_t enc_frames = out_dims.d[1];
    const auto output_elems = static_cast<std::size_t>(enc_codebooks) * enc_frames;
    const auto output_bytes = output_elems * sizeof(float);

    std::cerr << "[speech] Mimi encoder TRT: input [1,1,"
              << engine_input_samples << "], output ["
              << enc_codebooks << "," << enc_frames << "]" << std::endl;

    // Prepare input: pad or truncate to match engine size
    std::vector<float> input_buf(input_elems, 0.0F);
    auto copy_n = std::min(static_cast<std::size_t>(num_samples), input_elems);
    std::memcpy(input_buf.data(), samples, copy_n * sizeof(float));

    CudaBuffer d_input(input_bytes);
    CudaBuffer d_output(output_bytes);
    CudaStream stream;

    if (!d_input.ok() || !d_output.ok() || !stream.ok())
    {
        std::cerr << "[speech] Failed to allocate CUDA resources for Mimi encoder"
                  << std::endl;
        return {};
    }

    cudaMemcpyAsync(d_input.data(), input_buf.data(), input_bytes,
                    cudaMemcpyHostToDevice, stream.get());

    mMimiEncoderCtx->setTensorAddress("audio_input", d_input.data());
    mMimiEncoderCtx->setTensorAddress("codec_tokens", d_output.data());

    if (!mMimiEncoderCtx->enqueueV3(stream.get()))
    {
        std::cerr << "[speech] Mimi encoder TRT execution failed" << std::endl;
        return {};
    }

    std::vector<float> host_tokens_f(output_elems);
    cudaMemcpyAsync(host_tokens_f.data(), d_output.data(), output_bytes,
                    cudaMemcpyDeviceToHost, stream.get());
    cudaStreamSynchronize(stream.get());

    // Convert float -> int32 (TRT output is float32 cast of int indices)
    // Output is codebook-major: [enc_codebooks, enc_frames]
    // Convert to frame-major: [frame0_cb0, frame0_cb1, ..., frame1_cb0, ...]
    // so that codec_tokens[f * enc_codebooks + cb] is frame f, codebook cb
    std::vector<int32_t> tokens(output_elems);
    for (int32_t cb = 0; cb < enc_codebooks; ++cb)
    {
        for (int32_t f = 0; f < enc_frames; ++f)
        {
            auto src = static_cast<std::size_t>(cb) * enc_frames + f;
            auto dst = static_cast<std::size_t>(f) * enc_codebooks + cb;
            tokens[dst] = static_cast<int32_t>(std::round(host_tokens_f[src]));
        }
    }

    std::cerr << "[speech] Mimi encode (TRT): " << num_samples << " samples -> "
              << enc_frames << " frames x " << enc_codebooks
              << " codebooks" << std::endl;

    mLastEncodeFrames = enc_frames;
    mLastEncodeCodebooks = enc_codebooks;

    // Debug: print first few tokens
    std::cerr << "[speech] Encoder tokens [0:16]: ";
    for (int32_t i = 0; i < std::min(16, static_cast<int32_t>(tokens.size())); ++i)
        std::cerr << tokens[i] << " ";
    std::cerr << std::endl;

    return tokens;
}

// ---------------------------------------------------------------------------
// Stage 2: Temporal Transformer
// ---------------------------------------------------------------------------

std::vector<float> SpeechToSpeechBackend::run_temporal(
    const std::vector<int32_t>& codec_tokens, int32_t num_frames,
    int32_t num_codebooks)
{
    if (!mTemporalEngine)
    {
        throw std::runtime_error("Speech: temporal engine not available");
    }

    DeviceKvCache cache(*mTemporalEngine);
    DeviceResources resources(*mTemporalEngine);
    if (!cache.ok() || !resources.ok())
    {
        throw std::runtime_error("Speech: failed to allocate temporal resources");
    }

    // Check if the temporal engine has a hidden_state output tensor.
    // If so, we read the pre-LM-head hidden states for the depth transformer.
    // This is critical: the depth transformer needs the hidden representation,
    // not the text logits.
    const bool has_hidden_output = has_io_tensor(
        *mTemporalEngine->engine, "hidden_state");

    CudaBuffer d_hidden_state(0);
    const int32_t hidden = mTemporalEngine->hidden_size;

    if (has_hidden_output)
    {
        d_hidden_state = CudaBuffer(static_cast<std::size_t>(hidden) * sizeof(float));
        if (!d_hidden_state.ok())
        {
            throw std::runtime_error("Speech: failed to allocate hidden state buffer");
        }
        if (!mTemporalEngine->context->setTensorAddress(
                "hidden_state", d_hidden_state.data()))
        {
            std::cerr << "[speech] WARNING: failed to bind hidden_state tensor"
                      << std::endl;
        }
        else
        {
            std::cerr << "[speech] Temporal engine has hidden_state output "
                      << "(dim=" << hidden << ")" << std::endl;
        }
    }

    std::vector<float> logits;
    std::string error;

    // Feed codec tokens one frame at a time through the temporal transformer.
    // Moshi temporal input = SUM of per-codebook audio embeddings for each frame.
    // Tokens are in frame-major layout: [f0_cb0, f0_cb1, ..., f1_cb0, ...]
    //
    // If we have audio_embeddings in the config, compute the summed embedding
    // and pass via input_embed. Otherwise fall back to first codebook token only.
    const bool has_audio_emb = !mConfig.audio_embeddings.empty()
                                && mConfig.audio_vocab_size > 0;
    const bool has_input_embed = has_io_tensor(*mTemporalEngine->engine, "input_embed");

    if (has_audio_emb && has_input_embed)
    {
        std::cerr << "[speech] Temporal: using summed per-codebook audio embeddings"
                  << std::endl;
    }
    else if (has_audio_emb && !has_input_embed)
    {
        std::cerr << "[speech] WARNING: audio_embeddings available but engine has no "
                     "input_embed tensor; falling back to first codebook token"
                  << std::endl;
    }
    else
    {
        std::cerr << "[speech] WARNING: no audio_embeddings; using first codebook "
                     "token only (temporal input will be degraded)"
                  << std::endl;
    }

    std::vector<float> all_hidden;
    all_hidden.reserve(static_cast<std::size_t>(num_frames) * hidden);

    std::vector<float> frame_hidden(static_cast<std::size_t>(hidden));
    std::vector<float> summed_embed(static_cast<std::size_t>(hidden), 0.0F);

    for (int32_t frame = 0; frame < num_frames; ++frame)
    {
        if (has_audio_emb && has_input_embed)
        {
            // Compute summed embedding: sum emb[cb][token[frame, cb]] for all cb
            std::fill(summed_embed.begin(), summed_embed.end(), 0.0F);
            const int32_t audio_vocab = mConfig.audio_vocab_size;
            const auto emb_stride_cb = static_cast<std::size_t>(audio_vocab) * hidden;

            for (int32_t cb = 0; cb < num_codebooks; ++cb)
            {
                auto tok_idx = static_cast<std::size_t>(frame) * num_codebooks + cb;
                int32_t token = (tok_idx < codec_tokens.size())
                    ? codec_tokens[tok_idx] : 0;
                // Clamp to valid range
                token = std::max(0, std::min(token, audio_vocab - 1));

                // Lookup: audio_embeddings[cb * audio_vocab * hidden + token * hidden]
                auto emb_offset = static_cast<std::size_t>(cb) * emb_stride_cb
                                + static_cast<std::size_t>(token) * hidden;
                if (emb_offset + hidden <= mConfig.audio_embeddings.size())
                {
                    const float* emb_row = mConfig.audio_embeddings.data() + emb_offset;
                    for (int32_t d = 0; d < hidden; ++d)
                        summed_embed[d] += emb_row[d];
                }
            }

            // Pass summed embedding via input_embed (bypasses engine's embedding lookup)
            if (!run_decoder_step_device(*mTemporalEngine, cache, resources,
                    0 /* token_id unused */, logits, error,
                    summed_embed.data(), hidden, 1.0F))
            {
                throw std::runtime_error("Speech temporal step failed: " + error);
            }
        }
        else
        {
            // Fallback: use first codebook token as input
            auto tok_idx = static_cast<std::size_t>(frame) * num_codebooks;
            int32_t token = (tok_idx < codec_tokens.size())
                ? codec_tokens[tok_idx] : 0;

            if (!run_decoder_step_device(*mTemporalEngine, cache, resources,
                    token, logits, error))
            {
                throw std::runtime_error("Speech temporal step failed: " + error);
            }
        }

        if (has_hidden_output && d_hidden_state.ok())
        {
            // Read hidden state from GPU (the real pre-LM-head representation)
            cudaStreamSynchronize(resources.stream.get());
            cudaMemcpy(frame_hidden.data(), d_hidden_state.data(),
                       static_cast<std::size_t>(hidden) * sizeof(float),
                       cudaMemcpyDeviceToHost);
            all_hidden.insert(all_hidden.end(),
                              frame_hidden.begin(), frame_hidden.end());
        }
        else
        {
            // Fallback: use logits (first hidden elements)
            all_hidden.insert(all_hidden.end(), logits.begin(),
                              logits.begin() + std::min(
                                  static_cast<int32_t>(logits.size()), hidden));
            if (static_cast<int32_t>(logits.size()) < hidden)
            {
                all_hidden.resize(all_hidden.size() +
                    (hidden - static_cast<int32_t>(logits.size())), 0.0F);
            }
        }
    }

    std::cerr << "[speech] Temporal: processed " << num_frames << " frames"
              << (has_hidden_output ? " (hidden_state output)" : " (logits fallback)")
              << std::endl;
    return all_hidden;
}

// ---------------------------------------------------------------------------

std::vector<int32_t> SpeechToSpeechBackend::run_depth(
    const float* temporal_hidden, int32_t hidden_dim,
    int32_t text_token,
    const int32_t* forced_audio_tokens,
    const uint8_t* forced_audio_provided)
{
    const auto& cfg = mConfig;
    const int32_t num_cb = cfg.num_codebooks;
    const int32_t depth_hidden = cfg.depth_hidden_size;
    const int32_t temporal_hidden_dim = cfg.temporal_hidden_size;

    // Check if we have per-codebook engines
    const bool has_per_cb = !mDepthEngines.empty();
    if (!has_per_cb && !mDepthEngine)
    {
        return std::vector<int32_t>(num_cb, 0);
    }

    // Projection size per codebook: depth_hidden * temporal_hidden
    const auto proj_size_per_cb = static_cast<std::size_t>(depth_hidden) * temporal_hidden_dim;
    const bool has_proj = !cfg.depth_projection.empty()
                          && temporal_hidden != nullptr
                          && temporal_hidden_dim > 0 && depth_hidden > 0;

    // Resolve per-codebook engine pointers.
    // The depth sequence has num_cb + 1 positions:
    //   position 0: text token (uses engine cb=0)
    //   position 1: audio cb0 (uses engine cb=0)
    //   position 2: audio cb1 (uses engine cb=1)
    //   ...
    //   position N: audio cb{N-1} (uses engine cb={N-1})
    std::vector<DecoderStepEngine*> engines(static_cast<std::size_t>(num_cb));
    for (int32_t cb = 0; cb < num_cb; ++cb)
    {
        auto idx = static_cast<std::size_t>(cb);
        if (has_per_cb && idx < mDepthEngines.size() && mDepthEngines[idx])
            engines[idx] = mDepthEngines[idx].get();
        else if (mDepthEngine)
            engines[idx] = mDepthEngine.get();
        else
        {
            std::cerr << "[speech] No depth engine for codebook " << cb << std::endl;
            return std::vector<int32_t>(num_cb, 0);
        }
    }

    // Create shared KV cache and resources (shared across all depth steps).
    auto* first_engine = engines[0];
    auto shared_cache = std::make_unique<DeviceKvCache>(*first_engine);
    auto shared_resources = std::make_unique<DeviceResources>(*first_engine);
    if (!shared_cache->ok() || !shared_resources->ok())
    {
        return std::vector<int32_t>(num_cb, 0);
    }

    std::vector<int32_t> codebook_tokens;
    codebook_tokens.reserve(num_cb);
    const int32_t depth_call_idx = mDepthDebugCallCount++;

    std::vector<float> logits;
    std::string error;

    // Helper: compute depth_projection[proj_idx] @ temporal_hidden -> out[depth_hidden]
    auto apply_projection = [&](int32_t proj_idx, float* out)
    {
        if (!has_proj) return;
        auto proj_offset = static_cast<std::size_t>(proj_idx) * proj_size_per_cb;
        if (proj_offset + proj_size_per_cb > cfg.depth_projection.size()) return;
        const float* proj = cfg.depth_projection.data() + proj_offset;
        for (int32_t i = 0; i < depth_hidden; ++i)
        {
            float sum = 0.0F;
            const float* row = proj + static_cast<std::size_t>(i) * temporal_hidden_dim;
            for (int32_t j = 0; j < temporal_hidden_dim; ++j)
                sum += row[j] * temporal_hidden[j];
            out[i] += sum;  // ADD to existing embedding
        }
    };

    // Helper: look up text embedding: depth_text_embedding[token * depth_hidden]
    const bool has_text_emb = !cfg.depth_text_embedding.empty()
                               && cfg.depth_text_vocab > 0;

    // Helper: look up depth audio embedding: depth_audio_embeddings[emb_idx * vocab * hidden + token * hidden]
    const bool has_depth_audio_emb = !cfg.depth_audio_embeddings.empty()
                                      && cfg.num_depformer_emb > 0;
    const auto depth_audio_emb_stride = static_cast<std::size_t>(cfg.audio_vocab_size)
                                         * depth_hidden;

    std::vector<float> depth_embed(static_cast<std::size_t>(depth_hidden), 0.0F);

    // ---------------------------------------------------------------
    // Moshi/PersonaPlex depth transformer: num_cb autoregressive steps.
    //
    // The HF depth decoder generates num_codebooks audio tokens at positions
    // 0 through num_codebooks-1.  The input is a text token id; the output
    // sequence starts with this text id followed by num_codebooks generated
    // audio tokens.  The text input token is stripped from the final output.
    //
    //   Position 0: input = text_embed_tokens(text_token_id) + depformer_in.0(temporal_hidden)
    //               -> codebook 0 attention/MLP weights -> lm_heads[0] -> audio token cb0
    //   Position k (k>0): input = depformer_emb.{k-1}(prev_audio_token) + depformer_in.{k}(temporal_hidden)
    //               -> codebook k attention/MLP weights -> lm_heads[k] -> audio token cbk
    //
    // Key: input_projections (depformer_in) is applied at EVERY step, selecting
    // a different projection matrix per position. The token embedding and
    // projection are SUMMED to form the input.
    // ---------------------------------------------------------------

    int32_t prev_token = 0;  // not used for position 0 (text embed instead)

    for (int32_t cb = 0; cb < num_cb; ++cb)
    {
        auto* engine = engines[static_cast<std::size_t>(cb)];

        std::fill(depth_embed.begin(), depth_embed.end(), 0.0F);

        if (cb == 0)
        {
            // Position 0: text token embedding + projection[0]
            // Use depformer_text_emb(sampled_text_token) as the input embedding.
            // In the official code, the text token is sampled from temporal text_logits.
            if (has_text_emb)
            {
                int32_t ttok = std::max(0, std::min(text_token,
                    static_cast<int32_t>(cfg.depth_text_vocab) - 1));
                auto emb_offset = static_cast<std::size_t>(ttok) * depth_hidden;
                if (emb_offset + depth_hidden <= cfg.depth_text_embedding.size())
                {
                    const float* row = cfg.depth_text_embedding.data() + emb_offset;
                    for (int32_t d = 0; d < depth_hidden; ++d)
                        depth_embed[d] = row[d];
                }
            }
            // Add projection[0]
            apply_projection(0, depth_embed.data());
        }
        else
        {
            // Position cb (cb > 0): depformer_emb.{cb-1}(prev_token) + projection[cb]
            if (has_depth_audio_emb && (cb - 1) < cfg.num_depformer_emb)
            {
                int32_t tok = std::max(0, std::min(prev_token, cfg.audio_vocab_size - 1));
                auto emb_idx = static_cast<std::size_t>(cb - 1);
                auto emb_offset = emb_idx * depth_audio_emb_stride
                                + static_cast<std::size_t>(tok) * depth_hidden;
                if (emb_offset + depth_hidden <= cfg.depth_audio_embeddings.size())
                {
                    const float* row = cfg.depth_audio_embeddings.data() + emb_offset;
                    for (int32_t d = 0; d < depth_hidden; ++d)
                        depth_embed[d] = row[d];
                }
            }
            // Add projection[cb]
            apply_projection(cb, depth_embed.data());
        }

        // Run with input_embed (always bypass engine embedding for correct behavior)
        if (!run_decoder_step_device(*engine, *shared_cache, *shared_resources,
                0, logits, error,
                depth_embed.data(), depth_hidden, 1.0F))
        {
            std::cerr << "[speech] Depth step cb=" << cb << " failed: "
                      << error << std::endl;
            break;
        }

        int32_t best;
        if (cfg.depth_temperature > 0.0F && cfg.depth_top_k > 0)
        {
            best = sample_token_topk(logits, cfg.depth_temperature,
                                     cfg.depth_top_k, mRngState);
        }
        else
        {
            best = select_argmax_token(logits);
        }
        best = std::max(0, std::min(best, cfg.codebook_size - 1));
        codebook_tokens.push_back(best);
        if (forced_audio_tokens != nullptr && forced_audio_provided != nullptr &&
            forced_audio_provided[cb])
        {
            int32_t forced = forced_audio_tokens[cb];
            forced = std::max(0, std::min(forced, cfg.audio_vocab_size - 1));
            prev_token = forced;
        }
        else
        {
            prev_token = best;
        }

        if (cb == 0 && depth_call_idx < 12 && !logits.empty())
        {
            // Debug: inspect early cb0 ranking/margins.
            int32_t top1 = -1;
            int32_t top2 = -1;
            float v1 = -1.0e30F;
            float v2 = -1.0e30F;
            for (int32_t i = 0; i < static_cast<int32_t>(logits.size()); ++i)
            {
                const float v = logits[static_cast<std::size_t>(i)];
                if (v > v1)
                {
                    v2 = v1;
                    top2 = top1;
                    v1 = v;
                    top1 = i;
                }
                else if (v > v2)
                {
                    v2 = v;
                    top2 = i;
                }
            }
            std::cerr << "[speech] DepthDbg frame=" << depth_call_idx
                      << " cb0 top1=" << top1 << " v1=" << v1
                      << " top2=" << top2 << " v2=" << v2
                      << " margin=" << (v1 - v2)
                      << " sampled=" << best << std::endl;
        }
    }

    // Pad if we stopped early
    while (static_cast<int32_t>(codebook_tokens.size()) < num_cb)
        codebook_tokens.push_back(0);

    return codebook_tokens;
}

// ---------------------------------------------------------------------------
// Stage 4: Mimi Decode (codec tokens -> waveform)
// ---------------------------------------------------------------------------

std::vector<float> SpeechToSpeechBackend::run_mimi_decode(
    const std::vector<int32_t>& codec_tokens, int32_t num_frames)
{
    if (num_frames <= 0)
    {
        return {};
    }

    // Determine actual codebooks in input tokens (frame-major layout)
    int32_t actual_codebooks = 0;
    if (!codec_tokens.empty())
        actual_codebooks = static_cast<int32_t>(codec_tokens.size()) / num_frames;

    if (!mMimiDecoderEngine || !mMimiDecoderCtx)
    {
        std::cerr << "[speech] No Mimi TRT decoder available" << std::endl;
        return {};
    }

    // Query actual TRT engine I/O shapes
    auto in_dims = mMimiDecoderEngine->getTensorShape("codec_tokens");
    auto out_dims = mMimiDecoderEngine->getTensorShape("audio_output");

    // Decoder expects codec_tokens [dec_codebooks, dec_frames]
    const int32_t dec_codebooks = in_dims.d[0];
    const int32_t dec_frames = in_dims.d[1];

    int32_t total_output_elems = 1;
    for (int32_t d = 0; d < out_dims.nbDims; ++d)
        total_output_elems *= out_dims.d[d];

    std::cerr << "[speech] Mimi decoder TRT: input [" << dec_codebooks
              << "," << dec_frames << "], output " << total_output_elems
              << " samples" << std::endl;

    // Build the decoder input: codebook-major [dec_codebooks, dec_frames]
    // Input tokens are frame-major: [f0_cb0, f0_cb1, ..., f1_cb0, ...]
    const auto input_elems = static_cast<std::size_t>(dec_codebooks) * dec_frames;
    std::vector<float> input_tokens(input_elems, 0.0F);

    int32_t frames_to_copy = std::min(num_frames, dec_frames);
    int32_t cbs_to_copy = std::min(actual_codebooks, dec_codebooks);

    for (int32_t f = 0; f < frames_to_copy; ++f)
    {
        for (int32_t cb = 0; cb < cbs_to_copy; ++cb)
        {
            auto src_idx = static_cast<std::size_t>(f) * actual_codebooks + cb;
            auto dst_idx = static_cast<std::size_t>(cb) * dec_frames + f;
            if (src_idx < codec_tokens.size())
            {
                input_tokens[dst_idx] = static_cast<float>(codec_tokens[src_idx]);
            }
        }
    }

    const auto input_bytes = input_elems * sizeof(float);
    const auto output_bytes = static_cast<std::size_t>(total_output_elems) * sizeof(float);

    CudaBuffer d_input(input_bytes);
    CudaBuffer d_output(output_bytes);
    CudaStream stream;

    if (!d_input.ok() || !d_output.ok() || !stream.ok())
    {
        std::cerr << "[speech] Failed to allocate CUDA resources for Mimi decoder"
                  << std::endl;
        return {};
    }

    // Debug: print first few input tokens
    std::cerr << "[speech] Decoder input tokens [0:16]: ";
    for (int32_t i = 0; i < std::min(16, static_cast<int32_t>(input_elems)); ++i)
        std::cerr << input_tokens[i] << " ";
    std::cerr << std::endl;

    cudaMemcpyAsync(d_input.data(), input_tokens.data(), input_bytes,
                    cudaMemcpyHostToDevice, stream.get());

    mMimiDecoderCtx->setTensorAddress("codec_tokens", d_input.data());
    mMimiDecoderCtx->setTensorAddress("audio_output", d_output.data());

    if (!mMimiDecoderCtx->enqueueV3(stream.get()))
    {
        std::cerr << "[speech] Mimi decoder TRT execution failed" << std::endl;
        return {};
    }

    std::vector<float> waveform(static_cast<std::size_t>(total_output_elems));
    cudaMemcpyAsync(waveform.data(), d_output.data(), output_bytes,
                    cudaMemcpyDeviceToHost, stream.get());
    cudaStreamSynchronize(stream.get());

    // Debug: check output statistics
    float rms = 0.0F, mx = 0.0F;
    for (auto s : waveform)
    {
        rms += s * s;
        mx = std::max(mx, std::abs(s));
    }
    rms = std::sqrt(rms / std::max(1, total_output_elems));
    std::cerr << "[speech] Mimi decode (TRT): " << dec_frames << " frames -> "
              << total_output_elems << " samples (RMS=" << rms
              << ", Max=" << mx << ")" << std::endl;
    return waveform;
}

// ---------------------------------------------------------------------------
// Full pipeline
// ---------------------------------------------------------------------------

AudioResult SpeechToSpeechBackend::process_audio(
    const float* input_samples, int32_t num_input_samples,
    int32_t max_output_frames,
    int32_t input_sample_rate,
    int32_t tail_frames)
{
    AudioResult result;
    result.sample_rate = mConfig.sample_rate;

    if (!is_available())
    {
        std::cerr << "[speech] Backend not available" << std::endl;
        return result;
    }

    mDepthDebugCallCount = 0;

    std::cerr << "[speech] Starting pipeline with " << num_input_samples
              << " input samples" << std::endl;

    if (mConfig.depth_temperature > 0.0F && mConfig.depth_top_k > 0)
    {
        std::cerr << "[speech] Depth sampling: temperature="
                  << mConfig.depth_temperature << " top_k="
                  << mConfig.depth_top_k << std::endl;
    }
    else
    {
        std::cerr << "[speech] Depth decoding: greedy (argmax)" << std::endl;
    }

    // ---------------------------------------------------------------
    // Stage 1: Encode input audio via Mimi
    // ---------------------------------------------------------------
    auto codec_tokens = run_mimi_encode(
        input_samples, num_input_samples, input_sample_rate);

    // Mimi outputs 32 codebooks; we only use the first stream_cb for the user
    // stream.  PersonaPlex uses num_codebooks=16 total (8 moshi + 8 user).
    int32_t encode_codebooks = mLastEncodeCodebooks;
    int32_t num_frames = mLastEncodeFrames;
    if (encode_codebooks <= 0 || num_frames <= 0 ||
        static_cast<std::size_t>(encode_codebooks) *
            static_cast<std::size_t>(num_frames) != codec_tokens.size())
    {
        // Fallback if shape metadata is unavailable.
        if (mMimiEncoderEngine)
        {
            auto enc_out_dims = mMimiEncoderEngine->getTensorShape("codec_tokens");
            encode_codebooks = enc_out_dims.d[0];
        }
        else
        {
            encode_codebooks = mConfig.num_codebooks;
        }
        num_frames = (encode_codebooks > 0 && !codec_tokens.empty())
            ? static_cast<int32_t>(codec_tokens.size()) / encode_codebooks
            : 0;
    }

    std::cerr << "[speech] Encoder output: " << codec_tokens.size()
              << " tokens = " << num_frames << " frames x "
              << encode_codebooks << " codebooks" << std::endl;

    if (num_frames <= 0)
    {
        std::cerr << "[speech] Encoder produced no frames" << std::endl;
        return result;
    }

    // ---------------------------------------------------------------
    // Dual-stream architecture (Moshi/PersonaPlex):
    //
    // The temporal transformer receives 16 codebook embeddings per frame:
    //   cb 0..7  = "moshi" stream (model's own audio output)
    //   cb 8..15 = "user" stream (user's input audio)
    //
    // Prefill: moshi = BOS (2048), user = Mimi encoder output (first 8 cb)
    // Generation: moshi = depth output from prev frame, user = BOS (blank)
    //
    // The depth transformer generates 16 codebook tokens per frame.
    // Only the first 8 (moshi stream) are fed to the Mimi decoder.
    // ---------------------------------------------------------------
    const int32_t num_cb = mConfig.num_codebooks;  // 16
    const int32_t stream_cb = num_cb / 2;           // 8 per stream
    const int32_t bos_token = mConfig.codebook_size; // 2048 = BOS/pad
    const int32_t hidden = mTemporalEngine->hidden_size;
    const bool has_audio_emb = !mConfig.audio_embeddings.empty()
                                && mConfig.audio_vocab_size > 0;
    const bool has_input_embed = has_io_tensor(*mTemporalEngine->engine, "input_embed");
    const bool has_hidden_output = has_io_tensor(
        *mTemporalEngine->engine, "hidden_state");

    if (!has_audio_emb || !has_input_embed)
    {
        std::cerr << "[speech] ERROR: dual-stream requires audio_embeddings "
                     "and input_embed support" << std::endl;
        return result;
    }

    // ---------------------------------------------------------------
    // Stage 2 & 3: Temporal + Depth in lockstep
    //
    // Phase A: Prefill — process user input frames through temporal
    // Phase B: Generate — autoregressive temporal+depth loop
    // ---------------------------------------------------------------
    DeviceKvCache temporal_cache(*mTemporalEngine);
    DeviceResources temporal_resources(*mTemporalEngine);
    if (!temporal_cache.ok() || !temporal_resources.ok())
    {
        throw std::runtime_error("Speech: failed to allocate temporal resources");
    }

    CudaBuffer d_hidden_state(0);
    if (has_hidden_output)
    {
        d_hidden_state = CudaBuffer(static_cast<std::size_t>(hidden) * sizeof(float));
        if (d_hidden_state.ok())
        {
            mTemporalEngine->context->setTensorAddress(
                "hidden_state", d_hidden_state.data());
        }
    }

    // ---------------------------------------------------------------
    // Text prompt injection (primes temporal KV cache with system prompt)
    // Uses pre-tokenized IDs from bundle config (avoids runtime tokenization)
    // ---------------------------------------------------------------
    if (!mConfig.text_prompt_ids.empty())
    {
        run_text_prompt(temporal_cache, temporal_resources, d_hidden_state);
    }
    else if (!mConfig.system_prompt.empty() && !mConfig.hf_python.empty())
    {
        run_text_prompt(temporal_cache, temporal_resources, d_hidden_state);
    }

    const int32_t audio_vocab = mConfig.audio_vocab_size;
    const auto emb_stride_cb =
        static_cast<std::size_t>(audio_vocab) * hidden;

    // Text embedding for temporal input (official Moshi adds text_emb at each step)
    const bool has_temporal_text_emb = !mConfig.temporal_text_embedding.empty()
                                       && mConfig.temporal_text_vocab > 0;
    const int32_t text_pad_id = mConfig.text_padding_id;

    // Helper: compute summed embedding for 16 codebooks + text token
    // Official Moshi: input_ = text_emb(text_token) + sum(emb[cb](audio_token[cb]))
    auto compute_summed_embed = [&](
        const int32_t* moshi_tokens,
        const int32_t* user_tokens,
        int32_t text_token,
        float* out_embed)
    {
        std::fill(out_embed, out_embed + hidden, 0.0F);

        // Add text embedding (official: text_emb(text_token))
        if (has_temporal_text_emb)
        {
            int32_t ttok = std::max(0, std::min(text_token,
                mConfig.temporal_text_vocab - 1));
            auto text_offset = static_cast<std::size_t>(ttok) * hidden;
            if (text_offset + hidden <= mConfig.temporal_text_embedding.size())
            {
                const float* row = mConfig.temporal_text_embedding.data() + text_offset;
                for (int32_t d = 0; d < hidden; ++d)
                    out_embed[d] += row[d];
            }
        }

        // cb 0..stream_cb-1: moshi stream -> emb[0..7]
        for (int32_t cb = 0; cb < stream_cb; ++cb)
        {
            int32_t tok = std::max(0, std::min(moshi_tokens[cb], audio_vocab - 1));
            auto emb_offset = static_cast<std::size_t>(cb) * emb_stride_cb
                            + static_cast<std::size_t>(tok) * hidden;
            if (emb_offset + hidden <= mConfig.audio_embeddings.size())
            {
                const float* row = mConfig.audio_embeddings.data() + emb_offset;
                for (int32_t d = 0; d < hidden; ++d)
                    out_embed[d] += row[d];
            }
        }
        // cb stream_cb..num_cb-1: user stream -> emb[8..15]
        for (int32_t cb = 0; cb < stream_cb; ++cb)
        {
            int32_t tok = std::max(0, std::min(user_tokens[cb], audio_vocab - 1));
            auto emb_idx = static_cast<std::size_t>(cb + stream_cb);
            auto emb_offset = emb_idx * emb_stride_cb
                            + static_cast<std::size_t>(tok) * hidden;
            if (emb_offset + hidden <= mConfig.audio_embeddings.size())
            {
                const float* row = mConfig.audio_embeddings.data() + emb_offset;
                for (int32_t d = 0; d < hidden; ++d)
                    out_embed[d] += row[d];
            }
        }
    };

    std::vector<float> summed_embed(static_cast<std::size_t>(hidden));
    std::vector<float> frame_hidden(static_cast<std::size_t>(hidden));
    std::vector<float> logits;
    std::string error;

    // ---------------------------------------------------------------
    // Delay pattern implementation (matches official PersonaPlex/Moshi).
    //
    // The official code maintains a circular delay cache. At each frame:
    //   - Write new tokens at position (offset + delay[k])
    //   - Read model input from position (offset - 1)
    //   - For delay=0: token is read 1 step after writing (current frame's data)
    //   - For delay=1: token is read 2 steps after writing (previous frame's data)
    //
    // delays = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    //   k=0: text       delay=0
    //   k=1: moshi cb0  delay=0
    //   k=2-8: moshi cb1-7  delay=1
    //   k=9: user cb0   delay=0
    //   k=10-16: user cb1-7  delay=1
    //
    // The initial token for all codebooks at the start is:
    //   text: text_initial_token_id (32000)
    //   audio: audio_initial_token_id (2048 = BOS)
    // ---------------------------------------------------------------

    // Setup delay pattern
    const int32_t total_k = num_cb + 1;  // 17 = 1 text + 16 audio
    std::vector<int32_t> delays(static_cast<std::size_t>(total_k));
    if (!mConfig.delays.empty() &&
        static_cast<int32_t>(mConfig.delays.size()) >= total_k)
    {
        for (int32_t k = 0; k < total_k; ++k)
            delays[k] = mConfig.delays[k];
    }
    else
    {
        // Default PersonaPlex delays
        delays = {0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1};
    }

    int32_t max_delay = 0;
    for (auto d : delays) max_delay = std::max(max_delay, d);

    // Circular delay cache: [total_k, cache_size]
    // Cache size = max_delay + 3 (matching official code)
    const int32_t cache_size = max_delay + 3;
    const int32_t ungenerated = -2;
    std::vector<int32_t> delay_cache(
        static_cast<std::size_t>(total_k) * cache_size, ungenerated);
    std::vector<uint8_t> delay_provided(
        static_cast<std::size_t>(total_k) * cache_size, 0);

    // Initial tokens
    const int32_t text_bos = mConfig.text_initial_token_id;   // 32000
    const int32_t audio_bos = mConfig.audio_initial_token_id;  // 2048

    // Helper: read/write delay cache at [k, pos]
    auto dc_idx = [&](int32_t k, int32_t pos) -> std::size_t {
        return static_cast<std::size_t>(k) * cache_size + (pos % cache_size);
    };

    // ---------------------------------------------------------------
    // Interleaved temporal + depth loop with delay pattern.
    //
    // For each frame (offset):
    //   1. Write new tokens into delay cache at (offset + delay[k])
    //   2. If offset == 0: write initial tokens and skip to next
    //   3. Read model input from cache at (offset - 1)
    //   4. Compute embedding, run temporal, run depth
    //   5. If offset > max_delay: collect output from (offset - max_delay)
    //   6. Increment offset
    // ---------------------------------------------------------------
    // The streaming reference path does not emit one final frame for each
    // encoded chunk; initial/final delay warmup consumes ~2 frames.
    // Also bound by nominal input frame count from audio duration.
    const int32_t nominal_frame_size = (mConfig.frame_rate > 0.0F)
        ? static_cast<int32_t>(std::lround(
            static_cast<float>(mConfig.sample_rate) / mConfig.frame_rate))
        : 0;
    int32_t nominal_input_frames = num_frames;
    if (nominal_frame_size > 0)
    {
        int64_t effective_input_samples = num_input_samples;
        if (input_sample_rate > 0 && input_sample_rate != mConfig.sample_rate)
        {
            effective_input_samples =
                (effective_input_samples * mConfig.sample_rate) / input_sample_rate;
        }
        nominal_input_frames = static_cast<int32_t>(
            effective_input_samples / nominal_frame_size);
    }
    int32_t effective_frames = std::min(num_frames, nominal_input_frames);
    effective_frames = std::max(0, effective_frames - 2);
    const int32_t extra_tail = std::max(0, tail_frames);
    int64_t target_frames = static_cast<int64_t>(effective_frames)
        + static_cast<int64_t>(extra_tail);
    target_frames = std::max<int64_t>(0, target_frames);
    const int32_t output_frames = std::min(
        static_cast<int32_t>(std::min<int64_t>(
            target_frames, static_cast<int64_t>(std::numeric_limits<int32_t>::max()))),
        max_output_frames);
    const int32_t mimi_cb = mConfig.mimi_decode_codebooks;
    std::vector<int32_t> output_codes;
    output_codes.reserve(
        static_cast<std::size_t>(mimi_cb) * output_frames);

    // We need to run (output_frames + max_delay + 1) iterations to produce
    // output_frames of output after the delay warmup
    const int32_t total_iters = output_frames + max_delay + 1;

    std::cerr << "[speech] Interleaved temporal+depth with delay pattern: "
              << output_frames << " output frames, " << total_iters
              << " total iterations (max_delay=" << max_delay
              << ", input_effective=" << effective_frames
              << ", tail_frames=" << extra_tail << ")"
              << std::endl;

    int32_t frames_collected = 0;

    for (int32_t offset = 0; offset < total_iters && frames_collected < output_frames; ++offset)
    {
        // User stream tokens are forced input (provided=true) with delays.
        // k = 9..16 in [text + 16 audio] layout.
        for (int32_t cb = 0; cb < stream_cb; ++cb)
        {
            int32_t k = stream_cb + 1 + cb;
            int32_t user_tok = audio_bos;
            if (offset < num_frames)
            {
                auto tok_idx = static_cast<std::size_t>(offset) * encode_codebooks + cb;
                if (tok_idx < codec_tokens.size())
                    user_tok = codec_tokens[tok_idx];
            }
            auto widx = dc_idx(k, offset + delays[k]);
            delay_cache[widx] = user_tok;
            delay_provided[widx] = 1;
        }

        // Initial fill for delayed streams at the beginning.
        for (int32_t k = 0; k < total_k; ++k)
        {
            if (offset <= delays[k])
            {
                int32_t init_tok = (k == 0) ? text_bos : audio_bos;
                auto idx = dc_idx(k, offset);
                delay_cache[idx] = init_tok;
                delay_provided[idx] = 1;
            }
        }

        // Offset 0: initialize cache row and skip generation.
        if (offset == 0)
        {
            for (int32_t k = 0; k < total_k; ++k)
            {
                delay_cache[dc_idx(k, 0)] = (k == 0) ? text_bos : audio_bos;
            }
            continue;
        }

        // Read model input from position (offset - 1).
        const int32_t model_input_pos = offset - 1;
        const int32_t target_pos = offset;

        int32_t text_input = delay_cache[dc_idx(0, model_input_pos)];
        std::vector<int32_t> moshi_input(static_cast<std::size_t>(stream_cb));
        std::vector<int32_t> user_input(static_cast<std::size_t>(stream_cb));

        for (int32_t cb = 0; cb < stream_cb; ++cb)
        {
            moshi_input[cb] = delay_cache[dc_idx(1 + cb, model_input_pos)];
            user_input[cb] = delay_cache[dc_idx(stream_cb + 1 + cb, model_input_pos)];
        }

        // Compute temporal input embedding from delayed cache.
        compute_summed_embed(moshi_input.data(), user_input.data(),
                             text_input, summed_embed.data());

        if (!run_decoder_step_device(*mTemporalEngine, temporal_cache,
                temporal_resources, 0, logits, error,
                summed_embed.data(), hidden, 1.0F))
        {
            throw std::runtime_error("Speech temporal step failed: " + error);
        }

        if (has_hidden_output && d_hidden_state.ok())
        {
            cudaStreamSynchronize(temporal_resources.stream.get());
            cudaMemcpy(frame_hidden.data(), d_hidden_state.data(),
                       static_cast<std::size_t>(hidden) * sizeof(float),
                       cudaMemcpyDeviceToHost);
        }
        else
        {
            for (int32_t d = 0; d < hidden; ++d)
            {
                frame_hidden[d] = (d < static_cast<int32_t>(logits.size()))
                    ? logits[d] : 0.0F;
            }
        }

        // Sample text token from temporal logits.
        // Match official LMGen behavior:
        //   - greedy mode: argmax
        //   - sampling mode: text_temp=0.7, text_top_k=25
        int32_t sampled_text_token = text_pad_id;
        if (!logits.empty())
        {
            if (mConfig.depth_temperature > 0.0F && mConfig.depth_top_k > 0)
            {
                constexpr float kTextTemp = 0.7F;
                constexpr int32_t kTextTopK = 25;
                sampled_text_token = sample_token_topk(
                    logits, kTextTemp, kTextTopK, mRngState);
            }
            else
            {
                sampled_text_token = select_argmax_token(logits);
            }
        }

        // If target text at this offset is forced/provided, use it for depformer.
        auto text_target_idx = dc_idx(0, target_pos);
        bool text_provided = delay_provided[text_target_idx] != 0;
        int32_t next_text_token = text_provided
            ? delay_cache[text_target_idx] : sampled_text_token;

        // Build forced-audio arrays from target position (official depformer behavior).
        std::vector<int32_t> target_audio_tokens(static_cast<std::size_t>(num_cb), audio_bos);
        std::vector<uint8_t> target_audio_provided(static_cast<std::size_t>(num_cb), 0);
        for (int32_t cb = 0; cb < num_cb; ++cb)
        {
            auto idx = dc_idx(1 + cb, target_pos);
            target_audio_tokens[static_cast<std::size_t>(cb)] = delay_cache[idx];
            target_audio_provided[static_cast<std::size_t>(cb)] =
                static_cast<uint8_t>(delay_provided[idx] != 0);
        }

        auto frame_codes = run_depth(
            frame_hidden.data(), hidden, next_text_token,
            target_audio_tokens.data(), target_audio_provided.data());

        // Clear provided flags at model_input position after use.
        for (int32_t k = 0; k < total_k; ++k)
        {
            delay_provided[dc_idx(k, model_input_pos)] = 0;
        }

        // Write generated tokens at target position where not provided.
        if (!text_provided)
            delay_cache[text_target_idx] = sampled_text_token;
        for (int32_t cb = 0; cb < std::min(static_cast<int32_t>(frame_codes.size()),
                                            num_cb); ++cb)
        {
            auto idx = dc_idx(1 + cb, target_pos);
            if (delay_provided[idx] == 0)
                delay_cache[idx] = frame_codes[cb];
        }

        // Collect delay-compensated output tokens.
        if (offset > max_delay)
        {
            int32_t out_pos = offset - max_delay;
            for (int32_t cb = 0; cb < mimi_cb; ++cb)
            {
                int32_t k = 1 + cb;  // audio codebook k in delay cache
                int32_t gather_pos = out_pos + delays[k];
                int32_t tok = delay_cache[dc_idx(k, gather_pos)];
                output_codes.push_back(tok);
            }
            frames_collected++;
        }

        // Debug: print first 5 active frames
        if (offset > 0 && offset <= 5)
        {
            float l2 = 0.0F;
            for (int32_t d = 0; d < hidden; ++d)
                l2 += frame_hidden[d] * frame_hidden[d];
            l2 = std::sqrt(l2);
            std::cerr << "[speech] Offset " << offset << " hidden L2=" << l2
                      << " text_in=" << text_input
                      << " text_out=" << sampled_text_token << " depth:";
            for (int32_t cb = 0; cb < std::min(4,
                    static_cast<int32_t>(frame_codes.size())); ++cb)
                std::cerr << " " << frame_codes[cb];
            std::cerr << "..." << std::endl;
        }
    }

    std::cerr << "[speech] Depth: generated " << output_frames
              << " frames x " << num_cb << " codebooks (decoding first "
              << mimi_cb << ")" << std::endl;
    if (!output_codes.empty())
    {
        const int32_t dbg_frames = output_frames;
        for (int32_t f = 0; f < dbg_frames; ++f)
        {
            std::cerr << "[speech] Output frame " << f << ":";
            for (int32_t cb = 0; cb < mimi_cb; ++cb)
            {
                auto idx = static_cast<std::size_t>(f) * mimi_cb + cb;
                if (idx < output_codes.size())
                    std::cerr << " " << output_codes[idx];
            }
            std::cerr << std::endl;
        }
    }

    // ---------------------------------------------------------------
    // Stage 4: Decode first mimi_cb codebook tokens to audio via Mimi decoder
    // ---------------------------------------------------------------
    auto waveform = run_mimi_decode(output_codes, output_frames);

    // Peak-normalize to [-1, 1] to avoid clipping in WAV output
    if (!waveform.empty())
    {
        float peak = 0.0F;
        for (auto s : waveform)
            peak = std::max(peak, std::abs(s));
        if (peak > 1.0F)
        {
            float scale = 0.95F / peak;  // leave small headroom
            for (auto& s : waveform)
                s *= scale;
            std::cerr << "[speech] Peak-normalized: peak=" << peak
                      << " scale=" << scale << std::endl;
        }
    }

    result.waveform = std::move(waveform);
    result.num_samples = static_cast<int32_t>(result.waveform.size());
    std::cerr << "[speech] Generated " << result.num_samples << " samples ("
              << static_cast<float>(result.num_samples) / result.sample_rate
              << "s @ " << result.sample_rate << " Hz)" << std::endl;

    return result;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<SpeechToSpeechBackend> CreateSpeechBackend(
    std::unique_ptr<DecoderStepEngine> temporal_engine,
    const FastPathModelConfig& cfg,
    const std::string& hf_python)
{
    SpeechConfig speech_cfg;
    speech_cfg.sample_rate = cfg.audio_sample_rate;
    speech_cfg.temporal_hidden_size = cfg.hidden_size;
    speech_cfg.temporal_num_layers = cfg.num_layers;

    // These fields are parsed from the speech_to_speech config block
    speech_cfg.num_codebooks = cfg.codec_n_codebooks;
    speech_cfg.codebook_size = cfg.codebook_size;
    speech_cfg.depth_num_layers = cfg.fine_num_layers;  // reuse fine fields
    speech_cfg.depth_hidden_size = cfg.fine_hidden_size;
    speech_cfg.depth_num_heads = cfg.fine_num_heads;

    // Delay pattern and initial tokens
    speech_cfg.delays = cfg.speech_delays;
    speech_cfg.text_initial_token_id = cfg.speech_text_initial_token_id;
    speech_cfg.audio_initial_token_id = cfg.speech_audio_initial_token_id;
    speech_cfg.text_padding_id = cfg.speech_text_padding_id;

    // Sampling parameters
    speech_cfg.depth_temperature = cfg.speech_depth_temperature;
    speech_cfg.depth_top_k = cfg.speech_depth_top_k;

    // System prompt
    speech_cfg.system_prompt = cfg.speech_system_prompt;
    speech_cfg.text_prompt_ids = cfg.speech_text_prompt_ids;

    // Optional Python path for runtime text tokenization fallback.
    speech_cfg.hf_python = hf_python;

    return std::make_unique<SpeechToSpeechBackend>(
        std::move(temporal_engine), std::move(speech_cfg));
}

} // namespace trtf

#endif // TRTF_HAS_TRT
