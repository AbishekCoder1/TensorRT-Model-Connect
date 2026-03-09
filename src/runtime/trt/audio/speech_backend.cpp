#include "runtime/trt/audio/speech_backend.h"
#include "runtime/trt/audio/speech_depth_plan.h"
#include "runtime/trt/audio/speech_delay_cache.h"
#include "runtime/trt/audio/speech_generation_policy.h"
#include "runtime/trt/audio/speech_mimi_decode_plan.h"
#include "runtime/trt/audio/speech_runtime_plan.h"
#include "runtime/trt/audio/speech_temporal_embed_plan.h"
#include "runtime/trt/audio/speech_waveform_postprocess.h"
#include "trtf/runtime/trt/audio/speech_decode_stop_policy.h"

#if TRTF_HAS_TRT

#include "runtime/trt/core/trt_decode_runtime.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
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
    SpeechConfig config,
    std::shared_ptr<ISubprocessRunner> subprocess_runner)
    : mTemporalEngine(std::move(temporal_engine))
    , mConfig(std::move(config))
    , mSubprocessRunner(std::move(subprocess_runner))
{
    if (!mSubprocessRunner)
    {
        mSubprocessRunner = CreateDefaultSubprocessRunner();
    }

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

namespace {

void close_fd_if_open(int fd)
{
    if (fd >= 0)
        close(fd);
}

void close_pipe_pair(int pipe_fds[2])
{
    close_fd_if_open(pipe_fds[0]);
    close_fd_if_open(pipe_fds[1]);
    pipe_fds[0] = -1;
    pipe_fds[1] = -1;
}

bool create_pipe_pair(int pipe_fds[2], std::string& error)
{
    if (pipe(pipe_fds) == 0)
        return true;
    error = "pipe() failed";
    return false;
}

bool create_subprocess_pipes(
    int stdin_pipe[2], int stdout_pipe[2], int stderr_pipe[2],
    std::string& error)
{
    if (!create_pipe_pair(stdin_pipe, error))
        return false;
    if (!create_pipe_pair(stdout_pipe, error))
    {
        close_pipe_pair(stdin_pipe);
        return false;
    }
    if (!create_pipe_pair(stderr_pipe, error))
    {
        close_pipe_pair(stdin_pipe);
        close_pipe_pair(stdout_pipe);
        return false;
    }
    return true;
}

void write_fd_all(int fd, const void* input_data, std::size_t input_size)
{
    if (input_data == nullptr || input_size == 0)
        return;
    const auto* p = static_cast<const char*>(input_data);
    std::size_t remaining = input_size;
    while (remaining > 0)
    {
        auto written = write(fd, p, remaining);
        if (written <= 0)
            break;
        p += written;
        remaining -= static_cast<std::size_t>(written);
    }
}

void read_fd_to_vector(int fd, std::vector<char>& out)
{
    out.clear();
    char buf[65536];
    for (;;)
    {
        auto n = read(fd, buf, sizeof(buf));
        if (n <= 0)
            break;
        out.insert(out.end(), buf, buf + n);
    }
}

void read_fd_to_string(int fd, std::string& out)
{
    out.clear();
    char buf[65536];
    for (;;)
    {
        auto n = read(fd, buf, sizeof(buf));
        if (n <= 0)
            break;
        out.append(buf, static_cast<std::size_t>(n));
    }
}

int32_t clamp_token(int32_t token, int32_t vocab_size)
{
    return clamp_speech_depth_token(token, vocab_size);
}

void read_hidden_from_device(
    const CudaBuffer& d_hidden_state,
    DeviceResources& resources,
    std::vector<float>& frame_hidden,
    int32_t hidden_size)
{
    cudaStreamSynchronize(resources.stream.get());
    cudaMemcpy(frame_hidden.data(), d_hidden_state.data(),
               static_cast<std::size_t>(hidden_size) * sizeof(float),
               cudaMemcpyDeviceToHost);
}

bool is_sampling_enabled(const SpeechConfig& cfg)
{
    return cfg.depth_temperature > 0.0F && cfg.depth_top_k > 0;
}

int32_t select_depth_token_greedy(
    const std::vector<float>& logits,
    const SpeechConfig&,
    uint64_t&)
{
    return select_argmax_token(logits);
}

int32_t select_depth_token_sampled(
    const std::vector<float>& logits,
    const SpeechConfig& cfg,
    uint64_t& rng_state)
{
    return sample_token_topk(
        logits, cfg.depth_temperature, cfg.depth_top_k, rng_state);
}

using DepthTokenSelectFn = int32_t (*)(
    const std::vector<float>&, const SpeechConfig&, uint64_t&);

DepthTokenSelectFn select_depth_token_dispatch(const SpeechConfig& cfg)
{
    static constexpr std::array<DepthTokenSelectFn, 2> kDispatch = {
        select_depth_token_greedy,
        select_depth_token_sampled
    };
    return kDispatch[is_sampling_enabled(cfg) ? 1 : 0];
}

int32_t sample_temporal_text_token(
    const std::vector<float>& logits,
    int32_t text_pad_id,
    const SpeechConfig& cfg,
    uint64_t& rng_state)
{
    if (logits.empty())
        return text_pad_id;
    if (is_sampling_enabled(cfg))
    {
        constexpr float kTextTemp = 0.7F;
        constexpr int32_t kTextTopK = 25;
        return sample_token_topk(logits, kTextTemp, kTextTopK, rng_state);
    }
    return select_argmax_token(logits);
}

void maybe_log_depth_debug(
    int32_t cb, int32_t depth_call_idx,
    const std::vector<float>& logits, int32_t best)
{
    if (cb != 0 || depth_call_idx >= 12 || logits.empty())
        return;
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

} // namespace

namespace {

class PosixSubprocessRunner final : public ISubprocessRunner {
public:
    int run(
        const std::vector<std::string>& argv,
        const void* input_data,
        std::size_t input_size,
        std::vector<char>& out_stdout,
        std::string& out_stderr) override
    {
        // Build argv for execvp
        std::vector<const char*> c_argv;
        for (const auto& arg : argv)
            c_argv.push_back(arg.c_str());
        c_argv.push_back(nullptr);

        int stdin_pipe[2] = {-1, -1};
        int stdout_pipe[2] = {-1, -1};
        int stderr_pipe[2] = {-1, -1};

        if (!create_subprocess_pipes(stdin_pipe, stdout_pipe, stderr_pipe, out_stderr))
            return -1;

        pid_t pid = fork();
        if (pid < 0)
        {
            out_stderr = "fork() failed";
            close_pipe_pair(stdin_pipe);
            close_pipe_pair(stdout_pipe);
            close_pipe_pair(stderr_pipe);
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
        write_fd_all(stdin_pipe[1], input_data, input_size);
        close(stdin_pipe[1]);

        // Read stdout
        read_fd_to_vector(stdout_pipe[0], out_stdout);
        close(stdout_pipe[0]);

        // Read stderr
        read_fd_to_string(stderr_pipe[0], out_stderr);
        close(stderr_pipe[0]);

        int status = 0;
        waitpid(pid, &status, 0);
        return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }
};

} // namespace

std::shared_ptr<ISubprocessRunner> CreateDefaultSubprocessRunner()
{
    static std::shared_ptr<ISubprocessRunner> runner =
        std::make_shared<PosixSubprocessRunner>();
    return runner;
}

namespace {

bool can_inject_text_prompt(
    const SpeechConfig& cfg,
    const DecoderStepEngine& temporal_engine)
{
    return !cfg.temporal_text_embedding.empty()
        && cfg.temporal_text_vocab > 0
        && has_io_tensor(*temporal_engine.engine, "input_embed");
}

void build_text_prompt_step_embedding(
    const SpeechConfig& cfg,
    int32_t hidden_size,
    int32_t text_token,
    int32_t num_codebooks,
    int32_t bos_token,
    std::vector<float>& summed_embed)
{
    std::fill(summed_embed.begin(), summed_embed.end(), 0.0F);

    const int32_t text_tok = clamp_token(text_token, cfg.temporal_text_vocab);
    const auto text_offset = static_cast<std::size_t>(text_tok) * hidden_size;
    add_speech_embedding_row(
        cfg.temporal_text_embedding,
        text_offset,
        hidden_size,
        summed_embed.data());

    const int32_t audio_vocab = cfg.audio_vocab_size;
    const int32_t bos = clamp_token(bos_token, audio_vocab);
    const auto emb_stride_cb = static_cast<std::size_t>(audio_vocab) * hidden_size;
    for (int32_t cb = 0; cb < num_codebooks; ++cb)
    {
        const auto emb_offset = static_cast<std::size_t>(cb) * emb_stride_cb
            + static_cast<std::size_t>(bos) * hidden_size;
        add_speech_embedding_row(
            cfg.audio_embeddings,
            emb_offset,
            hidden_size,
            summed_embed.data());
    }
}

bool should_tokenize_text_prompt_runtime(const SpeechConfig& cfg)
{
    return cfg.text_prompt_ids.empty()
        && !cfg.system_prompt.empty()
        && !cfg.hf_python.empty();
}

} // namespace

// ---------------------------------------------------------------------------
// Text Prompt Injection
// ---------------------------------------------------------------------------

void SpeechToSpeechBackend::run_text_prompt(
    DeviceKvCache& cache, DeviceResources& resources,
    CudaBuffer& d_hidden_state)
{
    (void) d_hidden_state;

    const auto& cfg = mConfig;
    const int32_t hidden = mTemporalEngine->hidden_size;
    if (!can_inject_text_prompt(cfg, *mTemporalEngine))
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
    else if (should_tokenize_text_prompt_runtime(cfg))
    {
        auto tokenization = TokenizeSpeechPromptRuntime(
            cfg.hf_python, cfg.system_prompt, *mSubprocessRunner);
        if (tokenization.rc != 0 || tokenization.tokens.empty())
        {
            std::cerr << "[speech] Text prompt tokenization failed (rc="
                      << tokenization.rc << "): "
                      << tokenization.stderr_data << std::endl;
            return;
        }

        text_tokens = std::move(tokenization.tokens);
        const auto num_tokens = text_tokens.size();

        std::cerr << "[speech] Injecting runtime-tokenized text prompt: \""
                  << cfg.system_prompt << "\" (" << num_tokens << " tokens)"
                  << std::endl;
    }
    else
    {
        return;
    }

    // Run each text token through temporal with silence audio.
    // This primes the KV cache with the system instruction context.
    const int32_t bos_token = cfg.codebook_size;  // silence/BOS token
    const int32_t num_cb = cfg.num_codebooks;

    std::vector<float> summed_embed(static_cast<std::size_t>(hidden));
    std::vector<float> logits;
    std::string error;
    auto num_tokens = text_tokens.size();

    for (std::size_t t = 0; t < num_tokens; ++t)
    {
        build_text_prompt_step_embedding(
            cfg,
            hidden,
            text_tokens[t],
            num_cb,
            bos_token,
            summed_embed);

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

namespace {

struct MimiEncodeLayout
{
    int32_t engine_input_samples{0};
    int32_t enc_codebooks{0};
    int32_t enc_frames{0};
    std::size_t input_elems{0};
    std::size_t input_bytes{0};
    std::size_t output_elems{0};
    std::size_t output_bytes{0};
};

MimiEncodeLayout get_mimi_encode_layout(const nvinfer1::ICudaEngine& engine)
{
    MimiEncodeLayout layout;
    const auto in_dims = engine.getTensorShape("audio_input");
    const auto out_dims = engine.getTensorShape("codec_tokens");
    layout.engine_input_samples = in_dims.d[in_dims.nbDims - 1];
    layout.enc_codebooks = out_dims.d[0];
    layout.enc_frames = out_dims.d[1];
    layout.input_elems = static_cast<std::size_t>(layout.engine_input_samples);
    layout.input_bytes = layout.input_elems * sizeof(float);
    layout.output_elems = static_cast<std::size_t>(layout.enc_codebooks) * layout.enc_frames;
    layout.output_bytes = layout.output_elems * sizeof(float);
    return layout;
}

std::vector<float> make_mimi_encoder_input(
    const float* samples,
    int32_t num_samples,
    std::size_t input_elems)
{
    std::vector<float> input_buf(input_elems, 0.0F);
    const auto copy_n = std::min(
        static_cast<std::size_t>(num_samples),
        input_elems);
    std::memcpy(input_buf.data(), samples, copy_n * sizeof(float));
    return input_buf;
}

enum class MimiTrtRunStatus
{
    kOk,
    kAllocFailed,
    kEnqueueFailed
};

MimiTrtRunStatus run_mimi_encoder_trt(
    nvinfer1::IExecutionContext& context,
    const std::vector<float>& input_buf,
    const MimiEncodeLayout& layout,
    std::vector<float>& host_tokens)
{
    CudaBuffer d_input(layout.input_bytes);
    CudaBuffer d_output(layout.output_bytes);
    CudaStream stream;
    if (!d_input.ok() || !d_output.ok() || !stream.ok())
        return MimiTrtRunStatus::kAllocFailed;

    cudaMemcpyAsync(
        d_input.data(),
        input_buf.data(),
        layout.input_bytes,
        cudaMemcpyHostToDevice,
        stream.get());

    context.setTensorAddress("audio_input", d_input.data());
    context.setTensorAddress("codec_tokens", d_output.data());
    if (!context.enqueueV3(stream.get()))
        return MimiTrtRunStatus::kEnqueueFailed;

    host_tokens.resize(layout.output_elems);
    cudaMemcpyAsync(
        host_tokens.data(),
        d_output.data(),
        layout.output_bytes,
        cudaMemcpyDeviceToHost,
        stream.get());
    cudaStreamSynchronize(stream.get());
    return MimiTrtRunStatus::kOk;
}

std::vector<int32_t> reorder_mimi_encoder_tokens(const std::vector<float>& host_tokens_f,
                                                 int32_t enc_codebooks,
                                                 int32_t enc_frames)
{
    const auto output_elems = static_cast<std::size_t>(enc_codebooks) * enc_frames;
    std::vector<int32_t> tokens(output_elems);
    for (int32_t cb = 0; cb < enc_codebooks; ++cb)
    {
        for (int32_t frame = 0; frame < enc_frames; ++frame)
        {
            const auto src = static_cast<std::size_t>(cb) * enc_frames + frame;
            const auto dst = static_cast<std::size_t>(frame) * enc_codebooks + cb;
            tokens[dst] = static_cast<int32_t>(std::round(host_tokens_f[src]));
        }
    }
    return tokens;
}

} // namespace

std::vector<int32_t> SpeechToSpeechBackend::run_mimi_encode(
    const float* samples, int32_t num_samples,
    int32_t input_sample_rate)
{
    (void) input_sample_rate;

    mLastEncodeFrames = 0;
    mLastEncodeCodebooks = 0;

    if (!mMimiEncoderEngine || !mMimiEncoderCtx)
    {
        std::cerr << "[speech] No Mimi TRT encoder available" << std::endl;
        return {};
    }

    const auto layout = get_mimi_encode_layout(*mMimiEncoderEngine);

    // Verify input matches engine expectation
    if (num_samples != layout.engine_input_samples)
    {
        std::cerr << "[speech] WARNING: input samples " << num_samples
                  << " != engine expects " << layout.engine_input_samples
                  << ", using engine size" << std::endl;
    }

    std::cerr << "[speech] Mimi encoder TRT: input [1,1,"
              << layout.engine_input_samples << "], output ["
              << layout.enc_codebooks << "," << layout.enc_frames << "]"
              << std::endl;

    // Prepare input: pad or truncate to match engine size
    const auto input_buf = make_mimi_encoder_input(
        samples, num_samples, layout.input_elems);
    std::vector<float> host_tokens_f;
    const auto run_status = run_mimi_encoder_trt(
        *mMimiEncoderCtx,
        input_buf,
        layout,
        host_tokens_f);
    if (run_status != MimiTrtRunStatus::kOk)
    {
        if (run_status == MimiTrtRunStatus::kAllocFailed)
        {
            std::cerr << "[speech] Failed to allocate CUDA resources for Mimi encoder"
                      << std::endl;
        }
        else
        {
            std::cerr << "[speech] Mimi encoder TRT execution failed" << std::endl;
        }
        return {};
    }

    const auto tokens = reorder_mimi_encoder_tokens(
        host_tokens_f, layout.enc_codebooks, layout.enc_frames);

    std::cerr << "[speech] Mimi encode (TRT): " << num_samples << " samples -> "
              << layout.enc_frames << " frames x " << layout.enc_codebooks
              << " codebooks" << std::endl;

    mLastEncodeFrames = layout.enc_frames;
    mLastEncodeCodebooks = layout.enc_codebooks;

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

namespace {

enum class TemporalInputMode
{
    kSummedEmbedding,
    kFirstCodebookToken
};

TemporalInputMode resolve_temporal_input_mode(
    const SpeechConfig& cfg,
    const DecoderStepEngine& temporal_engine)
{
    const bool has_audio_emb = !cfg.audio_embeddings.empty()
        && cfg.audio_vocab_size > 0;
    const bool has_input_embed = has_io_tensor(*temporal_engine.engine, "input_embed");
    if (has_audio_emb && has_input_embed)
    {
        std::cerr << "[speech] Temporal: using summed per-codebook audio embeddings"
                  << std::endl;
        return TemporalInputMode::kSummedEmbedding;
    }
    if (has_audio_emb)
    {
        std::cerr << "[speech] WARNING: audio_embeddings available but engine has no "
                     "input_embed tensor; falling back to first codebook token"
                  << std::endl;
        return TemporalInputMode::kFirstCodebookToken;
    }
    std::cerr << "[speech] WARNING: no audio_embeddings; using first codebook "
                 "token only (temporal input will be degraded)"
              << std::endl;
    return TemporalInputMode::kFirstCodebookToken;
}

void build_temporal_frame_embedding(
    const SpeechConfig& cfg,
    const std::vector<int32_t>& codec_tokens,
    int32_t frame,
    int32_t num_codebooks,
    int32_t hidden_size,
    std::vector<float>& summed_embed)
{
    std::fill(summed_embed.begin(), summed_embed.end(), 0.0F);
    const int32_t audio_vocab = cfg.audio_vocab_size;
    const auto emb_stride_cb = static_cast<std::size_t>(audio_vocab) * hidden_size;
    for (int32_t cb = 0; cb < num_codebooks; ++cb)
    {
        const auto tok_idx = static_cast<std::size_t>(frame) * num_codebooks + cb;
        const int32_t token = clamp_token(
            (tok_idx < codec_tokens.size()) ? codec_tokens[tok_idx] : 0,
            audio_vocab);
        const auto emb_offset = static_cast<std::size_t>(cb) * emb_stride_cb
            + static_cast<std::size_t>(token) * hidden_size;
        add_speech_embedding_row(
            cfg.audio_embeddings,
            emb_offset,
            hidden_size,
            summed_embed.data());
    }
}

int32_t temporal_fallback_token(
    const std::vector<int32_t>& codec_tokens,
    int32_t frame,
    int32_t num_codebooks)
{
    const auto tok_idx = static_cast<std::size_t>(frame) * num_codebooks;
    return (tok_idx < codec_tokens.size()) ? codec_tokens[tok_idx] : 0;
}

void bind_temporal_hidden_output(
    DecoderStepEngine& temporal_engine,
    int32_t hidden_size,
    CudaBuffer& d_hidden_state)
{
    if (!temporal_engine.context->setTensorAddress("hidden_state", d_hidden_state.data()))
    {
        std::cerr << "[speech] WARNING: failed to bind hidden_state tensor"
                  << std::endl;
        return;
    }
    std::cerr << "[speech] Temporal engine has hidden_state output "
              << "(dim=" << hidden_size << ")" << std::endl;
}

bool run_temporal_frame_step(
    DecoderStepEngine& temporal_engine,
    DeviceKvCache& cache,
    DeviceResources& resources,
    TemporalInputMode mode,
    const std::vector<int32_t>& codec_tokens,
    int32_t frame,
    int32_t num_codebooks,
    int32_t hidden_size,
    const SpeechConfig& cfg,
    std::vector<float>& summed_embed,
    std::vector<float>& logits,
    std::string& error)
{
    if (mode == TemporalInputMode::kSummedEmbedding)
    {
        build_temporal_frame_embedding(
            cfg, codec_tokens, frame, num_codebooks, hidden_size, summed_embed);
        return run_decoder_step_device(
            temporal_engine, cache, resources,
            0, logits, error,
            summed_embed.data(), hidden_size, 1.0F);
    }
    const int32_t token = temporal_fallback_token(codec_tokens, frame, num_codebooks);
    return run_decoder_step_device(
        temporal_engine, cache, resources, token, logits, error);
}

CudaBuffer allocate_temporal_hidden_buffer(
    bool has_hidden_output,
    int32_t hidden_size)
{
    if (!has_hidden_output)
    {
        return CudaBuffer(0);
    }

    CudaBuffer d_hidden_state(static_cast<std::size_t>(hidden_size) * sizeof(float));
    if (!d_hidden_state.ok())
    {
        throw std::runtime_error("Speech: failed to allocate hidden state buffer");
    }
    return d_hidden_state;
}

void append_temporal_hidden_for_frame(
    bool has_hidden_output,
    const CudaBuffer& d_hidden_state,
    DeviceResources& resources,
    std::vector<float>& frame_hidden,
    const std::vector<float>& logits,
    int32_t hidden_size,
    std::vector<float>& all_hidden)
{
    if (has_hidden_output && d_hidden_state.ok())
    {
        read_hidden_from_device(d_hidden_state, resources, frame_hidden, hidden_size);
        all_hidden.insert(all_hidden.end(), frame_hidden.begin(), frame_hidden.end());
        return;
    }
    append_hidden_from_logits(all_hidden, logits, hidden_size);
}

} // namespace

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

    const int32_t hidden = mTemporalEngine->hidden_size;
    CudaBuffer d_hidden_state = allocate_temporal_hidden_buffer(has_hidden_output, hidden);
    if (has_hidden_output)
    {
        bind_temporal_hidden_output(*mTemporalEngine, hidden, d_hidden_state);
    }

    std::vector<float> logits;
    std::string error;

    // Feed codec tokens one frame at a time through the temporal transformer.
    // Moshi temporal input = SUM of per-codebook audio embeddings for each frame.
    // Tokens are in frame-major layout: [f0_cb0, f0_cb1, ..., f1_cb0, ...]
    //
    // If we have audio_embeddings in the config, compute the summed embedding
    // and pass via input_embed. Otherwise fall back to first codebook token only.
    const auto input_mode = resolve_temporal_input_mode(mConfig, *mTemporalEngine);

    std::vector<float> all_hidden;
    all_hidden.reserve(static_cast<std::size_t>(num_frames) * hidden);

    std::vector<float> frame_hidden(static_cast<std::size_t>(hidden));
    std::vector<float> summed_embed(static_cast<std::size_t>(hidden), 0.0F);

    for (int32_t frame = 0; frame < num_frames; ++frame)
    {
        if (!run_temporal_frame_step(
                *mTemporalEngine,
                cache,
                resources,
                input_mode,
                codec_tokens,
                frame,
                num_codebooks,
                hidden,
                mConfig,
                summed_embed,
                logits,
                error))
        {
            throw std::runtime_error("Speech temporal step failed: " + error);
        }

        append_temporal_hidden_for_frame(
            has_hidden_output,
            d_hidden_state,
            resources,
            frame_hidden,
            logits,
            hidden,
            all_hidden);
    }

    std::cerr << "[speech] Temporal: processed " << num_frames << " frames"
              << (has_hidden_output ? " (hidden_state output)" : " (logits fallback)")
              << std::endl;
    return all_hidden;
}

// ---------------------------------------------------------------------------

namespace {

bool resolve_depth_engines(
    int32_t num_cb,
    const std::vector<std::unique_ptr<DecoderStepEngine>>& per_codebook_engines,
    const std::unique_ptr<DecoderStepEngine>& fallback_engine,
    std::vector<DecoderStepEngine*>& engines)
{
    const bool has_per_cb = !per_codebook_engines.empty();
    engines.assign(static_cast<std::size_t>(num_cb), nullptr);
    for (int32_t cb = 0; cb < num_cb; ++cb)
    {
        const auto idx = static_cast<std::size_t>(cb);
        if (has_per_cb && idx < per_codebook_engines.size() && per_codebook_engines[idx])
        {
            engines[idx] = per_codebook_engines[idx].get();
        }
        else if (fallback_engine)
        {
            engines[idx] = fallback_engine.get();
        }
        else
        {
            std::cerr << "[speech] No depth engine for codebook " << cb << std::endl;
            return false;
        }
    }
    return true;
}

} // namespace

std::vector<int32_t> SpeechToSpeechBackend::run_depth(
    const float* temporal_hidden, int32_t hidden_dim,
    int32_t text_token,
    const int32_t* forced_audio_tokens,
    const uint8_t* forced_audio_provided)
{
    (void) hidden_dim;

    const auto& cfg = mConfig;
    const int32_t num_cb = cfg.num_codebooks;
    const int32_t depth_hidden = cfg.depth_hidden_size;
    if (mDepthEngines.empty() && !mDepthEngine)
    {
        return std::vector<int32_t>(num_cb, 0);
    }

    std::vector<DecoderStepEngine*> engines;
    if (!resolve_depth_engines(num_cb, mDepthEngines, mDepthEngine, engines))
        return std::vector<int32_t>(num_cb, 0);

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
    const auto projection_view = make_depth_projection_view(cfg, temporal_hidden);

    std::vector<float> depth_embed(static_cast<std::size_t>(depth_hidden), 0.0F);
    const auto select_token = select_depth_token_dispatch(cfg);

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
        build_depth_input_embedding(
            cfg,
            projection_view,
            cb,
            text_token,
            prev_token,
            depth_hidden,
            depth_embed);

        // Run with input_embed (always bypass engine embedding for correct behavior)
        if (!run_decoder_step_device(*engine, *shared_cache, *shared_resources,
                0, logits, error,
                depth_embed.data(), depth_hidden, 1.0F))
        {
            std::cerr << "[speech] Depth step cb=" << cb << " failed: "
                      << error << std::endl;
            break;
        }

        int32_t best = select_token(logits, cfg, mRngState);
        best = std::max(0, std::min(best, cfg.codebook_size - 1));
        codebook_tokens.push_back(best);
        prev_token = resolve_depth_prev_token(
            cb, best, cfg, forced_audio_tokens, forced_audio_provided);
        maybe_log_depth_debug(cb, depth_call_idx, logits, best);
    }

    // Pad if we stopped early
    while (static_cast<int32_t>(codebook_tokens.size()) < num_cb)
        codebook_tokens.push_back(0);

    return codebook_tokens;
}

// ---------------------------------------------------------------------------
// Stage 4: Mimi Decode (codec tokens -> waveform)
// ---------------------------------------------------------------------------

namespace {

MimiDecodeLayout get_mimi_decode_layout(const nvinfer1::ICudaEngine& engine)
{
    const auto in_dims = engine.getTensorShape("codec_tokens");
    const auto out_dims = engine.getTensorShape("audio_output");
    std::vector<int32_t> output_dims;
    output_dims.reserve(static_cast<std::size_t>(out_dims.nbDims));
    for (int32_t d = 0; d < out_dims.nbDims; ++d)
    {
        output_dims.push_back(out_dims.d[d]);
    }
    return build_mimi_decode_layout(in_dims.d[0], in_dims.d[1], output_dims);
}

MimiTrtRunStatus run_mimi_decoder_trt(
    nvinfer1::IExecutionContext& context,
    const std::vector<float>& input_tokens,
    const MimiDecodeLayout& layout,
    std::vector<float>& waveform)
{
    CudaBuffer d_input(layout.input_bytes);
    CudaBuffer d_output(layout.output_bytes);
    CudaStream stream;
    if (!d_input.ok() || !d_output.ok() || !stream.ok())
        return MimiTrtRunStatus::kAllocFailed;

    cudaMemcpyAsync(
        d_input.data(),
        input_tokens.data(),
        layout.input_bytes,
        cudaMemcpyHostToDevice,
        stream.get());

    context.setTensorAddress("codec_tokens", d_input.data());
    context.setTensorAddress("audio_output", d_output.data());
    if (!context.enqueueV3(stream.get()))
        return MimiTrtRunStatus::kEnqueueFailed;

    waveform.resize(static_cast<std::size_t>(layout.total_output_elems));
    cudaMemcpyAsync(
        waveform.data(),
        d_output.data(),
        layout.output_bytes,
        cudaMemcpyDeviceToHost,
        stream.get());
    cudaStreamSynchronize(stream.get());
    return MimiTrtRunStatus::kOk;
}

} // namespace

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

    const auto layout = get_mimi_decode_layout(*mMimiDecoderEngine);

    std::cerr << "[speech] Mimi decoder TRT: input [" << layout.dec_codebooks
              << "," << layout.dec_frames << "], output " << layout.total_output_elems
              << " samples" << std::endl;

    // Build the decoder input: codebook-major [dec_codebooks, dec_frames]
    // Input tokens are frame-major: [f0_cb0, f0_cb1, ..., f1_cb0, ...]
    auto input_tokens = build_mimi_decoder_input(
        codec_tokens,
        num_frames,
        actual_codebooks,
        layout.dec_frames,
        layout.dec_codebooks);

    // Debug: print first few input tokens
    std::cerr << "[speech] Decoder input tokens [0:16]: ";
    for (int32_t i = 0; i < std::min(16, static_cast<int32_t>(layout.input_elems)); ++i)
        std::cerr << input_tokens[i] << " ";
    std::cerr << std::endl;

    std::vector<float> waveform;
    const auto run_status = run_mimi_decoder_trt(
        *mMimiDecoderCtx,
        input_tokens,
        layout,
        waveform);
    if (run_status != MimiTrtRunStatus::kOk)
    {
        if (run_status == MimiTrtRunStatus::kAllocFailed)
        {
            std::cerr << "[speech] Failed to allocate CUDA resources for Mimi decoder"
                      << std::endl;
        }
        else
        {
            std::cerr << "[speech] Mimi decoder TRT execution failed" << std::endl;
        }
        return {};
    }

    // Debug: check output statistics
    float rms = 0.0F;
    float mx = 0.0F;
    waveform_stats(waveform, layout.total_output_elems, rms, mx);
    std::cerr << "[speech] Mimi decode (TRT): " << layout.dec_frames << " frames -> "
              << layout.total_output_elems << " samples (RMS=" << rms
              << ", Max=" << mx << ")" << std::endl;
    return waveform;
}

// ---------------------------------------------------------------------------
// Full pipeline
// ---------------------------------------------------------------------------

namespace {

EncoderShapeInfo resolve_encoder_shape(
    nvinfer1::ICudaEngine* encoder_engine,
    const SpeechConfig& cfg,
    int32_t last_encode_codebooks,
    int32_t last_encode_frames,
    const std::vector<int32_t>& codec_tokens)
{
    const int32_t fallback_codebooks = encoder_engine != nullptr
        ? encoder_engine->getTensorShape("codec_tokens").d[0]
        : cfg.num_codebooks;
    return resolve_encoder_shape_with_fallback_codebooks(
        cfg,
        last_encode_codebooks,
        last_encode_frames,
        codec_tokens.size(),
        fallback_codebooks);
}

void log_depth_mode(const SpeechConfig& cfg)
{
    if (is_sampling_enabled(cfg))
    {
        std::cerr << "[speech] Depth sampling: temperature="
                  << cfg.depth_temperature << " top_k="
                  << cfg.depth_top_k << std::endl;
        return;
    }
    std::cerr << "[speech] Depth decoding: greedy (argmax)" << std::endl;
}

using OutputPlan = SpeechOutputPlan;

void log_stop_configuration(
    const SpeechConfig& cfg,
    int32_t extra_tail)
{
    if (cfg.text_eos_token_id >= 0)
    {
        std::cerr << "[speech] Text EOS early-stop enabled: eos_token_id="
                  << cfg.text_eos_token_id << " (min_streak="
                  << kSpeechMinConsecutiveTextEos << ")" << std::endl;
    }
    if (extra_tail <= 0)
        return;
    std::cerr << "[speech] Text PAD fallback stop enabled after input "
                 "(pad_id="
              << cfg.text_padding_id << ", min_streak="
              << kSpeechMinConsecutiveTextPadAfterInput << ")" << std::endl;
    std::cerr << "[speech] Post-input continuation cap: "
              << kSpeechMaxContinuationFramesAfterInput << " frames" << std::endl;
}

void maybe_log_stop_decision(
    SpeechDecodeStopReason reason,
    const SpeechDecodeStopState& stop_state,
    int32_t offset)
{
    switch (reason)
    {
    case SpeechDecodeStopReason::kNone:
        return;
    case SpeechDecodeStopReason::kTextEos:
        std::cerr << "[speech] Text EOS detected at offset " << offset
                  << " (streak=" << stop_state.text_eos_streak
                  << "), draining delayed frames until offset "
                  << stop_state.stop_collect_until_offset << std::endl;
        return;
    case SpeechDecodeStopReason::kTextPadFallback:
        std::cerr << "[speech] Text PAD fallback stop at offset " << offset
                  << " (streak=" << stop_state.text_pad_streak
                  << "), draining delayed frames until offset "
                  << stop_state.stop_collect_until_offset << std::endl;
        return;
    case SpeechDecodeStopReason::kContinuationCap:
        std::cerr << "[speech] Continuation cap reached at offset " << offset
                  << ", draining delayed frames until offset "
                  << stop_state.stop_collect_until_offset << std::endl;
        return;
    }
}

void maybe_log_interleaved_debug(
    int32_t offset,
    int32_t hidden,
    const std::vector<float>& frame_hidden,
    int32_t text_input,
    int32_t sampled_text_token,
    const std::vector<int32_t>& frame_codes)
{
    if (offset <= 0 || offset > 5)
        return;
    float l2 = 0.0F;
    for (int32_t d = 0; d < hidden; ++d)
        l2 += frame_hidden[static_cast<std::size_t>(d)]
            * frame_hidden[static_cast<std::size_t>(d)];
    l2 = std::sqrt(l2);
    std::cerr << "[speech] Offset " << offset << " hidden L2=" << l2
              << " text_in=" << text_input
              << " text_out=" << sampled_text_token << " depth:";
    for (int32_t cb = 0; cb < std::min(4, static_cast<int32_t>(frame_codes.size())); ++cb)
        std::cerr << " " << frame_codes[static_cast<std::size_t>(cb)];
    std::cerr << "..." << std::endl;
}

using DepthRunnerFn = std::function<std::vector<int32_t>(
    const float*,
    int32_t,
    int32_t,
    const int32_t*,
    const uint8_t*)>;

bool run_generation_iteration(
    DecoderStepEngine& temporal_engine,
    DeviceKvCache& temporal_cache,
    DeviceResources& temporal_resources,
    CudaBuffer& d_hidden_state,
    bool has_hidden_output,
    const SpeechConfig& cfg,
    const SpeechGenerationSettings& settings,
    const std::vector<int32_t>& codec_tokens,
    DelayCacheState& delay_state,
    int32_t offset,
    const OutputPlan& plan,
    DepthRunnerFn depth_runner,
    uint64_t& rng_state,
    std::vector<float>& summed_embed,
    std::vector<float>& frame_hidden,
    std::vector<float>& logits,
    std::string& error,
    std::vector<int32_t>& moshi_input,
    std::vector<int32_t>& user_input,
    std::vector<int32_t>& target_audio_tokens,
    std::vector<uint8_t>& target_audio_provided,
    SpeechDecodeStopState& stop_state,
    int32_t& frames_collected,
    std::vector<int32_t>& output_codes)
{
    write_user_tokens_to_delay_cache(
        delay_state,
        codec_tokens,
        offset,
        settings.stream_cb,
        settings.num_frames,
        settings.encode_codebooks,
        settings.audio_bos);
    fill_initial_delay_tokens(delay_state, offset, settings.text_bos, settings.audio_bos);
    if (offset == 0)
    {
        seed_delay_offset_zero(delay_state, settings.text_bos, settings.audio_bos);
        return true;
    }

    const int32_t model_input_pos = offset - 1;
    const int32_t target_pos = offset;

    int32_t text_input = settings.text_pad_id;
    read_model_inputs_from_delay_cache(
        delay_state,
        model_input_pos,
        settings.stream_cb,
        text_input,
        moshi_input,
        user_input);
    compute_dual_stream_summed_embed(
        cfg,
        settings.hidden,
        settings.stream_cb,
        moshi_input.data(),
        user_input.data(),
        text_input,
        summed_embed.data());

    if (!run_decoder_step_device(
            temporal_engine,
            temporal_cache,
            temporal_resources,
            0,
            logits,
            error,
            summed_embed.data(),
            settings.hidden,
            1.0F))
    {
        throw std::runtime_error("Speech temporal step failed: " + error);
    }

    if (has_hidden_output && d_hidden_state.ok())
        read_hidden_from_device(d_hidden_state, temporal_resources, frame_hidden, settings.hidden);
    else
        fill_hidden_from_logits(frame_hidden, logits, settings.hidden);

    const int32_t sampled_text_token = sample_temporal_text_token(
        logits, settings.text_pad_id, cfg, rng_state);
    const auto text_target_idx = delay_cache_index(delay_state, 0, target_pos);
    const bool text_provided = delay_state.provided[text_target_idx] != 0;
    const int32_t next_text_token =
        text_provided ? delay_state.cache[text_target_idx] : sampled_text_token;

    build_target_audio_arrays(
        delay_state,
        target_pos,
        settings.num_cb,
        settings.audio_bos,
        target_audio_tokens,
        target_audio_provided);
    auto frame_codes = depth_runner(
        frame_hidden.data(),
        settings.hidden,
        next_text_token,
        target_audio_tokens.data(),
        target_audio_provided.data());

    clear_provided_flags_at_pos(delay_state, model_input_pos);
    write_generated_tokens_to_delay_cache(
        delay_state,
        target_pos,
        sampled_text_token,
        text_provided,
        frame_codes,
        settings.num_cb);
    if (collect_output_codes_from_delay_cache(
            delay_state, offset, delay_state.max_delay, settings.mimi_cb, output_codes))
    {
        ++frames_collected;
    }

    SpeechDecodeStopInput stop_input;
    stop_input.text_eos_token_id = cfg.text_eos_token_id;
    stop_input.text_padding_id = cfg.text_padding_id;
    stop_input.effective_frames = plan.effective_frames;
    stop_input.extra_tail = plan.extra_tail;
    stop_input.target_pos = target_pos;
    stop_input.sampled_text_token = sampled_text_token;
    stop_input.offset = offset;
    stop_input.max_delay = delay_state.max_delay;
    stop_input.text_provided = text_provided;
    const auto stop_decision = UpdateSpeechDecodeStopState(stop_state, stop_input);
    stop_state = stop_decision.state;
    maybe_log_stop_decision(stop_decision.reason, stop_state, offset);
    maybe_log_interleaved_debug(
        offset, settings.hidden, frame_hidden, text_input, sampled_text_token, frame_codes);
    return !stop_decision.should_break;
}

int32_t run_interleaved_generation(
    DecoderStepEngine& temporal_engine,
    DeviceKvCache& temporal_cache,
    DeviceResources& temporal_resources,
    CudaBuffer& d_hidden_state,
    bool has_hidden_output,
    const SpeechConfig& cfg,
    const SpeechGenerationSettings& settings,
    const OutputPlan& plan,
    const std::vector<int32_t>& codec_tokens,
    DelayCacheState& delay_state,
    DepthRunnerFn depth_runner,
    uint64_t& rng_state,
    std::vector<int32_t>& output_codes)
{
    SpeechDecodeStopState stop_state;
    log_stop_configuration(cfg, plan.extra_tail);

    std::vector<float> summed_embed(static_cast<std::size_t>(settings.hidden));
    std::vector<float> frame_hidden(static_cast<std::size_t>(settings.hidden));
    std::vector<float> logits;
    std::string error;
    std::vector<int32_t> moshi_input(static_cast<std::size_t>(settings.stream_cb));
    std::vector<int32_t> user_input(static_cast<std::size_t>(settings.stream_cb));
    std::vector<int32_t> target_audio_tokens(static_cast<std::size_t>(settings.num_cb));
    std::vector<uint8_t> target_audio_provided(static_cast<std::size_t>(settings.num_cb));

    int32_t frames_collected = 0;
    for (int32_t offset = 0;
         offset < plan.total_iters && frames_collected < plan.output_frames;
         ++offset)
    {
        const bool continue_running = run_generation_iteration(
            temporal_engine,
            temporal_cache,
            temporal_resources,
            d_hidden_state,
            has_hidden_output,
            cfg,
            settings,
            codec_tokens,
            delay_state,
            offset,
            plan,
            depth_runner,
            rng_state,
            summed_embed,
            frame_hidden,
            logits,
            error,
            moshi_input,
            user_input,
            target_audio_tokens,
            target_audio_provided,
            stop_state,
            frames_collected,
            output_codes);
        if (!continue_running)
            break;
    }
    return frames_collected;
}

void maybe_bind_hidden_output_for_process(
    DecoderStepEngine& temporal_engine,
    bool has_hidden_output,
    int32_t hidden_size,
    CudaBuffer& d_hidden_state)
{
    if (!has_hidden_output)
        return;
    d_hidden_state = CudaBuffer(static_cast<std::size_t>(hidden_size) * sizeof(float));
    if (!d_hidden_state.ok())
        return;
    temporal_engine.context->setTensorAddress("hidden_state", d_hidden_state.data());
}

void log_output_frames_debug(
    const std::vector<int32_t>& output_codes,
    int32_t generated_frames,
    int32_t mimi_cb)
{
    if (output_codes.empty())
        return;
    for (int32_t frame = 0; frame < generated_frames; ++frame)
    {
        std::cerr << "[speech] Output frame " << frame << ":";
        for (int32_t cb = 0; cb < mimi_cb; ++cb)
        {
            const auto idx = static_cast<std::size_t>(frame) * mimi_cb + cb;
            if (idx < output_codes.size())
                std::cerr << " " << output_codes[idx];
        }
        std::cerr << std::endl;
    }
}

LegacyAudioResult finalize_speech_audio_result(
    const SpeechConfig& config,
    int32_t generated_frames,
    std::vector<float> waveform)
{
    LegacyAudioResult result;
    result.sample_rate = config.sample_rate;

    const auto trim_result = trim_speech_waveform_to_generated_frames(
        config.sample_rate, config.frame_rate, generated_frames, waveform);
    if (trim_result.trimmed)
    {
        std::cerr << "[speech] Trimmed decoded waveform to "
                  << trim_result.expected_samples << " samples (" << generated_frames
                  << " generated frames)" << std::endl;
    }

    const auto normalize_result = peak_normalize_speech_waveform(waveform);
    if (normalize_result.normalized)
    {
        std::cerr << "[speech] Peak-normalized: peak=" << normalize_result.peak
                  << " scale=" << normalize_result.scale << std::endl;
    }

    result.waveform = std::move(waveform);
    result.num_samples = static_cast<int32_t>(result.waveform.size());
    std::cerr << "[speech] Generated " << result.num_samples << " samples ("
              << static_cast<float>(result.num_samples) / result.sample_rate
              << "s @ " << result.sample_rate << " Hz)" << std::endl;
    return result;
}

} // namespace

LegacyAudioResult SpeechToSpeechBackend::process_audio(
    const float* input_samples, int32_t num_input_samples,
    int32_t max_output_frames,
    int32_t input_sample_rate,
    int32_t tail_frames)
{
    LegacyAudioResult result;
    result.sample_rate = mConfig.sample_rate;

    if (!is_available())
    {
        std::cerr << "[speech] Backend not available" << std::endl;
        return result;
    }

    mDepthDebugCallCount = 0;

    std::cerr << "[speech] Starting pipeline with " << num_input_samples
              << " input samples" << std::endl;

    log_depth_mode(mConfig);

    // ---------------------------------------------------------------
    // Stage 1: Encode input audio via Mimi
    // ---------------------------------------------------------------
    auto codec_tokens = run_mimi_encode(
        input_samples, num_input_samples, input_sample_rate);

    // Mimi outputs 32 codebooks; we only use the first stream_cb for the user
    // stream.  PersonaPlex uses num_codebooks=16 total (8 moshi + 8 user).
    const auto encoder_shape = resolve_encoder_shape(
        mMimiEncoderEngine.get(),
        mConfig,
        mLastEncodeCodebooks,
        mLastEncodeFrames,
        codec_tokens);
    const int32_t encode_codebooks = encoder_shape.encode_codebooks;
    const int32_t num_frames = encoder_shape.num_frames;

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
    maybe_bind_hidden_output_for_process(
        *mTemporalEngine, has_hidden_output, hidden, d_hidden_state);

    // ---------------------------------------------------------------
    // Text prompt injection (primes temporal KV cache with system prompt)
    // Uses pre-tokenized IDs from bundle config (avoids runtime tokenization)
    // ---------------------------------------------------------------
    if (should_run_text_prompt_injection(mConfig))
        run_text_prompt(temporal_cache, temporal_resources, d_hidden_state);
    auto delay_state = make_delay_cache_state(mConfig.delays, num_cb);
    SpeechOutputPlanInput plan_input;
    plan_input.sample_rate = mConfig.sample_rate;
    plan_input.frame_rate = mConfig.frame_rate;
    plan_input.num_frames = num_frames;
    plan_input.num_input_samples = num_input_samples;
    plan_input.input_sample_rate = input_sample_rate;
    plan_input.tail_frames = tail_frames;
    plan_input.max_output_frames = max_output_frames;
    plan_input.max_delay = delay_state.max_delay;
    const auto plan = ComputeSpeechOutputPlan(plan_input);
    const int32_t mimi_cb = mConfig.mimi_decode_codebooks;
    std::vector<int32_t> output_codes;
    output_codes.reserve(
        static_cast<std::size_t>(mimi_cb) * plan.output_frames);
    std::cerr << "[speech] Interleaved temporal+depth with delay pattern: "
              << plan.output_frames << " output frames, " << plan.total_iters
              << " total iterations (max_delay=" << delay_state.max_delay
              << ", input_effective=" << plan.effective_frames
              << ", tail_frames=" << plan.extra_tail << ")"
              << std::endl;

    const SpeechGenerationSettings settings = make_speech_generation_settings(
        mConfig,
        hidden,
        encoder_shape);

    const DepthRunnerFn depth_runner = [this](
                                          const float* hidden_data,
                                          int32_t hidden_dim,
                                          int32_t text_token,
                                          const int32_t* forced_tokens,
                                          const uint8_t* forced_provided)
    {
        return run_depth(
            hidden_data, hidden_dim, text_token, forced_tokens, forced_provided);
    };

    const int32_t generated_frames = run_interleaved_generation(
        *mTemporalEngine,
        temporal_cache,
        temporal_resources,
        d_hidden_state,
        has_hidden_output,
        mConfig,
        settings,
        plan,
        codec_tokens,
        delay_state,
        depth_runner,
        mRngState,
        output_codes);
    std::cerr << "[speech] Depth: generated " << generated_frames
              << " frames x " << num_cb << " codebooks (decoding first "
              << mimi_cb << ")" << std::endl;
    log_output_frames_debug(output_codes, generated_frames, mimi_cb);

    // ---------------------------------------------------------------
    // Stage 4: Decode first mimi_cb codebook tokens to audio via Mimi decoder
    // ---------------------------------------------------------------
    return finalize_speech_audio_result(
        mConfig, generated_frames, run_mimi_decode(output_codes, generated_frames));
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
    speech_cfg.text_eos_token_id = cfg.id_eos;

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
