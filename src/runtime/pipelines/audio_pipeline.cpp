#include "runtime/pipelines/audio_pipeline.h"

#if TRTF_HAS_TRT

#include "runtime/trt/audio/audio_configs.h"
#include "runtime/trt/audio/omni_audio_plan.h"
#include "runtime/trt/audio/mel_spectrogram.h"
#include "runtime/trt/audio/whisper_host_plan.h"
#include "runtime/trt/audio/whisper_cross_kv_plan.h"
#include "runtime/trt/audio/whisper_cross_kv_apply.h"
#include "runtime/trt/audio/whisper_decode_policy.h"
#include "runtime/trt/audio/bark_generation_plan.h"
#include "runtime/trt/audio/magpie_codec_plan.h"
#include "runtime/trt/audio/magpie_decode_policy.h"
#include "runtime/trt/audio/magpie_decoder_plan.h"
#include "runtime/trt/audio/magpie_text_completion_policy.h"
#include "runtime/trt/audio/speech_depth_plan.h"
#include "runtime/trt/audio/speech_delay_cache.h"
#include "runtime/trt/audio/speech_generation_policy.h"
#include "runtime/trt/audio/speech_mimi_decode_plan.h"
#include "runtime/trt/audio/speech_runtime_plan.h"
#include "runtime/trt/audio/speech_temporal_embed_plan.h"
#include "runtime/trt/audio/speech_waveform_postprocess.h"
#include "trtf/runtime/trt/audio/speech_decode_stop_policy.h"
#include "runtime/trt/core/trt_engine_lifecycle.h"
#include "runtime/trt/core/trt_decode_runtime.h"
#include "utils/wav_reader.h"
#include "trtf/tokenizer.h"
#include "trtf/runtime/trt/audio/subprocess_runner.h"

#ifndef TRTF_HAS_CUDA_KERNELS
#define TRTF_HAS_CUDA_KERNELS 0
#endif

#if TRTF_HAS_CUDA_KERNELS
#include "runtime/trt/audio/magpie_kernels.h"
#endif

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include <sys/wait.h>
#include <unistd.h>

namespace trtf {

// ─── PosixSubprocessRunner (moved from speech_backend.cpp) ───

namespace {

// ---------------------------------------------------------------------------
// Subprocess pipe helpers (extracted from PosixSubprocessRunner::run)
// ---------------------------------------------------------------------------

bool create_subprocess_pipes(int (&stdin_pipe)[2], int (&stdout_pipe)[2],
                              int (&stderr_pipe)[2])
{
    return pipe(stdin_pipe) == 0 && pipe(stdout_pipe) == 0 && pipe(stderr_pipe) == 0;
}

void write_all_to_fd(int fd, const void* data, std::size_t size)
{
    const auto* p = static_cast<const char*>(data);
    std::size_t remaining = size;
    while (remaining > 0)
    {
        auto written = write(fd, p, remaining);
        if (written <= 0) break;
        p += written;
        remaining -= static_cast<std::size_t>(written);
    }
}

void read_all_from_fd(int fd, std::vector<char>& out)
{
    out.clear();
    char buf[65536];
    for (;;)
    {
        auto n = read(fd, buf, sizeof(buf));
        if (n <= 0) break;
        out.insert(out.end(), buf, buf + n);
    }
}

void read_all_string_from_fd(int fd, std::string& out)
{
    out.clear();
    char buf[65536];
    for (;;)
    {
        auto n = read(fd, buf, sizeof(buf));
        if (n <= 0) break;
        out.append(buf, static_cast<std::size_t>(n));
    }
}

void exec_child_process(int (&stdin_pipe)[2], int (&stdout_pipe)[2],
                         int (&stderr_pipe)[2],
                         const std::vector<const char*>& c_argv)
{
    dup2(stdin_pipe[0], STDIN_FILENO);
    dup2(stdout_pipe[1], STDOUT_FILENO);
    dup2(stderr_pipe[1], STDERR_FILENO);
    close(stdin_pipe[0]); close(stdin_pipe[1]);
    close(stdout_pipe[0]); close(stdout_pipe[1]);
    close(stderr_pipe[0]); close(stderr_pipe[1]);
    execvp(c_argv[0], const_cast<char* const*>(c_argv.data()));
    _exit(127);
}

class PosixSubprocessRunner final : public ISubprocessRunner {
public:
    int run(
        const std::vector<std::string>& argv,
        const void* input_data,
        std::size_t input_size,
        std::vector<char>& out_stdout,
        std::string& out_stderr) override
    {
        std::vector<const char*> c_argv;
        for (const auto& arg : argv)
            c_argv.push_back(arg.c_str());
        c_argv.push_back(nullptr);

        int stdin_pipe[2] = {-1, -1};
        int stdout_pipe[2] = {-1, -1};
        int stderr_pipe[2] = {-1, -1};

        if (!create_subprocess_pipes(stdin_pipe, stdout_pipe, stderr_pipe))
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
            exec_child_process(stdin_pipe, stdout_pipe, stderr_pipe, c_argv);

        close(stdin_pipe[0]);
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);

        if (input_data && input_size > 0)
            write_all_to_fd(stdin_pipe[1], input_data, input_size);
        close(stdin_pipe[1]);

        read_all_from_fd(stdout_pipe[0], out_stdout);
        close(stdout_pipe[0]);

        read_all_string_from_fd(stderr_pipe[0], out_stderr);
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

constexpr int32_t kBarkMaxTextLen = 256;
constexpr float kNegInf = -1e9F;

// ─── Bark helpers (moved from bark_backend.cpp) ───

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

void maybe_dump_tokens(const char* suffix, const std::vector<int32_t>& tokens)
{
    const char* dump_path = std::getenv("TRTF_BARK_DUMP");
    if (dump_path == nullptr) return;
    std::ofstream dump(std::string(dump_path) + suffix);
    for (int32_t token : tokens)
    {
        dump << token << "\n";
    }
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
    if (env == nullptr || *env == '\0') return;
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

} // namespace

// ═══════════════════════════════════════════════════════════════════════════
// WhisperPipeline
// ═══════════════════════════════════════════════════════════════════════════

WhisperPipeline::WhisperPipeline(
    std::unique_ptr<TrtModule> encoder,
    std::unique_ptr<TrtModule> decoder,
    std::unique_ptr<KvCache> cache,
    WhisperConfig whisper_config,
    int32_t hidden_size,
    int32_t num_decoder_layers,
    MelFilterbank mel_fb,
    int32_t mel_n_fft,
    int32_t mel_hop_length,
    int32_t mel_chunk_length,
    int32_t mel_sampling_rate,
    cudaStream_t stream,
    std::shared_ptr<ITokenizer> tokenizer,
    std::string model_id_str)
    : encoder_(std::move(encoder))
    , decoder_(std::move(decoder))
    , cache_(std::move(cache))
    , whisper_config_(std::move(whisper_config))
    , hidden_size_(hidden_size)
    , num_decoder_layers_(num_decoder_layers)
    , mel_fb_(std::make_unique<MelFilterbank>(std::move(mel_fb)))
    , mel_n_fft_(mel_n_fft)
    , mel_hop_length_(mel_hop_length)
    , mel_chunk_length_(mel_chunk_length)
    , mel_sampling_rate_(mel_sampling_rate)
    , stream_(stream)
    , tokenizer_(std::move(tokenizer))
    , model_id_(std::move(model_id_str))
{
    if (!encoder_ || !encoder_->ok())
        throw std::runtime_error("WhisperPipeline: invalid encoder module");
    if (!decoder_ || !decoder_->ok())
        throw std::runtime_error("WhisperPipeline: invalid decoder module");
    if (!cache_ || !cache_->ok())
        throw std::runtime_error("WhisperPipeline: invalid KvCache");

    // Allocate cross-attention K/V device buffers
    cross_kv_bytes_ = static_cast<std::size_t>(whisper_config_.max_source_positions)
        * static_cast<std::size_t>(hidden_size_) * sizeof(float);

    cross_k_ptrs_.resize(static_cast<std::size_t>(num_decoder_layers_), nullptr);
    cross_v_ptrs_.resize(static_cast<std::size_t>(num_decoder_layers_), nullptr);
    for (int32_t i = 0; i < num_decoder_layers_; ++i)
    {
        cudaMalloc(&cross_k_ptrs_[static_cast<std::size_t>(i)], cross_kv_bytes_);
        cudaMalloc(&cross_v_ptrs_[static_cast<std::size_t>(i)], cross_kv_bytes_);
    }
}

WhisperPipeline::~WhisperPipeline()
{
    for (auto* ptr : cross_k_ptrs_) { if (ptr) cudaFree(ptr); }
    for (auto* ptr : cross_v_ptrs_) { if (ptr) cudaFree(ptr); }
}

TextResult WhisperPipeline::transcribe(
    const float* audio_data, int32_t num_samples, int32_t max_new_tokens,
    int32_t input_sample_rate)
{
    // Step 0: Resample if needed
    const float* samples_ptr = audio_data;
    int32_t samples_count = num_samples;
    std::vector<float> resampled_buf;

    if (input_sample_rate > 0 && input_sample_rate != mel_sampling_rate_)
    {
        std::cerr << "[whisper] Resampling audio from " << input_sample_rate
                  << " Hz to " << mel_sampling_rate_ << " Hz" << std::endl;
        resampled_buf = resample_linear(
            audio_data, num_samples, input_sample_rate, mel_sampling_rate_);
        samples_ptr = resampled_buf.data();
        samples_count = static_cast<int32_t>(resampled_buf.size());
    }

    // Step 1: Extract mel spectrogram
    MelResult mel;
    if (mel_fb_ && !mel_fb_->data.empty())
    {
        mel = extract_mel_spectrogram(
            samples_ptr, samples_count,
            mel_fb_->data.data(), mel_fb_->n_freq_bins, mel_fb_->n_mel_bins,
            mel_n_fft_, mel_hop_length_,
            mel_chunk_length_, mel_sampling_rate_);
    }

    if (mel.data.empty())
    {
        return TextResult{"[mel extraction failed]", {}};
    }

    // Step 2: Run encoder
    std::cerr << "[whisper] Running encoder ..." << std::endl;
    run_encoder(mel.data.data(), mel.n_mels, mel.n_frames);

    // Compute actual encoder sequence length for masking
    const int32_t mel_full = resolve_whisper_expected_mel_length(whisper_config_);
    int32_t actual_enc_seq_len = compute_whisper_actual_encoder_length(
        mel.n_frames, mel_full, whisper_config_.max_source_positions);
    if (actual_enc_seq_len > 0)
    {
        std::cerr << "[whisper] Actual encoder seq len: " << actual_enc_seq_len
                  << " / " << whisper_config_.max_source_positions << std::endl;
    }

    // Step 3: Set up cross-attention K/V
    std::cerr << "[whisper] Computing cross-attention K/V ..." << std::endl;
    setup_cross_attention(actual_enc_seq_len);

    // Step 4: Run decoder
    std::vector<int32_t> initial_tokens = make_whisper_initial_decoder_tokens(whisper_config_);
    std::cerr << "[whisper] Running decoder ..." << std::endl;
    auto output_ids = run_decoder(initial_tokens, max_new_tokens);

    // Step 5: Decode token IDs
    TextResult out;
    out.token_ids = std::move(output_ids);
    if (tokenizer_ && !out.token_ids.empty())
    {
        out.text = tokenizer_->decode(out.token_ids);
    }
    return out;
}

void WhisperPipeline::run_encoder(
    const float* mel_data, int32_t mel_bins, int32_t mel_length)
{
    const int32_t expected_length = resolve_whisper_expected_mel_length(whisper_config_);
    const std::size_t mel_size = static_cast<std::size_t>(mel_bins) *
        static_cast<std::size_t>(expected_length);

    // Prepare mel input (pad if needed)
    std::vector<float> mel_host;
    if (mel_length == expected_length)
    {
        mel_host.assign(mel_data, mel_data + mel_size);
    }
    else
    {
        mel_host = build_whisper_padded_mel_input(
            mel_data, mel_bins, mel_length, expected_length);
    }

    // Build input TensorMap
    TensorMap inputs;
    Tensor mel_tensor;
    mel_tensor.data = mel_host.data();
    mel_tensor.shape = {mel_bins, expected_length};
    mel_tensor.dtype = DType::kFloat32;
    inputs["mel_features"] = mel_tensor;

    // Optional encoder_mask input
    const int32_t enc_seq = whisper_config_.max_source_positions;
    std::vector<float> enc_mask;
    if (encoder_->has_input("encoder_mask"))
    {
        int32_t actual_enc = compute_whisper_actual_encoder_length(
            mel_length, expected_length, enc_seq);
        if (actual_enc <= 0) actual_enc = enc_seq;
        enc_mask = build_whisper_encoder_mask_values(enc_seq, actual_enc);

        Tensor mask_tensor;
        mask_tensor.data = enc_mask.data();
        mask_tensor.shape = {static_cast<int64_t>(enc_mask.size())};
        mask_tensor.dtype = DType::kFloat32;
        inputs["encoder_mask"] = mask_tensor;
    }

    // Run encoder (we need the output to stay on device, so use forward_async + sync)
    encoder_->forward_async(inputs);
    encoder_->sync();
}

void WhisperPipeline::setup_cross_attention(int32_t actual_enc_seq_len)
{
    // Get encoder output device pointer
    void* enc_output_device = encoder_->device_ptr("encoder_output");

    // Apply cross-KV plan: optionally zero-pad encoder output, then copy to each layer
    const auto plan = make_whisper_cross_kv_plan(
        whisper_config_.max_source_positions,
        hidden_size_,
        actual_enc_seq_len);

    std::string error;
    const bool ok = apply_whisper_cross_kv_plan(
        plan,
        static_cast<std::size_t>(num_decoder_layers_),
        [enc_output_device](std::size_t valid_bytes, std::size_t pad_bytes)
        {
            return cudaMemset(
                static_cast<char*>(enc_output_device) + valid_bytes,
                0,
                pad_bytes) == cudaSuccess;
        },
        [this, enc_output_device](std::size_t layer, WhisperCrossKvBufferKind kind, std::size_t bytes)
        {
            void* dst = kind == WhisperCrossKvBufferKind::K
                ? cross_k_ptrs_[layer]
                : cross_v_ptrs_[layer];
            return cudaMemcpy(dst, enc_output_device, bytes, cudaMemcpyDeviceToDevice)
                == cudaSuccess;
        },
        error);
    if (!ok)
    {
        throw std::runtime_error(error);
    }

    // Bind cross-K/V to decoder module
    for (int32_t i = 0; i < num_decoder_layers_; ++i)
    {
        const std::string suffix = "_" + std::to_string(i);
        decoder_->bind_external("cross_k" + suffix, cross_k_ptrs_[static_cast<std::size_t>(i)]);
        decoder_->bind_external("cross_v" + suffix, cross_v_ptrs_[static_cast<std::size_t>(i)]);
    }
}

std::vector<int32_t> WhisperPipeline::run_decoder(
    const std::vector<int32_t>& initial_tokens,
    int32_t max_new_tokens)
{
    cache_->reset();
    cache_->bind_to(*decoder_);

    const int32_t eot_id = whisper_config_.eot_token_id;

    auto result = run_whisper_decode_loop(
        initial_tokens,
        max_new_tokens,
        eot_id,
        [this](int32_t token, std::vector<float>& logits, std::string&)
        {
            run_decoder_step(token, logits);
            return true;
        },
        [](const std::vector<float>& logits)
        {
            return select_argmax_token(logits);
        });

    if (result.prefill_failed)
    {
        std::cerr << "[whisper] Prefill step failed: " << result.error << std::endl;
    }
    else if (result.decode_failed)
    {
        std::cerr << "[whisper] Decode step failed: " << result.error << std::endl;
    }

    return result.output_ids;
}

void WhisperPipeline::run_decoder_step(int32_t token_id, std::vector<float>& logits)
{
    std::vector<float> mask;
    cache_->build_attention_mask(mask);
    int32_t position = cache_->position();

    Tensor token_tensor;
    token_tensor.data = &token_id;
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

    TensorMap inputs;
    inputs["token_id"] = token_tensor;
    if (decoder_->has_input("position_id"))
    {
        inputs["position_id"] = position_tensor;
    }
    inputs["attention_mask"] = mask_tensor;

    TensorMap outputs = decoder_->forward(inputs);

    auto it = outputs.find("logits");
    if (it == outputs.end())
    {
        throw std::runtime_error("WhisperPipeline: no 'logits' output");
    }

    const auto& logits_tensor = it->second;
    auto num_logits = logits_tensor.numel();
    logits.resize(static_cast<std::size_t>(num_logits));
    std::memcpy(logits.data(), logits_tensor.data,
                num_logits * sizeof(float));

    cache_->advance();
}

// ═══════════════════════════════════════════════════════════════════════════
// BarkPipeline
// ═══════════════════════════════════════════════════════════════════════════

BarkPipeline::BarkPipeline(
    std::unique_ptr<TrtModule> semantic,
    std::unique_ptr<TrtModule> coarse,
    std::unique_ptr<KvCache> semantic_cache,
    std::unique_ptr<KvCache> coarse_cache,
    std::vector<float> semantic_embed,
    std::vector<float> coarse_embed,
    BarkConfig config,
    cudaStream_t stream,
    std::shared_ptr<ITokenizer> tokenizer,
    std::string model_id_str)
    : semantic_(std::move(semantic))
    , coarse_(std::move(coarse))
    , semantic_cache_(std::move(semantic_cache))
    , coarse_cache_(std::move(coarse_cache))
    , semantic_embed_(std::move(semantic_embed))
    , coarse_embed_(std::move(coarse_embed))
    , config_(std::move(config))
    , stream_(stream)
    , tokenizer_(std::move(tokenizer))
    , model_id_(std::move(model_id_str))
{
    if (!semantic_ || !semantic_->ok())
        throw std::runtime_error("BarkPipeline: invalid semantic module");
    if (!coarse_ || !coarse_->ok())
        throw std::runtime_error("BarkPipeline: invalid coarse module");
    if (semantic_embed_.empty() || coarse_embed_.empty())
        throw std::runtime_error("BarkPipeline: empty embedding tables");
}

BarkPipeline::~BarkPipeline() = default;

void BarkPipeline::set_codec_module(std::unique_ptr<TrtModule> codec)
{
    codec_ = std::move(codec);
}

void BarkPipeline::set_fine_module(std::unique_ptr<TrtModule> fine)
{
    fine_ = std::move(fine);
}

void BarkPipeline::set_fine_embeddings(
    std::vector<float> embed, std::vector<float> pos_embed)
{
    fine_embed_ = std::move(embed);
    fine_position_embed_ = std::move(pos_embed);
}

AudioResult BarkPipeline::generate_audio(
    const std::string& prompt, const GenerateConfig& cfg)
{
    // Tokenize the prompt
    std::vector<int32_t> input_ids;
    if (tokenizer_)
        input_ids = tokenizer_->encode(prompt);

    int32_t max_tokens = cfg.max_new_tokens > 0 ? cfg.max_new_tokens : 768;

    maybe_enable_bark_greedy(config_);
    maybe_seed_bark_rng(rng_);

    std::cerr << "[trtf] Bark: starting pipeline with " << input_ids.size()
              << " text tokens, max_semantic=" << max_tokens
              << (config_.greedy ? " (greedy)" : "") << std::endl;

    // Stage 1: Text -> Semantic tokens
    auto semantic_tokens = run_semantic(input_ids, max_tokens);
    if (semantic_tokens.empty())
    {
        std::cerr << "[trtf] Bark: semantic stage produced no tokens" << std::endl;
        AudioResult out;
        out.sample_rate = config_.sample_rate;
        return out;
    }

    // Stage 2: Semantic -> Coarse acoustic codes
    auto coarse_tokens = run_coarse(semantic_tokens);
    if (coarse_tokens.empty())
    {
        std::cerr << "[trtf] Bark: coarse stage produced no tokens" << std::endl;
        AudioResult out;
        out.sample_rate = config_.sample_rate;
        return out;
    }

    // Stage 2.5: Fine (coarse codes -> 8 codebook codes)
    auto fine_codes = run_fine(coarse_tokens);
    const BarkCodecPlan codec_plan = make_bark_codec_plan(
        fine_codes,
        static_cast<bool>(fine_),
        coarse_tokens,
        config_.n_coarse_codebooks);

    std::vector<float> waveform = codec_plan.use_fine_codes
        ? run_codec(fine_codes, codec_plan.frame_count)
        : run_codec(coarse_tokens);
    if (waveform.empty())
    {
        std::cerr << "[trtf] Bark: codec produced no audio" << std::endl;
        AudioResult out;
        out.sample_rate = config_.sample_rate;
        return out;
    }

    AudioResult out;
    out.samples = std::move(waveform);
    out.num_samples = static_cast<int32_t>(out.samples.size());
    out.sample_rate = config_.sample_rate;
    std::cerr << "[trtf] Bark: generated " << out.num_samples << " samples ("
              << static_cast<float>(out.num_samples) / out.sample_rate
              << "s @ " << out.sample_rate << " Hz)" << std::endl;
    return out;
}

// ─── MagpiePipeline (TrtModule-based) ───

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
void BarkPipeline::run_step_with_embed(
    TrtModule& module, KvCache& cache,
    const float* embed, int32_t embed_dim,
    std::vector<float>& logits)
{
    std::vector<float> mask;
    cache.build_attention_mask(mask);
    int32_t position = cache.position();

    Tensor embed_tensor;
    embed_tensor.data = const_cast<float*>(embed);
    embed_tensor.shape = {embed_dim};
    embed_tensor.dtype = DType::kFloat32;

    float use_embed = 1.0F;
    Tensor use_embed_tensor;
    use_embed_tensor.data = &use_embed;
    use_embed_tensor.shape = {1};
    use_embed_tensor.dtype = DType::kFloat32;

    int32_t dummy_token = 0;
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

    TensorMap inputs;
    if (module.has_input("token_id"))
        inputs["token_id"] = token_tensor;
    if (module.has_input("input_embed"))
        inputs["input_embed"] = embed_tensor;
    if (module.has_input("use_input_embed"))
        inputs["use_input_embed"] = use_embed_tensor;
    if (module.has_input("position_id"))
        inputs["position_id"] = position_tensor;
    inputs["attention_mask"] = mask_tensor;

    TensorMap outputs = module.forward(inputs);

    auto it = outputs.find("logits");
    if (it == outputs.end())
        throw std::runtime_error("BarkPipeline: no 'logits' output");

    const auto& logits_tensor = it->second;
    logits.resize(static_cast<std::size_t>(logits_tensor.numel()));
    std::memcpy(logits.data(), logits_tensor.data,
                logits_tensor.numel() * sizeof(float));

    cache.advance();
}

void BarkPipeline::run_step_with_token(
    TrtModule& module, KvCache& cache,
    int32_t token_id,
    std::vector<float>& logits)
{
    std::vector<float> mask;
    cache.build_attention_mask(mask);
    int32_t position = cache.position();

    Tensor token_tensor;
    token_tensor.data = &token_id;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;

    float use_embed = 0.0F;
    Tensor use_embed_tensor;
    use_embed_tensor.data = &use_embed;
    use_embed_tensor.shape = {1};
    use_embed_tensor.dtype = DType::kFloat32;

    Tensor position_tensor;
    position_tensor.data = &position;
    position_tensor.shape = {1};
    position_tensor.dtype = DType::kInt32;

    Tensor mask_tensor;
    mask_tensor.data = mask.data();
    mask_tensor.shape = {static_cast<int64_t>(mask.size())};
    mask_tensor.dtype = DType::kFloat32;

    TensorMap inputs;
    inputs["token_id"] = token_tensor;
    if (module.has_input("use_input_embed"))
        inputs["use_input_embed"] = use_embed_tensor;
    if (module.has_input("position_id"))
        inputs["position_id"] = position_tensor;
    inputs["attention_mask"] = mask_tensor;

    TensorMap outputs = module.forward(inputs);

    auto it = outputs.find("logits");
    if (it == outputs.end())
        throw std::runtime_error("BarkPipeline: no 'logits' output");

    const auto& logits_tensor = it->second;
    logits.resize(static_cast<std::size_t>(logits_tensor.numel()));
    std::memcpy(logits.data(), logits_tensor.data,
                logits_tensor.numel() * sizeof(float));

    cache.advance();
}

int32_t BarkPipeline::sample_top_k(const float* logits, int32_t vocab_size,
                                     float temperature, int32_t top_k)
{
    if (config_.greedy)
    {
        int32_t best = 0;
        for (int32_t i = 1; i < vocab_size; ++i)
        {
            if (logits[i] > logits[best]) best = i;
        }
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
    for (int32_t i = 0; i < top_k; ++i) probs[i] /= sum;

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

// ---------------------------------------------------------------------------
// Stage 1: Semantic (text tokens -> semantic audio tokens)
// ---------------------------------------------------------------------------

std::vector<int32_t> BarkPipeline::run_semantic(
    const std::vector<int32_t>& text_ids,
    int32_t max_tokens)
{
    const auto& cfg = config_;

    semantic_cache_->reset();
    semantic_cache_->bind_to(*semantic_);

    std::vector<float> logits;
    std::vector<float> embed_a(static_cast<std::size_t>(cfg.hidden_size));
    std::vector<float> embed_b(static_cast<std::size_t>(cfg.hidden_size));
    std::vector<float> embed_buf(static_cast<std::size_t>(cfg.hidden_size));

    // Prefill text tokens
    const auto copy_len = std::min(static_cast<int32_t>(text_ids.size()), kBarkMaxTextLen);
    for (int32_t pos = 0; pos < kBarkMaxTextLen; ++pos)
    {
        const int32_t text_tok = semantic_text_token(text_ids, pos, copy_len, cfg);
        copy_embed_row(semantic_embed_.data(), text_tok, cfg.hidden_size, embed_a.data());
        copy_embed_row(
            semantic_embed_.data(), cfg.semantic_pad_token, cfg.hidden_size, embed_b.data());
        sum_embed_rows(embed_a.data(), embed_b.data(), cfg.hidden_size, embed_buf.data());
        run_step_with_embed(*semantic_, *semantic_cache_,
            embed_buf.data(), cfg.hidden_size, logits);
    }

    // Prefill infer token
    copy_embed_row(
        semantic_embed_.data(), cfg.semantic_infer_token, cfg.hidden_size, embed_buf.data());
    run_step_with_embed(*semantic_, *semantic_cache_,
        embed_buf.data(), cfg.hidden_size, logits);

    // Autoregressive generation
    std::vector<int32_t> semantic_tokens;
    semantic_tokens.reserve(static_cast<std::size_t>(max_tokens));

    for (int32_t step = 0; step < max_tokens; ++step)
    {
        if (semantic_eos_threshold_hit(logits, cfg)) break;
        suppress_semantic_logits(logits, cfg.semantic_pad_token);

        const int32_t token = sample_top_k(
            logits.data(),
            cfg.semantic_pad_token + 1,
            cfg.semantic_temperature,
            cfg.top_k);
        if (token == cfg.semantic_pad_token) break;

        semantic_tokens.push_back(token);

        // During decode, use token_id path (no embedding injection)
        run_step_with_token(*semantic_, *semantic_cache_, token, logits);
    }

    std::cerr << "[trtf] Bark semantic: generated " << semantic_tokens.size()
              << " tokens" << std::endl;
    maybe_dump_tokens(".sem_tokens", semantic_tokens);
    return semantic_tokens;
}

// ---------------------------------------------------------------------------
// Stage 2: Coarse (semantic tokens -> coarse acoustic codes)
// ---------------------------------------------------------------------------

std::vector<int32_t> BarkPipeline::run_coarse(
    const std::vector<int32_t>& semantic_tokens)
{
    const auto& cfg = config_;
    const BarkCoarsePlan coarse_plan = make_bark_coarse_plan(semantic_tokens, cfg);
    const int32_t n_steps = coarse_plan.total_steps;

    if (n_steps == 0)
    {
        std::cerr << "[trtf] Bark coarse: no steps to generate" << std::endl;
        return {};
    }

    std::vector<int32_t> x_coarse;
    x_coarse.reserve(static_cast<std::size_t>(n_steps));

    std::vector<float> logits;
    std::vector<float> embed_buf(static_cast<std::size_t>(cfg.hidden_size));

    for (int32_t win = 0; win < coarse_plan.num_windows; ++win)
    {
        const BarkCoarseWindowPlan window_plan = make_bark_coarse_window_plan(
            coarse_plan, x_coarse, cfg);
        if (window_plan.generated_this_window <= 0) break;

        // Reset cache for each window
        coarse_cache_->reset();
        coarse_cache_->bind_to(*coarse_);

        // Prefill coarse window
        for (std::size_t i = 0; i + 1 < window_plan.input_tokens.size(); ++i)
        {
            copy_embed_row(coarse_embed_.data(), window_plan.input_tokens[i],
                cfg.hidden_size, embed_buf.data());
            run_step_with_embed(*coarse_, *coarse_cache_,
                embed_buf.data(), cfg.hidden_size, logits);
        }

        // Last prefill token
        copy_embed_row(coarse_embed_.data(), window_plan.input_tokens.back(),
            cfg.hidden_size, embed_buf.data());
        run_step_with_embed(*coarse_, *coarse_cache_,
            embed_buf.data(), cfg.hidden_size, logits);

        // Generate
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
                copy_embed_row(coarse_embed_.data(), token,
                    cfg.hidden_size, embed_buf.data());
                run_step_with_embed(*coarse_, *coarse_cache_,
                    embed_buf.data(), cfg.hidden_size, logits);
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

std::vector<int32_t> BarkPipeline::run_fine(const std::vector<int32_t>& coarse_tokens)
{
    const auto& cfg = config_;
    const BarkFinePlan plan = make_bark_fine_plan(
        cfg,
        coarse_tokens.size(),
        static_cast<bool>(fine_),
        static_cast<bool>(fine_));
    std::vector<int32_t> codes = initialize_bark_fine_codes(
        coarse_tokens, plan.n_frames, cfg);

    if (!plan.should_run_trt)
    {
        std::cerr << "[trtf] Bark fine: no TRT fine engine, "
                  << "codebooks 2-7 will be zero" << std::endl;
        return codes;
    }

    const int32_t fine_hidden = cfg.fine_hidden_size;
    const int32_t fine_cb_size = cfg.fine_codebook_size;
    const int32_t max_seq = cfg.fine_seq_length;

    std::vector<float> host_embeds(
        static_cast<std::size_t>(max_seq) * fine_hidden, 0.0F);
    std::vector<float> host_logits(
        static_cast<std::size_t>(max_seq) * fine_cb_size);

    for (int32_t cb_idx = plan.first_predicted_codebook;
         cb_idx < plan.last_predicted_codebook;
         ++cb_idx)
    {
        // Build input embeddings on host
        build_fine_input_embeddings(
            host_embeds, codes, cb_idx, plan.n_frames, plan.actual_frames,
            fine_hidden, fine_cb_size, fine_embed_, fine_position_embed_);

        // Build TensorMap
        Tensor embed_tensor;
        embed_tensor.data = host_embeds.data();
        embed_tensor.shape = {max_seq, fine_hidden};
        embed_tensor.dtype = DType::kFloat32;

        TensorMap inputs;
        inputs["input_embeds"] = embed_tensor;

        TensorMap outputs = fine_->forward(inputs);

        // Read the correct codebook head output
        const int32_t head_idx = cb_idx - 1;
        const std::string head_name = "logits_cb" + std::to_string(head_idx + 1);
        auto it = outputs.find(head_name);
        if (it == outputs.end())
        {
            std::cerr << "[trtf] Bark fine: missing output " << head_name << std::endl;
            return codes;
        }

        const auto& logits_tensor = it->second;
        const std::size_t logits_bytes = std::min(
            static_cast<std::size_t>(max_seq) * fine_cb_size * sizeof(float),
            logits_tensor.nbytes());
        std::memcpy(host_logits.data(), logits_tensor.data, logits_bytes);

        update_fine_codes_from_logits(
            codes, host_logits, cb_idx, plan.n_frames, plan.actual_frames,
            fine_cb_size, cfg.codebook_size);
    }

    std::cerr << "[trtf] Bark fine: predicted codebooks 2-7 for "
              << plan.n_frames << " frames" << std::endl;
    return codes;
}

// ---------------------------------------------------------------------------
// Stage 3: Codec (codes -> waveform)
// ---------------------------------------------------------------------------

std::vector<float> BarkPipeline::run_codec(const std::vector<int32_t>& coarse_tokens)
{
    const auto& cfg = config_;
    const int32_t n_frames = static_cast<int32_t>(coarse_tokens.size()) /
        cfg.n_coarse_codebooks;

    if (n_frames == 0) return {};

    // De-interleave coarse tokens to [n_codebooks, n_frames]
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

    if (!codec_ || !codec_->ok() || cfg.codec_seq_length <= 0)
    {
        std::cerr << "[trtf] Bark codec: no TRT codec engine, "
                  << "generating simple waveform from codes" << std::endl;
        return synthesize_simple_waveform(codes, n_frames, cfg);
    }
    return run_codec(codes, n_frames);
}

std::vector<float> BarkPipeline::run_codec(const std::vector<int32_t>& codes_flat,
                                            int32_t n_frames)
{
    const auto& cfg = config_;

    if (n_frames <= 0) return {};

    if (!codec_ || !codec_->ok() || cfg.codec_seq_length <= 0)
    {
        std::cerr << "[trtf] Bark codec: no TRT codec engine, "
                  << "generating simple waveform from codes" << std::endl;
        return synthesize_simple_waveform(codes_flat, n_frames, cfg);
    }

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

    // Determine source codebooks from codes_flat layout
    const int32_t source_codebooks = static_cast<int32_t>(codes_flat.size()) / std::max(n_frames, 1);
    std::vector<int32_t> input_codes = make_bark_codec_input_codes(
        codes_flat, source_codebooks, n_frames, n_cb, max_T, actual_frames);

    // Build TensorMap for codec
    Tensor codes_tensor;
    codes_tensor.data = input_codes.data();
    codes_tensor.shape = {n_cb, max_T};
    codes_tensor.dtype = DType::kInt32;

    TensorMap inputs;
    inputs["audio_codes"] = codes_tensor;

    TensorMap outputs = codec_->forward(inputs);

    auto it = outputs.find("waveform");
    if (it == outputs.end())
    {
        std::cerr << "[trtf] Bark codec: no 'waveform' output" << std::endl;
        return {};
    }

    const auto& wav_tensor = it->second;
    const auto total_elems = static_cast<std::size_t>(max_T) * upsample;
    const auto trimmed = static_cast<std::size_t>(actual_frames) * upsample;
    const auto* wav_data = static_cast<const float*>(wav_tensor.data);

    std::vector<float> waveform(wav_data, wav_data + std::min(trimmed, total_elems));

    std::cerr << "[trtf] Bark codec: TRT decode " << actual_frames << " frames -> "
              << waveform.size() << " samples" << std::endl;
    return waveform;
}

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

// ---------------------------------------------------------------------------
// Constructor helpers (extracted to reduce cyclomatic complexity)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Embedding helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// run_encoder() -- text IDs -> encoder_output via TrtModule
// ---------------------------------------------------------------------------

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

// ─── SpeechPipeline (TrtModule-based) ───

SpeechPipeline::SpeechPipeline(
    std::unique_ptr<TrtModule> mimi_encoder,
    std::unique_ptr<TrtModule> temporal,
    std::unique_ptr<KvCache> temporal_cache,
    std::vector<std::unique_ptr<TrtModule>> depth_engines,
    std::unique_ptr<KvCache> depth_cache,
    std::unique_ptr<TrtModule> mimi_decoder,
    SpeechConfig config,
    cudaStream_t stream,
    std::shared_ptr<ISubprocessRunner> subprocess_runner,
    std::string model_id_str)
    : mimi_encoder_(std::move(mimi_encoder))
    , temporal_(std::move(temporal))
    , temporal_cache_(std::move(temporal_cache))
    , depth_engines_(std::move(depth_engines))
    , depth_cache_(std::move(depth_cache))
    , mimi_decoder_(std::move(mimi_decoder))
    , stream_(stream)
    , config_(std::move(config))
    , subprocess_runner_(std::move(subprocess_runner))
    , model_id_(std::move(model_id_str))
{
    if (!temporal_ || !temporal_->ok())
        throw std::runtime_error("SpeechPipeline: invalid temporal module");
    if (!temporal_cache_ || !temporal_cache_->ok())
        throw std::runtime_error("SpeechPipeline: invalid temporal cache");

    if (!subprocess_runner_)
        subprocess_runner_ = CreateDefaultSubprocessRunner();

    // Fixed deterministic seed for reproducible audio output.
    // PersonaPlex depth sampling (temperature=0.8, top_k=250) is sensitive
    // to the RNG sequence; a fixed seed ensures identical output across runs
    // and between the old and new pipeline paths.
    rng_state_ = 0x5EEDC0DECAFE1234ULL;
}

SpeechPipeline::~SpeechPipeline() = default;

// ---------------------------------------------------------------------------
// Mimi Encoder: audio waveform -> codec tokens
// ---------------------------------------------------------------------------

namespace {

struct MimiEncoderShapes {
    int32_t engine_input_samples{0};
    int32_t enc_codebooks{0};
    int32_t enc_frames{0};
};

MimiEncoderShapes query_mimi_encoder_shapes(const TrtModule& module)
{
    MimiEncoderShapes s;
    for (const auto& info : module.input_info())
    {
        if (info.name == "audio_input" && !info.shape.empty())
            s.engine_input_samples = static_cast<int32_t>(info.shape.back());
    }
    for (const auto& info : module.output_info())
    {
        if (info.name == "codec_tokens" && info.shape.size() >= 2)
        {
            s.enc_codebooks = static_cast<int32_t>(info.shape[0]);
            s.enc_frames = static_cast<int32_t>(info.shape[1]);
        }
    }
    return s;
}

std::vector<int32_t> transpose_codec_tokens_to_frame_major(
    const float* data, int32_t codebooks, int32_t frames)
{
    const auto output_elems = static_cast<std::size_t>(codebooks) * frames;
    std::vector<int32_t> tokens(output_elems);
    for (int32_t cb = 0; cb < codebooks; ++cb)
    {
        for (int32_t frame = 0; frame < frames; ++frame)
        {
            const auto src = static_cast<std::size_t>(cb) * frames + frame;
            const auto dst = static_cast<std::size_t>(frame) * codebooks + cb;
            tokens[dst] = static_cast<int32_t>(std::round(data[src]));
        }
    }
    return tokens;
}

void log_first_n_tokens(const char* label, const std::vector<int32_t>& tokens, int32_t n = 16)
{
    std::cerr << label;
    for (int32_t i = 0; i < std::min(n, static_cast<int32_t>(tokens.size())); ++i)
        std::cerr << tokens[static_cast<std::size_t>(i)] << " ";
    std::cerr << std::endl;
}

} // anonymous namespace

std::vector<int32_t> SpeechPipeline::run_mimi_encode(
    const float* samples, int32_t num_samples)
{
    last_encode_frames_ = 0;
    last_encode_codebooks_ = 0;

    if (!mimi_encoder_ || !mimi_encoder_->ok())
    {
        std::cerr << "[speech] No Mimi TRT encoder available" << std::endl;
        return {};
    }

    const auto shapes = query_mimi_encoder_shapes(*mimi_encoder_);

    if (num_samples != shapes.engine_input_samples)
    {
        std::cerr << "[speech] WARNING: input samples " << num_samples
                  << " != engine expects " << shapes.engine_input_samples
                  << ", using engine size" << std::endl;
    }

    std::cerr << "[speech] Mimi encoder TRT: input [1,1,"
              << shapes.engine_input_samples << "], output ["
              << shapes.enc_codebooks << "," << shapes.enc_frames << "]"
              << std::endl;

    // Prepare input: pad or truncate to match engine size.
    const auto input_elems = static_cast<std::size_t>(shapes.engine_input_samples);
    std::vector<float> input_buf(input_elems, 0.0F);
    const auto copy_n = std::min(static_cast<std::size_t>(num_samples), input_elems);
    std::memcpy(input_buf.data(), samples, copy_n * sizeof(float));

    Tensor audio_input_tensor;
    audio_input_tensor.data = input_buf.data();
    audio_input_tensor.shape = {1, 1, static_cast<int64_t>(shapes.engine_input_samples)};
    audio_input_tensor.dtype = DType::kFloat32;

    TensorMap inputs;
    inputs["audio_input"] = audio_input_tensor;

    TensorMap outputs = mimi_encoder_->forward(inputs);

    auto it = outputs.find("codec_tokens");
    if (it == outputs.end())
    {
        std::cerr << "[speech] Mimi encoder: no 'codec_tokens' output" << std::endl;
        return {};
    }

    const auto& out_tensor = it->second;
    auto tokens = transpose_codec_tokens_to_frame_major(
        static_cast<const float*>(out_tensor.data),
        shapes.enc_codebooks, shapes.enc_frames);

    std::cerr << "[speech] Mimi encode (TRT): " << num_samples << " samples -> "
              << shapes.enc_frames << " frames x " << shapes.enc_codebooks
              << " codebooks" << std::endl;

    last_encode_frames_ = shapes.enc_frames;
    last_encode_codebooks_ = shapes.enc_codebooks;

    log_first_n_tokens("[speech] Encoder tokens [0:16]: ", tokens);

    return tokens;
}

// ---------------------------------------------------------------------------
// Temporal step with KvCache: input_embed -> logits (+ hidden_state)
// ---------------------------------------------------------------------------

void SpeechPipeline::run_temporal_embed_step(
    const float* embed_ptr, int32_t embed_size,
    std::vector<float>& logits,
    std::vector<float>& hidden_out)
{
    std::vector<float> mask;
    temporal_cache_->build_attention_mask(mask);
    temporal_cache_->bind_to(*temporal_);

    int32_t position = temporal_cache_->position();
    float use_input_embed = 1.0F;
    int32_t dummy_token = 0;

    // Copy embed to mutable buffer (Tensor requires non-const pointer)
    std::vector<float> embed_buf(embed_ptr, embed_ptr + embed_size);

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
    if (temporal_->has_input("position_id"))
        inputs["position_id"] = position_tensor;
    inputs["attention_mask"] = mask_tensor;
    inputs["input_embed"] = embed_tensor;
    inputs["use_input_embed"] = use_embed_tensor;

    TensorMap outputs = temporal_->forward(inputs);

    // Extract logits
    auto logits_it = outputs.find("logits");
    if (logits_it == outputs.end())
        throw std::runtime_error("SpeechPipeline temporal: no 'logits' output");

    const auto& lt = logits_it->second;
    auto n = lt.numel();
    logits.resize(static_cast<std::size_t>(n));
    std::memcpy(logits.data(), lt.data, n * sizeof(float));

    // Extract hidden_state if available
    auto hidden_it = outputs.find("hidden_state");
    if (hidden_it != outputs.end())
    {
        const auto& ht = hidden_it->second;
        auto hn = ht.numel();
        hidden_out.resize(static_cast<std::size_t>(hn));
        std::memcpy(hidden_out.data(), ht.data, hn * sizeof(float));
    }
    else
    {
        // Fallback: use logits as hidden representation (truncated/padded to hidden_size)
        hidden_out.clear();
    }

    temporal_cache_->advance();
}

// ---------------------------------------------------------------------------
// Depth step: generate num_codebooks tokens
// ---------------------------------------------------------------------------

namespace {

int32_t speech_clamp_token(int32_t token, int32_t vocab_size)
{
    return clamp_speech_depth_token(token, vocab_size);
}

bool speech_is_sampling_enabled(const SpeechConfig& cfg)
{
    return cfg.depth_temperature > 0.0F && cfg.depth_top_k > 0;
}

int32_t speech_select_depth_token_greedy(
    const std::vector<float>& logits,
    const SpeechConfig&,
    uint64_t&)
{
    return select_argmax_token(logits);
}

int32_t speech_select_depth_token_sampled(
    const std::vector<float>& logits,
    const SpeechConfig& cfg,
    uint64_t& rng_state)
{
    return sample_token_topk(
        logits, cfg.depth_temperature, cfg.depth_top_k, rng_state);
}

using SpeechDepthTokenSelectFn = int32_t (*)(
    const std::vector<float>&, const SpeechConfig&, uint64_t&);

SpeechDepthTokenSelectFn speech_select_depth_token_dispatch(const SpeechConfig& cfg)
{
    return speech_is_sampling_enabled(cfg)
        ? speech_select_depth_token_sampled
        : speech_select_depth_token_greedy;
}

int32_t speech_sample_temporal_text_token(
    const std::vector<float>& logits,
    int32_t text_pad_id,
    const SpeechConfig& cfg,
    uint64_t& rng_state)
{
    if (logits.empty())
        return text_pad_id;
    if (speech_is_sampling_enabled(cfg))
    {
        constexpr float kTextTemp = 0.7F;
        constexpr int32_t kTextTopK = 25;
        return sample_token_topk(logits, kTextTemp, kTextTopK, rng_state);
    }
    return select_argmax_token(logits);
}

void speech_maybe_log_depth_debug(
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

} // anonymous namespace

std::vector<int32_t> SpeechPipeline::run_depth(
    const float* temporal_hidden, int32_t hidden_dim,
    int32_t text_token,
    const int32_t* forced_audio_tokens,
    const uint8_t* forced_audio_provided)
{
    (void) hidden_dim;

    const auto& cfg = config_;
    const int32_t num_cb = cfg.num_codebooks;
    const int32_t depth_hidden = cfg.depth_hidden_size;

    if (depth_engines_.empty())
        return std::vector<int32_t>(static_cast<std::size_t>(num_cb), 0);

    // Reset shared depth cache for this frame
    depth_cache_->reset();

    std::vector<int32_t> codebook_tokens;
    codebook_tokens.reserve(static_cast<std::size_t>(num_cb));
    const int32_t depth_call_idx = depth_debug_call_count_++;

    std::vector<float> logits;
    const auto projection_view = make_depth_projection_view(cfg, temporal_hidden);

    std::vector<float> depth_embed(static_cast<std::size_t>(depth_hidden), 0.0F);
    const auto select_token = speech_select_depth_token_dispatch(cfg);

    int32_t prev_token = 0;

    for (int32_t cb = 0; cb < num_cb; ++cb)
    {
        const auto cb_idx = static_cast<std::size_t>(cb);
        auto* engine = (cb_idx < depth_engines_.size() && depth_engines_[cb_idx])
            ? depth_engines_[cb_idx].get()
            : depth_engines_[0].get();

        build_depth_input_embedding(
            cfg,
            projection_view,
            cb,
            text_token,
            prev_token,
            depth_hidden,
            depth_embed);

        // Bind depth cache and run with input_embed
        std::vector<float> mask;
        depth_cache_->build_attention_mask(mask);
        depth_cache_->bind_to(*engine);

        int32_t position = depth_cache_->position();
        float use_input_embed = 1.0F;
        int32_t dummy_token = 0;

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
        embed_tensor.data = depth_embed.data();
        embed_tensor.shape = {static_cast<int64_t>(depth_hidden)};
        embed_tensor.dtype = DType::kFloat32;

        Tensor use_embed_tensor;
        use_embed_tensor.data = &use_input_embed;
        use_embed_tensor.shape = {1};
        use_embed_tensor.dtype = DType::kFloat32;

        TensorMap inputs;
        inputs["token_id"] = token_tensor;
        if (engine->has_input("position_id"))
            inputs["position_id"] = position_tensor;
        inputs["attention_mask"] = mask_tensor;
        inputs["input_embed"] = embed_tensor;
        inputs["use_input_embed"] = use_embed_tensor;

        TensorMap outputs = engine->forward(inputs);

        auto logits_it = outputs.find("logits");
        if (logits_it == outputs.end())
        {
            std::cerr << "[speech] Depth step cb=" << cb << " failed: no logits"
                      << std::endl;
            break;
        }

        const auto& lt = logits_it->second;
        auto n = lt.numel();
        logits.resize(static_cast<std::size_t>(n));
        std::memcpy(logits.data(), lt.data, n * sizeof(float));

        depth_cache_->advance();

        int32_t best = select_token(logits, cfg, rng_state_);
        best = std::max(0, std::min(best, cfg.codebook_size - 1));
        codebook_tokens.push_back(best);
        prev_token = resolve_depth_prev_token(
            cb, best, cfg, forced_audio_tokens, forced_audio_provided);
        speech_maybe_log_depth_debug(cb, depth_call_idx, logits, best);
    }

    // Pad if we stopped early
    while (static_cast<int32_t>(codebook_tokens.size()) < num_cb)
        codebook_tokens.push_back(0);

    return codebook_tokens;
}

// ---------------------------------------------------------------------------
// Mimi Decoder: codec tokens -> waveform
// ---------------------------------------------------------------------------

namespace {

struct MimiDecoderShapes {
    int32_t dec_codebooks{0};
    int32_t dec_frames{0};
    std::vector<int32_t> output_dims;
};

MimiDecoderShapes query_mimi_decoder_shapes(const TrtModule& module)
{
    MimiDecoderShapes s;
    for (const auto& info : module.input_info())
    {
        if (info.name == "codec_tokens" && info.shape.size() >= 2)
        {
            s.dec_codebooks = static_cast<int32_t>(info.shape[0]);
            s.dec_frames = static_cast<int32_t>(info.shape[1]);
        }
    }
    for (const auto& info : module.output_info())
    {
        if (info.name == "audio_output")
        {
            s.output_dims.reserve(info.shape.size());
            for (auto d : info.shape)
                s.output_dims.push_back(static_cast<int32_t>(d));
        }
    }
    return s;
}

} // anonymous namespace

std::vector<float> SpeechPipeline::run_mimi_decode(
    const std::vector<int32_t>& codec_tokens, int32_t num_frames)
{
    if (num_frames <= 0)
        return {};

    int32_t actual_codebooks = 0;
    if (!codec_tokens.empty())
        actual_codebooks = static_cast<int32_t>(codec_tokens.size()) / num_frames;

    if (!mimi_decoder_ || !mimi_decoder_->ok())
    {
        std::cerr << "[speech] No Mimi TRT decoder available" << std::endl;
        return {};
    }

    const auto shapes = query_mimi_decoder_shapes(*mimi_decoder_);
    const auto layout = build_mimi_decode_layout(
        shapes.dec_codebooks, shapes.dec_frames, shapes.output_dims);

    std::cerr << "[speech] Mimi decoder TRT: input [" << layout.dec_codebooks
              << "," << layout.dec_frames << "], output " << layout.total_output_elems
              << " samples" << std::endl;

    auto input_tokens = build_mimi_decoder_input(
        codec_tokens, num_frames, actual_codebooks,
        layout.dec_frames, layout.dec_codebooks);

    // Debug: print first few input tokens
    std::cerr << "[speech] Decoder input tokens [0:16]: ";
    for (int32_t i = 0; i < std::min(16, static_cast<int32_t>(layout.input_elems)); ++i)
        std::cerr << input_tokens[static_cast<std::size_t>(i)] << " ";
    std::cerr << std::endl;

    Tensor codec_tensor;
    codec_tensor.data = input_tokens.data();
    codec_tensor.shape = {static_cast<int64_t>(shapes.dec_codebooks),
                          static_cast<int64_t>(shapes.dec_frames)};
    codec_tensor.dtype = DType::kFloat32;

    TensorMap inputs;
    inputs["codec_tokens"] = codec_tensor;

    TensorMap outputs = mimi_decoder_->forward(inputs);

    auto it = outputs.find("audio_output");
    if (it == outputs.end())
    {
        std::cerr << "[speech] Mimi decoder: no 'audio_output' output" << std::endl;
        return {};
    }

    const auto& out_tensor = it->second;
    const auto total_elems = static_cast<std::size_t>(layout.total_output_elems);
    std::vector<float> waveform(total_elems);
    std::memcpy(waveform.data(), out_tensor.data, total_elems * sizeof(float));

    float rms = 0.0F;
    float mx = 0.0F;
    waveform_stats(waveform, layout.total_output_elems, rms, mx);
    std::cerr << "[speech] Mimi decode (TRT): " << layout.dec_frames << " frames -> "
              << layout.total_output_elems << " samples (RMS=" << rms
              << ", Max=" << mx << ")" << std::endl;
    return waveform;
}

// ---------------------------------------------------------------------------
// Text Prompt Injection
// ---------------------------------------------------------------------------

namespace {

// Resolve text prompt tokens: use pre-tokenized, runtime-tokenize, or empty.
// Returns true if tokens were resolved; false means skip text prompt.
bool resolve_text_prompt_tokens(
    const SpeechConfig& cfg,
    ISubprocessRunner& subprocess_runner,
    std::vector<int32_t>& text_tokens)
{
    text_tokens = cfg.text_prompt_ids;
    if (!text_tokens.empty())
    {
        std::cerr << "[speech] Injecting pre-tokenized text prompt ("
                  << text_tokens.size() << " tokens)" << std::endl;
        return true;
    }
    if (!cfg.system_prompt.empty() && !cfg.hf_python.empty())
    {
        auto tokenization = TokenizeSpeechPromptRuntime(
            cfg.hf_python, cfg.system_prompt, subprocess_runner);
        if (tokenization.rc != 0 || tokenization.tokens.empty())
        {
            std::cerr << "[speech] Text prompt tokenization failed (rc="
                      << tokenization.rc << "): "
                      << tokenization.stderr_data << std::endl;
            return false;
        }
        text_tokens = std::move(tokenization.tokens);
        std::cerr << "[speech] Injecting runtime-tokenized text prompt: \""
                  << cfg.system_prompt << "\" (" << text_tokens.size()
                  << " tokens)" << std::endl;
        return true;
    }
    return false;
}

void compute_text_prompt_frame_embed(
    const SpeechConfig& cfg, int32_t text_token_id,
    int32_t hidden, float* summed_embed)
{
    std::fill(summed_embed, summed_embed + hidden, 0.0F);
    const int32_t text_tok = speech_clamp_token(text_token_id, cfg.temporal_text_vocab);
    const auto text_offset = static_cast<std::size_t>(text_tok) * hidden;
    add_speech_embedding_row(
        cfg.temporal_text_embedding, text_offset, hidden, summed_embed);

    const int32_t audio_vocab = cfg.audio_vocab_size;
    const int32_t bos = speech_clamp_token(cfg.codebook_size, audio_vocab);
    const auto emb_stride_cb = static_cast<std::size_t>(audio_vocab) * hidden;
    for (int32_t cb = 0; cb < cfg.num_codebooks; ++cb)
    {
        const auto emb_offset = static_cast<std::size_t>(cb) * emb_stride_cb
            + static_cast<std::size_t>(bos) * hidden;
        add_speech_embedding_row(
            cfg.audio_embeddings, emb_offset, hidden, summed_embed);
    }
}

} // anonymous namespace

void SpeechPipeline::run_text_prompt()
{
    const auto& cfg = config_;
    const int32_t hidden = cfg.temporal_hidden_size;

    if (cfg.temporal_text_embedding.empty() || cfg.temporal_text_vocab <= 0
        || !temporal_->has_input("input_embed"))
    {
        std::cerr << "[speech] Cannot inject text prompt: missing embeddings"
                  << std::endl;
        return;
    }

    std::vector<int32_t> text_tokens;
    if (!resolve_text_prompt_tokens(cfg, *subprocess_runner_, text_tokens))
        return;

    std::vector<float> summed_embed(static_cast<std::size_t>(hidden));
    std::vector<float> logits;
    std::vector<float> hidden_out;

    for (std::size_t t = 0; t < text_tokens.size(); ++t)
    {
        compute_text_prompt_frame_embed(cfg, text_tokens[t], hidden, summed_embed.data());
        run_temporal_embed_step(summed_embed.data(), hidden, logits, hidden_out);
    }

    std::cerr << "[speech] Text prompt injection complete ("
              << text_tokens.size() << " temporal steps)" << std::endl;
}

// ---------------------------------------------------------------------------
// Interleaved generation helpers (free functions, SpeechPipeline-specific)
// ---------------------------------------------------------------------------

namespace {

void speech_log_depth_mode(const SpeechConfig& cfg)
{
    if (speech_is_sampling_enabled(cfg))
    {
        std::cerr << "[speech] Depth sampling: temperature="
                  << cfg.depth_temperature << " top_k="
                  << cfg.depth_top_k << std::endl;
        return;
    }
    std::cerr << "[speech] Depth decoding: greedy (argmax)" << std::endl;
}

void speech_log_stop_configuration(
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

void speech_maybe_log_stop_decision(
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

void speech_maybe_log_interleaved_debug(
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

void speech_log_output_frames_debug(
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

} // anonymous namespace

// ---------------------------------------------------------------------------
// speak(): Full speech-to-speech pipeline
// ---------------------------------------------------------------------------

bool SpeechPipeline::speak_validate_dual_stream() const
{
    const bool has_audio_emb = !config_.audio_embeddings.empty()
                                && config_.audio_vocab_size > 0;
    const bool has_input_embed = temporal_->has_input("input_embed");
    (void) temporal_->has_output("hidden_state");
    if (!has_audio_emb || !has_input_embed)
    {
        std::cerr << "[speech] ERROR: dual-stream requires audio_embeddings "
                     "and input_embed support" << std::endl;
        return false;
    }
    return true;
}

void SpeechPipeline::speak_run_generation_loop(
    const SpeechGenerationSettings& settings,
    const SpeechOutputPlan& plan,
    DelayCacheState& delay_state,
    const std::vector<int32_t>& codec_tokens,
    std::vector<int32_t>& output_codes,
    int32_t& frames_collected)
{
    const int32_t hidden = settings.hidden;
    SpeechDecodeStopState stop_state;
    speech_log_stop_configuration(config_, plan.extra_tail);

    std::vector<float> summed_embed(static_cast<std::size_t>(hidden));
    std::vector<float> frame_hidden(static_cast<std::size_t>(hidden));
    std::vector<float> logits;
    std::vector<float> hidden_out;
    std::vector<int32_t> moshi_input(static_cast<std::size_t>(settings.stream_cb));
    std::vector<int32_t> user_input(static_cast<std::size_t>(settings.stream_cb));
    std::vector<int32_t> target_audio_tokens(static_cast<std::size_t>(settings.num_cb));
    std::vector<uint8_t> target_audio_provided(static_cast<std::size_t>(settings.num_cb));

    frames_collected = 0;
    for (int32_t offset = 0;
         offset < plan.total_iters && frames_collected < plan.output_frames;
         ++offset)
    {
        write_user_tokens_to_delay_cache(
            delay_state, codec_tokens, offset,
            settings.stream_cb, settings.num_frames,
            settings.encode_codebooks, settings.audio_bos);
        fill_initial_delay_tokens(delay_state, offset,
            settings.text_bos, settings.audio_bos);
        if (offset == 0)
        {
            seed_delay_offset_zero(delay_state,
                settings.text_bos, settings.audio_bos);
            continue;
        }

        const int32_t model_input_pos = offset - 1;
        const int32_t target_pos = offset;

        int32_t text_input = settings.text_pad_id;
        read_model_inputs_from_delay_cache(
            delay_state, model_input_pos, settings.stream_cb,
            text_input, moshi_input, user_input);
        compute_dual_stream_summed_embed(
            config_, settings.hidden, settings.stream_cb,
            moshi_input.data(), user_input.data(),
            text_input, summed_embed.data());

        run_temporal_embed_step(
            summed_embed.data(), settings.hidden, logits, hidden_out);

        if (!hidden_out.empty())
        {
            frame_hidden.resize(static_cast<std::size_t>(settings.hidden));
            const auto copy_sz = std::min(hidden_out.size(),
                static_cast<std::size_t>(settings.hidden));
            std::memcpy(frame_hidden.data(), hidden_out.data(),
                copy_sz * sizeof(float));
        }
        else
        {
            fill_hidden_from_logits(frame_hidden, logits, settings.hidden);
        }

        const int32_t sampled_text_token = speech_sample_temporal_text_token(
            logits, settings.text_pad_id, config_, rng_state_);
        const auto text_target_idx = delay_cache_index(delay_state, 0, target_pos);
        const bool text_provided = delay_state.provided[text_target_idx] != 0;
        const int32_t next_text_token =
            text_provided ? delay_state.cache[text_target_idx] : sampled_text_token;

        build_target_audio_arrays(
            delay_state, target_pos, settings.num_cb, settings.audio_bos,
            target_audio_tokens, target_audio_provided);
        auto frame_codes = run_depth(
            frame_hidden.data(), settings.hidden,
            next_text_token,
            target_audio_tokens.data(),
            target_audio_provided.data());

        clear_provided_flags_at_pos(delay_state, model_input_pos);
        write_generated_tokens_to_delay_cache(
            delay_state, target_pos, sampled_text_token,
            text_provided, frame_codes, settings.num_cb);
        if (collect_output_codes_from_delay_cache(
                delay_state, offset, delay_state.max_delay,
                settings.mimi_cb, output_codes))
        {
            ++frames_collected;
        }

        SpeechDecodeStopInput stop_input;
        stop_input.text_eos_token_id = config_.text_eos_token_id;
        stop_input.text_padding_id = config_.text_padding_id;
        stop_input.effective_frames = plan.effective_frames;
        stop_input.extra_tail = plan.extra_tail;
        stop_input.target_pos = target_pos;
        stop_input.sampled_text_token = sampled_text_token;
        stop_input.offset = offset;
        stop_input.max_delay = delay_state.max_delay;
        stop_input.text_provided = text_provided;
        const auto stop_decision = UpdateSpeechDecodeStopState(stop_state, stop_input);
        stop_state = stop_decision.state;
        speech_maybe_log_stop_decision(stop_decision.reason, stop_state, offset);
        speech_maybe_log_interleaved_debug(
            offset, settings.hidden, frame_hidden,
            text_input, sampled_text_token, frame_codes);
        if (stop_decision.should_break)
            break;
    }
}

void SpeechPipeline::speak_postprocess_waveform(
    std::vector<float>& waveform, int32_t generated_frames) const
{
    const auto trim_result = trim_speech_waveform_to_generated_frames(
        config_.sample_rate, config_.frame_rate, generated_frames, waveform);
    if (trim_result.trimmed)
    {
        std::cerr << "[speech] Trimmed decoded waveform to "
                  << trim_result.expected_samples << " samples ("
                  << generated_frames << " generated frames)" << std::endl;
    }

    const auto normalize_result = peak_normalize_speech_waveform(waveform);
    if (normalize_result.normalized)
    {
        std::cerr << "[speech] Peak-normalized: peak=" << normalize_result.peak
                  << " scale=" << normalize_result.scale << std::endl;
    }
}

AudioResult SpeechPipeline::speak(
    const float* audio_in, int32_t num_samples, const GenerateConfig& cfg,
    int32_t input_sample_rate)
{
    AudioResult result;
    result.sample_rate = config_.sample_rate;

    depth_debug_call_count_ = 0;

    const int32_t max_output_frames = cfg.max_new_tokens > 0
        ? cfg.max_new_tokens : 375;

    // Resample if the input sample rate differs from the model's expected rate.
    const float* samples_ptr = audio_in;
    int32_t samples_count = num_samples;
    std::vector<float> resampled_buf;
    const int32_t target_rate = config_.sample_rate;

    if (input_sample_rate > 0 && target_rate > 0
        && input_sample_rate != target_rate)
    {
        std::cerr << "[speech] Resampling audio from " << input_sample_rate
                  << " Hz to " << target_rate << " Hz" << std::endl;
        resampled_buf = resample_linear(
            audio_in, num_samples, input_sample_rate, target_rate);
        samples_ptr = resampled_buf.data();
        samples_count = static_cast<int32_t>(resampled_buf.size());
        input_sample_rate = target_rate;
    }

    std::cerr << "[speech] Starting pipeline with " << samples_count
              << " input samples" << std::endl;

    speech_log_depth_mode(config_);

    // Stage 1: Encode input audio via Mimi
    auto codec_tokens = run_mimi_encode(samples_ptr, samples_count);

    const auto encoder_shape = resolve_encoder_shape_without_engine(
        config_,
        last_encode_codebooks_,
        last_encode_frames_,
        codec_tokens.size());
    const int32_t num_frames = encoder_shape.num_frames;

    std::cerr << "[speech] Encoder output: " << codec_tokens.size()
              << " tokens = " << num_frames << " frames x "
              << encoder_shape.encode_codebooks << " codebooks" << std::endl;

    if (num_frames <= 0)
    {
        std::cerr << "[speech] Encoder produced no frames" << std::endl;
        return result;
    }

    if (!speak_validate_dual_stream())
        return result;

    temporal_cache_->reset();

    if (should_run_text_prompt_injection(config_))
        run_text_prompt();

    const int32_t num_cb = config_.num_codebooks;
    const int32_t hidden = config_.temporal_hidden_size;
    auto delay_state = make_delay_cache_state(config_.delays, num_cb);
    SpeechOutputPlanInput plan_input;
    plan_input.sample_rate = config_.sample_rate;
    plan_input.frame_rate = config_.frame_rate;
    plan_input.num_frames = num_frames;
    plan_input.num_input_samples = samples_count;
    plan_input.input_sample_rate = input_sample_rate;
    plan_input.tail_frames = cfg.tail_frames;
    plan_input.max_output_frames = max_output_frames;
    plan_input.max_delay = delay_state.max_delay;
    const auto plan = ComputeSpeechOutputPlan(plan_input);
    const int32_t mimi_cb = config_.mimi_decode_codebooks;
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
        config_, hidden, encoder_shape);

    int32_t frames_collected = 0;
    speak_run_generation_loop(
        settings, plan, delay_state, codec_tokens,
        output_codes, frames_collected);

    const int32_t generated_frames = frames_collected;
    std::cerr << "[speech] Depth: generated " << generated_frames
              << " frames x " << num_cb << " codebooks (decoding first "
              << mimi_cb << ")" << std::endl;
    speech_log_output_frames_debug(output_codes, generated_frames, mimi_cb);

    // Stage 4: Decode output tokens to audio via Mimi decoder
    auto waveform = run_mimi_decode(output_codes, generated_frames);
    speak_postprocess_waveform(waveform, generated_frames);

    result.samples = std::move(waveform);
    result.num_samples = static_cast<int32_t>(result.samples.size());
    std::cerr << "[speech] Generated " << result.num_samples << " samples ("
              << static_cast<float>(result.num_samples) / result.sample_rate
              << "s @ " << result.sample_rate << " Hz)" << std::endl;
    return result;
}

// ─── OmniPipeline (TrtModule-based) ───

OmniPipeline::OmniPipeline(
    std::unique_ptr<TrtModule> thinker,
    std::unique_ptr<KvCache> thinker_cache,
    std::unique_ptr<TrtModule> talker,
    std::unique_ptr<KvCache> talker_cache,
    std::unique_ptr<TrtModule> code2wav,
    OmniConfig config,
    cudaStream_t stream,
    std::shared_ptr<ITokenizer> tokenizer,
    std::string model_id_str)
    : thinker_(std::move(thinker))
    , thinker_cache_(std::move(thinker_cache))
    , talker_(std::move(talker))
    , talker_cache_(std::move(talker_cache))
    , code2wav_(std::move(code2wav))
    , config_(std::make_unique<OmniConfig>(std::move(config)))
    , stream_(stream)
    , tokenizer_(std::move(tokenizer))
    , model_id_(std::move(model_id_str))
{
    if (!thinker_ || !thinker_->ok())
        throw std::runtime_error("OmniPipeline: invalid thinker module");
    if (!thinker_cache_ || !thinker_cache_->ok())
        throw std::runtime_error("OmniPipeline: invalid thinker cache");
}

OmniPipeline::~OmniPipeline() = default;

void OmniPipeline::run_thinker_step(int32_t token_id, std::vector<float>& logits)
{
    std::vector<float> mask;
    thinker_cache_->build_attention_mask(mask);
    int32_t position = thinker_cache_->position();

    Tensor token_tensor;
    token_tensor.data = &token_id;
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

    TensorMap inputs;
    inputs["token_id"] = token_tensor;
    if (thinker_->has_input("position_id"))
        inputs["position_id"] = position_tensor;
    inputs["attention_mask"] = mask_tensor;

    TensorMap outputs = thinker_->forward(inputs);

    auto it = outputs.find("logits");
    if (it == outputs.end())
        throw std::runtime_error("OmniPipeline thinker: no 'logits' output");

    const auto& lt = it->second;
    auto n = lt.numel();
    logits.resize(static_cast<std::size_t>(n));
    std::memcpy(logits.data(), lt.data, n * sizeof(float));

    thinker_cache_->advance();
}

void OmniPipeline::run_talker_embed_step(
    const float* embed_ptr, int32_t embed_size,
    std::vector<float>& logits)
{
    std::vector<float> mask;
    talker_cache_->build_attention_mask(mask);
    int32_t position = talker_cache_->position();
    float use_input_embed = 1.0F;

    std::vector<float> embed_buf(embed_ptr, embed_ptr + embed_size);

    Tensor token_tensor;
    int32_t dummy_token = 0;
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
    if (talker_->has_input("position_id"))
        inputs["position_id"] = position_tensor;
    inputs["attention_mask"] = mask_tensor;
    inputs["input_embed"] = embed_tensor;
    inputs["use_input_embed"] = use_embed_tensor;

    TensorMap outputs = talker_->forward(inputs);

    auto it = outputs.find("logits");
    if (it == outputs.end())
        throw std::runtime_error("OmniPipeline talker: no 'logits' output");

    const auto& lt = it->second;
    auto n = lt.numel();
    logits.resize(static_cast<std::size_t>(n));
    std::memcpy(logits.data(), lt.data, n * sizeof(float));

    talker_cache_->advance();
}

static int32_t omni_argmax(const std::vector<float>& logits)
{
    if (logits.empty()) return 0;
    return static_cast<int32_t>(
        std::distance(logits.begin(),
                      std::max_element(logits.begin(), logits.end())));
}

std::vector<int32_t> OmniPipeline::run_thinker(
    const std::vector<int32_t>& input_ids,
    int32_t max_tokens,
    std::vector<float>& hidden_states_out)
{
    (void)hidden_states_out;

    thinker_cache_->reset();
    thinker_cache_->bind_to(*thinker_);

    std::vector<float> logits;

    for (std::size_t i = 0; i + 1 < input_ids.size(); ++i)
        run_thinker_step(input_ids[i], logits);

    if (!input_ids.empty())
        run_thinker_step(input_ids.back(), logits);

    std::vector<int32_t> output_ids;
    output_ids.reserve(static_cast<std::size_t>(max_tokens));

    for (int32_t step = 0; step < max_tokens; ++step)
    {
        if (logits.empty()) break;
        int32_t token = omni_argmax(logits);
        if (token == 0) break;
        output_ids.push_back(token);
        run_thinker_step(token, logits);
    }

    std::cerr << "[trtf] Omni Thinker: generated " << output_ids.size()
              << " text tokens" << std::endl;
    return output_ids;
}

std::vector<int32_t> OmniPipeline::run_talker(
    const std::vector<float>& hidden_states,
    int32_t num_tokens)
{
    if (!talker_ || !talker_cache_)
    {
        std::cerr << "[trtf] Omni: no Talker engine" << std::endl;
        return {};
    }

    talker_cache_->reset();
    talker_cache_->bind_to(*talker_);

    const int32_t n_codebooks = config_->talker_n_codebooks;
    const int32_t codebook_size = config_->talker_codebook_size;
    const int32_t talker_hidden = config_->talker_hidden_size;

    const OmniTalkerDecodePlan decode_plan = make_omni_talker_decode_plan(
        n_codebooks, codebook_size, num_tokens);

    std::vector<int32_t> all_codes;
    all_codes.reserve(static_cast<std::size_t>(num_tokens) * n_codebooks);

    std::vector<float> logits;
    for (int32_t t = 0; t < num_tokens; ++t)
    {
        const float* ep =
            hidden_states.data() + static_cast<std::size_t>(t) * talker_hidden;
        run_talker_embed_step(ep, talker_hidden, logits);
        append_omni_talker_codes_from_logits(logits, decode_plan, all_codes);
    }

    std::cerr << "[trtf] Omni Talker: generated " << all_codes.size()
              << " codec tokens (" << num_tokens << " frames x "
              << n_codebooks << " codebooks)" << std::endl;
    return all_codes;
}

std::vector<float> OmniPipeline::run_code2wav(
    const std::vector<int32_t>& codec_tokens,
    int32_t n_codebooks,
    int32_t n_frames)
{
    if (!code2wav_)
    {
        std::cerr << "[trtf] Omni: no Code2Wav engine, generating simple waveform"
                  << std::endl;
        const int32_t samples_per_frame = config_->sample_rate / 75;
        const int32_t total_samples = n_frames * samples_per_frame;
        std::vector<float> waveform(static_cast<std::size_t>(total_samples), 0.0F);
        for (int32_t f = 0; f < n_frames; ++f)
        {
            const float freq = 200.0F +
                static_cast<float>(codec_tokens[f * n_codebooks]) * 800.0F /
                static_cast<float>(config_->talker_codebook_size);
            const float amp = 0.3F;
            for (int32_t s = 0; s < samples_per_frame; ++s)
            {
                const auto idx = static_cast<std::size_t>(f) * samples_per_frame + s;
                const float t = static_cast<float>(s) /
                                static_cast<float>(config_->sample_rate);
                waveform[idx] = amp * std::sin(2.0F * 3.14159265F * freq * t);
            }
        }
        return waveform;
    }

    const int32_t max_frames = config_->code2wav_max_frames;
    const int32_t actual_frames = std::min(n_frames, max_frames);
    const int32_t upsample = config_->code2wav_upsample_factor;

    std::vector<int32_t> input_codes = build_omni_code2wav_input_codes(
        codec_tokens, n_codebooks, max_frames, actual_frames);

    Tensor codes_tensor;
    codes_tensor.data = input_codes.data();
    codes_tensor.shape = {static_cast<int64_t>(n_codebooks),
                          static_cast<int64_t>(max_frames)};
    codes_tensor.dtype = DType::kInt32;

    TensorMap inputs;
    inputs["codec_tokens"] = codes_tensor;

    TensorMap outputs = code2wav_->forward(inputs);

    auto it = outputs.find("waveform");
    if (it == outputs.end())
    {
        std::cerr << "[trtf] Omni Code2Wav: no 'waveform' output" << std::endl;
        return {};
    }

    const auto& wt = it->second;
    const auto total_out = wt.numel();
    const auto trimmed = static_cast<std::size_t>(actual_frames)
        * static_cast<std::size_t>(upsample);
    const auto copy_n = std::min(total_out, trimmed);

    std::vector<float> waveform(static_cast<std::size_t>(copy_n));
    std::memcpy(waveform.data(), wt.data, copy_n * sizeof(float));

    std::cerr << "[trtf] Omni Code2Wav: " << actual_frames << " frames -> "
              << waveform.size() << " samples" << std::endl;
    return waveform;
}

AudioResult OmniPipeline::generate_audio(
    const std::string& prompt, const GenerateConfig& cfg)
{
    std::vector<int32_t> input_ids;
    if (tokenizer_)
        input_ids = tokenizer_->encode(prompt);

    int32_t max_tokens = cfg.max_new_tokens > 0 ? cfg.max_new_tokens : 768;

    AudioResult result;
    result.sample_rate = config_->sample_rate;

    std::cerr << "[trtf] Omni: starting pipeline with " << input_ids.size()
              << " input tokens" << std::endl;

    std::vector<float> hidden_states;
    auto text_tokens = run_thinker(input_ids, max_tokens, hidden_states);
    if (text_tokens.empty())
    {
        std::cerr << "[trtf] Omni: Thinker produced no tokens" << std::endl;
        return result;
    }

    const OmniTalkerPlan talker_plan = make_omni_talker_plan(
        text_tokens.size(),
        hidden_states.size(),
        talker_ != nullptr);
    if (talker_plan.should_run_talker)
    {
        auto codec_tokens = run_talker(hidden_states, talker_plan.num_tokens);

        const OmniCodecPlan codec_plan = make_omni_codec_plan(
            *config_, codec_tokens.size());
        if (codec_plan.should_run_codec)
        {
            auto waveform = run_code2wav(
                codec_tokens,
                codec_plan.n_codebooks,
                codec_plan.n_frames);
            if (!waveform.empty())
            {
                result.samples = std::move(waveform);
                result.num_samples = static_cast<int32_t>(result.samples.size());
            }
        }
    }

    std::cerr << "[trtf] Omni: generated " << result.num_samples << " samples ("
              << (result.num_samples > 0
                  ? static_cast<float>(result.num_samples) / result.sample_rate
                  : 0.0F)
              << "s @ " << result.sample_rate << " Hz)" << std::endl;

    return result;
}

} // namespace trtf

#endif // TRTF_HAS_TRT
