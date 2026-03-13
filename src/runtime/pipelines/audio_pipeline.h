#pragma once

// Audio pipelines: WhisperPipeline, BarkPipeline, MagpiePipeline,
// SpeechPipeline, OmniPipeline.
//
// Whisper and Bark own old-style backends and delegate to them.
// MagpiePipeline, SpeechPipeline, and OmniPipeline are migrated to TrtModule + KvCache (new runtime).

#include "trtf/pipeline.h"
#include "trtf/tokenizer.h"
#include "trtf/runtime/trt_module.h"
#include "trtf/runtime/kv_cache.h"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#if TRTF_HAS_TRT

#include "runtime/trt/audio/audio_configs.h"
#include "runtime/trt/audio/speech_delay_cache.h"
#include "runtime/trt/audio/speech_generation_policy.h"
#include "runtime/trt/audio/speech_runtime_plan.h"
#include "runtime/trt/core/trt_common.h"

// Forward-declare old backends and interfaces to avoid pulling in heavy headers.
namespace trtf {
class WhisperBackend;
class BarkBackend;
class ISubprocessRunner;
struct MelFilterbank;
struct OmniConfig;
} // namespace trtf

namespace trtf {

class WhisperPipeline final : public IPipeline {
public:
    /// Construct with a fully-initialized WhisperBackend.
    WhisperPipeline(
        std::unique_ptr<WhisperBackend> backend,
        MelFilterbank mel_fb,
        int32_t mel_n_fft,
        int32_t mel_hop_length,
        int32_t mel_chunk_length,
        int32_t mel_sampling_rate,
        std::shared_ptr<ITokenizer> tokenizer = nullptr,
        std::string model_id_str = "");

    ~WhisperPipeline() override;

    TextResult transcribe(const float* audio_data, int32_t num_samples,
                          int32_t max_new_tokens,
                          int32_t input_sample_rate = 0) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "WhisperPipeline"; }

private:
    std::unique_ptr<WhisperBackend> backend_;
    std::unique_ptr<MelFilterbank> mel_fb_;
    std::shared_ptr<ITokenizer> tokenizer_;
    int32_t mel_n_fft_;
    int32_t mel_hop_length_;
    int32_t mel_chunk_length_;
    int32_t mel_sampling_rate_;
    std::string model_id_;
};

class BarkPipeline final : public IPipeline {
public:
    /// Construct with a fully-initialized BarkBackend.
    BarkPipeline(
        std::unique_ptr<BarkBackend> backend,
        std::shared_ptr<ITokenizer> tokenizer,
        std::string model_id_str = "");

    ~BarkPipeline() override;

    AudioResult generate_audio(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "BarkPipeline"; }

private:
    std::unique_ptr<BarkBackend> backend_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
};

class MagpiePipeline final : public IPipeline {
public:
    /// Construct with TrtModules + KvCaches (new runtime).
    MagpiePipeline(
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
        std::shared_ptr<ITokenizer> tokenizer = nullptr,
        std::string model_id_str = "");

    ~MagpiePipeline() override;

    AudioResult generate_audio(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "MagpiePipeline"; }

private:
    // --- Decoder loop state ---
    struct DecoderLoopState {
        int32_t hidden{0};
        int32_t num_cb{0};
        int32_t cb_size{0};
        int32_t total_logits{0};
        bool use_cfg{false};
        bool use_gpu_kernels{false};
        bool use_gpu_greedy{false};
        bool use_cross_attn_tracking{false};
        int32_t estimated_frames{0};
        int32_t finished_limit{0};
        int32_t max_source_positions{0};
        int32_t text_consumed_threshold{1};
        bool text_consumed{false};
        int32_t frames_past_text_consumed{0};
        int32_t max_peak_pos{0};
        std::vector<float> logits;
        std::vector<float> embed_buf;
        std::vector<float> cb_embed;
        std::string error;
        double prof_prefill_ms{0.0};
        double prof_embed_ms{0.0};
        double prof_trt_step_ms{0.0};
        double prof_sample_ms{0.0};
    };

    // Pipeline stages
    void run_encoder(const std::vector<int32_t>& text_ids);
    void compute_cross_kv();
    void bind_cross_kv();
    void compute_cross_kv_uncond();
    void bind_cross_kv_uncond();

    DecoderLoopState init_decoder_state() const;
    int32_t prefill_context(DecoderLoopState& state);
    std::vector<int32_t> run_decoder(int32_t max_frames);
    std::vector<float> run_codec(const std::vector<int32_t>& codes, int32_t num_frames);

    // Decoder step: feeds embed via input_embed, returns logits
    void run_decoder_step(const float* embed, int32_t embed_size,
                          std::vector<float>& logits_out);
    void run_decoder_step_uncond(const float* embed, int32_t embed_size,
                                 std::vector<float>& logits_out);

    // CPU sampling loop
    std::vector<int32_t> run_cpu_sampling_loop(DecoderLoopState& state, int32_t max_frames);
    void cpu_compute_frame_embed(DecoderLoopState& state,
                                  const std::vector<int32_t>& prev_codes);

    // GPU greedy loop
    std::vector<int32_t> run_gpu_greedy_loop(DecoderLoopState& state, int32_t max_frames);
    bool gpu_greedy_frame_step(DecoderLoopState& state, int32_t frame,
                                CudaBuffer& d_eos_flag);
    void gpu_greedy_update_text_consumed(DecoderLoopState& state, int32_t frame);

    // CFG passes
    bool run_cfg_uncond_pass_gpu(DecoderLoopState& state, int32_t frame);
    bool run_cfg_uncond_pass_cpu(DecoderLoopState& state, int32_t frame);

    // Text completion tracking
    void update_text_completion(DecoderLoopState& state, int32_t frame);
    bool check_finished_limit(DecoderLoopState& state, int32_t frame);

    // Constructor helpers
    void upload_embeddings_to_gpu();
    void init_cross_attn_resources();
    void init_cfg_logit_buffers();

    // Helpers
    void apply_env_overrides();
    void ensure_cfg_resources();
    void run_cfg_encoder(const std::vector<int32_t>& text_ids);
    void log_decoder_profiling(const DecoderLoopState& state,
                                int32_t ctx_frames, int32_t gen_frames) const;
    void log_pipeline_profiling(int32_t num_frames, int32_t num_samples,
                                 double ms_encoder, double ms_decoder,
                                 double ms_codec, double ms_total) const;
    void lookup_embed(const float* table, int32_t token_id, float* out) const;
    void sum_embeds(const float* a, const float* b, float* out) const;
    int32_t sample_top_k(const float* logits, int32_t vocab_size,
                          float temperature, int32_t top_k);

    // Engines (TrtModule-based)
    std::unique_ptr<TrtModule> encoder_;
    std::unique_ptr<TrtModule> decoder_;
    std::unique_ptr<KvCache> decoder_cache_;
    std::unique_ptr<TrtModule> codec_;

    // CFG: unconditional path
    std::unique_ptr<KvCache> decoder_cache_uncond_;

    // Cross-attention buffers
    std::vector<CudaBuffer> cross_k_, cross_v_;
    std::vector<CudaBuffer> cross_k_uncond_, cross_v_uncond_;
    CudaBuffer encoder_output_, encoder_output_uncond_;

    // Cross-attention weights tracking
    CudaBuffer cross_attn_weights_, cross_attn_weights_scratch_;
    bool has_cross_attn_output_{false};

    // Embedding tables
    std::vector<float> audio_embed_, text_embed_, context_embed_;
    std::vector<int32_t> context_lengths_;

    // GPU-side buffers for greedy path
    CudaBuffer audio_embed_device_, context_embed_device_;
    CudaBuffer device_codes_, device_full_argmax_, device_prev_codes_;
    CudaBuffer device_all_codes_;
    CudaBuffer device_logits_cond_, device_logits_uncond_;

    cudaStream_t stream_;
    MagpieTTSConfig config_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    std::mt19937 rng_;
    int32_t text_length_{0};
};

class SpeechPipeline final : public IPipeline {
public:
    /// Construct with TrtModules + KvCaches (new runtime).
    SpeechPipeline(
        std::unique_ptr<TrtModule> mimi_encoder,
        std::unique_ptr<TrtModule> temporal,
        std::unique_ptr<KvCache> temporal_cache,
        std::vector<std::unique_ptr<TrtModule>> depth_engines,
        std::unique_ptr<KvCache> depth_cache,
        std::unique_ptr<TrtModule> mimi_decoder,
        SpeechConfig config,
        cudaStream_t stream,
        std::shared_ptr<ISubprocessRunner> subprocess_runner = nullptr,
        std::string model_id_str = "");

    ~SpeechPipeline() override;

    AudioResult speak(const float* audio_in, int32_t num_samples,
                      const GenerateConfig& cfg = {},
                      int32_t input_sample_rate = 0) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "SpeechPipeline"; }

private:
    // --- Mimi encoder: audio waveform -> codec tokens ---
    std::vector<int32_t> run_mimi_encode(
        const float* samples, int32_t num_samples);

    // --- Temporal step: input_embed -> logits (+ hidden_state) ---
    void run_temporal_embed_step(
        const float* embed_ptr, int32_t embed_size,
        std::vector<float>& logits,
        std::vector<float>& hidden_out);

    // --- Depth step: generate num_codebooks tokens ---
    std::vector<int32_t> run_depth(
        const float* temporal_hidden, int32_t hidden_dim,
        int32_t text_token,
        const int32_t* forced_audio_tokens = nullptr,
        const uint8_t* forced_audio_provided = nullptr);

    // --- Mimi decoder: codec tokens -> waveform ---
    std::vector<float> run_mimi_decode(
        const std::vector<int32_t>& codec_tokens, int32_t num_frames);

    // --- Text prompt injection ---
    void run_text_prompt();

    // --- speak() helpers ---
    bool speak_validate_dual_stream() const;
    void speak_run_generation_loop(
        const SpeechGenerationSettings& settings,
        const SpeechOutputPlan& plan,
        DelayCacheState& delay_state,
        const std::vector<int32_t>& codec_tokens,
        std::vector<int32_t>& output_codes,
        int32_t& frames_collected);
    void speak_postprocess_waveform(
        std::vector<float>& waveform, int32_t generated_frames) const;

    // Engines (TrtModule-based)
    std::unique_ptr<TrtModule> mimi_encoder_;
    std::unique_ptr<TrtModule> temporal_;
    std::unique_ptr<KvCache> temporal_cache_;
    std::vector<std::unique_ptr<TrtModule>> depth_engines_;
    std::unique_ptr<KvCache> depth_cache_;
    std::unique_ptr<TrtModule> mimi_decoder_;

    cudaStream_t stream_;
    SpeechConfig config_;
    std::shared_ptr<ISubprocessRunner> subprocess_runner_;
    std::string model_id_;
    uint64_t rng_state_{0};

    // Mimi encode output shape (set by run_mimi_encode).
    int32_t last_encode_frames_{0};
    int32_t last_encode_codebooks_{0};

    // Debug counter for depth calls within one speak() invocation.
    int32_t depth_debug_call_count_{0};
};

class OmniPipeline final : public IPipeline {
public:
    /// Construct with TrtModules + KvCaches (new runtime).
    OmniPipeline(
        std::unique_ptr<TrtModule> thinker,
        std::unique_ptr<KvCache> thinker_cache,
        std::unique_ptr<TrtModule> talker,
        std::unique_ptr<KvCache> talker_cache,
        std::unique_ptr<TrtModule> code2wav,
        OmniConfig config,
        cudaStream_t stream,
        std::shared_ptr<ITokenizer> tokenizer = nullptr,
        std::string model_id_str = "");

    ~OmniPipeline() override;

    AudioResult generate_audio(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "OmniPipeline"; }

private:
    // Run one thinker decoder step: token_id -> logits. Updates cache.
    void run_thinker_step(int32_t token_id, std::vector<float>& logits);

    // Run one talker decoder step with input embedding. Updates cache.
    void run_talker_embed_step(const float* embed_ptr, int32_t embed_size,
                               std::vector<float>& logits);

    // Stage 0: Thinker generates text tokens and hidden states.
    std::vector<int32_t> run_thinker(
        const std::vector<int32_t>& input_ids,
        int32_t max_tokens,
        std::vector<float>& hidden_states_out);

    // Stage 1: Talker converts hidden states to RVQ codec tokens.
    std::vector<int32_t> run_talker(
        const std::vector<float>& hidden_states,
        int32_t num_tokens);

    // Stage 2: Code2Wav synthesizes waveform from RVQ tokens.
    std::vector<float> run_code2wav(
        const std::vector<int32_t>& codec_tokens,
        int32_t n_codebooks,
        int32_t n_frames);

    std::unique_ptr<TrtModule> thinker_;
    std::unique_ptr<KvCache> thinker_cache_;
    std::unique_ptr<TrtModule> talker_;
    std::unique_ptr<KvCache> talker_cache_;
    std::unique_ptr<TrtModule> code2wav_;
    std::unique_ptr<OmniConfig> config_;
    cudaStream_t stream_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
};

} // namespace trtf

#endif // TRTF_HAS_TRT
