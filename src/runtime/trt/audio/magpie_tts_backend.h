#pragma once

#include "runtime/trt/audio/audio_configs.h"
#include "runtime/trt/core/trt_common.h"
#include "runtime/trt/core/trt_engine_lifecycle.h"
#include "runtime/trt/core/device_kv_cache.h"
#include "runtime/trt/audio/bark_backend.h"  // for LegacyAudioResult, write_wav
#include "cabi/config/fast_path_config.h"

#if TRTF_HAS_TRT

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace trtf {

class MagpieTTSBackend {
public:
    MagpieTTSBackend(
        std::unique_ptr<DecoderStepEngine> decoder_engine,
        TrtUniquePtr<nvinfer1::ICudaEngine> encoder_engine,
        TrtUniquePtr<nvinfer1::IExecutionContext> encoder_context,
        std::vector<float> audio_embed,     // [8 * codebook_size * hidden]
        std::vector<float> text_embed,      // [text_vocab * hidden]
        std::vector<float> context_embed,   // [num_speakers * context_frames * hidden]
        std::vector<int32_t> context_lengths, // [num_speakers]
        MagpieTTSConfig config);

    ~MagpieTTSBackend();

    bool is_available() const;

    LegacyAudioResult generate_audio(
        const std::vector<int32_t>& text_ids,
        int32_t max_frames = 500);

    const MagpieTTSConfig& config() const { return mConfig; }

    // Set codec engine for waveform synthesis
    void set_codec_engine(
        TrtUniquePtr<nvinfer1::ICudaEngine> engine,
        TrtUniquePtr<nvinfer1::IExecutionContext> context);


private:
    // Pipeline stages (Whisper pattern)
    void run_encoder(const std::vector<int32_t>& text_ids,
                     int32_t speaker_id, int32_t language_id);
    void compute_cross_kv();
    void bind_cross_kv();
    void compute_cross_kv_uncond();
    void bind_cross_kv_uncond();

    // Decoder loop (multi-codebook autoregressive)
    std::vector<int32_t> run_decoder(int32_t max_frames);

    // Codec (Bark pattern)
    std::vector<float> run_codec(const std::vector<int32_t>& codes, int32_t num_frames);

    // Helpers (Bark pattern)
    void lookup_embed(const float* table, int32_t token_id,
                      float* out) const;
    void sum_embeds(const float* a, const float* b, float* out) const;
    int32_t sample_top_k(const float* logits, int32_t vocab_size,
                         float temperature, int32_t top_k);

    // --- run_decoder() sub-phases ---

    // Shared mutable state for the decoder loop.
    struct DecoderLoopState {
        // Constants derived from config
        int32_t hidden{0};
        int32_t num_cb{0};
        int32_t cb_size{0};
        int32_t total_logits{0};
        bool use_cfg{false};
        bool use_gpu_kernels{false};
        bool use_gpu_greedy{false};
        // Text-completion tracking
        bool use_cross_attn_tracking{false};
        int32_t estimated_frames{0};
        int32_t finished_limit{0};
        int32_t max_source_positions{0};
        int32_t text_consumed_threshold{1};
        bool text_consumed{false};
        int32_t frames_past_text_consumed{0};
        int32_t max_peak_pos{0};
        // Working buffers
        std::vector<float> logits;
        std::vector<float> embed_buf;
        std::vector<float> cb_embed;
        std::string error;
        // Profiling accumulators (ms)
        double prof_prefill_ms{0.0};
        double prof_embed_ms{0.0};
        double prof_trt_step_ms{0.0};
        double prof_sample_ms{0.0};
    };

    // Initialize DecoderLoopState from current config/member state.
    DecoderLoopState init_decoder_state() const;

    // Phase 1: Feed baked speaker context embeddings into the decoder.
    // Returns number of context frames prefilled, or -1 on error.
    int32_t prefill_context(DecoderLoopState& state);

    // Phase 2a (GPU greedy): Deferred-sync decode loop. Returns generated codes.
    std::vector<int32_t> run_gpu_greedy_loop(DecoderLoopState& state, int32_t max_frames);

    // Phase 2b (CPU / non-greedy): Per-step sync decode loop. Returns generated codes.
    std::vector<int32_t> run_cpu_sampling_loop(DecoderLoopState& state, int32_t max_frames);

    // CFG unconditional pass for GPU greedy path (device-side blend).
    // Returns false on failure.
    bool run_cfg_uncond_pass_gpu(DecoderLoopState& state, int32_t frame);

    // CFG unconditional pass for CPU path (host-side blend into state.logits).
    // Returns false on failure.
    bool run_cfg_uncond_pass_cpu(DecoderLoopState& state, int32_t frame);

    // Check cross-attention text completion. Updates state.text_consumed.
    void update_text_completion(DecoderLoopState& state, int32_t frame);

    // Check finished_limit_with_eot. Returns true if generation should stop.
    bool check_finished_limit(DecoderLoopState& state, int32_t frame);

    // Log decoder profiling breakdown to stderr.
    void log_decoder_profiling(const DecoderLoopState& state, int32_t ctx_frames,
                               int32_t gen_frames) const;

    // Constructor helpers
    void allocate_cross_kv_buffers();
    void allocate_cfg_buffers_ctor();
    void detect_cross_attn_output();
    void upload_embed_tables_to_device();

    // prefill_context() helpers
    bool prefill_context_gpu(DecoderLoopState& state, DeviceKvCache& cache,
                             DeviceResources& resources, int32_t ctx_frames,
                             const char* label);
    bool prefill_context_cpu(DecoderLoopState& state, DeviceKvCache& cache,
                             DeviceResources& resources, int32_t ctx_frames,
                             const char* label);
    bool prefill_context_cfg(DecoderLoopState& state, int32_t ctx_frames);

    // GPU greedy loop helpers
    bool gpu_greedy_frame_step(DecoderLoopState& state, int32_t frame,
                               CudaBuffer& d_eos_flag);
    bool gpu_check_stop_conditions(DecoderLoopState& state, int32_t frame,
                                   CudaBuffer& d_eos_flag, int32_t& h_eos_flag,
                                   int32_t& gen_frames_actual);
    void gpu_update_text_completion(DecoderLoopState& state, int32_t frame);

    // CPU sampling loop helpers
    void cpu_compute_frame_embed(DecoderLoopState& state,
                                 const std::vector<int32_t>& prev_codes);
    bool cpu_run_conditioned_step(DecoderLoopState& state, int32_t frame);
    bool cpu_sample_frame_codes(DecoderLoopState& state,
                                std::vector<int32_t>& frame_codes, bool& eos);

    // generate_audio() helpers
    void apply_env_overrides();
    void ensure_cfg_resources();
    void run_cfg_encoder(const std::vector<int32_t>& text_ids);
    void log_pipeline_profiling(int32_t num_frames, int32_t num_samples,
                                double ms_encoder, double ms_decoder,
                                double ms_codec, double ms_total) const;

    // Engines
    std::unique_ptr<DecoderStepEngine> mDecoderEngine;
    TrtUniquePtr<nvinfer1::ICudaEngine> mEncoderEngine;
    TrtUniquePtr<nvinfer1::IExecutionContext> mEncoderContext;
    TrtUniquePtr<nvinfer1::ICudaEngine> mCodecEngine;
    TrtUniquePtr<nvinfer1::IExecutionContext> mCodecCtx;

    // Decoder KV cache (Whisper pattern)
    std::unique_ptr<DeviceKvCache> mCache;
    std::unique_ptr<DeviceResources> mResources;

    // CFG: unconditional path KV cache + resources (allocated when cfg_scale > 1)
    std::unique_ptr<DeviceKvCache> mCacheUncond;
    std::unique_ptr<DeviceResources> mResourcesUncond;

    // Encoder output + per-layer cross-attention buffers
    CudaBuffer mEncoderOutput;
    CudaBuffer mEncoderOutputUncond;  // CFG: encoder output from empty/PAD text
    std::vector<CudaBuffer> mCrossK;  // [decoder_layers] — per-layer copy of encoder output
    std::vector<CudaBuffer> mCrossV;  // [decoder_layers] — per-layer copy of encoder output
    // CFG: unconditional cross-KV from null-text encoder output
    std::vector<CudaBuffer> mCrossKUncond;
    std::vector<CudaBuffer> mCrossVUncond;

    // Embedding tables (Bark pattern)
    std::vector<float> mAudioEmbed;      // [8 * codebook_size * hidden]
    std::vector<float> mTextEmbed;       // [text_vocab * hidden]
    std::vector<float> mContextEmbed;    // [num_speakers * max_context_frames * hidden]
    std::vector<int32_t> mContextLengths; // [num_speakers]

    // GPU-side buffers for greedy argmax (Task #3)
    CudaBuffer mDeviceCodes;        // [num_codebooks] int32_t
    CudaBuffer mDeviceFullArgmax;   // [num_codebooks] int32_t

    // GPU-side embedding tables and buffers (Task #4)
    CudaBuffer mAudioEmbedDevice;   // [8 * codebook_size * hidden] float
    CudaBuffer mContextEmbedDevice; // [num_speakers * max_context_frames * hidden] float
    CudaBuffer mDevicePrevCodes;    // [num_codebooks] int32_t

    // Device-side code accumulator for deferred-sync generation
    CudaBuffer mDeviceAllCodes;     // [max_frames * num_codebooks] int32_t

    // CFG: device-side logit buffers for conditioned and unconditional passes
    CudaBuffer mDeviceLogitsCond;   // [num_codebooks * codebook_size] float
    CudaBuffer mDeviceLogitsUncond; // [num_codebooks * codebook_size] float

    // Cross-attention text-completion tracking (from last decoder layer)
    CudaBuffer mDeviceCrossAttnWeights;       // [max_source_positions] float — conditioned pass
    CudaBuffer mDeviceCrossAttnWeightsScratch; // scratch for uncond pass (CFG only)
    bool mHasCrossAttnOutput{false};

    // Text length for finished_limit_with_eot estimation
    int32_t mTextLength{0};

    MagpieTTSConfig mConfig;
    std::mt19937 mRng{std::random_device{}()};
};

std::unique_ptr<MagpieTTSBackend> CreateMagpieTTSBackend(
    std::unique_ptr<DecoderStepEngine> decoder_engine,
    TrtUniquePtr<nvinfer1::ICudaEngine> encoder_engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> encoder_context,
    std::vector<float> audio_embed,
    std::vector<float> text_embed,
    std::vector<float> context_embed,
    std::vector<int32_t> context_lengths,
    const FastPathModelConfig& cfg);

} // namespace trtf

#endif // TRTF_HAS_TRT
