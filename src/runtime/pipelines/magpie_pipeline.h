#pragma once

// MagpiePipeline: Magpie TTS encoder-decoder pipeline with optional CFG.
// Uses TrtModule(encoder) + TrtModule(decoder) + KvCache + TrtModule(codec).

#include "trtf/pipeline.h"
#include "trtf/tokenizer.h"
#include "trtf/runtime/trt_module.h"
#include "trtf/runtime/inference_state.h"
#include "trtf/runtime/kv_cache.h"
#include "runtime/plugins/shared/plugin_helpers.h"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#if TRTF_HAS_TRT

#include "runtime/domains/audio/audio_configs.h"
#include "runtime/core/trt_common.h"

#include <cuda_runtime_api.h>

namespace trtf {

class MagpiePipeline final : public IPipeline {
public:
    MagpiePipeline(
        std::unique_ptr<TrtModule> encoder,
        std::unique_ptr<TrtModule> decoder,
        std::unique_ptr<IInferenceState> decoder_state,
        std::unique_ptr<TrtModule> codec,
        std::unique_ptr<IInferenceState> decoder_state_uncond,
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

    void run_encoder(const std::vector<int32_t>& text_ids);
    void compute_cross_kv();
    void bind_cross_kv();
    void compute_cross_kv_uncond();
    void bind_cross_kv_uncond();

    DecoderLoopState init_decoder_state() const;
    int32_t prefill_context(DecoderLoopState& state);
    std::vector<int32_t> run_decoder(int32_t max_frames);
    std::vector<float> run_codec(const std::vector<int32_t>& codes, int32_t num_frames);

    void run_decoder_step(const float* embed, int32_t embed_size,
                          std::vector<float>& logits_out);
    void run_decoder_step_uncond(const float* embed, int32_t embed_size,
                                 std::vector<float>& logits_out);

    std::vector<int32_t> run_cpu_sampling_loop(DecoderLoopState& state, int32_t max_frames);
    void cpu_compute_frame_embed(DecoderLoopState& state,
                                  const std::vector<int32_t>& prev_codes);

    std::vector<int32_t> run_gpu_greedy_loop(DecoderLoopState& state, int32_t max_frames);
    bool gpu_greedy_frame_step(DecoderLoopState& state, int32_t frame,
                                CudaBuffer& d_eos_flag);
    void gpu_greedy_update_text_consumed(DecoderLoopState& state, int32_t frame);

    bool run_cfg_uncond_pass_gpu(DecoderLoopState& state, int32_t frame);
    bool run_cfg_uncond_pass_cpu(DecoderLoopState& state, int32_t frame);

    void update_text_completion(DecoderLoopState& state, int32_t frame);
    bool check_finished_limit(DecoderLoopState& state, int32_t frame);

    void upload_embeddings_to_gpu();
    void init_cross_attn_resources();
    void init_cfg_logit_buffers();

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

    std::unique_ptr<TrtModule> encoder_;
    std::unique_ptr<TrtModule> decoder_;
    std::unique_ptr<IInferenceState> decoder_state_;
    std::unique_ptr<TrtModule> codec_;

    std::unique_ptr<IInferenceState> decoder_state_uncond_;

    std::vector<CudaBuffer> cross_k_, cross_v_;
    std::vector<CudaBuffer> cross_k_uncond_, cross_v_uncond_;
    CudaBuffer encoder_output_, encoder_output_uncond_;

    CudaBuffer cross_attn_weights_, cross_attn_weights_scratch_;
    bool has_cross_attn_output_{false};

    std::vector<float> audio_embed_, text_embed_, context_embed_;
    std::vector<int32_t> context_lengths_;

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

} // namespace trtf

#endif // TRTF_HAS_TRT
