#pragma once

// TextGenerationPipeline: serves ALL decoder-only LLMs.
// HF equivalent: TextGenerationPipeline (one class, many models).
//
// Composes: TrtModule (decoder) + KvCache + ITokenizer.
// The model-specific architecture (GQA, RoPE, SwiGLU, etc.) is baked into
// the TRT engine. This pipeline just runs prefill → decode loop.

#include "runtime/core/chat_template.h"
#include "trtf/pipeline.h"
#include "trtf/runtime/inference_state.h"
#include "trtf/runtime/sampler.h"
#include "trtf/runtime/trt_module.h"
#include "trtf/tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtf {

struct TextGenConfig {
    int32_t vocab_size{0};
    int32_t id_bos{0};
    int32_t id_eos{0};
    bool has_position_input{true};
    ChatTemplateFormat chat_template_format{ChatTemplateFormat::kNone};
    std::string token_id_name{"token_id"};
    std::string logits_output_name{"logits"};

    // Dual-profile unified engine: when a prefill TrtModule (same engine,
    // different optimization profile) is attached, the pipeline runs the
    // whole prompt through it at profile 0 (batched-MHA kernels at opt
    // Sq), copies the per-layer K/V into the shared decode KV cache, and
    // switches to profile 1 (GEMV fast-path at Sq=1) for autoregressive
    // decode. The I/O names are shared by both profiles.
    std::string present_k_pattern{"present_k_{i}"};
    std::string present_v_pattern{"present_v_{i}"};
    int32_t prefill_max_length{0};
    int32_t num_layers{0};
    int32_t kv_dim{0};
};

class TextGenerationPipeline final : public IPipeline {
  public:
    TextGenerationPipeline(std::unique_ptr<TrtModule> decoder,
                           std::unique_ptr<IInferenceState> state, TextGenConfig config,
                           cudaStream_t stream, std::shared_ptr<ITokenizer> tokenizer = nullptr,
                           std::string model_id_str = "",
                           std::unique_ptr<ISampler> sampler = nullptr,
                           std::unique_ptr<TrtModule> prefill = nullptr);

    // Public API: takes raw text, returns typed result.
    TextResult generate(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "TextGenerationPipeline"; }

    // Token-ID-based generation (for unit tests and internal callers).
    struct GenerationResult {
        std::vector<int32_t> token_ids;
    };
    GenerationResult generate_ids(const std::vector<int32_t>& input_ids, const GenerateConfig& cfg);

    // Argmax over logits (public for testing).
    static int32_t argmax(const std::vector<float>& logits);

  private:
    std::unique_ptr<TrtModule> decoder_;
    std::unique_ptr<TrtModule> prefill_;
    std::unique_ptr<IInferenceState> state_;
    TextGenConfig config_;
    cudaStream_t stream_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    std::unique_ptr<ISampler> sampler_;
    const float* d_logits_ptr_{nullptr}; // device logits pointer (for GPU sampling)
    std::string logits_output_name_;

    // Internal: generate from token IDs with sampling parameters and optional timing.
    struct TimedGenResult {
        std::vector<int32_t> token_ids;
        double prefill_ms{0.0};
        double decode_ms{0.0};
    };
    TimedGenResult generate_from_ids(const std::vector<int32_t>& input_ids, int32_t max_new_tokens,
                                     const SamplingParams& params, bool collect_timing = false);

    // Run one decoder step: token_id → logits (D2H to host). Updates cache.
    void run_step(int32_t token_id, std::vector<float>& logits);

    // Run one decoder step: logits stay on device (d_logits_ptr_ updated).
    void run_step_device(int32_t token_id);

    // Decode loop (extracted for CCN).
    int32_t run_decode_loop(ISampler* sampler, const SamplingParams& params,
                            std::vector<int32_t>& output, std::vector<float>& logits,
                            int32_t max_new_tokens, bool gpu_sampling);

    // Run the batched prefill engine on the whole prompt: populate the decode
    // KV cache from the prefill outputs and return the last-token logits in
    // ``logits`` (host side). Returns true on success; false if the prefill
    // engine cannot handle this prompt length (caller falls back).
    bool run_prefill_batched(const std::vector<int32_t>& input_ids, std::vector<float>& logits);

    // Dispatch prompt through the batched prefill engine when available +
    // supported; otherwise run the token-by-token fallback on the decode
    // engine. Post-condition: KV cache is populated and ``logits`` holds
    // host-side logits for the last prompt token (when not gpu_sampling).
    void run_prefill_stage(const std::vector<int32_t>& input_ids, std::vector<float>& logits,
                           bool gpu_sampling);
};

} // namespace trtf
