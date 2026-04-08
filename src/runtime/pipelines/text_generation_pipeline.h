#pragma once

// TextGenerationPipeline: serves ALL decoder-only LLMs.
// HF equivalent: TextGenerationPipeline (one class, many models).
//
// Composes: TrtModule (decoder) + KvCache + ITokenizer.
// The model-specific architecture (GQA, RoPE, SwiGLU, etc.) is baked into
// the TRT engine. This pipeline just runs prefill → decode loop.

#include "trtf/pipeline.h"
#include "trtf/runtime/inference_state.h"
#include "trtf/runtime/sampler.h"
#include "trtf/runtime/trt_module.h"
#include "trtf/tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if TRTF_HAS_TRT

namespace trtf {

struct TextGenConfig {
    int32_t vocab_size{0};
    int32_t id_bos{0};
    int32_t id_eos{0};
    bool has_position_input{true};
    std::string logits_output_name{"logits"};
};

class TextGenerationPipeline final : public IPipeline {
  public:
    TextGenerationPipeline(std::unique_ptr<TrtModule> decoder,
                           std::unique_ptr<IInferenceState> state, TextGenConfig config,
                           cudaStream_t stream, std::shared_ptr<ITokenizer> tokenizer = nullptr,
                           std::string model_id_str = "",
                           std::unique_ptr<ISampler> sampler = nullptr);

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
    std::unique_ptr<IInferenceState> state_;
    TextGenConfig config_;
    cudaStream_t stream_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    std::unique_ptr<ISampler> sampler_;

    // Internal: generate from token IDs with sampling parameters and optional timing.
    struct TimedGenResult {
        std::vector<int32_t> token_ids;
        double prefill_ms{0.0};
        double decode_ms{0.0};
    };
    TimedGenResult generate_from_ids(const std::vector<int32_t>& input_ids, int32_t max_new_tokens,
                                     const SamplingParams& params, bool collect_timing = false);

    // Run one decoder step: token_id → logits. Updates cache.
    void run_step(int32_t token_id, std::vector<float>& logits);
};

} // namespace trtf

#endif // TRTF_HAS_TRT
