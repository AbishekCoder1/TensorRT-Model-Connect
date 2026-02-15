#include "runtime/trt/trt_backend_shared.h"
#include "runtime/trt/kv_cache_step_state.h"
#include "runtime/trt/trt_decode_runtime.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

namespace {

// Lightweight TRT backend for the bundle load path (pre-built engine, no weight data).
class TrtBackendFastPath final : public IGenerationBackend {
public:
    explicit TrtBackendFastPath(std::unique_ptr<DecoderStepEngine> engine)
        : mDecoderStepEngine(std::move(engine))
    {
    }

    bool is_available() const override { return static_cast<bool>(mDecoderStepEngine); }
    const char* name() const override { return "trt"; }

    std::vector<int32_t> generate(const std::vector<int32_t>& input_ids, const GenerationConfig& config) override
    {
        if (!mDecoderStepEngine)
        {
            throw std::runtime_error("TRT backend not initialized");
        }

        std::vector<int32_t> output = input_ids;
        if (config.max_new_tokens == 0) return output;

        auto state = std::make_unique<KvCacheStepState>(*mDecoderStepEngine);
        std::vector<float> logits;
        std::vector<std::vector<float>> present_k, present_v;

        // Prefill
        if (input_ids.size() > 1)
        {
            for (std::size_t i = 0; i + 1 < input_ids.size(); ++i)
            {
                int32_t position_id{};
                std::vector<float> mask;
                state->prepare_step(position_id, mask);
                std::string error;
                if (!run_decoder_step(*mDecoderStepEngine, input_ids[i], position_id,
                        state->cache_k_by_layer(), state->cache_v_by_layer(), mask,
                        logits, present_k, present_v, error))
                {
                    throw std::runtime_error("prefill step failed: " + error);
                }
                state->update_after_step(present_k, present_v);
            }
        }

        // Decode
        int32_t current_token = input_ids.empty() ? mDecoderStepEngine->id_bos : input_ids.back();
        for (std::size_t step = 0; step < config.max_new_tokens; ++step)
        {
            int32_t position_id{};
            std::vector<float> mask;
            state->prepare_step(position_id, mask);
            std::string error;
            if (!run_decoder_step(*mDecoderStepEngine, current_token, position_id,
                    state->cache_k_by_layer(), state->cache_v_by_layer(), mask,
                    logits, present_k, present_v, error))
            {
                throw std::runtime_error("decode step failed: " + error);
            }
            state->update_after_step(present_k, present_v);
            const int32_t next_token = select_argmax_token(logits);
            output.push_back(next_token);
            current_token = next_token;
            if (next_token == mDecoderStepEngine->id_eos) break;
        }
        return output;
    }

private:
    std::unique_ptr<DecoderStepEngine> mDecoderStepEngine;
};

} // namespace

std::unique_ptr<IGenerationBackend> CreateTrtBackendFromEngine(
    std::unique_ptr<DecoderStepEngine> engine)
{
    return std::make_unique<TrtBackendFastPath>(std::move(engine));
}

#endif // TRTF_HAS_TRT

} // namespace trtf
