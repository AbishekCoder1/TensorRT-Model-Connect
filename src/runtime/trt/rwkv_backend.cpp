#include "runtime/trt/rwkv_backend.h"
#include "runtime/trt/rwkv_step_state.h"
#include "runtime/trt/trt_decode_runtime.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

namespace {

// RWKV TRT backend for the bundle load path.
class RwkvBackendFastPath final : public IGenerationBackend {
public:
    explicit RwkvBackendFastPath(std::unique_ptr<RwkvStepEngine> engine)
        : mEngine(std::move(engine))
    {
    }

    bool is_available() const override { return static_cast<bool>(mEngine); }
    const char* name() const override { return "trt_rwkv"; }

    std::vector<int32_t> generate(const std::vector<int32_t>& input_ids, const GenerationConfig& config) override
    {
        if (!mEngine)
        {
            throw std::runtime_error("RWKV TRT backend not initialized");
        }

        std::vector<int32_t> output = input_ids;
        if (config.max_new_tokens == 0) return output;

        auto state = std::make_unique<RwkvStepState>(
            mEngine->num_layers, mEngine->hidden_size);
        std::vector<float> logits;
        std::vector<std::vector<float>> present_attn, present_ff;
        std::vector<std::vector<float>> present_num, present_den, present_max;

        // Prefill: process input tokens one by one (RWKV is recurrent, no batch prefill)
        if (input_ids.size() > 1)
        {
            for (std::size_t i = 0; i + 1 < input_ids.size(); ++i)
            {
                std::string error;
                if (!run_rwkv_step(*mEngine, input_ids[i],
                        state->attn_state_by_layer(), state->ff_state_by_layer(),
                        state->num_state_by_layer(), state->den_state_by_layer(),
                        state->max_state_by_layer(),
                        logits,
                        present_attn, present_ff, present_num, present_den, present_max,
                        error))
                {
                    throw std::runtime_error("rwkv prefill step failed: " + error);
                }
                state->update_after_step(present_attn, present_ff, present_num, present_den, present_max);
            }
        }

        // Decode
        int32_t current_token = input_ids.empty() ? mEngine->id_bos : input_ids.back();
        for (std::size_t step = 0; step < config.max_new_tokens; ++step)
        {
            std::string error;
            if (!run_rwkv_step(*mEngine, current_token,
                    state->attn_state_by_layer(), state->ff_state_by_layer(),
                    state->num_state_by_layer(), state->den_state_by_layer(),
                    state->max_state_by_layer(),
                    logits,
                    present_attn, present_ff, present_num, present_den, present_max,
                    error))
            {
                throw std::runtime_error("rwkv decode step failed: " + error);
            }
            state->update_after_step(present_attn, present_ff, present_num, present_den, present_max);
            const int32_t next_token = select_argmax_token(logits);
            output.push_back(next_token);
            current_token = next_token;
            if (next_token == mEngine->id_eos) break;
        }
        return output;
    }

private:
    std::unique_ptr<RwkvStepEngine> mEngine;
};

} // namespace

std::unique_ptr<IGenerationBackend> CreateRwkvBackendFromEngine(
    std::unique_ptr<RwkvStepEngine> engine)
{
    return std::make_unique<RwkvBackendFastPath>(std::move(engine));
}

#endif // TRTF_HAS_TRT

} // namespace trtf
