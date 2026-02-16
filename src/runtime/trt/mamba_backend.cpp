#include "runtime/trt/mamba_backend.h"
#include "runtime/trt/mamba_step_state.h"
#include "runtime/trt/trt_decode_runtime.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

namespace {

// Mamba/SSM TRT backend for the bundle load path.
class MambaBackendFastPath final : public IGenerationBackend {
public:
    explicit MambaBackendFastPath(std::unique_ptr<MambaStepEngine> engine)
        : mEngine(std::move(engine))
    {
    }

    bool is_available() const override { return static_cast<bool>(mEngine); }
    const char* name() const override { return "trt_mamba"; }

    std::vector<int32_t> generate(const std::vector<int32_t>& input_ids, const GenerationConfig& config) override
    {
        if (!mEngine)
        {
            throw std::runtime_error("Mamba TRT backend not initialized");
        }

        std::vector<int32_t> output = input_ids;
        if (config.max_new_tokens == 0) return output;

        auto state = std::make_unique<MambaStepState>(
            mEngine->num_layers, mEngine->d_inner, mEngine->state_size, mEngine->conv_kernel);
        std::vector<float> logits;
        std::vector<std::vector<float>> present_conv, present_ssm;

        // Prefill: process input tokens one by one (Mamba is recurrent, no batch prefill)
        if (input_ids.size() > 1)
        {
            for (std::size_t i = 0; i + 1 < input_ids.size(); ++i)
            {
                std::string error;
                if (!run_mamba_step(*mEngine, input_ids[i],
                        state->conv_state_by_layer(), state->ssm_state_by_layer(),
                        logits, present_conv, present_ssm, error))
                {
                    throw std::runtime_error("mamba prefill step failed: " + error);
                }
                state->update_after_step(present_conv, present_ssm);
            }
        }

        // Decode
        int32_t current_token = input_ids.empty() ? mEngine->id_bos : input_ids.back();
        for (std::size_t step = 0; step < config.max_new_tokens; ++step)
        {
            std::string error;
            if (!run_mamba_step(*mEngine, current_token,
                    state->conv_state_by_layer(), state->ssm_state_by_layer(),
                    logits, present_conv, present_ssm, error))
            {
                throw std::runtime_error("mamba decode step failed: " + error);
            }
            state->update_after_step(present_conv, present_ssm);
            const int32_t next_token = select_argmax_token(logits);
            output.push_back(next_token);
            current_token = next_token;
            if (next_token == mEngine->id_eos) break;
        }
        return output;
    }

private:
    std::unique_ptr<MambaStepEngine> mEngine;
};

} // namespace

std::unique_ptr<IGenerationBackend> CreateMambaBackendFromEngine(
    std::unique_ptr<MambaStepEngine> engine)
{
    return std::make_unique<MambaBackendFastPath>(std::move(engine));
}

#endif // TRTF_HAS_TRT

} // namespace trtf
