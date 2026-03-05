#include "runtime/trt/recurrent/mamba_backend.h"
#include "runtime/trt/recurrent/mamba_step_state.h"
#include "runtime/trt/core/trt_decode_runtime.h"

#include <cstdint>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

namespace {

void run_mamba_step_or_throw(
    const MambaStepEngine& engine,
    int32_t token_id,
    const MambaStepState& state,
    std::vector<float>& logits,
    std::vector<std::vector<float>>& present_conv,
    std::vector<std::vector<float>>& present_ssm,
    const char* error_prefix)
{
    std::string error;
    if (!run_mamba_step(engine, token_id,
            state.conv_state_by_layer(), state.ssm_state_by_layer(),
            logits, present_conv, present_ssm, error))
    {
        throw std::runtime_error(std::string(error_prefix) + error);
    }
}

void run_prefill_steps(
    const MambaStepEngine& engine,
    const std::vector<int32_t>& input_ids,
    MambaStepState& state,
    std::vector<float>& logits,
    std::vector<std::vector<float>>& present_conv,
    std::vector<std::vector<float>>& present_ssm)
{
    if (input_ids.size() <= 1)
    {
        return;
    }
    for (std::size_t i = 0; i + 1 < input_ids.size(); ++i)
    {
        run_mamba_step_or_throw(engine, input_ids[i], state,
            logits, present_conv, present_ssm, "mamba prefill step failed: ");
        state.update_after_step(present_conv, present_ssm);
    }
}

int32_t get_current_token(const std::vector<int32_t>& input_ids, int32_t bos_token_id)
{
    return input_ids.empty() ? bos_token_id : input_ids.back();
}

void run_decode_steps(
    const MambaStepEngine& engine,
    const GenerationConfig& config,
    MambaStepState& state,
    std::vector<float>& logits,
    std::vector<std::vector<float>>& present_conv,
    std::vector<std::vector<float>>& present_ssm,
    std::vector<int32_t>& output,
    int32_t current_token)
{
    for (std::size_t step = 0; step < config.max_new_tokens; ++step)
    {
        run_mamba_step_or_throw(engine, current_token, state,
            logits, present_conv, present_ssm, "mamba decode step failed: ");
        state.update_after_step(present_conv, present_ssm);
        const int32_t next_token = select_argmax_token(logits);
        output.push_back(next_token);
        current_token = next_token;
        if (next_token == engine.id_eos)
        {
            break;
        }
    }
}

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
        run_prefill_steps(*mEngine, input_ids, *state, logits, present_conv, present_ssm);
        run_decode_steps(*mEngine, config, *state, logits, present_conv, present_ssm,
            output, get_current_token(input_ids, mEngine->id_bos));
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
