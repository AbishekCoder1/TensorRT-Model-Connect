#include "runtime/trt/recurrent/rwkv_backend.h"
#include "runtime/trt/recurrent/rwkv_step_state.h"
#include "runtime/trt/core/trt_decode_runtime.h"

#include <cstdint>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

namespace {

void run_rwkv_step_or_throw(
    const RwkvStepEngine& engine,
    int32_t token_id,
    const RwkvStepState& state,
    std::vector<float>& logits,
    std::vector<std::vector<float>>& present_attn,
    std::vector<std::vector<float>>& present_ff,
    std::vector<std::vector<float>>& present_num,
    std::vector<std::vector<float>>& present_den,
    std::vector<std::vector<float>>& present_max,
    const char* error_prefix)
{
    std::string error;
    if (!run_rwkv_step(engine, token_id,
            state.attn_state_by_layer(), state.ff_state_by_layer(),
            state.num_state_by_layer(), state.den_state_by_layer(),
            state.max_state_by_layer(),
            logits,
            present_attn, present_ff, present_num, present_den, present_max,
            error))
    {
        throw std::runtime_error(std::string(error_prefix) + error);
    }
}

void run_prefill_steps(
    const RwkvStepEngine& engine,
    const std::vector<int32_t>& input_ids,
    RwkvStepState& state,
    std::vector<float>& logits,
    std::vector<std::vector<float>>& present_attn,
    std::vector<std::vector<float>>& present_ff,
    std::vector<std::vector<float>>& present_num,
    std::vector<std::vector<float>>& present_den,
    std::vector<std::vector<float>>& present_max)
{
    if (input_ids.size() <= 1)
    {
        return;
    }
    for (std::size_t i = 0; i + 1 < input_ids.size(); ++i)
    {
        run_rwkv_step_or_throw(engine, input_ids[i], state,
            logits, present_attn, present_ff, present_num, present_den, present_max,
            "rwkv prefill step failed: ");
        state.update_after_step(present_attn, present_ff, present_num, present_den, present_max);
    }
}

int32_t get_current_token(const std::vector<int32_t>& input_ids, int32_t bos_token_id)
{
    return input_ids.empty() ? bos_token_id : input_ids.back();
}

void run_decode_steps(
    const RwkvStepEngine& engine,
    const GenerationConfig& config,
    RwkvStepState& state,
    std::vector<float>& logits,
    std::vector<std::vector<float>>& present_attn,
    std::vector<std::vector<float>>& present_ff,
    std::vector<std::vector<float>>& present_num,
    std::vector<std::vector<float>>& present_den,
    std::vector<std::vector<float>>& present_max,
    std::vector<int32_t>& output,
    int32_t current_token)
{
    for (std::size_t step = 0; step < config.max_new_tokens; ++step)
    {
        run_rwkv_step_or_throw(engine, current_token, state,
            logits, present_attn, present_ff, present_num, present_den, present_max,
            "rwkv decode step failed: ");
        state.update_after_step(present_attn, present_ff, present_num, present_den, present_max);
        const int32_t next_token = select_argmax_token(logits);
        output.push_back(next_token);
        current_token = next_token;
        if (next_token == engine.id_eos)
        {
            break;
        }
    }
}

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
        run_prefill_steps(*mEngine, input_ids, *state, logits,
            present_attn, present_ff, present_num, present_den, present_max);
        run_decode_steps(*mEngine, config, *state, logits,
            present_attn, present_ff, present_num, present_den, present_max,
            output, get_current_token(input_ids, mEngine->id_bos));
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
