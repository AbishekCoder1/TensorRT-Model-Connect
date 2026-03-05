#include "runtime/trt/core/trt_backend_shared.h"
#include "runtime/trt/core/device_kv_cache.h"
#include "runtime/trt/core/trt_decode_runtime.h"

#include <cstdint>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

namespace {

void run_decoder_step_or_throw(
    const DecoderStepEngine& engine,
    DeviceKvCache& cache,
    DeviceResources& resources,
    int32_t token_id,
    std::vector<float>& logits,
    const char* error_prefix)
{
    std::string error;
    if (!run_decoder_step_device(engine, cache, resources, token_id, logits, error))
    {
        throw std::runtime_error(std::string(error_prefix) + error);
    }
}

void run_prefill_steps(
    const DecoderStepEngine& engine,
    DeviceKvCache& cache,
    DeviceResources& resources,
    const std::vector<int32_t>& input_ids,
    std::vector<float>& logits)
{
    if (input_ids.size() <= 1)
    {
        return;
    }
    for (std::size_t i = 0; i + 1 < input_ids.size(); ++i)
    {
        run_decoder_step_or_throw(
            engine, cache, resources, input_ids[i], logits, "prefill step failed: ");
    }
}

int32_t get_current_token(const std::vector<int32_t>& input_ids, int32_t bos_token_id)
{
    return input_ids.empty() ? bos_token_id : input_ids.back();
}

void run_decode_steps(
    const DecoderStepEngine& engine,
    DeviceKvCache& cache,
    DeviceResources& resources,
    const GenerationConfig& config,
    std::vector<float>& logits,
    std::vector<int32_t>& output,
    int32_t current_token)
{
    for (std::size_t step = 0; step < config.max_new_tokens; ++step)
    {
        run_decoder_step_or_throw(
            engine, cache, resources, current_token, logits, "decode step failed: ");
        const int32_t next_token = select_argmax_token(logits);
        output.push_back(next_token);
        current_token = next_token;
        if (next_token == engine.id_eos)
        {
            break;
        }
    }
}

// Lightweight TRT backend for the bundle load path (pre-built engine, no weight data).
// Uses device-resident KV cache — only small inputs transferred H2D per step.
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

        DeviceKvCache cache(*mDecoderStepEngine);
        DeviceResources resources(*mDecoderStepEngine);
        if (!cache.ok() || !resources.ok())
        {
            throw std::runtime_error("Failed to allocate device resources");
        }

        std::vector<float> logits;
        run_prefill_steps(*mDecoderStepEngine, cache, resources, input_ids, logits);
        run_decode_steps(
            *mDecoderStepEngine, cache, resources, config, logits, output,
            get_current_token(input_ids, mDecoderStepEngine->id_bos));
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
