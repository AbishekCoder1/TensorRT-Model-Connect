#include "runtime/trt/trt_backend_shared.h"
#include "runtime/trt/trt_decode_runtime.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

namespace {

class TrtBackendShared final : public IGenerationBackend {
public:
    TrtBackendShared(const ITokenizer& tokenizer, const DecoderModel& model, DecoderStepEngineFactory factory)
        : mTokenizer(tokenizer)
    {
        const cudaError_t init_err = cudaFree(nullptr);
        if (init_err != cudaSuccess)
        {
            mInitError = std::string("cudaFree(nullptr) failed with cudaError=")
                + std::to_string(static_cast<int>(init_err));
            mAvailable = false;
            return;
        }

        try
        {
            mLogger.clear_error();
            mWeights = BuildTrtDecoderWeights(mTokenizer, model);
            mDecoderStepEngine = factory(mWeights, mLogger);
            mAvailable = static_cast<bool>(mDecoderStepEngine);
            if (!mAvailable)
            {
                mInitError = mLogger.last_error();
                if (mInitError.empty())
                {
                    mInitError = "create_decoder_step_engine returned null";
                }
            }
        }
        catch (const std::exception& e)
        {
            mInitError = e.what();
            mAvailable = false;
        }
    }

    bool is_available() const override
    {
        return mAvailable;
    }

    const char* name() const override
    {
        return "trt";
    }

    const char* unavailable_reason() const override
    {
        return mInitError.c_str();
    }

    std::vector<int32_t> generate(const std::vector<int32_t>& input_ids, const GenerationConfig& config) override
    {
        if (!mAvailable || !mDecoderStepEngine)
        {
            throw std::runtime_error("TRT backend is unavailable: " + mInitError);
        }

        std::vector<int32_t> output = input_ids;
        if (config.max_new_tokens == 0)
        {
            return output;
        }

        const int32_t cache_state_size = mDecoderStepEngine->cache_state_size;
        const int32_t max_cache_length = mDecoderStepEngine->max_cache_length;
        const int32_t num_layers = mDecoderStepEngine->num_layers;
        const bool include_current_slot = mDecoderStepEngine->requires_position_input;
        const int32_t position_limit = include_current_slot ? max_cache_length : std::max(max_cache_length - 1, 0);
        const std::size_t cache_elems
            = static_cast<std::size_t>(max_cache_length) * static_cast<std::size_t>(cache_state_size);

        std::vector<std::vector<float>> cache_k(
            static_cast<std::size_t>(num_layers), std::vector<float>(cache_elems, 0.0F));
        std::vector<std::vector<float>> cache_v(
            static_cast<std::size_t>(num_layers), std::vector<float>(cache_elems, 0.0F));
        int32_t cache_length = 0;

        std::vector<float> logits;
        std::vector<std::vector<float>> present_k;
        std::vector<std::vector<float>> present_v;
        const int32_t debug_logits_topk = parse_positive_env_int("TRTF_DEBUG_LOGITS_TOPK", 0);

        if (input_ids.size() > 1)
        {
            for (std::size_t i = 0; i + 1 < input_ids.size(); ++i)
            {
                const std::vector<float> mask = build_attention_mask(cache_length, max_cache_length, include_current_slot);
                const int32_t position_id = std::min(cache_length, position_limit);
                std::string error;
                if (!run_decoder_step(*mDecoderStepEngine, input_ids[i], position_id, cache_k, cache_v, mask, logits,
                        present_k, present_v, error))
                {
                    throw std::runtime_error("prefill step failed: " + error);
                }
                for (int32_t layer = 0; layer < num_layers; ++layer)
                {
                    const std::size_t idx = static_cast<std::size_t>(layer);
                    append_cache_state(cache_k[idx], present_k[idx], cache_state_size, max_cache_length, cache_length);
                    append_cache_state(cache_v[idx], present_v[idx], cache_state_size, max_cache_length, cache_length);
                }
                cache_length = std::min(cache_length + 1, max_cache_length);
            }
        }

        int32_t current_token = input_ids.empty() ? mWeights.id_bos : input_ids.back();

        for (std::size_t step = 0; step < config.max_new_tokens; ++step)
        {
            const std::vector<float> mask = build_attention_mask(cache_length, max_cache_length, include_current_slot);
            const int32_t position_id = std::min(cache_length, position_limit);
            std::string error;
            if (!run_decoder_step(*mDecoderStepEngine, current_token, position_id, cache_k, cache_v, mask, logits,
                    present_k, present_v, error))
            {
                throw std::runtime_error("decode step failed: " + error);
            }
            for (int32_t layer = 0; layer < num_layers; ++layer)
            {
                const std::size_t idx = static_cast<std::size_t>(layer);
                append_cache_state(cache_k[idx], present_k[idx], cache_state_size, max_cache_length, cache_length);
                append_cache_state(cache_v[idx], present_v[idx], cache_state_size, max_cache_length, cache_length);
            }
            cache_length = std::min(cache_length + 1, max_cache_length);

            if (debug_logits_topk > 0)
            {
                const std::vector<int32_t> top_ids = select_topk_tokens(logits, debug_logits_topk);
                std::cerr << "TRTF_DEBUG_LOGITS step=" << step;
                for (int32_t token_id : top_ids)
                {
                    const std::size_t idx = static_cast<std::size_t>(token_id);
                    std::cerr << ' ' << token_id << ':' << logits[idx];
                }
                std::cerr << '\n';
            }

            const int32_t next_token = select_argmax_token(logits);
            output.push_back(next_token);
            current_token = next_token;

            if (next_token == mWeights.id_eos)
            {
                break;
            }
        }

        return output;
    }

private:
    const ITokenizer& mTokenizer;
    bool mAvailable{false};
    std::string mInitError;
    TrtLogger mLogger;
    TrtDecoderDefinition mWeights;
    std::unique_ptr<DecoderStepEngine> mDecoderStepEngine;
};

} // namespace

std::unique_ptr<IGenerationBackend> CreateTrtBackendWithFactory(
    const ITokenizer& tokenizer, const DecoderModel& model, DecoderStepEngineFactory factory)
{
    return std::make_unique<TrtBackendShared>(tokenizer, model, factory);
}

#endif // TRTF_HAS_TRT

} // namespace trtf
