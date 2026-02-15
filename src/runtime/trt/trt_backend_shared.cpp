#include "runtime/trt/trt_backend_shared.h"
#include "runtime/trt/kv_cache_step_state.h"
#include "runtime/trt/trt_decode_runtime.h"
#include "utils/json_helpers.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
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
            auto tw0 = std::chrono::steady_clock::now();
            std::cerr << "[trtf] Loading model weights (vocab=" << model.vocab.size()
                      << ", layers=" << model.architecture.num_layers
                      << ", hidden=" << model.checkpoint.hidden_size << ") ..." << std::endl;
            mWeights = BuildTrtDecoderWeights(mTokenizer, model);
            auto tw1 = std::chrono::steady_clock::now();
            std::cerr << "[trtf] Weights ready ["
                      << std::chrono::duration_cast<std::chrono::milliseconds>(tw1 - tw0).count()
                      << " ms]" << std::endl;
            std::cerr << "[trtf] Building TRT engine (this may take a while) ..." << std::endl;
            mDecoderStepEngine = factory(mWeights, mLogger);
            auto tw2 = std::chrono::steady_clock::now();
            std::cerr << "[trtf] Engine ready ["
                      << std::chrono::duration_cast<std::chrono::milliseconds>(tw2 - tw1).count()
                      << " ms]" << std::endl;
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

        auto state = std::make_unique<KvCacheStepState>(*mDecoderStepEngine);

        std::vector<float> logits;
        std::vector<std::vector<float>> present_k;
        std::vector<std::vector<float>> present_v;
        const int32_t debug_logits_topk = parse_positive_env_int("TRTF_DEBUG_LOGITS_TOPK", 0);

        // Prefill: process all input tokens except the last
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

        // Decode: generate new tokens autoregressively
        int32_t current_token = input_ids.empty() ? mWeights.id_bos : input_ids.back();

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
    return std::make_unique<TrtBackendShared>(tokenizer, model, std::move(factory));
}

std::unique_ptr<IGenerationBackend> CreateTrtBackendWithBuilder(
    const ITokenizer& tokenizer, const DecoderModel& model, ITrtGraphBuilder& builder)
{
    return CreateTrtBackendWithFactory(tokenizer, model,
        [&builder](const TrtDecoderDefinition& weights, TrtLogger& logger) {
            return builder.build_decoder_step_engine(weights, logger);
        });
}

#endif // TRTF_HAS_TRT

} // namespace trtf
