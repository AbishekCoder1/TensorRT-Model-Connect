#include "runtime/trt/vl_backend.h"
#include "runtime/trt/kv_cache_step_state.h"
#include "runtime/trt/trt_decode_runtime.h"
#include "runtime/trt/image_preprocessor.h"
#include "runtime/trt/vision_engine.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

VLBackendFastPath::VLBackendFastPath(
    std::unique_ptr<DecoderStepEngine> decoder_engine,
    std::unique_ptr<VisionStepEngine> vision_engine,
    const FastPathModelConfig& config,
    VLPreprocessConfig vl_config)
    : mDecoderEngine(std::move(decoder_engine))
    , mVisionEngine(std::move(vision_engine))
    , mConfig(config)
    , mVLConfig(std::move(vl_config))
    , mHasVision(mVisionEngine != nullptr)
{
}

bool VLBackendFastPath::is_available() const
{
    return static_cast<bool>(mDecoderEngine);
}

const char* VLBackendFastPath::name() const
{
    return "trt_vl";
}

bool VLBackendFastPath::supports_vision() const
{
    return mHasVision;
}

std::vector<int32_t> VLBackendFastPath::generate(
    const std::vector<int32_t>& input_ids,
    const GenerationConfig& config)
{
    // Text-only generation: same as TrtBackendFastPath
    if (!mDecoderEngine)
    {
        throw std::runtime_error("VL TRT backend not initialized");
    }

    std::vector<int32_t> output = input_ids;
    if (config.max_new_tokens == 0) return output;

    auto state = std::make_unique<KvCacheStepState>(*mDecoderEngine);
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
            if (!run_decoder_step(*mDecoderEngine, input_ids[i], position_id,
                    state->cache_k_by_layer(), state->cache_v_by_layer(), mask,
                    logits, present_k, present_v, error))
            {
                throw std::runtime_error("VL prefill step failed: " + error);
            }
            state->update_after_step(present_k, present_v);
        }
    }

    // Decode
    int32_t current_token = input_ids.empty() ? mDecoderEngine->id_bos : input_ids.back();
    for (std::size_t step = 0; step < config.max_new_tokens; ++step)
    {
        int32_t position_id{};
        std::vector<float> mask;
        state->prepare_step(position_id, mask);
        std::string error;
        if (!run_decoder_step(*mDecoderEngine, current_token, position_id,
                state->cache_k_by_layer(), state->cache_v_by_layer(), mask,
                logits, present_k, present_v, error))
        {
            throw std::runtime_error("VL decode step failed: " + error);
        }
        state->update_after_step(present_k, present_v);
        const int32_t next_token = select_argmax_token(logits);
        output.push_back(next_token);
        current_token = next_token;
        if (next_token == mDecoderEngine->id_eos) break;
    }
    return output;
}

std::vector<int32_t> VLBackendFastPath::generate_vl(
    const std::vector<int32_t>& input_ids,
    const float* image_features, int32_t num_features,
    int32_t feature_dim,
    const std::vector<int32_t>& /*image_positions*/,
    const GenerationConfig& config)
{
    if (!mDecoderEngine)
    {
        throw std::runtime_error("VL TRT backend not initialized");
    }

    // If no image features, fall back to text-only
    if (image_features == nullptr || num_features <= 0)
    {
        return generate(input_ids, config);
    }

    // VL prefill with embedding fusion
    std::vector<int32_t> output = input_ids;
    if (config.max_new_tokens == 0) return output;

    auto state = std::make_unique<KvCacheStepState>(*mDecoderEngine);
    std::vector<float> logits;
    std::vector<std::vector<float>> present_k, present_v;

    int32_t feat_idx = 0;

    // Prefill: process all input tokens
    if (input_ids.size() > 1)
    {
        for (std::size_t i = 0; i + 1 < input_ids.size(); ++i)
        {
            int32_t position_id{};
            std::vector<float> mask;
            state->prepare_step(position_id, mask);

            const int32_t token = input_ids[i];
            const float* embed_ptr = nullptr;
            float use_embed = 0.0F;

            if (token == mConfig.image_token_id && feat_idx < num_features)
            {
                embed_ptr = image_features + static_cast<std::size_t>(feat_idx) * feature_dim;
                use_embed = 1.0F;
                ++feat_idx;
            }

            std::string error;
            if (!run_decoder_step(*mDecoderEngine, token, position_id,
                    state->cache_k_by_layer(), state->cache_v_by_layer(), mask,
                    logits, present_k, present_v, error,
                    embed_ptr, feature_dim, use_embed))
            {
                throw std::runtime_error("VL prefill step failed: " + error);
            }
            state->update_after_step(present_k, present_v);
        }
    }

    // Handle the last input token
    int32_t current_token = input_ids.empty() ? mDecoderEngine->id_bos : input_ids.back();
    {
        const float* embed_ptr = nullptr;
        float use_embed = 0.0F;
        if (current_token == mConfig.image_token_id && feat_idx < num_features)
        {
            embed_ptr = image_features + static_cast<std::size_t>(feat_idx) * feature_dim;
            use_embed = 1.0F;
            ++feat_idx;
        }

        int32_t position_id{};
        std::vector<float> mask;
        state->prepare_step(position_id, mask);
        std::string error;
        if (!run_decoder_step(*mDecoderEngine, current_token, position_id,
                state->cache_k_by_layer(), state->cache_v_by_layer(), mask,
                logits, present_k, present_v, error,
                embed_ptr, feature_dim, use_embed))
        {
            throw std::runtime_error("VL last-prefill step failed: " + error);
        }
        state->update_after_step(present_k, present_v);
    }

    // Decode: autoregressive generation (text-only, no more image features)
    for (std::size_t step = 0; step < config.max_new_tokens; ++step)
    {
        const int32_t next_token = select_argmax_token(logits);
        output.push_back(next_token);
        if (next_token == mDecoderEngine->id_eos) break;

        int32_t position_id{};
        std::vector<float> mask;
        state->prepare_step(position_id, mask);
        std::string error;
        if (!run_decoder_step(*mDecoderEngine, next_token, position_id,
                state->cache_k_by_layer(), state->cache_v_by_layer(), mask,
                logits, present_k, present_v, error))
        {
            throw std::runtime_error("VL decode step failed: " + error);
        }
        state->update_after_step(present_k, present_v);
    }
    return output;
}

bool VLBackendFastPath::prepare_image(
    const std::string& image_path,
    std::vector<float>& image_features,
    int32_t& num_features,
    int32_t& feature_dim,
    std::string& error)
{
    if (!mVisionEngine)
    {
        error = "No vision engine available";
        return false;
    }

    // Step 1: Load and preprocess image (C++ stb)
    std::cerr << "[trtf] Loading and preprocessing image ..." << std::endl;
    auto preprocessed = load_and_preprocess_image(image_path, mVLConfig);
    if (!preprocessed.ok)
    {
        error = "Image preprocessing failed";
        return false;
    }

    // Step 2: Run vision encoder (C++ TRT)
    std::cerr << "[trtf] Running vision encoder ..." << std::endl;
    const std::size_t pixel_bytes = preprocessed.pixel_values.size() * sizeof(float);
    if (!run_vision_encoder(*mVisionEngine,
            preprocessed.pixel_values.data(), pixel_bytes,
            image_features, error))
    {
        return false;
    }

    num_features = mVisionEngine->num_output_features;
    feature_dim = mVisionEngine->feature_dim;

    std::cerr << "[trtf] Vision encoder done (features=" << num_features
              << ", dim=" << feature_dim << ")" << std::endl;
    return true;
}

std::unique_ptr<IGenerationBackend> CreateVLBackendFromEngines(
    std::unique_ptr<DecoderStepEngine> decoder_engine,
    std::unique_ptr<VisionStepEngine> vision_engine,
    const FastPathModelConfig& config,
    VLPreprocessConfig vl_config)
{
    return std::make_unique<VLBackendFastPath>(
        std::move(decoder_engine),
        std::move(vision_engine),
        config,
        std::move(vl_config));
}

#endif // TRTF_HAS_TRT

} // namespace trtf
