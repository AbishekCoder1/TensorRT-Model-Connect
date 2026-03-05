#include "runtime/trt/multimodal/vl_backend.h"
#include "runtime/trt/core/device_kv_cache.h"
#include "runtime/trt/core/trt_decode_runtime.h"
#include "runtime/trt/multimodal/image_preprocessor.h"
#include "runtime/trt/multimodal/vision_engine.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <stdexcept>
#include <utility>
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

void run_text_prefill_steps(
    const DecoderStepEngine& engine,
    DeviceKvCache& cache,
    DeviceResources& resources,
    const std::vector<int32_t>& input_ids,
    std::vector<float>& logits,
    const char* error_prefix)
{
    if (input_ids.size() <= 1)
    {
        return;
    }
    for (std::size_t i = 0; i + 1 < input_ids.size(); ++i)
    {
        run_decoder_step_or_throw(engine, cache, resources, input_ids[i], logits, error_prefix);
    }
}

int32_t get_current_token(const std::vector<int32_t>& input_ids, int32_t bos_token_id)
{
    return input_ids.empty() ? bos_token_id : input_ids.back();
}

void run_text_decode_steps(
    const DecoderStepEngine& engine,
    DeviceKvCache& cache,
    DeviceResources& resources,
    const GenerationConfig& config,
    std::vector<float>& logits,
    std::vector<int32_t>& output,
    int32_t current_token,
    const char* error_prefix)
{
    for (std::size_t step = 0; step < config.max_new_tokens; ++step)
    {
        run_decoder_step_or_throw(engine, cache, resources, current_token, logits, error_prefix);
        const int32_t next_token = select_argmax_token(logits);
        output.push_back(next_token);
        current_token = next_token;
        if (next_token == engine.id_eos)
        {
            break;
        }
    }
}

struct VlTokenEmbedding {
    const float* input_embed{nullptr};
    float use_input_embed{0.0F};
    std::vector<const float*> deepstack_embeds;
    float deepstack_active{0.0F};
};

VlTokenEmbedding build_vl_token_embedding(
    int32_t token_id,
    int32_t image_token_id,
    const float* image_features,
    int32_t num_features,
    int32_t feature_dim,
    int32_t& feature_index,
    const std::vector<std::vector<float>>& deepstack_features)
{
    VlTokenEmbedding embedding;
    if (token_id != image_token_id || feature_index >= num_features)
    {
        return embedding;
    }

    const int32_t used_feature_index = feature_index;
    embedding.input_embed = image_features + static_cast<std::size_t>(feature_index) * feature_dim;
    embedding.use_input_embed = 1.0F;
    ++feature_index;

    if (deepstack_features.empty())
    {
        return embedding;
    }

    embedding.deepstack_embeds.reserve(deepstack_features.size());
    for (const auto& ds : deepstack_features)
    {
        const int32_t ds_feature_count = static_cast<int32_t>(ds.size() / feature_dim);
        if (used_feature_index < ds_feature_count)
        {
            embedding.deepstack_embeds.push_back(
                ds.data() + static_cast<std::size_t>(used_feature_index) * feature_dim);
        }
        else
        {
            embedding.deepstack_embeds.push_back(nullptr);
        }
    }
    embedding.deepstack_active = 1.0F;
    return embedding;
}

void run_decoder_step_with_embedding_or_throw(
    const DecoderStepEngine& engine,
    DeviceKvCache& cache,
    DeviceResources& resources,
    int32_t token_id,
    std::vector<float>& logits,
    const VlTokenEmbedding& embedding,
    int32_t feature_dim,
    const char* error_prefix)
{
    std::string error;
    if (!run_decoder_step_device(engine, cache, resources,
            token_id, logits, error,
            embedding.input_embed, feature_dim, embedding.use_input_embed,
            embedding.deepstack_embeds, embedding.deepstack_active))
    {
        throw std::runtime_error(std::string(error_prefix) + error);
    }
}

void run_vl_prefill_steps(
    const DecoderStepEngine& engine,
    DeviceKvCache& cache,
    DeviceResources& resources,
    const std::vector<int32_t>& input_ids,
    const float* image_features,
    int32_t num_features,
    int32_t feature_dim,
    int32_t image_token_id,
    int32_t& feature_index,
    const std::vector<std::vector<float>>& deepstack_features,
    std::vector<float>& logits)
{
    if (input_ids.size() <= 1)
    {
        return;
    }

    for (std::size_t i = 0; i + 1 < input_ids.size(); ++i)
    {
        const VlTokenEmbedding embedding = build_vl_token_embedding(
            input_ids[i], image_token_id, image_features, num_features, feature_dim,
            feature_index, deepstack_features);
        run_decoder_step_with_embedding_or_throw(
            engine, cache, resources, input_ids[i], logits,
            embedding, feature_dim, "VL prefill step failed: ");
    }
}

void run_vl_last_prefill_step(
    const DecoderStepEngine& engine,
    DeviceKvCache& cache,
    DeviceResources& resources,
    int32_t token_id,
    const float* image_features,
    int32_t num_features,
    int32_t feature_dim,
    int32_t image_token_id,
    int32_t& feature_index,
    const std::vector<std::vector<float>>& deepstack_features,
    std::vector<float>& logits)
{
    const VlTokenEmbedding embedding = build_vl_token_embedding(
        token_id, image_token_id, image_features, num_features, feature_dim,
        feature_index, deepstack_features);
    run_decoder_step_with_embedding_or_throw(
        engine, cache, resources, token_id, logits,
        embedding, feature_dim, "VL last-prefill step failed: ");
}

void run_vl_decode_steps(
    const DecoderStepEngine& engine,
    DeviceKvCache& cache,
    DeviceResources& resources,
    const GenerationConfig& config,
    std::vector<float>& logits,
    std::vector<int32_t>& output)
{
    for (std::size_t step = 0; step < config.max_new_tokens; ++step)
    {
        const int32_t next_token = select_argmax_token(logits);
        output.push_back(next_token);
        if (next_token == engine.id_eos)
        {
            break;
        }
        run_decoder_step_or_throw(
            engine, cache, resources, next_token, logits, "VL decode step failed: ");
    }
}

} // namespace

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

    DeviceKvCache cache(*mDecoderEngine);
    DeviceResources resources(*mDecoderEngine);
    if (!cache.ok() || !resources.ok())
    {
        throw std::runtime_error("Failed to allocate VL device resources");
    }

    std::vector<float> logits;
    run_text_prefill_steps(
        *mDecoderEngine, cache, resources, input_ids, logits, "VL prefill step failed: ");
    run_text_decode_steps(
        *mDecoderEngine, cache, resources, config, logits, output,
        get_current_token(input_ids, mDecoderEngine->id_bos),
        "VL decode step failed: ");
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

    DeviceKvCache cache(*mDecoderEngine);
    DeviceResources resources(*mDecoderEngine);
    if (!cache.ok() || !resources.ok())
    {
        throw std::runtime_error("Failed to allocate VL device resources");
    }

    std::vector<float> logits;
    int32_t feat_idx = 0;
    run_vl_prefill_steps(
        *mDecoderEngine, cache, resources, input_ids,
        image_features, num_features, feature_dim, mConfig.image_token_id,
        feat_idx, deepstack_features, logits);
    run_vl_last_prefill_step(
        *mDecoderEngine, cache, resources,
        get_current_token(input_ids, mDecoderEngine->id_bos),
        image_features, num_features, feature_dim, mConfig.image_token_id,
        feat_idx, deepstack_features, logits);
    run_vl_decode_steps(*mDecoderEngine, cache, resources, config, logits, output);
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

    // Try deepstack-aware encoder first, falls back to standard
    deepstack_features.clear();
    if (!run_vision_encoder_with_deepstack(*mVisionEngine,
            preprocessed.pixel_values.data(), pixel_bytes,
            image_features, deepstack_features, error))
    {
        return false;
    }

    num_features = mVisionEngine->num_output_features;
    feature_dim = mVisionEngine->feature_dim;

    std::cerr << "[trtf] Vision encoder done (features=" << num_features
              << ", dim=" << feature_dim;
    if (!deepstack_features.empty())
    {
        std::cerr << ", deepstack_levels=" << deepstack_features.size();
    }
    std::cerr << ")" << std::endl;
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
