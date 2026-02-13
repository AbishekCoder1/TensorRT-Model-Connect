#include "runtime/trt/trt_engine_lifecycle.h"
#include "utils/trt/engine_cache.h"

namespace trtf {

#if TRTF_HAS_TRT

bool has_io_tensor(const nvinfer1::ICudaEngine& engine, const std::string& tensor_name)
{
    const int32_t count = engine.getNbIOTensors();
    for (int32_t i = 0; i < count; ++i)
    {
        const char* candidate = engine.getIOTensorName(i);
        if (candidate != nullptr && tensor_name == candidate)
        {
            return true;
        }
    }
    return false;
}

bool has_all_required_tensors(const DecoderStepEngine& engine)
{
    if (!has_io_tensor(*engine.engine, engine.token_input_name) || !has_io_tensor(*engine.engine, engine.mask_input_name)
        || !has_io_tensor(*engine.engine, engine.logits_output_name))
    {
        return false;
    }
    if (engine.requires_position_input && !has_io_tensor(*engine.engine, engine.position_input_name))
    {
        return false;
    }

    for (int32_t i = 0; i < engine.num_layers; ++i)
    {
        if (!has_io_tensor(*engine.engine, engine.cache_k_input_names[static_cast<std::size_t>(i)])
            || !has_io_tensor(*engine.engine, engine.cache_v_input_names[static_cast<std::size_t>(i)])
            || !has_io_tensor(*engine.engine, engine.present_k_output_names[static_cast<std::size_t>(i)])
            || !has_io_tensor(*engine.engine, engine.present_v_output_names[static_cast<std::size_t>(i)]))
        {
            return false;
        }
    }
    return true;
}

std::unique_ptr<DecoderStepEngine> finalize_decoder_step_engine(nvinfer1::IBuilder& builder,
    nvinfer1::INetworkDefinition& network, nvinfer1::IBuilderConfig& config, TrtLogger& logger,
    const TrtDecoderDefinition& weights, const std::vector<std::string>& cache_k_input_names,
    const std::vector<std::string>& cache_v_input_names, const std::vector<std::string>& present_k_output_names,
    const std::vector<std::string>& present_v_output_names, bool requires_position_input)
{
    auto runtime = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!runtime)
    {
        return nullptr;
    }

    TrtEngineCacheKeyParams cache_key_params;
    cache_key_params.requires_position_input = requires_position_input;
    cache_key_params.num_layers = static_cast<int32_t>(cache_k_input_names.size());
    const std::string cache_key = BuildTrtEngineCacheKey(weights, cache_key_params);

    TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine;
    if (const auto cached_plan = LoadTrtEnginePlanFromCache(cache_key))
    {
        trt_engine = TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(cached_plan->data(), cached_plan->size()));
    }

    if (!trt_engine)
    {
        auto plan = TrtUniquePtr<nvinfer1::IHostMemory>(builder.buildSerializedNetwork(network, config));
        if (!plan)
        {
            return nullptr;
        }

        trt_engine = TrtUniquePtr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
        if (!trt_engine)
        {
            return nullptr;
        }

        SaveTrtEnginePlanToCache(cache_key, plan->data(), plan->size());
    }

    auto execution_context = TrtUniquePtr<nvinfer1::IExecutionContext>(trt_engine->createExecutionContext());
    if (!execution_context)
    {
        return nullptr;
    }

    auto out = std::make_unique<DecoderStepEngine>();
    out->engine = std::move(trt_engine);
    out->context = std::move(execution_context);
    out->vocab_size = weights.vocab_size;
    out->hidden_size = weights.hidden_size;
    out->cache_state_size = (weights.has_qwen_layers && weights.attention_size > 0)
        ? weights.attention_size
        : weights.hidden_size;
    out->attention_mask_size = requires_position_input ? (weights.max_cache_length + 1) : weights.max_cache_length;
    out->max_cache_length = weights.max_cache_length;
    out->num_layers = static_cast<int32_t>(cache_k_input_names.size());
    out->requires_position_input = requires_position_input;
    out->cache_k_input_names = cache_k_input_names;
    out->cache_v_input_names = cache_v_input_names;
    out->present_k_output_names = present_k_output_names;
    out->present_v_output_names = present_v_output_names;

    if (!has_all_required_tensors(*out))
    {
        return nullptr;
    }
    return out;
}

#endif // TRTF_HAS_TRT

} // namespace trtf
