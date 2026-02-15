#include "runtime/trt/trt_engine_lifecycle.h"
#include "runtime/trt/trt_graph_ops.h"
#include "utils/trt/engine_cache.h"

#include <chrono>
#include <iostream>
#include <string>

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

std::vector<const DecoderStepEngine::TensorBinding*> find_extra_bindings(
    const DecoderStepEngine& engine, const std::string& prefix, bool is_input)
{
    std::vector<const DecoderStepEngine::TensorBinding*> result;
    for (const DecoderStepEngine::TensorBinding& binding : engine.extra_bindings)
    {
        if (binding.is_input == is_input
            && binding.logical_name.size() >= prefix.size()
            && binding.logical_name.compare(0, prefix.size(), prefix) == 0)
        {
            result.push_back(&binding);
        }
    }
    return result;
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

    for (const DecoderStepEngine::TensorBinding& binding : engine.extra_bindings)
    {
        if (!has_io_tensor(*engine.engine, binding.engine_name))
        {
            return false;
        }
    }
    return true;
}

std::unique_ptr<DecoderStepEngine> try_load_cached_engine(TrtLogger& logger,
    const TrtDecoderDefinition& weights, int32_t num_layers, bool requires_position_input)
{
    TrtEngineCacheKeyParams cache_key_params;
    cache_key_params.requires_position_input = requires_position_input;
    cache_key_params.num_layers = num_layers;
    const std::string cache_key = BuildTrtEngineCacheKey(weights, cache_key_params);
    const auto cached_plan = LoadTrtEnginePlanFromCache(cache_key);
    if (!cached_plan)
    {
        return nullptr;
    }

    std::cerr << "[trtf] Loading cached TRT engine plan (" << cached_plan->size() / (1024 * 1024) << " MB) ..." << std::endl;
    auto tdeser0 = std::chrono::steady_clock::now();

    auto runtime = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!runtime)
    {
        return nullptr;
    }

    auto trt_engine = TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(cached_plan->data(), cached_plan->size()));
    if (!trt_engine)
    {
        return nullptr;
    }

    auto tdeser1 = std::chrono::steady_clock::now();
    std::cerr << "[trtf] Engine deserialized ["
              << std::chrono::duration_cast<std::chrono::milliseconds>(tdeser1 - tdeser0).count()
              << " ms]" << std::endl;

    auto execution_context = TrtUniquePtr<nvinfer1::IExecutionContext>(trt_engine->createExecutionContext());
    if (!execution_context)
    {
        return nullptr;
    }

    const int32_t attention_size = (weights.has_decoder_layers && weights.attention_size > 0)
        ? weights.attention_size
        : weights.hidden_size;

    auto out = std::make_unique<DecoderStepEngine>();
    out->engine = std::move(trt_engine);
    out->context = std::move(execution_context);
    out->vocab_size = weights.vocab_size;
    out->hidden_size = weights.hidden_size;
    out->cache_state_size = attention_size;
    out->attention_mask_size = requires_position_input ? (weights.max_cache_length + 1) : weights.max_cache_length;
    out->max_cache_length = weights.max_cache_length;
    out->num_layers = num_layers;
    out->requires_position_input = requires_position_input;
    out->id_bos = weights.id_bos;
    out->id_eos = weights.id_eos;

    for (int32_t i = 0; i < num_layers; ++i)
    {
        out->cache_k_input_names.push_back(layer_tensor_name("cache_k", i));
        out->cache_v_input_names.push_back(layer_tensor_name("cache_v", i));
        out->present_k_output_names.push_back(layer_tensor_name("present_k", i));
        out->present_v_output_names.push_back(layer_tensor_name("present_v", i));
    }

    if (!has_all_required_tensors(*out))
    {
        return nullptr;
    }
    return out;
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
        std::cerr << "[trtf] Loading cached TRT engine plan (" << cached_plan->size() / (1024 * 1024) << " MB) ..." << std::endl;
        auto tdeser0 = std::chrono::steady_clock::now();
        trt_engine = TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(cached_plan->data(), cached_plan->size()));
        auto tdeser1 = std::chrono::steady_clock::now();
        std::cerr << "[trtf] Engine deserialized ["
                  << std::chrono::duration_cast<std::chrono::milliseconds>(tdeser1 - tdeser0).count()
                  << " ms]" << std::endl;
    }

    if (!trt_engine)
    {
        std::cerr << "[trtf] No cached engine found. Compiling TRT engine from scratch ..." << std::endl;
        std::cerr << "[trtf] (Use --engine-cache-dir to cache for faster subsequent runs)" << std::endl;
        auto tcomp0 = std::chrono::steady_clock::now();
        auto plan = TrtUniquePtr<nvinfer1::IHostMemory>(builder.buildSerializedNetwork(network, config));
        if (!plan)
        {
            return nullptr;
        }
        auto tcomp1 = std::chrono::steady_clock::now();
        std::cerr << "[trtf] Engine compiled (" << plan->size() / (1024 * 1024) << " MB plan) ["
                  << std::chrono::duration_cast<std::chrono::milliseconds>(tcomp1 - tcomp0).count()
                  << " ms]" << std::endl;

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
    out->cache_state_size = (weights.has_decoder_layers && weights.attention_size > 0)
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
    out->id_bos = weights.id_bos;
    out->id_eos = weights.id_eos;

    if (!has_all_required_tensors(*out))
    {
        return nullptr;
    }
    return out;
}

std::unique_ptr<DecoderStepEngine> finalize_decoder_step_engine(nvinfer1::IBuilder& builder,
    nvinfer1::INetworkDefinition& network, nvinfer1::IBuilderConfig& config, TrtLogger& logger,
    const TrtDecoderDefinition& weights, const std::vector<std::string>& cache_k_input_names,
    const std::vector<std::string>& cache_v_input_names, const std::vector<std::string>& present_k_output_names,
    const std::vector<std::string>& present_v_output_names, bool requires_position_input,
    const std::vector<DecoderStepEngine::TensorBinding>& extra_bindings)
{
    auto result = finalize_decoder_step_engine(builder, network, config, logger, weights,
        cache_k_input_names, cache_v_input_names, present_k_output_names, present_v_output_names,
        requires_position_input);
    if (result)
    {
        result->extra_bindings = extra_bindings;
        if (!has_all_required_tensors(*result))
        {
            return nullptr;
        }
    }
    return result;
}

std::vector<char> SerializeEnginePlan(const DecoderStepEngine& engine)
{
    if (!engine.engine)
    {
        return {};
    }

    auto plan = TrtUniquePtr<nvinfer1::IHostMemory>(engine.engine->serialize());
    if (!plan || plan->size() == 0)
    {
        return {};
    }

    const auto* data = static_cast<const char*>(plan->data());
    return std::vector<char>(data, data + plan->size());
}

#endif // TRTF_HAS_TRT

} // namespace trtf
