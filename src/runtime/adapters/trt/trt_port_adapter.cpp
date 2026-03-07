#include "runtime/adapters/trt/trt_port_adapter.h"

#if TRTF_HAS_TRT
#include <NvInferRuntime.h>
#endif

namespace trtf::runtime::adapters::trt {

namespace {

nvinfer1::ICudaEngine* default_deserialize_engine(
    nvinfer1::IRuntime* runtime,
    const void* serialized_bytes,
    std::size_t serialized_size)
{
#if TRTF_HAS_TRT
    return runtime->deserializeCudaEngine(serialized_bytes, serialized_size);
#else
    (void) runtime;
    (void) serialized_bytes;
    (void) serialized_size;
    return nullptr;
#endif
}

nvinfer1::IExecutionContext* default_create_execution_context(nvinfer1::ICudaEngine* engine)
{
#if TRTF_HAS_TRT
    return engine->createExecutionContext();
#else
    (void) engine;
    return nullptr;
#endif
}

int32_t default_get_nb_io_tensors(const nvinfer1::ICudaEngine* engine)
{
#if TRTF_HAS_TRT
    return engine->getNbIOTensors();
#else
    (void) engine;
    return -1;
#endif
}

const char* default_get_io_tensor_name(const nvinfer1::ICudaEngine* engine, int32_t index)
{
#if TRTF_HAS_TRT
    return engine->getIOTensorName(index);
#else
    (void) engine;
    (void) index;
    return nullptr;
#endif
}

} // namespace

TrtPortAdapter::Api TrtPortAdapter::Api::defaults()
{
    Api api;
    api.deserialize_engine = default_deserialize_engine;
    api.create_execution_context = default_create_execution_context;
    api.get_nb_io_tensors = default_get_nb_io_tensors;
    api.get_io_tensor_name = default_get_io_tensor_name;
    return api;
}

TrtPortAdapter::TrtPortAdapter(Api api)
    : mApi(api)
{
    const Api defaults_api = Api::defaults();
    if (mApi.deserialize_engine == nullptr)
    {
        mApi.deserialize_engine = defaults_api.deserialize_engine;
    }
    if (mApi.create_execution_context == nullptr)
    {
        mApi.create_execution_context = defaults_api.create_execution_context;
    }
    if (mApi.get_nb_io_tensors == nullptr)
    {
        mApi.get_nb_io_tensors = defaults_api.get_nb_io_tensors;
    }
    if (mApi.get_io_tensor_name == nullptr)
    {
        mApi.get_io_tensor_name = defaults_api.get_io_tensor_name;
    }
}

TrtPortResult<nvinfer1::ICudaEngine*> TrtPortAdapter::deserialize_engine(
    nvinfer1::IRuntime* runtime,
    const void* serialized_bytes,
    std::size_t serialized_size) const
{
    if (runtime == nullptr)
    {
        return TrtPortResult<nvinfer1::ICudaEngine*>::failure(
            TrtPortStatus::kInvalidArgument, "runtime pointer must not be null");
    }
    if (serialized_bytes == nullptr)
    {
        return TrtPortResult<nvinfer1::ICudaEngine*>::failure(
            TrtPortStatus::kInvalidArgument, "serialized bytes pointer must not be null");
    }
    if (serialized_size == 0)
    {
        return TrtPortResult<nvinfer1::ICudaEngine*>::failure(
            TrtPortStatus::kInvalidArgument, "serialized bytes must not be empty");
    }

    nvinfer1::ICudaEngine* const engine =
        mApi.deserialize_engine(runtime, serialized_bytes, serialized_size);
    if (engine == nullptr)
    {
        return TrtPortResult<nvinfer1::ICudaEngine*>::failure(
            TrtPortStatus::kDeserializeFailed,
            "deserializeCudaEngine returned null");
    }
    return TrtPortResult<nvinfer1::ICudaEngine*>::success(engine);
}

TrtPortResult<nvinfer1::IExecutionContext*> TrtPortAdapter::create_execution_context(
    nvinfer1::ICudaEngine* engine) const
{
    if (engine == nullptr)
    {
        return TrtPortResult<nvinfer1::IExecutionContext*>::failure(
            TrtPortStatus::kInvalidArgument, "engine pointer must not be null");
    }

    nvinfer1::IExecutionContext* const context = mApi.create_execution_context(engine);
    if (context == nullptr)
    {
        return TrtPortResult<nvinfer1::IExecutionContext*>::failure(
            TrtPortStatus::kContextCreationFailed,
            "createExecutionContext returned null");
    }
    return TrtPortResult<nvinfer1::IExecutionContext*>::success(context);
}

TrtPortResult<bool> TrtPortAdapter::has_io_tensor_named(
    const nvinfer1::ICudaEngine* engine,
    const std::string& tensor_name) const
{
    if (engine == nullptr)
    {
        return TrtPortResult<bool>::failure(
            TrtPortStatus::kInvalidArgument, "engine pointer must not be null");
    }
    if (tensor_name.empty())
    {
        return TrtPortResult<bool>::failure(
            TrtPortStatus::kInvalidArgument, "tensor_name must not be empty");
    }

    const int32_t count = mApi.get_nb_io_tensors(engine);
    if (count < 0)
    {
        return TrtPortResult<bool>::failure(
            TrtPortStatus::kMetadataFailed, "getNbIOTensors returned an invalid count");
    }

    for (int32_t i = 0; i < count; ++i)
    {
        const char* const candidate = mApi.get_io_tensor_name(engine, i);
        if (candidate != nullptr && tensor_name == candidate)
        {
            return TrtPortResult<bool>::success(true);
        }
    }
    return TrtPortResult<bool>::success(false);
}

} // namespace trtf::runtime::adapters::trt
