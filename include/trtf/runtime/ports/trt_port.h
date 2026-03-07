#pragma once

#include <cstddef>
#include <string>
#include <utility>

namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
} // namespace nvinfer1

namespace trtf::runtime {

enum class TrtPortStatus {
    kOk = 0,
    kInvalidArgument,
    kDeserializeFailed,
    kContextCreationFailed,
    kMetadataFailed,
};

template <typename T>
struct TrtPortResult {
    TrtPortStatus status{TrtPortStatus::kOk};
    std::string message;
    T value{};

    bool ok() const noexcept
    {
        return status == TrtPortStatus::kOk;
    }

    explicit operator bool() const noexcept
    {
        return ok();
    }

    static TrtPortResult<T> success(T value)
    {
        TrtPortResult<T> result;
        result.value = std::move(value);
        return result;
    }

    static TrtPortResult<T> failure(TrtPortStatus status_code, std::string error_message)
    {
        TrtPortResult<T> result;
        result.status = status_code;
        result.message = std::move(error_message);
        return result;
    }
};

class ITrtPort {
public:
    virtual ~ITrtPort() = default;

    virtual TrtPortResult<nvinfer1::ICudaEngine*> deserialize_engine(
        nvinfer1::IRuntime* runtime,
        const void* serialized_bytes,
        std::size_t serialized_size) const = 0;

    virtual TrtPortResult<nvinfer1::IExecutionContext*> create_execution_context(
        nvinfer1::ICudaEngine* engine) const = 0;

    virtual TrtPortResult<bool> has_io_tensor_named(
        const nvinfer1::ICudaEngine* engine,
        const std::string& tensor_name) const = 0;
};

} // namespace trtf::runtime
