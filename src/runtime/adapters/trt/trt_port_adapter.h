#pragma once

#include "trtf/runtime/ports/trt_port.h"

#include <cstdint>

namespace trtf::runtime::adapters::trt {

class TrtPortAdapter final : public ITrtPort {
public:
    struct Api {
        nvinfer1::ICudaEngine* (*deserialize_engine)(
            nvinfer1::IRuntime* runtime, const void* serialized_bytes, std::size_t serialized_size){nullptr};
        nvinfer1::IExecutionContext* (*create_execution_context)(nvinfer1::ICudaEngine* engine){nullptr};
        int32_t (*get_nb_io_tensors)(const nvinfer1::ICudaEngine* engine){nullptr};
        const char* (*get_io_tensor_name)(const nvinfer1::ICudaEngine* engine, int32_t index){nullptr};

        static Api defaults();
    };

    explicit TrtPortAdapter(Api api = Api::defaults());

    TrtPortResult<nvinfer1::ICudaEngine*> deserialize_engine(
        nvinfer1::IRuntime* runtime,
        const void* serialized_bytes,
        std::size_t serialized_size) const override;

    TrtPortResult<nvinfer1::IExecutionContext*> create_execution_context(
        nvinfer1::ICudaEngine* engine) const override;

    TrtPortResult<bool> has_io_tensor_named(
        const nvinfer1::ICudaEngine* engine,
        const std::string& tensor_name) const override;

private:
    Api mApi;
};

} // namespace trtf::runtime::adapters::trt
