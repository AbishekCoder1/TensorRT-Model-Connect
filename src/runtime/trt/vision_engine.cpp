#include "runtime/trt/vision_engine.h"
#include "runtime/trt/trt_common.h"

#include <cstring>
#include <memory>
#include <string_view>

namespace trtf {

#if TRTF_HAS_TRT

bool run_vision_encoder(
    const VisionStepEngine& engine,
    const float* pixel_values, std::size_t pixel_bytes,
    std::vector<float>& image_features,
    std::string& error)
{
    auto fail = [&error](std::string_view stage) {
        error = std::string(stage);
        return false;
    };

    if (!engine.engine || !engine.context)
    {
        return fail("vision engine not initialized");
    }

    const std::size_t output_count =
        static_cast<std::size_t>(engine.num_output_features) * engine.feature_dim;
    image_features.assign(output_count, 0.0F);

    CudaStream stream;
    if (!stream.ok())
    {
        return fail("cudaStreamCreate failed");
    }

    std::vector<std::unique_ptr<CudaBuffer>> device_buffers;

    struct PendingCopy {
        void* host_ptr{nullptr};
        void* device_ptr{nullptr};
        std::size_t bytes{0};
    };
    std::vector<PendingCopy> output_copies;

    auto bind_input = [&](const std::string& name, const void* host_ptr, std::size_t bytes) -> bool {
        const auto location = engine.engine->getTensorLocation(name.c_str());
        if (location == nvinfer1::TensorLocation::kHOST)
        {
            return engine.context->setTensorAddress(name.c_str(), const_cast<void*>(host_ptr));
        }

        auto buffer = std::make_unique<CudaBuffer>(bytes);
        if (!buffer->ok())
        {
            return false;
        }
        if (cudaMemcpyAsync(buffer->data(), host_ptr, bytes, cudaMemcpyHostToDevice, stream.get()) != cudaSuccess)
        {
            return false;
        }

        void* device_ptr = buffer->data();
        if (!engine.context->setTensorAddress(name.c_str(), device_ptr))
        {
            return false;
        }
        device_buffers.push_back(std::move(buffer));
        return true;
    };

    auto bind_output = [&](const std::string& name, void* host_ptr, std::size_t bytes) -> bool {
        const auto location = engine.engine->getTensorLocation(name.c_str());
        if (location == nvinfer1::TensorLocation::kHOST)
        {
            return engine.context->setTensorAddress(name.c_str(), host_ptr);
        }

        auto buffer = std::make_unique<CudaBuffer>(bytes);
        if (!buffer->ok())
        {
            return false;
        }

        void* device_ptr = buffer->data();
        if (!engine.context->setTensorAddress(name.c_str(), device_ptr))
        {
            return false;
        }
        output_copies.push_back(PendingCopy{host_ptr, device_ptr, bytes});
        device_buffers.push_back(std::move(buffer));
        return true;
    };

    // Bind input: pixel_values [T*C, H, W]
    if (!bind_input(engine.pixel_input_name, pixel_values, pixel_bytes))
    {
        return fail("bind vision input pixel_values failed");
    }

    // Bind output: image_features [num_merged, feature_dim]
    const std::size_t output_bytes = output_count * sizeof(float);
    if (!bind_output(engine.features_output_name, image_features.data(), output_bytes))
    {
        return fail("bind vision output image_features failed");
    }

    // Execute
    if (!engine.context->enqueueV3(stream.get()))
    {
        return fail("vision enqueueV3 failed");
    }

    // Copy outputs back to host
    for (const PendingCopy& copy : output_copies)
    {
        if (cudaMemcpyAsync(copy.host_ptr, copy.device_ptr, copy.bytes, cudaMemcpyDeviceToHost, stream.get())
            != cudaSuccess)
        {
            return fail("vision cudaMemcpyAsync output failed");
        }
    }

    if (cudaStreamSynchronize(stream.get()) != cudaSuccess)
    {
        return fail("vision cudaStreamSynchronize failed");
    }

    return true;
}

#endif // TRTF_HAS_TRT

} // namespace trtf
