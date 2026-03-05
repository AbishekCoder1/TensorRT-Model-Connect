#include "runtime/trt/multimodal/vision_engine.h"
#include "runtime/trt/core/trt_common.h"
#include "runtime/trt/core/trt_engine_lifecycle.h"

#include <cstring>
#include <memory>
#include <string_view>

namespace trtf {

#if TRTF_HAS_TRT

namespace {

struct PendingCopy {
    void* host_ptr{nullptr};
    void* device_ptr{nullptr};
    std::size_t bytes{0};
};

struct VisionRunState {
    std::vector<std::unique_ptr<CudaBuffer>> device_buffers;
    std::vector<PendingCopy> output_copies;
};

bool fail_with(std::string& error, std::string_view stage)
{
    error = std::string(stage);
    return false;
}

bool validate_engine(const VisionStepEngine& engine, std::string& error)
{
    if (engine.engine && engine.context)
    {
        return true;
    }
    return fail_with(error, "vision engine not initialized");
}

std::size_t feature_count(const VisionStepEngine& engine)
{
    return static_cast<std::size_t>(engine.num_output_features) * engine.feature_dim;
}

bool bind_input_tensor(
    const VisionStepEngine& engine,
    const std::string& name,
    const void* host_ptr,
    std::size_t bytes,
    CudaStream& stream,
    VisionRunState& state,
    bool allow_host_tensor)
{
    if (allow_host_tensor &&
        engine.engine->getTensorLocation(name.c_str()) == nvinfer1::TensorLocation::kHOST)
    {
        return engine.context->setTensorAddress(name.c_str(), const_cast<void*>(host_ptr));
    }

    auto buffer = std::make_unique<CudaBuffer>(bytes);
    if (!buffer->ok())
    {
        return false;
    }
    if (cudaMemcpyAsync(buffer->data(), host_ptr, bytes, cudaMemcpyHostToDevice, stream.get())
        != cudaSuccess)
    {
        return false;
    }
    if (!engine.context->setTensorAddress(name.c_str(), buffer->data()))
    {
        return false;
    }
    state.device_buffers.push_back(std::move(buffer));
    return true;
}

bool bind_output_tensor(
    const VisionStepEngine& engine,
    const std::string& name,
    void* host_ptr,
    std::size_t bytes,
    VisionRunState& state,
    bool allow_host_tensor)
{
    if (allow_host_tensor &&
        engine.engine->getTensorLocation(name.c_str()) == nvinfer1::TensorLocation::kHOST)
    {
        return engine.context->setTensorAddress(name.c_str(), host_ptr);
    }

    auto buffer = std::make_unique<CudaBuffer>(bytes);
    if (!buffer->ok())
    {
        return false;
    }
    if (!engine.context->setTensorAddress(name.c_str(), buffer->data()))
    {
        return false;
    }
    state.output_copies.push_back(PendingCopy{host_ptr, buffer->data(), bytes});
    state.device_buffers.push_back(std::move(buffer));
    return true;
}

bool run_and_copy_outputs(
    const VisionStepEngine& engine,
    CudaStream& stream,
    const std::vector<PendingCopy>& output_copies,
    std::string& error)
{
    if (!engine.context->enqueueV3(stream.get()))
    {
        return fail_with(error, "vision enqueueV3 failed");
    }

    for (const PendingCopy& copy : output_copies)
    {
        if (cudaMemcpyAsync(
                copy.host_ptr,
                copy.device_ptr,
                copy.bytes,
                cudaMemcpyDeviceToHost,
                stream.get())
            != cudaSuccess)
        {
            return fail_with(error, "vision cudaMemcpyAsync output failed");
        }
    }

    if (cudaStreamSynchronize(stream.get()) != cudaSuccess)
    {
        return fail_with(error, "vision cudaStreamSynchronize failed");
    }
    return true;
}

bool bind_main_outputs(
    const VisionStepEngine& engine,
    std::vector<float>& image_features,
    std::size_t output_bytes,
    VisionRunState& state,
    bool allow_host_tensor,
    std::string& error)
{
    if (bind_output_tensor(
            engine,
            engine.features_output_name,
            image_features.data(),
            output_bytes,
            state,
            allow_host_tensor))
    {
        return true;
    }
    return fail_with(error, "bind vision output image_features failed");
}

bool bind_pixel_input(
    const VisionStepEngine& engine,
    const float* pixel_values,
    std::size_t pixel_bytes,
    CudaStream& stream,
    VisionRunState& state,
    bool allow_host_tensor,
    std::string& error)
{
    if (bind_input_tensor(
            engine,
            engine.pixel_input_name,
            pixel_values,
            pixel_bytes,
            stream,
            state,
            allow_host_tensor))
    {
        return true;
    }
    return fail_with(error, "bind vision input pixel_values failed");
}

bool bind_deepstack_outputs(
    const VisionStepEngine& engine,
    std::size_t output_count,
    std::size_t output_bytes,
    VisionRunState& state,
    std::vector<std::vector<float>>& deepstack_features,
    std::string& error)
{
    deepstack_features.clear();
    for (int32_t i = 0;; ++i)
    {
        std::string ds_name = "deepstack_features_" + std::to_string(i);
        if (!has_io_tensor(*engine.engine, ds_name))
        {
            break;
        }

        deepstack_features.emplace_back(output_count, 0.0F);
        if (bind_output_tensor(
                engine,
                ds_name,
                deepstack_features.back().data(),
                output_bytes,
                state,
                false))
        {
            continue;
        }

        return fail_with(error, "bind deepstack output " + ds_name + " failed");
    }
    return true;
}

bool initialize_stream(CudaStream& stream, std::string& error)
{
    if (stream.ok())
    {
        return true;
    }
    return fail_with(error, "cudaStreamCreate failed");
}

} // namespace

bool run_vision_encoder(
    const VisionStepEngine& engine,
    const float* pixel_values,
    std::size_t pixel_bytes,
    std::vector<float>& image_features,
    std::string& error)
{
    if (!validate_engine(engine, error))
    {
        return false;
    }

    const std::size_t output_count = feature_count(engine);
    const std::size_t output_bytes = output_count * sizeof(float);
    image_features.assign(output_count, 0.0F);

    CudaStream stream;
    if (!initialize_stream(stream, error))
    {
        return false;
    }

    VisionRunState state;
    if (!bind_pixel_input(
            engine,
            pixel_values,
            pixel_bytes,
            stream,
            state,
            true,
            error))
    {
        return false;
    }

    if (!bind_main_outputs(engine, image_features, output_bytes, state, true, error))
    {
        return false;
    }

    return run_and_copy_outputs(engine, stream, state.output_copies, error);
}

bool run_vision_encoder_with_deepstack(
    const VisionStepEngine& engine,
    const float* pixel_values,
    std::size_t pixel_bytes,
    std::vector<float>& image_features,
    std::vector<std::vector<float>>& deepstack_features,
    std::string& error)
{
    if (!validate_engine(engine, error))
    {
        return false;
    }

    const std::size_t output_count = feature_count(engine);
    const std::size_t output_bytes = output_count * sizeof(float);
    image_features.assign(output_count, 0.0F);

    CudaStream stream;
    if (!initialize_stream(stream, error))
    {
        return false;
    }

    VisionRunState state;
    if (!bind_pixel_input(
            engine,
            pixel_values,
            pixel_bytes,
            stream,
            state,
            false,
            error))
    {
        return false;
    }

    if (!bind_main_outputs(engine, image_features, output_bytes, state, false, error))
    {
        return false;
    }

    if (!bind_deepstack_outputs(
            engine,
            output_count,
            output_bytes,
            state,
            deepstack_features,
            error))
    {
        return false;
    }

    return run_and_copy_outputs(engine, stream, state.output_copies, error);
}

#endif // TRTF_HAS_TRT

} // namespace trtf
