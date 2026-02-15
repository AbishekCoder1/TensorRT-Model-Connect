#include "runtime/trt/trt_engine_lifecycle.h"

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

std::string layer_tensor_name(const char* stem, int32_t layer)
{
    return std::string(stem) + "_" + std::to_string(layer);
}

#endif // TRTF_HAS_TRT

} // namespace trtf
