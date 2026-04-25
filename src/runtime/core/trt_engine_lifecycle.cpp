#include "runtime/core/trt_engine_lifecycle.h"

#include <string>

namespace trtf {

// expand_layer_name: pure string logic, no TRT dependency.
std::string expand_layer_name(const std::string& pattern, int32_t layer) {
    std::string result = pattern;
    auto replace_all = [&](const std::string& tok, const std::string& val) {
        std::size_t pos = 0;
        while ((pos = result.find(tok, pos)) != std::string::npos) {
            result.replace(pos, tok.size(), val);
            pos += val.size();
        }
    };
    // Replace longer tokens first to avoid partial matches.
    replace_all("{2i+2}", std::to_string(2 * layer + 2));
    replace_all("{2i+1}", std::to_string(2 * layer + 1));
    replace_all("{2i}", std::to_string(2 * layer));
    replace_all("{i}", std::to_string(layer));
    return result;
}

namespace {

bool has_required_base_tensors(const DecoderStepEngine& engine) {
    if (!has_io_tensor(*engine.engine, engine.token_input_name)) {
        return false;
    }
    if (!has_io_tensor(*engine.engine, engine.mask_input_name)) {
        return false;
    }
    if (!has_io_tensor(*engine.engine, engine.logits_output_name)) {
        return false;
    }

    return true;
}

bool has_required_layer_tensors(const DecoderStepEngine& engine, std::size_t layer_idx) {
    if (!has_io_tensor(*engine.engine, engine.cache_k_input_names[layer_idx])) {
        return false;
    }
    if (!has_io_tensor(*engine.engine, engine.cache_v_input_names[layer_idx])) {
        return false;
    }
    if (!has_io_tensor(*engine.engine, engine.present_k_output_names[layer_idx])) {
        return false;
    }
    if (!has_io_tensor(*engine.engine, engine.present_v_output_names[layer_idx])) {
        return false;
    }

    return true;
}

} // namespace

bool has_io_tensor(const nvinfer1::ICudaEngine& engine, const std::string& tensor_name) {
    const int32_t count = engine.getNbIOTensors();
    for (int32_t i = 0; i < count; ++i) {
        const char* candidate = engine.getIOTensorName(i);
        if (candidate != nullptr && tensor_name == candidate) {
            return true;
        }
    }
    return false;
}

bool has_all_required_tensors(const DecoderStepEngine& engine) {
    if (!has_required_base_tensors(engine)) {
        return false;
    }

    if (engine.requires_position_input) {
        if (!has_io_tensor(*engine.engine, engine.position_input_name)) {
            return false;
        }
    }

    for (int32_t i = 0; i < engine.num_layers; ++i) {
        if (!has_required_layer_tensors(engine, static_cast<std::size_t>(i))) {
            return false;
        }
    }
    return true;
}

std::string layer_tensor_name(const char* stem, int32_t layer) {
    return std::string(stem) + "_" + std::to_string(layer);
}

} // namespace trtf
