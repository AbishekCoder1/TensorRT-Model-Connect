#pragma once

// BackendLoader: loads backend DSOs via dlopen and caches them.

#include "trtf/runtime/trt_backend.h"
#include <string>

namespace trtf {

class BackendLoader {
public:
    // Load backend by name ("trt" or "trt_rtx").
    // Caches: second call with same name returns same IBackend*.
    // Throws std::runtime_error if DSO not found or factory missing.
    static IBackend* load(const std::string& backend_name);
};

} // namespace trtf
