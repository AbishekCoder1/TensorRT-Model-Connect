#pragma once

#include <cstddef>
#include <cstdint>

namespace trtf {

class IPipeline {
public:
    virtual ~IPipeline() = default;

    // Generate text. Returns pointer valid until next generate() or destruction.
    virtual const char* generate(const char* prompt, std::size_t max_new_tokens = 0) = 0;

    // Metadata -- pointers valid for lifetime of the pipeline.
    virtual const char* model_id() const = 0;
    virtual const char* backend_name() const = 0;
};

} // namespace trtf

extern "C" {

struct TrtfPipelineOptions {
    int max_new_tokens;           // 0 = use model default (20)
    const char* hf_python;        // nullptr = auto-detect
};

// Create a pipeline from a .trtfb bundle file.
// Returns owning pointer. Caller deletes with `delete`.
// Returns nullptr on failure (call trtf_last_error() for message).
trtf::IPipeline* trtf_create_pipeline(const char* bundle_path, int flags);

// Extended creation with options.
// Pass nullptr for options to use defaults.
trtf::IPipeline* trtf_create_pipeline_ex(const char* bundle_path, const TrtfPipelineOptions* options);

// Last error message (thread-local). Valid until next trtf_ call on this thread.
const char* trtf_last_error(void);

// Version string.
const char* trtf_version(void);

// 1 if compiled with TRT support, 0 otherwise.
int trtf_has_trt(void);

}
