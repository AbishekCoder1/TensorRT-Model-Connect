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

    // Bundle: serialize current engine + tokenizer to a single .trtfb file.
    virtual bool save_bundle(const char* output_path) = 0;
};

} // namespace trtf

extern "C" {

#define TRTF_PREFER_TRT  0
#define TRTF_FORCE_TRT   1
#define TRTF_CPU_ONLY    2

struct TrtfPipelineOptions {
    int flags;                    // TRTF_PREFER_TRT / TRTF_FORCE_TRT / TRTF_CPU_ONLY
    int max_new_tokens;           // 0 = use model default (20)
    int max_cache_length;         // 0 = use config.json value (capped at 4096)
};

// Create a pipeline from a model directory, model alias, or .trtfb bundle.
// Returns owning pointer. Caller deletes with `delete`.
// Returns nullptr on failure (call trtf_last_error() for message).
trtf::IPipeline* trtf_create_pipeline(const char* model_or_bundle, int flags);

// Extended creation with options for max_new_tokens and max_cache_length.
// Pass nullptr for options to use defaults.
trtf::IPipeline* trtf_create_pipeline_ex(const char* model_or_bundle, const TrtfPipelineOptions* options);

// Last error message (thread-local). Valid until next trtf_ call on this thread.
const char* trtf_last_error(void);

// Version string.
const char* trtf_version(void);

// 1 if compiled with TRT support, 0 otherwise.
int trtf_has_trt(void);

}
