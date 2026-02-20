#pragma once

#include <cstddef>
#include <cstdint>

namespace trtf {

class IPipeline {
public:
    virtual ~IPipeline() = default;

    // Generate text. Returns pointer valid until next generate() or destruction.
    virtual const char* generate(const char* prompt, std::size_t max_new_tokens = 0) = 0;

    // Generate text with an image (VL models). Default: ignores image, calls text-only.
    virtual const char* generate(const char* prompt, const char* image_path,
                                 std::size_t max_new_tokens = 0)
    {
        (void) image_path;
        return generate(prompt, max_new_tokens);
    }

    // Returns true if this pipeline has a vision encoder and supports images.
    virtual bool supports_vision() const { return false; }

    // Returns true if this pipeline supports video generation.
    virtual bool supports_video() const { return false; }

    // Generate video frames. Returns number of frames written to output_dir,
    // or -1 on failure. Only works if supports_video() is true.
    virtual int32_t generate_video(const char* prompt, const char* output_dir,
                                   int32_t num_steps = -1, float guidance_scale = -1.0F)
    {
        (void) prompt; (void) output_dir; (void) num_steps; (void) guidance_scale;
        return -1;
    }

    // Metadata -- pointers valid for lifetime of the pipeline.
    virtual const char* model_id() const = 0;
    virtual const char* backend_name() const = 0;
};

} // namespace trtf

extern "C" {

struct TrtfPipelineOptions {
    int max_new_tokens;           // 0 = use model default (20)
    const char* hf_python;        // nullptr = auto-detect
    const char* image_path;       // nullptr = text-only (no image)
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
