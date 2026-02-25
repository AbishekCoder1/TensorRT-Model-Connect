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

    // Returns true if this pipeline supports semantic segmentation.
    virtual bool supports_segmentation() const { return false; }

    // Segment an image. Writes a PNG with raw class indices to output_path.
    // Returns 0 on success, -1 on failure.
    virtual int32_t segment(const char* image_path, const char* output_path)
    {
        (void) image_path; (void) output_path;
        return -1;
    }

    // Returns true if this pipeline supports encoder-only inference (e.g. BERT).
    virtual bool supports_encoding() const { return false; }

    // Encode text and return hidden states as a float array.
    // Returns pointer valid until next encode() or destruction.
    // out_seq_len and out_hidden_size are populated with dimensions.
    virtual const float* encode(const char* text, int32_t* out_seq_len,
                                int32_t* out_hidden_size)
    {
        (void) text; (void) out_seq_len; (void) out_hidden_size;
        return nullptr;
    }

    // Returns true if this pipeline supports neural operator solve.
    virtual bool supports_solve() const { return false; }

    // Solve a neural operator problem. branch_input [branch_len], trunk_input [trunk_len].
    // Returns pointer to output array valid until next solve() or destruction.
    // out_dim is populated with the output dimension.
    virtual const float* solve(const float* branch_input, int32_t branch_len,
                               const float* trunk_input, int32_t trunk_len,
                               int32_t* out_dim)
    {
        (void) branch_input; (void) branch_len;
        (void) trunk_input; (void) trunk_len; (void) out_dim;
        return nullptr;
    }

    // Solve an FNO neural operator: field_input [in_channels * H * W].
    // Returns pointer to output field [out_channels * H * W].
    // out_channels, out_h, out_w are populated with output shape.
    virtual const float* solve_field(const float* field_input, int32_t input_size,
                                     int32_t* out_channels, int32_t* out_h,
                                     int32_t* out_w)
    {
        (void) field_input; (void) input_size;
        (void) out_channels; (void) out_h; (void) out_w;
        return nullptr;
    }

    // Returns true if this pipeline supports object detection.
    virtual bool supports_detection() const { return false; }

    // Detect objects in an image. Writes JSON results to output_path.
    // Returns number of detections on success, -1 on failure.
    virtual int32_t detect(const char* image_path, const char* output_path,
                           float conf_threshold = -1.0F)
    {
        (void) image_path; (void) output_path; (void) conf_threshold;
        return -1;
    }

    // Returns true if this pipeline supports speech-to-text transcription.
    virtual bool supports_transcription() const { return false; }

    // Transcribe audio. audio_path is a .wav file path.
    // Returns transcribed text (valid until next call or destruction).
    virtual const char* transcribe(const char* audio_path, int32_t max_new_tokens = 224)
    {
        (void) audio_path; (void) max_new_tokens;
        return nullptr;
    }

    // Returns true if this pipeline supports audio generation.
    virtual bool supports_audio() const { return false; }

    // Generate audio from text. Writes a WAV file to output_path.
    // Returns number of samples on success, -1 on failure.
    virtual int32_t generate_audio(const char* prompt, const char* output_path,
                                   int32_t max_tokens = -1)
    {
        (void) prompt; (void) output_path; (void) max_tokens;
        return -1;
    }

    // Returns true if this pipeline supports text/image embedding.
    virtual bool supports_embedding() const { return false; }

    // Embed text into a float vector. Returns pointer to embedding array
    // valid until next embed() or destruction. out_dim is populated with dimension.
    virtual const float* embed(const char* text, int32_t* out_dim)
    {
        (void) text; (void) out_dim;
        return nullptr;
    }

    // Returns true if this pipeline supports reranking.
    virtual bool supports_reranking() const { return false; }

    // Rerank: cross-encode query + document, return relevance score.
    virtual float rerank(const char* query, const char* document)
    {
        (void) query; (void) document;
        return 0.0F;
    }

    // Returns true if this pipeline supports prompted segmentation (SAM).
    virtual bool supports_prompted_segmentation() const { return false; }

    // Segment an image with a point prompt. Writes per-mask PNGs to output_dir.
    // point_x, point_y: normalized coordinates [0, 1].
    // is_foreground: true for foreground point, false for background.
    // Returns number of masks on success, -1 on failure.
    virtual int32_t segment_sam(const char* image_path, const char* output_dir,
                                float point_x, float point_y, bool is_foreground)
    {
        (void) image_path; (void) output_dir;
        (void) point_x; (void) point_y; (void) is_foreground;
        return -1;
    }

    // Returns true if this pipeline supports speech-to-speech.
    virtual bool supports_speech() const { return false; }

    // Speech-to-speech: read audio from audio_in WAV, process, write to
    // audio_out WAV. Returns number of output samples on success, -1 on failure.
    // tail_frames extends generation budget beyond input-derived frames.
    virtual int32_t speak(const char* audio_in, const char* audio_out,
                          int32_t max_output_frames = -1,
                          int32_t tail_frames = 0)
    {
        (void) audio_in; (void) audio_out; (void) max_output_frames;
        (void) tail_frames;
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
