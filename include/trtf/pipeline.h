#pragma once

// trtf public C++ API — the only header users need.
//
// Usage:
//   auto pipe = trtf::load("model.trtfb");
//   auto result = pipe->generate("Hello", {.max_new_tokens = 20});
//   std::cout << result.text << std::endl;

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtf {

// --- Result types (all value types, user owns the data) ---

struct TextResult {
    std::string text;
    std::vector<int32_t> token_ids;
    double prefill_ms{0.0}; // populated when GenerateConfig::collect_timing is true
    double decode_ms{0.0};  // populated when GenerateConfig::collect_timing is true
};

struct ImageResult {
    std::vector<float> pixels; // [C, H, W] float32 in [0,1]
    int32_t height{0};
    int32_t width{0};
    int32_t channels{3};
    int32_t num_frames{1}; // >1 for video
};

struct AudioResult {
    std::vector<float> samples; // mono float32 [-1,1]
    int32_t num_samples{0};
    int32_t sample_rate{24000};
};

struct EmbeddingResult {
    std::vector<float> data;
    int32_t dim{0};
};

struct SegmentResult {
    std::vector<int32_t> mask; // class indices [H, W]
    int32_t height{0};
    int32_t width{0};
};

struct TextEmbedding {
    std::vector<float> data;
    std::vector<int64_t> shape;
};

struct GenerateConfig {
    int32_t max_new_tokens{128};
    float temperature{1.0f};
    int32_t top_k{1}; // 1 = greedy
    int32_t seed{-1};
    float guidance_scale{-1.0f}; // diffusion
    int32_t num_steps{-1};       // diffusion
    int32_t eos_token_id{-1};
    int32_t tail_frames{0};        // speech-to-speech: extra frames after input
    bool collect_timing{false};    // if true, populate TextResult::prefill_ms / decode_ms
    bool use_chat_template{false}; ///< Apply chat template before tokenization
    bool enable_thinking{true};    ///< Qwen3: if false, disable thinking mode
};

// --- Pipeline interface ---

class IPipeline {
  public:
    virtual ~IPipeline() = default;

    // -- Text generation (decoder, mamba, rwkv, VL) --
    virtual TextResult generate(const std::string& prompt, const GenerateConfig& cfg = {}) {
        (void)prompt;
        (void)cfg;
        throw std::runtime_error(std::string(pipeline_type()) + " does not support generate()");
    }

    // -- Text generation with image (VL models) --
    virtual TextResult generate(const std::string& prompt, const float* image_pixels,
                                int32_t image_height, int32_t image_width,
                                const GenerateConfig& cfg = {}) {
        (void)image_pixels;
        (void)image_height;
        (void)image_width;
        return generate(prompt, cfg);
    }

    // -- Text encoding (reusable embeddings for diffusion) --
    virtual TextEmbedding encode_text(const std::string& prompt) {
        (void)prompt;
        throw std::runtime_error(std::string(pipeline_type()) + " does not support encode_text()");
    }

    // -- Image generation (diffusion) --
    virtual ImageResult generate_image(const std::string& prompt, const GenerateConfig& cfg = {}) {
        (void)prompt;
        (void)cfg;
        throw std::runtime_error(std::string(pipeline_type()) +
                                 " does not support generate_image()");
    }

    virtual ImageResult generate_image(const TextEmbedding& emb, const GenerateConfig& cfg = {}) {
        (void)emb;
        (void)cfg;
        throw std::runtime_error(std::string(pipeline_type()) +
                                 " does not support generate_image(TextEmbedding)");
    }

    // -- Audio generation (bark, magpie) --
    virtual AudioResult generate_audio(const std::string& prompt, const GenerateConfig& cfg = {}) {
        (void)prompt;
        (void)cfg;
        throw std::runtime_error(std::string(pipeline_type()) +
                                 " does not support generate_audio()");
    }

    // -- Streaming audio generation (magpie) --
    // Callback receives (pcm_samples, num_samples, sample_rate) per chunk.
    using AudioChunkCallback = std::function<void(const float*, int32_t, int32_t)>;
    virtual int32_t generate_audio_streaming(const std::string& prompt, const GenerateConfig& cfg,
                                             AudioChunkCallback callback,
                                             int32_t chunk_frames = 32) {
        (void)prompt;
        (void)cfg;
        (void)callback;
        (void)chunk_frames;
        throw std::runtime_error(std::string(pipeline_type()) + " does not support streaming");
    }

    // -- Transcription (whisper, canary) --
    // input_sample_rate: source audio sample rate. 0 = assume already at model rate.
    // When non-zero and different from the model's expected rate, the pipeline
    // resamples the audio before mel extraction.
    virtual TextResult transcribe(const float* audio_samples, int32_t num_samples,
                                  int32_t max_tokens = 224, int32_t input_sample_rate = 0) {
        (void)audio_samples;
        (void)num_samples;
        (void)max_tokens;
        (void)input_sample_rate;
        throw std::runtime_error(std::string(pipeline_type()) + " does not support transcribe()");
    }

    // -- Speech to speech --
    virtual AudioResult speak(const float* audio_in, int32_t num_samples,
                              const GenerateConfig& cfg = {}, int32_t input_sample_rate = 0) {
        (void)audio_in;
        (void)num_samples;
        (void)cfg;
        (void)input_sample_rate;
        throw std::runtime_error(std::string(pipeline_type()) + " does not support speak()");
    }

    // -- Embedding --
    virtual EmbeddingResult embed(const std::string& text) {
        (void)text;
        throw std::runtime_error(std::string(pipeline_type()) + " does not support embed()");
    }

    // -- Reranking --
    virtual float rerank(const std::string& query, const std::string& document) {
        (void)query;
        (void)document;
        throw std::runtime_error(std::string(pipeline_type()) + " does not support rerank()");
    }

    // -- Segmentation --
    virtual SegmentResult segment(const float* pixels, int32_t height, int32_t width) {
        (void)pixels;
        (void)height;
        (void)width;
        throw std::runtime_error(std::string(pipeline_type()) + " does not support segment()");
    }

    // -- Encoder-only hidden states (BERT) --
    virtual EmbeddingResult encode(const std::string& text) {
        (void)text;
        throw std::runtime_error(std::string(pipeline_type()) + " does not support encode()");
    }

    // -- Neural operator --
    virtual EmbeddingResult solve(const float* branch_input, int32_t branch_len,
                                  const float* trunk_input, int32_t trunk_len) {
        (void)branch_input;
        (void)branch_len;
        (void)trunk_input;
        (void)trunk_len;
        throw std::runtime_error(std::string(pipeline_type()) + " does not support solve()");
    }

    // -- Object detection --
    virtual std::string detect(const float* pixels, int32_t height, int32_t width,
                               float conf_threshold = 0.5f) {
        (void)pixels;
        (void)height;
        (void)width;
        (void)conf_threshold;
        throw std::runtime_error(std::string(pipeline_type()) + " does not support detect()");
    }

    // -- Metadata --
    virtual const char* model_id() const = 0;
    virtual const char* pipeline_type() const = 0;
};

// --- Factory ---
std::unique_ptr<IPipeline> load(const std::string& bundle_path, const std::string& hf_python = "",
                                const std::string& runtime_cache_path = "",
                                bool cuda_graphs = false);

} // namespace trtf

// --- C ABI (backward compatibility) ---

extern "C" {

struct TrtfPipelineOptions {
    int max_new_tokens;        // 0 = use model default
    const char* hf_python;     // nullptr = auto-detect
    const char* image_path;    // nullptr = text-only
    const char* runtime_cache; // nullptr = no RTX cache
    int cuda_graphs;           // 0 = disabled
};

trtf::IPipeline* trtf_create_pipeline(const char* bundle_path, int flags);
trtf::IPipeline* trtf_create_pipeline_ex(const char* bundle_path,
                                         const TrtfPipelineOptions* options);
const char* trtf_last_error(void);
const char* trtf_version(void);
int trtf_has_trt(void);
}
