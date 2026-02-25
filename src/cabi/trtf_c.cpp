#include "trtf/pipeline.h"
#include "trtf/bundle.h"
#include "trtf/tokenizer.h"
#include "bundle/bundle_format.h"
#include "cabi/fast_path_config.h"
#include "cabi/bundle_helpers.h"
#include "runtime/trt/trt_engine_lifecycle.h"
#include "runtime/trt/trt_backend_shared.h"
#include "runtime/trt/mamba_backend.h"
#include "runtime/trt/mamba_decode_runtime.h"
#include "runtime/trt/rwkv_backend.h"
#include "runtime/trt/rwkv_decode_runtime.h"
#include "runtime/trt/whisper_backend.h"
#include "runtime/trt/vl_backend.h"
#include "runtime/trt/vision_engine.h"
#include "runtime/trt/image_preprocessor.h"
#include "runtime/trt/diffusion_backend.h"
#include "runtime/trt/wan_diffusion_backend.h"
#include "runtime/trt/z_image_diffusion_backend.h"
#include "runtime/trt/encoder_backend.h"
#include "runtime/trt/embedding_backend.h"
#include "runtime/trt/reranking_backend.h"
#include "runtime/trt/segmentation_backend.h"
#include "runtime/trt/detection_backend.h"
#include "runtime/trt/sam_backend.h"
#include "runtime/trt/neural_operator_backend.h"
#include "runtime/trt/hybrid_backend.h"
#include "runtime/trt/bark_backend.h"
#include "runtime/trt/omni_backend.h"
#include "runtime/trt/speech_backend.h"
#include "runtime/trt/trt_common.h"

#include "stb_image_write.h"

#include "utils/data_dir.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include <unistd.h>

#ifndef TRTF_VERSION_STRING
#define TRTF_VERSION_STRING "0.1.0"
#endif

namespace {

thread_local std::string g_last_error;

void set_last_error(const std::string& msg)
{
    g_last_error = msg;
}

void clear_last_error()
{
    g_last_error.clear();
}

std::string shell_quote(const std::string& value)
{
    std::string out;
    out.reserve(value.size() + 2);
    out.push_back('\'');
    for (char ch : value)
    {
        if (ch == '\'')
        {
            out += "'\"'\"'";
        }
        else
        {
            out.push_back(ch);
        }
    }
    out.push_back('\'');
    return out;
}

struct SubprocessResult {
    int exit_code{1};
    std::string output;
};

SubprocessResult run_subprocess(const std::string& command)
{
    std::string cmd = command + " 2>&1";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (pipe == nullptr)
    {
        return SubprocessResult{1, "popen failed"};
    }

    std::array<char, 512> buffer{};
    std::string output;
    while (true)
    {
        const std::size_t n = std::fread(buffer.data(), 1, buffer.size(), pipe);
        if (n > 0)
        {
            output.append(buffer.data(), n);
        }
        if (n < buffer.size())
        {
            if (std::feof(pipe) != 0 || std::ferror(pipe) != 0)
            {
                break;
            }
        }
    }

    const int status = pclose(pipe);
    int exit_code = 1;
    if (WIFEXITED(status))
    {
        exit_code = WEXITSTATUS(status);
    }
    return SubprocessResult{exit_code, output};
}

class PipelineImpl final : public trtf::IPipeline {
public:
    PipelineImpl(std::string model_id, std::unique_ptr<trtf::ITokenizer> tokenizer,
        std::unique_ptr<trtf::IGenerationBackend> backend, std::string backend_name,
        trtf::GenerationConfig gen_config)
        : mModelId(std::move(model_id))
        , mTokenizer(std::move(tokenizer))
        , mBackend(std::move(backend))
        , mBackendName(std::move(backend_name))
        , mGenConfig(gen_config)
    {
    }

    ~PipelineImpl() override
    {
        if (!mBundleTempDir.empty())
        {
            std::error_code ec;
            std::filesystem::remove_all(mBundleTempDir, ec);
        }
    }

    const char* generate(const char* prompt, std::size_t max_new_tokens) override
    {
        if (prompt == nullptr)
        {
            mLastOutput = "";
            return mLastOutput.c_str();
        }

        trtf::GenerationConfig config = mGenConfig;
        if (max_new_tokens > 0)
        {
            config.max_new_tokens = max_new_tokens;
        }

#if TRTF_HAS_TRT
        // Omni backend: use Thinker decoder for text generation
        if (mOmniBackend != nullptr && mTokenizer)
        {
            auto input_ids = mTokenizer->encode(prompt);
            const int32_t max_tok = config.max_new_tokens > 0
                ? static_cast<int32_t>(config.max_new_tokens) : 128;
            auto output_ids = mOmniBackend->generate_text(input_ids, max_tok);
            mLastOutput = mTokenizer->decode(output_ids);
            return mLastOutput.c_str();
        }
#endif

        if (!mBackend || !mTokenizer)
        {
            mLastOutput = "";
            return mLastOutput.c_str();
        }

        auto input_ids = mTokenizer->encode(prompt);
        auto output_ids = mBackend->generate(input_ids, config);
        mLastOutput = mTokenizer->decode(output_ids);
        return mLastOutput.c_str();
    }

    const char* generate(const char* prompt, const char* image_path,
                         std::size_t max_new_tokens) override
    {
        if (!mBackend->supports_vision() || image_path == nullptr)
        {
            return generate(prompt, max_new_tokens);
        }

        if (prompt == nullptr)
        {
            mLastOutput = "";
            return mLastOutput.c_str();
        }

        trtf::GenerationConfig config = mGenConfig;
        if (max_new_tokens > 0)
        {
            config.max_new_tokens = max_new_tokens;
        }

#if TRTF_HAS_TRT
        // Native VL pipeline: preprocess image in C++, run vision TRT, tokenize, generate
        auto* vl = dynamic_cast<trtf::VLBackendFastPath*>(mBackend.get());
        if (vl != nullptr)
        {
            // Step 1-2: Load image + run vision encoder (C++ native)
            std::vector<float> image_features;
            int32_t num_features = 0;
            int32_t feature_dim = 0;
            std::string prep_error;
            if (!vl->prepare_image(std::string(image_path),
                    image_features, num_features, feature_dim, prep_error))
            {
                std::cerr << "[trtf] VL image preparation failed: " << prep_error
                          << ", falling back to text-only" << std::endl;
                auto input_ids = mTokenizer->encode(prompt);
                auto output_ids = mBackend->generate(input_ids, config);
                mLastOutput = mTokenizer->decode(output_ids);
                return mLastOutput.c_str();
            }

            // Step 3: Format VL prompt with image pad tokens (C++ string)
            const std::string formatted = trtf::format_vl_prompt(
                std::string(prompt), vl->vl_config());

            // Step 4: Tokenize via Python tokenizer subprocess
            auto input_ids = mTokenizer->encode(formatted.c_str());

            // Step 5-6: Generate with image features + decode
            auto output_ids = vl->generate_vl(
                input_ids,
                image_features.data(), num_features, feature_dim,
                {}, config);
            mLastOutput = mTokenizer->decode(output_ids);
        }
        else
#endif // TRTF_HAS_TRT
        {
            auto input_ids = mTokenizer->encode(prompt);
            auto output_ids = mBackend->generate(input_ids, config);
            mLastOutput = mTokenizer->decode(output_ids);
        }
        return mLastOutput.c_str();
    }

    bool supports_vision() const override
    {
        return mBackend->supports_vision();
    }

    bool supports_video() const override
    {
#if TRTF_HAS_TRT
        auto* diff = dynamic_cast<trtf::IDiffusionBackend*>(mBackend.get());
        return diff != nullptr;
#else
        return false;
#endif
    }

    int32_t generate_video(const char* prompt, const char* output_dir,
                           int32_t num_steps, float guidance_scale) override
    {
#if TRTF_HAS_TRT
        auto* diff = dynamic_cast<trtf::IDiffusionBackend*>(mBackend.get());
        if (diff == nullptr)
        {
            return -1;
        }

        // Pass raw prompt text for subprocess-based backends (Z-Image, etc.)
        if (prompt != nullptr)
        {
            diff->set_prompt(std::string(prompt));
        }

        // Tokenize prompt (apply chat template if backend requires it)
        std::vector<int32_t> input_ids;
        if (mTokenizer && prompt != nullptr)
        {
            const std::string prepared = diff->prepare_prompt(std::string(prompt));
            input_ids = mTokenizer->encode(prepared);
        }

        // Generate video
        auto video = diff->generate_video(input_ids, num_steps, guidance_scale);
        if (video.frames.empty() || video.num_frames <= 0)
        {
            return -1;
        }

        // Write PNG frames
        const std::string dir(output_dir != nullptr ? output_dir : "/tmp/trtf_frames");
        std::filesystem::create_directories(dir);

        for (int32_t t = 0; t < video.num_frames; ++t)
        {
            const auto frame_size = static_cast<std::size_t>(video.height) *
                                    static_cast<std::size_t>(video.width) * 3;
            const float* frame_f = video.frames.data() +
                static_cast<std::size_t>(t) * frame_size;

            // Convert float32 [0,1] -> uint8 [0,255]
            std::vector<uint8_t> pixels(frame_size);
            for (std::size_t i = 0; i < frame_size; ++i)
            {
                pixels[i] = static_cast<uint8_t>(
                    std::max(0.0F, std::min(255.0F, frame_f[i] * 255.0F)));
            }

            char fname[64];
            std::snprintf(fname, sizeof(fname), "/frame_%04d.png", t);
            const std::string path = dir + fname;

            stbi_write_png(path.c_str(), video.width, video.height, 3,
                           pixels.data(), video.width * 3);
        }

        std::cerr << "[trtf] Wrote " << video.num_frames << " PNG frames to "
                  << dir << std::endl;
        return video.num_frames;
#else
        (void) prompt; (void) output_dir; (void) num_steps; (void) guidance_scale;
        return -1;
#endif
    }

    const char* model_id() const override
    {
        return mModelId.c_str();
    }

    const char* backend_name() const override
    {
        return mBackendName.c_str();
    }

    bool supports_segmentation() const override
    {
#if TRTF_HAS_TRT
        return mSegBackend != nullptr;
#else
        return false;
#endif
    }

    int32_t segment(const char* image_path, const char* output_path) override
    {
#if TRTF_HAS_TRT
        if (mSegBackend == nullptr || image_path == nullptr || output_path == nullptr)
            return -1;

        try
        {
            auto result = mSegBackend->segment_image(std::string(image_path));

            // Write PNG with raw class indices (NOT scaled to 255).
            // Each pixel stores the class index as a grayscale value.
            const int32_t w = result.width;
            const int32_t h = result.height;
            std::vector<uint8_t> pixels(static_cast<std::size_t>(w) * h);
            for (int32_t i = 0; i < w * h; ++i)
            {
                // Store raw class index (0-149 for ADE20K)
                pixels[i] = static_cast<uint8_t>(
                    std::min(result.class_map[i], 255));
            }

            stbi_write_png(output_path, w, h, 1, pixels.data(), w);
            std::cerr << "[trtf] Segmentation saved: " << output_path
                      << " (" << w << "x" << h << ", "
                      << result.num_classes << " classes)" << std::endl;
            return 0;
        }
        catch (const std::exception& e)
        {
            std::cerr << "[trtf] Segmentation error: " << e.what() << std::endl;
            return -1;
        }
#else
        (void) image_path; (void) output_path;
        return -1;
#endif
    }

    bool supports_audio() const override
    {
#if TRTF_HAS_TRT
        return mBarkBackend != nullptr || mOmniBackend != nullptr;
#else
        return false;
#endif
    }

    int32_t generate_audio(const char* prompt, const char* output_path,
                           int32_t max_tokens) override
    {
#if TRTF_HAS_TRT
        // Try Omni backend first
        if (mOmniBackend != nullptr && prompt != nullptr && output_path != nullptr)
        {
            try
            {
                std::vector<int32_t> input_ids;
                if (mTokenizer)
                {
                    input_ids = mTokenizer->encode(prompt);
                }

                auto result = mOmniBackend->generate_audio(
                    input_ids, max_tokens > 0 ? max_tokens : 768);

                if (result.num_samples <= 0)
                    return -1;

                trtf::write_wav(std::string(output_path),
                    result.waveform.data(), result.num_samples, result.sample_rate);

                std::cerr << "[trtf] Omni audio saved: " << output_path
                          << " (" << result.num_samples << " samples @ "
                          << result.sample_rate << " Hz)" << std::endl;
                return result.num_samples;
            }
            catch (const std::exception& e)
            {
                std::cerr << "[trtf] Omni audio error: " << e.what() << std::endl;
                return -1;
            }
        }

        if (mBarkBackend == nullptr || prompt == nullptr || output_path == nullptr)
            return -1;

        try
        {
            std::vector<int32_t> input_ids;
            if (mTokenizer)
            {
                input_ids = mTokenizer->encode(prompt);
            }

            auto result = mBarkBackend->generate_audio(
                input_ids, max_tokens > 0 ? max_tokens : 768);

            if (result.num_samples <= 0)
                return -1;

            trtf::write_wav(std::string(output_path),
                result.waveform.data(), result.num_samples, result.sample_rate);

            std::cerr << "[trtf] Audio saved: " << output_path
                      << " (" << result.num_samples << " samples @ "
                      << result.sample_rate << " Hz)" << std::endl;
            return result.num_samples;
        }
        catch (const std::exception& e)
        {
            std::cerr << "[trtf] Audio generation error: " << e.what() << std::endl;
            return -1;
        }
#else
        (void) prompt; (void) output_path; (void) max_tokens;
        return -1;
#endif
    }

    bool supports_transcription() const override
    {
#if TRTF_HAS_TRT
        return mWhisperBackend != nullptr;
#else
        return false;
#endif
    }

    const char* transcribe(const char* audio_path, int32_t max_new_tokens) override
    {
#if TRTF_HAS_TRT
        if (mWhisperBackend == nullptr || audio_path == nullptr)
            return nullptr;

        try
        {
            char mel_temp[] = "/tmp/trtf_mel_XXXXXX";
            const int fd = mkstemp(mel_temp);
            if (fd < 0)
            {
                std::cerr << "[trtf] Failed to create temp file for mel" << std::endl;
                return nullptr;
            }
            close(fd);

            const std::string script = trtf::script_path("hf_mel_extract.py");
            const std::string python = mHfPython.empty() ? "python3" : mHfPython;
            const std::string cmd = shell_quote(python) + " " + shell_quote(script)
                + " --audio " + shell_quote(std::string(audio_path))
                + " --output " + shell_quote(std::string(mel_temp));

            const auto sub = run_subprocess(cmd);
            if (sub.exit_code != 0)
            {
                std::cerr << "[trtf] Mel extraction failed: " << sub.output << std::endl;
                std::error_code ec;
                std::filesystem::remove(mel_temp, ec);
                return nullptr;
            }

            std::ifstream in(mel_temp, std::ios::binary);
            if (!in)
            {
                std::cerr << "[trtf] Failed to read mel temp file" << std::endl;
                std::error_code ec;
                std::filesystem::remove(mel_temp, ec);
                return nullptr;
            }

            int32_t shape[2] = {0, 0};
            in.read(reinterpret_cast<char*>(shape), sizeof(shape));
            if (!in || shape[0] <= 0 || shape[1] <= 0)
            {
                std::cerr << "[trtf] Invalid mel shape" << std::endl;
                std::error_code ec;
                std::filesystem::remove(mel_temp, ec);
                return nullptr;
            }

            const auto nf = static_cast<std::size_t>(shape[0]) * static_cast<std::size_t>(shape[1]);
            std::vector<float> mel_data(nf);
            in.read(reinterpret_cast<char*>(mel_data.data()),
                static_cast<std::streamsize>(nf * sizeof(float)));
            in.close();
            { std::error_code ec; std::filesystem::remove(mel_temp, ec); }

            std::cerr << "[trtf] Mel spectrogram: " << shape[0] << " bins x "
                      << shape[1] << " frames" << std::endl;

            auto tr = mWhisperBackend->transcribe(
                mel_data.data(), shape[0], shape[1],
                max_new_tokens > 0 ? max_new_tokens : 224);

            if (mTokenizer && !tr.output_ids.empty())
                mLastOutput = mTokenizer->decode(tr.output_ids);
            else
                mLastOutput = tr.text;

            std::cerr << "[trtf] Transcribed " << tr.num_tokens << " tokens" << std::endl;
            return mLastOutput.c_str();
        }
        catch (const std::exception& e)
        {
            std::cerr << "[trtf] Transcription error: " << e.what() << std::endl;
            return nullptr;
        }
#else
        (void) audio_path; (void) max_new_tokens;
        return nullptr;
#endif
    }

    void set_bundle_temp_dir(std::string dir) { mBundleTempDir = std::move(dir); }

#if TRTF_HAS_TRT
    void set_seg_backend(std::unique_ptr<trtf::SegmentationBackend> backend)
    {
        mSegBackend = std::move(backend);
    }
    void set_bark_backend(std::unique_ptr<trtf::BarkBackend> backend)
    {
        mBarkBackend = std::move(backend);
    }
    void set_omni_backend(std::unique_ptr<trtf::OmniBackend> backend)
    {
        mOmniBackend = std::move(backend);
    }
    void set_encoder_backend(std::unique_ptr<trtf::EncoderBackend> backend)
    {
        mEncoderBackend = std::move(backend);
    }
    void set_neural_operator_backend(std::unique_ptr<trtf::NeuralOperatorBackend> backend)
    {
        mNeuralOpBackend = std::move(backend);
    }
    void set_detection_backend(std::unique_ptr<trtf::DetectionBackend> backend)
    {
        mDetBackend = std::move(backend);
    }
    void set_speech_backend(std::unique_ptr<trtf::SpeechToSpeechBackend> backend)
    {
        mSpeechBackend = std::move(backend);
    }
    void set_embedding_backend(std::unique_ptr<trtf::EmbeddingBackend> backend)
    {
        mEmbeddingBackend = std::move(backend);
    }
    void set_reranking_backend(std::unique_ptr<trtf::RerankingBackend> backend)
    {
        mRerankBackend = std::move(backend);
    }
    void set_whisper_backend(std::unique_ptr<trtf::WhisperBackend> backend)
    {
        mWhisperBackend = std::move(backend);
    }
    void set_sam_backend(std::unique_ptr<trtf::SamBackend> backend)
    {
        mSamBackend = std::move(backend);
    }
#endif

    void set_hf_python(std::string path) { mHfPython = std::move(path); }

    bool supports_embedding() const override
    {
#if TRTF_HAS_TRT
        return mEmbeddingBackend != nullptr;
#else
        return false;
#endif
    }

    const float* embed(const char* text, int32_t* out_dim) override
    {
#if TRTF_HAS_TRT
        if (mEmbeddingBackend == nullptr || text == nullptr) return nullptr;
        try {
            std::vector<int32_t> ids;
            if (mTokenizer) ids = mTokenizer->encode(text);
            mLastEmbResult = mEmbeddingBackend->embed(ids);
            if (out_dim) *out_dim = mLastEmbResult.embedding_dim;
            return mLastEmbResult.embedding.data();
        } catch (const std::exception& e) {
            std::cerr << "[trtf] Embed error: " << e.what() << std::endl;
            return nullptr;
        }
#else
        (void) text; (void) out_dim; return nullptr;
#endif
    }

    bool supports_reranking() const override
    {
#if TRTF_HAS_TRT
        return mRerankBackend != nullptr;
#else
        return false;
#endif
    }

    float rerank(const char* query, const char* document) override
    {
#if TRTF_HAS_TRT
        if (mRerankBackend == nullptr || !query || !document) return 0.0F;
        try {
            std::string combined = std::string(query) + " " + std::string(document);
            std::vector<int32_t> ids;
            if (mTokenizer) ids = mTokenizer->encode(combined.c_str());
            return mRerankBackend->rerank(ids).score;
        } catch (const std::exception& e) {
            std::cerr << "[trtf] Rerank error: " << e.what() << std::endl;
            return 0.0F;
        }
#else
        (void) query; (void) document; return 0.0F;
#endif
    }

    bool supports_encoding() const override
    {
#if TRTF_HAS_TRT
        return mEncoderBackend != nullptr;
#else
        return false;
#endif
    }

    const float* encode(const char* text, int32_t* out_seq_len,
                        int32_t* out_hidden_size) override
    {
#if TRTF_HAS_TRT
        if (mEncoderBackend == nullptr || text == nullptr)
            return nullptr;

        try
        {
            std::vector<int32_t> input_ids;
            if (mTokenizer)
            {
                input_ids = mTokenizer->encode(text);
            }

            mLastEncoderResult = mEncoderBackend->encode(input_ids);

            if (out_seq_len != nullptr)
                *out_seq_len = mLastEncoderResult.seq_length;
            if (out_hidden_size != nullptr)
                *out_hidden_size = mLastEncoderResult.hidden_size;

            return mLastEncoderResult.hidden_states.data();
        }
        catch (const std::exception& e)
        {
            std::cerr << "[trtf] Encode error: " << e.what() << std::endl;
            return nullptr;
        }
#else
        (void) text; (void) out_seq_len; (void) out_hidden_size;
        return nullptr;
#endif
    }

    bool supports_solve() const override
    {
#if TRTF_HAS_TRT
        return mNeuralOpBackend != nullptr;
#else
        return false;
#endif
    }

    const float* solve(const float* branch_input, int32_t branch_len,
                       const float* trunk_input, int32_t trunk_len,
                       int32_t* out_dim) override
    {
#if TRTF_HAS_TRT
        if (mNeuralOpBackend == nullptr || branch_input == nullptr || trunk_input == nullptr)
            return nullptr;

        try
        {
            mLastSolveResult = mNeuralOpBackend->solve(
                branch_input, branch_len, trunk_input, trunk_len);

            if (out_dim != nullptr)
                *out_dim = mLastSolveResult.output_dim;

            return mLastSolveResult.output.data();
        }
        catch (const std::exception& e)
        {
            std::cerr << "[trtf] Solve error: " << e.what() << std::endl;
            return nullptr;
        }
#else
        (void) branch_input; (void) branch_len;
        (void) trunk_input; (void) trunk_len; (void) out_dim;
        return nullptr;
#endif
    }

    const float* solve_field(const float* field_input, int32_t input_size,
                             int32_t* out_channels, int32_t* out_h,
                             int32_t* out_w) override
    {
#if TRTF_HAS_TRT
        if (mNeuralOpBackend == nullptr || field_input == nullptr)
            return nullptr;

        try
        {
            mLastSolveResult = mNeuralOpBackend->solve_field(
                field_input, input_size);

            if (out_channels != nullptr)
                *out_channels = mLastSolveResult.out_channels;
            if (out_h != nullptr)
                *out_h = mLastSolveResult.height;
            if (out_w != nullptr)
                *out_w = mLastSolveResult.width;

            return mLastSolveResult.output.data();
        }
        catch (const std::exception& e)
        {
            std::cerr << "[trtf] Solve field error: " << e.what() << std::endl;
            return nullptr;
        }
#else
        (void) field_input; (void) input_size;
        (void) out_channels; (void) out_h; (void) out_w;
        return nullptr;
#endif
    }

    bool supports_detection() const override
    {
#if TRTF_HAS_TRT
        return mDetBackend != nullptr;
#else
        return false;
#endif
    }

    int32_t detect(const char* image_path, const char* output_path,
                   float conf_threshold) override
    {
#if TRTF_HAS_TRT
        if (mDetBackend == nullptr || image_path == nullptr || output_path == nullptr)
            return -1;

        try
        {
            (void) conf_threshold;
            auto result = mDetBackend->detect_image(std::string(image_path));

            std::string json = "[\n";
            for (std::size_t i = 0; i < result.detections.size(); ++i)
            {
                const auto& d = result.detections[i];
                if (i > 0) json += ",\n";
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "  {\"class_id\": %d, \"confidence\": %.4f, "
                    "\"x1\": %.1f, \"y1\": %.1f, \"x2\": %.1f, \"y2\": %.1f}",
                    d.class_id, d.confidence, d.x1, d.y1, d.x2, d.y2);
                json += buf;
            }
            json += "\n]\n";

            std::FILE* fp = std::fopen(output_path, "w");
            if (fp == nullptr)
            {
                std::cerr << "[trtf] Failed to open output: " << output_path << std::endl;
                return -1;
            }
            std::fwrite(json.data(), 1, json.size(), fp);
            std::fclose(fp);

            std::cerr << "[trtf] Detection saved: " << output_path
                      << " (" << result.detections.size() << " objects)" << std::endl;
            return static_cast<int32_t>(result.detections.size());
        }
        catch (const std::exception& e)
        {
            std::cerr << "[trtf] Detection error: " << e.what() << std::endl;
            return -1;
        }
#else
        (void) image_path; (void) output_path; (void) conf_threshold;
        return -1;
#endif
    }

    bool supports_speech() const override
    {
#if TRTF_HAS_TRT
        return mSpeechBackend != nullptr;
#else
        return false;
#endif
    }

    int32_t speak(const char* audio_in, const char* audio_out,
                  int32_t max_output_frames) override
    {
#if TRTF_HAS_TRT
        if (!mSpeechBackend || !audio_in || !audio_out) return -1;
        try {
            std::ifstream infile(audio_in, std::ios::binary);
            if (!infile) return -1;

            // Read full WAV file and parse chunks (do not assume 44-byte header).
            std::vector<char> wav_bytes((std::istreambuf_iterator<char>(infile)),
                                        std::istreambuf_iterator<char>());
            if (wav_bytes.size() < 44) return -1;
            if (std::memcmp(wav_bytes.data(), "RIFF", 4) != 0 ||
                std::memcmp(wav_bytes.data() + 8, "WAVE", 4) != 0)
            {
                std::cerr << "[trtf] Invalid WAV container" << std::endl;
                return -1;
            }

            uint16_t fmt_tag = 0;
            uint16_t channels = 0;
            uint32_t sample_rate = 0;
            uint16_t bits_per_sample = 0;
            const char* raw_ptr = nullptr;
            std::size_t raw_size = 0;
            bool have_fmt = false;
            bool have_data = false;

            std::size_t pos = 12;
            while (pos + 8 <= wav_bytes.size())
            {
                const char* chunk = wav_bytes.data() + pos;
                const char* chunk_data = chunk + 8;
                auto chunk_size = *reinterpret_cast<const uint32_t*>(chunk + 4);
                if (pos + 8 + static_cast<std::size_t>(chunk_size) > wav_bytes.size())
                    break;

                if (std::memcmp(chunk, "fmt ", 4) == 0)
                {
                    if (chunk_size < 16)
                    {
                        std::cerr << "[trtf] WAV fmt chunk too small" << std::endl;
                        return -1;
                    }
                    fmt_tag = *reinterpret_cast<const uint16_t*>(chunk_data + 0);
                    channels = *reinterpret_cast<const uint16_t*>(chunk_data + 2);
                    sample_rate = *reinterpret_cast<const uint32_t*>(chunk_data + 4);
                    bits_per_sample = *reinterpret_cast<const uint16_t*>(chunk_data + 14);
                    have_fmt = true;
                }
                else if (std::memcmp(chunk, "data", 4) == 0)
                {
                    raw_ptr = chunk_data;
                    raw_size = static_cast<std::size_t>(chunk_size);
                    have_data = true;
                    // Keep parsing in case fmt appears later, but first data is enough.
                }

                pos += 8 + static_cast<std::size_t>(chunk_size);
                if ((chunk_size & 1U) != 0)
                    ++pos;  // RIFF chunks are word-aligned.
            }

            if (!have_fmt || !have_data || raw_ptr == nullptr || raw_size == 0)
            {
                std::cerr << "[trtf] WAV missing fmt/data chunk" << std::endl;
                return -1;
            }

            // Convert to float32 mono samples
            std::vector<float> samples;
            if (fmt_tag == 3 && bits_per_sample == 32)
            {
                // IEEE float32
                auto ns = static_cast<int32_t>(raw_size / sizeof(float));
                samples.resize(ns);
                std::memcpy(samples.data(), raw_ptr, ns * sizeof(float));
            }
            else if (fmt_tag == 1 && bits_per_sample == 16)
            {
                // PCM int16
                auto ns = static_cast<int32_t>(raw_size / sizeof(int16_t));
                samples.resize(ns);
                const auto* pcm = reinterpret_cast<const int16_t*>(raw_ptr);
                for (int32_t i = 0; i < ns; ++i)
                    samples[i] = static_cast<float>(pcm[i]) / 32768.0F;
            }
            else
            {
                std::cerr << "[trtf] Unsupported WAV format: tag="
                          << fmt_tag << " bits=" << bits_per_sample << std::endl;
                return -1;
            }

            // Convert stereo to mono (average channels)
            if (channels == 2)
            {
                auto mono_len = static_cast<int32_t>(samples.size()) / 2;
                std::vector<float> mono(mono_len);
                for (int32_t i = 0; i < mono_len; ++i)
                    mono[i] = (samples[2 * i] + samples[2 * i + 1]) * 0.5F;
                samples = std::move(mono);
            }
            else if (channels > 2)
            {
                // Take first channel
                auto mono_len = static_cast<int32_t>(samples.size()) / channels;
                std::vector<float> mono(mono_len);
                for (int32_t i = 0; i < mono_len; ++i)
                    mono[i] = samples[i * channels];
                samples = std::move(mono);
            }

            // Simple resample to 24kHz if needed (Mimi codec rate)
            const int32_t target_rate = mSpeechBackend->config().sample_rate;
            const bool prefer_python_codec =
                !mSpeechBackend->config().hf_python.empty();
            if (static_cast<int32_t>(sample_rate) != target_rate && sample_rate > 0)
            {
                if (prefer_python_codec)
                {
                    std::cerr << "[trtf] Keeping input at " << sample_rate
                              << " Hz; Python Mimi encoder will resample to "
                              << target_rate << " Hz" << std::endl;
                }
                else
                {
                    auto in_len = static_cast<int32_t>(samples.size());
                    auto out_len = static_cast<int32_t>(
                        static_cast<int64_t>(in_len) * target_rate / sample_rate);
                    std::vector<float> resampled(out_len);
                    for (int32_t i = 0; i < out_len; ++i)
                    {
                        float src_pos = static_cast<float>(i) *
                            static_cast<float>(sample_rate) /
                            static_cast<float>(target_rate);
                        auto idx = static_cast<int32_t>(src_pos);
                        float frac = src_pos - static_cast<float>(idx);
                        if (idx + 1 < in_len)
                            resampled[i] = samples[idx] * (1.0F - frac)
                                          + samples[idx + 1] * frac;
                        else if (idx < in_len)
                            resampled[i] = samples[idx];
                    }
                    samples = std::move(resampled);
                    std::cerr << "[trtf] Resampled " << sample_rate << " -> "
                              << target_rate << " Hz (" << in_len << " -> "
                              << out_len << " samples)" << std::endl;
                    sample_rate = static_cast<uint32_t>(target_rate);
                }
            }

            auto ns = static_cast<int32_t>(samples.size());
            if (ns <= 0) return -1;
            auto r = mSpeechBackend->process_audio(
                samples.data(), ns,
                max_output_frames > 0 ? max_output_frames : 375,
                static_cast<int32_t>(sample_rate));
            if (r.num_samples <= 0) return -1;
            trtf::write_wav(std::string(audio_out),
                r.waveform.data(), r.num_samples, r.sample_rate);
            return r.num_samples;
        } catch (const std::exception& e) {
            std::cerr << "[trtf] Speech error: " << e.what() << std::endl;
            return -1;
        }
#else
        (void) audio_in; (void) audio_out; (void) max_output_frames;
        return -1;
#endif
    }

    bool supports_prompted_segmentation() const override
    {
#if TRTF_HAS_TRT
        return mSamBackend != nullptr;
#else
        return false;
#endif
    }

    int32_t segment_sam(const char* image_path, const char* output_dir,
                        float point_x, float point_y, bool is_foreground) override
    {
#if TRTF_HAS_TRT
        if (mSamBackend == nullptr || image_path == nullptr || output_dir == nullptr)
            return -1;

        try
        {
            // Encode image (cached on GPU)
            if (!mSamBackend->encode_image(std::string(image_path)))
            {
                std::cerr << "[trtf] SAM: Failed to encode image" << std::endl;
                return -1;
            }

            // Segment with point prompt
            auto result = mSamBackend->segment_point(point_x, point_y, is_foreground);

            // Write each mask as a separate PNG
            std::filesystem::create_directories(output_dir);
            for (int32_t m = 0; m < result.num_masks; ++m)
            {
                const auto offset = static_cast<std::size_t>(m) *
                    static_cast<std::size_t>(result.mask_height) * result.mask_width;
                std::vector<uint8_t> pixels(
                    static_cast<std::size_t>(result.mask_height) * result.mask_width);
                for (int32_t i = 0; i < result.mask_height * result.mask_width; ++i)
                {
                    // Sigmoid threshold at 0 -> binary mask
                    pixels[i] = result.masks[offset + i] > 0.0F
                        ? static_cast<uint8_t>(255) : static_cast<uint8_t>(0);
                }

                char fname[64];
                std::snprintf(fname, sizeof(fname), "/mask_%d_iou_%.4f.png",
                              m, result.iou_scores[m]);
                const std::string path = std::string(output_dir) + fname;

                stbi_write_png(path.c_str(), result.mask_width, result.mask_height,
                               1, pixels.data(), result.mask_width);
            }

            std::cerr << "[trtf] SAM: " << result.num_masks << " masks saved to "
                      << output_dir << std::endl;
            return result.num_masks;
        }
        catch (const std::exception& e)
        {
            std::cerr << "[trtf] SAM error: " << e.what() << std::endl;
            return -1;
        }
#else
        (void) image_path; (void) output_dir;
        (void) point_x; (void) point_y; (void) is_foreground;
        return -1;
#endif
    }

private:
    std::string mModelId;
    std::unique_ptr<trtf::ITokenizer> mTokenizer;
    std::unique_ptr<trtf::IGenerationBackend> mBackend;
    std::string mBackendName;
    trtf::GenerationConfig mGenConfig;
    std::string mLastOutput;
    std::string mBundleTempDir;
#if TRTF_HAS_TRT
    std::unique_ptr<trtf::SegmentationBackend> mSegBackend;
    std::unique_ptr<trtf::BarkBackend> mBarkBackend;
    std::unique_ptr<trtf::OmniBackend> mOmniBackend;
    std::unique_ptr<trtf::EncoderBackend> mEncoderBackend;
    trtf::EncoderResult mLastEncoderResult;
    std::unique_ptr<trtf::NeuralOperatorBackend> mNeuralOpBackend;
    trtf::NeuralOperatorResult mLastSolveResult;
    std::unique_ptr<trtf::SpeechToSpeechBackend> mSpeechBackend;
    std::unique_ptr<trtf::DetectionBackend> mDetBackend;
    std::unique_ptr<trtf::EmbeddingBackend> mEmbeddingBackend;
    trtf::EmbeddingResult mLastEmbResult;
    std::unique_ptr<trtf::RerankingBackend> mRerankBackend;
    std::unique_ptr<trtf::SamBackend> mSamBackend;
    std::unique_ptr<trtf::WhisperBackend> mWhisperBackend;
#endif
    std::string mHfPython;
};

#if TRTF_HAS_TRT

// Helper: assemble a PipelineImpl from backend components + tokenizer.
PipelineImpl* make_pipeline(
    const std::string& model_id,
    trtf::TokenizerResult tok,
    std::unique_ptr<trtf::IGenerationBackend> backend,
    const std::string& backend_name)
{
    auto* pipeline = new PipelineImpl(
        model_id, std::move(tok.tokenizer), std::move(backend), backend_name, trtf::GenerationConfig{});
    pipeline->set_bundle_temp_dir(std::move(tok.temp_dir));
    return pipeline;
}

// --- Per-strategy creation functions ---

PipelineImpl* create_mamba_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    auto mamba_engine = std::make_unique<trtf::MambaStepEngine>();
    mamba_engine->engine = std::move(trt_engine);
    mamba_engine->context = std::move(exec_ctx);
    mamba_engine->vocab_size = fp_cfg.vocab_size;
    mamba_engine->hidden_size = fp_cfg.hidden_size;
    mamba_engine->d_inner = fp_cfg.d_inner;
    mamba_engine->state_size = fp_cfg.state_size;
    mamba_engine->conv_kernel = fp_cfg.conv_kernel;
    mamba_engine->num_layers = fp_cfg.num_layers;
    mamba_engine->id_bos = fp_cfg.id_bos;
    mamba_engine->id_eos = fp_cfg.id_eos;

    for (int32_t i = 0; i < fp_cfg.num_layers; ++i)
    {
        mamba_engine->conv_state_input_names.push_back(trtf::layer_tensor_name("conv_state", i));
        mamba_engine->ssm_state_input_names.push_back(trtf::layer_tensor_name("ssm_state", i));
        mamba_engine->present_conv_output_names.push_back(trtf::layer_tensor_name("present_conv", i));
        mamba_engine->present_ssm_output_names.push_back(trtf::layer_tensor_name("present_ssm", i));
    }

    if (!trtf::has_all_required_mamba_tensors(*mamba_engine))
    {
        throw std::runtime_error("Bundle engine missing required Mamba tensors: " + bundle_path);
    }

    auto tok = trtf::extract_tokenizer_from_bundle(sections, hf_python);
    auto backend = trtf::CreateMambaBackendFromEngine(std::move(mamba_engine));
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create Mamba TRT backend from bundle engine");
    }

    std::cerr << "[trtf] Runtime ready (backend=trt_mamba, strategy=ssm_recurrent)" << std::endl;
    return make_pipeline(model_id, std::move(tok), std::move(backend), "trt_mamba");
}

PipelineImpl* create_rwkv_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    auto rwkv_engine = std::make_unique<trtf::RwkvStepEngine>();
    rwkv_engine->engine = std::move(trt_engine);
    rwkv_engine->context = std::move(exec_ctx);
    rwkv_engine->vocab_size = fp_cfg.vocab_size;
    rwkv_engine->hidden_size = fp_cfg.hidden_size;
    rwkv_engine->num_layers = fp_cfg.num_layers;
    rwkv_engine->id_bos = fp_cfg.id_bos;
    rwkv_engine->id_eos = fp_cfg.id_eos;

    for (int32_t i = 0; i < fp_cfg.num_layers; ++i)
    {
        rwkv_engine->attn_state_input_names.push_back(trtf::layer_tensor_name("attn_state", i));
        rwkv_engine->ff_state_input_names.push_back(trtf::layer_tensor_name("ff_state", i));
        rwkv_engine->num_state_input_names.push_back(trtf::layer_tensor_name("num_state", i));
        rwkv_engine->den_state_input_names.push_back(trtf::layer_tensor_name("den_state", i));
        rwkv_engine->max_state_input_names.push_back(trtf::layer_tensor_name("max_state", i));
        rwkv_engine->present_attn_output_names.push_back(trtf::layer_tensor_name("present_attn", i));
        rwkv_engine->present_ff_output_names.push_back(trtf::layer_tensor_name("present_ff", i));
        rwkv_engine->present_num_output_names.push_back(trtf::layer_tensor_name("present_num", i));
        rwkv_engine->present_den_output_names.push_back(trtf::layer_tensor_name("present_den", i));
        rwkv_engine->present_max_output_names.push_back(trtf::layer_tensor_name("present_max", i));
    }

    if (!trtf::has_all_required_rwkv_tensors(*rwkv_engine))
    {
        throw std::runtime_error("Bundle engine missing required RWKV tensors: " + bundle_path);
    }

    auto tok = trtf::extract_tokenizer_from_bundle(sections, hf_python);
    auto backend = trtf::CreateRwkvBackendFromEngine(std::move(rwkv_engine));
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create RWKV TRT backend from bundle engine");
    }

    std::cerr << "[trtf] Runtime ready (backend=trt_rwkv, strategy=rwkv_recurrent)" << std::endl;
    return make_pipeline(model_id, std::move(tok), std::move(backend), "trt_rwkv");
}

PipelineImpl* create_whisper_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    // Build text decoder engine (standard KV cache decoder)
    auto decoder_engine = trtf::make_decoder_engine(std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!trtf::has_all_required_tensors(*decoder_engine))
    {
        throw std::runtime_error("Bundle engine missing required decoder tensors: " + bundle_path);
    }

    // Deserialize encoder engine from vision_engine_plan section
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> encoder_engine;
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> encoder_ctx;
    if (sections.vision_plan_data != nullptr && !sections.vision_plan_data->empty())
    {
        std::cerr << "[trtf] Deserializing Whisper encoder TRT engine ("
                  << sections.vision_plan_data->size() / (1024 * 1024) << " MB) ..." << std::endl;

        encoder_engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime_ptr->deserializeCudaEngine(
                sections.vision_plan_data->data(), sections.vision_plan_data->size()));
        if (!encoder_engine)
        {
            throw std::runtime_error("Failed to deserialize Whisper encoder engine: " + bundle_path);
        }
        encoder_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
            encoder_engine->createExecutionContext());
        if (!encoder_ctx)
        {
            throw std::runtime_error("Failed to create Whisper encoder execution context");
        }
    }

    trtf::WhisperConfig whisper_cfg;
    whisper_cfg.num_mel_bins = fp_cfg.num_mel_bins;
    whisper_cfg.max_source_positions = fp_cfg.max_source_positions;
    whisper_cfg.max_target_positions = fp_cfg.max_target_positions;
    whisper_cfg.encoder_layers = fp_cfg.encoder_layers;
    whisper_cfg.decoder_layers = fp_cfg.decoder_layers;

    auto tok = trtf::extract_tokenizer_from_bundle(sections, hf_python);
    auto backend = trtf::CreateWhisperBackend(
        std::move(decoder_engine), std::move(encoder_engine), std::move(encoder_ctx),
        whisper_cfg, fp_cfg);
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create Whisper TRT backend from bundle engine");
    }

    std::cerr << "[trtf] Runtime ready (backend=trt_whisper, strategy=speech_to_text)" << std::endl;

    auto* pipeline = new PipelineImpl(
        model_id, std::move(tok.tokenizer), nullptr, "trt_whisper", trtf::GenerationConfig{});
    if (!tok.temp_dir.empty())
        pipeline->set_bundle_temp_dir(std::move(tok.temp_dir));
    pipeline->set_whisper_backend(std::move(backend));
    pipeline->set_hf_python(hf_python);
    return pipeline;
}

PipelineImpl* create_vl_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    // Build text decoder engine
    auto decoder_engine = trtf::make_decoder_engine(std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!trtf::has_all_required_tensors(*decoder_engine))
    {
        throw std::runtime_error("Bundle engine missing required tensors: " + bundle_path);
    }

    // Deserialize vision engine (if present)
    std::unique_ptr<trtf::VisionStepEngine> vision_step_engine;
    if (sections.vision_plan_data != nullptr && !sections.vision_plan_data->empty())
    {
        std::cerr << "[trtf] Deserializing vision TRT engine from bundle ("
                  << sections.vision_plan_data->size() / (1024 * 1024) << " MB) ..." << std::endl;

        auto vision_trt_engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime_ptr->deserializeCudaEngine(
                sections.vision_plan_data->data(), sections.vision_plan_data->size()));
        if (!vision_trt_engine)
        {
            throw std::runtime_error("Failed to deserialize vision engine from bundle: " + bundle_path);
        }

        auto vision_exec_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
            vision_trt_engine->createExecutionContext());
        if (!vision_exec_ctx)
        {
            throw std::runtime_error("Failed to create vision execution context");
        }

        vision_step_engine = std::make_unique<trtf::VisionStepEngine>();
        vision_step_engine->engine = std::move(vision_trt_engine);
        vision_step_engine->context = std::move(vision_exec_ctx);
        vision_step_engine->num_output_features = fp_cfg.num_image_pad_tokens;
        vision_step_engine->feature_dim = (fp_cfg.vision_output_dim > 0)
            ? fp_cfg.vision_output_dim : fp_cfg.hidden_size;

        std::cerr << "[trtf] Vision engine deserialized (features="
                  << vision_step_engine->num_output_features
                  << ", dim=" << vision_step_engine->feature_dim << ")" << std::endl;
    }

    // Parse VL preprocessing config
    std::string config_text_vl;
    if (sections.config_json_data != nullptr && !sections.config_json_data->empty())
    {
        config_text_vl.assign(sections.config_json_data->begin(), sections.config_json_data->end());
    }
    std::string preproc_text;
    if (sections.preprocessor_config_data != nullptr && !sections.preprocessor_config_data->empty())
    {
        preproc_text.assign(
            sections.preprocessor_config_data->begin(), sections.preprocessor_config_data->end());
    }
    trtf::VLPreprocessConfig vl_preproc = trtf::parse_vl_preprocess_config(config_text_vl, preproc_text);

    auto tok = trtf::extract_tokenizer_from_bundle(sections, hf_python, fp_cfg.tokenizer_add_special_tokens);
    auto backend = trtf::CreateVLBackendFromEngines(
        std::move(decoder_engine), std::move(vision_step_engine), fp_cfg, std::move(vl_preproc));
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create VL TRT backend from bundle engine");
    }

    std::cerr << "[trtf] Runtime ready (backend=trt_vl, strategy=vision_language)" << std::endl;
    return make_pipeline(model_id, std::move(tok), std::move(backend), "trt_vl");
}

PipelineImpl* create_segmentation_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const std::string& model_id,
    const std::string& bundle_path)
{
    auto seg_backend = trtf::CreateSegmentationBackend(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!seg_backend || !seg_backend->is_available())
    {
        throw std::runtime_error("Failed to create segmentation backend from bundle engine");
    }

    // Segmentation pipelines don't need a tokenizer or generation backend
    auto* pipeline = new PipelineImpl(
        model_id, nullptr, nullptr, "trt_segmentation", trtf::GenerationConfig{});
    pipeline->set_seg_backend(std::move(seg_backend));

    std::cerr << "[trtf] Runtime ready (backend=trt_segmentation, strategy=segmentation)" << std::endl;
    return pipeline;
}

PipelineImpl* create_detection_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const std::string& model_id,
    const std::string& bundle_path)
{
    auto det_backend = trtf::CreateDetectionBackend(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!det_backend || !det_backend->is_available())
    {
        throw std::runtime_error("Failed to create detection backend from bundle engine");
    }

    // Detection pipelines don't need a tokenizer or generation backend
    auto* pipeline = new PipelineImpl(
        model_id, nullptr, nullptr, "trt_detection", trtf::GenerationConfig{});
    pipeline->set_detection_backend(std::move(det_backend));

    std::cerr << "[trtf] Runtime ready (backend=trt_detection, strategy=object_detection)" << std::endl;
    return pipeline;
}

PipelineImpl* create_sam_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& bundle_path)
{
    // Primary engine is the ViT image encoder
    // Deserialize mask decoder engine from vision_engine_plan section
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> decoder_engine;
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> decoder_ctx;
    if (sections.vision_plan_data != nullptr && !sections.vision_plan_data->empty())
    {
        std::cerr << "[trtf] Deserializing SAM mask decoder TRT engine ("
                  << sections.vision_plan_data->size() / (1024 * 1024) << " MB) ..." << std::endl;

        decoder_engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime_ptr->deserializeCudaEngine(
                sections.vision_plan_data->data(), sections.vision_plan_data->size()));
        if (!decoder_engine)
        {
            throw std::runtime_error("Failed to deserialize SAM mask decoder engine: " + bundle_path);
        }
        decoder_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
            decoder_engine->createExecutionContext());
        if (!decoder_ctx)
        {
            throw std::runtime_error("Failed to create SAM mask decoder execution context");
        }
    }
    else
    {
        throw std::runtime_error("SAM bundle missing vision_engine_plan (mask decoder): " + bundle_path);
    }

    auto sam_backend = trtf::CreateSamBackend(
        std::move(trt_engine), std::move(exec_ctx),
        std::move(decoder_engine), std::move(decoder_ctx), fp_cfg);
    if (!sam_backend || !sam_backend->is_available())
    {
        throw std::runtime_error("Failed to create SAM backend from bundle engines");
    }

    // SAM pipelines don't need a tokenizer or generation backend
    auto* pipeline = new PipelineImpl(
        model_id, nullptr, nullptr, "trt_sam", trtf::GenerationConfig{});
    pipeline->set_sam_backend(std::move(sam_backend));

    std::cerr << "[trtf] Runtime ready (backend=trt_sam, strategy=prompted_segmentation)" << std::endl;
    return pipeline;
}

PipelineImpl* create_neural_operator_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const std::string& model_id,
    const std::string& bundle_path)
{
    auto no_backend = trtf::CreateNeuralOperatorBackend(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!no_backend || !no_backend->is_available())
    {
        throw std::runtime_error("Failed to create neural operator backend from bundle engine");
    }

    // Neural operator pipelines don't need a tokenizer or generation backend
    auto* pipeline = new PipelineImpl(
        model_id, nullptr, nullptr, "trt_neural_operator", trtf::GenerationConfig{});
    pipeline->set_neural_operator_backend(std::move(no_backend));

    std::cerr << "[trtf] Runtime ready (backend=trt_neural_operator, strategy=neural_operator)" << std::endl;
    return pipeline;
}

PipelineImpl* create_bark_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    // Build semantic decoder engine (primary engine)
    auto semantic_engine = trtf::make_decoder_engine(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!trtf::has_all_required_tensors(*semantic_engine))
    {
        throw std::runtime_error("Bundle engine missing required semantic tensors: " + bundle_path);
    }

    // Override vocab_size from engine output shape.
    // Bark semantic: input_vocab=129600 but lm_head outputs only 10048.
    // The config.json vocab_size is the input embedding size, not the output.
    {
        auto logits_shape = semantic_engine->engine->getTensorShape("logits");
        if (logits_shape.nbDims >= 2)
        {
            int32_t actual_vocab = logits_shape.d[logits_shape.nbDims - 1];
            if (actual_vocab > 0 && actual_vocab != semantic_engine->vocab_size)
            {
                std::cerr << "[trtf] Semantic: output vocab " << actual_vocab
                          << " (config says " << semantic_engine->vocab_size << ")" << std::endl;
                semantic_engine->vocab_size = actual_vocab;
            }
        }
    }

    // Load embedding tables from bundle sections
    std::vector<float> semantic_embed;
    if (sections.semantic_embed_data != nullptr && !sections.semantic_embed_data->empty())
    {
        const auto n_floats = sections.semantic_embed_data->size() / sizeof(float);
        semantic_embed.resize(n_floats);
        std::memcpy(semantic_embed.data(), sections.semantic_embed_data->data(),
                     sections.semantic_embed_data->size());
        std::cerr << "[trtf] Loaded semantic embedding table ("
                  << n_floats / std::max(fp_cfg.hidden_size, 1) << " x "
                  << fp_cfg.hidden_size << ")" << std::endl;
    }
    else
    {
        throw std::runtime_error("Bundle missing semantic_embed section: " + bundle_path);
    }

    std::vector<float> coarse_embed;
    if (sections.coarse_embed_data != nullptr && !sections.coarse_embed_data->empty())
    {
        const auto n_floats = sections.coarse_embed_data->size() / sizeof(float);
        coarse_embed.resize(n_floats);
        std::memcpy(coarse_embed.data(), sections.coarse_embed_data->data(),
                     sections.coarse_embed_data->size());
        std::cerr << "[trtf] Loaded coarse embedding table ("
                  << n_floats / std::max(fp_cfg.hidden_size, 1) << " x "
                  << fp_cfg.hidden_size << ")" << std::endl;
    }
    else
    {
        throw std::runtime_error("Bundle missing coarse_embed section: " + bundle_path);
    }

    // Deserialize coarse engine
    std::unique_ptr<trtf::DecoderStepEngine> coarse_engine;
    if (sections.coarse_engine_plan_data != nullptr && !sections.coarse_engine_plan_data->empty())
    {
        std::cerr << "[trtf] Deserializing coarse TRT engine ("
                  << sections.coarse_engine_plan_data->size() / (1024 * 1024)
                  << " MB) ..." << std::endl;

        auto coarse_trt = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime_ptr->deserializeCudaEngine(
                sections.coarse_engine_plan_data->data(),
                sections.coarse_engine_plan_data->size()));
        if (!coarse_trt)
        {
            throw std::runtime_error("Failed to deserialize coarse engine: " + bundle_path);
        }
        auto coarse_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
            coarse_trt->createExecutionContext());
        if (!coarse_ctx)
        {
            throw std::runtime_error("Failed to create coarse execution context");
        }

        // Build coarse-specific config (may differ from semantic)
        trtf::FastPathModelConfig coarse_cfg = fp_cfg;
        if (fp_cfg.coarse_hidden_size > 0)
            coarse_cfg.hidden_size = fp_cfg.coarse_hidden_size;
        if (fp_cfg.coarse_num_layers > 0)
            coarse_cfg.num_layers = fp_cfg.coarse_num_layers;
        if (fp_cfg.coarse_num_heads > 0)
        {
            coarse_cfg.num_heads = fp_cfg.coarse_num_heads;
            coarse_cfg.num_kv_heads = fp_cfg.coarse_num_heads;
        }
        coarse_cfg.vocab_size = fp_cfg.coarse_input_vocab;
        coarse_cfg.head_dim = coarse_cfg.hidden_size / std::max(coarse_cfg.num_heads, 1);
        coarse_cfg.attention_size = coarse_cfg.num_heads * coarse_cfg.head_dim;
        coarse_cfg.max_cache_length = fp_cfg.coarse_max_cache_length;

        coarse_engine = trtf::make_decoder_engine(
            std::move(coarse_trt), std::move(coarse_ctx), coarse_cfg);
        if (!trtf::has_all_required_tensors(*coarse_engine))
        {
            throw std::runtime_error("Bundle coarse engine missing required tensors: " + bundle_path);
        }
    }
    else
    {
        throw std::runtime_error("Bundle missing coarse_engine section: " + bundle_path);
    }

    // Create BarkBackend with both engines + embedding tables
    auto bark_backend = trtf::CreateBarkBackend(
        std::move(semantic_engine), std::move(coarse_engine),
        std::move(semantic_embed), std::move(coarse_embed), fp_cfg);
    if (!bark_backend || !bark_backend->is_available())
    {
        throw std::runtime_error("Failed to create Bark backend from bundle engines");
    }

    // Deserialize optional codec engine
    if (sections.codec_engine_plan_data != nullptr && !sections.codec_engine_plan_data->empty())
    {
        auto codec_trt = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime_ptr->deserializeCudaEngine(
                sections.codec_engine_plan_data->data(),
                sections.codec_engine_plan_data->size()));
        if (codec_trt)
        {
            auto codec_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
                codec_trt->createExecutionContext());
            bark_backend->set_codec_engine(std::move(codec_trt), std::move(codec_ctx));
        }
    }

    // Deserialize optional fine engine
    if (sections.fine_engine_plan_data != nullptr && !sections.fine_engine_plan_data->empty())
    {
        std::cerr << "[trtf] Deserializing fine TRT engine ("
                  << sections.fine_engine_plan_data->size() / (1024 * 1024)
                  << " MB) ..." << std::endl;

        auto fine_trt = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime_ptr->deserializeCudaEngine(
                sections.fine_engine_plan_data->data(),
                sections.fine_engine_plan_data->size()));
        if (fine_trt)
        {
            auto fine_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
                fine_trt->createExecutionContext());
            bark_backend->set_fine_engine(std::move(fine_trt), std::move(fine_ctx));
        }
    }

    // Load fine embedding tables
    if (sections.fine_embed_data != nullptr && !sections.fine_embed_data->empty())
    {
        const auto n_floats = sections.fine_embed_data->size() / sizeof(float);
        std::vector<float> fine_embed(n_floats);
        std::memcpy(fine_embed.data(), sections.fine_embed_data->data(),
                     sections.fine_embed_data->size());

        std::vector<float> fine_pos_embed;
        if (sections.fine_position_embed_data != nullptr &&
            !sections.fine_position_embed_data->empty())
        {
            const auto pos_n = sections.fine_position_embed_data->size() / sizeof(float);
            fine_pos_embed.resize(pos_n);
            std::memcpy(fine_pos_embed.data(),
                         sections.fine_position_embed_data->data(),
                         sections.fine_position_embed_data->size());
        }

        bark_backend->set_fine_embeddings(std::move(fine_embed),
                                           std::move(fine_pos_embed));
        std::cerr << "[trtf] Loaded fine embedding tables ("
                  << n_floats << " floats)" << std::endl;
    }

    // Tokenizer (optional for Bark)
    trtf::TokenizerResult tok = {nullptr, ""};
    try
    {
        tok = trtf::extract_tokenizer_from_bundle(sections, hf_python);
    }
    catch (const std::exception& e)
    {
        std::cerr << "[trtf] Warning: no tokenizer for Bark (" << e.what() << ")" << std::endl;
    }

    std::cerr << "[trtf] Runtime ready (backend=trt_bark, strategy=text_to_audio)" << std::endl;

    auto* pipeline = new PipelineImpl(
        model_id, std::move(tok.tokenizer), nullptr, "trt_bark", trtf::GenerationConfig{});
    if (!tok.temp_dir.empty())
        pipeline->set_bundle_temp_dir(std::move(tok.temp_dir));
    pipeline->set_bark_backend(std::move(bark_backend));
    return pipeline;
}

PipelineImpl* create_encoder_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& /* bundle_path */)
{
    auto enc_backend = trtf::CreateEncoderBackend(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!enc_backend || !enc_backend->is_available())
    {
        throw std::runtime_error("Failed to create encoder backend from bundle engine");
    }

    // Tokenizer — encoder models need special tokens ([CLS], [SEP]) added
    trtf::TokenizerResult tok = {nullptr, ""};
    try
    {
        tok = trtf::extract_tokenizer_from_bundle(sections, hf_python, /*add_special_tokens=*/true);
    }
    catch (const std::exception& e)
    {
        std::cerr << "[trtf] Warning: no tokenizer for encoder (" << e.what() << ")" << std::endl;
    }

    auto* pipeline = new PipelineImpl(
        model_id, std::move(tok.tokenizer), nullptr, "trt_encoder", trtf::GenerationConfig{});
    if (!tok.temp_dir.empty())
        pipeline->set_bundle_temp_dir(std::move(tok.temp_dir));
    pipeline->set_encoder_backend(std::move(enc_backend));

    std::cerr << "[trtf] Runtime ready (backend=trt_encoder, strategy=encoder_only)" << std::endl;
    return pipeline;
}

PipelineImpl* create_embedding_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& /* bundle_path */)
{
    auto emb_backend = trtf::CreateEmbeddingBackend(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!emb_backend || !emb_backend->is_available())
    {
        throw std::runtime_error("Failed to create embedding backend from bundle engine");
    }

    trtf::TokenizerResult tok = {nullptr, ""};
    try
    {
        // Embedding models need special tokens (BOS) for correct encoding.
        tok = trtf::extract_tokenizer_from_bundle(sections, hf_python, /*add_special_tokens=*/true);
    }
    catch (const std::exception& e)
    {
        std::cerr << "[trtf] Warning: no tokenizer for embedding (" << e.what() << ")" << std::endl;
    }

    auto* pipeline = new PipelineImpl(
        model_id, std::move(tok.tokenizer), nullptr, "trt_embedding", trtf::GenerationConfig{});
    if (!tok.temp_dir.empty())
        pipeline->set_bundle_temp_dir(std::move(tok.temp_dir));
    pipeline->set_embedding_backend(std::move(emb_backend));

    std::cerr << "[trtf] Runtime ready (backend=trt_embedding, strategy=embedding)" << std::endl;
    return pipeline;
}

PipelineImpl* create_reranking_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& /* bundle_path */)
{
    auto rerank_backend = trtf::CreateRerankingBackend(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!rerank_backend || !rerank_backend->is_available())
    {
        throw std::runtime_error("Failed to create reranking backend from bundle engine");
    }

    trtf::TokenizerResult tok = {nullptr, ""};
    try
    {
        // Reranking models need special tokens (BOS) for correct encoding.
        tok = trtf::extract_tokenizer_from_bundle(sections, hf_python, /*add_special_tokens=*/true);
    }
    catch (const std::exception& e)
    {
        std::cerr << "[trtf] Warning: no tokenizer for reranking (" << e.what() << ")" << std::endl;
    }

    auto* pipeline = new PipelineImpl(
        model_id, std::move(tok.tokenizer), nullptr, "trt_reranking", trtf::GenerationConfig{});
    if (!tok.temp_dir.empty())
        pipeline->set_bundle_temp_dir(std::move(tok.temp_dir));
    pipeline->set_reranking_backend(std::move(rerank_backend));

    std::cerr << "[trtf] Runtime ready (backend=trt_reranking, strategy=reranking)" << std::endl;
    return pipeline;
}

PipelineImpl* create_omni_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    // Build Thinker decoder engine (primary engine = MoE text decoder)
    auto thinker_engine = trtf::make_decoder_engine(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!trtf::has_all_required_tensors(*thinker_engine))
    {
        throw std::runtime_error("Bundle engine missing required Thinker tensors: " + bundle_path);
    }

    // Create OmniBackend from Thinker engine + config
    auto omni_backend = trtf::CreateOmniBackend(std::move(thinker_engine), fp_cfg);
    if (!omni_backend || !omni_backend->is_available())
    {
        throw std::runtime_error("Failed to create Omni backend from bundle engine");
    }

    // Deserialize optional audio encoder engine
    if (sections.audio_encoder_plan_data != nullptr && !sections.audio_encoder_plan_data->empty())
    {
        std::cerr << "[trtf] Deserializing Omni audio encoder TRT engine ("
                  << sections.audio_encoder_plan_data->size() / (1024 * 1024)
                  << " MB) ..." << std::endl;

        auto audio_trt = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime_ptr->deserializeCudaEngine(
                sections.audio_encoder_plan_data->data(),
                sections.audio_encoder_plan_data->size()));
        if (audio_trt)
        {
            auto audio_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
                audio_trt->createExecutionContext());
            omni_backend->set_audio_encoder(std::move(audio_trt), std::move(audio_ctx));
        }
    }

    // Deserialize optional Talker decoder engine
    if (sections.talker_engine_plan_data != nullptr && !sections.talker_engine_plan_data->empty())
    {
        std::cerr << "[trtf] Deserializing Omni Talker TRT engine ("
                  << sections.talker_engine_plan_data->size() / (1024 * 1024)
                  << " MB) ..." << std::endl;

        auto talker_trt = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime_ptr->deserializeCudaEngine(
                sections.talker_engine_plan_data->data(),
                sections.talker_engine_plan_data->size()));
        if (talker_trt)
        {
            auto talker_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
                talker_trt->createExecutionContext());

            // Build Talker decoder config from omni config
            trtf::FastPathModelConfig talker_cfg;
            talker_cfg.hidden_size = fp_cfg.omni_talker_hidden_size;
            talker_cfg.num_layers = fp_cfg.omni_talker_num_layers;
            talker_cfg.num_heads = std::max(fp_cfg.omni_talker_hidden_size / 64, 1);
            talker_cfg.num_kv_heads = talker_cfg.num_heads;
            talker_cfg.head_dim = talker_cfg.hidden_size / std::max(talker_cfg.num_heads, 1);
            talker_cfg.attention_size = talker_cfg.num_heads * talker_cfg.head_dim;
            talker_cfg.max_cache_length = fp_cfg.omni_talker_max_cache_length;
            talker_cfg.vocab_size = fp_cfg.omni_codebook_size * fp_cfg.omni_n_codebooks;

            auto talker_engine = trtf::make_decoder_engine(
                std::move(talker_trt), std::move(talker_ctx), talker_cfg);
            omni_backend->set_talker_engine(std::move(talker_engine));
        }
    }

    // Deserialize optional Code2Wav engine
    if (sections.code2wav_engine_plan_data != nullptr && !sections.code2wav_engine_plan_data->empty())
    {
        std::cerr << "[trtf] Deserializing Omni Code2Wav TRT engine ("
                  << sections.code2wav_engine_plan_data->size() / (1024 * 1024)
                  << " MB) ..." << std::endl;

        auto code2wav_trt = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime_ptr->deserializeCudaEngine(
                sections.code2wav_engine_plan_data->data(),
                sections.code2wav_engine_plan_data->size()));
        if (code2wav_trt)
        {
            auto code2wav_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
                code2wav_trt->createExecutionContext());
            omni_backend->set_code2wav_engine(std::move(code2wav_trt), std::move(code2wav_ctx));
        }
    }

    // Tokenizer
    trtf::TokenizerResult tok = {nullptr, ""};
    try
    {
        tok = trtf::extract_tokenizer_from_bundle(sections, hf_python);
    }
    catch (const std::exception& e)
    {
        std::cerr << "[trtf] Warning: no tokenizer for Omni (" << e.what() << ")" << std::endl;
    }

    std::cerr << "[trtf] Runtime ready (backend=trt_omni, strategy=omni_multimodal)" << std::endl;

    auto* pipeline = new PipelineImpl(
        model_id, std::move(tok.tokenizer), nullptr, "trt_omni", trtf::GenerationConfig{});
    if (!tok.temp_dir.empty())
        pipeline->set_bundle_temp_dir(std::move(tok.temp_dir));
    pipeline->set_omni_backend(std::move(omni_backend));
    return pipeline;
}

PipelineImpl* create_hybrid_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    auto hybrid_engine = std::make_unique<trtf::HybridStepEngine>();
    hybrid_engine->engine = std::move(trt_engine);
    hybrid_engine->context = std::move(exec_ctx);
    hybrid_engine->vocab_size = fp_cfg.vocab_size;
    hybrid_engine->hidden_size = fp_cfg.hidden_size;
    hybrid_engine->attention_size = fp_cfg.attention_size;
    hybrid_engine->max_cache_length = fp_cfg.max_cache_length;
    hybrid_engine->d_inner = fp_cfg.d_inner;
    hybrid_engine->d_state = fp_cfg.mamba_d_state;
    hybrid_engine->d_conv = fp_cfg.mamba_d_conv;
    hybrid_engine->nheads = fp_cfg.mamba_nheads;
    hybrid_engine->head_dim = fp_cfg.mamba_head_dim;
    hybrid_engine->conv_dim = fp_cfg.conv_dim;
    hybrid_engine->num_mamba_layers = fp_cfg.num_mamba_layers;
    hybrid_engine->num_attention_layers = fp_cfg.num_attention_layers;
    hybrid_engine->layer_types = fp_cfg.layer_types;
    hybrid_engine->id_bos = fp_cfg.id_bos;
    hybrid_engine->id_eos = fp_cfg.id_eos;

    for (int32_t i = 0; i < fp_cfg.num_mamba_layers; ++i)
    {
        hybrid_engine->conv_state_input_names.push_back(trtf::layer_tensor_name("conv_state", i));
        hybrid_engine->ssm_state_input_names.push_back(trtf::layer_tensor_name("ssm_state", i));
        hybrid_engine->present_conv_output_names.push_back(trtf::layer_tensor_name("present_conv", i));
        hybrid_engine->present_ssm_output_names.push_back(trtf::layer_tensor_name("present_ssm", i));
    }

    for (int32_t i = 0; i < fp_cfg.num_attention_layers; ++i)
    {
        hybrid_engine->cache_k_input_names.push_back(trtf::layer_tensor_name("cache_k", i));
        hybrid_engine->cache_v_input_names.push_back(trtf::layer_tensor_name("cache_v", i));
        hybrid_engine->present_k_output_names.push_back(trtf::layer_tensor_name("present_k", i));
        hybrid_engine->present_v_output_names.push_back(trtf::layer_tensor_name("present_v", i));
    }

    if (!trtf::has_all_required_hybrid_tensors(*hybrid_engine))
    {
        throw std::runtime_error("Bundle engine missing required hybrid tensors: " + bundle_path);
    }

    auto tok = trtf::extract_tokenizer_from_bundle(sections, hf_python);
    auto backend = trtf::CreateHybridBackendFromEngine(std::move(hybrid_engine));
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create hybrid TRT backend from bundle engine");
    }

    std::cerr << "[trtf] Runtime ready (backend=trt_hybrid, strategy=hybrid_mamba_attention)" << std::endl;
    return make_pipeline(model_id, std::move(tok), std::move(backend), "trt_hybrid");
}

PipelineImpl* create_decoder_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    auto engine_struct = trtf::make_decoder_engine(std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!trtf::has_all_required_tensors(*engine_struct))
    {
        throw std::runtime_error("Bundle engine missing required tensors: " + bundle_path);
    }

    auto tok = trtf::extract_tokenizer_from_bundle(sections, hf_python);

    const auto& strategy = fp_cfg.runtime_strategy;
    if (strategy != "decoder_kv_cache" && strategy != "decoder_moe")
    {
        throw std::runtime_error("Unsupported runtime_strategy: " + strategy
            + " (supported: decoder_kv_cache, decoder_moe, ssm_recurrent, rwkv_recurrent, speech_to_text, vision_language)");
    }

    auto backend = trtf::CreateTrtBackendFromEngine(std::move(engine_struct));
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create TRT backend from bundle engine");
    }

    std::cerr << "[trtf] Runtime ready (backend=trt, strategy=" << strategy << ")" << std::endl;
    return make_pipeline(model_id, std::move(tok), std::move(backend), "trt");
}

PipelineImpl* create_diffusion_pipeline(
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    std::cerr << "[trtf] Creating diffusion pipeline ..." << std::endl;

    // Deserialize text encoder engines
    std::vector<trtf::DiffusionEngine> text_encoders;
    for (std::size_t i = 0; i < sections.text_encoder_plans.size(); ++i)
    {
        const auto* plan = sections.text_encoder_plans[i];
        if (plan == nullptr || plan->empty()) continue;

        std::cerr << "[trtf] Deserializing text encoder " << i
                  << " (" << plan->size() / (1024 * 1024) << " MB) ..." << std::endl;

        trtf::DiffusionEngine te;
        te.name = "text_encoder_" + std::to_string(i);
        te.engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime_ptr->deserializeCudaEngine(plan->data(), plan->size()));
        if (!te.engine)
            throw std::runtime_error("Failed to deserialize text encoder " + std::to_string(i));
        te.context = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
            te.engine->createExecutionContext());
        if (!te.context)
            throw std::runtime_error("Failed to create text encoder context " + std::to_string(i));
        text_encoders.push_back(std::move(te));
    }

    // Deserialize denoiser engine
    if (sections.denoiser_plan_data == nullptr || sections.denoiser_plan_data->empty())
        throw std::runtime_error("Bundle has no denoiser_plan section: " + bundle_path);

    std::cerr << "[trtf] Deserializing denoiser ("
              << sections.denoiser_plan_data->size() / (1024 * 1024) << " MB) ..." << std::endl;

    trtf::DiffusionEngine denoiser;
    denoiser.name = "denoiser";
    denoiser.engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime_ptr->deserializeCudaEngine(
            sections.denoiser_plan_data->data(), sections.denoiser_plan_data->size()));
    if (!denoiser.engine)
        throw std::runtime_error("Failed to deserialize denoiser engine");
    denoiser.context = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
        denoiser.engine->createExecutionContext());

    // Deserialize VAE decoder engine
    if (sections.vae_decoder_plan_data == nullptr || sections.vae_decoder_plan_data->empty())
        throw std::runtime_error("Bundle has no vae_decoder_plan section: " + bundle_path);

    std::cerr << "[trtf] Deserializing VAE decoder ("
              << sections.vae_decoder_plan_data->size() / (1024 * 1024) << " MB) ..." << std::endl;

    trtf::DiffusionEngine vae_decoder;
    vae_decoder.name = "vae_decoder";
    vae_decoder.engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime_ptr->deserializeCudaEngine(
            sections.vae_decoder_plan_data->data(), sections.vae_decoder_plan_data->size()));
    if (!vae_decoder.engine)
        throw std::runtime_error("Failed to deserialize VAE decoder engine");
    vae_decoder.context = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
        vae_decoder.engine->createExecutionContext());

    // Create backend
    auto backend = trtf::CreateDiffusionBackend(
        std::move(text_encoders), std::move(denoiser), std::move(vae_decoder), fp_cfg);
    if (!backend || !backend->is_available())
        throw std::runtime_error("Failed to create diffusion backend");

    // Load preprocessor weights (patch embedding, timestep MLP, text projection)
    if (sections.preprocessor_weights_data != nullptr &&
        !sections.preprocessor_weights_data->empty())
    {
        auto pp_weights = trtf::parse_preprocessor_weights(*sections.preprocessor_weights_data);
        backend->set_preprocessor_weights(std::move(pp_weights));

        // Z-Image uses a custom preprocessor weight format
        auto* z_image = dynamic_cast<trtf::ZImageDiffusionBackend*>(backend.get());
        if (z_image != nullptr) {
            z_image->load_z_image_preprocessor_weights(*sections.preprocessor_weights_data);
        }
    }

    // Set paths for VAE subprocess and bundle info
    backend->set_hf_python(hf_python);
    backend->set_bundle_path(bundle_path);

    // CLIP tokenizer (for dual-tokenizer models like FLUX)
    if (sections.clip_vocab_json_data != nullptr && !sections.clip_vocab_json_data->empty())
    {
        try
        {
            auto clip_tok = trtf::extract_clip_tokenizer_from_bundle(sections, hf_python);
            backend->set_clip_tokenizer(std::move(clip_tok.tokenizer));
            std::cerr << "[trtf] CLIP tokenizer loaded for dual-tokenizer model" << std::endl;
        }
        catch (const std::exception& e)
        {
            std::cerr << "[trtf] Warning: CLIP tokenizer extraction failed (" << e.what() << ")" << std::endl;
        }
    }

    // Tokenizer (optional for diffusion — some models use sentencepiece)
    trtf::TokenizerResult tok = {nullptr, ""};
    try
    {
        tok = trtf::extract_tokenizer_from_bundle(sections, hf_python, /*add_special_tokens=*/true);
    }
    catch (const std::exception& e)
    {
        std::cerr << "[trtf] Warning: no tokenizer in bundle (" << e.what() << ")" << std::endl;
    }

    std::cerr << "[trtf] Runtime ready (backend=trt_diffusion, strategy=diffusion)" << std::endl;

    if (tok.tokenizer)
    {
        return make_pipeline(model_id, std::move(tok), std::move(backend), "trt_diffusion");
    }

    // No tokenizer — create pipeline with a null tokenizer
    auto* pipeline = new PipelineImpl(
        model_id, nullptr, std::move(backend), "trt_diffusion", trtf::GenerationConfig{});
    return pipeline;
}

PipelineImpl* create_speech_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    // Build temporal decoder engine (primary engine, standard KV cache decoder)
    auto temporal_engine = trtf::make_decoder_engine(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!trtf::has_all_required_tensors(*temporal_engine))
    {
        throw std::runtime_error(
            "Bundle engine missing required temporal tensors: " + bundle_path);
    }

    // Create SpeechToSpeechBackend from temporal engine + config
    // Pass hf_python for Python-based Mimi codec bridge
    auto speech_backend = trtf::CreateSpeechBackend(
        std::move(temporal_engine), fp_cfg, hf_python);
    if (!speech_backend || !speech_backend->is_available())
    {
        throw std::runtime_error(
            "Failed to create speech backend from bundle engine");
    }

    // Load depth projection matrix (if present)
    if (sections.depth_projection_data != nullptr &&
        !sections.depth_projection_data->empty())
    {
        auto& proj_data = *sections.depth_projection_data;
        auto num_floats = proj_data.size() / sizeof(float);
        auto& cfg_ref = const_cast<trtf::SpeechConfig&>(speech_backend->config());
        cfg_ref.depth_projection.resize(num_floats);
        std::memcpy(cfg_ref.depth_projection.data(), proj_data.data(),
                    proj_data.size());
        cfg_ref.temporal_hidden_for_proj = fp_cfg.hidden_size;
        std::cerr << "[trtf] Loaded depth_projection: "
                  << num_floats << " floats" << std::endl;
    }

    // Load per-codebook audio embedding tables (if present)
    // Layout: [num_codebooks, audio_vocab, temporal_hidden] as float32
    if (sections.audio_embeddings_data != nullptr &&
        !sections.audio_embeddings_data->empty())
    {
        auto& emb_data = *sections.audio_embeddings_data;
        auto num_floats = emb_data.size() / sizeof(float);
        auto& cfg_ref = const_cast<trtf::SpeechConfig&>(speech_backend->config());
        cfg_ref.audio_embeddings.resize(num_floats);
        std::memcpy(cfg_ref.audio_embeddings.data(), emb_data.data(),
                    emb_data.size());
        // Infer audio_vocab_size: total_floats / (num_codebooks * hidden_size)
        if (fp_cfg.speech_num_codebooks > 0 && fp_cfg.hidden_size > 0)
        {
            cfg_ref.audio_vocab_size = static_cast<int32_t>(
                num_floats / (static_cast<std::size_t>(fp_cfg.speech_num_codebooks)
                              * fp_cfg.hidden_size));
        }
        std::cerr << "[trtf] Loaded audio_embeddings: "
                  << num_floats << " floats ("
                  << fp_cfg.speech_num_codebooks << " codebooks x "
                  << cfg_ref.audio_vocab_size << " vocab x "
                  << fp_cfg.hidden_size << " hidden)"
                  << std::endl;
    }

    // Load temporal text embedding table (text_emb.weight) for temporal input.
    // The official Moshi code adds text_emb(text_token) at every temporal step.
    if (sections.temporal_text_embedding_data != nullptr &&
        !sections.temporal_text_embedding_data->empty())
    {
        auto& tte_data = *sections.temporal_text_embedding_data;
        auto num_floats = tte_data.size() / sizeof(float);
        auto& cfg_ref = const_cast<trtf::SpeechConfig&>(speech_backend->config());
        cfg_ref.temporal_text_embedding.resize(num_floats);
        std::memcpy(cfg_ref.temporal_text_embedding.data(), tte_data.data(),
                    tte_data.size());
        // Infer text vocab: total_floats / temporal_hidden_size
        if (fp_cfg.hidden_size > 0)
        {
            cfg_ref.temporal_text_vocab = static_cast<int32_t>(
                num_floats / static_cast<std::size_t>(fp_cfg.hidden_size));
        }
        std::cerr << "[trtf] Loaded temporal_text_embedding: "
                  << num_floats << " floats ("
                  << cfg_ref.temporal_text_vocab << " vocab x "
                  << fp_cfg.hidden_size << " hidden)"
                  << std::endl;
    }

    // Load depth text embedding table (depformer_text_emb) for depth position 0
    if (sections.depth_text_embedding_data != nullptr &&
        !sections.depth_text_embedding_data->empty())
    {
        auto& te_data = *sections.depth_text_embedding_data;
        auto num_floats = te_data.size() / sizeof(float);
        auto& cfg_ref = const_cast<trtf::SpeechConfig&>(speech_backend->config());
        cfg_ref.depth_text_embedding.resize(num_floats);
        std::memcpy(cfg_ref.depth_text_embedding.data(), te_data.data(),
                    te_data.size());
        // Infer text vocab: total_floats / depth_hidden_size
        if (fp_cfg.speech_depth_hidden_size > 0)
        {
            cfg_ref.depth_text_vocab = static_cast<int32_t>(
                num_floats / static_cast<std::size_t>(fp_cfg.speech_depth_hidden_size));
        }
        std::cerr << "[trtf] Loaded depth_text_embedding: "
                  << num_floats << " floats ("
                  << cfg_ref.depth_text_vocab << " vocab x "
                  << fp_cfg.speech_depth_hidden_size << " hidden)"
                  << std::endl;
    }

    // Load depth per-codebook audio embedding tables (depformer_emb.{0-N})
    if (sections.depth_audio_embeddings_data != nullptr &&
        !sections.depth_audio_embeddings_data->empty())
    {
        auto& dae_data = *sections.depth_audio_embeddings_data;
        auto num_floats = dae_data.size() / sizeof(float);
        auto& cfg_ref = const_cast<trtf::SpeechConfig&>(speech_backend->config());
        cfg_ref.depth_audio_embeddings.resize(num_floats);
        std::memcpy(cfg_ref.depth_audio_embeddings.data(), dae_data.data(),
                    dae_data.size());
        // Infer num_depformer_emb: total / (audio_vocab * depth_hidden)
        int32_t audio_vocab = cfg_ref.audio_vocab_size;
        int32_t d_hidden = fp_cfg.speech_depth_hidden_size;
        if (audio_vocab > 0 && d_hidden > 0)
        {
            cfg_ref.num_depformer_emb = static_cast<int32_t>(
                num_floats / (static_cast<std::size_t>(audio_vocab) * d_hidden));
        }
        std::cerr << "[trtf] Loaded depth_audio_embeddings: "
                  << num_floats << " floats ("
                  << cfg_ref.num_depformer_emb << " codebooks x "
                  << audio_vocab << " vocab x "
                  << d_hidden << " hidden)"
                  << std::endl;
    }

    // Build depth-specific config (shared across all codebook engines)
    trtf::FastPathModelConfig depth_cfg = fp_cfg;
    if (fp_cfg.speech_depth_hidden_size > 0)
        depth_cfg.hidden_size = fp_cfg.speech_depth_hidden_size;
    if (fp_cfg.speech_depth_num_layers > 0)
        depth_cfg.num_layers = fp_cfg.speech_depth_num_layers;
    if (fp_cfg.speech_depth_num_heads > 0)
    {
        depth_cfg.num_heads = fp_cfg.speech_depth_num_heads;
        depth_cfg.num_kv_heads = fp_cfg.speech_depth_num_kv_heads > 0
            ? fp_cfg.speech_depth_num_kv_heads : fp_cfg.speech_depth_num_heads;
    }
    depth_cfg.vocab_size = fp_cfg.speech_codebook_size;
    depth_cfg.head_dim = depth_cfg.hidden_size / std::max(depth_cfg.num_heads, 1);
    depth_cfg.attention_size = depth_cfg.num_heads * depth_cfg.head_dim;
    depth_cfg.max_cache_length = fp_cfg.speech_num_codebooks + 2;

    // Deserialize per-codebook depth engines (depth_engine_plan_0, _1, ...)
    if (!sections.depth_engine_plans.empty())
    {
        std::cerr << "[trtf] Deserializing " << sections.depth_engine_plans.size()
                  << " per-codebook depth engines ..." << std::endl;
        for (std::size_t cb = 0; cb < sections.depth_engine_plans.size(); ++cb)
        {
            auto* plan = sections.depth_engine_plans[cb];
            if (plan == nullptr || plan->empty()) continue;

            std::cerr << "[trtf] Deserializing depth engine cb=" << cb
                      << " (" << plan->size() / (1024 * 1024) << " MB) ..."
                      << std::endl;

            auto depth_trt = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
                runtime_ptr->deserializeCudaEngine(plan->data(), plan->size()));
            if (depth_trt)
            {
                auto depth_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
                    depth_trt->createExecutionContext());
                auto depth_engine = trtf::make_decoder_engine(
                    std::move(depth_trt), std::move(depth_ctx), depth_cfg);
                speech_backend->set_depth_engine(
                    static_cast<int32_t>(cb), std::move(depth_engine));
            }
        }
    }
    // Fallback: single depth engine (backward compatibility)
    else if (sections.depth_engine_plan_data != nullptr &&
             !sections.depth_engine_plan_data->empty())
    {
        std::cerr << "[trtf] Deserializing depth TRT engine ("
                  << sections.depth_engine_plan_data->size() / (1024 * 1024)
                  << " MB) ..." << std::endl;

        auto depth_trt = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime_ptr->deserializeCudaEngine(
                sections.depth_engine_plan_data->data(),
                sections.depth_engine_plan_data->size()));
        if (depth_trt)
        {
            auto depth_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
                depth_trt->createExecutionContext());
            auto depth_engine = trtf::make_decoder_engine(
                std::move(depth_trt), std::move(depth_ctx), depth_cfg);
            speech_backend->set_depth_engine(std::move(depth_engine));
        }
    }

    // Deserialize Mimi encoder engine (if present)
    if (sections.mimi_encoder_plan_data != nullptr &&
        !sections.mimi_encoder_plan_data->empty())
    {
        std::cerr << "[trtf] Deserializing Mimi encoder TRT engine ("
                  << sections.mimi_encoder_plan_data->size() / (1024 * 1024)
                  << " MB) ..." << std::endl;

        auto enc_trt = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime_ptr->deserializeCudaEngine(
                sections.mimi_encoder_plan_data->data(),
                sections.mimi_encoder_plan_data->size()));
        if (enc_trt)
        {
            auto enc_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
                enc_trt->createExecutionContext());
            speech_backend->set_mimi_encoder(
                std::move(enc_trt), std::move(enc_ctx));
        }
    }

    // Deserialize Mimi decoder engine (if present)
    if (sections.mimi_decoder_plan_data != nullptr &&
        !sections.mimi_decoder_plan_data->empty())
    {
        std::cerr << "[trtf] Deserializing Mimi decoder TRT engine ("
                  << sections.mimi_decoder_plan_data->size() / (1024 * 1024)
                  << " MB) ..." << std::endl;

        auto dec_trt = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime_ptr->deserializeCudaEngine(
                sections.mimi_decoder_plan_data->data(),
                sections.mimi_decoder_plan_data->size()));
        if (dec_trt)
        {
            auto dec_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
                dec_trt->createExecutionContext());
            speech_backend->set_mimi_decoder(
                std::move(dec_trt), std::move(dec_ctx));
        }
    }

    std::cerr << "[trtf] Runtime ready (backend=trt_speech, strategy=speech_to_speech)"
              << std::endl;

    // Speech pipelines don't need a tokenizer or generation backend
    auto* pipeline = new PipelineImpl(
        model_id, nullptr, nullptr, "trt_speech", trtf::GenerationConfig{});
    pipeline->set_speech_backend(std::move(speech_backend));
    return pipeline;
}

// --- Main dispatch ---

PipelineImpl* try_create_from_bundle(const std::string& bundle_path, const std::string& hf_python)
{
    trtf::BundleFile bundle = trtf::ReadBundleFile(bundle_path);
    auto sections = trtf::find_bundle_sections(bundle);

    // Check for diffusion bundle early (no engine_plan needed)
    // Parse config to detect strategy before engine deserialization
    trtf::FastPathModelConfig fp_cfg_early;
    if (sections.config_json_data != nullptr && !sections.config_json_data->empty())
    {
        const std::string config_text_early(
            sections.config_json_data->begin(), sections.config_json_data->end());
        fp_cfg_early = trtf::parse_fast_path_config(config_text_early, bundle.info.max_cache_length);
    }

    if (fp_cfg_early.runtime_strategy == "diffusion")
    {
        trtf::TrtLogger logger;
        auto runtime_ptr = trtf::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
        if (!runtime_ptr) throw std::runtime_error("Failed to create TRT runtime");
        return create_diffusion_pipeline(
            fp_cfg_early, sections, runtime_ptr,
            bundle.info.model_id, hf_python, bundle_path);
    }

    if (sections.plan_data == nullptr || sections.plan_data->empty())
    {
        throw std::runtime_error("Bundle has no engine_plan section: " + bundle_path);
    }

    // Deserialize TRT engine
    std::cerr << "[trtf] Deserializing TRT engine from bundle ("
              << sections.plan_data->size() / (1024 * 1024) << " MB) ..." << std::endl;

    trtf::TrtLogger logger;
    auto runtime_ptr = trtf::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!runtime_ptr)
    {
        throw std::runtime_error("Failed to create TRT runtime");
    }

    auto tdeser0 = std::chrono::steady_clock::now();
    auto trt_engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime_ptr->deserializeCudaEngine(sections.plan_data->data(), sections.plan_data->size()));
    if (!trt_engine)
    {
        throw std::runtime_error("Failed to deserialize engine from bundle: " + bundle_path);
    }
    auto tdeser1 = std::chrono::steady_clock::now();
    std::cerr << "[trtf] Engine deserialized ["
              << std::chrono::duration_cast<std::chrono::milliseconds>(tdeser1 - tdeser0).count()
              << " ms]" << std::endl;

    auto exec_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(trt_engine->createExecutionContext());
    if (!exec_ctx)
    {
        throw std::runtime_error("Failed to create execution context from bundle engine");
    }

    // Parse config.json for model metadata
    trtf::FastPathModelConfig fp_cfg;
    if (sections.config_json_data != nullptr && !sections.config_json_data->empty())
    {
        const std::string config_text(sections.config_json_data->begin(), sections.config_json_data->end());
        fp_cfg = trtf::parse_fast_path_config(config_text, bundle.info.max_cache_length);
    }
    else
    {
        fp_cfg.vocab_size = bundle.info.vocab_size;
        fp_cfg.hidden_size = bundle.info.hidden_size;
        fp_cfg.num_layers = bundle.info.num_layers;
        fp_cfg.num_heads = bundle.info.num_attention_heads;
        fp_cfg.num_kv_heads = bundle.info.num_key_value_heads;
        fp_cfg.head_dim = fp_cfg.hidden_size / std::max(fp_cfg.num_heads, 1);
        fp_cfg.attention_size = fp_cfg.num_heads * fp_cfg.head_dim;
        fp_cfg.max_cache_length = bundle.info.max_cache_length;
    }

    // Dispatch to per-strategy factory
    const auto& strategy = fp_cfg.runtime_strategy;

    if (strategy == "ssm_recurrent")
    {
        return create_mamba_pipeline(
            std::move(trt_engine), std::move(exec_ctx), fp_cfg,
            sections, bundle.info.model_id, hf_python, bundle_path);
    }

    if (strategy == "rwkv_recurrent")
    {
        return create_rwkv_pipeline(
            std::move(trt_engine), std::move(exec_ctx), fp_cfg,
            sections, bundle.info.model_id, hf_python, bundle_path);
    }

    if (strategy == "speech_to_text")
    {
        return create_whisper_pipeline(
            std::move(trt_engine), std::move(exec_ctx), fp_cfg,
            sections, runtime_ptr, bundle.info.model_id, hf_python, bundle_path);
    }

    if (strategy == "vision_language")
    {
        return create_vl_pipeline(
            std::move(trt_engine), std::move(exec_ctx), fp_cfg,
            sections, runtime_ptr, bundle.info.model_id, hf_python, bundle_path);
    }

    if (strategy == "segmentation")
    {
        return create_segmentation_pipeline(
            std::move(trt_engine), std::move(exec_ctx), fp_cfg,
            bundle.info.model_id, bundle_path);
    }

    if (strategy == "object_detection")
    {
        return create_detection_pipeline(
            std::move(trt_engine), std::move(exec_ctx), fp_cfg,
            bundle.info.model_id, bundle_path);
    }

    if (strategy == "prompted_segmentation")
    {
        return create_sam_pipeline(
            std::move(trt_engine), std::move(exec_ctx), fp_cfg,
            sections, runtime_ptr, bundle.info.model_id, bundle_path);
    }

    if (strategy == "neural_operator")
    {
        return create_neural_operator_pipeline(
            std::move(trt_engine), std::move(exec_ctx), fp_cfg,
            bundle.info.model_id, bundle_path);
    }

    if (strategy == "encoder_only")
    {
        return create_encoder_pipeline(
            std::move(trt_engine), std::move(exec_ctx), fp_cfg,
            sections, bundle.info.model_id, hf_python, bundle_path);
    }

    if (strategy == "embedding")
    {
        return create_embedding_pipeline(
            std::move(trt_engine), std::move(exec_ctx), fp_cfg,
            sections, bundle.info.model_id, hf_python, bundle_path);
    }

    if (strategy == "reranking")
    {
        return create_reranking_pipeline(
            std::move(trt_engine), std::move(exec_ctx), fp_cfg,
            sections, bundle.info.model_id, hf_python, bundle_path);
    }

    if (strategy == "text_to_audio")
    {
        return create_bark_pipeline(
            std::move(trt_engine), std::move(exec_ctx), fp_cfg,
            sections, runtime_ptr, bundle.info.model_id, hf_python, bundle_path);
    }

    if (strategy == "omni_multimodal")
    {
        return create_omni_pipeline(
            std::move(trt_engine), std::move(exec_ctx), fp_cfg,
            sections, runtime_ptr, bundle.info.model_id, hf_python, bundle_path);
    }

    if (strategy == "speech_to_speech")
    {
        return create_speech_pipeline(
            std::move(trt_engine), std::move(exec_ctx), fp_cfg,
            sections, runtime_ptr, bundle.info.model_id, hf_python, bundle_path);
    }

    if (strategy == "diffusion")
    {
        return create_diffusion_pipeline(
            fp_cfg, sections, runtime_ptr,
            bundle.info.model_id, hf_python, bundle_path);
    }

    if (strategy == "hybrid_mamba_attention")
    {
        return create_hybrid_pipeline(
            std::move(trt_engine), std::move(exec_ctx), fp_cfg,
            sections, bundle.info.model_id, hf_python, bundle_path);
    }

    return create_decoder_pipeline(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg,
        sections, bundle.info.model_id, hf_python, bundle_path);
}
#endif

} // namespace

extern "C" {

trtf::IPipeline* trtf_create_pipeline(const char* bundle_path, int flags)
{
    (void) flags;
    TrtfPipelineOptions opts{};
    opts.max_new_tokens = 0;
    opts.hf_python = nullptr;
    return trtf_create_pipeline_ex(bundle_path, &opts);
}

trtf::IPipeline* trtf_create_pipeline_ex(const char* bundle_path, const TrtfPipelineOptions* options)
{
    clear_last_error();

    if (bundle_path == nullptr || bundle_path[0] == '\0')
    {
        set_last_error("bundle_path must not be null or empty");
        return nullptr;
    }

    (void)(options ? options->max_new_tokens : 0); // reserved for future use
    const std::string hf_python = (options && options->hf_python) ? options->hf_python : "";

    try
    {
        const std::string input(bundle_path);

        if (!trtf::IsBundle(input))
        {
            set_last_error("Not a valid .trtfb bundle: " + input);
            return nullptr;
        }

#if TRTF_HAS_TRT
        auto t0 = std::chrono::steady_clock::now();
        auto* pipeline = try_create_from_bundle(input, hf_python);
        if (pipeline != nullptr)
        {
            auto t1 = std::chrono::steady_clock::now();
            std::cerr << "[trtf] Total startup ["
                      << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                      << " ms]" << std::endl;
            return pipeline;
        }
        set_last_error("Failed to load bundle: " + input);
        return nullptr;
#else
        set_last_error("Bundle loading requires TRT support (compile with TRT)");
        return nullptr;
#endif
    }
    catch (const std::exception& e)
    {
        set_last_error(e.what());
        return nullptr;
    }
    catch (...)
    {
        set_last_error("Unknown error creating pipeline");
        return nullptr;
    }
}

const char* trtf_last_error(void)
{
    return g_last_error.c_str();
}

const char* trtf_version(void)
{
    return TRTF_VERSION_STRING;
}

int trtf_has_trt(void)
{
#if TRTF_HAS_TRT
    return 1;
#else
    return 0;
#endif
}

} // extern "C"
