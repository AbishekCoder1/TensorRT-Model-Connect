#include "cabi/pipeline/pipeline_impl.h"

#include "cabi/factories/factory_decls.h"
#include "cabi/factories/factories_audio.h"
#include "cabi/factories/factories_diffusion.h"
#include "cabi/factories/factories_encoder.h"
#include "cabi/factories/factories_multimodal.h"
#include "cabi/factories/factories_text.h"
#include "cabi/factories/factories_vision.h"
#include "runtime/trt/audio/bark_backend.h"
#include "runtime/trt/perception/detection_backend.h"
#include "runtime/trt/diffusion/diffusion_backend.h"
#include "runtime/trt/encoder/embedding_backend.h"
#include "runtime/trt/encoder/encoder_backend.h"
#include "runtime/trt/multimodal/image_preprocessor.h"
#include "runtime/trt/audio/magpie_tts_backend.h"
#include "runtime/trt/audio/mel_spectrogram.h"
#include "runtime/trt/perception/neural_operator_backend.h"
#include "runtime/trt/audio/omni_backend.h"
#include "runtime/trt/encoder/reranking_backend.h"
#include "runtime/trt/perception/sam_backend.h"
#include "runtime/trt/perception/segmentation_backend.h"
#include "runtime/trt/audio/speech_backend.h"
#include "runtime/trt/core/trt_backend_shared.h"
#include "runtime/trt/core/trt_engine_lifecycle.h"
#include "runtime/trt/multimodal/vl_backend.h"
#include "runtime/trt/multimodal/vision_engine.h"
#include "runtime/trt/audio/whisper_backend.h"
#include "stb_image_write.h"
#include "trtf/backend.h"
#include "trtf/tokenizer.h"
#include "utils/wav_reader.h"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

namespace {

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
            return clear_and_return_empty_output();

        const auto config = resolve_generation_config(max_new_tokens);

#if TRTF_HAS_TRT
        // Omni backend: use Thinker decoder for text generation
        if (can_use_omni_text_generation())
        {
            return run_omni_text_generation(prompt, config);
        }
#endif

        if (!can_run_text_generation())
            return clear_and_return_empty_output();
        return run_text_generation(prompt, config);
    }

    const char* generate(const char* prompt, const char* image_path,
                         std::size_t max_new_tokens) override
    {
        // Fall back to text-only generation when vision path is unavailable.
        if (!can_run_vision_generation(image_path))
        {
            return generate(prompt, max_new_tokens);
        }

        if (prompt == nullptr)
            return clear_and_return_empty_output();

        const auto config = resolve_generation_config(max_new_tokens);

#if TRTF_HAS_TRT
        // Native VL pipeline: preprocess image in C++, run vision TRT, tokenize, generate
        auto* vl = dynamic_cast<trtf::VLBackendFastPath*>(mBackend.get());
        if (vl != nullptr)
        {
            return run_vl_generation(vl, prompt, image_path, config);
        }
#endif // TRTF_HAS_TRT

        return run_text_generation(prompt, config);
    }

    bool supports_vision() const override
    {
        return mBackend != nullptr && mBackend->supports_vision();
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
        auto* diff = get_diffusion_backend();
        if (diff == nullptr)
        {
            return -1;
        }

        auto input_ids = prepare_video_input_ids(diff, prompt);

        // Generate video
        auto video = diff->generate_video(input_ids, num_steps, guidance_scale);
        if (video.frames.empty() || video.num_frames <= 0)
        {
            return -1;
        }
        return write_video_frames(video, output_dir);
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
        return mVision.segmentation != nullptr;
#else
        return false;
#endif
    }

    int32_t segment(const char* image_path, const char* output_path) override
    {
#if TRTF_HAS_TRT
        if (mVision.segmentation == nullptr || image_path == nullptr || output_path == nullptr)
            return -1;

        try
        {
            auto result = mVision.segmentation->segment_image(std::string(image_path));

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
        return mAudio.bark != nullptr || mMultimodal.omni != nullptr || mAudio.magpie_tts != nullptr;
#else
        return false;
#endif
    }

    int32_t generate_audio(const char* prompt, const char* output_path,
                           int32_t max_tokens) override
    {
#if TRTF_HAS_TRT
        if (prompt == nullptr || output_path == nullptr)
            return -1;

        if (mMultimodal.omni != nullptr)
            return generate_omni_audio(prompt, output_path, max_tokens);
        if (mAudio.magpie_tts != nullptr)
            return generate_magpie_audio(prompt, output_path, max_tokens);
        if (mAudio.bark == nullptr)
            return -1;
        return generate_bark_audio(prompt, output_path, max_tokens);
#else
        (void) prompt; (void) output_path; (void) max_tokens;
        return -1;
#endif
    }

    bool supports_transcription() const override
    {
#if TRTF_HAS_TRT
        return mAudio.whisper != nullptr;
#else
        return false;
#endif
    }

    const char* transcribe(const char* audio_path, int32_t max_new_tokens) override
    {
#if TRTF_HAS_TRT
        if (!can_transcribe(audio_path))
            return nullptr;

        if (!has_mel_filterbank())
        {
            std::cerr << "[trtf] Bundle missing mel_filterbank section; "
                         "rebuild with latest trtf-build" << std::endl;
            return nullptr;
        }

        try
        {
            auto wav = trtf::read_wav(std::string(audio_path));
            std::cerr << "[trtf] WAV: " << wav.samples.size()
                      << " samples @ " << wav.sample_rate << " Hz" << std::endl;

            auto mel = extract_mel_features(wav);

            std::cerr << "[trtf] Mel spectrogram: " << mel.n_mels << " bins x "
                      << mel.n_frames << " frames" << std::endl;

            auto tr = mAudio.whisper->transcribe(
                mel.data.data(), mel.n_mels, mel.n_frames,
                effective_limit(max_new_tokens, 224));

            assign_transcription_output(tr);

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
    void set_default_max_new_tokens(std::size_t max_new_tokens)
    {
        if (max_new_tokens > 0)
        {
            mGenConfig.max_new_tokens = max_new_tokens;
        }
    }

#if TRTF_HAS_TRT
    void set_seg_backend(std::unique_ptr<trtf::SegmentationBackend> backend)
    {
        mVision.segmentation = std::move(backend);
    }
    void set_bark_backend(std::unique_ptr<trtf::BarkBackend> backend)
    {
        mAudio.bark = std::move(backend);
    }
    void set_magpie_tts_backend(std::unique_ptr<trtf::MagpieTTSBackend> backend)
    {
        mAudio.magpie_tts = std::move(backend);
    }
    void set_omni_backend(std::unique_ptr<trtf::OmniBackend> backend)
    {
        mMultimodal.omni = std::move(backend);
    }
    void set_encoder_backend(std::unique_ptr<trtf::EncoderBackend> backend)
    {
        mEncoding.encoder = std::move(backend);
    }
    void set_neural_operator_backend(std::unique_ptr<trtf::NeuralOperatorBackend> backend)
    {
        mOperator.neural_op = std::move(backend);
    }
    void set_detection_backend(std::unique_ptr<trtf::DetectionBackend> backend)
    {
        mVision.detection = std::move(backend);
    }
    void set_speech_backend(std::unique_ptr<trtf::SpeechToSpeechBackend> backend)
    {
        mSpeech.speech_to_speech = std::move(backend);
    }
    void set_embedding_backend(std::unique_ptr<trtf::EmbeddingBackend> backend)
    {
        mEmbedding.embedding = std::move(backend);
    }
    void set_reranking_backend(std::unique_ptr<trtf::RerankingBackend> backend)
    {
        mEmbedding.reranker = std::move(backend);
    }
    void set_whisper_backend(std::unique_ptr<trtf::WhisperBackend> backend,
                             trtf::MelFilterbank mel_fb = {},
                             const trtf::FastPathModelConfig* fp_cfg = nullptr)
    {
        mAudio.whisper = std::move(backend);
        mAudio.mel_filterbank = std::move(mel_fb);
        if (fp_cfg)
        {
            mAudio.mel_nfft = fp_cfg->mel_n_fft;
            mAudio.mel_hop_length = fp_cfg->mel_hop_length;
            mAudio.mel_chunk_length = fp_cfg->mel_chunk_length;
            mAudio.mel_sampling_rate = fp_cfg->mel_sampling_rate;
        }
    }
    void set_sam_backend(std::unique_ptr<trtf::SamBackend> backend)
    {
        mVision.prompted_segmentation = std::move(backend);
    }
#endif

    void set_hf_python(std::string path) { mHfPython = std::move(path); }

    bool supports_embedding() const override
    {
#if TRTF_HAS_TRT
        return mEmbedding.embedding != nullptr;
#else
        return false;
#endif
    }

    const float* embed(const char* text, int32_t* out_dim) override
    {
#if TRTF_HAS_TRT
        if (mEmbedding.embedding == nullptr || text == nullptr) return nullptr;
        try {
            std::vector<int32_t> ids;
            if (mTokenizer) ids = mTokenizer->encode(text);
            mEmbedding.last_result = mEmbedding.embedding->embed(ids);
            if (out_dim) *out_dim = mEmbedding.last_result.embedding_dim;
            return mEmbedding.last_result.embedding.data();
        } catch (const std::exception& e) {
            std::cerr << "[trtf] Embed error: " << e.what() << std::endl;
            return nullptr;
        }
#else
        (void) text; (void) out_dim; return nullptr;
#endif
    }

    const float* embed_image(const char* image_path, int32_t* out_dim) override
    {
#if TRTF_HAS_TRT
        if (mEmbedding.embedding == nullptr || !mEmbedding.embedding->has_vision() ||
            image_path == nullptr) return nullptr;
        try {
            // For image-only embedding, create token sequence with image placeholders
            const auto& vl_cfg = mEmbedding.embedding->vl_config();
            std::string vl_prompt = trtf::format_vl_prompt("", vl_cfg);
            std::vector<int32_t> ids;
            if (mTokenizer) ids = mTokenizer->encode(vl_prompt.c_str());
            mEmbedding.last_result = mEmbedding.embedding->embed_with_image(ids, std::string(image_path));
            if (out_dim) *out_dim = mEmbedding.last_result.embedding_dim;
            return mEmbedding.last_result.embedding.data();
        } catch (const std::exception& e) {
            std::cerr << "[trtf] Embed image error: " << e.what() << std::endl;
            return nullptr;
        }
#else
        (void) image_path; (void) out_dim; return nullptr;
#endif
    }

    const float* embed_image_text(const char* text, const char* image_path,
                                   int32_t* out_dim) override
    {
#if TRTF_HAS_TRT
        if (mEmbedding.embedding == nullptr || !mEmbedding.embedding->has_vision() ||
            text == nullptr || image_path == nullptr) return nullptr;
        try {
            const auto& vl_cfg = mEmbedding.embedding->vl_config();
            std::string vl_prompt = trtf::format_vl_prompt(std::string(text), vl_cfg);
            std::vector<int32_t> ids;
            if (mTokenizer) ids = mTokenizer->encode(vl_prompt.c_str());
            mEmbedding.last_result = mEmbedding.embedding->embed_with_image(ids, std::string(image_path));
            if (out_dim) *out_dim = mEmbedding.last_result.embedding_dim;
            return mEmbedding.last_result.embedding.data();
        } catch (const std::exception& e) {
            std::cerr << "[trtf] Embed image+text error: " << e.what() << std::endl;
            return nullptr;
        }
#else
        (void) text; (void) image_path; (void) out_dim; return nullptr;
#endif
    }

    bool supports_reranking() const override
    {
#if TRTF_HAS_TRT
        return mEmbedding.reranker != nullptr;
#else
        return false;
#endif
    }

    float rerank(const char* query, const char* document) override
    {
#if TRTF_HAS_TRT
        if (mEmbedding.reranker == nullptr || !query || !document) return 0.0F;
        try {
            std::string combined = std::string(query) + " " + std::string(document);
            std::vector<int32_t> ids;
            if (mTokenizer) ids = mTokenizer->encode(combined.c_str());
            return mEmbedding.reranker->rerank(ids).score;
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
        return mEncoding.encoder != nullptr;
#else
        return false;
#endif
    }

    const float* encode(const char* text, int32_t* out_seq_len,
                        int32_t* out_hidden_size) override
    {
#if TRTF_HAS_TRT
        if (mEncoding.encoder == nullptr || text == nullptr)
            return nullptr;

        try
        {
            std::vector<int32_t> input_ids;
            if (mTokenizer)
            {
                input_ids = mTokenizer->encode(text);
            }

            mEncoding.last_result = mEncoding.encoder->encode(input_ids);

            if (out_seq_len != nullptr)
                *out_seq_len = mEncoding.last_result.seq_length;
            if (out_hidden_size != nullptr)
                *out_hidden_size = mEncoding.last_result.hidden_size;

            return mEncoding.last_result.hidden_states.data();
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
        return mOperator.neural_op != nullptr;
#else
        return false;
#endif
    }

    const float* solve(const float* branch_input, int32_t branch_len,
                       const float* trunk_input, int32_t trunk_len,
                       int32_t* out_dim) override
    {
#if TRTF_HAS_TRT
        if (mOperator.neural_op == nullptr || branch_input == nullptr || trunk_input == nullptr)
            return nullptr;

        try
        {
            mOperator.last_result = mOperator.neural_op->solve(
                branch_input, branch_len, trunk_input, trunk_len);

            if (out_dim != nullptr)
                *out_dim = mOperator.last_result.output_dim;

            return mOperator.last_result.output.data();
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
        if (mOperator.neural_op == nullptr || field_input == nullptr)
            return nullptr;

        try
        {
            mOperator.last_result = mOperator.neural_op->solve_field(
                field_input, input_size);

            if (out_channels != nullptr)
                *out_channels = mOperator.last_result.out_channels;
            if (out_h != nullptr)
                *out_h = mOperator.last_result.height;
            if (out_w != nullptr)
                *out_w = mOperator.last_result.width;

            return mOperator.last_result.output.data();
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
        return mVision.detection != nullptr;
#else
        return false;
#endif
    }

    int32_t detect(const char* image_path, const char* output_path,
                   float conf_threshold) override
    {
#if TRTF_HAS_TRT
        if (mVision.detection == nullptr || image_path == nullptr || output_path == nullptr)
            return -1;

        try
        {
            auto result = mVision.detection->detect_image(std::string(image_path));
            std::size_t written = 0;
            auto json = build_detection_json(result, conf_threshold, written);
            if (!write_text_file(output_path, json))
            {
                std::cerr << "[trtf] Failed to open output: " << output_path << std::endl;
                return -1;
            }

            std::cerr << "[trtf] Detection saved: " << output_path
                      << " (" << written << " objects)" << std::endl;
            return static_cast<int32_t>(written);
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
        return mSpeech.speech_to_speech != nullptr;
#else
        return false;
#endif
    }

    int32_t speak(const char* audio_in, const char* audio_out,
                  int32_t max_output_frames,
                  int32_t tail_frames) override
    {
#if TRTF_HAS_TRT
        if (!can_speak(audio_in, audio_out))
            return -1;
        try {
            auto wav = trtf::read_wav(std::string(audio_in));
            auto input = prepare_speech_input(std::move(wav.samples), wav.sample_rate);
            if (input.samples.empty())
                return -1;
            auto r = mSpeech.speech_to_speech->process_audio(
                input.samples.data(),
                static_cast<int32_t>(input.samples.size()),
                effective_limit(max_output_frames, 375),
                input.sample_rate,
                std::max(0, tail_frames));
            if (r.num_samples <= 0)
                return -1;
            trtf::write_wav(std::string(audio_out),
                r.waveform.data(), r.num_samples, r.sample_rate);
            return r.num_samples;
        } catch (const std::exception& e) {
            std::cerr << "[trtf] Speech error: " << e.what() << std::endl;
            return -1;
        }
#else
        (void) audio_in; (void) audio_out; (void) max_output_frames; (void) tail_frames;
        return -1;
#endif
    }

    bool supports_prompted_segmentation() const override
    {
#if TRTF_HAS_TRT
        return mVision.prompted_segmentation != nullptr;
#else
        return false;
#endif
    }

    int32_t segment_sam(const char* image_path, const char* output_dir,
                        float point_x, float point_y, bool is_foreground) override
    {
#if TRTF_HAS_TRT
        if (mVision.prompted_segmentation == nullptr || image_path == nullptr || output_dir == nullptr)
            return -1;

        try
        {
            // Encode image (cached on GPU)
            if (!mVision.prompted_segmentation->encode_image(std::string(image_path)))
            {
                std::cerr << "[trtf] SAM: Failed to encode image" << std::endl;
                return -1;
            }

            // Segment with point prompt
            auto result = mVision.prompted_segmentation->segment_point(point_x, point_y, is_foreground);

            // Write each mask as a separate PNG
            write_sam_masks(result, output_dir);

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
    const char* clear_and_return_empty_output()
    {
        mLastOutput.clear();
        return mLastOutput.c_str();
    }

    trtf::GenerationConfig resolve_generation_config(std::size_t max_new_tokens) const
    {
        trtf::GenerationConfig config = mGenConfig;
        if (max_new_tokens > 0)
        {
            config.max_new_tokens = max_new_tokens;
        }
        return config;
    }

    bool can_run_text_generation() const
    {
        return mBackend != nullptr && mTokenizer != nullptr;
    }

    bool can_run_vision_generation(const char* image_path) const
    {
        return mBackend != nullptr
            && mBackend->supports_vision()
            && image_path != nullptr
            && mTokenizer != nullptr;
    }

    std::vector<int32_t> encode_if_tokenizer(const char* text) const
    {
        if (mTokenizer == nullptr || text == nullptr)
        {
            return {};
        }
        return mTokenizer->encode(text);
    }

    const char* run_text_generation(const char* prompt, const trtf::GenerationConfig& config)
    {
        auto input_ids = mTokenizer->encode(prompt);
        auto output_ids = mBackend->generate(input_ids, config);
        mLastOutput = mTokenizer->decode(output_ids);
        return mLastOutput.c_str();
    }

    static int32_t effective_limit(int32_t value, int32_t fallback)
    {
        return value > 0 ? value : fallback;
    }

#if TRTF_HAS_TRT
    bool can_use_omni_text_generation() const
    {
        return mMultimodal.omni != nullptr && mTokenizer != nullptr;
    }

    const char* run_omni_text_generation(
        const char* prompt,
        const trtf::GenerationConfig& config)
    {
        auto input_ids = mTokenizer->encode(prompt);
        const int32_t max_tok = config.max_new_tokens > 0
            ? static_cast<int32_t>(config.max_new_tokens) : 128;
        auto output_ids = mMultimodal.omni->generate_text(input_ids, max_tok);
        mLastOutput = mTokenizer->decode(output_ids);
        return mLastOutput.c_str();
    }

    const char* run_vl_generation(
        trtf::VLBackendFastPath* vl,
        const char* prompt,
        const char* image_path,
        const trtf::GenerationConfig& config)
    {
        std::vector<float> image_features;
        int32_t num_features = 0;
        int32_t feature_dim = 0;
        std::string prep_error;
        if (!vl->prepare_image(
                std::string(image_path),
                image_features, num_features, feature_dim, prep_error))
        {
            std::cerr << "[trtf] VL image preparation failed: " << prep_error
                      << ", falling back to text-only" << std::endl;
            return run_text_generation(prompt, config);
        }

        const std::string formatted = trtf::format_vl_prompt(
            std::string(prompt), vl->vl_config());
        auto input_ids = mTokenizer->encode(formatted.c_str());
        auto output_ids = vl->generate_vl(
            input_ids,
            image_features.data(), num_features, feature_dim,
            {}, config);
        mLastOutput = mTokenizer->decode(output_ids);
        return mLastOutput.c_str();
    }

    trtf::IDiffusionBackend* get_diffusion_backend() const
    {
        return dynamic_cast<trtf::IDiffusionBackend*>(mBackend.get());
    }

    std::vector<int32_t> prepare_video_input_ids(
        trtf::IDiffusionBackend* diff,
        const char* prompt) const
    {
        if (prompt != nullptr)
        {
            diff->set_prompt(std::string(prompt));
        }
        if (mTokenizer == nullptr || prompt == nullptr)
        {
            return {};
        }
        const std::string prepared = diff->prepare_prompt(std::string(prompt));
        return mTokenizer->encode(prepared);
    }

    static std::vector<uint8_t> to_u8_pixels(const float* frame_f, std::size_t frame_size)
    {
        std::vector<uint8_t> pixels(frame_size);
        for (std::size_t i = 0; i < frame_size; ++i)
        {
            pixels[i] = static_cast<uint8_t>(
                std::max(0.0F, std::min(255.0F, frame_f[i] * 255.0F)));
        }
        return pixels;
    }

    int32_t write_video_frames(const trtf::VideoResult& video, const char* output_dir) const
    {
        const std::string dir(output_dir != nullptr ? output_dir : "/tmp/trtf_frames");
        std::filesystem::create_directories(dir);

        for (int32_t t = 0; t < video.num_frames; ++t)
        {
            const auto frame_size = static_cast<std::size_t>(video.height)
                                  * static_cast<std::size_t>(video.width) * 3;
            const float* frame_f = video.frames.data()
                + static_cast<std::size_t>(t) * frame_size;

            auto pixels = to_u8_pixels(frame_f, frame_size);
            char fname[64];
            std::snprintf(fname, sizeof(fname), "/frame_%04d.png", t);
            const std::string path = dir + fname;
            stbi_write_png(path.c_str(), video.width, video.height, 3,
                           pixels.data(), video.width * 3);
        }

        std::cerr << "[trtf] Wrote " << video.num_frames << " PNG frames to "
                  << dir << std::endl;
        return video.num_frames;
    }

    template <typename AudioResultT>
    int32_t write_audio_output(
        const AudioResultT& result,
        const char* output_path,
        const char* success_prefix) const
    {
        if (result.num_samples <= 0)
        {
            return -1;
        }
        trtf::write_wav(
            std::string(output_path),
            result.waveform.data(), result.num_samples, result.sample_rate);
        std::cerr << success_prefix << output_path
                  << " (" << result.num_samples << " samples @ "
                  << result.sample_rate << " Hz)" << std::endl;
        return result.num_samples;
    }

    int32_t generate_omni_audio(const char* prompt, const char* output_path, int32_t max_tokens)
    {
        try
        {
            auto input_ids = encode_if_tokenizer(prompt);
            auto result = mMultimodal.omni->generate_audio(
                input_ids, effective_limit(max_tokens, 768));
            return write_audio_output(result, output_path, "[trtf] Omni audio saved: ");
        }
        catch (const std::exception& e)
        {
            std::cerr << "[trtf] Omni audio error: " << e.what() << std::endl;
            return -1;
        }
    }

    int32_t generate_magpie_audio(const char* prompt, const char* output_path, int32_t max_tokens)
    {
        try
        {
            auto input_ids = encode_if_tokenizer(prompt);
            auto result = mAudio.magpie_tts->generate_audio(
                input_ids, effective_limit(max_tokens, 500));
            return write_audio_output(result, output_path, "[trtf] MagpieTTS audio saved: ");
        }
        catch (const std::exception& e)
        {
            std::cerr << "[trtf] MagpieTTS audio error: " << e.what() << std::endl;
            return -1;
        }
    }

    int32_t generate_bark_audio(const char* prompt, const char* output_path, int32_t max_tokens)
    {
        try
        {
            auto input_ids = encode_if_tokenizer(prompt);
            auto result = mAudio.bark->generate_audio(
                input_ids, effective_limit(max_tokens, 768));
            return write_audio_output(result, output_path, "[trtf] Audio saved: ");
        }
        catch (const std::exception& e)
        {
            std::cerr << "[trtf] Audio generation error: " << e.what() << std::endl;
            return -1;
        }
    }

    bool can_transcribe(const char* audio_path) const
    {
        return mAudio.whisper != nullptr && audio_path != nullptr;
    }

    bool has_mel_filterbank() const
    {
        return !mAudio.mel_filterbank.data.empty();
    }

    trtf::MelResult extract_mel_features(const trtf::WavData& wav) const
    {
        const float* audio_ptr = wav.samples.data();
        int32_t audio_len = static_cast<int32_t>(wav.samples.size());
        std::vector<float> resampled;
        if (wav.sample_rate != mAudio.mel_sampling_rate && wav.sample_rate > 0)
        {
            resampled = trtf::resample_linear(
                wav.samples.data(), audio_len,
                wav.sample_rate, mAudio.mel_sampling_rate);
            audio_ptr = resampled.data();
            audio_len = static_cast<int32_t>(resampled.size());
            std::cerr << "[trtf] Resampled " << wav.sample_rate
                      << " -> " << mAudio.mel_sampling_rate << " Hz ("
                      << wav.samples.size() << " -> " << audio_len
                      << " samples)" << std::endl;
        }
        return trtf::extract_mel_spectrogram(
            audio_ptr, audio_len,
            mAudio.mel_filterbank.data.data(),
            mAudio.mel_filterbank.n_freq_bins, mAudio.mel_filterbank.n_mel_bins,
            mAudio.mel_nfft, mAudio.mel_hop_length, mAudio.mel_chunk_length, mAudio.mel_sampling_rate);
    }

    void assign_transcription_output(const trtf::TranscriptionResult& tr)
    {
        if (mTokenizer != nullptr && !tr.output_ids.empty())
        {
            mLastOutput = mTokenizer->decode(tr.output_ids);
            return;
        }
        mLastOutput = tr.text;
    }

    static std::string build_detection_json(
        const trtf::DetectionResult& result,
        float conf_threshold,
        std::size_t& written)
    {
        const bool apply_threshold = conf_threshold >= 0.0F;
        std::string json = "[\n";
        written = 0;
        for (const auto& d : result.detections)
        {
            if (apply_threshold && d.confidence < conf_threshold)
            {
                continue;
            }
            if (written > 0)
            {
                json += ",\n";
            }
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "  {\"class_id\": %d, \"confidence\": %.4f, "
                "\"x1\": %.1f, \"y1\": %.1f, \"x2\": %.1f, \"y2\": %.1f}",
                d.class_id, d.confidence, d.x1, d.y1, d.x2, d.y2);
            json += buf;
            ++written;
        }
        json += "\n]\n";
        return json;
    }

    static bool write_text_file(const char* output_path, const std::string& text)
    {
        std::FILE* fp = std::fopen(output_path, "w");
        if (fp == nullptr)
        {
            return false;
        }
        std::fwrite(text.data(), 1, text.size(), fp);
        std::fclose(fp);
        return true;
    }

    bool can_speak(const char* audio_in, const char* audio_out) const
    {
        return mSpeech.speech_to_speech != nullptr && audio_in != nullptr && audio_out != nullptr;
    }

    struct SpeechInput {
        std::vector<float> samples;
        int32_t sample_rate{0};
    };

    SpeechInput prepare_speech_input(std::vector<float> samples, int32_t sample_rate) const
    {
        const int32_t target_rate = mSpeech.speech_to_speech->config().sample_rate;
        const bool prefer_python_codec =
            !mSpeech.speech_to_speech->config().hf_python.empty();
        if (sample_rate != target_rate && sample_rate > 0)
        {
            if (prefer_python_codec)
            {
                std::cerr << "[trtf] Keeping input at " << sample_rate
                          << " Hz; Python Mimi encoder will resample to "
                          << target_rate << " Hz" << std::endl;
            }
            else
            {
                const auto in_len = static_cast<int32_t>(samples.size());
                samples = trtf::resample_linear(
                    samples.data(), in_len, sample_rate, target_rate);
                std::cerr << "[trtf] Resampled " << sample_rate << " -> "
                          << target_rate << " Hz (" << in_len << " -> "
                          << samples.size() << " samples)" << std::endl;
                sample_rate = target_rate;
            }
        }
        return SpeechInput{std::move(samples), sample_rate};
    }

    static std::vector<uint8_t> build_sam_mask_pixels(
        const trtf::SamResult& result,
        int32_t mask_index)
    {
        const auto mask_size = static_cast<std::size_t>(result.mask_height) * result.mask_width;
        const auto offset = static_cast<std::size_t>(mask_index) * mask_size;
        std::vector<uint8_t> pixels(mask_size);
        for (int32_t i = 0; i < result.mask_height * result.mask_width; ++i)
        {
            pixels[i] = result.masks[offset + i] > 0.0F
                ? static_cast<uint8_t>(255) : static_cast<uint8_t>(0);
        }
        return pixels;
    }

    static void write_sam_masks(const trtf::SamResult& result, const char* output_dir)
    {
        std::filesystem::create_directories(output_dir);
        for (int32_t m = 0; m < result.num_masks; ++m)
        {
            auto pixels = build_sam_mask_pixels(result, m);
            char fname[64];
            std::snprintf(fname, sizeof(fname), "/mask_%d_iou_%.4f.png",
                          m, result.iou_scores[m]);
            const std::string path = std::string(output_dir) + fname;
            stbi_write_png(path.c_str(), result.mask_width, result.mask_height,
                           1, pixels.data(), result.mask_width);
        }
    }
#endif

    std::string mModelId;
    std::unique_ptr<trtf::ITokenizer> mTokenizer;
    std::unique_ptr<trtf::IGenerationBackend> mBackend;
    std::string mBackendName;
    trtf::GenerationConfig mGenConfig;
    std::string mLastOutput;
    std::string mBundleTempDir;
#if TRTF_HAS_TRT
    struct VisionBackends {
        std::unique_ptr<trtf::SegmentationBackend> segmentation;
        std::unique_ptr<trtf::DetectionBackend> detection;
        std::unique_ptr<trtf::SamBackend> prompted_segmentation;
    };

    struct AudioBackends {
        std::unique_ptr<trtf::BarkBackend> bark;
        std::unique_ptr<trtf::MagpieTTSBackend> magpie_tts;
        std::unique_ptr<trtf::WhisperBackend> whisper;
        trtf::MelFilterbank mel_filterbank;
        int32_t mel_nfft{400};
        int32_t mel_hop_length{160};
        int32_t mel_chunk_length{30};
        int32_t mel_sampling_rate{16000};
    };

    struct MultimodalBackends {
        std::unique_ptr<trtf::OmniBackend> omni;
    };

    struct EncodingBackends {
        std::unique_ptr<trtf::EncoderBackend> encoder;
        trtf::EncoderResult last_result;
    };

    struct OperatorBackends {
        std::unique_ptr<trtf::NeuralOperatorBackend> neural_op;
        trtf::NeuralOperatorResult last_result;
    };

    struct SpeechBackends {
        std::unique_ptr<trtf::SpeechToSpeechBackend> speech_to_speech;
    };

    struct EmbeddingBackends {
        std::unique_ptr<trtf::EmbeddingBackend> embedding;
        trtf::EmbeddingResult last_result;
        std::unique_ptr<trtf::RerankingBackend> reranker;
    };

    VisionBackends mVision;
    AudioBackends mAudio;
    MultimodalBackends mMultimodal;
    EncodingBackends mEncoding;
    OperatorBackends mOperator;
    SpeechBackends mSpeech;
    EmbeddingBackends mEmbedding;
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

#endif

} // namespace

#if TRTF_HAS_TRT
namespace trtf {
namespace cabi {
namespace detail {

trtf::IPipeline* create_text_pipeline_impl(
    const std::string& model_id,
    trtf::TokenizerResult tok,
    std::unique_ptr<trtf::IGenerationBackend> backend,
    const std::string& backend_name)
{
    return make_pipeline(
        model_id, std::move(tok), std::move(backend), backend_name);
}

trtf::IPipeline* create_pipeline_impl(
    const std::string& model_id,
    std::unique_ptr<trtf::ITokenizer> tokenizer,
    std::unique_ptr<trtf::IGenerationBackend> backend,
    const std::string& backend_name)
{
    return new PipelineImpl(
        model_id, std::move(tokenizer), std::move(backend), backend_name, trtf::GenerationConfig{});
}

trtf::IPipeline* create_magpie_tts_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    return trtf::cabi::create_magpie_tts_pipeline(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg, sections,
        runtime_ptr, model_id, hf_python, bundle_path);
}

trtf::IPipeline* create_bark_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    return trtf::cabi::create_bark_pipeline(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg, sections,
        runtime_ptr, model_id, hf_python, bundle_path);
}

trtf::IPipeline* create_omni_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    return trtf::cabi::create_omni_pipeline(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg, sections,
        runtime_ptr, model_id, hf_python, bundle_path);
}

trtf::IPipeline* create_speech_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    return trtf::cabi::create_speech_pipeline(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg, sections,
        runtime_ptr, model_id, hf_python, bundle_path);
}

trtf::IPipeline* create_vision_pipeline_impl(
    const std::string& model_id,
    const std::string& backend_name)
{
    return new PipelineImpl(
        model_id, nullptr, nullptr, backend_name, trtf::GenerationConfig{});
}

void set_segmentation_backend(
    trtf::IPipeline* pipeline,
    std::unique_ptr<trtf::SegmentationBackend> backend)
{
    auto* impl = static_cast<PipelineImpl*>(pipeline);
    impl->set_seg_backend(std::move(backend));
}

void set_detection_backend(
    trtf::IPipeline* pipeline,
    std::unique_ptr<trtf::DetectionBackend> backend)
{
    auto* impl = static_cast<PipelineImpl*>(pipeline);
    impl->set_detection_backend(std::move(backend));
}

void set_sam_backend(
    trtf::IPipeline* pipeline,
    std::unique_ptr<trtf::SamBackend> backend)
{
    auto* impl = static_cast<PipelineImpl*>(pipeline);
    impl->set_sam_backend(std::move(backend));
}

void set_neural_operator_backend(
    trtf::IPipeline* pipeline,
    std::unique_ptr<trtf::NeuralOperatorBackend> backend)
{
    auto* impl = static_cast<PipelineImpl*>(pipeline);
    impl->set_neural_operator_backend(std::move(backend));
}

void set_bark_backend(
    trtf::IPipeline* pipeline,
    std::unique_ptr<trtf::BarkBackend> backend)
{
    auto* impl = static_cast<PipelineImpl*>(pipeline);
    impl->set_bark_backend(std::move(backend));
}

void set_magpie_tts_backend(
    trtf::IPipeline* pipeline,
    std::unique_ptr<trtf::MagpieTTSBackend> backend)
{
    auto* impl = static_cast<PipelineImpl*>(pipeline);
    impl->set_magpie_tts_backend(std::move(backend));
}

void set_omni_backend(
    trtf::IPipeline* pipeline,
    std::unique_ptr<trtf::OmniBackend> backend)
{
    auto* impl = static_cast<PipelineImpl*>(pipeline);
    impl->set_omni_backend(std::move(backend));
}

void set_speech_backend(
    trtf::IPipeline* pipeline,
    std::unique_ptr<trtf::SpeechToSpeechBackend> backend)
{
    auto* impl = static_cast<PipelineImpl*>(pipeline);
    impl->set_speech_backend(std::move(backend));
}

void set_whisper_backend(
    trtf::IPipeline* pipeline,
    std::unique_ptr<trtf::WhisperBackend> backend,
    trtf::MelFilterbank mel_filterbank,
    const trtf::FastPathModelConfig& fp_cfg)
{
    auto* impl = static_cast<PipelineImpl*>(pipeline);
    impl->set_whisper_backend(
        std::move(backend), std::move(mel_filterbank), &fp_cfg);
}

void set_hf_python(
    trtf::IPipeline* pipeline,
    std::string hf_python)
{
    auto* impl = static_cast<PipelineImpl*>(pipeline);
    impl->set_hf_python(std::move(hf_python));
}

trtf::IPipeline* create_encoder_pipeline_impl(
    const std::string& model_id,
    std::unique_ptr<trtf::ITokenizer> tokenizer,
    const std::string& backend_name)
{
    return new PipelineImpl(
        model_id, std::move(tokenizer), nullptr, backend_name, trtf::GenerationConfig{});
}

void set_bundle_temp_dir(
    trtf::IPipeline* pipeline,
    std::string temp_dir)
{
    auto* impl = static_cast<PipelineImpl*>(pipeline);
    impl->set_bundle_temp_dir(std::move(temp_dir));
}

void set_encoder_backend(
    trtf::IPipeline* pipeline,
    std::unique_ptr<trtf::EncoderBackend> backend)
{
    auto* impl = static_cast<PipelineImpl*>(pipeline);
    impl->set_encoder_backend(std::move(backend));
}

void set_embedding_backend(
    trtf::IPipeline* pipeline,
    std::unique_ptr<trtf::EmbeddingBackend> backend)
{
    auto* impl = static_cast<PipelineImpl*>(pipeline);
    impl->set_embedding_backend(std::move(backend));
}

void set_reranking_backend(
    trtf::IPipeline* pipeline,
    std::unique_ptr<trtf::RerankingBackend> backend)
{
    auto* impl = static_cast<PipelineImpl*>(pipeline);
    impl->set_reranking_backend(std::move(backend));
}

void set_default_max_new_tokens(
    trtf::IPipeline* pipeline,
    std::size_t max_new_tokens)
{
    if (auto* impl = dynamic_cast<PipelineImpl*>(pipeline))
    {
        impl->set_default_max_new_tokens(max_new_tokens);
    }
}

} // namespace detail
} // namespace cabi
} // namespace trtf
#endif
