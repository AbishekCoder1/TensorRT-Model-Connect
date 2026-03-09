#pragma once

// Audio pipelines: WhisperPipeline, BarkPipeline, MagpiePipeline,
// SpeechPipeline, OmniPipeline.
//
// Each pipeline owns an old-style backend and delegates inference to it.

#include "trtf/pipeline.h"
#include "trtf/tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if TRTF_HAS_TRT

// Forward-declare old backends to avoid pulling in heavy headers.
namespace trtf {
class WhisperBackend;
class BarkBackend;
class MagpieTTSBackend;
class SpeechToSpeechBackend;
class OmniBackend;
struct MelFilterbank;
} // namespace trtf

namespace trtf {

class WhisperPipeline final : public IPipeline {
public:
    /// Construct with a fully-initialized WhisperBackend.
    WhisperPipeline(
        std::unique_ptr<WhisperBackend> backend,
        MelFilterbank mel_fb,
        int32_t mel_n_fft,
        int32_t mel_hop_length,
        int32_t mel_chunk_length,
        int32_t mel_sampling_rate,
        std::shared_ptr<ITokenizer> tokenizer = nullptr,
        std::string model_id_str = "");

    ~WhisperPipeline() override;

    TextResult transcribe(const float* audio_data, int32_t num_samples,
                          int32_t max_new_tokens,
                          int32_t input_sample_rate = 0) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "WhisperPipeline"; }

private:
    std::unique_ptr<WhisperBackend> backend_;
    std::unique_ptr<MelFilterbank> mel_fb_;
    std::shared_ptr<ITokenizer> tokenizer_;
    int32_t mel_n_fft_;
    int32_t mel_hop_length_;
    int32_t mel_chunk_length_;
    int32_t mel_sampling_rate_;
    std::string model_id_;
};

class BarkPipeline final : public IPipeline {
public:
    /// Construct with a fully-initialized BarkBackend.
    BarkPipeline(
        std::unique_ptr<BarkBackend> backend,
        std::shared_ptr<ITokenizer> tokenizer,
        std::string model_id_str = "");

    ~BarkPipeline() override;

    AudioResult generate_audio(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "BarkPipeline"; }

private:
    std::unique_ptr<BarkBackend> backend_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
};

class MagpiePipeline final : public IPipeline {
public:
    /// Construct with a fully-initialized MagpieTTSBackend.
    MagpiePipeline(
        std::unique_ptr<MagpieTTSBackend> backend,
        std::shared_ptr<ITokenizer> tokenizer,
        std::string model_id_str = "");

    ~MagpiePipeline() override;

    AudioResult generate_audio(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "MagpiePipeline"; }

private:
    std::unique_ptr<MagpieTTSBackend> backend_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
};

class SpeechPipeline final : public IPipeline {
public:
    /// Construct with a fully-initialized SpeechToSpeechBackend.
    SpeechPipeline(
        std::unique_ptr<SpeechToSpeechBackend> backend,
        std::string model_id_str = "");

    ~SpeechPipeline() override;

    AudioResult speak(const float* audio_in, int32_t num_samples,
                      const GenerateConfig& cfg = {},
                      int32_t input_sample_rate = 0) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "SpeechPipeline"; }

private:
    std::unique_ptr<SpeechToSpeechBackend> backend_;
    std::string model_id_;
};

class OmniPipeline final : public IPipeline {
public:
    /// Construct with a fully-initialized OmniBackend.
    OmniPipeline(
        std::unique_ptr<OmniBackend> backend,
        std::shared_ptr<ITokenizer> tokenizer,
        std::string model_id_str = "");

    ~OmniPipeline() override;

    AudioResult generate_audio(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "OmniPipeline"; }

private:
    std::unique_ptr<OmniBackend> backend_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
};

} // namespace trtf

#endif // TRTF_HAS_TRT
