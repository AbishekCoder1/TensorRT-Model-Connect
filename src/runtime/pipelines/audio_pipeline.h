#pragma once

// Audio pipelines: WhisperPipeline, BarkPipeline, MagpiePipeline,
// SpeechPipeline, OmniPipeline.
//
// Whisper, Bark, Magpie, and Speech own old-style backends and delegate to them.
// OmniPipeline is migrated to TrtModule + KvCache (new runtime).

#include "trtf/pipeline.h"
#include "trtf/tokenizer.h"
#include "trtf/runtime/trt_module.h"
#include "trtf/runtime/kv_cache.h"

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
struct MelFilterbank;
struct OmniConfig;
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
    /// Construct with TrtModules + KvCaches (new runtime).
    OmniPipeline(
        std::unique_ptr<TrtModule> thinker,
        std::unique_ptr<KvCache> thinker_cache,
        std::unique_ptr<TrtModule> talker,
        std::unique_ptr<KvCache> talker_cache,
        std::unique_ptr<TrtModule> code2wav,
        OmniConfig config,
        cudaStream_t stream,
        std::shared_ptr<ITokenizer> tokenizer = nullptr,
        std::string model_id_str = "");

    ~OmniPipeline() override;

    AudioResult generate_audio(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "OmniPipeline"; }

private:
    // Run one thinker decoder step: token_id -> logits. Updates cache.
    void run_thinker_step(int32_t token_id, std::vector<float>& logits);

    // Run one talker decoder step with input embedding. Updates cache.
    void run_talker_embed_step(const float* embed_ptr, int32_t embed_size,
                               std::vector<float>& logits);

    // Stage 0: Thinker generates text tokens and hidden states.
    std::vector<int32_t> run_thinker(
        const std::vector<int32_t>& input_ids,
        int32_t max_tokens,
        std::vector<float>& hidden_states_out);

    // Stage 1: Talker converts hidden states to RVQ codec tokens.
    std::vector<int32_t> run_talker(
        const std::vector<float>& hidden_states,
        int32_t num_tokens);

    // Stage 2: Code2Wav synthesizes waveform from RVQ tokens.
    std::vector<float> run_code2wav(
        const std::vector<int32_t>& codec_tokens,
        int32_t n_codebooks,
        int32_t n_frames);

    std::unique_ptr<TrtModule> thinker_;
    std::unique_ptr<KvCache> thinker_cache_;
    std::unique_ptr<TrtModule> talker_;
    std::unique_ptr<KvCache> talker_cache_;
    std::unique_ptr<TrtModule> code2wav_;
    std::unique_ptr<OmniConfig> config_;
    cudaStream_t stream_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
};

} // namespace trtf

#endif // TRTF_HAS_TRT
