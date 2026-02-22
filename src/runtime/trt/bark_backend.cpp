#include "runtime/trt/bark_backend.h"

#if TRTF_HAS_TRT

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace trtf {

BarkBackend::BarkBackend(
    std::unique_ptr<DecoderStepEngine> semantic_engine,
    BarkConfig config)
    : mSemanticEngine(std::move(semantic_engine))
    , mConfig(std::move(config))
{
}

BarkBackend::~BarkBackend() = default;

bool BarkBackend::is_available() const
{
    return mSemanticEngine != nullptr;
}

AudioResult BarkBackend::generate_audio(
    const std::vector<int32_t>& input_ids,
    int32_t max_semantic_tokens)
{
    AudioResult result;
    result.sample_rate = mConfig.sample_rate;

    if (!mSemanticEngine)
    {
        std::cerr << "[trtf] Bark: no semantic engine available" << std::endl;
        return result;
    }

    // Stage 1: Semantic generation
    // This is a standard autoregressive decoder with KV cache.
    // For now, return a placeholder result since the full 4-stage
    // pipeline requires all engines.
    std::cerr << "[trtf] Bark: semantic stage with " << input_ids.size()
              << " input tokens, max_semantic=" << max_semantic_tokens << std::endl;

    // Placeholder: generate silence
    result.num_samples = mConfig.sample_rate;  // 1 second of silence
    result.waveform.resize(static_cast<std::size_t>(result.num_samples), 0.0F);

    return result;
}

void BarkBackend::set_coarse_engine(std::unique_ptr<DecoderStepEngine> engine)
{
    mCoarseEngine = std::move(engine);
}

void BarkBackend::set_fine_engine(
    TrtUniquePtr<nvinfer1::ICudaEngine> engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> context)
{
    mFineEngine = std::move(engine);
    mFineCtx = std::move(context);
}

void BarkBackend::set_codec_engine(
    TrtUniquePtr<nvinfer1::ICudaEngine> engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> context)
{
    mCodecEngine = std::move(engine);
    mCodecCtx = std::move(context);
}

bool write_wav(const std::string& path, const float* samples,
               int32_t num_samples, int32_t sample_rate)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        return false;
    }

    const int32_t num_channels = 1;
    const int32_t bits_per_sample = 32;
    const int32_t byte_rate = sample_rate * num_channels * (bits_per_sample / 8);
    const int32_t block_align = num_channels * (bits_per_sample / 8);
    const int32_t data_size = num_samples * block_align;
    const int32_t chunk_size = 36 + data_size;

    // RIFF header (44 bytes)
    out.write("RIFF", 4);
    out.write(reinterpret_cast<const char*>(&chunk_size), 4);
    out.write("WAVE", 4);

    // fmt sub-chunk
    out.write("fmt ", 4);
    const int32_t fmt_size = 16;
    out.write(reinterpret_cast<const char*>(&fmt_size), 4);
    const int16_t audio_format = 3;  // IEEE float
    out.write(reinterpret_cast<const char*>(&audio_format), 2);
    const int16_t channels = static_cast<int16_t>(num_channels);
    out.write(reinterpret_cast<const char*>(&channels), 2);
    out.write(reinterpret_cast<const char*>(&sample_rate), 4);
    out.write(reinterpret_cast<const char*>(&byte_rate), 4);
    const int16_t ba = static_cast<int16_t>(block_align);
    out.write(reinterpret_cast<const char*>(&ba), 2);
    const int16_t bps = static_cast<int16_t>(bits_per_sample);
    out.write(reinterpret_cast<const char*>(&bps), 2);

    // data sub-chunk
    out.write("data", 4);
    out.write(reinterpret_cast<const char*>(&data_size), 4);
    out.write(reinterpret_cast<const char*>(samples),
              static_cast<std::streamsize>(data_size));

    return out.good();
}

std::unique_ptr<BarkBackend> CreateBarkBackend(
    std::unique_ptr<DecoderStepEngine> semantic_engine,
    const FastPathModelConfig& cfg)
{
    BarkConfig bark_cfg;
    bark_cfg.sample_rate = cfg.audio_sample_rate;
    bark_cfg.semantic_vocab_size = cfg.semantic_vocab_size;
    bark_cfg.coarse_vocab_size = cfg.coarse_vocab_size;
    bark_cfg.n_coarse_codebooks = cfg.n_coarse_codebooks;
    bark_cfg.n_fine_codebooks = cfg.n_fine_codebooks;
    bark_cfg.semantic_pad_token = cfg.semantic_pad_token;
    bark_cfg.semantic_infer_token = cfg.semantic_infer_token;

    return std::make_unique<BarkBackend>(
        std::move(semantic_engine), std::move(bark_cfg));
}

} // namespace trtf

#endif // TRTF_HAS_TRT
