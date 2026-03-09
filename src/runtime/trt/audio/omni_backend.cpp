#include "runtime/trt/audio/omni_backend.h"
#include "runtime/trt/audio/omni_audio_plan.h"

#if TRTF_HAS_TRT

#include "runtime/trt/core/trt_decode_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace trtf {

namespace {

void throw_if_device_resources_unavailable(
    const DeviceKvCache& cache,
    const DeviceResources& resources,
    const char* stage)
{
    if (!cache.ok() || !resources.ok())
    {
        throw std::runtime_error(std::string(stage) + ": failed to allocate device resources");
    }
}

void run_decoder_step_or_throw(
    DecoderStepEngine& engine,
    DeviceKvCache& cache,
    DeviceResources& resources,
    int32_t token,
    std::vector<float>& logits,
    std::string& error,
    const char* error_prefix)
{
    if (!run_decoder_step_device(engine, cache, resources, token, logits, error))
    {
        throw std::runtime_error(std::string(error_prefix) + error);
    }
}

void run_decoder_embed_step_or_throw(
    DecoderStepEngine& engine,
    DeviceKvCache& cache,
    DeviceResources& resources,
    const float* embed_ptr,
    int32_t embed_size,
    std::vector<float>& logits,
    std::string& error,
    const char* error_prefix)
{
    if (!run_decoder_step_device(
            engine,
            cache,
            resources,
            0,
            logits,
            error,
            embed_ptr,
            embed_size,
            1.0F))
    {
        throw std::runtime_error(std::string(error_prefix) + error);
    }
}

void prefill_thinker_or_throw(
    DecoderStepEngine& engine,
    const std::vector<int32_t>& input_ids,
    DeviceKvCache& cache,
    DeviceResources& resources,
    std::vector<float>& logits,
    std::string& error)
{
    if (input_ids.empty())
    {
        return;
    }

    const std::size_t last_idx = input_ids.size() - 1;
    for (std::size_t i = 0; i < last_idx; ++i)
    {
        run_decoder_step_or_throw(
            engine,
            cache,
            resources,
            input_ids[i],
            logits,
            error,
            "Omni Thinker prefill failed: ");
    }

    run_decoder_step_or_throw(
        engine,
        cache,
        resources,
        input_ids[last_idx],
        logits,
        error,
        "Omni Thinker last prefill failed: ");
}

std::vector<int32_t> decode_thinker_tokens_or_throw(
    DecoderStepEngine& engine,
    int32_t max_tokens,
    int32_t eos_id,
    DeviceKvCache& cache,
    DeviceResources& resources,
    std::vector<float>& logits,
    std::string& error)
{
    std::vector<int32_t> output_ids;
    output_ids.reserve(static_cast<std::size_t>(max_tokens));

    for (int32_t step = 0; step < max_tokens; ++step)
    {
        if (logits.empty())
        {
            break;
        }

        const int32_t token = select_argmax_token(logits);
        if (token == eos_id)
        {
            break;
        }
        output_ids.push_back(token);

        run_decoder_step_or_throw(
            engine,
            cache,
            resources,
            token,
            logits,
            error,
            "Omni Thinker decode failed: ");
    }

    return output_ids;
}

void prefill_talker_codes_or_throw(
    DecoderStepEngine& engine,
    const std::vector<float>& hidden_states,
    int32_t num_tokens,
    int32_t talker_hidden,
    int32_t n_codebooks,
    int32_t codebook_size,
    DeviceKvCache& cache,
    DeviceResources& resources,
    std::vector<int32_t>& all_codes)
{
    const OmniTalkerDecodePlan decode_plan = make_omni_talker_decode_plan(
        n_codebooks,
        codebook_size,
        num_tokens);
    std::vector<float> logits;
    std::string error;

    for (int32_t t = 0; t < num_tokens; ++t)
    {
        const float* embed_ptr =
            hidden_states.data() + static_cast<std::size_t>(t) * talker_hidden;
        run_decoder_embed_step_or_throw(
            engine,
            cache,
            resources,
            embed_ptr,
            talker_hidden,
            logits,
            error,
            "Omni Talker prefill failed: ");
        append_omni_talker_codes_from_logits(logits, decode_plan, all_codes);
    }
}

std::vector<float> generate_fallback_waveform(
    const std::vector<int32_t>& codec_tokens,
    int32_t n_codebooks,
    int32_t n_frames,
    const OmniConfig& config)
{
    const int32_t samples_per_frame = config.sample_rate / 75;
    const int32_t total_samples = n_frames * samples_per_frame;
    std::vector<float> waveform(static_cast<std::size_t>(total_samples), 0.0F);

    for (int32_t f = 0; f < n_frames; ++f)
    {
        const float freq = 200.0F +
            static_cast<float>(codec_tokens[f * n_codebooks]) * 800.0F /
            static_cast<float>(config.talker_codebook_size);
        const float amp = 0.3F;
        for (int32_t s = 0; s < samples_per_frame; ++s)
        {
            const auto idx = static_cast<std::size_t>(f) * samples_per_frame + s;
            const float t = static_cast<float>(s) /
                            static_cast<float>(config.sample_rate);
            waveform[idx] = amp * std::sin(2.0F * 3.14159265F * freq * t);
        }
    }

    return waveform;
}

std::vector<float> run_code2wav_trt(
    const std::vector<int32_t>& codec_tokens,
    int32_t n_codebooks,
    int32_t n_frames,
    const OmniConfig& config,
    nvinfer1::IExecutionContext& code2wav_ctx)
{
    const int32_t max_frames = config.code2wav_max_frames;
    const int32_t actual_frames = std::min(n_frames, max_frames);
    const int32_t upsample = config.code2wav_upsample_factor;

    std::vector<int32_t> input_codes = build_omni_code2wav_input_codes(
        codec_tokens, n_codebooks, max_frames, actual_frames);
    const auto input_size = static_cast<std::size_t>(n_codebooks) * max_frames;
    const auto output_elems = static_cast<std::size_t>(max_frames) * upsample;

    CudaBuffer d_input(input_size * sizeof(int32_t));
    CudaBuffer d_output(output_elems * sizeof(float));
    CudaStream stream;

    if (!d_input.ok() || !d_output.ok() || !stream.ok())
    {
        std::cerr << "[trtf] Omni Code2Wav: CUDA alloc failed" << std::endl;
        return {};
    }

    cudaMemcpyAsync(
        d_input.data(),
        input_codes.data(),
        input_size * sizeof(int32_t),
        cudaMemcpyHostToDevice,
        stream.get());

    if (!code2wav_ctx.setTensorAddress("codec_tokens", d_input.data()) ||
        !code2wav_ctx.setTensorAddress("waveform", d_output.data()))
    {
        std::cerr << "[trtf] Omni Code2Wav: tensor binding failed" << std::endl;
        return {};
    }

    if (!code2wav_ctx.enqueueV3(stream.get()))
    {
        std::cerr << "[trtf] Omni Code2Wav: TRT execution failed" << std::endl;
        return {};
    }

    std::vector<float> full_waveform(output_elems);
    cudaMemcpyAsync(
        full_waveform.data(),
        d_output.data(),
        output_elems * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream.get());
    cudaStreamSynchronize(stream.get());

    const auto trimmed = static_cast<std::size_t>(actual_frames) * upsample;
    std::vector<float> waveform(
        full_waveform.begin(),
        full_waveform.begin() + static_cast<std::ptrdiff_t>(trimmed));

    std::cerr << "[trtf] Omni Code2Wav: " << actual_frames << " frames -> "
              << waveform.size() << " samples" << std::endl;
    return waveform;
}

} // namespace

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

OmniBackend::OmniBackend(
    std::unique_ptr<DecoderStepEngine> thinker_engine,
    OmniConfig config)
    : mThinkerEngine(std::move(thinker_engine))
    , mConfig(std::move(config))
{
}

OmniBackend::~OmniBackend() = default;

bool OmniBackend::is_available() const
{
    return mThinkerEngine != nullptr;
}

void OmniBackend::set_audio_encoder(
    TrtUniquePtr<nvinfer1::ICudaEngine> engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> context)
{
    mAudioEncoderEngine = std::move(engine);
    mAudioEncoderCtx = std::move(context);
}

void OmniBackend::set_talker_engine(
    std::unique_ptr<DecoderStepEngine> engine)
{
    mTalkerEngine = std::move(engine);
}

void OmniBackend::set_code2wav_engine(
    TrtUniquePtr<nvinfer1::ICudaEngine> engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> context)
{
    mCode2WavEngine = std::move(engine);
    mCode2WavCtx = std::move(context);
}

// ---------------------------------------------------------------------------
// Stage 0: Thinker (MoE text decoder)
// ---------------------------------------------------------------------------

std::vector<int32_t> OmniBackend::run_thinker(
    const std::vector<int32_t>& input_ids,
    int32_t max_tokens,
    std::vector<float>* hidden_states_out)
{
    (void) hidden_states_out;

    DeviceKvCache cache(*mThinkerEngine);
    DeviceResources resources(*mThinkerEngine);
    throw_if_device_resources_unavailable(cache, resources, "Omni Thinker");

    std::vector<float> logits;
    std::string error;
    prefill_thinker_or_throw(
        *mThinkerEngine,
        input_ids,
        cache,
        resources,
        logits,
        error);

    std::vector<int32_t> output_ids = decode_thinker_tokens_or_throw(
        *mThinkerEngine,
        max_tokens,
        mThinkerEngine->id_eos,
        cache,
        resources,
        logits,
        error);

    std::cerr << "[trtf] Omni Thinker: generated " << output_ids.size()
              << " text tokens" << std::endl;

    return output_ids;
}

std::vector<int32_t> OmniBackend::generate_text(
    const std::vector<int32_t>& input_ids,
    int32_t max_new_tokens)
{
    if (!is_available())
    {
        std::cerr << "[trtf] Omni: backend not initialized" << std::endl;
        return {};
    }

    return run_thinker(input_ids, max_new_tokens);
}

// ---------------------------------------------------------------------------
// Audio encoder
// ---------------------------------------------------------------------------

std::vector<float> OmniBackend::encode_audio(
    const float* mel_features,
    int32_t num_mel_bins,
    int32_t num_frames)
{
    if (!mAudioEncoderEngine || !mAudioEncoderCtx)
    {
        std::cerr << "[trtf] Omni: no audio encoder engine" << std::endl;
        return {};
    }

    const OmniAudioEncodePlan plan = make_omni_audio_encode_plan(
        mConfig,
        num_mel_bins,
        num_frames);
    std::vector<float> input_padded = build_omni_audio_encoder_input(
        mel_features,
        plan);

    // Allocate device buffers
    CudaBuffer d_input(plan.input_size * sizeof(float));
    CudaBuffer d_output(plan.output_elements * sizeof(float));
    CudaStream stream;

    if (!d_input.ok() || !d_output.ok() || !stream.ok())
    {
        std::cerr << "[trtf] Omni audio encoder: CUDA alloc failed" << std::endl;
        return {};
    }

    cudaMemcpyAsync(d_input.data(), input_padded.data(),
                    plan.input_size * sizeof(float),
                    cudaMemcpyHostToDevice, stream.get());

    if (!mAudioEncoderCtx->setTensorAddress("mel_features", d_input.data()) ||
        !mAudioEncoderCtx->setTensorAddress("audio_features", d_output.data()))
    {
        std::cerr << "[trtf] Omni audio encoder: tensor binding failed" << std::endl;
        return {};
    }

    if (!mAudioEncoderCtx->enqueueV3(stream.get()))
    {
        std::cerr << "[trtf] Omni audio encoder: TRT execution failed" << std::endl;
        return {};
    }

    std::vector<float> features(plan.output_elements);
    cudaMemcpyAsync(features.data(), d_output.data(),
                    features.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream.get());
    cudaStreamSynchronize(stream.get());

    std::cerr << "[trtf] Omni audio encoder: " << plan.output_frames
              << " frames, dim=" << plan.embed_dim << std::endl;

    return features;
}

// ---------------------------------------------------------------------------
// Stage 1: Talker (RVQ codec token prediction)
// ---------------------------------------------------------------------------

std::vector<int32_t> OmniBackend::run_talker(
    const std::vector<float>& hidden_states,
    int32_t num_tokens)
{
    if (!mTalkerEngine)
    {
        std::cerr << "[trtf] Omni: no Talker engine" << std::endl;
        return {};
    }

    // The Talker takes Thinker hidden states as input embeddings
    // and produces RVQ codec tokens codebook-by-codebook
    DeviceKvCache cache(*mTalkerEngine);
    DeviceResources resources(*mTalkerEngine);
    throw_if_device_resources_unavailable(cache, resources, "Omni Talker");

    const int32_t n_codebooks = mConfig.talker_n_codebooks;
    const int32_t codebook_size = mConfig.talker_codebook_size;
    const int32_t talker_hidden = mConfig.talker_hidden_size;

    std::vector<int32_t> all_codes;
    all_codes.reserve(static_cast<std::size_t>(num_tokens) * n_codebooks);
    prefill_talker_codes_or_throw(
        *mTalkerEngine,
        hidden_states,
        num_tokens,
        talker_hidden,
        n_codebooks,
        codebook_size,
        cache,
        resources,
        all_codes);

    std::cerr << "[trtf] Omni Talker: generated " << all_codes.size()
              << " codec tokens (" << num_tokens << " frames x "
              << n_codebooks << " codebooks)" << std::endl;

    return all_codes;
}

// ---------------------------------------------------------------------------
// Stage 2: Code2Wav (waveform synthesis)
// ---------------------------------------------------------------------------

std::vector<float> OmniBackend::run_code2wav(
    const std::vector<int32_t>& codec_tokens,
    int32_t n_codebooks,
    int32_t n_frames)
{
    if (!mCode2WavEngine || !mCode2WavCtx)
    {
        std::cerr << "[trtf] Omni: no Code2Wav engine, generating simple waveform"
                  << std::endl;
        return generate_fallback_waveform(codec_tokens, n_codebooks, n_frames, mConfig);
    }

    return run_code2wav_trt(
        codec_tokens,
        n_codebooks,
        n_frames,
        mConfig,
        *mCode2WavCtx);
}

// ---------------------------------------------------------------------------
// Full pipeline
// ---------------------------------------------------------------------------

LegacyAudioResult OmniBackend::generate_audio(
    const std::vector<int32_t>& input_ids,
    int32_t max_semantic_tokens)
{
    LegacyAudioResult result;
    result.sample_rate = mConfig.sample_rate;

    if (!is_available())
    {
        std::cerr << "[trtf] Omni: backend not initialized" << std::endl;
        return result;
    }

    std::cerr << "[trtf] Omni: starting pipeline with " << input_ids.size()
              << " input tokens" << std::endl;

    // Stage 0: Thinker generates text tokens (+ hidden states for Talker)
    std::vector<float> hidden_states;
    auto text_tokens = run_thinker(input_ids, max_semantic_tokens,
                                    &hidden_states);
    if (text_tokens.empty())
    {
        std::cerr << "[trtf] Omni: Thinker produced no tokens" << std::endl;
        return result;
    }

    // Stage 1: Talker converts hidden states to RVQ codec tokens
    const OmniTalkerPlan talker_plan = make_omni_talker_plan(
        text_tokens.size(),
        hidden_states.size(),
        static_cast<bool>(mTalkerEngine));
    if (talker_plan.should_run_talker)
    {
        auto codec_tokens = run_talker(
            hidden_states,
            talker_plan.num_tokens);

        const OmniCodecPlan codec_plan = make_omni_codec_plan(mConfig, codec_tokens.size());
        if (codec_plan.should_run_codec)
        {
            // Stage 2: Code2Wav synthesizes waveform
            auto waveform = run_code2wav(
                codec_tokens,
                codec_plan.n_codebooks,
                codec_plan.n_frames);
            if (!waveform.empty())
            {
                result.waveform = std::move(waveform);
                result.num_samples = static_cast<int32_t>(result.waveform.size());
            }
        }
    }

    std::cerr << "[trtf] Omni: generated " << result.num_samples << " samples ("
              << (result.num_samples > 0
                  ? static_cast<float>(result.num_samples) / result.sample_rate
                  : 0.0F)
              << "s @ " << result.sample_rate << " Hz)" << std::endl;

    return result;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<OmniBackend> CreateOmniBackend(
    std::unique_ptr<DecoderStepEngine> thinker_engine,
    const FastPathModelConfig& cfg)
{
    OmniConfig omni_cfg;
    omni_cfg.sample_rate = cfg.omni_sample_rate;
    omni_cfg.thinker_hidden_size = cfg.hidden_size;
    omni_cfg.thinker_num_layers = cfg.num_layers;
    omni_cfg.thinker_num_heads = cfg.num_heads;
    omni_cfg.num_experts = cfg.omni_num_experts;
    omni_cfg.num_experts_per_tok = cfg.omni_num_experts_per_tok;
    omni_cfg.talker_hidden_size = cfg.omni_talker_hidden_size;
    omni_cfg.talker_num_layers = cfg.omni_talker_num_layers;
    omni_cfg.talker_n_codebooks = cfg.omni_n_codebooks;
    omni_cfg.talker_codebook_size = cfg.omni_codebook_size;

    return std::make_unique<OmniBackend>(
        std::move(thinker_engine), std::move(omni_cfg));
}

} // namespace trtf

#endif // TRTF_HAS_TRT
