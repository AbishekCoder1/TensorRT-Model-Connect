#include "runtime/trt/audio/whisper_backend.h"
#include "runtime/trt/audio/whisper_cross_kv_apply.h"
#include "runtime/trt/audio/whisper_cross_kv_plan.h"
#include "runtime/trt/audio/whisper_decode_policy.h"
#include "runtime/trt/audio/whisper_host_plan.h"
#include "runtime/trt/core/trt_decode_runtime.h"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

namespace {

bool has_encoder_mask_input(const nvinfer1::ICudaEngine& encoder_engine)
{
    const int32_t nt = encoder_engine.getNbIOTensors();
    for (int32_t i = 0; i < nt; ++i)
    {
        if (std::string(encoder_engine.getIOTensorName(i)) == "encoder_mask")
        {
            return true;
        }
    }
    return false;
}

void bind_encoder_tensors(
    nvinfer1::ICudaEngine& encoder_engine,
    nvinfer1::IExecutionContext& encoder_context,
    CudaBuffer& mel_device,
    CudaBuffer& encoder_output,
    CudaBuffer* mask_device,
    bool has_mask_input)
{
    const int32_t num_tensors = encoder_engine.getNbIOTensors();
    for (int32_t i = 0; i < num_tensors; ++i)
    {
        const char* tensor_name = encoder_engine.getIOTensorName(i);
        const std::string name(tensor_name);
        if (name == "mel_features")
        {
            encoder_context.setTensorAddress(tensor_name, mel_device.data());
            continue;
        }
        if (name == "encoder_output")
        {
            encoder_context.setTensorAddress(tensor_name, encoder_output.data());
            continue;
        }
        if (name == "encoder_mask" && has_mask_input)
        {
            encoder_context.setTensorAddress(tensor_name, mask_device->data());
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// WhisperBackend construction
// ---------------------------------------------------------------------------

WhisperBackend::WhisperBackend(
    std::unique_ptr<DecoderStepEngine> decoder_engine,
    TrtUniquePtr<nvinfer1::ICudaEngine> encoder_engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> encoder_context,
    WhisperConfig config,
    const FastPathModelConfig& fp_cfg)
    : mDecoderEngine(std::move(decoder_engine))
    , mEncoderEngine(std::move(encoder_engine))
    , mEncoderContext(std::move(encoder_context))
    , mEncoderOutput(static_cast<std::size_t>(config.max_source_positions) *
                     static_cast<std::size_t>(fp_cfg.hidden_size) * sizeof(float))
    , mWhisperConfig(config)
    , mFpCfg(fp_cfg)
{
    if (mDecoderEngine)
    {
        mCache = std::make_unique<DeviceKvCache>(*mDecoderEngine);
        mResources = std::make_unique<DeviceResources>(*mDecoderEngine);
    }

    // Cross-attention K/V: one buffer per decoder layer, same size as encoder output
    const std::size_t enc_buf_size = mEncoderOutput.size();
    const int32_t dec_layers = config.decoder_layers > 0
        ? config.decoder_layers : fp_cfg.num_layers;
    mCrossK.reserve(dec_layers);
    mCrossV.reserve(dec_layers);
    for (int32_t i = 0; i < dec_layers; ++i)
    {
        mCrossK.emplace_back(enc_buf_size);
        mCrossV.emplace_back(enc_buf_size);
    }
}

bool WhisperBackend::is_available() const
{
    return mDecoderEngine && mEncoderEngine && mCache && mResources;
}

// ---------------------------------------------------------------------------
// generate() -- IGenerationBackend interface
// ---------------------------------------------------------------------------

std::vector<int32_t> WhisperBackend::generate(
    const std::vector<int32_t>& input_ids,
    const GenerationConfig& config)
{
    // For Whisper, generate() assumes encoder has already run and cross-K/V
    // are set up. The input_ids are the initial decoder tokens.
    const int32_t max_tokens = config.max_new_tokens > 0
        ? static_cast<int32_t>(config.max_new_tokens) : 224;
    return run_decoder(input_ids, max_tokens);
}

// ---------------------------------------------------------------------------
// transcribe() -- mel spectrogram to text
// ---------------------------------------------------------------------------

TranscriptionResult WhisperBackend::transcribe(
    const float* mel_data, int32_t mel_bins, int32_t mel_length,
    int32_t max_new_tokens)
{
    std::cerr << "[whisper] Running encoder ..." << std::endl;
    run_encoder(mel_data, mel_bins, mel_length);

    const int32_t mel_full = resolve_whisper_expected_mel_length(mWhisperConfig);
    mActualEncSeqLen = compute_whisper_actual_encoder_length(
        mel_length,
        mel_full,
        mWhisperConfig.max_source_positions);
    if (mActualEncSeqLen > 0)
    {
        std::cerr << "[whisper] Actual encoder seq len: " << mActualEncSeqLen
                  << " / " << mWhisperConfig.max_source_positions << std::endl;
    }

    std::cerr << "[whisper] Computing cross-attention K/V ..." << std::endl;
    compute_cross_kv();

    std::vector<int32_t> initial_tokens = make_whisper_initial_decoder_tokens(mWhisperConfig);

    std::cerr << "[whisper] Running decoder ..." << std::endl;
    auto output_ids = run_decoder(initial_tokens, max_new_tokens);

    TranscriptionResult result;
    result.output_ids = std::move(output_ids);
    result.num_tokens = static_cast<int32_t>(result.output_ids.size());
    return result;
}

// ---------------------------------------------------------------------------
// run_encoder()
// ---------------------------------------------------------------------------

void WhisperBackend::run_encoder(
    const float* mel_data, int32_t mel_bins, int32_t mel_length)
{
    if (!mEncoderEngine || !mEncoderContext)
    {
        return;
    }

    const int32_t expected_length = resolve_whisper_expected_mel_length(mWhisperConfig);
    const std::size_t mel_size = static_cast<std::size_t>(mel_bins) *
                                 static_cast<std::size_t>(expected_length) * sizeof(float);

    CudaBuffer mel_device(mel_size);
    if (mel_length == expected_length)
    {
        cudaMemcpy(mel_device.data(), mel_data, mel_size, cudaMemcpyHostToDevice);
    }
    else
    {
        const auto padded = build_whisper_padded_mel_input(
            mel_data,
            mel_bins,
            mel_length,
            expected_length);
        cudaMemcpy(mel_device.data(), padded.data(), mel_size, cudaMemcpyHostToDevice);
    }

    const int32_t enc_seq = mWhisperConfig.max_source_positions;
    const bool mask_input = has_encoder_mask_input(*mEncoderEngine);
    CudaBuffer mask_device(mask_input ? static_cast<std::size_t>(enc_seq) * sizeof(float) : 0);
    if (mask_input)
    {
        int32_t actual_enc = compute_whisper_actual_encoder_length(mel_length, expected_length, enc_seq);
        if (actual_enc <= 0)
        {
            actual_enc = enc_seq;
        }
        const auto mask = build_whisper_encoder_mask_values(enc_seq, actual_enc);
        cudaMemcpy(
            mask_device.data(),
            mask.data(),
            static_cast<std::size_t>(enc_seq) * sizeof(float),
            cudaMemcpyHostToDevice);
    }

    bind_encoder_tensors(
        *mEncoderEngine,
        *mEncoderContext,
        mel_device,
        mEncoderOutput,
        &mask_device,
        mask_input);

    CudaStream enc_stream;
    if (!mEncoderContext->enqueueV3(enc_stream.get()))
    {
        throw std::runtime_error("Whisper encoder execution failed");
    }
    cudaStreamSynchronize(enc_stream.get());
}

// ---------------------------------------------------------------------------
// compute_cross_kv() -- copy encoder output to per-layer cross K/V buffers
// ---------------------------------------------------------------------------

void WhisperBackend::compute_cross_kv()
{
    // The decoder engine bakes per-layer K/V projections into the graph.
    // Cross_k/cross_v inputs ARE the raw encoder output — the graph
    // applies the projection weights internally.
    const auto plan = make_whisper_cross_kv_plan(
        mWhisperConfig.max_source_positions,
        mFpCfg.hidden_size,
        mActualEncSeqLen);
    std::string error;
    const bool ok = apply_whisper_cross_kv_plan(
        plan,
        mCrossK.size(),
        [this](std::size_t valid_bytes, std::size_t pad_bytes)
        {
            return cudaMemset(
                       static_cast<char*>(mEncoderOutput.data()) + valid_bytes,
                       0,
                       pad_bytes)
                == cudaSuccess;
        },
        [this](std::size_t layer, WhisperCrossKvBufferKind kind, std::size_t bytes)
        {
            void* dst = kind == WhisperCrossKvBufferKind::K
                ? mCrossK[layer].data()
                : mCrossV[layer].data();
            return cudaMemcpy(dst, mEncoderOutput.data(), bytes, cudaMemcpyDeviceToDevice)
                == cudaSuccess;
        },
        error);
    if (!ok)
    {
        throw std::runtime_error(error);
    }
}

// ---------------------------------------------------------------------------
// bind_cross_kv() -- set cross_k/cross_v tensor addresses on decoder context
// ---------------------------------------------------------------------------

void WhisperBackend::bind_cross_kv()
{
    const int32_t dec_layers = static_cast<int32_t>(mCrossK.size());
    for (int32_t i = 0; i < dec_layers; ++i)
    {
        const std::string cross_k_name = layer_tensor_name("cross_k", i);
        const std::string cross_v_name = layer_tensor_name("cross_v", i);
        mDecoderEngine->context->setTensorAddress(cross_k_name.c_str(), mCrossK[i].data());
        mDecoderEngine->context->setTensorAddress(cross_v_name.c_str(), mCrossV[i].data());
    }
}

// ---------------------------------------------------------------------------
// run_decoder() -- autoregressive decode with cross-attention
// ---------------------------------------------------------------------------

std::vector<int32_t> WhisperBackend::run_decoder(
    const std::vector<int32_t>& initial_tokens,
    int32_t max_new_tokens)
{
    if (!mDecoderEngine || !mCache || !mResources) return {};

    auto& engine = *mDecoderEngine;
    auto& cache = *mCache;
    auto& resources = *mResources;
    const int32_t eot_id = mWhisperConfig.eot_token_id;

    // Reset KV cache
    cache.reset(resources.stream.get());
    cudaStreamSynchronize(resources.stream.get());

    // Bind cross-attention K/V (persistent until changed)
    bind_cross_kv();

    const auto result = run_whisper_decode_loop(
        initial_tokens,
        max_new_tokens,
        eot_id,
        [&engine, &cache, &resources](int32_t token, std::vector<float>& logits, std::string& error)
        {
            return run_decoder_step_device(engine, cache, resources, token, logits, error);
        },
        [](const std::vector<float>& logits)
        {
            return select_argmax_token(logits);
        });

    if (result.prefill_failed)
    {
        std::cerr << "[whisper] Prefill step failed: " << result.error << std::endl;
    }
    else if (result.decode_failed)
    {
        std::cerr << "[whisper] Decode step failed: " << result.error << std::endl;
    }

    return result.output_ids;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<WhisperBackend> CreateWhisperBackend(
    std::unique_ptr<DecoderStepEngine> decoder_engine,
    TrtUniquePtr<nvinfer1::ICudaEngine> encoder_engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> encoder_context,
    WhisperConfig config,
    const FastPathModelConfig& fp_cfg)
{
    auto backend = std::make_unique<WhisperBackend>(
        std::move(decoder_engine), std::move(encoder_engine),
        std::move(encoder_context), config, fp_cfg);
    if (!backend->is_available())
    {
        return nullptr;
    }
    return backend;
}

#endif // TRTF_HAS_TRT
} // namespace trtf
