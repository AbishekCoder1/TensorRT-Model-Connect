#include "runtime/trt/whisper_backend.h"
#include "runtime/trt/trt_decode_runtime.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

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

    // Compute actual encoder output length from mel input length.
    // The encoder subsamples via strided convolutions. We compute the
    // subsampling factor from the configured mel_length / max_source_positions
    // and apply the same integer-division formula for each stride-2 stage.
    const int32_t mel_full = mWhisperConfig.mel_length > 0
        ? mWhisperConfig.mel_length
        : mWhisperConfig.max_source_positions * 2;
    if (mel_full > 0 && mel_length < mel_full)
    {
        const int32_t enc_full = mWhisperConfig.max_source_positions;
        // Determine number of stride-2 stages: mel_full / enc_full ≈ 2^stages
        int32_t ratio = (enc_full > 0) ? mel_full / enc_full : 2;
        int32_t stages = 0;
        while (ratio > 1) { ratio /= 2; ++stages; }
        // Apply stride-2 subsampling formula for each stage
        int32_t t = mel_length;
        for (int32_t s = 0; s < stages; ++s)
            t = (t + 2 - 3) / 2 + 1;
        mActualEncSeqLen = t;
        std::cerr << "[whisper] Actual encoder seq len: " << mActualEncSeqLen
                  << " / " << enc_full << std::endl;
    }
    else
    {
        mActualEncSeqLen = 0;  // use full
    }

    std::cerr << "[whisper] Computing cross-attention K/V ..." << std::endl;
    compute_cross_kv();

    // Initial decoder tokens: use custom sequence if provided (e.g. Canary),
    // otherwise Whisper default: <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>
    std::vector<int32_t> initial_tokens;
    if (!mWhisperConfig.decoder_start_token_ids.empty())
    {
        initial_tokens = mWhisperConfig.decoder_start_token_ids;
    }
    else
    {
        initial_tokens.push_back(mWhisperConfig.decoder_start_token_id);
        initial_tokens.push_back(mWhisperConfig.language_token_id);
        initial_tokens.push_back(mWhisperConfig.transcribe_token_id);
        initial_tokens.push_back(mWhisperConfig.notimestamps_token_id);
    }

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
    if (!mEncoderEngine || !mEncoderContext) return;

    const int32_t expected_mel_length = mWhisperConfig.mel_length > 0
        ? mWhisperConfig.mel_length
        : mWhisperConfig.max_source_positions * 2;
    const std::size_t mel_size = static_cast<std::size_t>(mel_bins) *
                                 static_cast<std::size_t>(expected_mel_length) * sizeof(float);

    // Allocate and copy mel input to device
    CudaBuffer mel_device(mel_size);
    if (mel_length == expected_mel_length)
    {
        cudaMemcpy(mel_device.data(), mel_data, mel_size, cudaMemcpyHostToDevice);
    }
    else
    {
        // Pad or truncate: copy row-by-row since source [mel_bins, mel_length]
        // and destination [mel_bins, expected_mel_length] have different strides.
        std::vector<float> padded(mel_bins * expected_mel_length, 0.0F);
        const int32_t copy_len = std::min(mel_length, expected_mel_length);
        for (int32_t bin = 0; bin < mel_bins; ++bin)
        {
            std::memcpy(
                padded.data() + static_cast<std::size_t>(bin) * expected_mel_length,
                mel_data + static_cast<std::size_t>(bin) * mel_length,
                static_cast<std::size_t>(copy_len) * sizeof(float));
        }
        cudaMemcpy(mel_device.data(), padded.data(), mel_size, cudaMemcpyHostToDevice);
    }

    // Compute encoder attention mask if the engine has an encoder_mask input.
    // Mask shape: [1, 1, enc_seq]. 0.0 for valid, -10000.0 for padded.
    const int32_t enc_seq = mWhisperConfig.max_source_positions;
    CudaBuffer mask_device(0);
    bool has_mask_input = false;
    {
        const int32_t nt = mEncoderEngine->getNbIOTensors();
        for (int32_t i = 0; i < nt; ++i)
        {
            if (std::string(mEncoderEngine->getIOTensorName(i)) == "encoder_mask")
            {
                has_mask_input = true;
                break;
            }
        }
    }
    if (has_mask_input)
    {
        // Compute actual encoder seq len from mel_length
        int32_t actual_enc = enc_seq;
        if (mel_length < expected_mel_length)
        {
            int32_t ratio = (enc_seq > 0) ? expected_mel_length / enc_seq : 2;
            int32_t stages = 0;
            while (ratio > 1) { ratio /= 2; ++stages; }
            int32_t t = mel_length;
            for (int32_t s = 0; s < stages; ++s)
                t = (t + 2 - 3) / 2 + 1;
            actual_enc = t;
        }
        std::vector<float> mask(enc_seq, 0.0F);
        for (int32_t i = actual_enc; i < enc_seq; ++i)
            mask[i] = -10000.0F;
        mask_device = CudaBuffer(enc_seq * sizeof(float));
        cudaMemcpy(mask_device.data(), mask.data(),
                    enc_seq * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Set up encoder bindings
    const int32_t num_tensors = mEncoderEngine->getNbIOTensors();
    for (int32_t i = 0; i < num_tensors; ++i)
    {
        const char* tensor_name = mEncoderEngine->getIOTensorName(i);
        if (std::string(tensor_name) == "mel_features")
        {
            mEncoderContext->setTensorAddress(tensor_name, mel_device.data());
        }
        else if (std::string(tensor_name) == "encoder_output")
        {
            mEncoderContext->setTensorAddress(tensor_name, mEncoderOutput.data());
        }
        else if (std::string(tensor_name) == "encoder_mask" && has_mask_input)
        {
            mEncoderContext->setTensorAddress(tensor_name, mask_device.data());
        }
    }

    // Execute encoder
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
    const int32_t enc_seq = mWhisperConfig.max_source_positions;
    const int32_t hidden = mFpCfg.hidden_size;
    const std::size_t buf_size = static_cast<std::size_t>(enc_seq) *
                                 static_cast<std::size_t>(hidden) * sizeof(float);

    // Zero-mask encoder output beyond valid positions so the decoder's
    // cross-attention effectively ignores padded positions.
    if (mActualEncSeqLen > 0 && mActualEncSeqLen < enc_seq)
    {
        const std::size_t valid_bytes = static_cast<std::size_t>(mActualEncSeqLen) *
                                        static_cast<std::size_t>(hidden) * sizeof(float);
        const std::size_t pad_bytes = buf_size - valid_bytes;
        // Zero the padded region of encoder output on device
        cudaMemset(
            static_cast<char*>(mEncoderOutput.data()) + valid_bytes,
            0, pad_bytes);
    }

    for (std::size_t i = 0; i < mCrossK.size(); ++i)
    {
        cudaMemcpy(mCrossK[i].data(), mEncoderOutput.data(),
                    buf_size, cudaMemcpyDeviceToDevice);
        cudaMemcpy(mCrossV[i].data(), mEncoderOutput.data(),
                    buf_size, cudaMemcpyDeviceToDevice);
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

    std::vector<int32_t> output_ids;
    std::vector<float> logits;
    std::string error;

    // Prefill: feed initial tokens one by one
    for (std::size_t t = 0; t < initial_tokens.size(); ++t)
    {
        if (!run_decoder_step_device(engine, cache, resources,
                initial_tokens[t], logits, error))
        {
            std::cerr << "[whisper] Prefill step failed: " << error << std::endl;
            return output_ids;
        }
    }

    // Autoregressive generation
    for (int32_t step = 0; step < max_new_tokens; ++step)
    {
        const int32_t next_token = select_argmax_token(logits);
        output_ids.push_back(next_token);

        if (next_token == eot_id) break;

        if (!run_decoder_step_device(engine, cache, resources,
                next_token, logits, error))
        {
            std::cerr << "[whisper] Decode step failed: " << error << std::endl;
            break;
        }
    }

    return output_ids;
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
