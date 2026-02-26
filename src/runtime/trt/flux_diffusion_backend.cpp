#include "runtime/trt/flux_diffusion_backend.h"

#if TRTF_HAS_TRT

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

namespace trtf {

// ---------------------------------------------------------------------------
// Flow Match Euler Scheduler (same as Wan, shared logic)
// ---------------------------------------------------------------------------
namespace {

struct FlowMatchEulerState {
    std::vector<double> sigmas;
    std::vector<float> timesteps;
    int32_t num_train_timesteps{1000};
    float shift{1.0F};
    bool use_dynamic_shifting{false};
    float base_shift{0.5F};
    float max_shift{1.15F};
    int32_t image_seq_len{4096};

    void set_timesteps(int32_t num_steps) {
        const double N = static_cast<double>(num_train_timesteps);

        if (use_dynamic_shifting) {
            // HF FLUX scheduler:
            // 1) base sigmas = linspace(1.0, 1.0/num_steps, num_steps)
            //    (current FluxPipeline path when scheduler.use_flow_sigmas is false)
            // 2) apply exponential dynamic shifting with mu
            const double base_seq = 256.0;
            const double max_seq = 4096.0;
            const double m = (static_cast<double>(max_shift) -
                              static_cast<double>(base_shift)) /
                             (max_seq - base_seq);
            const double b = static_cast<double>(base_shift) -
                             m * base_seq;
            const double mu = static_cast<double>(image_seq_len) * m + b;

            // Generate raw sigmas in scheduler sigma-space (not inference-step space).
            const double sigma_max = 1.0;
            const double sigma_min = 1.0 / static_cast<double>(std::max(num_steps, 1));
            std::vector<double> raw_sigmas(static_cast<std::size_t>(num_steps));
            for (int32_t i = 0; i < num_steps; ++i) {
                const double frac = static_cast<double>(i) /
                    static_cast<double>(std::max(num_steps - 1, 1));
                raw_sigmas[static_cast<std::size_t>(i)] =
                    sigma_max + frac * (sigma_min - sigma_max);
            }

            // Apply exponential time shift: sigma = exp(mu)/(exp(mu)+(1/t-1))
            const double exp_mu = std::exp(mu);
            sigmas.resize(static_cast<std::size_t>(num_steps) + 1);
            for (int32_t i = 0; i < num_steps; ++i) {
                const double t = raw_sigmas[static_cast<std::size_t>(i)];
                // Clamp to avoid division by zero
                const double t_clamped = std::max(t, 1e-10);
                const double shifted = exp_mu /
                    (exp_mu + (1.0 / t_clamped - 1.0));
                sigmas[static_cast<std::size_t>(i)] = shifted;
            }
            sigmas[static_cast<std::size_t>(num_steps)] = 0.0;

            timesteps.resize(static_cast<std::size_t>(num_steps));
            for (int32_t i = 0; i < num_steps; ++i) {
                timesteps[static_cast<std::size_t>(i)] =
                    static_cast<float>(sigmas[static_cast<std::size_t>(i)] * N);
            }

            std::cerr << "[flux-scheduler] Dynamic shifting: mu=" << mu
                      << ", exp_mu=" << exp_mu
                      << ", image_seq_len=" << image_seq_len << "\n";
        } else {
            // Original linear shift (for non-FLUX models)
            const double s = static_cast<double>(shift);
            const double raw_sigma_min = 1.0 / N;
            const double sigma_min = s * raw_sigma_min /
                (1.0 + (s - 1.0) * raw_sigma_min);
            const double t_max = 1.0 * N;
            const double t_min = sigma_min * N;

            sigmas.resize(static_cast<std::size_t>(num_steps) + 1);
            for (int32_t i = 0; i < num_steps; ++i) {
                const double frac = static_cast<double>(i) /
                    static_cast<double>(std::max(num_steps - 1, 1));
                const double t_val = t_max + frac * (t_min - t_max);
                double sigma = t_val / N;
                if (std::abs(shift - 1.0F) > 1e-6F) {
                    sigma = s * sigma / (1.0 + (s - 1.0) * sigma);
                }
                sigmas[static_cast<std::size_t>(i)] = sigma;
            }
            sigmas[static_cast<std::size_t>(num_steps)] = 0.0;

            timesteps.resize(static_cast<std::size_t>(num_steps));
            for (int32_t i = 0; i < num_steps; ++i) {
                timesteps[static_cast<std::size_t>(i)] =
                    static_cast<float>(sigmas[static_cast<std::size_t>(i)] *
                                       num_train_timesteps);
            }
        }
    }

    void step(const float* velocity, const float* sample, float* output,
              std::size_t count, int32_t step_index) const {
        const double sigma = sigmas[static_cast<std::size_t>(step_index)];
        const double sigma_next = sigmas[static_cast<std::size_t>(step_index) + 1];
        const double dt = sigma_next - sigma;
        for (std::size_t i = 0; i < count; ++i) {
            output[i] = static_cast<float>(
                static_cast<double>(sample[i]) +
                dt * static_cast<double>(velocity[i]));
        }
    }
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// FluxDiffusionBackend
// ---------------------------------------------------------------------------

FluxDiffusionBackend::FluxDiffusionBackend(
    std::vector<DiffusionEngine> text_encoders,
    DiffusionEngine denoiser,
    DiffusionEngine vae_decoder,
    DiffusionConfig config)
    : DiffusionBackendBase(
          std::move(text_encoders),
          std::move(denoiser),
          std::move(vae_decoder),
          std::move(config))
    , mD_FluxHidden(0)
    , mD_FluxEncoder(0)
    , mD_FluxTemb(0)
    , mD_FluxCos(0)
    , mD_FluxSin(0)
    , mD_FluxOutput(0)
    , mD_ClipInputIds(0)
    , mD_ClipTextEmb(0)
    , mD_ClipPooled(0)
{
    const int32_t dit_dim = mConfig.dit_dim;
    const int32_t head_dim = dit_dim / std::max(mConfig.dit_num_heads, 1);
    mHLatent = mConfig.video_height / mConfig.scale_factor_spatial;
    mWLatent = mConfig.video_width / mConfig.scale_factor_spatial;

    // FLUX applies 2x2 packing: patch_size=[1, ph, pw] where ph=pw=2
    int32_t ph = 2, pw = 2;
    if (mConfig.patch_size.size() >= 3) {
        ph = mConfig.patch_size[1];
        pw = mConfig.patch_size[2];
    }
    mNumImgTokens = (mHLatent / ph) * (mWLatent / pw);
    const int32_t text_seq = mConfig.text_seq_len;
    const int32_t total_seq = text_seq + mNumImgTokens;
    // Output channels = z_dim * ph * pw (packed channels)
    const int32_t out_channels = mConfig.z_dim * ph * pw;

    // FLUX-specific buffers
    mD_FluxHidden = CudaBuffer(
        static_cast<std::size_t>(mNumImgTokens) * static_cast<std::size_t>(dit_dim) * sizeof(float));
    mD_FluxEncoder = CudaBuffer(
        static_cast<std::size_t>(text_seq) * static_cast<std::size_t>(dit_dim) * sizeof(float));
    mD_FluxTemb = CudaBuffer(
        static_cast<std::size_t>(dit_dim) * sizeof(float));
    mD_FluxCos = CudaBuffer(
        static_cast<std::size_t>(total_seq) * static_cast<std::size_t>(head_dim) * sizeof(float));
    mD_FluxSin = CudaBuffer(
        static_cast<std::size_t>(total_seq) * static_cast<std::size_t>(head_dim) * sizeof(float));
    mD_FluxOutput = CudaBuffer(
        static_cast<std::size_t>(mNumImgTokens) * static_cast<std::size_t>(out_channels) * sizeof(float));

    // CLIP buffers (for first text encoder if it exists)
    const int32_t clip_seq = 77;
    const int32_t clip_dim = 768;
    mD_ClipInputIds = CudaBuffer(
        static_cast<std::size_t>(clip_seq) * sizeof(int32_t));
    mD_ClipTextEmb = CudaBuffer(
        static_cast<std::size_t>(clip_seq) * static_cast<std::size_t>(clip_dim) * sizeof(float));
    mD_ClipPooled = CudaBuffer(
        static_cast<std::size_t>(clip_dim) * sizeof(float));

    std::cerr << "[flux] FLUX buffers allocated: img_tokens=" << mNumImgTokens
              << ", dit_dim=" << dit_dim << ", total_seq=" << total_seq
              << ", pack=" << ph << "x" << pw << "\n";
}

// ---------------------------------------------------------------------------
// FLUX timestep embedding
// ---------------------------------------------------------------------------

void FluxDiffusionBackend::compute_flux_timestep_embedding(
    float timestep, float guidance,
    const std::vector<float>& pooled_text,
    std::vector<float>& temb) const
{
    const int32_t dim = mConfig.dit_dim;
    const int32_t freq_dim = mConfig.freq_dim;

    // Sinusoidal timestep embedding
    std::vector<float> t_emb(static_cast<std::size_t>(freq_dim));
    {
        const float t = timestep * 1000.0F;
        const int32_t half = freq_dim / 2;
        for (int32_t i = 0; i < half; ++i) {
            const float freq = std::exp(
                -std::log(10000.0F) * static_cast<float>(i) /
                static_cast<float>(half));
            t_emb[static_cast<std::size_t>(i)] = std::cos(t * freq);
            t_emb[static_cast<std::size_t>(i + half)] = std::sin(t * freq);
        }
    }

    // timestep_embedder MLP: sinusoidal -> Linear -> SiLU -> Linear
    std::vector<float> t_proj(static_cast<std::size_t>(dim));
    cpu_matmul_bias(t_emb.data(),
        mWeights.time_emb_0_weight.data(),
        mWeights.time_emb_0_bias.data(),
        t_proj.data(), 1, freq_dim, dim);
    cpu_silu_inplace(t_proj.data(), static_cast<std::size_t>(dim));

    std::vector<float> t_proj2(static_cast<std::size_t>(dim));
    cpu_matmul_bias(t_proj.data(),
        mWeights.time_emb_2_weight.data(),
        mWeights.time_emb_2_bias.data(),
        t_proj2.data(), 1, dim, dim);

    // text_embedder MLP: pooled -> Linear -> SiLU -> Linear
    std::vector<float> text_proj(static_cast<std::size_t>(dim));
    if (!mWeights.text_proj_weight.empty() && !pooled_text.empty()) {
        const int32_t text_in_dim = static_cast<int32_t>(pooled_text.size());
        cpu_matmul_bias(pooled_text.data(),
            mWeights.text_proj_weight.data(),
            mWeights.text_proj_bias.data(),
            text_proj.data(), 1, text_in_dim, dim);
        cpu_silu_inplace(text_proj.data(), static_cast<std::size_t>(dim));

        if (!mWeights.text_proj_2_weight.empty()) {
            std::vector<float> text_proj2(static_cast<std::size_t>(dim));
            cpu_matmul_bias(text_proj.data(),
                mWeights.text_proj_2_weight.data(),
                mWeights.text_proj_2_bias.data(),
                text_proj2.data(), 1, dim, dim);
            text_proj = std::move(text_proj2);
        }
    }

    // Guidance embedding MLP (if guidance_embeds is enabled)
    std::vector<float> guidance_proj(static_cast<std::size_t>(dim), 0.0F);
    if (mConfig.guidance_embeds &&
        !mWeights.guidance_emb_0_weight.empty()) {
        // Diffusers FLUX forward currently scales guidance by 1000 before
        // feeding it into time_text_embed (same convention as timestep).
        std::vector<float> g_emb(static_cast<std::size_t>(freq_dim));
        {
            const float g = guidance * 1000.0F;
            const int32_t half = freq_dim / 2;
            for (int32_t i = 0; i < half; ++i) {
                const float freq = std::exp(
                    -std::log(10000.0F) * static_cast<float>(i) /
                    static_cast<float>(half));
                g_emb[static_cast<std::size_t>(i)] = std::cos(g * freq);
                g_emb[static_cast<std::size_t>(i + half)] = std::sin(g * freq);
            }
        }

        // Linear -> SiLU -> Linear
        std::vector<float> g_proj(static_cast<std::size_t>(dim));
        cpu_matmul_bias(g_emb.data(),
            mWeights.guidance_emb_0_weight.data(),
            mWeights.guidance_emb_0_bias.data(),
            g_proj.data(), 1, freq_dim, dim);
        cpu_silu_inplace(g_proj.data(), static_cast<std::size_t>(dim));

        cpu_matmul_bias(g_proj.data(),
            mWeights.guidance_emb_2_weight.data(),
            mWeights.guidance_emb_2_bias.data(),
            guidance_proj.data(), 1, dim, dim);
    }

    // Combined: temb = timestep_embed + text_embed + guidance_embed
    temb.resize(static_cast<std::size_t>(dim));
    for (int32_t i = 0; i < dim; ++i) {
        temb[static_cast<std::size_t>(i)] =
            t_proj2[static_cast<std::size_t>(i)] +
            text_proj[static_cast<std::size_t>(i)] +
            guidance_proj[static_cast<std::size_t>(i)];
    }

    // Diagnostic: print temb statistics
    {
        float tmin = temb[0], tmax = temb[0];
        double tsum = 0.0;
        for (auto v : temb) {
            tmin = std::min(tmin, v);
            tmax = std::max(tmax, v);
            tsum += static_cast<double>(v);
        }
        std::cerr << "[flux-temb] t=" << timestep << " g=" << guidance
                  << " temb=[" << tmin << "," << tmax
                  << ",mean=" << (tsum / static_cast<double>(dim)) << "]\n";
    }
}

// ---------------------------------------------------------------------------
// FLUX 2D RoPE
// ---------------------------------------------------------------------------

void FluxDiffusionBackend::compute_flux_rope(
    int32_t h_patches, int32_t w_patches, int32_t text_seq_len,
    std::vector<float>& cos_out,
    std::vector<float>& sin_out) const
{
    const int32_t head_dim = mConfig.dit_dim / std::max(mConfig.dit_num_heads, 1);
    const int32_t num_img_tokens = h_patches * w_patches;
    const int32_t total_seq = text_seq_len + num_img_tokens;

    cos_out.resize(static_cast<std::size_t>(total_seq) * static_cast<std::size_t>(head_dim), 1.0F);
    sin_out.resize(static_cast<std::size_t>(total_seq) * static_cast<std::size_t>(head_dim), 0.0F);

    // FLUX uses 3-axis RoPE: (text_pos, h_pos, w_pos)
    // axes_dims_rope default = (16, 56, 56) => total = 128 = head_dim
    // For text tokens: only text_pos dimension is used
    // For image tokens: h and w dimensions are used

    // HF FLUX RoPE: text_ids = zeros(text_seq, 3), image_ids[:,0]=0, [:,1]=h, [:,2]=w
    // ALL text tokens have position 0 on ALL axes (identity rotation).
    // Image tokens have position 0 on axis 0 (text), h on axis 1, w on axis 2.

    const float theta = 10000.0F;
    const int32_t text_dim = 16;    // First axis
    const int32_t h_dim = 56;       // Second axis
    const int32_t w_dim = 56;       // Third axis

    auto encode_pos = [&](float* cos_row, float* sin_row,
                          int32_t text_pos, int32_t h_pos, int32_t w_pos) {
        int32_t offset = 0;
        // Text dimension
        for (int32_t i = 0; i < text_dim / 2; ++i) {
            const float freq = 1.0F / std::pow(theta, 2.0F * static_cast<float>(i) / static_cast<float>(text_dim));
            const float angle = static_cast<float>(text_pos) * freq;
            cos_row[offset + 2 * i] = std::cos(angle);
            cos_row[offset + 2 * i + 1] = std::cos(angle);
            sin_row[offset + 2 * i] = std::sin(angle);
            sin_row[offset + 2 * i + 1] = std::sin(angle);
        }
        offset += text_dim;
        // Height dimension
        for (int32_t i = 0; i < h_dim / 2; ++i) {
            const float freq = 1.0F / std::pow(theta, 2.0F * static_cast<float>(i) / static_cast<float>(h_dim));
            const float angle = static_cast<float>(h_pos) * freq;
            cos_row[offset + 2 * i] = std::cos(angle);
            cos_row[offset + 2 * i + 1] = std::cos(angle);
            sin_row[offset + 2 * i] = std::sin(angle);
            sin_row[offset + 2 * i + 1] = std::sin(angle);
        }
        offset += h_dim;
        // Width dimension
        for (int32_t i = 0; i < w_dim / 2; ++i) {
            const float freq = 1.0F / std::pow(theta, 2.0F * static_cast<float>(i) / static_cast<float>(w_dim));
            const float angle = static_cast<float>(w_pos) * freq;
            cos_row[offset + 2 * i] = std::cos(angle);
            cos_row[offset + 2 * i + 1] = std::cos(angle);
            sin_row[offset + 2 * i] = std::sin(angle);
            sin_row[offset + 2 * i + 1] = std::sin(angle);
        }
    };

    // Text tokens: ALL positions are (0, 0, 0) -- identity rotation
    // cos_out/sin_out are already initialized to 1.0/0.0 respectively,
    // which is correct for position 0 on all axes.
    for (int32_t t = 0; t < text_seq_len; ++t) {
        encode_pos(
            cos_out.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(head_dim),
            sin_out.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(head_dim),
            0, 0, 0);
    }

    // Image tokens: position (0, h, w)
    for (int32_t h = 0; h < h_patches; ++h) {
        for (int32_t w = 0; w < w_patches; ++w) {
            const int32_t idx = text_seq_len + h * w_patches + w;
            encode_pos(
                cos_out.data() + static_cast<std::size_t>(idx) * static_cast<std::size_t>(head_dim),
                sin_out.data() + static_cast<std::size_t>(idx) * static_cast<std::size_t>(head_dim),
                0, h, w);
        }
    }
}

// ---------------------------------------------------------------------------
// CLIP encoder execution
// ---------------------------------------------------------------------------

bool FluxDiffusionBackend::run_clip_encoder(
    const std::vector<int32_t>& input_ids,
    std::vector<float>& pooled_output,
    std::string& error)
{
    // CLIP is the first text encoder (index 0) if present
    if (mTextEncoders.empty()) {
        pooled_output.assign(768, 0.0F);  // Zero pooled output
        return true;
    }

    auto& te = mTextEncoders[0];
    const int32_t seq_len = 77;
    const int32_t clip_dim = 768;

    std::vector<int32_t> padded(
        static_cast<std::size_t>(seq_len),
        std::max(mClipPadTokenId, 0));
    const auto n = std::min(static_cast<std::size_t>(seq_len), input_ids.size());
    std::copy_n(input_ids.begin(), n, padded.begin());

    cudaMemcpyAsync(mD_ClipInputIds.data(), padded.data(),
        padded.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice, mStream.get());

    te.context->setTensorAddress("input_ids", mD_ClipInputIds.data());
    te.context->setTensorAddress("text_embeddings", mD_ClipTextEmb.data());
    te.context->setTensorAddress("pooled_output", mD_ClipPooled.data());

    if (!te.context->enqueueV3(mStream.get())) {
        error = "CLIP enqueueV3 failed";
        return false;
    }

    // HF CLIP pools from the EOS-position logic (argmax over token IDs for
    // OpenAI-style CLIP tokenizers), not from the final padded token.
    // We reproduce that behavior from text_embeddings to avoid blurred/weak
    // conditioning when prompts are shorter than max length.
    std::vector<float> clip_hidden(
        static_cast<std::size_t>(seq_len) *
        static_cast<std::size_t>(clip_dim));
    cudaMemcpyAsync(clip_hidden.data(), mD_ClipTextEmb.data(),
        clip_hidden.size() * sizeof(float),
        cudaMemcpyDeviceToHost, mStream.get());
    cudaStreamSynchronize(mStream.get());

    int32_t pool_idx = -1;
    if (mClipEosTokenId >= 0) {
        for (int32_t i = 0; i < seq_len; ++i) {
            if (padded[static_cast<std::size_t>(i)] == mClipEosTokenId) {
                pool_idx = i;
                break;  // first EOS position, matching HF behavior
            }
        }
    }

    if (pool_idx < 0) {
        // Fallback to OpenAI CLIP-compatible pooling when EOS is unavailable:
        // choose the first position of the largest token id.
        int32_t max_token_id = padded[0];
        pool_idx = 0;
        for (int32_t i = 1; i < seq_len; ++i) {
            if (padded[static_cast<std::size_t>(i)] > max_token_id) {
                max_token_id = padded[static_cast<std::size_t>(i)];
                pool_idx = i;
            }
        }
    }
    if (pool_idx < 0) pool_idx = 0;
    if (pool_idx >= seq_len) pool_idx = seq_len - 1;

    pooled_output.resize(static_cast<std::size_t>(clip_dim));
    const float* pooled_src = clip_hidden.data() +
        static_cast<std::size_t>(pool_idx) * static_cast<std::size_t>(clip_dim);
    std::copy_n(pooled_src, static_cast<std::size_t>(clip_dim), pooled_output.begin());

    return true;
}

// ---------------------------------------------------------------------------
// FLUX DiT denoiser execution
// ---------------------------------------------------------------------------

bool FluxDiffusionBackend::run_flux_denoiser(
    const std::vector<float>& hidden,
    const std::vector<float>& encoder_hidden,
    const std::vector<float>& temb,
    const std::vector<float>& cos_vals,
    const std::vector<float>& sin_vals,
    std::vector<float>& output,
    std::string& error)
{
    auto& ctx = mDenoiser.context;

    cudaMemcpyAsync(mD_FluxHidden.data(), hidden.data(),
        hidden.size() * sizeof(float), cudaMemcpyHostToDevice, mStream.get());
    cudaMemcpyAsync(mD_FluxEncoder.data(), encoder_hidden.data(),
        encoder_hidden.size() * sizeof(float), cudaMemcpyHostToDevice, mStream.get());
    cudaMemcpyAsync(mD_FluxTemb.data(), temb.data(),
        temb.size() * sizeof(float), cudaMemcpyHostToDevice, mStream.get());
    cudaMemcpyAsync(mD_FluxCos.data(), cos_vals.data(),
        cos_vals.size() * sizeof(float), cudaMemcpyHostToDevice, mStream.get());
    cudaMemcpyAsync(mD_FluxSin.data(), sin_vals.data(),
        sin_vals.size() * sizeof(float), cudaMemcpyHostToDevice, mStream.get());

    ctx->setTensorAddress("hidden_states", mD_FluxHidden.data());
    ctx->setTensorAddress("encoder_hidden_states", mD_FluxEncoder.data());
    ctx->setTensorAddress("temb", mD_FluxTemb.data());
    ctx->setTensorAddress("rotary_cos", mD_FluxCos.data());
    ctx->setTensorAddress("rotary_sin", mD_FluxSin.data());
    ctx->setTensorAddress("output", mD_FluxOutput.data());

    if (!ctx->enqueueV3(mStream.get())) {
        error = "FLUX DiT enqueueV3 failed";
        return false;
    }

    const auto out_size = mD_FluxOutput.size() / sizeof(float);
    output.resize(out_size);
    cudaMemcpyAsync(output.data(), mD_FluxOutput.data(),
        mD_FluxOutput.size(), cudaMemcpyDeviceToHost, mStream.get());
    cudaStreamSynchronize(mStream.get());

    return true;
}

// ---------------------------------------------------------------------------
// Run T5 encoder at specific index
// ---------------------------------------------------------------------------

bool FluxDiffusionBackend::run_t5_encoder_at(
    int32_t encoder_idx,
    const std::vector<int32_t>& input_ids,
    std::vector<float>& text_embeddings,
    std::string& error)
{
    if (encoder_idx < 0 || encoder_idx >= static_cast<int32_t>(mTextEncoders.size())) {
        error = "T5 encoder index " + std::to_string(encoder_idx) + " out of range";
        return false;
    }

    auto& te = mTextEncoders[static_cast<std::size_t>(encoder_idx)];
    const int32_t seq_len = mConfig.text_seq_len;
    const int32_t te_dim = mConfig.text_encoder_dim;

    std::vector<int32_t> padded_ids(static_cast<std::size_t>(seq_len), 0);
    const auto copy_len = std::min(
        static_cast<std::size_t>(seq_len), input_ids.size());
    std::copy_n(input_ids.begin(), copy_len, padded_ids.begin());

    std::vector<float> mask(static_cast<std::size_t>(seq_len), -1e9F);
    for (int32_t i = 0; i < seq_len; ++i) {
        if (padded_ids[static_cast<std::size_t>(i)] != 0) {
            mask[static_cast<std::size_t>(i)] = 0.0F;
        }
    }

    cudaMemcpyAsync(mD_InputIds.data(), padded_ids.data(),
        padded_ids.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice, mStream.get());
    cudaMemcpyAsync(mD_AttentionMask.data(), mask.data(),
        mask.size() * sizeof(float),
        cudaMemcpyHostToDevice, mStream.get());

    te.context->setTensorAddress("input_ids", mD_InputIds.data());
    te.context->setTensorAddress("attention_mask", mD_AttentionMask.data());
    te.context->setTensorAddress("text_embeddings", mD_TextEmbeddings.data());

    if (!te.context->enqueueV3(mStream.get())) {
        error = "T5 enqueueV3 failed";
        return false;
    }

    const auto emb_size = static_cast<std::size_t>(seq_len) *
                          static_cast<std::size_t>(te_dim);
    text_embeddings.resize(emb_size);
    cudaMemcpyAsync(text_embeddings.data(), mD_TextEmbeddings.data(),
        emb_size * sizeof(float),
        cudaMemcpyDeviceToHost, mStream.get());
    cudaStreamSynchronize(mStream.get());

    return true;
}

// ---------------------------------------------------------------------------
// Parse FLUX-specific preprocessor weights
// ---------------------------------------------------------------------------

void FluxDiffusionBackend::parse_flux_preprocessor_weights(
    const PreprocessorWeights& /*base_weights*/)
{
    // FLUX preprocessor weights are stored in the bundle's preprocessor_weights
    // section with FLUX-specific key names. The base PreprocessorWeights uses
    // Wan-style keys which don't match. Instead, we parse directly from the
    // raw preprocessor data using our own key mapping.
    //
    // For now, the timestep/text embedder weights are loaded into the base
    // mWeights via a fallback mapping. The x_embedder and context_embedder
    // are loaded separately here.
    //
    // Key mapping:
    //   FLUX                                  -> Base PreprocessorWeights
    //   x_embedder.weight                    -> (custom mFluxXEmbedW)
    //   x_embedder.bias                      -> (custom mFluxXEmbedB)
    //   context_embedder.weight              -> (custom mFluxCtxEmbedW)
    //   context_embedder.bias                -> (custom mFluxCtxEmbedB)
    //   time_text_embed.timestep_embedder.*  -> time_emb_0/2
    //   time_text_embed.text_embedder.*      -> text_proj

    // The weights are already loaded by the base class parse_preprocessor_weights
    // but with wrong key names. We need to access the raw section data.
    // For now, we check the base weights and note that they may be incomplete.

    mFluxWeightsLoaded = !mWeights.time_emb_0_weight.empty();
    if (!mFluxWeightsLoaded) {
        std::cerr << "[flux] Warning: Preprocessor weights not loaded\n";
    }
}

// ---------------------------------------------------------------------------
// Override set_preprocessor_weights to extract FLUX-specific weights
// ---------------------------------------------------------------------------

void FluxDiffusionBackend::set_preprocessor_weights(PreprocessorWeights weights)
{
    // Copy x_embedder weights (serialized as patch_embedding by Python)
    mFluxXEmbedW = weights.patch_embed_weight;
    mFluxXEmbedB = weights.patch_embed_bias;

    // Copy context_embedder weights
    mFluxCtxEmbedW = weights.context_embed_weight;
    mFluxCtxEmbedB = weights.context_embed_bias;

    mFluxWeightsLoaded = !mFluxXEmbedW.empty();

    std::cerr << "[flux] Preprocessor: x_embedder="
              << (mFluxXEmbedW.empty() ? "MISSING" : "OK")
              << ", context_embedder="
              << (mFluxCtxEmbedW.empty() ? "MISSING" : "OK")
              << "\n";

    // Store in base class for timestep/text embedder usage
    DiffusionBackendBase::set_preprocessor_weights(std::move(weights));
}

// ---------------------------------------------------------------------------
// set_clip_tokenizer / set_prompt
// ---------------------------------------------------------------------------

void FluxDiffusionBackend::set_clip_tokenizer(std::unique_ptr<ITokenizer> tok)
{
    mClipTokenizer = std::move(tok);
    mClipEosTokenId = -1;
    mClipPadTokenId = 0;
    if (mClipTokenizer) {
        const char* kCandidates[] = {
            "<|endoftext|>",
            "</s>",
            "<eos>",
        };
        for (const char* tok_name : kCandidates) {
            try {
                const int32_t id = mClipTokenizer->id_for_token(tok_name);
                if (id >= 0) {
                    mClipEosTokenId = id;
                    break;
                }
            } catch (const std::exception&) {
                // Best-effort lookup; ignore and fall back.
            }
        }

        if (mClipEosTokenId >= 0) {
            // OpenAI CLIP-style tokenizers use EOS as pad token.
            mClipPadTokenId = mClipEosTokenId;
        } else {
            const char* kPadCandidates[] = {
                "<pad>",
                "</s>",
                "<eos>",
            };
            for (const char* tok_name : kPadCandidates) {
                try {
                    const int32_t id = mClipTokenizer->id_for_token(tok_name);
                    if (id >= 0) {
                        mClipPadTokenId = id;
                        break;
                    }
                } catch (const std::exception&) {
                    // Best-effort lookup; ignore and keep fallback.
                }
            }
        }
    }
    std::cerr << "[flux] CLIP tokenizer set (eos_id=" << mClipEosTokenId
              << ", pad_id=" << mClipPadTokenId << ")\n";
}

void FluxDiffusionBackend::set_prompt(std::string prompt)
{
    mRawPrompt = std::move(prompt);
}

// ---------------------------------------------------------------------------
// Generate image (full pipeline)
// ---------------------------------------------------------------------------

VideoResult FluxDiffusionBackend::generate_video(
    const std::vector<int32_t>& input_ids,
    int32_t num_inference_steps,
    float guidance_scale)
{
    VideoResult result;
    std::string error;

    if (num_inference_steps <= 0) num_inference_steps = mConfig.num_inference_steps;
    if (guidance_scale < 0.0F) guidance_scale = mConfig.guidance_scale;

    const int32_t dit_dim = mConfig.dit_dim;
    const int32_t text_seq = mConfig.text_seq_len;
    const int32_t z_dim = mConfig.z_dim;  // VAE latent channels (16)

    // FLUX 2x2 packing: [C, H, W] -> [H/2*W/2, C*4]
    int32_t ph = 2, pw = 2;
    if (mConfig.patch_size.size() >= 3) {
        ph = mConfig.patch_size[1];
        pw = mConfig.patch_size[2];
    }
    const int32_t packed_channels = z_dim * ph * pw;  // 16*2*2 = 64
    const int32_t h_packed = mHLatent / ph;  // 64
    const int32_t w_packed = mWLatent / pw;  // 64

    // 1. Run CLIP encoder (pooled output for conditioning)
    //    FLUX needs CLIP tokens (BPE vocab), not T5 tokens (sentencepiece).
    //    If a separate CLIP tokenizer is available, re-tokenize the raw prompt.
    std::vector<float> pooled_output;
    if (mTextEncoders.size() >= 2) {
        std::vector<int32_t> clip_ids;
        if (mClipTokenizer && !mRawPrompt.empty()) {
            clip_ids = mClipTokenizer->encode(mRawPrompt);
            std::cerr << "[flux] CLIP tokenized prompt (" << clip_ids.size()
                      << " tokens) from raw text\n";
        } else {
            // Fallback: use same input_ids (will produce wrong results
            // if these are T5 tokens, but better than nothing)
            clip_ids = input_ids;
            std::cerr << "[flux] Warning: no CLIP tokenizer, using T5 tokens for CLIP encoder\n";
        }
        if (!run_clip_encoder(clip_ids, pooled_output, error)) {
            std::cerr << "[flux] CLIP encoder failed: " << error << "\n";
            return result;
        }
        std::cerr << "[flux] CLIP encoder done\n";
    } else {
        pooled_output.assign(768, 0.0F);
        std::cerr << "[flux] No CLIP encoder, using zero pooled output\n";
    }

    // 2. Run T5 encoder (sequence output for cross-attention)
    //    Uses the original input_ids (T5 sentencepiece tokens).
    std::vector<float> text_embeddings;
    const int32_t t5_idx = (mTextEncoders.size() > 1) ? 1 : 0;
    if (!run_t5_encoder_at(t5_idx, input_ids, text_embeddings, error)) {
        std::cerr << "[flux] T5 encoder failed: " << error << "\n";
        return result;
    }
    std::cerr << "[flux] T5 encoder done\n";

    // 3. Project text to DiT dim: context_embedder
    const int32_t t5_dim = mConfig.text_encoder_dim;
    std::vector<float> encoder_hidden(
        static_cast<std::size_t>(text_seq) * static_cast<std::size_t>(dit_dim), 0.0F);
    if (!mFluxCtxEmbedW.empty()) {
        cpu_matmul_bias(
            text_embeddings.data(),
            mFluxCtxEmbedW.data(),
            mFluxCtxEmbedB.empty() ? nullptr : mFluxCtxEmbedB.data(),
            encoder_hidden.data(),
            text_seq, t5_dim, dit_dim);
        std::cerr << "[flux] Context embedder projection done\n";
    } else {
        std::cerr << "[flux] Warning: No context_embedder weights, encoder_hidden is zero\n";
    }

    // 4. Compute RoPE (uses packed spatial dims for image tokens)
    std::vector<float> cos_vals, sin_vals;
    compute_flux_rope(h_packed, w_packed, text_seq, cos_vals, sin_vals);

    // 5. Initialize random latents in VAE space: [z_dim, h_lat, w_lat]
    const auto latent_size = static_cast<std::size_t>(z_dim) *
                             static_cast<std::size_t>(mHLatent) *
                             static_cast<std::size_t>(mWLatent);
    std::vector<float> latents(latent_size);
    {
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0F, 1.0F);
        for (auto& v : latents) v = dist(gen);
    }

    // 6. Setup scheduler
    FlowMatchEulerState scheduler;
    scheduler.shift = mConfig.flow_shift;
    scheduler.use_dynamic_shifting = mConfig.use_dynamic_shifting;
    scheduler.base_shift = mConfig.base_shift;
    scheduler.max_shift = mConfig.max_shift;
    scheduler.image_seq_len = mNumImgTokens;
    scheduler.set_timesteps(num_inference_steps);

    std::cerr << "[flux] Starting denoising loop (" << num_inference_steps << " steps)"
              << " latents=[" << z_dim << "," << mHLatent << "," << mWLatent << "]"
              << " packed=[" << mNumImgTokens << "," << packed_channels << "] ...\n";

    // 7. Denoising loop
    std::vector<float> hidden(
        static_cast<std::size_t>(mNumImgTokens) * static_cast<std::size_t>(dit_dim));
    std::vector<float> denoiser_output;

    for (int32_t step = 0; step < num_inference_steps; ++step) {
        const float t = scheduler.timesteps[static_cast<std::size_t>(step)] / 1000.0F;

        // Compute timestep embedding
        std::vector<float> temb;
        compute_flux_timestep_embedding(t, guidance_scale, pooled_output, temb);

        // Pack latents 2x2: [C, H, W] -> [H/2*W/2, C*2*2]
        // Layout: for each 2x2 patch (py, px), gather C channels * 4 spatial positions
        std::vector<float> packed(
            static_cast<std::size_t>(mNumImgTokens) *
            static_cast<std::size_t>(packed_channels));
        for (int32_t py = 0; py < h_packed; ++py) {
            for (int32_t px = 0; px < w_packed; ++px) {
                const int32_t tok_idx = py * w_packed + px;
                float* dst = packed.data() +
                    static_cast<std::size_t>(tok_idx) *
                    static_cast<std::size_t>(packed_channels);
                int32_t off = 0;
                for (int32_t c = 0; c < z_dim; ++c) {
                    for (int32_t dy = 0; dy < ph; ++dy) {
                        for (int32_t dx = 0; dx < pw; ++dx) {
                            const int32_t y = py * ph + dy;
                            const int32_t x = px * pw + dx;
                            const auto src_idx =
                                static_cast<std::size_t>(c) *
                                static_cast<std::size_t>(mHLatent * mWLatent) +
                                static_cast<std::size_t>(y * mWLatent + x);
                            dst[off++] = latents[src_idx];
                        }
                    }
                }
            }
        }

        // x_embedder: [num_img_tokens, packed_channels] @ [packed_channels, dim] + bias
        if (!mFluxXEmbedW.empty()) {
            cpu_matmul_bias(
                packed.data(),
                mFluxXEmbedW.data(),
                mFluxXEmbedB.empty() ? nullptr : mFluxXEmbedB.data(),
                hidden.data(),
                mNumImgTokens, packed_channels, dit_dim);
        } else {
            std::fill(hidden.begin(), hidden.end(), 0.0F);
        }

        // Run denoiser
        if (!run_flux_denoiser(hidden, encoder_hidden, temb,
                               cos_vals, sin_vals, denoiser_output, error)) {
            std::cerr << "[flux] Denoiser step " << step << " failed: " << error << "\n";
            return result;
        }

        // Unpack output: [num_img_tokens, packed_channels] -> [C, H, W]
        std::vector<float> velocity(latent_size);
        for (int32_t py = 0; py < h_packed; ++py) {
            for (int32_t px = 0; px < w_packed; ++px) {
                const int32_t tok_idx = py * w_packed + px;
                const float* src = denoiser_output.data() +
                    static_cast<std::size_t>(tok_idx) *
                    static_cast<std::size_t>(packed_channels);
                int32_t off = 0;
                for (int32_t c = 0; c < z_dim; ++c) {
                    for (int32_t dy = 0; dy < ph; ++dy) {
                        for (int32_t dx = 0; dx < pw; ++dx) {
                            const int32_t y = py * ph + dy;
                            const int32_t x = px * pw + dx;
                            const auto dst_idx =
                                static_cast<std::size_t>(c) *
                                static_cast<std::size_t>(mHLatent * mWLatent) +
                                static_cast<std::size_t>(y * mWLatent + x);
                            velocity[dst_idx] = src[off++];
                        }
                    }
                }
            }
        }

        // Scheduler step: latents = latents + dt * velocity
        std::vector<float> next_latents(latent_size);
        scheduler.step(velocity.data(), latents.data(), next_latents.data(),
                      latent_size, step);
        latents = std::move(next_latents);

        // Diagnostics: print latent and output statistics
        {
            float lat_min = latents[0], lat_max = latents[0];
            double lat_sum = 0.0;
            for (auto v : latents) {
                lat_min = std::min(lat_min, v);
                lat_max = std::max(lat_max, v);
                lat_sum += static_cast<double>(v);
            }
            float vel_min = velocity[0], vel_max = velocity[0];
            double vel_sum = 0.0;
            for (auto v : velocity) {
                vel_min = std::min(vel_min, v);
                vel_max = std::max(vel_max, v);
                vel_sum += static_cast<double>(v);
            }
            float hid_min = hidden[0], hid_max = hidden[0];
            double hid_sum = 0.0;
            for (auto v : hidden) {
                hid_min = std::min(hid_min, v);
                hid_max = std::max(hid_max, v);
                hid_sum += static_cast<double>(v);
            }
            std::cerr << "[flux] Step " << (step + 1) << "/" << num_inference_steps
                      << " t=" << scheduler.timesteps[static_cast<std::size_t>(step)]
                      << " dt=" << (scheduler.sigmas[static_cast<std::size_t>(step) + 1] -
                                    scheduler.sigmas[static_cast<std::size_t>(step)])
                      << " latent=[" << lat_min << "," << lat_max
                      << ",mean=" << (lat_sum / static_cast<double>(latent_size))
                      << "] vel=[" << vel_min << "," << vel_max
                      << ",mean=" << (vel_sum / static_cast<double>(latent_size))
                      << "] hidden=[" << hid_min << "," << hid_max
                      << ",mean=" << (hid_sum / static_cast<double>(hidden.size()))
                      << "]\n";
        }
    }

    // Dump final latents to file for external comparison
    {
        const std::string dump_path = "/tmp/flux_final_latents.raw";
        std::ofstream dump(dump_path, std::ios::binary);
        if (dump.is_open()) {
            dump.write(reinterpret_cast<const char*>(latents.data()),
                       latents.size() * sizeof(float));
            dump.close();
            std::cerr << "[flux] Dumped final latents (" << latents.size()
                      << " floats) to " << dump_path << "\n";
        }
    }

    // 8. VAE decode via native TRT engine
    std::cerr << "[flux] Decoding latents via native TRT VAE engine ...\n";
    {
        const int32_t H_out = mConfig.video_height;
        const int32_t W_out = mConfig.video_width;
        const auto vae_in_size = static_cast<std::size_t>(z_dim) *
                                  static_cast<std::size_t>(mHLatent) *
                                  static_cast<std::size_t>(mWLatent);
        const auto vae_out_size = static_cast<std::size_t>(3) *
                                   static_cast<std::size_t>(H_out) *
                                   static_cast<std::size_t>(W_out);

        // Prepare input: [1, C, H, W] layout (latents already in CHW)
        CudaBuffer d_vae_in(vae_in_size * sizeof(float));
        CudaBuffer d_vae_out(vae_out_size * sizeof(float));

        cudaMemcpyAsync(d_vae_in.data(), latents.data(),
            vae_in_size * sizeof(float), cudaMemcpyHostToDevice, mStream.get());

        mVaeDecoder.context->setTensorAddress("latents", d_vae_in.data());
        mVaeDecoder.context->setTensorAddress("image", d_vae_out.data());

        if (!mVaeDecoder.context->enqueueV3(mStream.get())) {
            std::cerr << "[flux] VAE TRT enqueueV3 failed\n";
            return result;
        }

        std::vector<float> vae_output(vae_out_size);
        cudaMemcpyAsync(vae_output.data(), d_vae_out.data(),
            vae_out_size * sizeof(float), cudaMemcpyDeviceToHost, mStream.get());
        cudaStreamSynchronize(mStream.get());

        // Convert from CHW [-1,1] to THWC [0,1] (T=1 for FLUX)
        result.num_frames = 1;
        result.height = H_out;
        result.width = W_out;
        result.frames.resize(static_cast<std::size_t>(H_out * W_out * 3));

        for (int32_t fh = 0; fh < H_out; ++fh) {
            for (int32_t fw = 0; fw < W_out; ++fw) {
                for (int32_t fc = 0; fc < 3; ++fc) {
                    const auto src = static_cast<std::size_t>(fc) *
                        static_cast<std::size_t>(H_out * W_out) +
                        static_cast<std::size_t>(fh * W_out + fw);
                    const auto dst = static_cast<std::size_t>(fh * W_out * 3 + fw * 3 + fc);
                    float v = (vae_output[src] + 1.0F) * 0.5F;
                    v = std::max(0.0F, std::min(1.0F, v));
                    result.frames[dst] = v;
                }
            }
        }
    }

    std::cerr << "[flux] Image generated: " << result.width << "x" << result.height << "\n";
    return result;
}

} // namespace trtf

#endif // TRTF_HAS_TRT
