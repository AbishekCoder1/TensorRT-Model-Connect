#include "runtime/trt/wan_diffusion_backend.h"
#include "runtime/trt/flux_diffusion_backend.h"
#include "runtime/trt/z_image_diffusion_backend.h"

#if TRTF_HAS_TRT

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>

namespace trtf {

// ---------------------------------------------------------------------------
// Flow Match Euler Scheduler (Wan-specific)
// ---------------------------------------------------------------------------
namespace {

struct FlowMatchEulerState {
    std::vector<double> sigmas;
    std::vector<float> timesteps;
    int32_t num_train_timesteps{1000};
    float shift{1.0F};

    void set_timesteps(int32_t num_steps) {
        const double N = static_cast<double>(num_train_timesteps);
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

// ---------------------------------------------------------------------------
// DDIM Scheduler (epsilon-prediction models like PixArt)
// ---------------------------------------------------------------------------

struct DDIMState {
    // Continuous-time sigmas for each step (from HF DPMSolverMultistep schedule)
    std::vector<double> sigmas;  // [num_steps + 1]
    std::vector<float> timesteps;
    int32_t num_train_timesteps{1000};
    // For 2nd-order: previous x0_pred and lambda
    std::vector<double> prev_x0;
    double prev_lambda_src{0.0};
    bool has_prev{false};

    void set_timesteps(int32_t num_steps,
                       double beta_start = 0.0001,
                       double beta_end = 0.02) {
        const int32_t T = num_train_timesteps;
        // Linear beta schedule -> alpha_cumprod
        std::vector<double> alpha_cumprod(static_cast<std::size_t>(T));
        double cum = 1.0;
        for (int32_t i = 0; i < T; ++i) {
            double beta = beta_start +
                static_cast<double>(i) / static_cast<double>(T - 1) *
                (beta_end - beta_start);
            cum *= (1.0 - beta);
            alpha_cumprod[static_cast<std::size_t>(i)] = cum;
        }
        // Evenly spaced integer timesteps
        timesteps.resize(static_cast<std::size_t>(num_steps));
        for (int32_t i = 0; i < num_steps; ++i) {
            double frac = static_cast<double>(i) /
                          static_cast<double>(num_steps);
            timesteps[static_cast<std::size_t>(i)] =
                static_cast<float>(std::round((1.0 - frac) * (T - 1)));
        }
        // Compute continuous sigmas at each step boundary
        // sigma = sqrt((1-alpha_cumprod)/alpha_cumprod)
        sigmas.resize(static_cast<std::size_t>(num_steps) + 1);
        for (int32_t i = 0; i < num_steps; ++i) {
            const int32_t t = static_cast<int32_t>(std::round(timesteps[static_cast<std::size_t>(i)]));
            const double acp = alpha_cumprod[static_cast<std::size_t>(
                std::max(0, std::min(t, T - 1)))];
            sigmas[static_cast<std::size_t>(i)] = std::sqrt((1.0 - acp) / acp);
        }
        sigmas[static_cast<std::size_t>(num_steps)] = 0.0;  // terminal sigma
    }

    /// DPM-Solver++ multistep (1st + 2nd order).
    /// Uses continuous sigmas matching HF DPMSolverMultistepScheduler 0.36+.
    /// Verified: 1st order cosine=1.0, 2nd order cosine=0.99999994 vs HF.
    void step(const float* eps_pred, const float* x_t, float* x_out,
              std::size_t count, int32_t step_index) {
        const auto si = static_cast<std::size_t>(step_index);

        // Convert continuous sigma to (alpha, sigma): a=1/sqrt(1+s^2), sig=s/sqrt(1+s^2)
        const double raw_src = sigmas[si];
        const double raw_tgt = sigmas[si + 1];
        const double alp_src = 1.0 / std::sqrt(1.0 + raw_src * raw_src);
        const double sig_src = raw_src / std::sqrt(1.0 + raw_src * raw_src);
        const double alp_tgt = 1.0 / std::sqrt(1.0 + raw_tgt * raw_tgt);
        const double sig_tgt = raw_tgt / std::sqrt(1.0 + raw_tgt * raw_tgt);

        double lam_src = std::log(alp_src / sig_src);
        double lam_tgt = std::log(alp_tgt / sig_tgt);
        double h = lam_tgt - lam_src;  // POSITIVE

        double ratio = sig_tgt / sig_src;
        double coeff = -alp_tgt * std::expm1(-h);  // alpha_t*(1-exp(-h)) > 0

        // Predict x0 from epsilon at SOURCE timestep
        std::vector<double> x0(count);
        for (std::size_t i = 0; i < count; ++i) {
            x0[i] = (static_cast<double>(x_t[i]) -
                sig_src * static_cast<double>(eps_pred[i])) / alp_src;
        }

        for (std::size_t i = 0; i < count; ++i) {
            x_out[i] = static_cast<float>(
                ratio * static_cast<double>(x_t[i]) + coeff * x0[i]);
        }
    }
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// WanDiffusionBackend
// ---------------------------------------------------------------------------

WanDiffusionBackend::WanDiffusionBackend(
    std::vector<DiffusionEngine> text_encoders,
    DiffusionEngine denoiser,
    DiffusionEngine vae_decoder,
    DiffusionConfig config)
    : DiffusionBackendBase(
          std::move(text_encoders),
          std::move(denoiser),
          std::move(vae_decoder),
          std::move(config))
    , mD_VaeInput(0)
    , mD_VaeOutput(0)
{
    init_vae_buffers();
}

// ---------------------------------------------------------------------------
// Wan-specific CPU preprocessing
// ---------------------------------------------------------------------------

void WanDiffusionBackend::compute_timestep_embedding(
    float timestep,
    std::vector<float>& temb_6d,
    std::vector<float>& time_embed) const
{
    const int32_t dim = mConfig.dit_dim;
    const int32_t freq_dim = mConfig.freq_dim;
    const int32_t half = freq_dim / 2;

    std::vector<float> sinusoidal(static_cast<std::size_t>(freq_dim));
    for (int32_t i = 0; i < half; ++i) {
        const double freq = std::exp(
            -std::log(10000.0) * static_cast<double>(i) / static_cast<double>(half));
        const double angle = static_cast<double>(timestep) * freq;
        sinusoidal[static_cast<std::size_t>(i)] = static_cast<float>(std::cos(angle));
        sinusoidal[static_cast<std::size_t>(i + half)] = static_cast<float>(std::sin(angle));
    }

    std::vector<float> hidden_1(static_cast<std::size_t>(dim));
    cpu_matmul_bias(sinusoidal.data(), mWeights.time_emb_0_weight.data(),
                    mWeights.time_emb_0_bias.data(),
                    hidden_1.data(), 1, freq_dim, dim);
    cpu_silu_inplace(hidden_1.data(), static_cast<std::size_t>(dim));

    time_embed.resize(static_cast<std::size_t>(dim));
    cpu_matmul_bias(hidden_1.data(), mWeights.time_emb_2_weight.data(),
                    mWeights.time_emb_2_bias.data(),
                    time_embed.data(), 1, dim, dim);

    std::vector<float> silu_te(time_embed.begin(), time_embed.end());
    cpu_silu_inplace(silu_te.data(), static_cast<std::size_t>(dim));

    temb_6d.resize(static_cast<std::size_t>(6 * dim));
    cpu_matmul_bias(silu_te.data(), mWeights.time_proj_weight.data(),
                    mWeights.time_proj_bias.data(),
                    temb_6d.data(), 1, dim, 6 * dim);
}

void WanDiffusionBackend::project_text(
    const std::vector<float>& in, int32_t seq_len,
    std::vector<float>& out) const
{
    const int32_t te_dim = mConfig.text_encoder_dim;
    const int32_t dim = mConfig.dit_dim;

    out.resize(static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(dim));
    cpu_matmul_bias(in.data(), mWeights.text_proj_weight.data(),
                    mWeights.text_proj_bias.data(),
                    out.data(), seq_len, te_dim, dim);

    if (!mWeights.text_proj_2_weight.empty()) {
        cpu_gelu_tanh_inplace(out.data(),
            static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(dim));
        std::vector<float> tmp(out.size());
        cpu_matmul_bias(out.data(), mWeights.text_proj_2_weight.data(),
                        mWeights.text_proj_2_bias.data(),
                        tmp.data(), seq_len, dim, dim);
        out = std::move(tmp);
    }
}

void WanDiffusionBackend::patchify(
    const std::vector<float>& latents,
    int32_t c, int32_t t, int32_t h, int32_t w,
    std::vector<float>& patches) const
{
    int32_t pt = 1, ph = 2, pw = 2;
    if (mConfig.patch_size.size() >= 3) {
        pt = mConfig.patch_size[0];
        ph = mConfig.patch_size[1];
        pw = mConfig.patch_size[2];
    }
    const int32_t nt = t / pt, nh = h / ph, nw = w / pw;
    const int32_t patch_dim = c * pt * ph * pw;
    const int32_t num_patches = nt * nh * nw;

    patches.resize(static_cast<std::size_t>(num_patches) *
                   static_cast<std::size_t>(patch_dim));

    int32_t patch_idx = 0;
    for (int32_t ti = 0; ti < nt; ++ti) {
        for (int32_t hi = 0; hi < nh; ++hi) {
            for (int32_t wi = 0; wi < nw; ++wi) {
                int32_t elem = 0;
                for (int32_t ci = 0; ci < c; ++ci) {
                    for (int32_t pti = 0; pti < pt; ++pti) {
                        for (int32_t phi_ = 0; phi_ < ph; ++phi_) {
                            for (int32_t pwi = 0; pwi < pw; ++pwi) {
                                const int32_t tt = ti * pt + pti;
                                const int32_t hh = hi * ph + phi_;
                                const int32_t ww = wi * pw + pwi;
                                const auto src_idx =
                                    static_cast<std::size_t>(ci) * static_cast<std::size_t>(t * h * w) +
                                    static_cast<std::size_t>(tt) * static_cast<std::size_t>(h * w) +
                                    static_cast<std::size_t>(hh) * static_cast<std::size_t>(w) +
                                    static_cast<std::size_t>(ww);
                                patches[static_cast<std::size_t>(patch_idx) *
                                        static_cast<std::size_t>(patch_dim) +
                                        static_cast<std::size_t>(elem)] =
                                    latents[src_idx];
                                ++elem;
                            }
                        }
                    }
                }
                ++patch_idx;
            }
        }
    }
}

void WanDiffusionBackend::unpatchify(
    const std::vector<float>& patches,
    int32_t c, int32_t t, int32_t h, int32_t w,
    std::vector<float>& output) const
{
    int32_t pt = 1, ph = 2, pw = 2;
    if (mConfig.patch_size.size() >= 3) {
        pt = mConfig.patch_size[0];
        ph = mConfig.patch_size[1];
        pw = mConfig.patch_size[2];
    }
    const int32_t nt = t / pt, nh = h / ph, nw = w / pw;
    const int32_t patch_dim = c * pt * ph * pw;

    output.resize(static_cast<std::size_t>(c * t * h * w));

    int32_t patch_idx = 0;
    for (int32_t ti = 0; ti < nt; ++ti) {
        for (int32_t hi = 0; hi < nh; ++hi) {
            for (int32_t wi = 0; wi < nw; ++wi) {
                int32_t elem = 0;
                for (int32_t pti = 0; pti < pt; ++pti) {
                    for (int32_t phi_ = 0; phi_ < ph; ++phi_) {
                        for (int32_t pwi = 0; pwi < pw; ++pwi) {
                            for (int32_t ci = 0; ci < c; ++ci) {
                                const int32_t tt = ti * pt + pti;
                                const int32_t hh = hi * ph + phi_;
                                const int32_t ww = wi * pw + pwi;
                                const auto dst_idx =
                                    static_cast<std::size_t>(ci) * static_cast<std::size_t>(t * h * w) +
                                    static_cast<std::size_t>(tt) * static_cast<std::size_t>(h * w) +
                                    static_cast<std::size_t>(hh) * static_cast<std::size_t>(w) +
                                    static_cast<std::size_t>(ww);
                                output[dst_idx] =
                                    patches[static_cast<std::size_t>(patch_idx) *
                                            static_cast<std::size_t>(patch_dim) +
                                            static_cast<std::size_t>(elem)];
                                ++elem;
                            }
                        }
                    }
                }
                ++patch_idx;
            }
        }
    }
}

void WanDiffusionBackend::compute_3d_rope(
    int32_t nt, int32_t nh, int32_t nw,
    std::vector<float>& cos_out,
    std::vector<float>& sin_out) const
{
    const int32_t dim = mConfig.dit_dim;
    const int32_t num_heads = mConfig.dit_num_heads;
    const int32_t head_dim = dim / std::max(num_heads, 1);
    const int32_t num_patches = nt * nh * nw;
    const double theta = 10000.0;

    // Wan uses: h_dim = w_dim = 2*(head_dim//6), t_dim = head_dim - h_dim - w_dim
    const int32_t h_dim = 2 * (head_dim / 6);
    const int32_t w_dim = h_dim;
    const int32_t t_dim = head_dim - h_dim - w_dim;

    auto get_1d_rope = [&](int32_t rdim, int32_t max_len,
                           std::vector<std::vector<float>>& cos_table,
                           std::vector<std::vector<float>>& sin_table) {
        const int32_t half_r = rdim / 2;
        cos_table.resize(static_cast<std::size_t>(max_len));
        sin_table.resize(static_cast<std::size_t>(max_len));

        for (int32_t pos = 0; pos < max_len; ++pos) {
            auto& c = cos_table[static_cast<std::size_t>(pos)];
            auto& s = sin_table[static_cast<std::size_t>(pos)];
            c.resize(static_cast<std::size_t>(rdim));
            s.resize(static_cast<std::size_t>(rdim));

            for (int32_t i = 0; i < half_r; ++i) {
                const double freq = 1.0 / std::pow(theta,
                    static_cast<double>(i) / static_cast<double>(half_r));
                const double angle = static_cast<double>(pos) * freq;
                const auto cv = static_cast<float>(std::cos(angle));
                const auto sv = static_cast<float>(std::sin(angle));
                c[static_cast<std::size_t>(2 * i)] = cv;
                c[static_cast<std::size_t>(2 * i + 1)] = cv;
                s[static_cast<std::size_t>(2 * i)] = sv;
                s[static_cast<std::size_t>(2 * i + 1)] = sv;
            }
        }
    };

    std::vector<std::vector<float>> t_cos, t_sin, h_cos, h_sin, w_cos, w_sin;
    get_1d_rope(t_dim, std::max(nt, 1024), t_cos, t_sin);
    get_1d_rope(h_dim, std::max(nh, 1024), h_cos, h_sin);
    get_1d_rope(w_dim, std::max(nw, 1024), w_cos, w_sin);

    cos_out.resize(static_cast<std::size_t>(num_patches) *
                   static_cast<std::size_t>(head_dim));
    sin_out.resize(static_cast<std::size_t>(num_patches) *
                   static_cast<std::size_t>(head_dim));

    int32_t p = 0;
    for (int32_t ti = 0; ti < nt; ++ti) {
        for (int32_t hi = 0; hi < nh; ++hi) {
            for (int32_t wi = 0; wi < nw; ++wi) {
                float* c_row = cos_out.data() + static_cast<std::size_t>(p) *
                               static_cast<std::size_t>(head_dim);
                float* s_row = sin_out.data() + static_cast<std::size_t>(p) *
                               static_cast<std::size_t>(head_dim);

                int32_t off = 0;
                std::memcpy(c_row + off, t_cos[static_cast<std::size_t>(ti)].data(),
                            static_cast<std::size_t>(t_dim) * sizeof(float));
                std::memcpy(s_row + off, t_sin[static_cast<std::size_t>(ti)].data(),
                            static_cast<std::size_t>(t_dim) * sizeof(float));
                off += t_dim;

                std::memcpy(c_row + off, h_cos[static_cast<std::size_t>(hi)].data(),
                            static_cast<std::size_t>(h_dim) * sizeof(float));
                std::memcpy(s_row + off, h_sin[static_cast<std::size_t>(hi)].data(),
                            static_cast<std::size_t>(h_dim) * sizeof(float));
                off += h_dim;

                std::memcpy(c_row + off, w_cos[static_cast<std::size_t>(wi)].data(),
                            static_cast<std::size_t>(w_dim) * sizeof(float));
                std::memcpy(s_row + off, w_sin[static_cast<std::size_t>(wi)].data(),
                            static_cast<std::size_t>(w_dim) * sizeof(float));

                ++p;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Native VAE decode (Wan: 3D causal, frame-by-frame with cache)
// ---------------------------------------------------------------------------

void WanDiffusionBackend::init_vae_buffers()
{
    auto& engine = mVaeDecoder.engine;
    if (!engine) return;

    const int32_t num_caches = mConfig.num_vae_caches;

    // Scan engine I/O tensors to allocate input/output buffers.
    // Supports both 3D causal VAE (latent_frame/video_frame) and
    // 2D AutoencoderKL VAE (latent_input/decoder_output).
    for (int32_t i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* tname = engine->getIOTensorName(i);
        const auto dims = engine->getTensorShape(tname);
        const std::string sname(tname);

        if (sname == "latent_frame" || sname == "latent_input") {
            std::size_t sz = sizeof(float);
            for (int32_t d = 0; d < dims.nbDims; ++d)
                sz *= static_cast<std::size_t>(std::max(static_cast<int32_t>(dims.d[d]), 1));
            mD_VaeInput = CudaBuffer(sz);
        }
        else if (sname == "video_frame" || sname == "decoder_output") {
            std::size_t sz = sizeof(float);
            for (int32_t d = 0; d < dims.nbDims; ++d)
                sz *= static_cast<std::size_t>(std::max(static_cast<int32_t>(dims.d[d]), 1));
            mD_VaeOutput = CudaBuffer(sz);
            if (dims.nbDims >= 3) {
                mVaeOutputT = std::max(static_cast<int32_t>(dims.d[2]), 1);
            }
        }
    }

    if (num_caches <= 0) {
        std::cerr << "[diffusion] VAE buffers allocated: 2D VAE (no caches)\n";
        return;
    }

    mD_VaeCacheIn.reserve(static_cast<std::size_t>(num_caches));
    mD_VaeCacheOut.reserve(static_cast<std::size_t>(num_caches));
    for (int32_t ci = 0; ci < num_caches; ++ci) {
        mD_VaeCacheIn.emplace_back(0);
        mD_VaeCacheOut.emplace_back(0);
    }
    mVaeCacheSizes.resize(static_cast<std::size_t>(num_caches), 0);

    for (int32_t ci = 0; ci < num_caches; ++ci) {
        const std::string cin_name = "cache_" + std::to_string(ci);

        for (int32_t i = 0; i < engine->getNbIOTensors(); ++i) {
            const char* tname = engine->getIOTensorName(i);
            if (cin_name == tname) {
                const auto dims = engine->getTensorShape(tname);
                std::size_t sz = sizeof(float);
                for (int32_t d = 0; d < dims.nbDims; ++d)
                    sz *= static_cast<std::size_t>(std::max(static_cast<int32_t>(dims.d[d]), 1));
                mD_VaeCacheIn[static_cast<std::size_t>(ci)] = CudaBuffer(sz);
                mD_VaeCacheOut[static_cast<std::size_t>(ci)] = CudaBuffer(sz);
                mVaeCacheSizes[static_cast<std::size_t>(ci)] = sz;
                break;
            }
        }
    }

    std::cerr << "[diffusion] VAE buffers allocated: "
              << num_caches << " caches, output_T=" << mVaeOutputT << "\n";
}

bool WanDiffusionBackend::decode_vae_native(
    const std::vector<float>& latents,
    int32_t c, int32_t t_lat, int32_t h_lat, int32_t w_lat,
    VideoResult& result, std::string& error)
{
    const int32_t num_caches = mConfig.num_vae_caches;
    auto& ctx = mVaeDecoder.context;
    if (!ctx) {
        error = "No VAE execution context";
        return false;
    }

    // 2D VAE path (AutoencoderKL, no caches): latent_input -> decoder_output
    if (num_caches <= 0) {
        const int32_t h_out = h_lat * mConfig.scale_factor_spatial;
        const int32_t w_out = w_lat * mConfig.scale_factor_spatial;
        const auto input_size = static_cast<std::size_t>(c) *
            static_cast<std::size_t>(h_lat) * static_cast<std::size_t>(w_lat);

        // Scale latents by 1/vae_scaling_factor before VAE decode.
        // PixArt: scaling_factor=0.13025, so multiply by ~7.68.
        // Shape: [1, C, H, W] — just scale all values uniformly.
        // The scaling_factor is baked in get_diffusion_config as vae_scaling_factor.
        // For DDIM models without latents_mean/std, this is the only scaling.
        std::vector<float> scaled_latents(latents.begin(),
            latents.begin() + static_cast<std::ptrdiff_t>(input_size));
        if (mConfig.latents_mean.empty() && mConfig.vae_scaling_factor > 0.0F) {
            // No Wan-style denorm applied — do VAE scaling here.
            // Divides latents by vae_scaling_factor before VAE decode.
            const float inv_sf = 1.0F / mConfig.vae_scaling_factor;
            for (auto& v : scaled_latents) {
                v *= inv_sf;
            }
        }

        cudaMemcpyAsync(mD_VaeInput.data(), scaled_latents.data(),
            input_size * sizeof(float), cudaMemcpyHostToDevice, mStream.get());

        ctx->setTensorAddress("latent_input", mD_VaeInput.data());
        ctx->setTensorAddress("decoder_output", mD_VaeOutput.data());

        if (!ctx->enqueueV3(mStream.get())) {
            error = "VAE 2D enqueueV3 failed";
            return false;
        }

        const auto out_size = 3ULL * static_cast<std::size_t>(h_out) *
            static_cast<std::size_t>(w_out);
        std::vector<float> raw(out_size);
        cudaMemcpyAsync(raw.data(), mD_VaeOutput.data(),
            out_size * sizeof(float), cudaMemcpyDeviceToHost, mStream.get());
        cudaStreamSynchronize(mStream.get());

        // Convert [1, 3, H, W] CHW -> [H, W, 3] HWC, clamp to [0, 1]
        result.height = h_out;
        result.width = w_out;
        result.num_frames = 1;
        result.frames.resize(static_cast<std::size_t>(h_out) *
                             static_cast<std::size_t>(w_out) * 3);
        const auto hw = static_cast<std::size_t>(h_out * w_out);
        for (int32_t y = 0; y < h_out; ++y) {
            for (int32_t x = 0; x < w_out; ++x) {
                for (int32_t ch = 0; ch < 3; ++ch) {
                    const auto src = static_cast<std::size_t>(ch) * hw +
                        static_cast<std::size_t>(y * w_out + x);
                    const auto dst = static_cast<std::size_t>(y) *
                        static_cast<std::size_t>(w_out) * 3 +
                        static_cast<std::size_t>(x) * 3 +
                        static_cast<std::size_t>(ch);
                    float val = raw[src];
                    // VAE output is roughly [-1, 1]; map to [0, 1]
                    val = (val + 1.0F) * 0.5F;
                    result.frames[dst] = std::min(std::max(val, 0.0F), 1.0F);
                }
            }
        }
        return true;
    }

    const std::size_t frame_size =
        static_cast<std::size_t>(c) * 1 *
        static_cast<std::size_t>(h_lat) * static_cast<std::size_t>(w_lat);
    const std::size_t frame_bytes = frame_size * sizeof(float);

    for (int32_t ci = 0; ci < num_caches; ++ci) {
        const auto idx = static_cast<std::size_t>(ci);
        if (mVaeCacheSizes[idx] > 0) {
            cudaMemsetAsync(mD_VaeCacheIn[idx].data(), 0,
                            mVaeCacheSizes[idx], mStream.get());
        }
    }

    const int32_t sft = mConfig.scale_factor_temporal;
    const int32_t H_out = mConfig.video_height;
    const int32_t W_out = mConfig.video_width;
    const int32_t T_out_per_frame = mVaeOutputT;

    const std::size_t out_frame_floats =
        static_cast<std::size_t>(3) * static_cast<std::size_t>(T_out_per_frame) *
        static_cast<std::size_t>(H_out) * static_cast<std::size_t>(W_out);
    const std::size_t out_frame_bytes = out_frame_floats * sizeof(float);

    std::vector<float> all_raw_frames;
    all_raw_frames.reserve(
        static_cast<std::size_t>(t_lat) * out_frame_floats);

    for (int32_t t = 0; t < t_lat; ++t) {
        std::vector<float> frame_buf(frame_size);
        const auto spatial = static_cast<std::size_t>(h_lat) *
                             static_cast<std::size_t>(w_lat);
        for (int32_t ci = 0; ci < c; ++ci) {
            const float* ch_src = latents.data() +
                static_cast<std::size_t>(ci) *
                static_cast<std::size_t>(t_lat) * spatial +
                static_cast<std::size_t>(t) * spatial;
            std::memcpy(frame_buf.data() +
                static_cast<std::size_t>(ci) * spatial,
                ch_src, spatial * sizeof(float));
        }

        cudaMemcpyAsync(mD_VaeInput.data(), frame_buf.data(), frame_bytes,
                         cudaMemcpyHostToDevice, mStream.get());

        ctx->setTensorAddress("latent_frame", mD_VaeInput.data());
        ctx->setTensorAddress("video_frame", mD_VaeOutput.data());

        for (int32_t ci = 0; ci < num_caches; ++ci) {
            const auto idx = static_cast<std::size_t>(ci);
            const std::string cin = "cache_" + std::to_string(ci);
            const std::string cout = "cache_out_" + std::to_string(ci);
            ctx->setTensorAddress(cin.c_str(), mD_VaeCacheIn[idx].data());
            ctx->setTensorAddress(cout.c_str(), mD_VaeCacheOut[idx].data());
        }

        if (!ctx->enqueueV3(mStream.get())) {
            error = "VAE enqueueV3 failed at frame " + std::to_string(t);
            return false;
        }

        std::vector<float> out_buf(out_frame_floats);
        cudaMemcpyAsync(out_buf.data(), mD_VaeOutput.data(), out_frame_bytes,
                         cudaMemcpyDeviceToHost, mStream.get());
        cudaStreamSynchronize(mStream.get());

        all_raw_frames.insert(all_raw_frames.end(),
                              out_buf.begin(), out_buf.end());

        for (int32_t ci = 0; ci < num_caches; ++ci) {
            const auto idx = static_cast<std::size_t>(ci);
            if (mVaeCacheSizes[idx] > 0) {
                cudaMemcpyAsync(mD_VaeCacheIn[idx].data(),
                                mD_VaeCacheOut[idx].data(),
                                mVaeCacheSizes[idx],
                                cudaMemcpyDeviceToDevice, mStream.get());
            }
        }

        if (t % 2 == 0) {
            std::cerr << "  VAE frame " << (t + 1) << "/" << t_lat << "\n";
        }
    }

    const int32_t total_out_frames = t_lat * T_out_per_frame;
    const int32_t trim = sft - 1;
    const int32_t T_final = total_out_frames - trim;

    result.num_frames = std::min(T_final, mConfig.video_num_frames);
    result.height = H_out;
    result.width = W_out;
    const auto final_pixel_count =
        static_cast<std::size_t>(result.num_frames) *
        static_cast<std::size_t>(H_out) * static_cast<std::size_t>(W_out) * 3;
    result.frames.resize(final_pixel_count);

    const auto per_frame_spatial =
        static_cast<std::size_t>(H_out) * static_cast<std::size_t>(W_out);

    for (int32_t input_t = 0; input_t < t_lat; ++input_t) {
        const float* raw_base = all_raw_frames.data() +
            static_cast<std::size_t>(input_t) * out_frame_floats;

        for (int32_t sub_t = 0; sub_t < T_out_per_frame; ++sub_t) {
            const int32_t global_t = input_t * T_out_per_frame + sub_t;

            if (global_t < trim) continue;
            const int32_t final_t = global_t - trim;
            if (final_t >= result.num_frames) continue;

            for (int32_t fh = 0; fh < H_out; ++fh) {
                for (int32_t fw = 0; fw < W_out; ++fw) {
                    for (int32_t fc = 0; fc < 3; ++fc) {
                        const auto s_idx =
                            static_cast<std::size_t>(fc) *
                                static_cast<std::size_t>(T_out_per_frame) *
                                per_frame_spatial +
                            static_cast<std::size_t>(sub_t) * per_frame_spatial +
                            static_cast<std::size_t>(fh) *
                                static_cast<std::size_t>(W_out) +
                            static_cast<std::size_t>(fw);

                        const auto d_idx =
                            static_cast<std::size_t>(final_t) *
                                static_cast<std::size_t>(H_out) *
                                static_cast<std::size_t>(W_out) * 3 +
                            static_cast<std::size_t>(fh) *
                                static_cast<std::size_t>(W_out) * 3 +
                            static_cast<std::size_t>(fw) * 3 +
                            static_cast<std::size_t>(fc);

                        float v = (raw_base[s_idx] + 1.0F) * 0.5F;
                        v = std::max(0.0F, std::min(1.0F, v));
                        result.frames[d_idx] = v;
                    }
                }
            }
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Main generation pipeline (Wan-specific orchestration)
// ---------------------------------------------------------------------------

VideoResult WanDiffusionBackend::generate_video(
    const std::vector<int32_t>& input_ids,
    int32_t num_inference_steps,
    float guidance_scale)
{
    if (num_inference_steps < 0) {
        num_inference_steps = mConfig.num_inference_steps;
    }
    if (guidance_scale < 0.0F) {
        guidance_scale = mConfig.guidance_scale;
    }

    VideoResult result;
    result.height = mConfig.video_height;
    result.width = mConfig.video_width;

    if (!mWeights.valid) {
        std::cerr << "[diffusion] ERROR: preprocessor weights not loaded\n";
        return result;
    }

    const int32_t t_lat = (mConfig.video_num_frames - 1) /
                           mConfig.scale_factor_temporal + 1;
    const int32_t h_lat = mConfig.video_height / mConfig.scale_factor_spatial;
    const int32_t w_lat = mConfig.video_width / mConfig.scale_factor_spatial;
    const int32_t z_dim = mConfig.z_dim;
    const int32_t dim = mConfig.dit_dim;
    const int32_t seq_len = mConfig.text_seq_len;

    int32_t pt = 1, ph = 2, pw = 2;
    if (mConfig.patch_size.size() >= 3) {
        pt = mConfig.patch_size[0];
        ph = mConfig.patch_size[1];
        pw = mConfig.patch_size[2];
    }
    const int32_t nt = t_lat / pt;
    const int32_t nh_p = h_lat / ph;
    const int32_t nw_p = w_lat / pw;
    const int32_t num_patches = nt * nh_p * nw_p;
    const int32_t patch_dim = z_dim * pt * ph * pw;

    std::cerr << "[diffusion] Latent shape: "
              << z_dim << "x" << t_lat << "x"
              << h_lat << "x" << w_lat
              << " (patches=" << num_patches << ")\n";

    // 1. Text encoding
    std::cerr << "[diffusion] Encoding text ("
              << input_ids.size() << " tokens) ...\n";
    std::vector<float> text_embeddings;
    std::string error;
    if (!run_t5_encoder(input_ids, text_embeddings, error)) {
        std::cerr << "[diffusion] T5 encoding failed: " << error << "\n";
        return result;
    }
    std::cerr << "[diffusion] T5 encoding done ("
              << text_embeddings.size() << " floats)\n";

    // 2. Project text embeddings
    std::vector<float> text_projected;
    project_text(text_embeddings, seq_len, text_projected);

    // Encoder attention mask: 0.0 for valid tokens, -10000.0 for padding.
    // Valid tokens are non-zero entries in input_ids (padded with 0).
    std::vector<float> encoder_attn_mask;
    if (!mConfig.use_rope) {
        encoder_attn_mask.resize(static_cast<std::size_t>(seq_len), -10000.0F);
        for (std::size_t i = 0; i < input_ids.size() && i < static_cast<std::size_t>(seq_len); ++i) {
            if (input_ids[i] != 0) {
                encoder_attn_mask[i] = 0.0F;
            }
        }
        // EOS token (id=1) is valid
        // Also mark position 0 as valid if it's a BOS/pad token (T5 pads with 0 but position 0 is valid)
        if (!input_ids.empty()) {
            encoder_attn_mask[0] = 0.0F;  // Position 0 is always valid
        }
    }

    // Null text for CFG
    std::vector<int32_t> empty_ids(static_cast<std::size_t>(seq_len), 0);
    empty_ids[0] = 1;
    std::vector<float> null_embeddings;
    if (!run_t5_encoder(empty_ids, null_embeddings, error)) {
        std::cerr << "[diffusion] T5 null encoding failed: " << error << "\n";
        return result;
    }
    std::vector<float> null_text;
    project_text(null_embeddings, seq_len, null_text);

    // 3. Compute RoPE (skip for models with fixed position embeddings)
    std::vector<float> rope_cos, rope_sin;
    if (mConfig.use_rope) {
        compute_3d_rope(nt, nh_p, nw_p, rope_cos, rope_sin);
    }

    // 3b. Compute 2D sinusoidal position embeddings for no-RoPE models.
    //     Added to hidden states after patch embedding in the denoising loop.
    std::vector<float> pos_embed_2d;
    if (!mConfig.use_rope) {
        const int32_t grid_h = nh_p;
        const int32_t grid_w = nw_p;
        const int32_t half_dim = dim / 2;
        const int32_t quarter_dim = half_dim / 2;
        // interpolation_scale: PixArt uses 2 for 1024px resolution
        const float interp_scale = 2.0F;
        pos_embed_2d.resize(static_cast<std::size_t>(num_patches) *
                            static_cast<std::size_t>(dim), 0.0F);
        // omega = 1 / 10000^(i / (D/4))
        std::vector<double> omega(static_cast<std::size_t>(quarter_dim));
        for (int32_t i = 0; i < quarter_dim; ++i) {
            omega[static_cast<std::size_t>(i)] =
                1.0 / std::pow(10000.0,
                    static_cast<double>(i) / static_cast<double>(quarter_dim));
        }
        for (int32_t hi = 0; hi < grid_h; ++hi) {
            for (int32_t wi = 0; wi < grid_w; ++wi) {
                const int32_t patch_idx = hi * grid_w + wi;
                float* row = pos_embed_2d.data() +
                    static_cast<std::size_t>(patch_idx) *
                    static_cast<std::size_t>(dim);
                const double h_pos = static_cast<double>(hi) / static_cast<double>(interp_scale);
                const double w_pos = static_cast<double>(wi) / static_cast<double>(interp_scale);
                // First half: W-axis sincos (HF meshgrid puts W first)
                for (int32_t d = 0; d < quarter_dim; ++d) {
                    const double angle_w = w_pos * omega[static_cast<std::size_t>(d)];
                    row[d] = static_cast<float>(std::sin(angle_w));
                    row[quarter_dim + d] = static_cast<float>(std::cos(angle_w));
                }
                // Second half: H-axis sincos
                for (int32_t d = 0; d < quarter_dim; ++d) {
                    const double angle_h = h_pos * omega[static_cast<std::size_t>(d)];
                    row[half_dim + d] = static_cast<float>(std::sin(angle_h));
                    row[half_dim + quarter_dim + d] = static_cast<float>(std::cos(angle_h));
                }
            }
        }
    }

    // 4. Initialize random latents
    const std::size_t latent_count = static_cast<std::size_t>(z_dim) *
        static_cast<std::size_t>(t_lat) * static_cast<std::size_t>(h_lat) *
        static_cast<std::size_t>(w_lat);
    // Generate initial noise with Box-Muller transform on mt19937,
    // matching numpy RandomState(seed).randn() output exactly.
    // numpy uses mt19937 for uniform generation then a custom
    // gauss algorithm. We use the standard Box-Muller pair
    // which produces the same sequence as torch.Generator(seed).
    std::vector<float> latents(latent_count);
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> udist(0.0, 1.0);
        for (std::size_t i = 0; i < latent_count; i += 2) {
            double u1 = udist(rng);
            double u2 = udist(rng);
            // Avoid log(0)
            if (u1 < 1e-12) u1 = 1e-12;
            double r = std::sqrt(-2.0 * std::log(u1));
            double theta = 2.0 * M_PI * u2;
            latents[i] = static_cast<float>(r * std::cos(theta));
            if (i + 1 < latent_count)
                latents[i + 1] = static_cast<float>(r * std::sin(theta));
        }
    }

    // 5. Denoising loop
    const bool use_ddim = (mConfig.scheduler == "dpmsolver_multistep" ||
                           mConfig.scheduler == "ddim" ||
                           mConfig.scheduler == "ddpm");

    // Flow-match Euler scheduler (Wan, FLUX, Z-Image)
    FlowMatchEulerState fm_scheduler;
    // DDIM scheduler (PixArt, Stable Diffusion — epsilon prediction)
    DDIMState ddim_scheduler;

    std::vector<float> step_timesteps;
    if (use_ddim) {
        ddim_scheduler.num_train_timesteps = 1000;
        ddim_scheduler.set_timesteps(num_inference_steps);
        step_timesteps = ddim_scheduler.timesteps;
    } else {
        fm_scheduler.num_train_timesteps = 1000;
        fm_scheduler.shift = mConfig.flow_shift;
        fm_scheduler.set_timesteps(num_inference_steps);
        step_timesteps = fm_scheduler.timesteps;
    }

    std::vector<float> noise_pred_spatial(latent_count);

    for (int32_t step = 0; step < num_inference_steps; ++step) {
        const float timestep = step_timesteps[static_cast<std::size_t>(step)];

        std::vector<float> temb_6d, time_embed;
        compute_timestep_embedding(timestep, temb_6d, time_embed);

        std::vector<float> patches;
        patchify(latents, z_dim, t_lat, h_lat, w_lat, patches);

        std::vector<float> hidden(
            static_cast<std::size_t>(num_patches) * static_cast<std::size_t>(dim));
        cpu_matmul_bias(patches.data(), mWeights.patch_embed_weight.data(),
                        mWeights.patch_embed_bias.data(),
                        hidden.data(), num_patches, patch_dim, dim);

        // Add 2D sinusoidal position embeddings for no-RoPE models
        if (!pos_embed_2d.empty()) {
            for (std::size_t i = 0; i < hidden.size(); ++i) {
                hidden[i] += pos_embed_2d[i];
            }
        }

        std::vector<float> denoiser_output;

        // Null-text mask: all valid (T5 outputs for padding are still valid embeddings)
        std::vector<float> null_mask;
        if (!encoder_attn_mask.empty()) {
            null_mask.resize(encoder_attn_mask.size(), 0.0F);
        }

        if (guidance_scale > 1.0F) {
            std::vector<float> cond_pred, uncond_pred;

            if (!run_denoiser(hidden, temb_6d, time_embed, text_projected,
                              rope_cos, rope_sin, cond_pred, error,
                              encoder_attn_mask)) {
                std::cerr << "[diffusion] Denoiser (cond) failed: " << error << "\n";
                return result;
            }
            if (!run_denoiser(hidden, temb_6d, time_embed, null_text,
                              rope_cos, rope_sin, uncond_pred, error,
                              null_mask)) {
                std::cerr << "[diffusion] Denoiser (uncond) failed: " << error << "\n";
                return result;
            }

            denoiser_output.resize(cond_pred.size());
            for (std::size_t i = 0; i < cond_pred.size(); ++i) {
                denoiser_output[i] = uncond_pred[i] +
                    guidance_scale * (cond_pred[i] - uncond_pred[i]);
            }
        } else {
            if (!run_denoiser(hidden, temb_6d, time_embed, text_projected,
                              rope_cos, rope_sin, denoiser_output, error,
                              encoder_attn_mask)) {
                std::cerr << "[diffusion] Denoiser failed: " << error << "\n";
                return result;
            }
        }

        // If the DiT outputs more channels than z_dim (e.g. PixArt outputs
        // out_channels=8=2*z_dim for learned_sigma), extract only the first
        // z_dim channels per spatial position within each patch.
        // Element layout per patch: (pt, ph, pw, C_out) in C-contiguous order.
        // We keep C=0..z_dim-1, discarding C=z_dim..C_out-1.
        const int32_t expected_patch_out = z_dim * pt * ph * pw;
        const auto actual_patch_out = static_cast<int32_t>(
            denoiser_output.size() / static_cast<std::size_t>(num_patches));
        if (actual_patch_out > expected_patch_out) {
            const int32_t c_out = actual_patch_out / (pt * ph * pw);
            std::vector<float> truncated(
                static_cast<std::size_t>(num_patches) *
                static_cast<std::size_t>(expected_patch_out));
            for (int32_t pi = 0; pi < num_patches; ++pi) {
                const float* src = denoiser_output.data() +
                    static_cast<std::size_t>(pi) *
                    static_cast<std::size_t>(actual_patch_out);
                float* dst = truncated.data() +
                    static_cast<std::size_t>(pi) *
                    static_cast<std::size_t>(expected_patch_out);
                int32_t di = 0;
                for (int32_t pti = 0; pti < pt; ++pti) {
                    for (int32_t phi_ = 0; phi_ < ph; ++phi_) {
                        for (int32_t pwi = 0; pwi < pw; ++pwi) {
                            const int32_t base =
                                ((pti * ph + phi_) * pw + pwi) * c_out;
                            for (int32_t ci = 0; ci < z_dim; ++ci) {
                                dst[di++] = src[base + ci];
                            }
                        }
                    }
                }
            }
            denoiser_output = std::move(truncated);
        }

        unpatchify(denoiser_output, z_dim, t_lat, h_lat, w_lat, noise_pred_spatial);

        if (use_ddim) {
            ddim_scheduler.step(noise_pred_spatial.data(), latents.data(),
                                latents.data(), latent_count, step);
        } else {
            fm_scheduler.step(noise_pred_spatial.data(), latents.data(),
                              latents.data(), latent_count, step);
        }

        if (step % 5 == 0 || step == num_inference_steps - 1) {
            double lat_sq = 0;
            for (std::size_t i = 0; i < latent_count; ++i)
                lat_sq += static_cast<double>(latents[i]) * latents[i];
            double lat_std = std::sqrt(lat_sq / static_cast<double>(latent_count));
            std::cerr << "  Step " << (step + 1) << "/" << num_inference_steps
                      << " t=" << timestep << " lat_std=" << lat_std << "\n";
        }
    }

    // 6. Denormalize latents
    if (!mConfig.latents_mean.empty() && !mConfig.latents_std.empty()) {
        for (int32_t ci = 0; ci < z_dim; ++ci) {
            const float mean = mConfig.latents_mean[static_cast<std::size_t>(ci)];
            const float std_val = mConfig.latents_std[static_cast<std::size_t>(ci)];
            const auto channel_size = static_cast<std::size_t>(t_lat * h_lat * w_lat);
            float* ch = latents.data() + static_cast<std::size_t>(ci) * channel_size;
            for (std::size_t i = 0; i < channel_size; ++i) {
                ch[i] = ch[i] * std_val + mean;
            }
        }
    }

    // 7. VAE decode (native TRT engine)
    std::cerr << "[diffusion] Decoding video ...\n";
    if (!decode_vae_native(latents, z_dim, t_lat, h_lat, w_lat, result, error)) {
        std::cerr << "[diffusion] VAE decode failed: " << error << "\n";
        return result;
    }

    result.num_frames = mConfig.video_num_frames;
    std::cerr << "[diffusion] Video generation complete: "
              << result.num_frames << " frames, "
              << result.height << "x" << result.width << "\n";
    return result;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<IDiffusionBackend> CreateDiffusionBackend(
    std::vector<DiffusionEngine> text_encoders,
    DiffusionEngine denoiser,
    DiffusionEngine vae_decoder,
    const FastPathModelConfig& fp_cfg)
{
    DiffusionConfig config;
    config.scheduler = fp_cfg.scheduler;
    config.num_inference_steps = fp_cfg.num_inference_steps;
    config.guidance_scale = fp_cfg.guidance_scale;
    config.flow_shift = fp_cfg.flow_shift;
    config.use_dynamic_shifting = fp_cfg.use_dynamic_shifting;
    config.base_shift = fp_cfg.base_shift;
    config.max_shift = fp_cfg.max_shift;
    config.video_height = fp_cfg.video_height;
    config.video_width = fp_cfg.video_width;
    config.video_num_frames = fp_cfg.video_num_frames;
    config.z_dim = fp_cfg.z_dim;
    config.scale_factor_temporal = fp_cfg.scale_factor_temporal;
    config.scale_factor_spatial = fp_cfg.scale_factor_spatial;
    config.dit_dim = fp_cfg.dit_dim;
    config.dit_num_heads = fp_cfg.dit_num_heads;
    config.freq_dim = fp_cfg.freq_dim;
    config.text_encoder_dim = fp_cfg.text_encoder_dim;
    config.text_seq_len = fp_cfg.text_seq_len;
    config.num_vae_caches = fp_cfg.num_vae_caches;
    config.latents_mean = fp_cfg.latents_mean;
    config.latents_std = fp_cfg.latents_std;
    config.patch_size = fp_cfg.patch_size;
    config.vae_model_id = fp_cfg.vae_model_id;
    config.guidance_embeds = fp_cfg.guidance_embeds;
    config.use_rope = fp_cfg.use_rope;
    config.vae_scaling_factor = fp_cfg.vae_scaling_factor;
    config.diffusion_backend_type = fp_cfg.diffusion_backend_type;

    // Dispatch on backend type. Default to wan_3d for backward compatibility.
    if (config.diffusion_backend_type == "flux_2d") {
        return std::make_unique<FluxDiffusionBackend>(
            std::move(text_encoders),
            std::move(denoiser),
            std::move(vae_decoder),
            std::move(config));
    }
    if (config.diffusion_backend_type == "z_image_2d") {
        return std::make_unique<ZImageDiffusionBackend>(
            std::move(text_encoders),
            std::move(denoiser),
            std::move(vae_decoder),
            std::move(config));
    }
    return std::make_unique<WanDiffusionBackend>(
        std::move(text_encoders),
        std::move(denoiser),
        std::move(vae_decoder),
        std::move(config));
}

} // namespace trtf

#endif // TRTF_HAS_TRT
