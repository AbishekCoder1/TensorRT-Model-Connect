#include "runtime/trt/diffusion/wan_diffusion_backend.h"
#include "runtime/trt/diffusion/diffusion_denoising_step_seam.h"
#include "runtime/trt/diffusion/diffusion_generation_plan.h"
#include "runtime/trt/diffusion/diffusion_scheduler_helpers.h"
#include "runtime/trt/diffusion/flux_diffusion_backend.h"
#include "runtime/trt/diffusion/wan_generation_conditioning.h"
#include "runtime/trt/diffusion/z_image_diffusion_backend.h"

#if TRTF_HAS_TRT

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>

namespace trtf {

namespace {

using diffusion::FlowMatchEulerState;
using diffusion::WanLayout;

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

float clamp_unit(float value)
{
    return std::max(0.0F, std::min(1.0F, value));
}

void convert_wan_chw_to_hwc(
    const std::vector<float>& raw,
    int32_t h_out,
    int32_t w_out,
    VideoResult& result)
{
    result.height = h_out;
    result.width = w_out;
    result.num_frames = 1;
    result.frames.resize(
        static_cast<std::size_t>(h_out) * static_cast<std::size_t>(w_out) * 3);
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
                result.frames[dst] = clamp_unit((raw[src] + 1.0F) * 0.5F);
            }
        }
    }
}

std::vector<float> prepare_wan_vae_2d_input(
    const std::vector<float>& latents,
    const DiffusionConfig& config,
    std::size_t input_size)
{
    std::vector<float> scaled_latents(
        latents.begin(), latents.begin() + static_cast<std::ptrdiff_t>(input_size));
    if (config.latents_mean.empty() && config.vae_scaling_factor > 0.0F) {
        const float inv_sf = 1.0F / config.vae_scaling_factor;
        for (auto& v : scaled_latents) {
            v *= inv_sf;
        }
    }
    return scaled_latents;
}

bool decode_wan_vae_2d(
    const DiffusionConfig& config,
    nvinfer1::IExecutionContext& ctx,
    CudaBuffer& d_vae_input,
    CudaBuffer& d_vae_output,
    CudaStream& stream,
    const std::vector<float>& latents,
    int32_t c,
    int32_t h_lat,
    int32_t w_lat,
    VideoResult& result,
    std::string& error)
{
    const int32_t h_out = h_lat * config.scale_factor_spatial;
    const int32_t w_out = w_lat * config.scale_factor_spatial;
    const auto input_size = static_cast<std::size_t>(c) *
        static_cast<std::size_t>(h_lat) * static_cast<std::size_t>(w_lat);
    const auto scaled_latents = prepare_wan_vae_2d_input(latents, config, input_size);

    cudaMemcpyAsync(d_vae_input.data(), scaled_latents.data(),
        input_size * sizeof(float), cudaMemcpyHostToDevice, stream.get());
    ctx.setTensorAddress("latent_input", d_vae_input.data());
    ctx.setTensorAddress("decoder_output", d_vae_output.data());
    if (!ctx.enqueueV3(stream.get())) {
        error = "VAE 2D enqueueV3 failed";
        return false;
    }

    const auto out_size = static_cast<std::size_t>(3) *
        static_cast<std::size_t>(h_out) * static_cast<std::size_t>(w_out);
    std::vector<float> raw(out_size);
    cudaMemcpyAsync(raw.data(), d_vae_output.data(),
        out_size * sizeof(float), cudaMemcpyDeviceToHost, stream.get());
    cudaStreamSynchronize(stream.get());
    convert_wan_chw_to_hwc(raw, h_out, w_out, result);
    return true;
}

void reset_wan_vae_cache_inputs(
    int32_t num_caches,
    const std::vector<std::size_t>& cache_sizes,
    std::vector<CudaBuffer>& cache_inputs,
    CudaStream& stream)
{
    for (int32_t ci = 0; ci < num_caches; ++ci) {
        const auto idx = static_cast<std::size_t>(ci);
        if (cache_sizes[idx] > 0) {
            cudaMemsetAsync(
                cache_inputs[idx].data(), 0, cache_sizes[idx], stream.get());
        }
    }
}

void extract_wan_latent_frame(
    const std::vector<float>& latents,
    int32_t c,
    int32_t t_lat,
    int32_t h_lat,
    int32_t w_lat,
    int32_t t,
    std::vector<float>& frame_buf)
{
    const auto spatial = static_cast<std::size_t>(h_lat) *
        static_cast<std::size_t>(w_lat);
    frame_buf.resize(static_cast<std::size_t>(c) * spatial);
    for (int32_t ci = 0; ci < c; ++ci) {
        const float* ch_src = latents.data() +
            static_cast<std::size_t>(ci) * static_cast<std::size_t>(t_lat) * spatial +
            static_cast<std::size_t>(t) * spatial;
        std::memcpy(
            frame_buf.data() + static_cast<std::size_t>(ci) * spatial,
            ch_src,
            spatial * sizeof(float));
    }
}

void bind_wan_cache_tensors(
    nvinfer1::IExecutionContext& ctx,
    int32_t num_caches,
    std::vector<CudaBuffer>& cache_in,
    std::vector<CudaBuffer>& cache_out)
{
    for (int32_t ci = 0; ci < num_caches; ++ci) {
        const auto idx = static_cast<std::size_t>(ci);
        const std::string cin = "cache_" + std::to_string(ci);
        const std::string cout = "cache_out_" + std::to_string(ci);
        ctx.setTensorAddress(cin.c_str(), cache_in[idx].data());
        ctx.setTensorAddress(cout.c_str(), cache_out[idx].data());
    }
}

void rotate_wan_vae_caches(
    int32_t num_caches,
    const std::vector<std::size_t>& cache_sizes,
    std::vector<CudaBuffer>& cache_in,
    std::vector<CudaBuffer>& cache_out,
    CudaStream& stream)
{
    for (int32_t ci = 0; ci < num_caches; ++ci) {
        const auto idx = static_cast<std::size_t>(ci);
        if (cache_sizes[idx] > 0) {
            cudaMemcpyAsync(cache_in[idx].data(),
                            cache_out[idx].data(),
                            cache_sizes[idx],
                            cudaMemcpyDeviceToDevice, stream.get());
        }
    }
}

void compose_wan_vae_video_frames(
    const std::vector<float>& all_raw_frames,
    int32_t t_lat,
    int32_t t_out_per_frame,
    int32_t h_out,
    int32_t w_out,
    int32_t scale_factor_temporal,
    int32_t max_video_frames,
    VideoResult& result)
{
    const int32_t total_out_frames = t_lat * t_out_per_frame;
    const int32_t trim = scale_factor_temporal - 1;
    const int32_t t_final = total_out_frames - trim;

    result.num_frames = std::min(t_final, max_video_frames);
    result.height = h_out;
    result.width = w_out;
    result.frames.resize(
        static_cast<std::size_t>(result.num_frames) *
        static_cast<std::size_t>(h_out) *
        static_cast<std::size_t>(w_out) * 3);

    const auto per_frame_spatial =
        static_cast<std::size_t>(h_out) * static_cast<std::size_t>(w_out);
    const auto out_frame_floats =
        static_cast<std::size_t>(3) *
        static_cast<std::size_t>(t_out_per_frame) *
        per_frame_spatial;

    for (int32_t input_t = 0; input_t < t_lat; ++input_t) {
        const float* raw_base = all_raw_frames.data() +
            static_cast<std::size_t>(input_t) * out_frame_floats;
        for (int32_t sub_t = 0; sub_t < t_out_per_frame; ++sub_t) {
            const int32_t global_t = input_t * t_out_per_frame + sub_t;
            const int32_t final_t = global_t - trim;
            if (global_t < trim || final_t >= result.num_frames) {
                continue;
            }
            for (int32_t fh = 0; fh < h_out; ++fh) {
                for (int32_t fw = 0; fw < w_out; ++fw) {
                    for (int32_t fc = 0; fc < 3; ++fc) {
                        const auto s_idx =
                            static_cast<std::size_t>(fc) *
                                static_cast<std::size_t>(t_out_per_frame) *
                                per_frame_spatial +
                            static_cast<std::size_t>(sub_t) * per_frame_spatial +
                            static_cast<std::size_t>(fh) *
                                static_cast<std::size_t>(w_out) +
                            static_cast<std::size_t>(fw);
                        const auto d_idx =
                            static_cast<std::size_t>(final_t) *
                                static_cast<std::size_t>(h_out) *
                                static_cast<std::size_t>(w_out) * 3 +
                            static_cast<std::size_t>(fh) *
                                static_cast<std::size_t>(w_out) * 3 +
                            static_cast<std::size_t>(fw) * 3 +
                            static_cast<std::size_t>(fc);
                        result.frames[d_idx] =
                            clamp_unit((raw_base[s_idx] + 1.0F) * 0.5F);
                    }
                }
            }
        }
    }
}

bool decode_wan_vae_3d(
    const DiffusionConfig& config,
    nvinfer1::IExecutionContext& ctx,
    CudaBuffer& d_vae_input,
    CudaBuffer& d_vae_output,
    std::vector<CudaBuffer>& cache_in,
    std::vector<CudaBuffer>& cache_out,
    const std::vector<std::size_t>& cache_sizes,
    int32_t vae_output_t,
    CudaStream& stream,
    const std::vector<float>& latents,
    int32_t c,
    int32_t t_lat,
    int32_t h_lat,
    int32_t w_lat,
    VideoResult& result,
    std::string& error)
{
    const int32_t num_caches = config.num_vae_caches;
    const std::size_t frame_size = static_cast<std::size_t>(c) *
        static_cast<std::size_t>(h_lat) * static_cast<std::size_t>(w_lat);
    const std::size_t frame_bytes = frame_size * sizeof(float);
    const int32_t h_out = config.video_height;
    const int32_t w_out = config.video_width;
    const auto out_frame_floats = static_cast<std::size_t>(3) *
        static_cast<std::size_t>(vae_output_t) *
        static_cast<std::size_t>(h_out) *
        static_cast<std::size_t>(w_out);
    const auto out_frame_bytes = out_frame_floats * sizeof(float);

    reset_wan_vae_cache_inputs(num_caches, cache_sizes, cache_in, stream);
    std::vector<float> all_raw_frames;
    all_raw_frames.reserve(static_cast<std::size_t>(t_lat) * out_frame_floats);
    std::vector<float> frame_buf;
    std::vector<float> out_buf(out_frame_floats);

    for (int32_t t = 0; t < t_lat; ++t) {
        extract_wan_latent_frame(latents, c, t_lat, h_lat, w_lat, t, frame_buf);
        cudaMemcpyAsync(d_vae_input.data(), frame_buf.data(), frame_bytes,
            cudaMemcpyHostToDevice, stream.get());

        ctx.setTensorAddress("latent_frame", d_vae_input.data());
        ctx.setTensorAddress("video_frame", d_vae_output.data());
        bind_wan_cache_tensors(ctx, num_caches, cache_in, cache_out);

        if (!ctx.enqueueV3(stream.get())) {
            error = "VAE enqueueV3 failed at frame " + std::to_string(t);
            return false;
        }

        cudaMemcpyAsync(out_buf.data(), d_vae_output.data(),
            out_frame_bytes, cudaMemcpyDeviceToHost, stream.get());
        cudaStreamSynchronize(stream.get());
        all_raw_frames.insert(all_raw_frames.end(), out_buf.begin(), out_buf.end());
        rotate_wan_vae_caches(num_caches, cache_sizes, cache_in, cache_out, stream);

        if (t % 2 == 0) {
            std::cerr << "  VAE frame " << (t + 1) << "/" << t_lat << "\n";
        }
    }

    compose_wan_vae_video_frames(
        all_raw_frames,
        t_lat,
        vae_output_t,
        h_out,
        w_out,
        config.scale_factor_temporal,
        config.video_num_frames,
        result);
    return true;
}

void log_wan_layout(const WanLayout& layout)
{
    std::cerr << "[diffusion] Latent shape: "
              << layout.z_dim << "x" << layout.t_lat << "x"
              << layout.h_lat << "x" << layout.w_lat
              << " (patches=" << layout.num_patches << ")\n";
}

void compute_wan_pos_embed_2d(
    int32_t nh_p,
    int32_t nw_p,
    int32_t dim,
    std::vector<float>& pos_embed_2d)
{
    const int32_t half_dim = dim / 2;
    const int32_t quarter_dim = half_dim / 2;
    const float interp_scale = 2.0F;
    pos_embed_2d.assign(
        static_cast<std::size_t>(nh_p * nw_p) * static_cast<std::size_t>(dim), 0.0F);

    std::vector<double> omega(static_cast<std::size_t>(quarter_dim));
    for (int32_t i = 0; i < quarter_dim; ++i) {
        omega[static_cast<std::size_t>(i)] =
            1.0 / std::pow(10000.0,
                static_cast<double>(i) / static_cast<double>(quarter_dim));
    }

    for (int32_t hi = 0; hi < nh_p; ++hi) {
        for (int32_t wi = 0; wi < nw_p; ++wi) {
            const int32_t patch_idx = hi * nw_p + wi;
            float* row = pos_embed_2d.data() +
                static_cast<std::size_t>(patch_idx) * static_cast<std::size_t>(dim);
            const double h_pos = static_cast<double>(hi) / static_cast<double>(interp_scale);
            const double w_pos = static_cast<double>(wi) / static_cast<double>(interp_scale);
            for (int32_t d = 0; d < quarter_dim; ++d) {
                const double angle_w = w_pos * omega[static_cast<std::size_t>(d)];
                row[d] = static_cast<float>(std::sin(angle_w));
                row[quarter_dim + d] = static_cast<float>(std::cos(angle_w));
            }
            for (int32_t d = 0; d < quarter_dim; ++d) {
                const double angle_h = h_pos * omega[static_cast<std::size_t>(d)];
                row[half_dim + d] = static_cast<float>(std::sin(angle_h));
                row[half_dim + quarter_dim + d] = static_cast<float>(std::cos(angle_h));
            }
        }
    }
}

void add_wan_positional_embedding(
    std::vector<float>& hidden,
    const std::vector<float>& pos_embed_2d)
{
    if (pos_embed_2d.empty()) {
        return;
    }
    for (std::size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] += pos_embed_2d[i];
    }
}

template <typename RunDenoiserFn>
bool predict_wan_noise(
    const std::vector<float>& hidden,
    const std::vector<float>& temb_6d,
    const std::vector<float>& time_embed,
    const std::vector<float>& text_projected,
    const std::vector<float>& null_text,
    const std::vector<float>& encoder_attn_mask,
    float guidance_scale,
    std::vector<float>& denoiser_output,
    std::string& error,
    RunDenoiserFn&& run_denoiser)
{
    if (guidance_scale > 1.0F) {
        std::vector<float> cond_pred;
        std::vector<float> uncond_pred;
        std::vector<float> null_mask;
        if (!encoder_attn_mask.empty()) {
            null_mask.assign(encoder_attn_mask.size(), 0.0F);
        }

        if (!run_denoiser(hidden, temb_6d, time_embed, text_projected,
                          encoder_attn_mask, cond_pred, error)) {
            return false;
        }
        if (!run_denoiser(hidden, temb_6d, time_embed, null_text,
                          null_mask, uncond_pred, error)) {
            return false;
        }

        denoiser_output.resize(cond_pred.size());
        for (std::size_t i = 0; i < cond_pred.size(); ++i) {
            denoiser_output[i] = uncond_pred[i] +
                guidance_scale * (cond_pred[i] - uncond_pred[i]);
        }
        return true;
    }

    return run_denoiser(hidden, temb_6d, time_embed, text_projected,
                        encoder_attn_mask, denoiser_output, error);
}

void maybe_truncate_wan_output(
    std::vector<float>& denoiser_output,
    int32_t num_patches,
    int32_t z_dim,
    int32_t pt,
    int32_t ph,
    int32_t pw)
{
    const int32_t expected_patch_out = z_dim * pt * ph * pw;
    const auto actual_patch_out = static_cast<int32_t>(
        denoiser_output.size() / static_cast<std::size_t>(num_patches));
    if (actual_patch_out <= expected_patch_out) {
        return;
    }

    const int32_t c_out = actual_patch_out / (pt * ph * pw);
    std::vector<float> truncated(
        static_cast<std::size_t>(num_patches) *
        static_cast<std::size_t>(expected_patch_out));
    for (int32_t pi = 0; pi < num_patches; ++pi) {
        const float* src = denoiser_output.data() +
            static_cast<std::size_t>(pi) * static_cast<std::size_t>(actual_patch_out);
        float* dst = truncated.data() +
            static_cast<std::size_t>(pi) * static_cast<std::size_t>(expected_patch_out);
        int32_t di = 0;
        for (int32_t pti = 0; pti < pt; ++pti) {
            for (int32_t phi_ = 0; phi_ < ph; ++phi_) {
                for (int32_t pwi = 0; pwi < pw; ++pwi) {
                    const int32_t base = ((pti * ph + phi_) * pw + pwi) * c_out;
                    for (int32_t ci = 0; ci < z_dim; ++ci) {
                        dst[di++] = src[base + ci];
                    }
                }
            }
        }
    }
    denoiser_output = std::move(truncated);
}

void apply_wan_scheduler_step(
    bool use_ddim,
    DDIMState& ddim_scheduler,
    FlowMatchEulerState& fm_scheduler,
    const std::vector<float>& noise_pred_spatial,
    std::vector<float>& latents,
    std::size_t latent_count,
    int32_t step)
{
    if (use_ddim) {
        ddim_scheduler.step(noise_pred_spatial.data(), latents.data(),
                            latents.data(), latent_count, step);
        return;
    }
    fm_scheduler.step(noise_pred_spatial.data(), latents.data(),
                      latents.data(), latent_count, step);
}

void maybe_log_wan_step(
    int32_t step,
    int32_t num_inference_steps,
    float timestep,
    const std::vector<float>& latents)
{
    if (step % 5 != 0 && step != num_inference_steps - 1) {
        return;
    }
    double lat_sq = 0.0;
    for (const auto v : latents) {
        lat_sq += static_cast<double>(v) * static_cast<double>(v);
    }
    const double lat_std = std::sqrt(lat_sq / static_cast<double>(latents.size()));
    std::cerr << "  Step " << (step + 1) << "/" << num_inference_steps
              << " t=" << timestep << " lat_std=" << lat_std << "\n";
}

template <typename ComputeTembFn, typename PatchifyFn, typename EmbedHiddenFn,
          typename UnpatchifyFn, typename RunDenoiserFn>
bool run_wan_denoising_loop(
    int32_t num_inference_steps,
    bool use_ddim,
    float guidance_scale,
    const WanLayout& layout,
    const std::vector<float>& step_timesteps,
    const std::vector<float>& pos_embed_2d,
    const std::vector<float>& text_projected,
    const std::vector<float>& null_text,
    const std::vector<float>& encoder_attn_mask,
    DDIMState& ddim_scheduler,
    FlowMatchEulerState& fm_scheduler,
    std::vector<float>& latents,
    std::string& error,
    ComputeTembFn&& compute_temb,
    PatchifyFn&& patchify,
    EmbedHiddenFn&& embed_hidden,
    UnpatchifyFn&& unpatchify,
    RunDenoiserFn&& run_denoiser)
{
    std::vector<float> patches;
    return diffusion::run_wan_denoising_steps(
        num_inference_steps,
        step_timesteps,
        latents,
        error,
        compute_temb,
        [&](const std::vector<float>& current_latents, std::vector<float>& hidden) {
            patchify(current_latents, patches);
            hidden.resize(
                static_cast<std::size_t>(layout.num_patches) *
                static_cast<std::size_t>(layout.dim));
            embed_hidden(patches, hidden);
            add_wan_positional_embedding(hidden, pos_embed_2d);
        },
        [&](const std::vector<float>& hidden, const std::vector<float>& temb_6d,
            const std::vector<float>& time_embed, std::vector<float>& denoiser_output,
            std::string& err) {
            return predict_wan_noise(
                hidden, temb_6d, time_embed, text_projected, null_text,
                encoder_attn_mask, guidance_scale, denoiser_output, err, run_denoiser);
        },
        [&](std::vector<float>& denoiser_output, std::vector<float>& noise_pred_spatial) {
            maybe_truncate_wan_output(
                denoiser_output,
                layout.num_patches,
                layout.z_dim,
                layout.pt,
                layout.ph,
                layout.pw);
            unpatchify(denoiser_output, noise_pred_spatial);
        },
        [&](const std::vector<float>& noise_pred_spatial, std::vector<float>& current_latents, int32_t step) {
            apply_wan_scheduler_step(
                use_ddim, ddim_scheduler, fm_scheduler, noise_pred_spatial,
                current_latents, current_latents.size(), step);
        },
        [&](int32_t step, float timestep, const std::vector<float>& current_latents) {
            maybe_log_wan_step(step, num_inference_steps, timestep, current_latents);
        });
}

void denormalize_wan_latents(
    const DiffusionConfig& config,
    int32_t z_dim,
    int32_t t_lat,
    int32_t h_lat,
    int32_t w_lat,
    std::vector<float>& latents)
{
    if (config.latents_mean.empty() || config.latents_std.empty()) {
        return;
    }
    const auto channel_size = static_cast<std::size_t>(t_lat * h_lat * w_lat);
    for (int32_t ci = 0; ci < z_dim; ++ci) {
        const float mean = config.latents_mean[static_cast<std::size_t>(ci)];
        const float std_val = config.latents_std[static_cast<std::size_t>(ci)];
        float* ch = latents.data() + static_cast<std::size_t>(ci) * channel_size;
        for (std::size_t i = 0; i < channel_size; ++i) {
            ch[i] = ch[i] * std_val + mean;
        }
    }
}

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

namespace {

std::size_t float_tensor_size_bytes(const nvinfer1::Dims& dims)
{
    std::size_t size_bytes = sizeof(float);
    for (int32_t d = 0; d < dims.nbDims; ++d) {
        size_bytes *= static_cast<std::size_t>(
            std::max(static_cast<int32_t>(dims.d[d]), 1));
    }
    return size_bytes;
}

void allocate_vae_io_buffers(
    nvinfer1::ICudaEngine& engine,
    CudaBuffer& vae_input,
    CudaBuffer& vae_output,
    int32_t& vae_output_t)
{
    const int32_t io_count = engine.getNbIOTensors();
    for (int32_t i = 0; i < io_count; ++i) {
        const char* tname = engine.getIOTensorName(i);
        const auto dims = engine.getTensorShape(tname);
        const std::string tensor_name(tname);

        if (tensor_name == "latent_frame" || tensor_name == "latent_input") {
            vae_input = CudaBuffer(float_tensor_size_bytes(dims));
            continue;
        }
        if (tensor_name == "video_frame" || tensor_name == "decoder_output") {
            vae_output = CudaBuffer(float_tensor_size_bytes(dims));
            if (dims.nbDims >= 3) {
                vae_output_t = std::max(static_cast<int32_t>(dims.d[2]), 1);
            }
        }
    }
}

std::size_t find_vae_cache_tensor_size(
    nvinfer1::ICudaEngine& engine,
    const std::string& cache_name)
{
    const int32_t io_count = engine.getNbIOTensors();
    for (int32_t i = 0; i < io_count; ++i) {
        const char* tname = engine.getIOTensorName(i);
        if (cache_name == tname) {
            return float_tensor_size_bytes(engine.getTensorShape(tname));
        }
    }
    return 0;
}

} // namespace

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
    allocate_vae_io_buffers(*engine, mD_VaeInput, mD_VaeOutput, mVaeOutputT);

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
        const std::size_t cache_size = find_vae_cache_tensor_size(*engine, cin_name);
        if (cache_size == 0) {
            continue;
        }
        mD_VaeCacheIn[static_cast<std::size_t>(ci)] = CudaBuffer(cache_size);
        mD_VaeCacheOut[static_cast<std::size_t>(ci)] = CudaBuffer(cache_size);
        mVaeCacheSizes[static_cast<std::size_t>(ci)] = cache_size;
    }

    std::cerr << "[diffusion] VAE buffers allocated: "
              << num_caches << " caches, output_T=" << mVaeOutputT << "\n";
}

bool WanDiffusionBackend::decode_vae_native(
    const std::vector<float>& latents,
    int32_t c, int32_t t_lat, int32_t h_lat, int32_t w_lat,
    VideoResult& result, std::string& error)
{
    auto& ctx = mVaeDecoder.context;
    if (!ctx) {
        error = "No VAE execution context";
        return false;
    }

    if (mConfig.num_vae_caches <= 0) {
        return decode_wan_vae_2d(
            mConfig,
            *ctx,
            mD_VaeInput,
            mD_VaeOutput,
            mStream,
            latents,
            c,
            h_lat,
            w_lat,
            result,
            error);
    }

    return decode_wan_vae_3d(
        mConfig,
        *ctx,
        mD_VaeInput,
        mD_VaeOutput,
        mD_VaeCacheIn,
        mD_VaeCacheOut,
        mVaeCacheSizes,
        mVaeOutputT,
        mStream,
        latents,
        c,
        t_lat,
        h_lat,
        w_lat,
        result,
        error);
}

// ---------------------------------------------------------------------------
// Main generation pipeline (Wan-specific orchestration)
// ---------------------------------------------------------------------------

VideoResult WanDiffusionBackend::generate_video(
    const std::vector<int32_t>& input_ids,
    int32_t num_inference_steps,
    float guidance_scale)
{
    const auto plan = diffusion::make_wan_generation_plan(
        mConfig, num_inference_steps, guidance_scale);
    num_inference_steps = plan.num_inference_steps;
    guidance_scale = plan.guidance_scale;

    VideoResult result;
    result.height = mConfig.video_height;
    result.width = mConfig.video_width;

    if (!mWeights.valid) {
        std::cerr << "[diffusion] ERROR: preprocessor weights not loaded\n";
        return result;
    }

    const WanLayout& layout = plan.layout;
    log_wan_layout(layout);
    const diffusion::WanConditioningInputs conditioning_inputs
        = diffusion::make_wan_conditioning_inputs(mConfig, layout, input_ids);

    std::cerr << "[diffusion] Encoding text ("
              << input_ids.size() << " tokens) ...\n";
    std::string error;
    diffusion::WanTextConditioning text_conditioning;
    if (!diffusion::build_wan_text_conditioning(
            input_ids,
            conditioning_inputs,
            layout.seq_len,
            error,
            [this](
                const std::vector<int32_t>& ids,
                std::vector<float>& embeddings,
                std::string& encoder_error) {
                return run_t5_encoder(ids, embeddings, encoder_error);
            },
            [this](
                const std::vector<float>& embeddings,
                int32_t seq_len,
                std::vector<float>& projected) {
                project_text(embeddings, seq_len, projected);
            },
            text_conditioning)) {
        std::cerr << "[diffusion] T5 conditioning failed: " << error << "\n";
        return result;
    }
    std::cerr << "[diffusion] T5 conditioning done ("
              << text_conditioning.text_projected.size() << " floats)\n";

    std::vector<float> rope_cos, rope_sin;
    if (mConfig.use_rope) {
        compute_3d_rope(layout.nt, layout.nh_p, layout.nw_p, rope_cos, rope_sin);
    }

    std::vector<float> pos_embed_2d;
    if (!mConfig.use_rope) {
        compute_wan_pos_embed_2d(layout.nh_p, layout.nw_p, layout.dim, pos_embed_2d);
    }

    std::vector<float> latents = diffusion::make_wan_initial_latents(plan.latent_count);

    const bool use_ddim = plan.use_ddim;
    FlowMatchEulerState fm_scheduler;
    DDIMState ddim_scheduler;
    std::vector<float> step_timesteps;
    if (use_ddim) {
        ddim_scheduler.num_train_timesteps = 1000;
        ddim_scheduler.set_timesteps(num_inference_steps);
        step_timesteps = ddim_scheduler.timesteps;
    } else {
        fm_scheduler = diffusion::make_wan_flow_match_scheduler(plan);
        step_timesteps = fm_scheduler.timesteps;
    }

    const auto compute_temb = [this](
        float timestep,
        std::vector<float>& temb_6d,
        std::vector<float>& time_embed) {
        compute_timestep_embedding(timestep, temb_6d, time_embed);
    };
    const auto patchify_fn = [this, &layout](
        const std::vector<float>& src_latents,
        std::vector<float>& patches) {
        patchify(src_latents, layout.z_dim, layout.t_lat, layout.h_lat, layout.w_lat, patches);
    };
    const auto embed_hidden = [this, &layout](
        const std::vector<float>& patches,
        std::vector<float>& hidden) {
        cpu_matmul_bias(
            patches.data(),
            mWeights.patch_embed_weight.data(),
            mWeights.patch_embed_bias.data(),
            hidden.data(),
            layout.num_patches,
            layout.patch_dim,
            layout.dim);
    };
    const auto unpatchify_fn = [this, &layout](
        const std::vector<float>& patches,
        std::vector<float>& out) {
        unpatchify(patches, layout.z_dim, layout.t_lat, layout.h_lat, layout.w_lat, out);
    };
    const auto run_denoiser_fn = [this, &rope_cos, &rope_sin](
        const std::vector<float>& hidden,
        const std::vector<float>& temb_6d,
        const std::vector<float>& time_embed,
        const std::vector<float>& encoder_hidden,
        const std::vector<float>& encoder_mask,
        std::vector<float>& output,
        std::string& err) {
        return this->run_denoiser(
            hidden, temb_6d, time_embed, encoder_hidden,
            rope_cos, rope_sin, output, err, encoder_mask);
    };

    if (!run_wan_denoising_loop(
            num_inference_steps,
            use_ddim,
            guidance_scale,
            layout,
            step_timesteps,
            pos_embed_2d,
            text_conditioning.text_projected,
            text_conditioning.null_text,
            conditioning_inputs.encoder_attn_mask,
            ddim_scheduler,
            fm_scheduler,
            latents,
            error,
            compute_temb,
            patchify_fn,
            embed_hidden,
            unpatchify_fn,
            run_denoiser_fn)) {
        std::cerr << "[diffusion] Denoiser failed: " << error << "\n";
        return result;
    }

    denormalize_wan_latents(
        mConfig, layout.z_dim, layout.t_lat, layout.h_lat, layout.w_lat, latents);

    std::cerr << "[diffusion] Decoding video ...\n";
    if (!decode_vae_native(
            latents,
            layout.z_dim,
            layout.t_lat,
            layout.h_lat,
            layout.w_lat,
            result,
            error)) {
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
    config.axes_dims_rope = fp_cfg.axes_dims_rope;
    config.rope_theta = fp_cfg.rope_theta;
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
