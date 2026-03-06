#include "runtime/trt/diffusion/flux_diffusion_backend.h"

#if TRTF_HAS_TRT

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <functional>
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
    bool use_empirical_mu{false};  // FLUX.2 empirical mu formula

    void set_timesteps(int32_t num_steps) {
        const double N = static_cast<double>(num_train_timesteps);

        if (use_dynamic_shifting) {
            double mu;

            if (use_empirical_mu) {
                // FLUX.2 empirical mu formula (compute_empirical_mu in diffusers)
                const double a1 = 8.73809524e-05, b1 = 1.89833333;
                const double a2 = 0.00016927, b2 = 0.45666666;
                const double seq = static_cast<double>(image_seq_len);
                const double nsteps = static_cast<double>(num_steps);

                if (seq > 4300.0) {
                    mu = a2 * seq + b2;
                } else {
                    const double m_200 = a2 * seq + b2;
                    const double m_10 = a1 * seq + b1;
                    const double a = (m_200 - m_10) / 190.0;
                    const double b = m_200 - 200.0 * a;
                    mu = a * nsteps + b;
                }
            } else {
                // FLUX.1 linear mu formula
                const double base_seq = 256.0;
                const double max_seq = 4096.0;
                const double m = (static_cast<double>(max_shift) -
                                  static_cast<double>(base_shift)) /
                                 (max_seq - base_seq);
                const double b = static_cast<double>(base_shift) -
                                 m * base_seq;
                mu = static_cast<double>(image_seq_len) * m + b;
            }

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

constexpr int32_t kFluxClipSeqLen = 77;
constexpr int32_t kFluxClipDim = 768;

int32_t resolve_flux_steps(int32_t requested, int32_t fallback)
{
    return requested <= 0 ? fallback : requested;
}

float resolve_flux_guidance(float requested, float fallback)
{
    return requested < 0.0F ? fallback : requested;
}

struct FluxPackLayout {
    int32_t ph{2};
    int32_t pw{2};
    int32_t packed_channels{0};
    int32_t h_packed{0};
    int32_t w_packed{0};
};

FluxPackLayout make_flux_pack_layout(
    const DiffusionConfig& config,
    int32_t z_dim,
    int32_t h_lat,
    int32_t w_lat)
{
    FluxPackLayout layout;
    if (config.patch_size.size() >= 3) {
        layout.ph = config.patch_size[1];
        layout.pw = config.patch_size[2];
    }
    layout.packed_channels = z_dim * layout.ph * layout.pw;
    layout.h_packed = h_lat / layout.ph;
    layout.w_packed = w_lat / layout.pw;
    return layout;
}

std::vector<int32_t> make_clip_padded_ids(
    const std::vector<int32_t>& input_ids,
    int32_t pad_token_id)
{
    std::vector<int32_t> padded(
        static_cast<std::size_t>(kFluxClipSeqLen),
        std::max(pad_token_id, 0));
    const auto copy_len = std::min(
        static_cast<std::size_t>(kFluxClipSeqLen), input_ids.size());
    std::copy_n(input_ids.begin(), copy_len, padded.begin());
    return padded;
}

int32_t find_first_token(
    const std::vector<int32_t>& ids,
    int32_t token_id)
{
    for (int32_t i = 0; i < kFluxClipSeqLen; ++i) {
        if (ids[static_cast<std::size_t>(i)] == token_id) {
            return i;
        }
    }
    return -1;
}

int32_t find_first_max_token(const std::vector<int32_t>& ids)
{
    int32_t max_token_id = ids[0];
    int32_t max_index = 0;
    for (int32_t i = 1; i < kFluxClipSeqLen; ++i) {
        if (ids[static_cast<std::size_t>(i)] > max_token_id) {
            max_token_id = ids[static_cast<std::size_t>(i)];
            max_index = i;
        }
    }
    return max_index;
}

int32_t select_clip_pool_index(
    const std::vector<int32_t>& padded_ids,
    int32_t eos_token_id)
{
    int32_t pool_idx = -1;
    if (eos_token_id >= 0) {
        pool_idx = find_first_token(padded_ids, eos_token_id);
    }
    if (pool_idx < 0) {
        pool_idx = find_first_max_token(padded_ids);
    }
    pool_idx = std::max(pool_idx, 0);
    return std::min(pool_idx, kFluxClipSeqLen - 1);
}

void copy_clip_pooled_row(
    const std::vector<float>& clip_hidden,
    int32_t pool_idx,
    std::vector<float>& pooled_output)
{
    pooled_output.resize(static_cast<std::size_t>(kFluxClipDim));
    const float* pooled_src = clip_hidden.data() +
        static_cast<std::size_t>(pool_idx) * static_cast<std::size_t>(kFluxClipDim);
    std::copy_n(
        pooled_src, static_cast<std::size_t>(kFluxClipDim), pooled_output.begin());
}

std::vector<int32_t> build_flux_clip_ids(
    const std::vector<int32_t>& input_ids,
    ITokenizer* clip_tokenizer,
    const std::string& raw_prompt)
{
    if (clip_tokenizer != nullptr && !raw_prompt.empty()) {
        auto clip_ids = clip_tokenizer->encode(raw_prompt);
        std::cerr << "[flux] CLIP tokenized prompt (" << clip_ids.size()
                  << " tokens) from raw text\n";
        return clip_ids;
    }
    std::cerr << "[flux] Warning: no CLIP tokenizer, using T5 tokens for CLIP encoder\n";
    return input_ids;
}

template <typename RunClipFn>
bool prepare_flux_clip_conditioning(
    const std::vector<int32_t>& input_ids,
    int32_t num_text_encoders,
    ITokenizer* clip_tokenizer,
    const std::string& raw_prompt,
    RunClipFn&& run_clip,
    std::vector<float>& pooled_output,
    std::string& error)
{
    if (num_text_encoders < 2) {
        pooled_output.assign(static_cast<std::size_t>(kFluxClipDim), 0.0F);
        std::cerr << "[flux] No CLIP encoder, using zero pooled output\n";
        return true;
    }

    const auto clip_ids = build_flux_clip_ids(input_ids, clip_tokenizer, raw_prompt);
    if (!run_clip(clip_ids, pooled_output, error)) {
        return false;
    }
    std::cerr << "[flux] CLIP encoder done\n";
    return true;
}

template <typename RunT5Fn>
bool prepare_flux_t5_conditioning(
    const std::vector<int32_t>& input_ids,
    int32_t num_text_encoders,
    RunT5Fn&& run_t5,
    std::vector<float>& text_embeddings,
    std::string& error)
{
    const int32_t t5_idx = (num_text_encoders > 1) ? 1 : 0;
    if (!run_t5(t5_idx, input_ids, text_embeddings, error)) {
        return false;
    }
    std::cerr << "[flux] T5 encoder done\n";
    return true;
}

void initialize_flux_latents(std::vector<float>& latents)
{
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    for (auto& v : latents) {
        v = dist(gen);
    }
}

void fill_flux_sinusoidal_embedding(
    float value,
    int32_t freq_dim,
    std::vector<float>& embedding)
{
    embedding.resize(static_cast<std::size_t>(freq_dim));
    const int32_t half = freq_dim / 2;
    for (int32_t i = 0; i < half; ++i) {
        const float freq = std::exp(
            -std::log(10000.0F) * static_cast<float>(i) /
            static_cast<float>(half));
        embedding[static_cast<std::size_t>(i)] = std::cos(value * freq);
        embedding[static_cast<std::size_t>(i + half)] = std::sin(value * freq);
    }
}

void combine_flux_embeddings(
    const std::vector<float>& timestep_proj,
    const std::vector<float>& text_proj,
    const std::vector<float>& guidance_proj,
    std::vector<float>& temb)
{
    temb.resize(timestep_proj.size());
    for (std::size_t i = 0; i < timestep_proj.size(); ++i) {
        temb[i] = timestep_proj[i] + text_proj[i] + guidance_proj[i];
    }
}

void log_flux_temb_stats(
    float timestep,
    float guidance,
    const std::vector<float>& temb)
{
    float tmin = temb[0];
    float tmax = temb[0];
    double tsum = 0.0;
    for (const auto v : temb) {
        tmin = std::min(tmin, v);
        tmax = std::max(tmax, v);
        tsum += static_cast<double>(v);
    }
    std::cerr << "[flux-temb] t=" << timestep << " g=" << guidance
              << " temb=[" << tmin << "," << tmax
              << ",mean=" << (tsum / static_cast<double>(temb.size())) << "]\n";
}

void pack_flux2_latents(
    const std::vector<float>& latents,
    int32_t packed_channels,
    int32_t h_packed,
    int32_t w_packed,
    std::vector<float>& packed)
{
    // FLUX.2: latents are [packed_channels, h_packed, w_packed] in CHW
    // Pack = CHW -> HWC: tokens[h*W+w, c] = latents[c, h, w]
    const auto num_tokens = static_cast<std::size_t>(h_packed) *
        static_cast<std::size_t>(w_packed);
    packed.resize(num_tokens * static_cast<std::size_t>(packed_channels));
    for (int32_t h = 0; h < h_packed; ++h) {
        for (int32_t w = 0; w < w_packed; ++w) {
            const int32_t tok = h * w_packed + w;
            for (int32_t c = 0; c < packed_channels; ++c) {
                const auto src = static_cast<std::size_t>(c) *
                    static_cast<std::size_t>(h_packed * w_packed) +
                    static_cast<std::size_t>(h * w_packed + w);
                const auto dst = static_cast<std::size_t>(tok) *
                    static_cast<std::size_t>(packed_channels) +
                    static_cast<std::size_t>(c);
                packed[dst] = latents[src];
            }
        }
    }
}

void unpack_flux2_velocity(
    const std::vector<float>& denoiser_output,
    int32_t packed_channels,
    int32_t h_packed,
    int32_t w_packed,
    std::vector<float>& velocity)
{
    // FLUX.2: HWC -> CHW: velocity[c, h, w] = tokens[h*W+w, c]
    const auto total = static_cast<std::size_t>(packed_channels) *
        static_cast<std::size_t>(h_packed) *
        static_cast<std::size_t>(w_packed);
    velocity.resize(total);
    for (int32_t h = 0; h < h_packed; ++h) {
        for (int32_t w = 0; w < w_packed; ++w) {
            const int32_t tok = h * w_packed + w;
            for (int32_t c = 0; c < packed_channels; ++c) {
                const auto src_i = static_cast<std::size_t>(tok) *
                    static_cast<std::size_t>(packed_channels) +
                    static_cast<std::size_t>(c);
                const auto dst_i = static_cast<std::size_t>(c) *
                    static_cast<std::size_t>(h_packed * w_packed) +
                    static_cast<std::size_t>(h * w_packed + w);
                velocity[dst_i] = denoiser_output[src_i];
            }
        }
    }
}

void pack_flux_latents(
    const std::vector<float>& latents,
    int32_t z_dim,
    int32_t h_lat,
    int32_t w_lat,
    const FluxPackLayout& layout,
    std::vector<float>& packed)
{
    const auto num_img_tokens = static_cast<std::size_t>(layout.h_packed) *
        static_cast<std::size_t>(layout.w_packed);
    packed.resize(num_img_tokens * static_cast<std::size_t>(layout.packed_channels));
    for (int32_t py = 0; py < layout.h_packed; ++py) {
        for (int32_t px = 0; px < layout.w_packed; ++px) {
            const int32_t tok_idx = py * layout.w_packed + px;
            float* dst = packed.data() +
                static_cast<std::size_t>(tok_idx) *
                static_cast<std::size_t>(layout.packed_channels);
            int32_t off = 0;
            for (int32_t c = 0; c < z_dim; ++c) {
                for (int32_t dy = 0; dy < layout.ph; ++dy) {
                    for (int32_t dx = 0; dx < layout.pw; ++dx) {
                        const int32_t y = py * layout.ph + dy;
                        const int32_t x = px * layout.pw + dx;
                        const auto src_idx =
                            static_cast<std::size_t>(c) *
                            static_cast<std::size_t>(h_lat * w_lat) +
                            static_cast<std::size_t>(y * w_lat + x);
                        dst[off++] = latents[src_idx];
                    }
                }
            }
        }
    }
}

void unpack_flux_velocity(
    const std::vector<float>& denoiser_output,
    int32_t z_dim,
    int32_t h_lat,
    int32_t w_lat,
    const FluxPackLayout& layout,
    std::vector<float>& velocity)
{
    velocity.resize(
        static_cast<std::size_t>(z_dim) *
        static_cast<std::size_t>(h_lat) *
        static_cast<std::size_t>(w_lat));
    for (int32_t py = 0; py < layout.h_packed; ++py) {
        for (int32_t px = 0; px < layout.w_packed; ++px) {
            const int32_t tok_idx = py * layout.w_packed + px;
            const float* src = denoiser_output.data() +
                static_cast<std::size_t>(tok_idx) *
                static_cast<std::size_t>(layout.packed_channels);
            int32_t off = 0;
            for (int32_t c = 0; c < z_dim; ++c) {
                for (int32_t dy = 0; dy < layout.ph; ++dy) {
                    for (int32_t dx = 0; dx < layout.pw; ++dx) {
                        const int32_t y = py * layout.ph + dy;
                        const int32_t x = px * layout.pw + dx;
                        const auto dst_idx =
                            static_cast<std::size_t>(c) *
                            static_cast<std::size_t>(h_lat * w_lat) +
                            static_cast<std::size_t>(y * w_lat + x);
                        velocity[dst_idx] = src[off++];
                    }
                }
            }
        }
    }
}

void compute_vector_stats(
    const std::vector<float>& values,
    float& min_out,
    float& max_out,
    double& mean_out)
{
    min_out = values[0];
    max_out = values[0];
    double sum = 0.0;
    for (const auto v : values) {
        min_out = std::min(min_out, v);
        max_out = std::max(max_out, v);
        sum += static_cast<double>(v);
    }
    mean_out = sum / static_cast<double>(values.size());
}

void log_flux_step_stats(
    int32_t step,
    int32_t num_inference_steps,
    const FlowMatchEulerState& scheduler,
    const std::vector<float>& latents,
    const std::vector<float>& velocity,
    const std::vector<float>& hidden)
{
    float lat_min = 0.0F, lat_max = 0.0F;
    float vel_min = 0.0F, vel_max = 0.0F;
    float hid_min = 0.0F, hid_max = 0.0F;
    double lat_mean = 0.0;
    double vel_mean = 0.0;
    double hid_mean = 0.0;

    compute_vector_stats(latents, lat_min, lat_max, lat_mean);
    compute_vector_stats(velocity, vel_min, vel_max, vel_mean);
    compute_vector_stats(hidden, hid_min, hid_max, hid_mean);

    const auto si = static_cast<std::size_t>(step);
    std::cerr << "[flux] Step " << (step + 1) << "/" << num_inference_steps
              << " t=" << scheduler.timesteps[si]
              << " dt=" << (scheduler.sigmas[si + 1] - scheduler.sigmas[si])
              << " latent=[" << lat_min << "," << lat_max
              << ",mean=" << lat_mean
              << "] vel=[" << vel_min << "," << vel_max
              << ",mean=" << vel_mean
              << "] hidden=[" << hid_min << "," << hid_max
              << ",mean=" << hid_mean
              << "]\n";
}

void apply_bn_denorm_inplace(
    std::vector<float>& data,
    int32_t num_channels,
    int32_t spatial_size,
    const std::vector<float>& bn_mean,
    const std::vector<float>& bn_var,
    float eps)
{
    const int32_t bn_ch = static_cast<int32_t>(bn_mean.size());
    const auto spatial = static_cast<std::size_t>(spatial_size);
    for (int32_t c = 0; c < bn_ch && c < num_channels; ++c) {
        const float s = std::sqrt(bn_var[static_cast<std::size_t>(c)] + eps);
        const float m = bn_mean[static_cast<std::size_t>(c)];
        for (std::size_t i = 0; i < spatial; ++i) {
            const auto idx = static_cast<std::size_t>(c) * spatial + i;
            data[idx] = data[idx] * s + m;
        }
    }
}

void unpatchify_latents(
    const std::vector<float>& packed,
    const FluxPackLayout& layout,
    int32_t z_dim,
    int32_t h_lat, int32_t w_lat,
    std::vector<float>& out)
{
    const auto spatial = static_cast<std::size_t>(layout.h_packed * layout.w_packed);
    out.resize(static_cast<std::size_t>(z_dim) *
               static_cast<std::size_t>(h_lat) * static_cast<std::size_t>(w_lat));
    for (int32_t c = 0; c < z_dim; ++c) {
        for (int32_t py = 0; py < layout.h_packed; ++py) {
            for (int32_t px = 0; px < layout.w_packed; ++px) {
                for (int32_t dy = 0; dy < layout.ph; ++dy) {
                    for (int32_t dx = 0; dx < layout.pw; ++dx) {
                        const int32_t src_ch = c * layout.ph * layout.pw + dy * layout.pw + dx;
                        const auto si = static_cast<std::size_t>(src_ch) * spatial +
                                        static_cast<std::size_t>(py * layout.w_packed + px);
                        const auto di = static_cast<std::size_t>(c) *
                                        static_cast<std::size_t>(h_lat * w_lat) +
                                        static_cast<std::size_t>((py * layout.ph + dy) * w_lat +
                                                                  px * layout.pw + dx);
                        out[di] = packed[si];
                    }
                }
            }
        }
    }
}

std::function<void(const std::vector<float>&, std::vector<float>&)>
make_flux_pack_fn(bool is_flux2, int32_t z_dim, int32_t h_lat, int32_t w_lat,
                  const FluxPackLayout& layout)
{
    if (is_flux2) {
        return [&layout](const std::vector<float>& lat, std::vector<float>& packed) {
            pack_flux2_latents(lat, layout.packed_channels,
                               layout.h_packed, layout.w_packed, packed);
        };
    }
    return [z_dim, h_lat, w_lat, &layout](const std::vector<float>& lat,
                                           std::vector<float>& packed) {
        pack_flux_latents(lat, z_dim, h_lat, w_lat, layout, packed);
    };
}

std::function<void(const std::vector<float>&, std::vector<float>&)>
make_flux_unpack_fn(bool is_flux2, int32_t z_dim, int32_t h_lat, int32_t w_lat,
                    const FluxPackLayout& layout)
{
    if (is_flux2) {
        return [&layout](const std::vector<float>& out, std::vector<float>& vel) {
            unpack_flux2_velocity(out, layout.packed_channels,
                                  layout.h_packed, layout.w_packed, vel);
        };
    }
    return [z_dim, h_lat, w_lat, &layout](const std::vector<float>& out,
                                           std::vector<float>& vel) {
        unpack_flux_velocity(out, z_dim, h_lat, w_lat, layout, vel);
    };
}

void prepare_flux2_vae_input(
    std::vector<float>& latents,
    const FluxPackLayout& layout,
    int32_t z_dim,
    int32_t h_lat, int32_t w_lat,
    const std::vector<float>& bn_mean,
    const std::vector<float>& bn_var,
    bool is_flux2,
    std::vector<float>& vae_latents)
{
    if (!is_flux2 || bn_mean.empty()) {
        vae_latents = latents;
        return;
    }

    apply_bn_denorm_inplace(latents, layout.packed_channels,
                            layout.h_packed * layout.w_packed,
                            bn_mean, bn_var, 0.0001F);
    unpatchify_latents(latents, layout, z_dim, h_lat, w_lat, vae_latents);
    std::cerr << "[flux] Applied BN denorm + unpatchify ("
              << bn_mean.size() << " -> " << z_dim << " ch)\n";
}

void maybe_dump_flux_latents(const std::vector<float>& latents)
{
    const std::string dump_path = "/tmp/flux_final_latents.raw";
    std::ofstream dump(dump_path, std::ios::binary);
    if (!dump.is_open()) {
        return;
    }
    dump.write(reinterpret_cast<const char*>(latents.data()),
               latents.size() * sizeof(float));
    dump.close();
    std::cerr << "[flux] Dumped final latents (" << latents.size()
              << " floats) to " << dump_path << "\n";
}

void convert_flux_vae_output_to_video(
    const std::vector<float>& vae_output,
    int32_t h_out,
    int32_t w_out,
    VideoResult& result)
{
    result.num_frames = 1;
    result.height = h_out;
    result.width = w_out;
    result.frames.resize(static_cast<std::size_t>(h_out * w_out * 3));
    for (int32_t h = 0; h < h_out; ++h) {
        for (int32_t w = 0; w < w_out; ++w) {
            for (int32_t c = 0; c < 3; ++c) {
                const auto src = static_cast<std::size_t>(c) *
                    static_cast<std::size_t>(h_out * w_out) +
                    static_cast<std::size_t>(h * w_out + w);
                const auto dst = static_cast<std::size_t>(h * w_out * 3 + w * 3 + c);
                float v = (vae_output[src] + 1.0F) * 0.5F;
                result.frames[dst] = std::max(0.0F, std::min(1.0F, v));
            }
        }
    }
}

bool decode_flux_vae(
    DiffusionEngine& vae_decoder,
    CudaStream& stream,
    const DiffusionConfig& config,
    int32_t h_lat,
    int32_t w_lat,
    int32_t z_dim,
    const std::vector<float>& latents,
    VideoResult& result)
{
    const int32_t h_out = config.video_height;
    const int32_t w_out = config.video_width;
    const auto vae_in_size = static_cast<std::size_t>(z_dim) *
        static_cast<std::size_t>(h_lat) * static_cast<std::size_t>(w_lat);
    const auto vae_out_size = static_cast<std::size_t>(3) *
        static_cast<std::size_t>(h_out) * static_cast<std::size_t>(w_out);

    CudaBuffer d_vae_in(vae_in_size * sizeof(float));
    CudaBuffer d_vae_out(vae_out_size * sizeof(float));
    cudaMemcpyAsync(d_vae_in.data(), latents.data(),
        vae_in_size * sizeof(float), cudaMemcpyHostToDevice, stream.get());

    vae_decoder.context->setTensorAddress("latents", d_vae_in.data());
    vae_decoder.context->setTensorAddress("image", d_vae_out.data());
    if (!vae_decoder.context->enqueueV3(stream.get())) {
        return false;
    }

    std::vector<float> vae_output(vae_out_size);
    cudaMemcpyAsync(vae_output.data(), d_vae_out.data(),
        vae_out_size * sizeof(float), cudaMemcpyDeviceToHost, stream.get());
    cudaStreamSynchronize(stream.get());
    convert_flux_vae_output_to_video(vae_output, h_out, w_out, result);
    return true;
}

template <typename PackFn, typename UnpackFn, typename ComputeTembFn,
          typename EmbedHiddenFn, typename RunDenoiserFn>
bool run_flux_denoising_loop(
    FlowMatchEulerState& scheduler,
    int32_t num_inference_steps,
    std::vector<float>& latents,
    std::vector<float>& hidden,
    std::vector<float>& denoiser_output,
    std::string& error,
    PackFn&& pack_latents,
    UnpackFn&& unpack_velocity,
    ComputeTembFn&& compute_temb,
    EmbedHiddenFn&& embed_hidden,
    RunDenoiserFn&& run_denoiser)
{
    std::vector<float> temb;
    std::vector<float> packed;
    std::vector<float> velocity;
    std::vector<float> next_latents(latents.size());

    for (int32_t step = 0; step < num_inference_steps; ++step) {
        const float t = scheduler.timesteps[static_cast<std::size_t>(step)] / 1000.0F;
        compute_temb(t, temb);
        pack_latents(latents, packed);
        embed_hidden(packed, hidden);
        if (!run_denoiser(hidden, temb, denoiser_output, error)) {
            std::cerr << "[flux] Denoiser step " << step << " failed: " << error << "\n";
            return false;
        }
        unpack_velocity(denoiser_output, velocity);
        scheduler.step(velocity.data(), latents.data(), next_latents.data(),
            latents.size(), step);
        latents = next_latents;
        log_flux_step_stats(
            step, num_inference_steps, scheduler, latents, velocity, hidden);
    }
    return true;
}

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

    std::vector<float> t_emb;
    fill_flux_sinusoidal_embedding(timestep * 1000.0F, freq_dim, t_emb);

    // Helper: return nullptr for empty bias vectors, valid pointer otherwise
    auto bias_or_null = [](const std::vector<float>& v) -> const float* {
        return v.empty() ? nullptr : v.data();
    };

    // timestep_embedder MLP: sinusoidal -> Linear -> SiLU -> Linear
    std::vector<float> t_proj(static_cast<std::size_t>(dim));
    cpu_matmul_bias(t_emb.data(),
        mWeights.time_emb_0_weight.data(),
        bias_or_null(mWeights.time_emb_0_bias),
        t_proj.data(), 1, freq_dim, dim);
    cpu_silu_inplace(t_proj.data(), static_cast<std::size_t>(dim));

    std::vector<float> t_proj2(static_cast<std::size_t>(dim));
    cpu_matmul_bias(t_proj.data(),
        mWeights.time_emb_2_weight.data(),
        bias_or_null(mWeights.time_emb_2_bias),
        t_proj2.data(), 1, dim, dim);

    // text_embedder MLP: pooled -> Linear -> SiLU -> Linear
    std::vector<float> text_proj(static_cast<std::size_t>(dim));
    if (!mWeights.text_proj_weight.empty() && !pooled_text.empty()) {
        const int32_t text_in_dim = static_cast<int32_t>(pooled_text.size());
        cpu_matmul_bias(pooled_text.data(),
            mWeights.text_proj_weight.data(),
            bias_or_null(mWeights.text_proj_bias),
            text_proj.data(), 1, text_in_dim, dim);
        cpu_silu_inplace(text_proj.data(), static_cast<std::size_t>(dim));

        if (!mWeights.text_proj_2_weight.empty()) {
            std::vector<float> text_proj2(static_cast<std::size_t>(dim));
            cpu_matmul_bias(text_proj.data(),
                mWeights.text_proj_2_weight.data(),
                bias_or_null(mWeights.text_proj_2_bias),
                text_proj2.data(), 1, dim, dim);
            text_proj = std::move(text_proj2);
        }
    }

    // Guidance embedding MLP (if guidance_embeds is enabled)
    std::vector<float> guidance_proj(static_cast<std::size_t>(dim), 0.0F);
    if (timestep > 0.99F) {
        std::cerr << "[flux-temb] guidance_embeds=" << mConfig.guidance_embeds
                  << " g_w0=" << mWeights.guidance_emb_0_weight.size()
                  << " g_w2=" << mWeights.guidance_emb_2_weight.size()
                  << "\n";
    }
    if (mConfig.guidance_embeds &&
        !mWeights.guidance_emb_0_weight.empty()) {
        // Diffusers FLUX forward currently scales guidance by 1000 before
        // feeding it into time_text_embed (same convention as timestep).
        std::vector<float> g_emb;
        fill_flux_sinusoidal_embedding(guidance * 1000.0F, freq_dim, g_emb);

        // Linear -> SiLU -> Linear
        std::vector<float> g_proj(static_cast<std::size_t>(dim));
        cpu_matmul_bias(g_emb.data(),
            mWeights.guidance_emb_0_weight.data(),
            bias_or_null(mWeights.guidance_emb_0_bias),
            g_proj.data(), 1, freq_dim, dim);
        cpu_silu_inplace(g_proj.data(), static_cast<std::size_t>(dim));

        cpu_matmul_bias(g_proj.data(),
            mWeights.guidance_emb_2_weight.data(),
            bias_or_null(mWeights.guidance_emb_2_bias),
            guidance_proj.data(), 1, dim, dim);

        if (timestep > 0.99F) {
            float gmin = guidance_proj[0], gmax = guidance_proj[0];
            double gsum = 0.0;
            for (auto v : guidance_proj) {
                gmin = std::min(gmin, v);
                gmax = std::max(gmax, v);
                gsum += static_cast<double>(v);
            }
            std::cerr << "[flux-temb] guidance_proj=[" << gmin << "," << gmax
                      << ",mean=" << (gsum / static_cast<double>(dim)) << "]\n";
        }
    }

    combine_flux_embeddings(t_proj2, text_proj, guidance_proj, temb);
    log_flux_temb_stats(timestep, guidance, temb);
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

    // FLUX uses multi-axis RoPE: (text_pos, h_pos, w_pos [, extra_pos])
    // FLUX.1 default axes = (16, 56, 56) => 3D, total = 128 = head_dim
    // FLUX.2 default axes = (32, 32, 32, 32) => 4D, total = 128 = head_dim
    // For text tokens: ALL positions are 0 on ALL axes (identity rotation)
    // For image tokens: position 0 on axis 0, h on axis 1, w on axis 2, 0 on axis 3+

    const float theta = mConfig.rope_theta;

    // Read axis dimensions from config, fall back to FLUX.1 default
    std::vector<int32_t> axes = mConfig.axes_dims_rope;
    if (axes.empty()) {
        axes = {16, 56, 56};  // FLUX.1 default
    }

    auto encode_pos = [&](float* cos_row, float* sin_row,
                          int32_t text_pos, int32_t h_pos, int32_t w_pos) {
        int32_t offset = 0;
        for (std::size_t ax = 0; ax < axes.size(); ++ax) {
            const int32_t ax_dim = axes[ax];
            // Determine position value for this axis
            int32_t pos = 0;
            if (ax == 0) pos = text_pos;
            else if (ax == 1) pos = h_pos;
            else if (ax == 2) pos = w_pos;
            // axes[3+] default to position 0 (identity rotation for image tokens)

            for (int32_t i = 0; i < ax_dim / 2; ++i) {
                const float freq = 1.0F / std::pow(theta, 2.0F * static_cast<float>(i) / static_cast<float>(ax_dim));
                const float angle = static_cast<float>(pos) * freq;
                cos_row[offset + 2 * i] = std::cos(angle);
                cos_row[offset + 2 * i + 1] = std::cos(angle);
                sin_row[offset + 2 * i] = std::sin(angle);
                sin_row[offset + 2 * i + 1] = std::sin(angle);
            }
            offset += ax_dim;
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
    if (mTextEncoders.empty()) {
        pooled_output.assign(static_cast<std::size_t>(kFluxClipDim), 0.0F);
        return true;
    }

    auto& te = mTextEncoders[0];
    const auto padded = make_clip_padded_ids(input_ids, mClipPadTokenId);

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

    std::vector<float> clip_hidden(
        static_cast<std::size_t>(kFluxClipSeqLen) *
        static_cast<std::size_t>(kFluxClipDim));
    cudaMemcpyAsync(clip_hidden.data(), mD_ClipTextEmb.data(),
        clip_hidden.size() * sizeof(float),
        cudaMemcpyDeviceToHost, mStream.get());
    cudaStreamSynchronize(mStream.get());

    const int32_t pool_idx = select_clip_pool_index(padded, mClipEosTokenId);
    copy_clip_pooled_row(clip_hidden, pool_idx, pooled_output);
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

std::string FluxDiffusionBackend::prepare_prompt(const std::string& prompt) const
{
    // FLUX.2 uses Mistral chat template with system message
    // Detect FLUX.2 via VAE BN weights presence
    if (!mWeights.vae_bn_mean.empty()) {
        static const char* kSystemMsg =
            "You are an AI that reasons about image descriptions. "
            "You give structured responses focusing on object relationships, object\n"
            "attribution and actions without speculation.";
        return std::string("<s>[SYSTEM_PROMPT]") + kSystemMsg +
               "[/SYSTEM_PROMPT][INST]" + prompt + "[/INST]";
    }
    return prompt;
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

    num_inference_steps = resolve_flux_steps(
        num_inference_steps, mConfig.num_inference_steps);
    guidance_scale = resolve_flux_guidance(
        guidance_scale, mConfig.guidance_scale);

    const int32_t dit_dim = mConfig.dit_dim;
    const int32_t text_seq = mConfig.text_seq_len;
    const int32_t z_dim = mConfig.z_dim;
    const auto layout = make_flux_pack_layout(mConfig, z_dim, mHLatent, mWLatent);

    std::vector<float> pooled_output;
    auto run_clip = [this](
        const std::vector<int32_t>& ids,
        std::vector<float>& pooled,
        std::string& err) {
        return run_clip_encoder(ids, pooled, err);
    };
    if (!prepare_flux_clip_conditioning(
            input_ids,
            static_cast<int32_t>(mTextEncoders.size()),
            mClipTokenizer.get(),
            mRawPrompt,
            run_clip,
            pooled_output,
            error)) {
        std::cerr << "[flux] CLIP encoder failed: " << error << "\n";
        return result;
    }

    std::vector<float> text_embeddings;
    auto run_t5 = [this](
        int32_t idx,
        const std::vector<int32_t>& ids,
        std::vector<float>& embeddings,
        std::string& err) {
        return run_t5_encoder_at(idx, ids, embeddings, err);
    };
    if (!prepare_flux_t5_conditioning(
            input_ids,
            static_cast<int32_t>(mTextEncoders.size()),
            run_t5,
            text_embeddings,
            error)) {
        std::cerr << "[flux] T5 encoder failed: " << error << "\n";
        return result;
    }
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
        std::cerr << "[flux] Warning: No context_embedder weights\n";
    }

    std::vector<float> cos_vals, sin_vals;
    compute_flux_rope(layout.h_packed, layout.w_packed, text_seq, cos_vals, sin_vals);

    // 5. Initialize random latents
    // FLUX.2: sample as [packed_channels, h_packed, w_packed] = [128, 64, 64]
    //   then simple flatten to [4096, 128] tokens (channels-last)
    // FLUX.1: sample as [z_dim, h_lat, w_lat] = [16, 128, 128]
    //   then 2x2 spatial pack to [4096, 64] tokens
    const bool is_flux2 = !mWeights.vae_bn_mean.empty();
    const auto latent_size = is_flux2
        ? (static_cast<std::size_t>(layout.packed_channels) *
           static_cast<std::size_t>(layout.h_packed) *
           static_cast<std::size_t>(layout.w_packed))
        : (static_cast<std::size_t>(z_dim) *
           static_cast<std::size_t>(mHLatent) *
           static_cast<std::size_t>(mWLatent));
    std::vector<float> latents(latent_size);
    initialize_flux_latents(latents);

    FlowMatchEulerState scheduler;
    scheduler.shift = mConfig.flow_shift;
    scheduler.use_dynamic_shifting = mConfig.use_dynamic_shifting;
    scheduler.base_shift = mConfig.base_shift;
    scheduler.max_shift = mConfig.max_shift;
    scheduler.image_seq_len = mNumImgTokens;
    // FLUX.2 uses empirical mu formula (detected via VAE BN presence)
    scheduler.use_empirical_mu = !mWeights.vae_bn_mean.empty();
    scheduler.set_timesteps(num_inference_steps);

    std::cerr << "[flux] Starting denoising loop (" << num_inference_steps << " steps)"
              << " latents=[" << z_dim << "," << mHLatent << "," << mWLatent << "]"
              << " packed=[" << mNumImgTokens << "," << layout.packed_channels << "] ...\n";

    std::vector<float> hidden(
        static_cast<std::size_t>(mNumImgTokens) * static_cast<std::size_t>(dit_dim));
    std::vector<float> denoiser_output;

    const auto compute_temb = [this, guidance_scale, &pooled_output](
        float t,
        std::vector<float>& temb) {
        compute_flux_timestep_embedding(t, guidance_scale, pooled_output, temb);
    };
    const auto run_denoiser = [this, &encoder_hidden, &cos_vals, &sin_vals](
        const std::vector<float>& hidden_in,
        const std::vector<float>& temb_in,
        std::vector<float>& output,
        std::string& err) {
        return run_flux_denoiser(
            hidden_in, encoder_hidden, temb_in, cos_vals, sin_vals, output, err);
    };

    // Pack/unpack: FLUX.2 uses simple CHW->HWC, FLUX.1 uses 2x2 spatial packing
    auto pack_latents_fn = make_flux_pack_fn(is_flux2, z_dim, mHLatent, mWLatent, layout);
    auto unpack_velocity_fn = make_flux_unpack_fn(is_flux2, z_dim, mHLatent, mWLatent, layout);

    // Embed hidden states: pack -> x_embedder matmul
    std::function<void(const std::vector<float>&, std::vector<float>&)> embed_hidden;
    if (!mFluxXEmbedW.empty()) {
        embed_hidden = [this, &layout, dit_dim](
            const std::vector<float>& packed,
            std::vector<float>& hidden_out) {
            cpu_matmul_bias(
                packed.data(),
                mFluxXEmbedW.data(),
                mFluxXEmbedB.empty() ? nullptr : mFluxXEmbedB.data(),
                hidden_out.data(),
                mNumImgTokens,
                layout.packed_channels,
                dit_dim);
        };
    } else {
        std::cerr << "[flux] Warning: No x_embedder weights, hidden_states are zero\n";
        embed_hidden = [](
            const std::vector<float>& /*packed*/,
            std::vector<float>& hidden_out) {
            std::fill(hidden_out.begin(), hidden_out.end(), 0.0F);
        };
    }

    if (!run_flux_denoising_loop(
            scheduler,
            num_inference_steps,
            latents,
            hidden,
            denoiser_output,
            error,
            pack_latents_fn,
            unpack_velocity_fn,
            compute_temb,
            embed_hidden,
            run_denoiser)) {
        return result;
    }

    maybe_dump_flux_latents(latents);

    // Prepare VAE input: BN denorm + unpatchify for FLUX.2, identity for FLUX.1
    std::vector<float> vae_latents;
    prepare_flux2_vae_input(
        latents, layout, z_dim, mHLatent, mWLatent,
        mWeights.vae_bn_mean, mWeights.vae_bn_var,
        is_flux2, vae_latents);

    std::cerr << "[flux] Decoding latents via native TRT VAE engine ...\n";
    if (!decode_flux_vae(
            mVaeDecoder, mStream, mConfig, mHLatent, mWLatent, z_dim,
            vae_latents, result)) {
        std::cerr << "[flux] VAE TRT enqueueV3 failed\n";
        return result;
    }

    std::cerr << "[flux] Image generated: " << result.width << "x" << result.height << "\n";
    return result;
}

} // namespace trtf

#endif // TRTF_HAS_TRT
