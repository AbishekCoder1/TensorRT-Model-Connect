#pragma once

#include "runtime/domains/diffusion/diffusion_scheduler_helpers.h"
#include "runtime/domains/diffusion/diffusion_types.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {
namespace diffusion {

struct FluxPackLayout {
    int32_t ph{2};
    int32_t pw{2};
    int32_t packed_channels{0};
    int32_t h_packed{0};
    int32_t w_packed{0};
};

inline FluxPackLayout make_flux_pack_layout(const DiffusionConfig& config, int32_t z_dim,
                                            int32_t h_lat, int32_t w_lat) {
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

struct FluxGenerationPlan {
    int32_t num_inference_steps{0};
    float guidance_scale{0.0F};
    int32_t dit_dim{0};
    int32_t text_seq{0};
    int32_t z_dim{0};
    FluxPackLayout layout;
    bool is_flux2{false};
    std::size_t latent_size{0};
    FlowMatchEulerConfig scheduler_config;
};

inline FluxGenerationPlan make_flux_generation_plan(const DiffusionConfig& config,
                                                    const PreprocessorWeights& weights,
                                                    int32_t requested_steps,
                                                    float requested_guidance, int32_t h_lat,
                                                    int32_t w_lat, int32_t num_img_tokens) {
    FluxGenerationPlan plan;
    plan.num_inference_steps =
        resolve_requested_steps(requested_steps, config.num_inference_steps, true);
    plan.guidance_scale = resolve_requested_guidance(requested_guidance, config.guidance_scale);
    plan.dit_dim = config.dit_dim;
    plan.text_seq = config.text_seq_len;
    plan.z_dim = config.z_dim;
    plan.layout = make_flux_pack_layout(config, plan.z_dim, h_lat, w_lat);
    plan.is_flux2 = !weights.vae_bn_mean.empty();
    plan.latent_size = plan.is_flux2
                           ? (static_cast<std::size_t>(plan.layout.packed_channels) *
                              static_cast<std::size_t>(plan.layout.h_packed) *
                              static_cast<std::size_t>(plan.layout.w_packed))
                           : (static_cast<std::size_t>(plan.z_dim) *
                              static_cast<std::size_t>(h_lat) * static_cast<std::size_t>(w_lat));

    plan.scheduler_config.num_train_timesteps = 1000;
    plan.scheduler_config.shift = config.flow_shift;
    plan.scheduler_config.use_dynamic_shifting = config.use_dynamic_shifting;
    plan.scheduler_config.base_shift = config.base_shift;
    plan.scheduler_config.max_shift = config.max_shift;
    plan.scheduler_config.base_image_seq_len = config.base_image_seq_len;
    plan.scheduler_config.max_image_seq_len = config.max_image_seq_len;
    plan.scheduler_config.shift_terminal = config.shift_terminal;
    plan.scheduler_config.image_seq_len = num_img_tokens;
    plan.scheduler_config.use_empirical_mu = plan.is_flux2;
    return plan;
}

inline FlowMatchEulerState make_flux_scheduler_state(const FluxGenerationPlan& plan) {
    FlowMatchEulerState scheduler;
    scheduler.num_train_timesteps = plan.scheduler_config.num_train_timesteps;
    scheduler.shift = plan.scheduler_config.shift;
    scheduler.use_dynamic_shifting = plan.scheduler_config.use_dynamic_shifting;
    scheduler.base_shift = plan.scheduler_config.base_shift;
    scheduler.max_shift = plan.scheduler_config.max_shift;
    scheduler.base_image_seq_len = plan.scheduler_config.base_image_seq_len;
    scheduler.max_image_seq_len = plan.scheduler_config.max_image_seq_len;
    scheduler.shift_terminal = plan.scheduler_config.shift_terminal;
    scheduler.image_seq_len = plan.scheduler_config.image_seq_len;
    scheduler.use_empirical_mu = plan.scheduler_config.use_empirical_mu;
    scheduler.use_zero_sigma_min = plan.scheduler_config.use_zero_sigma_min;
    scheduler.set_timesteps(plan.num_inference_steps);
    return scheduler;
}

struct WanLayout {
    int32_t t_lat{0};
    int32_t h_lat{0};
    int32_t w_lat{0};
    int32_t z_dim{0};
    int32_t dim{0};
    int32_t seq_len{0};
    int32_t pt{1};
    int32_t ph{2};
    int32_t pw{2};
    int32_t nt{0};
    int32_t nh_p{0};
    int32_t nw_p{0};
    int32_t num_patches{0};
    int32_t patch_dim{0};
};

inline WanLayout make_wan_layout(const DiffusionConfig& config) {
    WanLayout layout;
    layout.t_lat = (config.video_num_frames - 1) / config.scale_factor_temporal + 1;
    layout.h_lat = config.video_height / config.scale_factor_spatial;
    layout.w_lat = config.video_width / config.scale_factor_spatial;
    layout.z_dim = config.z_dim;
    layout.dim = config.dit_dim;
    layout.seq_len = config.text_seq_len;
    if (config.patch_size.size() >= 3) {
        layout.pt = config.patch_size[0];
        layout.ph = config.patch_size[1];
        layout.pw = config.patch_size[2];
    }
    layout.nt = layout.t_lat / layout.pt;
    layout.nh_p = layout.h_lat / layout.ph;
    layout.nw_p = layout.w_lat / layout.pw;
    layout.num_patches = layout.nt * layout.nh_p * layout.nw_p;
    layout.patch_dim = layout.z_dim * layout.pt * layout.ph * layout.pw;
    return layout;
}

inline bool should_use_wan_ddim(const std::string& scheduler) {
    return scheduler == "dpmsolver_multistep" || scheduler == "ddim" || scheduler == "ddpm";
}

struct WanGenerationPlan {
    int32_t num_inference_steps{0};
    float guidance_scale{0.0F};
    WanLayout layout;
    bool use_ddim{false};
    std::size_t latent_count{0};
    FlowMatchEulerConfig flow_match_config;
};

inline WanGenerationPlan make_wan_generation_plan(const DiffusionConfig& config,
                                                  int32_t requested_steps,
                                                  float requested_guidance) {
    WanGenerationPlan plan;
    plan.num_inference_steps =
        resolve_requested_steps(requested_steps, config.num_inference_steps, false);
    plan.guidance_scale = resolve_requested_guidance(requested_guidance, config.guidance_scale);
    plan.layout = make_wan_layout(config);
    plan.use_ddim = should_use_wan_ddim(config.scheduler);
    plan.latent_count =
        static_cast<std::size_t>(plan.layout.z_dim) * static_cast<std::size_t>(plan.layout.t_lat) *
        static_cast<std::size_t>(plan.layout.h_lat) * static_cast<std::size_t>(plan.layout.w_lat);
    plan.flow_match_config.num_train_timesteps = 1000;
    plan.flow_match_config.shift = config.flow_shift;
    plan.flow_match_config.use_dynamic_shifting = config.use_dynamic_shifting;
    plan.flow_match_config.base_shift = config.base_shift;
    plan.flow_match_config.max_shift = config.max_shift;
    plan.flow_match_config.base_image_seq_len = config.base_image_seq_len;
    plan.flow_match_config.max_image_seq_len = config.max_image_seq_len;
    plan.flow_match_config.shift_terminal = config.shift_terminal;
    plan.flow_match_config.image_seq_len = plan.layout.num_patches;
    return plan;
}

inline FlowMatchEulerState make_wan_flow_match_scheduler(const WanGenerationPlan& plan) {
    FlowMatchEulerState scheduler;
    scheduler.num_train_timesteps = plan.flow_match_config.num_train_timesteps;
    scheduler.shift = plan.flow_match_config.shift;
    scheduler.use_dynamic_shifting = plan.flow_match_config.use_dynamic_shifting;
    scheduler.base_shift = plan.flow_match_config.base_shift;
    scheduler.max_shift = plan.flow_match_config.max_shift;
    scheduler.base_image_seq_len = plan.flow_match_config.base_image_seq_len;
    scheduler.max_image_seq_len = plan.flow_match_config.max_image_seq_len;
    scheduler.shift_terminal = plan.flow_match_config.shift_terminal;
    scheduler.image_seq_len = plan.flow_match_config.image_seq_len;
    scheduler.use_empirical_mu = plan.flow_match_config.use_empirical_mu;
    scheduler.use_zero_sigma_min = plan.flow_match_config.use_zero_sigma_min;
    scheduler.set_timesteps(plan.num_inference_steps);
    return scheduler;
}

} // namespace diffusion
} // namespace trtmc
