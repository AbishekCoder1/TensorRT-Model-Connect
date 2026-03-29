// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-DIFF-CPP-02
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-DIFF-01
// Intent:         Diffusion generation plan: FLUX/Wan layout derivation and scheduler mode selection
// Preconditions:  Diffusion config with valid latent dimensions
// Postconditions: Layout dimensions and scheduler parameters match expected values
// =============================================================================

#include "runtime/domains/diffusion/diffusion_generation_plan.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace {

int g_failures = 0;

void check(bool condition, const char* name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

void check_close(float actual, float expected, float tolerance, const char* name)
{
    if (std::fabs(actual - expected) > tolerance)
    {
        std::cerr << "FAIL: " << name
                  << " actual=" << actual
                  << " expected=" << expected << '\n';
        ++g_failures;
    }
}

void test_flux_generation_plan_derives_layout_and_scheduler()
{
    trtf::DiffusionConfig config;
    config.num_inference_steps = 28;
    config.guidance_scale = 4.5F;
    config.z_dim = 16;
    config.dit_dim = 3072;
    config.text_seq_len = 256;
    config.patch_size = {1, 2, 2};
    config.flow_shift = 1.2F;
    config.use_dynamic_shifting = true;
    config.base_shift = 0.7F;
    config.max_shift = 1.3F;

    trtf::PreprocessorWeights weights;
    weights.vae_bn_mean = {0.0F};

    const auto plan = trtf::diffusion::make_flux_generation_plan(
        config,
        weights,
        0,
        -1.0F,
        128,
        128,
        4096);

    check(plan.num_inference_steps == 28, "flux plan uses fallback steps when request is zero");
    check_close(plan.guidance_scale, 4.5F, 1e-6F, "flux plan uses fallback guidance");
    check(plan.layout.ph == 2 && plan.layout.pw == 2, "flux plan derives patch layout");
    check(plan.layout.packed_channels == 64, "flux plan derives packed channels");
    check(plan.layout.h_packed == 64 && plan.layout.w_packed == 64, "flux plan derives packed spatial size");
    check(plan.is_flux2, "flux plan detects flux2 from bn weights");
    check(plan.latent_size == static_cast<std::size_t>(64 * 64 * 64),
        "flux plan computes flux2 latent size");
    check(plan.scheduler_config.use_dynamic_shifting, "flux plan forwards dynamic shift flag");
    check(plan.scheduler_config.use_empirical_mu, "flux plan uses empirical mu for flux2");
    check(plan.scheduler_config.image_seq_len == 4096, "flux plan forwards image token count");

    const auto scheduler = trtf::diffusion::make_flux_scheduler_state(plan);
    check(scheduler.timesteps.size() == 28, "flux scheduler size matches plan");
    check(scheduler.last_used_dynamic_shifting, "flux scheduler records dynamic shifting");
}

void test_wan_generation_plan_derives_layout_and_scheduler_mode()
{
    trtf::DiffusionConfig config;
    config.scheduler = "ddim";
    config.num_inference_steps = 30;
    config.guidance_scale = 6.0F;
    config.video_num_frames = 81;
    config.video_height = 480;
    config.video_width = 832;
    config.scale_factor_temporal = 4;
    config.scale_factor_spatial = 8;
    config.z_dim = 16;
    config.dit_dim = 1536;
    config.text_seq_len = 512;
    config.patch_size = {1, 2, 2};
    config.flow_shift = 1.15F;

    const auto plan = trtf::diffusion::make_wan_generation_plan(
        config,
        -1,
        -1.0F);

    check(plan.num_inference_steps == 30, "wan plan uses fallback steps for negative request");
    check_close(plan.guidance_scale, 6.0F, 1e-6F, "wan plan uses fallback guidance");
    check(plan.use_ddim, "wan plan selects ddim scheduler family");
    check(plan.layout.t_lat == 21, "wan plan derives temporal latent size");
    check(plan.layout.h_lat == 60 && plan.layout.w_lat == 104, "wan plan derives spatial latent size");
    check(plan.layout.num_patches == 21 * 30 * 52, "wan plan derives patch count");
    check(plan.layout.patch_dim == 64, "wan plan derives patch dim");
    check(plan.latent_count == static_cast<std::size_t>(16 * 21 * 60 * 104),
        "wan plan computes latent count");
}

void test_wan_flow_match_scheduler_builds_when_not_using_ddim()
{
    trtf::DiffusionConfig config;
    config.scheduler = "flow_match_euler";
    config.num_inference_steps = 12;
    config.flow_shift = 1.35F;

    const auto plan = trtf::diffusion::make_wan_generation_plan(
        config,
        8,
        3.0F);
    check(!plan.use_ddim, "wan flow-match plan keeps native scheduler");
    check(plan.num_inference_steps == 8, "wan flow-match plan uses explicit request");
    check_close(plan.guidance_scale, 3.0F, 1e-6F, "wan flow-match plan uses explicit guidance");

    const auto scheduler = trtf::diffusion::make_wan_flow_match_scheduler(plan);
    check(scheduler.timesteps.size() == 8, "wan flow-match scheduler size matches request");
    check_close(scheduler.shift, 1.35F, 1e-6F, "wan flow-match scheduler forwards shift");
}

} // namespace

int main()
{
    test_flux_generation_plan_derives_layout_and_scheduler();
    test_wan_generation_plan_derives_layout_and_scheduler_mode();
    test_wan_flow_match_scheduler_builds_when_not_using_ddim();

    if (g_failures != 0)
    {
        std::cerr << g_failures << " diffusion generation plan test(s) failed\n";
        return 1;
    }
    return 0;
}
