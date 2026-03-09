#pragma once

// IScheduler: diffusion noise scheduler interface.
// HF equivalent: SchedulerMixin / FlowMatchEulerDiscreteScheduler.
//
// Pipelines compose IScheduler as an interchangeable component — swap
// FlowMatchEuler for DDPM without changing the pipeline code.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtf {

class IScheduler {
public:
    virtual ~IScheduler() = default;

    // Configure the timestep schedule for the given number of steps.
    virtual void set_timesteps(int32_t num_steps) = 0;

    // Access the timestep schedule (descending from ~1000 to ~0).
    virtual const std::vector<float>& timesteps() const = 0;

    // Access the sigma schedule (descending from ~1.0 to ~0.0).
    virtual const std::vector<float>& sigmas() const = 0;

    // Single scheduler step: update latents in-place.
    // latents += (sigma_next - sigma) * velocity
    virtual void step(float* latents, const float* velocity,
                      int32_t num_elements, int32_t step_index) = 0;
};

// Flow Matching Euler Discrete Scheduler.
// Used by: FLUX, Wan, Z-Image, SD3.
class FlowMatchEulerScheduler final : public IScheduler {
public:
    explicit FlowMatchEulerScheduler(float shift = 1.0f,
                                      int32_t num_train_timesteps = 1000);

    void set_timesteps(int32_t num_steps) override;
    const std::vector<float>& timesteps() const override { return timesteps_; }
    const std::vector<float>& sigmas() const override { return sigmas_; }

    void step(float* latents, const float* velocity,
              int32_t num_elements, int32_t step_index) override;

private:
    float shift_;
    int32_t num_train_timesteps_;
    std::vector<float> timesteps_;
    std::vector<float> sigmas_;
};

// Factory: create scheduler by name.
inline std::unique_ptr<IScheduler> create_scheduler(
    const std::string& name, float shift = 1.0f)
{
    if (name == "flow_match_euler")
    {
        return std::make_unique<FlowMatchEulerScheduler>(shift);
    }
    return nullptr;
}

} // namespace trtf
