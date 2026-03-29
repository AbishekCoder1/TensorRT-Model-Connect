#include "trtf/runtime/scheduler.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace trtf {

FlowMatchEulerScheduler::FlowMatchEulerScheduler(float shift,
                                                   int32_t num_train_timesteps)
    : shift_(shift)
    , num_train_timesteps_(num_train_timesteps)
{
}

void FlowMatchEulerScheduler::set_timesteps(int32_t num_steps)
{
    // Match HF FlowMatchEulerDiscreteScheduler:
    // 1. Linspace in t-space from N to sigma_min*N
    // 2. Convert to sigma-space and apply shift
    // 3. Append terminal sigma=0

    const double n = static_cast<double>(num_train_timesteps_);
    const double s = static_cast<double>(shift_);

    // sigma_min = shift * (1/N) / (1 + (shift-1)/N)
    const double raw_sigma_min = 1.0 / n;
    const double sigma_min = s * raw_sigma_min / (1.0 + (s - 1.0) * raw_sigma_min);

    const double t_max = n;
    const double t_min = sigma_min * n;

    // Linspace in t-space
    std::vector<double> t_steps(static_cast<std::size_t>(num_steps));
    for (int32_t i = 0; i < num_steps; ++i)
    {
        double frac = (num_steps <= 1) ? 0.0
            : static_cast<double>(i) / static_cast<double>(num_steps - 1);
        t_steps[static_cast<std::size_t>(i)] = t_max + frac * (t_min - t_max);
    }

    // Convert to sigmas
    std::vector<double> sig(static_cast<std::size_t>(num_steps));
    for (int32_t i = 0; i < num_steps; ++i)
    {
        sig[static_cast<std::size_t>(i)] = t_steps[static_cast<std::size_t>(i)] / n;
    }

    // Apply shift
    if (shift_ != 1.0f)
    {
        for (auto& sigma : sig)
        {
            sigma = s * sigma / (1.0 + (s - 1.0) * sigma);
        }
    }

    // Append terminal sigma=0
    sigmas_.resize(static_cast<std::size_t>(num_steps) + 1);
    for (int32_t i = 0; i < num_steps; ++i)
    {
        sigmas_[static_cast<std::size_t>(i)] = static_cast<float>(sig[static_cast<std::size_t>(i)]);
    }
    sigmas_[static_cast<std::size_t>(num_steps)] = 0.0f;

    // Timesteps = sigmas[:-1] * num_train_timesteps
    timesteps_.resize(static_cast<std::size_t>(num_steps));
    for (int32_t i = 0; i < num_steps; ++i)
    {
        timesteps_[static_cast<std::size_t>(i)] =
            sigmas_[static_cast<std::size_t>(i)] * static_cast<float>(num_train_timesteps_);
    }
}

void FlowMatchEulerScheduler::step(float* latents, const float* velocity,
                                    int32_t num_elements, int32_t step_index)
{
    auto si = static_cast<std::size_t>(step_index);
    if (si + 1 >= sigmas_.size()) return;

    const float sigma = sigmas_[si];
    const float sigma_next = sigmas_[si + 1];
    const float dt = sigma_next - sigma;  // negative (sigma decreasing)

    // Euler step: latents += dt * velocity
    for (int32_t i = 0; i < num_elements; ++i)
    {
        latents[i] += dt * velocity[i];
    }
}

} // namespace trtf
