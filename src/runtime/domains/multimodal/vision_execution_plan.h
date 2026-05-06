#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

struct VisionPendingCopy
{
    void* host_ptr{nullptr};
    void* device_ptr{nullptr};
    std::size_t bytes{0};
};

inline std::size_t vision_output_feature_count(int32_t num_output_features, int32_t feature_dim)
{
    return static_cast<std::size_t>(num_output_features) * feature_dim;
}

template <typename HasTensorFn>
inline std::vector<std::string> collect_vision_deepstack_output_names(HasTensorFn&& has_tensor)
{
    std::vector<std::string> names;
    for (int32_t i = 0;; ++i)
    {
        std::string name = "deepstack_features_" + std::to_string(i);
        if (!has_tensor(name))
        {
            break;
        }
        names.push_back(std::move(name));
    }
    return names;
}

template <typename EnqueueFn, typename CopyFn, typename SyncFn>
inline bool run_vision_copy_plan(
    const std::vector<VisionPendingCopy>& output_copies,
    std::string& error,
    EnqueueFn&& enqueue,
    CopyFn&& copy_output,
    SyncFn&& synchronize)
{
    if (!enqueue())
    {
        error = "vision enqueueV3 failed";
        return false;
    }

    for (const VisionPendingCopy& copy : output_copies)
    {
        if (!copy_output(copy))
        {
            error = "vision cudaMemcpyAsync output failed";
            return false;
        }
    }

    if (!synchronize())
    {
        error = "vision cudaStreamSynchronize failed";
        return false;
    }
    return true;
}

} // namespace trtmc
