#include "trtf/runtime/kv_cache.h"
#include "trtf/runtime/trt_module.h"

#if TRTF_HAS_TRT

#include <algorithm>
#include <cstring>

namespace trtf {

KvCache::KvCache(int32_t num_layers, int32_t max_length,
                 int32_t kv_dim, cudaStream_t stream)
    : num_layers_(num_layers)
    , max_length_(max_length)
    , kv_dim_(kv_dim)
    , stream_(stream)
{
    cache_k_.reserve(static_cast<std::size_t>(num_layers));
    cache_v_.reserve(static_cast<std::size_t>(num_layers));
    present_k_.reserve(static_cast<std::size_t>(num_layers));
    present_v_.reserve(static_cast<std::size_t>(num_layers));

    for (int32_t i = 0; i < num_layers; ++i)
    {
        cache_k_.emplace_back(
            std::vector<int64_t>{max_length, kv_dim}, DType::kFloat32, stream);
        cache_v_.emplace_back(
            std::vector<int64_t>{max_length, kv_dim}, DType::kFloat32, stream);
        present_k_.emplace_back(
            std::vector<int64_t>{1, kv_dim}, DType::kFloat32, stream);
        present_v_.emplace_back(
            std::vector<int64_t>{1, kv_dim}, DType::kFloat32, stream);
    }

    reset();
}

void KvCache::build_attention_mask(std::vector<float>& mask) const
{
    // Mask size = max_cache_length + 1 (last slot is current token position).
    // Format matches the TRT engine's attention_mask input: [1, max_length+1].
    const auto width = static_cast<std::size_t>(max_length_) + 1;
    mask.assign(width, -1e4f);
    const int32_t valid = std::max(0, std::min(position_, max_length_));
    for (int32_t i = 0; i < valid; ++i)
    {
        mask[static_cast<std::size_t>(i)] = 0.0f;
    }
    // Current token slot (last position) is always visible
    mask.back() = 0.0f;
}

void KvCache::bind_to(TrtModule& module)
{
    for (int32_t i = 0; i < num_layers_; ++i)
    {
        std::string suffix = "_" + std::to_string(i);
        module.bind_external("cache_k" + suffix, cache_k_[static_cast<std::size_t>(i)].data());
        module.bind_external("cache_v" + suffix, cache_v_[static_cast<std::size_t>(i)].data());
        module.bind_external("present_k" + suffix, present_k_[static_cast<std::size_t>(i)].data());
        module.bind_external("present_v" + suffix, present_v_[static_cast<std::size_t>(i)].data());
    }
}

void KvCache::advance()
{
    // Copy present K/V (single row) into cache at current position.
    // present_k_[layer] is [1, kv_dim] → copy to cache_k_[layer][position_, :]
    auto row_bytes = static_cast<std::size_t>(kv_dim_) * sizeof(float);
    for (int32_t i = 0; i < num_layers_; ++i)
    {
        auto li = static_cast<std::size_t>(i);
        auto* cache_k_ptr = static_cast<uint8_t*>(cache_k_[li].data());
        auto* cache_v_ptr = static_cast<uint8_t*>(cache_v_[li].data());
        auto offset = static_cast<std::size_t>(position_) * row_bytes;

        cudaMemcpyAsync(
            cache_k_ptr + offset, present_k_[li].data(), row_bytes,
            cudaMemcpyDeviceToDevice, stream_);
        cudaMemcpyAsync(
            cache_v_ptr + offset, present_v_[li].data(), row_bytes,
            cudaMemcpyDeviceToDevice, stream_);
    }

    ++position_;
    if (position_ >= max_length_) position_ = max_length_ - 1;
}

void KvCache::reset()
{
    position_ = 0;
    for (int32_t i = 0; i < num_layers_; ++i)
    {
        auto li = static_cast<std::size_t>(i);
        cudaMemsetAsync(cache_k_[li].data(), 0, cache_k_[li].nbytes(), stream_);
        cudaMemsetAsync(cache_v_[li].data(), 0, cache_v_[li].nbytes(), stream_);
        cudaMemsetAsync(present_k_[li].data(), 0, present_k_[li].nbytes(), stream_);
        cudaMemsetAsync(present_v_[li].data(), 0, present_v_[li].nbytes(), stream_);
    }
    cudaStreamSynchronize(stream_);
}

bool KvCache::ok() const
{
    if (cache_k_.size() != static_cast<std::size_t>(num_layers_)) return false;
    for (const auto& t : cache_k_)
    {
        if (!t.ok()) return false;
    }
    return true;
}

} // namespace trtf

#endif // TRTF_HAS_TRT
