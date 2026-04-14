#include "trtf/runtime/kv_cache.h"

#include "trtf/runtime/trt_module.h"

#if TRTF_HAS_TRT

#include <algorithm>
#include <cassert>
#include <cstring>

namespace trtf {

KvCache::KvCache(int32_t num_layers, int32_t max_length, int32_t kv_dim, cudaStream_t stream,
                 DType cache_dtype, KvCacheNames names)
    : num_layers_(num_layers), max_length_(max_length), kv_dim_(kv_dim), stream_(stream),
      cache_dtype_(cache_dtype), cache_element_size_(dtype_size(cache_dtype)),
      names_(std::move(names)) {

    // If names were not supplied, generate standard defaults.
    if (names_.cache_k.empty()) {
        names_.cache_k.reserve(static_cast<std::size_t>(num_layers));
        names_.cache_v.reserve(static_cast<std::size_t>(num_layers));
        names_.present_k.reserve(static_cast<std::size_t>(num_layers));
        names_.present_v.reserve(static_cast<std::size_t>(num_layers));
        for (int32_t i = 0; i < num_layers; ++i) {
            std::string suffix = "_" + std::to_string(i);
            names_.cache_k.push_back("cache_k" + suffix);
            names_.cache_v.push_back("cache_v" + suffix);
            names_.present_k.push_back("present_k" + suffix);
            names_.present_v.push_back("present_v" + suffix);
        }
    }

    cache_k_.reserve(static_cast<std::size_t>(num_layers));
    cache_v_.reserve(static_cast<std::size_t>(num_layers));
    present_k_.reserve(static_cast<std::size_t>(num_layers));
    present_v_.reserve(static_cast<std::size_t>(num_layers));

    for (int32_t i = 0; i < num_layers; ++i) {
        cache_k_.emplace_back(std::vector<int64_t>{max_length, kv_dim}, cache_dtype_, stream);
        cache_v_.emplace_back(std::vector<int64_t>{max_length, kv_dim}, cache_dtype_, stream);
        present_k_.emplace_back(std::vector<int64_t>{1, kv_dim}, cache_dtype_, stream);
        present_v_.emplace_back(std::vector<int64_t>{1, kv_dim}, cache_dtype_, stream);
    }

    // Pre-allocate mask buffer: [max_length + 1] for dense causal mask.
    mask_buf_.resize(static_cast<std::size_t>(max_length) + 1);

    reset();
}

// Masked score constant — must match trt_engine_lifecycle.h kMaskedScore.
static constexpr float kMaskedScore = -1.0e4F;

void KvCache::build_attention_mask(std::vector<float>& mask) const {
    // DEPRECATED: use prepare_step() instead.
    const auto width = static_cast<std::size_t>(max_length_) + 1;
    mask.assign(width, kMaskedScore);
    const int32_t valid = std::max(0, std::min(position_, max_length_));
    for (int32_t i = 0; i < valid; ++i)
        mask[static_cast<std::size_t>(i)] = 0.0f;
    mask.back() = 0.0f;
}

void KvCache::prepare_step(TensorMap& inputs, int32_t /*seq_len*/) {
    // Position input (discovered during bind_to).
    if (has_position_input_) {
        pos_buf_ = position_;
        Tensor pos_t;
        pos_t.data = &pos_buf_;
        pos_t.shape = {1};
        pos_t.dtype = DType::kInt32;
        inputs[names_.position_id] = pos_t;
    }

    // Dense causal mask: 0.0 = visible, -1e4 = masked.
    const int32_t valid = std::max(0, std::min(position_, max_length_));
    std::fill(mask_buf_.begin(), mask_buf_.end(), kMaskedScore);
    for (int32_t i = 0; i < valid; ++i)
        mask_buf_[static_cast<std::size_t>(i)] = 0.0f;
    mask_buf_.back() = 0.0f;

    Tensor mask_t;
    mask_t.data = mask_buf_.data();
    mask_t.shape = {static_cast<int64_t>(mask_buf_.size())};
    mask_t.dtype = DType::kFloat32;
    inputs[names_.attention_mask] = mask_t;
}

void KvCache::bind_to(TrtModule& module) {
    has_position_input_ = module.has_input(names_.position_id);

    for (int32_t i = 0; i < num_layers_; ++i) {
        auto li = static_cast<std::size_t>(i);
        module.bind_external(names_.cache_k[li], cache_k_[li].data());
        module.bind_external(names_.cache_v[li], cache_v_[li].data());
        module.bind_external(names_.present_k[li], present_k_[li].data());
        module.bind_external(names_.present_v[li], present_v_[li].data());
    }
}

void KvCache::advance(int32_t n_tokens) {
    // For now, only single-token advance is supported.
    // n_tokens > 1 reserved for future batched prefill (TASK-10).
    assert(n_tokens == 1 && "KvCache::advance: only n_tokens==1 supported");
    (void)n_tokens;

    // Copy present K/V (single row) into cache at current position.
    // present_k_[layer] is [1, kv_dim] → copy to cache_k_[layer][position_, :]
    auto row_bytes = static_cast<std::size_t>(kv_dim_) * cache_element_size_;

    if (position_ < max_length_) {
        // Normal append: write to position_ slot
        auto offset = static_cast<std::size_t>(position_) * row_bytes;
        for (int32_t i = 0; i < num_layers_; ++i) {
            auto li = static_cast<std::size_t>(i);
            cudaMemcpyAsync(static_cast<uint8_t*>(cache_k_[li].data()) + offset,
                            present_k_[li].data(), row_bytes, cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(static_cast<uint8_t*>(cache_v_[li].data()) + offset,
                            present_v_[li].data(), row_bytes, cudaMemcpyDeviceToDevice, stream_);
        }
        ++position_;
    } else {
        // Cache full: shift [1..max) → [0..max-1), then write at tail
        auto shift_bytes = static_cast<std::size_t>(max_length_ - 1) * row_bytes;
        auto tail_offset = shift_bytes;
        for (int32_t i = 0; i < num_layers_; ++i) {
            auto li = static_cast<std::size_t>(i);
            auto* ck = static_cast<uint8_t*>(cache_k_[li].data());
            auto* cv = static_cast<uint8_t*>(cache_v_[li].data());
            cudaMemcpyAsync(ck, ck + row_bytes, shift_bytes, cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(cv, cv + row_bytes, shift_bytes, cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(ck + tail_offset, present_k_[li].data(), row_bytes,
                            cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(cv + tail_offset, present_v_[li].data(), row_bytes,
                            cudaMemcpyDeviceToDevice, stream_);
        }
        // position_ stays at max_length_ (cache is full, all slots visible)
    }
}

void KvCache::reset() {
    position_ = 0;
    for (int32_t i = 0; i < num_layers_; ++i) {
        auto li = static_cast<std::size_t>(i);
        cudaMemsetAsync(cache_k_[li].data(), 0, cache_k_[li].nbytes(), stream_);
        cudaMemsetAsync(cache_v_[li].data(), 0, cache_v_[li].nbytes(), stream_);
        cudaMemsetAsync(present_k_[li].data(), 0, present_k_[li].nbytes(), stream_);
        cudaMemsetAsync(present_v_[li].data(), 0, present_v_[li].nbytes(), stream_);
    }
    cudaStreamSynchronize(stream_);
}

std::size_t KvCache::device_memory_bytes() const {
    std::size_t total = 0;
    for (const auto& t : cache_k_)
        total += t.nbytes();
    for (const auto& t : cache_v_)
        total += t.nbytes();
    for (const auto& t : present_k_)
        total += t.nbytes();
    for (const auto& t : present_v_)
        total += t.nbytes();
    return total;
}

bool KvCache::ok() const {
    if (cache_k_.size() != static_cast<std::size_t>(num_layers_))
        return false;
    for (const auto& t : cache_k_) {
        if (!t.ok())
            return false;
    }
    return true;
}

} // namespace trtf

#endif // TRTF_HAS_TRT
