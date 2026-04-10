#include "trtf/runtime/recurrent_state.h"
#include "trtf/runtime/trt_module.h"

namespace trtf {

RecurrentState::RecurrentState(int32_t num_layers, std::vector<TensorSpec> specs,
                               cudaStream_t stream)
    : specs_(std::move(specs))
    , num_layers_(num_layers)
    , stream_(stream)
{
    state_.resize(specs_.size());
    present_.resize(specs_.size());

    for (std::size_t si = 0; si < specs_.size(); ++si)
    {
        state_[si].reserve(static_cast<std::size_t>(num_layers));
        present_[si].reserve(static_cast<std::size_t>(num_layers));
        for (int32_t li = 0; li < num_layers; ++li)
        {
            state_[si].emplace_back(specs_[si].shape, DType::kFloat32, stream);
            present_[si].emplace_back(specs_[si].shape, DType::kFloat32, stream);
        }
    }

    reset();
}

void RecurrentState::bind_to(TrtModule& module)
{
    for (std::size_t si = 0; si < specs_.size(); ++si)
    {
        const auto& name = specs_[si].name;
        const auto& out_prefix = specs_[si].output_prefix.empty()
            ? ("present_" + name)
            : specs_[si].output_prefix;
        for (int32_t li = 0; li < num_layers_; ++li)
        {
            auto suffix = "_" + std::to_string(li);
            auto uli = static_cast<std::size_t>(li);

            // Input: {name}_{layer}
            module.bind_external(name + suffix, state_[si][uli].data());
            // Output: {output_prefix}_{layer}
            module.bind_external(out_prefix + suffix, present_[si][uli].data());
        }
    }
}

void RecurrentState::prepare_step(TensorMap& /*inputs*/, int32_t /*seq_len*/)
{
    // Recurrent state has no per-step inputs (no mask, no position).
    // State tensors are already bound via bind_to().
}

void RecurrentState::advance(int32_t n_tokens)
{
    for (std::size_t si = 0; si < specs_.size(); ++si)
    {
        for (int32_t li = 0; li < num_layers_; ++li)
        {
            auto uli = static_cast<std::size_t>(li);
            state_[si][uli].copy_from(present_[si][uli]);
        }
    }
    position_ += n_tokens;
}

void RecurrentState::reset()
{
    position_ = 0;
    for (std::size_t si = 0; si < specs_.size(); ++si)
    {
        for (int32_t li = 0; li < num_layers_; ++li)
        {
            auto uli = static_cast<std::size_t>(li);
            cudaMemsetAsync(state_[si][uli].data(), 0, state_[si][uli].nbytes(), stream_);
            cudaMemsetAsync(present_[si][uli].data(), 0, present_[si][uli].nbytes(), stream_);
        }
    }
    cudaStreamSynchronize(stream_);
}

std::size_t RecurrentState::device_memory_bytes() const
{
    std::size_t total = 0;
    for (std::size_t si = 0; si < specs_.size(); ++si)
    {
        for (const auto& t : state_[si]) total += t.nbytes();
        for (const auto& t : present_[si]) total += t.nbytes();
    }
    return total;
}

bool RecurrentState::ok() const
{
    for (std::size_t si = 0; si < specs_.size(); ++si)
    {
        if (state_[si].size() != static_cast<std::size_t>(num_layers_)) return false;
        for (const auto& t : state_[si])
        {
            if (!t.ok()) return false;
        }
    }
    return true;
}

} // namespace trtf
