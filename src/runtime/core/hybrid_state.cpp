#include "trtf/runtime/hybrid_state.h"

#include <cassert>

namespace trtf {

HybridState::HybridState(std::unique_ptr<KvCache> kv, std::unique_ptr<RecurrentState> ssm)
    : kv_(std::move(kv))
    , ssm_(std::move(ssm))
{
}

void HybridState::reset()
{
    kv_->reset();
    ssm_->reset();
}

void HybridState::bind_to(TrtModule& module)
{
    kv_->bind_to(module);
    ssm_->bind_to(module);
}

void HybridState::prepare_step(TensorMap& inputs, int32_t seq_len)
{
    kv_->prepare_step(inputs, seq_len);
    // ssm_ has nothing to prepare (state tensors already bound).
}

void HybridState::advance(int32_t n_tokens)
{
    kv_->advance(n_tokens);
    ssm_->advance(n_tokens);
}

int32_t HybridState::position() const
{
    return kv_->position();
}

int32_t HybridState::max_length() const
{
    return kv_->max_length();
}

int32_t HybridState::num_layers() const
{
    return kv_->num_layers() + ssm_->num_layers();
}

std::size_t HybridState::device_memory_bytes() const
{
    return kv_->device_memory_bytes() + ssm_->device_memory_bytes();
}

bool HybridState::ok() const
{
    return kv_ && kv_->ok() && ssm_ && ssm_->ok();
}

} // namespace trtf
