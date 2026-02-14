#pragma once

#include "model/checkpoint_mapper.h"

#include <string>

namespace trtf {

// Shared checkpoint mapper for standard HF decoder models that use the common
// tensor key naming convention (model.embed_tokens, model.layers.N.self_attn.*,
// model.layers.N.mlp.*, model.norm, lm_head). Works for LLaMA, Qwen, Mistral,
// Yi, Gemma, DeepSeek-dense, and most other modern decoder-only LLMs.
//
// Optional features (auto-detected from safetensors):
// - Per-head q_norm/k_norm (Qwen3 has them, LLaMA does not)
// - Tied lm_head (when lm_head.weight is absent, reuses embedding)
// - GQA (grouped-query attention) with num_kv_heads < num_attention_heads
//
// Subclasses only need to implement can_map() to match their family name.
class StandardCheckpointMapper : public ICheckpointMapper {
public:
    DecoderCheckpoint map_checkpoint(
        const TensorSource& reader, std::size_t vocab_size,
        const std::filesystem::path& path,
        const DecoderArchitectureConfig& architecture) const override;
};

} // namespace trtf
