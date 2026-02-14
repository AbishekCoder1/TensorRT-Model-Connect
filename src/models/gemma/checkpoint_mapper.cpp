#include "models/gemma/checkpoint_mapper.h"
#include "utils/text_parsers.h"

#include <cmath>

namespace trtf {
namespace gemma {

bool GemmaCheckpointMapper::can_map(const DecoderArchitectureConfig& architecture) const
{
    const std::string fam = to_lower_ascii(architecture.family);
    return starts_with(fam, "gemma");
}

DecoderCheckpoint GemmaCheckpointMapper::map_checkpoint(
    const TensorSource& reader, std::size_t vocab_size,
    const std::filesystem::path& path,
    const DecoderArchitectureConfig& architecture) const
{
    // Call base class to do standard HF checkpoint mapping.
    DecoderCheckpoint checkpoint = StandardCheckpointMapper::map_checkpoint(
        reader, vocab_size, path, architecture);

    // Fix 1: Gemma uses (1 + gamma) * normalized instead of gamma * normalized.
    // The HF weights store the raw gamma (which is ~0.0 at init), so we add 1.0
    // to convert to the standard convention expected by our RMSNorm implementation.
    for (auto& layer : checkpoint.decoder_layers)
    {
        for (auto& v : layer.input_norm) v += 1.0F;
        for (auto& v : layer.post_attn_norm) v += 1.0F;
    }
    for (auto& v : checkpoint.final_norm) v += 1.0F;

    // Fix 2: Gemma scales embedding output by sqrt(hidden_size) before feeding
    // into the decoder layers. Bake this scaling into the embedding weights so
    // the standard graph builder produces correct hidden states without needing
    // a custom embedding op.
    const float scale = std::sqrt(static_cast<float>(checkpoint.hidden_size));
    for (auto& v : checkpoint.embedding) v *= scale;

    return checkpoint;
}

} // namespace gemma
} // namespace trtf
