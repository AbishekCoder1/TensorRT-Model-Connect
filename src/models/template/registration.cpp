// Template for adding a new model family.
// Copy this directory to src/models/<family>/ and customize.
//
// Steps:
// 1. Rename "template" namespace and function names to your family name.
// 2. Implement the checkpoint mapper (HF tensor keys -> DecoderCheckpoint).
//    - For standard HF LLMs (model.layers.N.self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj),
//      consider subclassing StandardCheckpointMapper (src/model/standard_checkpoint_mapper.h).
//      You only need to override can_map() and optionally map_checkpoint() if your tensor
//      key naming differs from the standard pattern.
// 3. Choose a TRT graph builder (StandardDecoderGraphBuilder works for most LLMs).
// 4. Call RegisterYourFamily() from RegisterBuiltinHfModelFamilies() in hf_family_registry.cpp.
// 5. Add your .cpp files to CMakeLists.txt.
//
// What you get for free:
// - StandardTrtModelDefinitionPopulator is registered as a low-priority fallback in
//   RegisterBuiltinHfModelFamilies(). It handles any model with has_decoder_layers,
//   so you do NOT need to register your own populator for standard decoder architectures.
// - StandardDecoderGraphBuilder handles Pre-RMSNorm + GQA + RoPE + SwiGLU (LLaMA, Qwen,
//   Yi, Mistral-dense). Only create a custom ITrtGraphBuilder for non-standard architectures
//   (MoE, parallel attention, etc.).
//
// Testing your family:
// - Use tests/test_helpers.h for temp-dir creation, safetensors writing, and
//   write_standard_decoder_checkpoint() to create synthetic checkpoints.
// - See tests/test_llama_family.cpp for the minimal test pattern:
//   1. write config.json + write_standard_decoder_checkpoint()
//   2. ResolveTextGenerationModel() and validate checkpoint structure
// - Run: ctest --test-dir build -R test_your_family --output-on-failure

#include "trtf/hf_family_registry.h"
#include "model/checkpoint_mapper.h"
#include "runtime/trt/trt_graph_builder.h"
#include "runtime/trt/trt_common.h"
#include "runtime/trt/standard_decoder_graph_builder.h"
#include "utils/text_parsers.h"

#include <filesystem>
#include <memory>
#include <string>

namespace trtf {
namespace your_family {

// --- Checkpoint Mapper ---
// Maps HuggingFace safetensors tensor keys to the generic DecoderCheckpoint.
// See src/models/qwen/checkpoint_mapper.cpp for a complete example.
// See src/model/standard_checkpoint_mapper.h for the base class that handles
// the standard HF tensor naming convention (model.layers.N.self_attn.*, mlp.*).

class YourCheckpointMapper final : public ICheckpointMapper {
public:
    bool can_map(const DecoderArchitectureConfig& architecture) const override
    {
        // Return true if this mapper handles the given family.
        const std::string family = to_lower_ascii(architecture.family);
        return starts_with(family, "your_family");
    }

    DecoderCheckpoint map_checkpoint(
        const TensorSource& reader, std::size_t vocab_size,
        const std::filesystem::path& path,
        const DecoderArchitectureConfig& architecture) const override
    {
        // Map HF tensor keys to DecoderCheckpoint fields.
        // For a standard LLM decoder, you need:
        //   - embedding: model.embed_tokens.weight
        //   - Per-layer: input_norm, q/k/v/o projections, post_attn_norm, gate/up/down MLP
        //   - final_norm: model.norm.weight
        //   - lm_head: lm_head.weight (or tied to embedding)
        //
        // Set checkpoint.has_decoder_layers = true and populate checkpoint.decoder_layers.
        // See QwenCheckpointMapper for the full implementation.
        (void) reader;
        (void) vocab_size;
        (void) path;
        (void) architecture;
        throw std::runtime_error("YourCheckpointMapper::map_checkpoint not implemented");
    }
};

// --- Registration ---

void RegisterYourFamily()
{
    // Registry 2: Checkpoint mapper (HF tensor keys -> DecoderCheckpoint)
    RegisterCheckpointMapper("your_family", 100,
        std::make_unique<YourCheckpointMapper>());

    // Registry 3: TRT model definition populator — NOT needed for standard decoders.
    // StandardTrtModelDefinitionPopulator is registered as a low-priority fallback
    // in RegisterBuiltinHfModelFamilies() and handles any model with has_decoder_layers.
    // Only register a custom populator if your architecture has non-standard TRT
    // definition requirements (e.g., MoE expert routing, custom attention patterns).

    // Registry 4: TRT graph builder
    // StandardDecoderGraphBuilder handles the common pattern:
    //   Pre-RMSNorm -> QKV+RoPE+GQA -> residual -> Post-RMSNorm -> SwiGLU MLP -> residual
#if TRTF_HAS_TRT
    RegisterTrtGraphBuilder("your_family", std::make_unique<StandardDecoderGraphBuilder>());
#endif

    // Registry 1: HF family matcher + model loader
    RegisterHfModelFamily({
        "your-family",   // unique registration name
        100,             // priority (higher = checked first)
        [](const HfModelMetadata& metadata) {
            // Return true if this family handles the given HF model.
            const std::string mt = to_lower_ascii(metadata.model_type);
            return starts_with(mt, "your_family");
        },
        [](const HfModelMetadata& metadata) {
            // Load and return a DecoderModel.
            // For most families, LoadDecoderModel(model_dir) handles the heavy lifting
            // once your checkpoint mapper is registered.
            return LoadDecoderModel(metadata.model_dir);
        },
    });
}

} // namespace your_family
} // namespace trtf
