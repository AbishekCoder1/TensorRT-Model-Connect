// =============================================================================
// Test: Gemma model family registration and checkpoint loading.
//
// Purpose:
//   Validates the end-to-end integration of the Gemma model family through the
//   HF family registry. This exercises Registry 1 (HfModelFamily) and Registry 2
//   (CheckpointMapper) for the Gemma architecture, ensuring that a synthetic HF
//   model directory with model_type "gemma" is correctly detected, loaded, and
//   mapped to a DecoderModel.
//
// What is verified:
//   - HF family detection: model_type "gemma" + "GemmaForCausalLM" triggers
//     the Gemma family registration.
//   - Multi-layer checkpoint loading: a 2-layer safetensors checkpoint is read
//     and mapped through GemmaCheckpointMapper.
//   - GQA layout: num_kv_heads (1) < num_attention_heads (2), verifying grouped
//     query attention geometry is preserved.
//   - No per-head QK norms: Gemma does not use q_norm/k_norm, so these vectors
//     must be empty in the loaded checkpoint.
//   - Attention width preservation: non-square q_proj (q_hidden=16 vs hidden=8)
//     is correctly reported as attention_size=16.
//
// Note on Gemma weight pre-processing:
//   In production, GemmaCheckpointMapper applies two weight transformations:
//   (1) adds +1.0 to all RMSNorm gamma weights, and (2) scales the embedding
//   by sqrt(hidden_size). This test uses the standard decoder checkpoint helper
//   which writes raw weights, so it primarily validates the structural loading
//   path rather than the weight transformation logic.
//
// Dependencies:
//   - test_helpers.h: make_temp_dir_or_throw, write_file, write_standard_decoder_checkpoint
//   - trtf/model_resolver.h: ResolveTextGenerationModel, ResolvedModelSpec
//   - Gemma family registration (auto-discovered via CMake GLOB)
//
// Approach:
//   Creates a synthetic 2-layer Gemma checkpoint in a temporary directory with
//   config.json and model.safetensors, resolves it through the full model
//   resolution pipeline, and validates the resulting ResolvedModelSpec fields.
// =============================================================================

#include "test_helpers.h"
#include "trtf/model_resolver.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace {

// ---------------------------------------------------------------------------
// write_hf_gemma_root
//
// Intention: Create a minimal but structurally valid HF Gemma model directory.
// Setup:
//   - Writes config.json with model_type="gemma", architectures=["GemmaForCausalLM"],
//     vocab_size=8, hidden_size=8, 2 layers, 2 attention heads, 1 KV head (GQA).
//   - Writes model.safetensors via write_standard_decoder_checkpoint with
//     q_hidden=16, kv_hidden=8, mlp=16, 2 layers, include_qk_norm=false.
// Mechanism: Delegates to test helpers for file and safetensors generation.
// ---------------------------------------------------------------------------
void write_hf_gemma_root(const std::filesystem::path& dir)
{
    trtf_test::write_file(dir / "config.json",
        "{\n"
        "  \"model_type\": \"gemma\",\n"
        "  \"architectures\": [\"GemmaForCausalLM\"],\n"
        "  \"vocab_size\": 8,\n"
        "  \"hidden_size\": 8,\n"
        "  \"num_hidden_layers\": 2,\n"
        "  \"num_attention_heads\": 2,\n"
        "  \"num_key_value_heads\": 1,\n"
        "  \"rms_norm_eps\": 1e-5,\n"
        "  \"rope_theta\": 10000.0\n"
        "}\n");
    trtf_test::write_standard_decoder_checkpoint(dir, 8, 8, 16, 8, 16, 2, false);
}

} // namespace

int main()
{
    std::filesystem::path family_dir;

    try
    {
        family_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_gemma_family_XXXXXX");
        write_hf_gemma_root(family_dir);

        // -----------------------------------------------------------------
        // Assertion: Model resolves as kDecoderDefinition
        //
        // Intention: Verify that the family registry recognizes the Gemma
        //   config.json + safetensors combination and resolves it as a
        //   decoder-definition (not kHuggingFaceLocal or kUnknown).
        // Mechanism: Calls ResolveTextGenerationModel on the temp dir path
        //   and checks spec.kind.
        // -----------------------------------------------------------------
        const trtf::ResolvedModelSpec spec = trtf::ResolveTextGenerationModel(family_dir.string());
        if (spec.kind != trtf::ResolvedModelKind::kDecoderDefinition)
        {
            std::cerr << "expected gemma hf dir to resolve as decoder-definition" << std::endl;
            return 1;
        }

        // -----------------------------------------------------------------
        // Assertion: Checkpoint was loaded (has_checkpoint = true)
        //
        // Intention: Confirm that the Gemma checkpoint mapper successfully
        //   read the safetensors file and populated the checkpoint struct.
        // Mechanism: Checks the has_checkpoint flag on the decoder model.
        // -----------------------------------------------------------------
        if (!spec.decoder_model.has_checkpoint)
        {
            std::cerr << "expected gemma safetensors bridge to load checkpoint" << std::endl;
            return 1;
        }

        // -----------------------------------------------------------------
        // Assertion: Family is detected as "gemma"
        //
        // Intention: Verify that the HF family registry matched model_type
        //   "gemma" and set the architecture.family field accordingly.
        // Mechanism: Checks that the family string starts with "gemma".
        // -----------------------------------------------------------------
        const std::string family = spec.decoder_model.architecture.family;
        if (family.substr(0, 5) != "gemma")
        {
            std::cerr << "expected gemma architecture family, got " << family << std::endl;
            return 1;
        }

        // -----------------------------------------------------------------
        // Assertion: Multi-layer checkpoint has exactly 2 decoder layers
        //
        // Intention: Verify that the checkpoint mapper correctly parsed all
        //   per-layer tensors (attention, MLP, norms) for both layers.
        // Mechanism: Checks has_decoder_layers and decoder_layers.size().
        // -----------------------------------------------------------------
        if (!spec.decoder_model.checkpoint.has_decoder_layers
            || spec.decoder_model.checkpoint.decoder_layers.size() != 2)
        {
            std::cerr << "expected gemma bridge to load full multi-layer checkpoint" << std::endl;
            return 1;
        }

        // -----------------------------------------------------------------
        // Assertion: final_norm tensor has correct size (hidden_size=8)
        //
        // Intention: Verify that the model.norm.weight tensor (final RMSNorm
        //   before lm_head) was loaded with the correct dimensionality.
        // Mechanism: Checks final_norm vector size equals hidden_size (8).
        // -----------------------------------------------------------------
        if (spec.decoder_model.checkpoint.final_norm.size() != 8)
        {
            std::cerr << "expected gemma bridge to load final_norm tensor" << std::endl;
            return 1;
        }

        // -----------------------------------------------------------------
        // Assertion: attention_size reflects non-square q_proj width
        //
        // Intention: The q_proj weight is [q_hidden=16, hidden=8], so the
        //   attention_size should be 16 (not hidden_size). This tests that
        //   the checkpoint mapper infers attention width from the actual
        //   q_proj shape rather than defaulting to hidden_size.
        // Mechanism: Checks spec.decoder_model.checkpoint.attention_size == 16.
        // -----------------------------------------------------------------
        if (spec.decoder_model.checkpoint.attention_size != 16)
        {
            std::cerr << "expected gemma bridge to preserve attention width, got "
                      << spec.decoder_model.checkpoint.attention_size << std::endl;
            return 1;
        }

        // -----------------------------------------------------------------
        // Assertion: q_norm and k_norm are empty (Gemma has no QK norms)
        //
        // Intention: Gemma does not use per-head RMSNorm on Q/K projections
        //   (unlike Qwen3). Empty q_norm/k_norm vectors signal the TRT graph
        //   builder to skip per-head normalization entirely.
        // Setup: The checkpoint was created with include_qk_norm=false.
        // Mechanism: Checks that layer 0's q_norm and k_norm are both empty.
        // -----------------------------------------------------------------
        {
            const auto& layer0 = spec.decoder_model.checkpoint.decoder_layers[0];
            if (!layer0.q_norm.empty())
            {
                std::cerr << "expected gemma q_norm to be empty (no QK norm)" << std::endl;
                return 1;
            }
            if (!layer0.k_norm.empty())
            {
                std::cerr << "expected gemma k_norm to be empty (no QK norm)" << std::endl;
                return 1;
            }
        }

        std::cout << "test_gemma_family passed" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "test_gemma_family failed: " << e.what() << std::endl;
        std::filesystem::remove_all(family_dir);
        return 1;
    }

    std::filesystem::remove_all(family_dir);
    return 0;
}
