// =============================================================================
// Test: Qwen model family registration and checkpoint loading.
//
// Purpose:
//   Validates the end-to-end integration of the Qwen model family through the
//   HF family registry and the built-in QWEN3 model alias. This exercises
//   Registry 1 (HfModelFamily) and Registry 2 (CheckpointMapper) for the Qwen
//   architecture, plus the built-in model alias resolution and C ABI pipeline
//   creation.
//
// What is verified:
//   - HF family detection: model_type "qwen2" + "Qwen2ForCausalLM" triggers
//     the Qwen family registration.
//   - Multi-layer checkpoint loading: a 2-layer safetensors checkpoint is read
//     and mapped through QwenCheckpointMapper with q_norm/k_norm included.
//   - Decoder-definition precedence: synthetic HF dir resolves as
//     kDecoderDefinition (not kHuggingFaceLocal).
//   - Built-in QWEN3 alias: the string "QWEN3" resolves to the bundled Qwen3
//     model at models/hf/Qwen__Qwen3-0.6B or models/hf/qwen3.
//   - C ABI pipeline creation: trtf_create_pipeline("QWEN3", PREFER_TRT)
//     produces a working pipeline that generates non-empty text.
//   - Attention width preservation: non-square q_proj is correctly tracked.
//
// Dependencies:
//   - test_helpers.h: make_temp_dir_or_throw, write_file, write_standard_decoder_checkpoint
//   - trtf/model_resolver.h: ResolveTextGenerationModel, ResolvedModelSpec
//   - trtf/pipeline.h: trtf_create_pipeline, trtf_last_error (C ABI)
//   - Qwen family registration (auto-discovered via CMake GLOB)
//   - Built-in QWEN3 model data at models/hf/ (bundled with the project)
//
// Approach:
//   1. Creates a synthetic 2-layer Qwen checkpoint in a temporary directory with
//      config.json (model_type="qwen2") and model.safetensors (with QK norms),
//      resolves it, and validates the ResolvedModelSpec.
//   2. Resolves the built-in "QWEN3" alias and validates it also produces a
//      decoder-definition with loaded checkpoint.
//   3. Creates a pipeline via the C ABI and generates text to validate the full
//      end-to-end path. Pipeline creation may fail gracefully if no TRT/HF-Python
//      backend is available in the test environment.
// =============================================================================

#include "test_helpers.h"
#include "trtf/model_resolver.h"
#include "trtf/pipeline.h"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

// ---------------------------------------------------------------------------
// write_hf_qwen_root
//
// Intention: Create a minimal but structurally valid HF Qwen model directory.
// Setup:
//   - Writes config.json with model_type="qwen2", architectures=["Qwen2ForCausalLM"],
//     vocab_size=8, hidden_size=8, 2 layers, 2 attention heads, 1 KV head (GQA).
//   - Writes model.safetensors via write_standard_decoder_checkpoint with
//     q_hidden=16, kv_hidden=8, mlp=16, 2 layers, include_qk_norm=true.
//     The include_qk_norm=true flag adds self_attn.q_norm.weight and
//     self_attn.k_norm.weight tensors per layer, which is a distinguishing
//     feature of the Qwen3 architecture.
// Mechanism: Delegates to test helpers for file and safetensors generation.
// ---------------------------------------------------------------------------
void write_hf_qwen_root(const std::filesystem::path& dir)
{
    trtf_test::write_file(dir / "config.json",
        "{\n"
        "  \"model_type\": \"qwen2\",\n"
        "  \"architectures\": [\"Qwen2ForCausalLM\"],\n"
        "  \"vocab_size\": 8,\n"
        "  \"hidden_size\": 8,\n"
        "  \"num_hidden_layers\": 2,\n"
        "  \"num_attention_heads\": 2,\n"
        "  \"num_key_value_heads\": 1,\n"
        "  \"rms_norm_eps\": 1e-5,\n"
        "  \"rope_theta\": 10000.0\n"
        "}\n");
    trtf_test::write_standard_decoder_checkpoint(dir, 8, 8, 16, 8, 16, 2, true);
}

} // namespace

int main()
{
    std::filesystem::path without_definition;

    try
    {
        without_definition = trtf_test::make_temp_dir_or_throw("/tmp/trtf_qwen_family_without_def_XXXXXX");

        write_hf_qwen_root(without_definition);

        // =================================================================
        // Section 1: Synthetic Qwen HF directory resolution
        // =================================================================

        // -----------------------------------------------------------------
        // Assertion: Synthetic Qwen dir resolves as kDecoderDefinition
        //
        // Intention: Verify that the family registry recognizes model_type
        //   "qwen2" + safetensors and resolves it as a decoder-definition
        //   (not kHuggingFaceLocal). This confirms that the Qwen family
        //   registration has higher priority than the generic HF fallback.
        // Mechanism: Calls ResolveTextGenerationModel on the temp dir path
        //   and checks fallback_spec.kind.
        // -----------------------------------------------------------------
        const trtf::ResolvedModelSpec fallback_spec = trtf::ResolveTextGenerationModel(without_definition.string());
        if (fallback_spec.kind != trtf::ResolvedModelKind::kDecoderDefinition)
        {
            std::cerr << "expected qwen hf dir without decoder-definition to resolve through qwen safetensors bridge"
                      << std::endl;
            return 1;
        }

        // -----------------------------------------------------------------
        // Assertion: Checkpoint was loaded (has_checkpoint = true)
        //
        // Intention: Confirm that the Qwen checkpoint mapper successfully
        //   read the safetensors file and populated the checkpoint struct.
        // Mechanism: Checks the has_checkpoint flag on the decoder model.
        // -----------------------------------------------------------------
        if (!fallback_spec.decoder_model.has_checkpoint)
        {
            std::cerr << "expected qwen root safetensors bridge path to load checkpoint" << std::endl;
            return 1;
        }

        // -----------------------------------------------------------------
        // Assertion: Family is detected as "qwen" or "qwq"
        //
        // Intention: Verify that the HF family registry matched model_type
        //   "qwen2" and set a Qwen-family architecture string. The family
        //   could be "qwen", "qwen2", "qwen3", or "qwq" depending on which
        //   registration entry matched.
        // Mechanism: Checks the family string starts with "qwen" or "qwq".
        // -----------------------------------------------------------------
        {
            const std::string family = fallback_spec.decoder_model.architecture.family;
            if (family.substr(0, 4) != "qwen" && family.substr(0, 3) != "qwq")
            {
                std::cerr << "expected qwen root path to set a qwen/qwq architecture family, got "
                          << family << std::endl;
                return 1;
            }
        }

        // -----------------------------------------------------------------
        // Assertion: Multi-layer checkpoint has exactly 2 decoder layers
        //
        // Intention: Verify that the checkpoint mapper correctly parsed all
        //   per-layer tensors (attention, MLP, norms, q_norm, k_norm) for
        //   both layers.
        // Mechanism: Checks has_decoder_layers and decoder_layers.size().
        // -----------------------------------------------------------------
        if (!fallback_spec.decoder_model.checkpoint.has_decoder_layers
            || fallback_spec.decoder_model.checkpoint.decoder_layers.size() != 2)
        {
            std::cerr << "expected qwen root bridge to load full multi-layer qwen checkpoint tensors" << std::endl;
            return 1;
        }

        // -----------------------------------------------------------------
        // Assertion: final_norm tensor has correct size (hidden_size=8)
        //
        // Intention: Verify that the model.norm.weight tensor (final RMSNorm
        //   before lm_head) was loaded with the correct dimensionality.
        // Mechanism: Checks final_norm vector size equals hidden_size (8).
        // -----------------------------------------------------------------
        if (fallback_spec.decoder_model.checkpoint.final_norm.size() != 8)
        {
            std::cerr << "expected qwen root bridge to load final_norm tensor" << std::endl;
            return 1;
        }

        // -----------------------------------------------------------------
        // Assertion: attention_size reflects non-square q_proj width
        //
        // Intention: The q_proj weight is [q_hidden=16, hidden=8], so the
        //   attention_size should be 16. This verifies the checkpoint mapper
        //   infers attention width from the actual q_proj shape.
        // Mechanism: Checks attention_size == 16.
        // -----------------------------------------------------------------
        if (fallback_spec.decoder_model.checkpoint.attention_size != 16)
        {
            std::cerr << "expected qwen root bridge to preserve non-square q attention width" << std::endl;
            return 1;
        }

        // =================================================================
        // Section 2: Built-in QWEN3 alias resolution
        // =================================================================

        // -----------------------------------------------------------------
        // Assertion: "QWEN3" alias resolves as kDecoderDefinition
        //
        // Intention: Verify that the built-in model alias "QWEN3" (which
        //   points to models/hf/Qwen__Qwen3-0.6B or models/hf/qwen3) is
        //   recognized by the model resolver and produces a valid
        //   decoder-definition with loaded checkpoint data.
        // Setup: Uses the bundled model data shipped with the project.
        // Mechanism: Resolves "QWEN3" string and checks kind, model_id,
        //   and has_checkpoint.
        // -----------------------------------------------------------------
        const trtf::ResolvedModelSpec builtin_alias_spec = trtf::ResolveTextGenerationModel("QWEN3");
        if (builtin_alias_spec.kind != trtf::ResolvedModelKind::kDecoderDefinition)
        {
            std::cerr << "expected built-in QWEN3 alias to resolve as decoder-definition" << std::endl;
            return 1;
        }
        if (builtin_alias_spec.decoder_model.model_id != "QWEN3")
        {
            std::cerr << "expected built-in QWEN3 alias to preserve requested model_id" << std::endl;
            return 1;
        }
        if (!builtin_alias_spec.decoder_model.has_checkpoint)
        {
            std::cerr << "expected built-in QWEN3 alias to load checkpoint tensors" << std::endl;
            return 1;
        }

        // =================================================================
        // Section 3: C ABI end-to-end pipeline test
        // =================================================================

        // -----------------------------------------------------------------
        // Assertion: C ABI pipeline generates non-empty text
        //
        // Intention: Validate the full end-to-end path from C ABI pipeline
        //   creation through text generation. Uses TRTF_PREFER_TRT which
        //   will use TRT if available, otherwise falls back to HF-Python.
        // Setup: Calls trtf_create_pipeline("QWEN3", TRTF_PREFER_TRT).
        // Mechanism: If pipeline creation succeeds, generates 3 tokens from
        //   "Hello" and verifies the result is non-null and non-empty. If
        //   pipeline creation fails (no backend available in test env), the
        //   failure is logged but treated as acceptable (not a test failure).
        // -----------------------------------------------------------------
        auto* pipeline = trtf_create_pipeline("QWEN3", TRTF_PREFER_TRT);
        if (pipeline != nullptr)
        {
            const char* result = pipeline->generate("Hello", 3);
            if (result == nullptr || std::strlen(result) == 0)
            {
                std::cerr << "expected non-empty generated text for QWEN3 via C ABI" << std::endl;
                delete pipeline;
                std::filesystem::remove_all(without_definition);
                return 1;
            }
            delete pipeline;
        }
        else
        {
            // Acceptable if no TRT or HF-Python backend is available in test env
            std::cerr << "note: QWEN3 pipeline creation skipped (no backend): "
                      << trtf_last_error() << std::endl;
        }

        std::cout << "test_qwen_family passed" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "test_qwen_family failed: " << e.what() << std::endl;
        std::filesystem::remove_all(without_definition);
        return 1;
    }

    std::filesystem::remove_all(without_definition);
    return 0;
}
