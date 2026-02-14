# Model Family Implementation Agent: __Family__

## Your Task

Implement the **__Family__** model family for the trt-transformers-cpp framework. This is a self-contained prompt — all code patterns are included inline. You do NOT need to read reference files.

The framework uses a 4-registry plug-and-play architecture. For standard dense decoders (Pre-RMSNorm + GQA + RoPE + SwiGLU), you only need to create 3 source files + 1 test file, then add integration lines to 2 existing files.

## Parameters

| Parameter | Value |
|-----------|-------|
| `family` (lowercase, C++ namespace) | `__family__` |
| `Family` (PascalCase, class names) | `__Family__` |
| `model_type_prefix` (HF config.json match) | `__model_type_prefix__` |
| `architectures` (HF architectures array) | `__architectures__` |
| `has_qk_norm` (per-head Q/K RMSNorm) | `__has_qk_norm__` |
| `hf_repo_id` (for E2E validation) | `__hf_repo_id__` |

## CRITICAL: Sandbox and Execution Requirements

- **ALL docker exec commands MUST use `dangerouslyDisableSandbox: true`** — the container has its own network access for HuggingFace downloads and GPU access for TRT inference. The host sandbox cannot see inside the container.
- **You MUST complete ALL validation steps** (build, test, download, TRT E2E) before returning. Do not return with "blocked" status — debug and fix issues autonomously.
- **Verify `backend=trt`** in inference output. If you see `backend=hf-transformers` or `backend=cpu-reference`, the TRT path is not being exercised.
- **If sharded model downloads fail**, check that both `"model.safetensors"` AND `"model-*.safetensors"` patterns are in the download call — sharded models use `model-00001-of-NNNNN.safetensors` filenames.

## Step 0: Create Branch

```bash
git checkout -b model/__family__
```

## Step 1: Create Source Files

### File: `src/models/__family__/checkpoint_mapper.h`

```cpp
#pragma once

#include "model/standard_checkpoint_mapper.h"

namespace trtf {
namespace __family__ {

class __Family__CheckpointMapper final : public StandardCheckpointMapper {
public:
    bool can_map(const DecoderArchitectureConfig& architecture) const override;
};

} // namespace __family__
} // namespace trtf
```

### File: `src/models/__family__/checkpoint_mapper.cpp`

```cpp
#include "models/__family__/checkpoint_mapper.h"
#include "utils/text_parsers.h"

namespace trtf {
namespace __family__ {

bool __Family__CheckpointMapper::can_map(const DecoderArchitectureConfig& architecture) const
{
    const std::string fam = to_lower_ascii(architecture.family);
    return starts_with(fam, "__model_type_prefix__");
}

} // namespace __family__
} // namespace trtf
```

### File: `src/models/__family__/registration.h`

```cpp
#pragma once

namespace trtf {
namespace __family__ {

void Register__Family__Family();

} // namespace __family__
} // namespace trtf
```

### File: `src/models/__family__/registration.cpp`

```cpp
#include "models/__family__/registration.h"
#include "models/__family__/checkpoint_mapper.h"
#include "trtf/hf_family_registry.h"
#include "model/checkpoint_mapper.h"
#include "runtime/trt/trt_graph_builder.h"
#include "runtime/trt/trt_common.h"
#include "runtime/trt/standard_decoder_graph_builder.h"
#include "utils/text_parsers.h"
#include "utils/json_helpers.h"

#include <filesystem>
#include <memory>
#include <string>

namespace trtf {
namespace __family__ {
namespace {

bool is___family___model_type(const std::string& model_type)
{
    if (model_type.empty())
    {
        return false;
    }
    const std::string lowered = to_lower_ascii(model_type);
    return starts_with(lowered, "__model_type_prefix__");
}

bool has___family___root_checkpoint(const HfModelMetadata& metadata)
{
    const std::filesystem::path model_dir(metadata.model_dir);
    return std::filesystem::exists(model_dir / "model.safetensors")
        || std::filesystem::exists(model_dir / "model.safetensors.index.json");
}

DecoderModel load___family___model(const HfModelMetadata& metadata)
{
    DecoderModel model = LoadDecoderModel(metadata.model_dir);

    const int32_t env_max_cache = parse_positive_env_int("TRTF_MAX_CACHE_LENGTH", -1);
    if (env_max_cache <= 0 && model.max_cache_length > 4096)
    {
        model.max_cache_length = 4096;
    }
    return model;
}

} // namespace

void Register__Family__Family()
{
    RegisterCheckpointMapper("__family__", 100, std::make_unique<__Family__CheckpointMapper>());

#if TRTF_HAS_TRT
    RegisterTrtGraphBuilder("__family__", std::make_unique<StandardDecoderGraphBuilder>());
    if (!FindTrtGraphBuilder("standard-decoder"))
    {
        RegisterTrtGraphBuilder("standard-decoder", std::make_unique<StandardDecoderGraphBuilder>());
    }
#endif

    RegisterHfModelFamily({
        "__family__-safetensors",
        100,
        [](const HfModelMetadata& metadata) {
            if (!is___family___model_type(metadata.model_type))
            {
                return false;
            }
            return has___family___root_checkpoint(metadata);
        },
        [](const HfModelMetadata& metadata) { return load___family___model(metadata); },
    });
}

} // namespace __family__
} // namespace trtf
```

## Step 2: Create Test File

### File: `tests/test___family___family.cpp`

```cpp
// Test: __Family__ model family registration and checkpoint loading.
// Verifies: HF family detection for model_type "__model_type_prefix__", multi-layer safetensors
// bridge __qk_norm_test_comment__, GQA layout with num_kv_heads < num_attention_heads.

#include "test_helpers.h"
#include "trtf/model_resolver.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace {

void write_hf___family___root(const std::filesystem::path& dir)
{
    trtf_test::write_file(dir / "config.json",
        "{\n"
        "  \"model_type\": \"__model_type_prefix__\",\n"
        "  \"architectures\": [\"__architectures__\"],\n"
        "  \"vocab_size\": 8,\n"
        "  \"hidden_size\": 8,\n"
        "  \"num_hidden_layers\": 2,\n"
        "  \"num_attention_heads\": 2,\n"
        "  \"num_key_value_heads\": 1,\n"
        "  \"rms_norm_eps\": 1e-5,\n"
        "  \"rope_theta\": 10000.0\n"
        "}\n");
    trtf_test::write_standard_decoder_checkpoint(dir, 8, 8, 16, 8, 16, 2, __has_qk_norm__);
}

} // namespace

int main()
{
    std::filesystem::path family_dir;

    try
    {
        family_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf___family___family_XXXXXX");

        write_hf___family___root(family_dir);

        const trtf::ResolvedModelSpec spec = trtf::ResolveTextGenerationModel(family_dir.string());
        if (spec.kind != trtf::ResolvedModelKind::kDecoderDefinition)
        {
            std::cerr << "expected __family__ hf dir to resolve as decoder-definition" << std::endl;
            return 1;
        }
        if (spec.decoder_model.model_id != family_dir.string())
        {
            std::cerr << "expected resolved model_id to track original hf dir" << std::endl;
            return 1;
        }
        if (!spec.decoder_model.has_checkpoint)
        {
            std::cerr << "expected __family__ safetensors bridge to load checkpoint" << std::endl;
            return 1;
        }

        // Verify the family is detected as __family__.
        {
            const std::string family = spec.decoder_model.architecture.family;
            if (family.substr(0, __model_type_prefix_len__) != "__model_type_prefix__")
            {
                std::cerr << "expected __family__ architecture family, got " << family << std::endl;
                return 1;
            }
        }

        // Verify multi-layer checkpoint was loaded.
        if (!spec.decoder_model.checkpoint.has_decoder_layers
            || spec.decoder_model.checkpoint.decoder_layers.size() != 2)
        {
            std::cerr << "expected __family__ bridge to load full multi-layer checkpoint tensors" << std::endl;
            return 1;
        }

        // Verify final_norm was loaded.
        if (spec.decoder_model.checkpoint.final_norm.size() != 8)
        {
            std::cerr << "expected __family__ bridge to load final_norm tensor" << std::endl;
            return 1;
        }

        // Verify attention_size is preserved (q_proj has q_hidden=16 with hidden=8).
        if (spec.decoder_model.checkpoint.attention_size != 16)
        {
            std::cerr << "expected __family__ bridge to preserve non-square q attention width, got "
                      << spec.decoder_model.checkpoint.attention_size << std::endl;
            return 1;
        }

        // Verify q_norm/k_norm based on family's has_qk_norm setting.
        {
            const auto& layer0 = spec.decoder_model.checkpoint.decoder_layers[0];
__qk_norm_assertions__
        }

        std::cout << "test___family___family passed" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "test___family___family failed: " << e.what() << std::endl;
        std::filesystem::remove_all(family_dir);
        return 1;
    }

    std::filesystem::remove_all(family_dir);
    return 0;
}
```

## Step 3: Integrate into Existing Files

### Edit `src/model/hf_family_registry.cpp`

Add this include near the other model family includes (after `#include "models/llama/registration.h"`):

```cpp
#include "models/__family__/registration.h"
```

Add this call inside `RegisterBuiltinHfModelFamilies()`, after `llama::RegisterLlamaFamily();`:

```cpp
    __family__::Register__Family__Family();
```

### Edit `CMakeLists.txt`

Add these source files to the `trtf_core` library (after the llama source lines):

```cmake
  src/models/__family__/registration.cpp
  src/models/__family__/checkpoint_mapper.cpp
```

Add the test executable (after the `test_llama_family` block):

```cmake
add_executable(test___family___family tests/test___family___family.cpp)
target_link_libraries(test___family___family PRIVATE trtf_core)
add_test(NAME test___family___family COMMAND test___family___family)
```

## Step 4: Host Build and Test

```bash
cmake -S . -B build -G Ninja
cmake --build build -j
ctest --test-dir build -R test___family___family --output-on-failure
```

Expected: `test___family___family` passes. Other tests should remain unchanged.

Run full test suite:
```bash
ctest --test-dir build --output-on-failure
```

Expected: Same pass/fail as baseline. Tests that use temp dirs (test_model_loader, test_model_resolver, test_hf_family_registry, test_qwen_family, test_llama_family, test___family___family) may fail on read-only `/tmp` but pass in container.

## Step 5: Container Build and Test

```bash
docker exec trtf-dev bash -c 'cmake --build build-container-phase1 -j'
docker exec trtf-dev bash -c 'ctest --test-dir build-container-phase1 --output-on-failure'
```

Expected: ALL tests pass including `test___family___family`.

## Step 6: Download Real Model Weights (Container)

```bash
docker exec trtf-dev bash -c '
source .venv-hf/bin/activate
python3 - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="__hf_repo_id__",
    local_dir="models/hf/__hf_local_dir__",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "config.json", "generation_config.json",
        "model.safetensors", "model.safetensors.index.json",
        "model-*.safetensors",
        "tokenizer.json", "tokenizer_config.json", "vocab.json",
        "merges.txt", "special_tokens_map.json", "*.model",
    ],
)
PYEOF
'
```

## Step 7: TRT E2E Validation (Container, GPU required)

```bash
docker exec trtf-dev bash -c '
TRTF_HF_PYTHON=$PWD/.venv-hf/bin/python \
TRTF_MAX_NEW_TOKENS=30 \
TRTF_MAX_CACHE_LENGTH=256 \
./build-container-phase1/trtf_load_model --force-trt models/hf/__hf_local_dir__ "The capital of France is"
'
```

Expected: Coherent text output, `backend=trt`.

## Step 8: Numerical Parity Check (Container, GPU required)

```bash
docker exec trtf-dev bash -c '
source .venv-hf/bin/activate
python3 scripts/diff_logits.py \
  --model-dir models/hf/__hf_local_dir__ \
  --binary ./build-container-phase1/trtf_load_model \
  --backend-flag --force-trt --atol 1e-3 --battery
'
```

Expected: All logit comparisons within tolerance.

## Step 9: Commit

```bash
git add src/models/__family__/ tests/test___family___family.cpp src/model/hf_family_registry.cpp CMakeLists.txt
git commit -m "feat: Add __Family__ model family support

Registers __Family__ into the 4-registry plug-and-play architecture:
- Registry 1 (HF Family): matches model_type '__model_type_prefix__'
- Registry 2 (Checkpoint Mapper): StandardCheckpointMapper subclass
- Registry 3 (TRT Populator): uses StandardTrtModelDefinitionPopulator fallback
- Registry 4 (TRT Graph Builder): StandardDecoderGraphBuilder

Validated: host build, container build, unit tests, TRT E2E, diff_logits parity."
```

## Important Notes

- The `StandardCheckpointMapper` base class handles ALL standard HF tensor key mapping (model.embed_tokens, model.layers.N.self_attn.*, model.layers.N.mlp.*, model.norm, lm_head). Your subclass only overrides `can_map()`.
- The `StandardTrtModelDefinitionPopulator` is registered as a priority-0 fallback globally. You do NOT register your own populator (Registry 3 is automatic).
- The `StandardDecoderGraphBuilder` handles Pre-RMSNorm + GQA + RoPE + SwiGLU. Register it for your family name in Registry 4.
- Per-head Q/K RMSNorm is auto-detected from safetensors (present = loaded, absent = skipped). The `__has_qk_norm__` parameter only affects the test assertions.
- All utility functions (`starts_with`, `to_lower_ascii`, `parse_positive_env_int`, `read_file`, `extract_json_string`, etc.) are already available in `src/utils/`.

## Tier 2 Extension: Custom Graph Builder

If the model uses a non-standard architecture (MoE, parallel attention, partial rotary, LayerNorm instead of RMSNorm), you need a custom `ITrtGraphBuilder` instead of `StandardDecoderGraphBuilder`.

### Custom Graph Builder Template

Create `src/models/__family__/trt_graph_builder.h`:

```cpp
#pragma once

#include "runtime/trt/trt_graph_builder.h"

namespace trtf {
namespace __family__ {

#if TRTF_HAS_TRT

class __Family__TrtGraphBuilder final : public ITrtGraphBuilder {
public:
    std::unique_ptr<DecoderStepEngine> build_decoder_step_engine(
        const TrtDecoderDefinition& weights, TrtLogger& logger) override;
};

#endif

} // namespace __family__
} // namespace trtf
```

Create `src/models/__family__/trt_graph_builder.cpp` implementing the build method using reusable TRT graph ops from `src/runtime/trt/trt_graph_ops.h`:

Available ops:
- `add_rms_norm(network, input, hidden_size, gamma, eps_tensor)` - RMSNorm
- `add_rms_norm_per_head(network, input, num_heads, head_dim, gamma, eps_tensor)` - Per-head RMSNorm
- `add_apply_rope(network, input, position_id, cos_table, sin_table, rotate_half_matrix)` - RoPE
- `add_matmul_rhs_constant(network, lhs, lhs_width, rhs_width, rhs_weights)` - MatMul with constant RHS
- `add_bias_sum(network, input, width, bias)` - Add bias
- `add_constant_tensor(network, dims, values)` - Constant tensor
- `make_rope_table(max_cache_length, hidden_size, num_attention_heads, rope_theta, cosine)` - RoPE cos/sin tables
- `make_rotate_half_matrix(hidden_size, num_attention_heads)` - Rotate-half matrix for RoPE
- `make_dims_1d/2d/3d(...)` - TRT dimension helpers
- `layer_tensor_name(stem, layer)` - "stem_L{layer}" naming

Reference `src/runtime/trt/standard_decoder_graph_builder.cpp` for the full standard decoder implementation to adapt from.

In registration.cpp, replace `StandardDecoderGraphBuilder` with your custom builder:

```cpp
#if TRTF_HAS_TRT
    RegisterTrtGraphBuilder("__family__", std::make_unique<__Family__TrtGraphBuilder>());
#endif
```
