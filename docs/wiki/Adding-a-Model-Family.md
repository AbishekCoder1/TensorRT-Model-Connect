# Adding a New Model Family

Adding support for a new HuggingFace model family is a **Python-only task** in the `trtf_build/` package. No C++ changes are needed — the C++ runtime is family-agnostic and only loads pre-built `.trtfb` bundles.

## Prerequisites

Verify your model follows the standard dense decoder pattern:
- **Pre-RMSNorm** before attention
- **Grouped Query Attention** (GQA) or standard multi-head attention
- **Rotary Position Embeddings** (RoPE)
- **SwiGLU MLP** (gate + up projection with SiLU activation)
- **Post-Attention RMSNorm** before MLP

If yes, the standard decoder builder handles the graph — you only need a plugin file with weight mapping.

If your model diverges (MoE, parallel attention, different norm types), you will need a custom `build_engine()` — see [Advanced: Custom Build Engine](#advanced-custom-build-engine) below.

## Quick Path: Scaffolding Script

The fastest way to add a new family:

```bash
# 1. Generate a plugin from a HuggingFace model's config.json
python3 scripts/new_family.py \
  --model-type phi3 \
  --hf-repo microsoft/Phi-3-mini-4k-instruct \
  --family-name phi

# 2. Review the generated plugin (customize if needed)
$EDITOR trtf_build/trtf_build/families/phi.py

# 3. Validate end-to-end (build + diff_logits + diff_layers + runner parity)
./scripts/validate_family.sh microsoft/Phi-3-mini-4k-instruct
```

The scaffolding script:
- Downloads `config.json` from the HF repo
- Detects architecture features (GQA, tied embeddings, explicit head_dim, MoE, etc.)
- Generates a plugin `.py` with correct `matches()`, standard `load_weights()` and `build_engine()`
- Adds comments noting detected features that may need attention

## Manual Path: Step-by-Step

### Step 1: Create the plugin file

Create `trtf_build/trtf_build/families/<family>.py`. The file must:
- Define a class implementing the `FamilyPlugin` protocol (see `base.py`)
- Expose a module-level `plugin` attribute (instance of the class)

```python
"""Yi family plugin."""

from __future__ import annotations

from ..config import ModelConfig
from ..checkpoint_mapper import WeightDict, load_standard_weights
from ..standard_decoder_builder import build_standard_decoder_engine


class YiPlugin:
    name = "yi"

    def matches(self, model_type: str) -> bool:
        return model_type.lower().startswith("yi")

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        return load_standard_weights(model_dir, config)

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, verbose: bool = False,
    ) -> bytes:
        return build_standard_decoder_engine(
            config, weights, max_cache_length, verbose=verbose)


plugin = YiPlugin()
```

That's it — the plugin is auto-discovered. `families/__init__.py` uses `pkgutil.iter_modules()` to find any `.py` file with a `plugin` attribute. No registration code needed.

### Step 2: Customize weight loading (if needed)

Most models use standard HF tensor naming (same as LLaMA):
```
model.embed_tokens.weight
model.layers.N.input_layernorm.weight
model.layers.N.self_attn.{q,k,v,o}_proj.weight
model.layers.N.post_attention_layernorm.weight
model.layers.N.mlp.{gate,up,down}_proj.weight
model.norm.weight
lm_head.weight
```

If your model uses standard naming, `load_standard_weights()` handles everything — including transposing, GQA expansion, tied embeddings, and optional q/k norms.

For non-standard models, customize `load_weights()`. Example from Gemma (adds +1.0 to RMSNorm gamma, scales embedding):

```python
def load_weights(self, model_dir: str, config: ModelConfig) -> WeightDict:
    weights = load_standard_weights(model_dir, config)

    # Gemma: (1 + gamma) * normalized
    for i in range(config.num_hidden_layers):
        weights[f"layer.{i}.input_norm"] += 1.0
        weights[f"layer.{i}.post_attn_norm"] += 1.0
    weights["final_norm"] += 1.0

    # Gemma: scale embedding by sqrt(hidden_size)
    weights["embedding"] *= math.sqrt(config.hidden_size)
    return weights
```

Some models use fused projections (e.g., Phi-3 ships a single `qkv_proj` instead of separate Q/K/V, and a single `gate_up_proj` instead of separate gate/up). In these cases, split the fused tensor during weight loading. See `trtf_build/trtf_build/families/phi.py` for an example.

### Step 3: Validate

Run the one-command validation gate:

```bash
./scripts/validate_family.sh <hf-repo-or-local-path>
```

This runs:
1. `trtf-build build` — builds a `.trtfb` bundle
2. `diff_logits.py --battery` — E2E logit comparison (4 prompts)
3. `diff_layers.py` — per-layer hidden state comparison
4. `test_runner_parity.py` — Python-vs-C++ cross-validation

Or run each step individually:

```bash
# Build bundle
trtf-build build <model> -o /tmp/test.trtfb --max-cache-length 256

# E2E logit comparison (per-step, all tokens)
python3 scripts/diff_logits.py --model <model> --atol 1e-3 --battery

# Per-layer hidden state comparison
python3 scripts/diff_layers.py --model <model> --atol 0.05

# Python-vs-C++ runner parity
python3 scripts/test_runner_parity.py \
  --bundle /tmp/test.trtfb --binary ./build/trtf \
  --hf-python .venv/bin/python --max-new-tokens 20
```

For models that require custom tokenizer code (e.g., Phi-3), add `--trust-remote-code` to diff_logits.py, diff_layers.py, and validate_family.sh.

**Memory note**: Large models (3B+ parameters) can require significant RAM during TRT engine compilation. Phi-3-mini (3.8B) peaks at ~44GB. On 64GB machines, 16GB swap is recommended.

## Checklist

| Step | Files | Lines |
|------|-------|-------|
| Plugin file | `families/<family>.py` | ~30 (standard), more if custom |
| **Total** | **1 new file, 0 existing files edited** | **~30** |

## FamilyPlugin Protocol

From `trtf_build/trtf_build/families/base.py`:

```python
class FamilyPlugin(Protocol):
    name: str

    def matches(self, model_type: str) -> bool: ...
    def load_weights(self, model_dir: str, config: ModelConfig) -> WeightDict: ...
    def build_engine(self, config: ModelConfig, weights: WeightDict,
                     max_cache_length: int, *, verbose: bool = False) -> bytes: ...
```

## Advanced: Custom Build Engine

If your model has a non-standard architecture (MoE, parallel attention, different norm types), override `build_engine()` to use custom graph construction. The shared TRT graph ops in `trtf_build/graph_ops.py` (RMSNorm, RoPE, matmul, attention, SwiGLU, etc.) are reusable building blocks — compose them differently for your architecture.

Examples of when custom `build_engine()` is needed:
- **MoE (Mixture of Experts)**: Expert routing instead of dense MLP
- **MLA (Multi-head Latent Attention)**: Compressed KV cache
- **Parallel attention**: Attention and MLP computed in parallel (GPT-J style)
- **Different normalization**: LayerNorm instead of RMSNorm
