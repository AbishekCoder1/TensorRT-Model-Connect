# Adding a New Model Family

This guide walks through adding support for a new HuggingFace model family. Since engine building has migrated to Python, adding a new family is now a **Python-only task** in the `trtf_build/` package. No C++ changes are needed.

## Prerequisites

Before starting, verify your model follows the standard dense decoder pattern:
- **Pre-RMSNorm** before attention
- **Grouped Query Attention** (GQA) or standard multi-head attention
- **Rotary Position Embeddings** (RoPE)
- **SwiGLU MLP** (gate + up projection with SiLU activation)
- **Post-Attention RMSNorm** before MLP

If yes, you only need to write a checkpoint mapper. The shared graph builder handles the rest.

If your model has a non-standard architecture (MoE, parallel attention, different norm types), you will need a custom graph builder -- see [Advanced: Custom Graph Builder](#advanced-custom-graph-builder) at the bottom.

## Step-by-Step Guide

We will use a hypothetical "Yi" model family as our example.

### Step 1: Create the plugin directory

```
trtf_build/families/yi/
  __init__.py
  plugin.py
  checkpoint.py
```

### Step 2: Implement the checkpoint mapper

First, check if your model uses the standard HF tensor naming convention:
```
model.embed_tokens.weight
model.layers.N.input_layernorm.weight
model.layers.N.self_attn.q_proj.weight
model.layers.N.self_attn.k_proj.weight
model.layers.N.self_attn.v_proj.weight
model.layers.N.self_attn.o_proj.weight
model.layers.N.post_attention_layernorm.weight
model.layers.N.mlp.gate_proj.weight
model.layers.N.mlp.up_proj.weight
model.layers.N.mlp.down_proj.weight
model.norm.weight
lm_head.weight
```

**If yes** (most models), use the standard checkpoint mapper and only specify the family name:

**`checkpoint.py`:**
```python
from trtf_build.checkpoint import StandardCheckpointMapper

class YiCheckpointMapper(StandardCheckpointMapper):
    """Yi uses standard HF tensor naming, identical to LLaMA."""
    pass
```

The `StandardCheckpointMapper` base class handles:
- Reading all weight tensors from safetensors
- Transposing weight matrices from `[out, in]` to `[in, out]`
- Expanding K/V projections for GQA (when `num_key_value_heads < num_attention_heads`)
- Repeating per-head norms across all heads
- Handling tied `lm_head` (when `lm_head.weight` is absent, reuses embedding)
- Reading optional `q_norm`/`k_norm` weights

**If your model has non-standard tensor keys**, override the mapping methods. See `trtf_build/families/gemma/checkpoint.py` for an example that adds +1.0 to RMSNorm gamma and scales the embedding.

### Step 3: Write the plugin registration

**`plugin.py`:**
```python
from trtf_build.registry import FamilyPlugin
from .checkpoint import YiCheckpointMapper


def register():
    """Register the Yi model family plugin."""
    return FamilyPlugin(
        family_name="yi",
        model_types=["yi"],       # Matches model_type from config.json
        checkpoint_mapper=YiCheckpointMapper(),
        # Uses the standard decoder graph builder by default.
        # Set graph_builder=YiGraphBuilder() for custom architectures.
    )
```

**`__init__.py`:**
```python
from .plugin import register
```

### Step 4: Write tests

Create Python tests for the new family:

```python
# tests/test_yi_family.py

def test_yi_checkpoint_mapping():
    """Verify Yi checkpoint mapper produces correct canonical tensors."""
    # Create mock safetensors with Yi tensor naming
    # Run mapper
    # Assert canonical format is correct
    ...

def test_yi_family_detection():
    """Verify Yi model_type is correctly matched."""
    # Create mock config.json with model_type: "yi"
    # Assert family plugin claims the model
    ...
```

### Step 5: Build and validate

```bash
# Build a bundle from a Yi model directory
trtf-build build path/to/yi-model -o yi.trtfb --max-cache-length 256

# Inspect the bundle
trtf-build inspect yi.trtfb

# Run inference from the bundle (C++)
trtf run yi.trtfb --prompt "Hello" --max-new-tokens 10 --hf-python $PWD/.venv-hf/bin/python

# E2E logit comparison against HF (in container with GPU)
python3 scripts/diff_logits.py \
  --model-dir path/to/yi-model --binary ./build/trtf \
  --backend-flag --force-trt --atol 1e-3 --battery
```

---

## Checklist Summary

| Step | Files | Approximate lines |
|------|-------|-------------------|
| Checkpoint mapper | `checkpoint.py` | ~10 (if using standard base) |
| Plugin registration | `plugin.py` | ~15 |
| Init | `__init__.py` | ~1 |
| Tests | `test_yi_family.py` | ~30 |
| **Total** | **4 new files, 0 existing files edited** | **~56** |

---

## Advanced: Custom Graph Builder

If your model does not follow the Pre-RMSNorm + GQA + RoPE + SwiGLU pattern, you need a custom graph builder. The shared TRT graph ops in `trtf_build/graph_ops.py` provide reusable building blocks.

Examples of when this is needed:

- **MoE (Mixture of Experts)**: Expert routing layer instead of dense MLP
- **MLA (Multi-head Latent Attention)**: Compressed KV cache
- **Mamba/SSM**: Fundamentally different architecture (no attention)
- **Parallel attention**: Attention and MLP computed in parallel (GPT-J style)
- **Different normalization**: LayerNorm instead of RMSNorm

### Custom graph builder example (MoE)

```python
# trtf_build/families/mixtral/graph_builder.py

from trtf_build.graph_ops import (
    add_rms_norm, add_matmul, add_rope, add_attention,
    add_swiglu_expert, add_topk_routing,
)

class MixtralGraphBuilder:
    """Builds TRT network for Mixtral MoE architecture."""

    def build_layer(self, network, hidden, layer_weights, config):
        # Standard attention (reuse shared ops)
        norm1 = add_rms_norm(network, hidden, layer_weights.input_norm)
        q, k, v = self.project_qkv(network, norm1, layer_weights)
        q, k = add_rope(network, q, k, config)
        attn_out = add_attention(network, q, k, v, config)
        hidden = add_residual(network, hidden, attn_out)

        # MoE MLP (custom)
        norm2 = add_rms_norm(network, hidden, layer_weights.post_attn_norm)
        routing = add_topk_routing(network, norm2, layer_weights.router, top_k=2)
        mlp_out = add_swiglu_expert(network, norm2, layer_weights.experts, routing)
        hidden = add_residual(network, hidden, mlp_out)

        return hidden
```

Register it in the plugin:

```python
# trtf_build/families/mixtral/plugin.py

from trtf_build.registry import FamilyPlugin
from .checkpoint import MixtralCheckpointMapper
from .graph_builder import MixtralGraphBuilder

def register():
    return FamilyPlugin(
        family_name="mixtral",
        model_types=["mixtral"],
        checkpoint_mapper=MixtralCheckpointMapper(),
        graph_builder=MixtralGraphBuilder(),
    )
```

The shared graph ops from `trtf_build/graph_ops.py` (RMSNorm, matmul, RoPE, attention, etc.) are reusable by any custom graph builder -- compose them differently for your architecture.
