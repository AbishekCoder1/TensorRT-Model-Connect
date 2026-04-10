---
name: debug-trt-mismatch
description: >-
  Investigate numerical mismatches between TRT and HuggingFace inference. Escalating
  levels: token-level logit comparison, per-layer hidden state diff, vision-language
  feature analysis, and C++ vs Python parity. Use when an E2E test fails, TRT output
  diverges from HF, or the user reports wrong model output. Invoke via /debug-trt-mismatch.
---

# Debug TRT Mismatch: Numerical Divergence Investigation

## Overview

You are a numerical debugging agent. When TRT inference produces different output
than HuggingFace, systematically isolate where the divergence originates — which
step, which layer, which operation — and recommend fixes. All commands run inside
the dev container.

## When to Use

- E2E test failure (token mismatch, cosine similarity too low)
- User reports TRT output is wrong/garbled
- New family plugin produces incorrect results
- After weight mapping changes or graph op modifications

## Environment Check (ALWAYS run first)

Before running any diff tool, verify the environment:

```bash
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null && echo "GPU: OK" || echo "GPU: MISSING"
python3 -c "import tensorrt as trt; print(f'TRT: {trt.__version__}')" 2>/dev/null || echo "TRT: MISSING"
python3 -c "import trtf_build; print('trtf_build: OK')" 2>/dev/null || echo "trtf_build: MISSING (run: pip install --no-deps -e trtf_build/)"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch: MISSING"
```

For Level 4 (runner parity) only — also check C++ binary:
```bash
test -x ./build/trtf && echo "C++ binary: OK" || echo "C++ binary: MISSING"
```

**If any check fails, auto-recover:**

Check if a team container already exists:
```bash
docker ps -a --filter "name=trtf-dev-gb300" --format "{{.Names}} {{.Status}}"
```

- **Running container exists**: Use `docker exec trtf-dev-gb300-<team-id>` for all
  subsequent commands.
- **Stopped container exists**: `docker start trtf-dev-gb300-<team-id>`, then exec.
- **No container exists**: Bootstrap one:
  ```bash
  ./scripts/bootstrap_workspace.sh --id <team-id> --branch $(git branch --show-current) --detach
  ```
  This creates a container with editable install + C++ build. Then use docker exec.
- **No Docker image**: Build it first: `./scripts/docker_build_gb300.sh`
- **Inside container but trtf_build missing**: `pip install --no-deps -e trtf_build/`
- **Inside container but C++ binary missing** (Level 4 only): `./scripts/setup_container.sh`

After recovery, re-run the environment check. Do NOT proceed to Level 1 until all
required checks pass. When running via docker exec, prefix every diff command with
`docker exec trtf-dev-gb300-<team-id>`.

## Escalation Levels

Always start at Level 1. Only escalate if the lower level doesn't pinpoint the issue.

```
Level 1: diff_logits   →  "Which decode step diverges?"
Level 2: diff_layers   →  "Which transformer layer diverges?"
Level 3: diff_vl       →  "Is it the vision encoder or text decoder?"
Level 4: runner_parity →  "Is it Python TrtRunner or C++ binary?"
Level 5: Manual        →  Graph op isolation (single-op TRT graph vs PyTorch)
```

---

## Level 1: Token-Level Logit Comparison (diff_logits.py)

**Purpose:** Compare per-step logits between TRT and HF to find the first diverging step.

### Quick check (single prompt):

```bash
python tools/diff_logits.py \
  --model <model> \
  --prompt "The capital of France is" \
  --max-new-tokens 10 \
  --atol 1e-3 \
  --verbose
```

### Full battery (recommended for new models):

```bash
python tools/diff_logits.py \
  --model <model> \
  --atol 1e-3 \
  --battery \
  --json /tmp/diff_logits.json
```

The `--battery` flag runs 4 standard prompts covering factual, reasoning, code, and
multi-turn patterns.

### For models requiring custom code:

```bash
python tools/diff_logits.py \
  --model microsoft/Phi-3-mini-4k-instruct \
  --atol 1e-3 --battery --trust-remote-code
```

### Reading the output:

**Key metrics to check:**

| Metric | Good | Concerning | Fail |
|--------|------|------------|------|
| `cosine_p5` (5th percentile cosine sim) | > 0.999 | 0.99 - 0.999 | < 0.99 |
| `top1_match_rate` (argmax agreement) | 1.0 | > 0.9 | < 0.9 |
| `token_agreement` (text match fraction) | 1.0 | > 0.8 | < 0.8 |
| `max_diff` per step | < 1e-3 | 1e-3 to 0.1 | > 0.1 |

**Interpreting failures:**

| Pattern | Likely Cause |
|---------|-------------|
| Step 0 diverges (prefill) | Weight mapping bug in family plugin |
| Divergence grows each step | Accumulating error — check norm precision (FP32 boundary) |
| Sudden divergence at step N | RoPE position mismatch or KV cache corruption |
| High max_diff but correct text | Numerical noise — probably OK, loosen `--atol` |
| Wrong text but low max_diff | Sampling issue (top-1 tie-breaking), not a real bug |
| All prompts fail identically | Systematic issue: weight loading, config parsing, or graph op |

### JSON output fields:

```json
{
  "pass": false,
  "cosine_p5": 0.9987,
  "top1_match_rate": 0.85,
  "token_agreement": 0.5,
  "prompts": [
    {
      "label": "factual",
      "passed": false,
      "max_diff": 0.0456,
      "trt_text": "Paris is the capital...",
      "hf_text": "Paris is the capital...",
      "num_steps": 10
    }
  ]
}
```

---

## Level 2: Per-Layer Hidden State Comparison (diff_layers.py)

**Purpose:** Build a debug engine with per-layer outputs and compare hidden states
against HF. Pinpoints the exact layer where divergence starts.

```bash
python tools/diff_layers.py \
  --model <model> \
  --prompt "Hello" \
  --atol 0.05 \
  --verbose
```

This builds an engine with `debug_layer_outputs=True`, which adds output tensors for:
- `debug_embed`: Embedding layer output
- `debug_hidden_0` through `debug_hidden_{N-1}`: Per-layer outputs
- `logits`: Final logits

### Reading the output:

```
Layer                  Shape         MaxDiff    MeanDiff    Status
embed                  (1024,)      0.0001     0.00001     OK
layer.0.hidden         (1024,)      0.0123     0.0012      OK
layer.1.hidden         (1024,)      0.0456     0.0045      OK
layer.2.hidden         (1024,)      0.1234     0.0123      FAIL  ← divergence starts here
layer.3.hidden         (1024,)      0.5678     0.0567      FAIL
...
logits                 (151936,)    1.2345     0.1234      FAIL
```

### Interpreting the layer diff:

| Pattern | Likely Cause | Fix |
|---------|-------------|-----|
| Embed diverges | Wrong embedding weight key mapping | Check `checkpoint_mapper.py` embed key |
| Layer 0 diverges, rest cascades | First attention or MLP weights wrong | Check family plugin `load_weights()` |
| Error grows linearly per layer | Precision accumulation | Add FP32 precision boundary in `graph_ops.add_rms_norm` |
| Specific layer jumps | Fused weight splitting issue | Check QKV or gate_up weight decomposition in plugin |
| Only logits diverge | LM head weight wrong or tied embedding issue | Check `tie_word_embeddings` handling |
| All layers OK but logits wrong | Post-norm or LM head scaling | Check final RMSNorm + LM head pipeline |

### Threshold guidance:

| `--atol` | When to use |
|----------|-------------|
| 0.01 | Strict — expect near-exact match (FP32 engine) |
| 0.05 | Default — reasonable for FP16/mixed precision |
| 0.1 | Loose — use when you expect some divergence (e.g., different RoPE impl) |

---

## Level 3: Vision-Language Analysis (diff_vl.py)

**Purpose:** For VL models, isolate whether the mismatch is in the vision encoder
or the text decoder.

### Vision encoder only (no HF model needed):

```bash
python tools/diff_vl.py \
  --bundle /tmp/model.trtfb \
  --image /path/to/test.jpg \
  --vision-only
```

This runs sanity checks: non-zero output, no NaN, no Inf, reasonable value range.

### Vision encoder vs HF reference:

```bash
python tools/diff_vl.py \
  --bundle /tmp/model.trtfb \
  --image /path/to/test.jpg \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --atol 0.1
```

**Key metric:** Cosine similarity between TRT and HF vision features.
- `cosine_sim > 0.9`: Vision encoder is fine; issue is in text decoder
- `cosine_sim < 0.5`: Hard fail — vision encoder has a bug
- `cosine_sim 0.5 - 0.9`: Partial match — check image preprocessing

### Full VL generation + C++ parity:

```bash
python tools/diff_vl.py \
  --bundle /tmp/model.trtfb \
  --image /path/to/test.jpg \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --binary ./build/trtf \
  --hf-python /opt/venv/bin/python \
  --max-new-tokens 30
```

### Preprocessor debugging:

If vision features don't match, the image preprocessing might be wrong:

```bash
# Try different preprocessor strategies
python tools/diff_vl.py \
  --bundle /tmp/model.trtfb \
  --image /path/to/test.jpg \
  --vision-only \
  --preprocessor-type simple_chw
```

Available preprocessor types: `qwen_merge_group`, `simple_chw`, and model-specific types.

---

## Level 4: Python vs C++ Binary Parity (test_runner_parity.py)

**Purpose:** Check if the Python `TrtRunner` and C++ `trtf` binary produce identical
output from the same bundle. If they diverge, the bug is in the C++ runtime, not the
engine.

```bash
python tools/test_runner_parity.py \
  --bundle /tmp/model.trtfb \
  --binary ./build/trtf \
  --hf-python /opt/venv/bin/python \
  --prompt "The capital of France is" \
  --max-new-tokens 20
```

### Reading the output:

```
C++:    'The capital of France is Paris, the largest city in France.'
Python: 'The capital of France is Paris, the largest city in France.'

Exact match: True
PASS
```

### If parity fails:

| Divergence | Likely Cause |
|-----------|-------------|
| First token differs | Tokenizer mismatch — check `tokenizer_add_special_tokens` in bundle config |
| Mid-sequence divergence | KV cache or mask implementation difference — check `device_kv_cache.cpp` vs `debug_runner.py` |
| Whitespace-only difference | Tokenizer decode normalization — usually benign |
| Complete gibberish from C++ | Bundle loading bug, wrong engine section, or TRT version mismatch |

**Critical rule from CLAUDE.md:** If you change the C++ mask/cache/position logic
(`trt_decode_runtime.cpp`, `device_kv_cache.cpp`), you MUST also update
`debug_runner.py` and verify with `test_runner_parity.py`.

---

## Level 5: Graph Op Isolation (Manual)

When Levels 1-4 identify the diverging layer but not the specific operation, isolate
individual graph ops using the TRT graph testing framework.

### Using the graph op test infrastructure:

```bash
# Run all graph op tests
pytest tests/builder/test_graph_ops.py -v -m trt

# Run a specific op test (e.g., RoPE)
pytest tests/builder/test_graph_ops.py::test_rope -v

# Run extended ops (YaRN, T5 bias, conv, etc.)
pytest tests/builder/test_graph_ops_extended.py -v -m trt

# Run composable blocks (attention, SwiGLU MLP)
pytest tests/builder/test_graph_blocks.py -v -m trt
```

### Building a minimal reproducer:

If the existing graph op tests pass but the model still diverges, build a single-op
TRT graph and compare against PyTorch:

```python
# In a Python script or notebook:
import numpy as np
import torch
from trtf_build.graph_ops import add_rms_norm  # or any op
# ... build minimal TRT network with just that op
# ... compare output against torch reference
```

The `trt_runner` fixture in `tests/builder/conftest.py` provides a `build_fn(network, inputs)`
closure that builds a small TRT graph, runs inference, and returns NumPy outputs for
comparison via `np.testing.assert_allclose`.

---

## Diff Framework (Unified Test Runner)

For running multiple diff tests at once:

```python
# List available tests for a strategy
from diff_framework import list_tests, run_tests, TestContext

# See what tests exist
tests = list_tests(runtime_strategy="decoder_kv_cache")

# Run all applicable tests
ctx = TestContext(
    model="Qwen/Qwen3-0.6B",
    runtime_strategy="decoder_kv_cache",
    bundle_path="/tmp/qwen3.trtfb",
    binary_path="./build/trtf",
    hf_python="/opt/venv/bin/python",
    max_cache_length=256,
    max_new_tokens=20,
    atol=1e-3,
    layer_atol=0.05,
)
results = run_tests(ctx)
for r in results:
    print(f"{r.test_name}: {r.status} — {r.message}")
```

Registered checks: `logit_diff`, `layer_diff`, `runner_parity`, `vl_pipeline`,
`diffusion_components`, `layer_profile`, `perf_benchmark`.

---

## Full Validation (validate_family.sh)

For validating a new or modified family plugin end-to-end:

```bash
./scripts/validate_family.sh <hf-repo-or-path>
```

This runs: build bundle → diff_logits (battery) → diff_layers → runner parity.

---

## Common Scenarios

### Scenario: New family plugin output is wrong

```bash
# Level 1: Which steps diverge?
python tools/diff_logits.py --model <model> --battery --atol 1e-3 --verbose

# Level 2: Which layer diverges?
python tools/diff_layers.py --model <model> --atol 0.05 --verbose

# Then: Read the family plugin in trtf_build/trtf_build/families/<family>.py
# Compare load_weights() key mapping against HF model's state_dict keys
# Check weight transforms (e.g., Gemma +1.0 gamma, Phi fused QKV split)
```

### Scenario: E2E test regression after code change

```bash
# Level 1: Quick check
python tools/diff_logits.py --model <model> --atol 1e-3 --battery

# Level 4: Is it Python or C++?
python tools/test_runner_parity.py --bundle <bundle> --binary ./build/trtf

# If Python matches HF but C++ doesn't → C++ runtime bug
# If Python doesn't match HF → engine builder or weight mapping bug
```

### Scenario: VL model produces wrong image description

```bash
# Level 3: Is vision encoder OK?
python tools/diff_vl.py --bundle <bundle> --image test.jpg --vision-only

# If vision OK → Level 1 on text decoder
python tools/diff_logits.py --model <model> --atol 1e-3

# If vision bad → check image preprocessor
python tools/diff_vl.py --bundle <bundle> --image test.jpg \
  --model <hf-model> --atol 0.1
```

### Scenario: Numerical drift across long sequences

```bash
# Level 1 with longer generation
python tools/diff_logits.py --model <model> \
  --prompt "Write a detailed essay about" \
  --max-new-tokens 50 --max-cache-length 128 \
  --atol 1e-2 --verbose

# Check if error grows linearly with step count
# If yes → precision accumulation in norms, check FP32 boundaries in graph_ops.py
```

## Real Output Examples

### diff_logits.py --battery output (passing model):

```
[diff_logits] Building TRT engine for Qwen/Qwen3-0.6B ...
[diff_logits] Engine built: 28 layers, cache=64

=== Prompt 1/4: factual ===
  Prompt: "The capital of France is"
  TRT:  ' Paris, which is also the largest city in France.'
  HF:   ' Paris, which is also the largest city in France.'
  Steps: 10 | Max diff: 0.000234 | Cosine p5: 0.99998 | Top-1 match: 10/10
  ✓ PASS (max_diff 0.000234 <= atol 0.001)

=== Prompt 2/4: reasoning ===
  ...
  ✓ PASS

=== Battery Summary ===
  Prompts: 4/4 passed
  cosine_p5:       0.99997
  top1_match_rate: 1.000
  token_agreement: 1.000
  mean_abs_diff:   0.000089
  ✓ ALL PASSED
```

### diff_logits.py output (failing model — weight mapping bug):

```
=== Prompt 1/4: factual ===
  Prompt: "The capital of France is"
  TRT:  ' the the the the the the the the the the'
  HF:   ' Paris, which is also the largest city in France.'
  Steps: 10 | Max diff: 45.234 | Cosine p5: 0.12345 | Top-1 match: 1/10
  ✗ FAIL (max_diff 45.234 > atol 0.001)
```

This pattern (repeating tokens + huge max_diff + low cosine) = **weight mapping is fundamentally broken**.

### diff_layers.py output (divergence at layer 5):

```
  Layer                  Shape         MaxDiff    MeanDiff    Status
  ─────────────────────────────────────────────────────────────────
  embed                  (1024,)      0.0000     0.0000      OK
  layer.0.hidden         (1024,)      0.0001     0.0000      OK
  layer.1.hidden         (1024,)      0.0003     0.0000      OK
  layer.2.hidden         (1024,)      0.0008     0.0001      OK
  layer.3.hidden         (1024,)      0.0021     0.0002      OK
  layer.4.hidden         (1024,)      0.0054     0.0005      OK
  layer.5.hidden         (1024,)      0.0823     0.0078      FAIL  ← jump here
  layer.6.hidden         (1024,)      0.2345     0.0234      FAIL
  ...
```

This pattern (gradual growth then sudden jump at layer 5) = **layer 5 has a weight mapping
or op mismatch**. Check the family plugin's weight keys for layer 5 attention/MLP.

## Composability with Other Skills

- After fixing a mismatch, use `/profile-model` to verify no performance regression.
- For new family plugins: run `/debug-trt-mismatch` Level 1 (battery) as the first
  correctness gate, before running the full E2E suite.
- If Level 4 (runner parity) fails but Python TrtRunner matches HF, the issue is in
  the **C++ runtime** — check `src/runtime/` code, not `trtf_build/`.

## Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| OOM during debug engine build | `debug_layer_outputs=True` doubles memory | Use smaller `--max-cache-length` (32 or 16) |
| TrtRunner import fails | trtf_build not installed | `pip install --no-deps -e trtf_build/` |
| HF model not found | Typo or gated model | Check spelling; use `--trust-remote-code` if needed |
| diff_layers all OK but diff_logits fails | Single-step clean, error accumulates | Precision issue — add FP32 boundary in norm ops |
| diff_vl cosine_sim = 0 | Vision encoder returns zeros | Bundle missing `vision_engine_plan` section; rebuild with VL plugin |
| runner_parity complete gibberish from C++ | Wrong TRT version or bundle corrupt | Verify `trtf-build inspect <bundle>` shows valid metadata |
| "No module named trtf_build" | Python path wrong | Run from repo root, ensure `PYTHONPATH` includes trtf_build/ |
