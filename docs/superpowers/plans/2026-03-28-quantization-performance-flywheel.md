# Quantization Performance Flywheel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a supervisor + skill + tool system where an agent autonomously finds the best low-precision config for any model, with deterministic validation and persistent progress that survives agent restarts.

**Architecture:** Three layers — a bash supervisor loop that restarts agents until the progress file shows success, a Claude Code skill that gives the agent full autonomy over search strategy, and deterministic tools (`diff_logits --json`, `perf_compare --json`) whose pass/fail verdict the agent cannot override.

**Tech Stack:** Bash (supervisor), Markdown (skill), Python (tool modifications), TensorRT Python API (quantization formats)

---

### Task 1: Add `--json` output to `diff_logits.py`

This is the **critical deterministic gate** — the tool that decides pass/fail for every candidate.

**Files:**
- Modify: `tools/diff_logits.py`
- Test: `tests/tools/test_diff_logits.py`

- [ ] **Step 1: Write the failing test**

Add a test that verifies JSON output mode produces a parseable file with the required fields.

```python
# In tests/tools/test_diff_logits.py — add new test class at end of file

class TestJsonOutput:
    """Verify --json flag produces machine-readable evaluation results."""

    def test_json_output_structure(self, tmp_path):
        """JSON output must contain pass, cosine_p5, top1_match_rate, token_agreement."""
        json_path = tmp_path / "eval.json"
        # Simulate what _write_json_report would produce
        mod = importlib.import_module("diff_logits")
        report = mod._build_json_report(
            prompts=["Hello world"],
            per_step_results=[
                {"label": "step_0", "max_diff": 0.001, "cosine_sim": 0.99,
                 "top1_match": True, "text_match": True, "passed": True}
            ],
            overall_pass=True,
            atol=0.001,
        )
        assert "pass" in report
        assert "cosine_p5" in report
        assert "top1_match_rate" in report
        assert "token_agreement" in report
        assert isinstance(report["pass"], bool)

    def test_json_pass_true_when_all_steps_pass(self, tmp_path):
        mod = importlib.import_module("diff_logits")
        report = mod._build_json_report(
            prompts=["test"],
            per_step_results=[
                {"label": "s0", "max_diff": 0.0005, "cosine_sim": 0.99,
                 "top1_match": True, "text_match": True, "passed": True},
                {"label": "s1", "max_diff": 0.0008, "cosine_sim": 0.98,
                 "top1_match": True, "text_match": True, "passed": True},
            ],
            overall_pass=True,
            atol=0.001,
        )
        assert report["pass"] is True
        assert report["cosine_p5"] >= 0.98

    def test_json_pass_false_when_step_fails(self, tmp_path):
        mod = importlib.import_module("diff_logits")
        report = mod._build_json_report(
            prompts=["test"],
            per_step_results=[
                {"label": "s0", "max_diff": 0.5, "cosine_sim": 0.3,
                 "top1_match": False, "text_match": False, "passed": False},
            ],
            overall_pass=False,
            atol=0.001,
        )
        assert report["pass"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec trtmc-dev-gb300-agent-3 bash -c "cd /workspace/tensorrt-model-connect && /opt/venv/bin/python -m pytest tests/tools/test_diff_logits.py::TestJsonOutput -v"`

Expected: FAIL — `_build_json_report` does not exist yet

- [ ] **Step 3: Implement `_build_json_report` and `--json` flag in diff_logits.py**

Read `tools/diff_logits.py` in full. Then add:

1. Import `json` at the top.
2. Add `--json` argument to the argparse section (around line 314):
   ```python
   parser.add_argument("--json", type=str, default=None,
                        help="Write machine-readable results to JSON file")
   ```
3. Add the `_build_json_report` function (before `main()`):
   ```python
   def _build_json_report(prompts, per_step_results, overall_pass, atol):
       cosine_sims = [r["cosine_sim"] for r in per_step_results if "cosine_sim" in r]
       top1_matches = [r["top1_match"] for r in per_step_results if "top1_match" in r]
       text_matches = [r["text_match"] for r in per_step_results if "text_match" in r]
       return {
           "pass": bool(overall_pass),
           "cosine_p5": float(np.percentile(cosine_sims, 5)) if cosine_sims else 0.0,
           "cosine_mean": float(np.mean(cosine_sims)) if cosine_sims else 0.0,
           "top1_match_rate": float(np.mean(top1_matches)) if top1_matches else 0.0,
           "token_agreement": float(np.mean(text_matches)) if text_matches else 0.0,
           "mean_abs_diff": float(np.mean([r["max_diff"] for r in per_step_results])),
           "num_steps": len(per_step_results),
           "atol": atol,
           "prompts": prompts,
       }
   ```
4. At the end of `main()`, after computing results, add JSON output:
   ```python
   if args.json:
       report = _build_json_report(prompts, per_step_results, overall_pass, args.atol)
       Path(args.json).write_text(json.dumps(report, indent=2))
   ```

Read the actual file to find the exact variable names for `per_step_results`, `overall_pass`, etc. and adapt. The function names may differ — find where the per-step comparison loop accumulates results and where the final pass/fail is determined.

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker exec trtmc-dev-gb300-agent-3 bash -c "cd /workspace/tensorrt-model-connect && /opt/venv/bin/python -m pytest tests/tools/test_diff_logits.py::TestJsonOutput -v"`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/diff_logits.py tests/tools/test_diff_logits.py
git commit -m "feat: add --json output to diff_logits for machine-readable accuracy validation"
```

---

### Task 2: Verify `perf_compare.py` JSON output

`perf_compare.py` already has `--json` in its docstring. Verify it works and produces the required fields.

**Files:**
- Modify: `tools/perf_compare.py` (if `--json` is incomplete)
- Test: `tests/tools/test_perf_compare.py`

- [ ] **Step 1: Check existing `--json` implementation**

Read `tools/perf_compare.py` in full. Search for `--json` in argparse and where JSON is written. If it already produces `{"prefill_ms", "decode_ms_per_token", "total_latency_ms", "tokens_per_second", "peak_memory_mb"}`, this task is a verification step only. If fields are missing, add them.

- [ ] **Step 2: Write a test for JSON output structure**

```python
# In tests/tools/test_perf_compare.py — add at end

class TestJsonOutputFields:
    def test_json_has_required_fields(self, tmp_path):
        mod = importlib.import_module("perf_compare")
        # Create a minimal PerfResult or equivalent and serialize
        # Verify all required fields exist in output
        required = {"prefill_ms", "decode_ms_per_token", "total_latency_ms",
                     "tokens_per_second", "peak_memory_mb"}
        # ... test that serialization produces these fields
```

Read the actual file to understand the data structures before writing the exact test.

- [ ] **Step 3: Fix any missing fields**

If the JSON output is missing fields, add them. If it works, this step is a no-op.

- [ ] **Step 4: Run tests**

Run: `docker exec trtmc-dev-gb300-agent-3 bash -c "cd /workspace/tensorrt-model-connect && /opt/venv/bin/python -m pytest tests/tools/test_perf_compare.py -v"`

Expected: PASS

- [ ] **Step 5: Commit** (only if changes were made)

```bash
git add tools/perf_compare.py tests/tools/test_perf_compare.py
git commit -m "fix: ensure perf_compare --json output includes all required fields"
```

---

### Task 3: Write the Agent Skill

The skill is the agent's instruction manual. It describes available tools, the goal, invariants, and how to write the progress file. No code — just markdown.

**Files:**
- Create: `.claude/skills/optimize-model-precision.md`

- [ ] **Step 1: Write the skill file**

```markdown
---
description: Autonomously optimize a model's inference precision and quantization. Tries parallel precision variants, validates each with deterministic tools, tracks progress in a persistent file. The agent has full strategy autonomy but cannot override tool pass/fail verdicts.
---

# Optimize Model Precision

## Goal

Given a HuggingFace model ID and an accuracy threshold, find the best
low-precision configuration that maximizes throughput while maintaining
accuracy. Write progress to a persistent file after every attempt.

## Reading This Skill

You are invoked by the optimize supervisor. You receive:
- `MODEL_ID`: HuggingFace model identifier
- `PROGRESS_FILE`: path to read/write persistent progress
- `ACCURACY_THRESHOLD`: minimum accuracy (default 0.95)

At startup, READ the progress file if it exists. Resume from where the
previous agent left off. Do not repeat completed attempts.

## Available Tools

All commands run inside the container:
`docker exec trtmc-dev-gb300-{agent-id} bash -c "cd /workspace/tensorrt-model-connect && COMMAND"`

### Build a variant
```
trtmc-build build MODEL_ID -o OUTPUT.trtfb \
  --precision fp16 \
  [--quantize fp8|int8|int4|nvfp4|w4a8] \
  [--quant-scales SCALES.json] \
  [--max-cache-length 256]
```

### Validate accuracy (DETERMINISTIC — this decides pass/fail)
```
python3 tools/diff_logits.py \
  --model MODEL_ID \
  --bundle BUNDLE.trtfb \
  --atol 0.01 --battery \
  --json /tmp/eval_VARIANT.json
```
Read the JSON output. The `pass` field is the single source of truth.
You CANNOT override this verdict.

### Measure performance
```
python3 tools/perf_compare.py \
  --bundle BUNDLE.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --json /tmp/perf_VARIANT.json
```

### Quick sanity check (not a pass/fail gate)
```
./build/trtmc run BUNDLE.trtfb \
  --prompt "Hello, how are you?" --max-new-tokens 10 \
  --hf-python /opt/venv/bin/python
```

### Inspect a bundle
```
trtmc-build inspect BUNDLE.trtfb
```

## Invariants (NEVER violate)

1. You are NOT DONE until the progress file contains a `best_passing`
   entry with `verified: true` that is NOT FP32.

2. After EVERY attempt (pass or fail), update the progress file.
   The supervisor reads this file to decide whether to restart you.

3. FP16 always works for standard decoder models. If nothing else
   passes, build FP16. It is your guaranteed fallback.

4. Accuracy is decided by `diff_logits.py --json`. The `pass` field
   is authoritative. Do not judge quality by reading generated text.

5. You may try as many variants in parallel as you have compute.
   Use the Agent tool to dispatch parallel builds.

## Progress File Format

Write to the path given as PROGRESS_FILE. Use this exact schema:

```json
{
  "model": "MODEL_ID",
  "goal": "best precision, diff_logits pass=true",
  "started": "ISO-8601 timestamp",
  "attempts": [
    {
      "id": 1,
      "precision": "fp16",
      "quantize": null,
      "status": "pass|build_failed|accuracy_failed|error",
      "accuracy": 0.99,
      "latency_ms": 48.2,
      "memory_mb": 1557,
      "bundle": "/absolute/path/to/bundle.trtfb",
      "verified": true,
      "eval_json": "/tmp/eval_fp16.json",
      "perf_json": "/tmp/perf_fp16.json",
      "error": null
    }
  ],
  "best_passing": {
    "precision": "fp16",
    "quantize": null,
    "accuracy": 0.99,
    "latency_ms": 48.2,
    "memory_mb": 1557,
    "bundle": "/absolute/path/to/bundle.trtfb",
    "verified": true
  }
}
```

`verified: true` means diff_logits.py confirmed the result.
`best_passing` is the candidate with the lowest latency among all
passing attempts. Update it after each successful attempt.

## Strategy Suggestions (NOT mandatory — you decide)

- Start with FP16 (fast, always works) to establish a floor
- Then try more aggressive formats in parallel
- If a format fails to build (ModelOpt missing, GPU incompatible),
  record the error and move on — do not retry the same failure
- If accuracy fails, try mixed precision or a less aggressive format
- Check if HuggingFace has pre-quantized variants (GPTQ/AWQ)
  before running expensive calibration
```

- [ ] **Step 2: Verify the skill file is readable**

```bash
cat .claude/skills/optimize-model-precision.md | head -5
```

Expected: frontmatter with `description:` field

- [ ] **Step 3: Force-add to git (past .gitignore) and commit**

```bash
git add -f .claude/skills/optimize-model-precision.md
git commit -m "feat: add optimize-model-precision agent skill"
```

---

### Task 4: Write the Supervisor Script

The supervisor is a bash loop that launches the agent and checks the progress file. No intelligence — just persistence.

**Files:**
- Create: `scripts/autopilot/optimize_supervisor.sh`

- [ ] **Step 1: Write the supervisor script**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Usage: optimize_supervisor.sh MODEL_ID [MAX_ATTEMPTS] [ACCURACY_THRESHOLD] [AGENT_ID]
#
# Launches a Claude Code agent to optimize MODEL_ID for low-precision inference.
# Restarts the agent up to MAX_ATTEMPTS times until the progress file shows a
# verified passing candidate. The supervisor reads the progress file (written by
# deterministic tools), NOT the agent's self-report.

MODEL_ID="${1:?Usage: optimize_supervisor.sh MODEL_ID [MAX_ATTEMPTS] [ACCURACY] [AGENT_ID]}"
MAX_ATTEMPTS="${2:-5}"
ACCURACY="${3:-0.95}"
AGENT_ID="${4:-agent-3}"

SAFE_NAME="$(echo "$MODEL_ID" | tr '/' '_' | tr '.' '_')"
PROGRESS_FILE="/tmp/optimize_progress_${SAFE_NAME}.json"
CONTAINER="trtmc-dev-gb300-${AGENT_ID}"
REPO_DIR="/workspace/users/yifeif/workspaces/${AGENT_ID}/tensorrt-model-connect"

echo "[supervisor] Model:      $MODEL_ID"
echo "[supervisor] Max attempts: $MAX_ATTEMPTS"
echo "[supervisor] Accuracy:    $ACCURACY"
echo "[supervisor] Progress:    $PROGRESS_FILE"
echo "[supervisor] Container:   $CONTAINER"
echo ""

check_success() {
    python3 -c "
import json, sys
try:
    p = json.load(open('${PROGRESS_FILE}'))
except (FileNotFoundError, json.JSONDecodeError):
    sys.exit(1)
best = p.get('best_passing')
if not best:
    sys.exit(1)
if not best.get('verified', False):
    sys.exit(1)
if best.get('accuracy', 0) < ${ACCURACY}:
    sys.exit(1)
prec = best.get('precision', 'fp32')
quant = best.get('quantize', 'none') or 'none'
if prec == 'fp32' and quant == 'none':
    sys.exit(1)  # Must be better than FP32
lat = best.get('latency_ms', 0)
mem = best.get('memory_mb', 0)
acc = best.get('accuracy', 0)
print(f'precision={prec} quantize={quant} accuracy={acc:.3f} latency={lat:.1f}ms memory={mem:.0f}MB')
sys.exit(0)
" 2>/dev/null
}

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
    attempt=$((attempt + 1))
    echo "[supervisor] === Attempt $attempt/$MAX_ATTEMPTS ==="

    claude --print --dangerously-skip-permissions -p "
You are optimizing ${MODEL_ID} for low-precision inference.

Read the skill file: cat ${REPO_DIR}/.claude/skills/optimize-model-precision.md

Progress file: ${PROGRESS_FILE}
Accuracy threshold: ${ACCURACY}
Container: ${CONTAINER}
Repo: ${REPO_DIR}

Read the progress file first. If it exists, resume from where the
previous agent left off. Do not repeat completed attempts.

Your goal: find the best non-FP32 precision config that passes
diff_logits validation. Write every result to the progress file.
" || true  # Don't exit on agent failure

    if check_success; then
        echo ""
        echo "[supervisor] SUCCESS after $attempt attempt(s):"
        check_success  # Print the result line
        echo "[supervisor] Progress file: $PROGRESS_FILE"
        exit 0
    fi

    echo "[supervisor] No verified passing candidate yet."
    if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
        echo "[supervisor] Restarting agent..."
        echo ""
    fi
done

echo ""
echo "[supervisor] FAILED after $MAX_ATTEMPTS attempts."
echo "[supervisor] Progress file: $PROGRESS_FILE"
exit 1
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/autopilot/optimize_supervisor.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/autopilot/optimize_supervisor.sh
git commit -m "feat: add optimize supervisor loop for agent-driven precision search"
```

---

### Task 5: Add `wrap_conv2d` to QuantFormat protocol

Extends quantization to cover vision, audio, and diffusion models that use convolutions.

**Files:**
- Modify: `tensorrt_model_connect/tensorrt_model_connect/quantization/formats.py`
- Test: `tests/builder/test_quantization.py` (new)

- [ ] **Step 1: Write failing test**

Create `tests/builder/test_quantization.py`:

```python
"""Tests for quantization framework abstractions."""
import numpy as np
import pytest

from tensorrt_model_connect.quantization import get_format, list_formats, QuantScaleMap, LayerScales
from tensorrt_model_connect.quantization.formats import QuantFormat


class TestFormatRegistry:
    def test_all_formats_registered(self):
        names = list_formats()
        assert "fp8" in names
        assert "int8_sq" in names
        assert "int4_awq" in names
        assert "nvfp4" in names
        assert "w4a8" in names

    def test_get_format_returns_protocol(self):
        fmt = get_format("fp8")
        assert isinstance(fmt, QuantFormat)
        assert fmt.name == "fp8"

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_format("nonexistent")


class TestScaleMapJsonRoundTrip:
    def test_roundtrip(self):
        original = QuantScaleMap(scales={
            "layer.0.w_q": LayerScales(input_scale=0.042, weight_scale=0.051),
            "layer.1.w_k": LayerScales(input_scale=0.1, weight_scale=0.2, block_size=128),
        })
        restored = QuantScaleMap.from_json(original.to_json())
        assert len(restored.scales) == 2
        assert abs(restored.scales["layer.0.w_q"].input_scale - 0.042) < 1e-6
        assert restored.scales["layer.1.w_k"].block_size == 128

    def test_dynamic_flag(self):
        m = QuantScaleMap(scales={}, dynamic=True)
        restored = QuantScaleMap.from_json(m.to_json())
        assert restored.dynamic is True


class TestQuantFormatProtocol:
    def test_all_formats_have_wrap_matmul(self):
        for name in list_formats():
            fmt = get_format(name)
            assert hasattr(fmt, "wrap_matmul"), f"{name} missing wrap_matmul"

    def test_all_formats_have_wrap_conv2d(self):
        for name in list_formats():
            fmt = get_format(name)
            assert hasattr(fmt, "wrap_conv2d"), f"{name} missing wrap_conv2d"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec trtmc-dev-gb300-agent-3 bash -c "cd /workspace/tensorrt-model-connect && /opt/venv/bin/python -m pytest tests/builder/test_quantization.py -v"`

Expected: `test_all_formats_have_wrap_conv2d` FAILS (method doesn't exist yet)

- [ ] **Step 3: Add `wrap_conv2d` to protocol and all 5 format classes**

In `tensorrt_model_connect/tensorrt_model_connect/quantization/formats.py`:

1. Add to `QuantFormat` protocol (after `wrap_matmul`):
```python
    def wrap_conv2d(
        self,
        network: trt.INetworkDefinition,
        activation: trt.ITensor,
        weight_array: np.ndarray,
        bias_array: np.ndarray | None,
        scales: LayerScales,
        *,
        out_channels: int,
        kernel_shape: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        groups: int = 1,
        dtype: np.dtype,
    ) -> trt.ITensor:
        """Insert format-specific Q/DQ around a Conv2D."""
        ...
```

2. Add default implementation to each of the 5 format classes. The pattern is the same as `wrap_matmul` but uses `network.add_convolution_nd()` instead of `add_matrix_multiply()`. For weight-only formats (INT4_AWQ), quantize only the conv weight, not the activation.

Read the existing `graph_ops.add_conv2d()` function to understand the exact TRT convolution API, then mirror the Q/DQ wrapping pattern.

- [ ] **Step 4: Run tests**

Run: `docker exec trtmc-dev-gb300-agent-3 bash -c "cd /workspace/tensorrt-model-connect && /opt/venv/bin/python -m pytest tests/builder/test_quantization.py -v"`

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add tensorrt_model_connect/tensorrt_model_connect/quantization/formats.py tests/builder/test_quantization.py
git commit -m "feat: add wrap_conv2d to QuantFormat protocol for vision/audio quantization"
```

---

### Task 6: Implement AWQ checkpoint scale extraction

Completes the pre-quantized checkpoint provider so users can load AWQ models from HuggingFace without re-calibrating.

**Files:**
- Modify: `tensorrt_model_connect/tensorrt_model_connect/quantization/scale_providers.py`
- Test: `tests/builder/test_quantization.py` (extend)

- [ ] **Step 1: Write failing test**

Add to `tests/builder/test_quantization.py`:

```python
class TestPreQuantizedCheckpointProvider:
    def test_detect_awq_format(self, tmp_path):
        """AWQ checkpoint has quantization_config.quant_method == 'awq'."""
        from tensorrt_model_connect.quantization.scale_providers import PreQuantizedCheckpointProvider
        from tensorrt_model_connect.config import ModelConfig

        config = ModelConfig.from_json(json.dumps({
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
                "zero_point": True,
            }
        }))
        provider = PreQuantizedCheckpointProvider()
        # Should not raise ValueError for AWQ format detection
        # (will fail on missing safetensors, but that tests the detection path)
        with pytest.raises(Exception):  # FileNotFoundError or similar
            provider.acquire_scales(str(tmp_path), config, get_format("int4_awq"), [])
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL with `NotImplementedError("AWQ scale extraction not yet implemented")`

- [ ] **Step 3: Implement `_extract_awq`**

In `scale_providers.py`, replace the `_extract_awq` stub with:

```python
def _extract_awq(self, model_dir, config, exclude_patterns):
    """Extract scales from AWQ checkpoint."""
    import re
    from safetensors import safe_open

    exclude_re = re.compile("|".join(
        f"({p.replace('*', '.*')})" for p in exclude_patterns
    )) if exclude_patterns else None

    quant_config = config.raw.get("quantization_config", {})
    group_size = quant_config.get("group_size", 128)

    scales: dict[str, LayerScales] = {}
    model_path = Path(model_dir)
    for sf_path in model_path.glob("*.safetensors"):
        with safe_open(str(sf_path), framework="numpy") as f:
            for key in f.keys():
                if not key.endswith(".scales"):
                    continue
                layer_name = key.rsplit(".scales", 1)[0]
                if exclude_re and exclude_re.search(layer_name):
                    continue
                scale_array = f.get_tensor(key)
                scales[layer_name] = LayerScales(
                    weight_scale=scale_array,
                    input_scale=1.0,
                    block_size=group_size,
                )

    logger.info("Extracted AWQ scales for %d layers", len(scales))
    return QuantScaleMap(scales=scales)
```

- [ ] **Step 4: Run tests**

Expected: `test_detect_awq_format` now raises a file-not-found error (not NotImplementedError) — AWQ path is reached.

- [ ] **Step 5: Commit**

```bash
git add tensorrt_model_connect/tensorrt_model_connect/quantization/scale_providers.py tests/builder/test_quantization.py
git commit -m "feat: implement AWQ checkpoint scale extraction for pre-quantized HF models"
```

---

### Task 7: Add `--optimize` flag to autopilot

Integrate the optimization phase into the existing autopilot onboarding flow.

**Files:**
- Modify: `scripts/autopilot/autorun.py`
- Modify: `scripts/autopilot/dispatch.py`

- [ ] **Step 1: Read current agent prompt in autorun.py and dispatch.py**

Read the `WORKER_PROMPT` strings in both files. Understand where the validation phase ends and where the commit phase begins.

- [ ] **Step 2: Add `--optimize` CLI flag to autorun.py**

In the argparse section, add:
```python
parser.add_argument("--optimize", action="store_true",
                    help="After validation, run precision optimization")
```

- [ ] **Step 3: Add optimization phase to WORKER_PROMPT**

After the validation gates section and before the commit section, add a conditional block. Read the exact prompt structure first, then insert:

```
### Phase: Performance Optimization (if --optimize)

After all validation gates pass, optimize the model for low precision:

1. Read the skill: cat .claude/skills/optimize-model-precision.md
2. Follow the skill instructions to find the best non-FP32 precision
3. Use the progress file at /tmp/optimize_progress_{family_name}.json
4. Update the E2E manifest with the recommended precision
5. Create a second manifest for the optimized variant (e.g., {name}-fp16.json)
```

Only include this section when `--optimize` is set. Use a conditional in the prompt construction.

- [ ] **Step 4: Apply same change to dispatch.py**

Mirror the `--optimize` flag and prompt addition.

- [ ] **Step 5: Commit**

```bash
git add scripts/autopilot/autorun.py scripts/autopilot/dispatch.py
git commit -m "feat: add --optimize flag to autopilot for precision tuning after validation"
```

---

### Task 8: End-to-End Verification

Verify the full flywheel works: supervisor → agent → tools → progress file → success.

**Files:**
- No new files — this is a validation task

- [ ] **Step 1: Run the supervisor on Qwen3-0.6B**

This is the gold-standard test. The supervisor launches an agent, the agent builds FP16, validates with `diff_logits --json`, writes the progress file, and the supervisor confirms success.

```bash
# From the host (not inside container)
cd /workspace/users/yifeif/workspaces/agent-3/tensorrt-model-connect
bash scripts/autopilot/optimize_supervisor.sh Qwen/Qwen3-0.6B 3 0.95 agent-3
```

Expected:
- Supervisor prints "Attempt 1/3"
- Agent builds FP16 bundle
- Agent runs `diff_logits --json`
- Agent writes progress file
- Supervisor reads progress file, finds verified passing candidate
- Supervisor prints "SUCCESS after 1 attempt(s)"

- [ ] **Step 2: Verify progress file**

```bash
cat /tmp/optimize_progress_Qwen_Qwen3-0_6B.json | python3 -m json.tool
```

Expected: JSON with at least one attempt with `"verified": true` and `"status": "pass"`.

- [ ] **Step 3: Verify the bundle works**

```bash
docker exec trtmc-dev-gb300-agent-3 bash -c "cd /workspace/tensorrt-model-connect && \
  ./build/trtmc run BUNDLE_PATH --prompt 'The capital of France is' --max-new-tokens 10 \
  --hf-python /opt/venv/bin/python"
```

Where BUNDLE_PATH is from the progress file's `best_passing.bundle`.

Expected: Coherent text output.

- [ ] **Step 4: Final commit with all verification artifacts**

```bash
git add -A
git commit -m "feat: quantization performance flywheel — supervisor, skill, tools, verification"
```
