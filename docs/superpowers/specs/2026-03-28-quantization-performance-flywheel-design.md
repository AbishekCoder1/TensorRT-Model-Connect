# Quantization Performance Flywheel — Design Spec

## Problem

We need an autonomous system that takes any HuggingFace model and finds
the best low-precision/quantized configuration — maximizing throughput
while maintaining accuracy. The system must:

- Be fully agentic (no hardcoded workflow)
- Try as many variants in parallel as compute allows
- Never give up until a valid solution is found
- Produce deterministic, machine-readable validation
- Integrate with the existing autopilot model onboarding system

## Architecture: Supervisor → Agent → Tools

Three layers with strict separation of concerns:

### Layer 1: Supervisor (enforces persistence)

A stateless loop that launches agents and checks results. No intelligence
— just persistence enforcement.

**Location:** `scripts/autopilot/optimize_supervisor.sh`

```bash
#!/usr/bin/env bash
# Usage: optimize_supervisor.sh MODEL_ID [--max-attempts 5] [--accuracy 0.95]
MODEL_ID="$1"
MAX_ATTEMPTS="${2:-5}"
ACCURACY="${3:-0.95}"
PROGRESS_FILE="/tmp/optimize_progress_$(echo $MODEL_ID | tr '/' '_').json"

attempt=0
while [ $attempt -lt $MAX_ATTEMPTS ]; do
    attempt=$((attempt + 1))
    echo "[supervisor] Attempt $attempt/$MAX_ATTEMPTS for $MODEL_ID"

    # Launch agent with goal + progress file
    claude --print -p "$(cat <<EOF
You are optimizing ${MODEL_ID} for low-precision inference.
Read the skill at .claude/skills/optimize-model-precision.md
Read progress from: ${PROGRESS_FILE}
Accuracy threshold: ${ACCURACY}
EOF
    )"

    # Check progress file (written by deterministic tools, not agent)
    if python3 -c "
import json, sys
p = json.load(open('${PROGRESS_FILE}'))
best = p.get('best_passing')
if best and best.get('verified') and best['accuracy'] >= ${ACCURACY}:
    print(f'SUCCESS: {best[\"precision\"]}/{best.get(\"quantize\",\"none\")} '
          f'accuracy={best[\"accuracy\"]:.3f} latency={best[\"latency_ms\"]:.1f}ms')
    sys.exit(0)
sys.exit(1)
" 2>/dev/null; then
        echo "[supervisor] Found valid solution after $attempt attempt(s)"
        exit 0
    fi

    echo "[supervisor] No valid solution yet, restarting agent..."
done

echo "[supervisor] FAILED after $MAX_ATTEMPTS attempts"
exit 1
```

**Key property:** The supervisor reads the progress file, not the agent's
output. The progress file is written by deterministic tools. The agent
cannot fake a pass.

### Layer 2: Agent (full autonomy)

The agent receives a goal and tools. It decides strategy, parallelism,
and iteration. It writes progress after every attempt.

**Location:** `.claude/skills/optimize-model-precision.md`

The skill tells the agent:
1. What tools are available
2. What the goal is
3. What invariants to maintain
4. How to write progress
5. What "done" means

The agent has **full autonomy** in:
- Which precision/quantization combos to try
- What order to try them
- How many to try in parallel
- Whether to use ModelOpt, pre-quantized checkpoints, or manual scales
- Whether to modify code (e.g., thread FP16 through a custom builder)
- When to stop searching (after finding a good-enough solution)

The agent has **zero autonomy** in:
- Success criteria (tool decides pass/fail, not the agent)
- Progress reporting (must write progress file after every attempt)
- The minimum bar (must deliver at least one non-FP32 passing variant)

### Layer 3: Tools (deterministic, machine-readable)

Each tool has a single responsibility and produces machine-readable output.

#### Tool 1: Build

```bash
trtf-build build HF_ID -o BUNDLE.trtfb \
  --precision fp16 \
  --quantize fp8 \
  --quant-scales scales.json  # optional
```

Output: bundle file (or build error on stderr)

#### Tool 2: Validate Accuracy (DETERMINISTIC)

```bash
python3 tools/diff_logits.py \
  --model HF_ID \
  --bundle BUNDLE.trtfb \
  --atol 0.01 \
  --json /tmp/eval_result.json
```

Output:
```json
{
  "pass": true,
  "cosine_p5": 0.97,
  "top1_match_rate": 0.92,
  "token_agreement": 0.88,
  "mean_abs_diff": 0.0032
}
```

The `pass` field is the **single source of truth**. Computed by the tool
using deterministic thresholds. The agent reads this value — it does not
interpret generated text quality.

#### Tool 3: Measure Performance

```bash
python3 tools/perf_compare.py \
  --bundle BUNDLE.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --json /tmp/perf_result.json
```

Output:
```json
{
  "prefill_ms": 12.3,
  "decode_ms_per_token": 8.7,
  "total_latency_ms": 186.3,
  "tokens_per_second": 114.9,
  "peak_memory_mb": 1557
}
```

#### Tool 4: Quick Sanity Check

```bash
./build/trtf run BUNDLE.trtfb \
  --prompt "Hello" --max-new-tokens 10 --hf-python /opt/venv/bin/python
```

Output: generated text (for agent to sanity-check, not for pass/fail)

#### Tool 5: Inspect Bundle

```bash
trtf-build inspect BUNDLE.trtfb
```

Output: precision, quantization, engine size, layer count, etc.

### Progress File Contract

After every attempt, the agent writes/updates:

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "goal": "best precision, diff_logits pass=true (cosine_p5 >= 0.95)",
  "started": "2026-03-28T10:00:00Z",
  "attempts": [
    {
      "id": 1,
      "precision": "fp16",
      "quantize": null,
      "status": "pass",
      "accuracy": 0.99,
      "latency_ms": 48.2,
      "memory_mb": 1557,
      "bundle": "/path/to/fp16.trtfb",
      "verified": true,
      "eval_json": "/tmp/eval_fp16.json"
    },
    {
      "id": 2,
      "precision": "fp16",
      "quantize": "fp8",
      "status": "build_failed",
      "error": "ModelOpt not installed",
      "verified": false
    },
    {
      "id": 3,
      "precision": "fp16",
      "quantize": "int8",
      "status": "accuracy_failed",
      "accuracy": 0.82,
      "verified": true,
      "eval_json": "/tmp/eval_int8.json"
    }
  ],
  "best_passing": {
    "precision": "fp16",
    "quantize": null,
    "accuracy": 0.99,
    "latency_ms": 48.2,
    "memory_mb": 1557,
    "bundle": "/path/to/fp16.trtfb",
    "verified": true
  }
}
```

**`verified: true`** means the `diff_logits.py --json` tool confirmed
the result. The supervisor only trusts entries with `verified: true`.

### Invariants

1. **FP16 is the guaranteed floor.** For standard decoder models, FP16
   always works (proven experimentally). The agent must try FP16 first
   if no other options succeed.

2. **Success is binary.** The `diff_logits.py --json` tool returns
   `pass: true/false`. The agent cannot override this.

3. **Progress survives restarts.** The agent reads the progress file at
   startup and skips already-completed attempts.

4. **The supervisor checks the file, not the agent.** Even if the agent
   says "I'm done" but the progress file shows no passing candidate,
   the supervisor restarts the agent.

5. **Parallel attempts are independent.** The agent can dispatch N builds
   simultaneously. A failure in one does not affect others.

## Fallback Chain (Agent's Toolbox, Not a Script)

The agent has access to multiple paths. It chooses which to try:

| Path | Requires | Speed | When to Use |
|------|----------|-------|-------------|
| FP16 (precision only) | Nothing | Fast (2-5 min build) | Always — guaranteed floor |
| FP8 via ModelOpt | `nvidia-modelopt` + GPU | Slow (calibration ~30min) | When ModelOpt is installed |
| FP8 via pre-computed scales | JSON file | Fast (no calibration) | When scales exist from prior run |
| INT8 SmoothQuant via ModelOpt | `nvidia-modelopt` + GPU | Slow (~30min) | Alternative to FP8 |
| INT4 AWQ via ModelOpt | `nvidia-modelopt` + GPU | Slow (~30min) | Large models, memory-bound |
| INT4 via pre-quantized HF checkpoint | GPTQ/AWQ checkpoint on HF | Medium | When model has GPTQ/AWQ variant |
| NVFP4 dynamic | Blackwell GPU | Fast (no calibration) | On GB300/Blackwell |
| Mixed precision (per-layer) | Prior failed attempts | Medium | When uniform quantization fails accuracy |

The agent decides which paths to explore based on:
- What's installed (ModelOpt? Blackwell GPU?)
- What's available (pre-quantized checkpoint on HF?)
- What already failed (read progress file)
- How much compute is available (parallel vs sequential)

## Integration with Autopilot

The existing autopilot onboarding flow:

```
discover → scaffold → build FP32 → validate (5 gates) → commit
```

The optimization plugs in as an optional post-validation step:

```
discover → scaffold → build FP32 → validate → [optimize] → commit
```

The agent prompt in `autorun.py` gains a clause:

```
After all validation gates pass:
  If --optimize flag is set:
    Run the optimization skill (read .claude/skills/optimize-model-precision.md)
    Update the E2E manifest with the best precision config
    Create a second manifest for the optimized variant if different from FP32
```

The supervisor (`optimize_supervisor.sh`) can also be invoked standalone
for existing models that already have FP32 support.

## Framework Gaps to Fix

For the flywheel to work across all modalities:

| Gap | Impact | Fix |
|-----|--------|-----|
| `diff_logits.py` lacks `--json` output | Agent can't parse accuracy | Add `--json` flag that writes metrics to JSON file |
| `perf_compare.py` JSON output incomplete | Agent can't parse latency reliably | Ensure `--json` includes all fields |
| Conv Q/DQ missing from `QuantFormat` | Vision/audio models can't be quantized | Add `wrap_conv2d()`/`wrap_conv1d()` to protocol |
| AWQ checkpoint provider is a stub | Can't load pre-quantized HF models | Implement `_extract_awq()` |
| Non-decoder builders lack FP16 threading | FP16 floor doesn't work for all modalities | Thread precision through custom builders (agent can do this) |
| E2E harness doesn't output per-metric JSON | Supervisor can't check accuracy deterministically | Add metrics JSON output to E2E harness |

## Success Criteria for This Spec

The flywheel is "working" when:

1. `optimize_supervisor.sh Qwen/Qwen3-0.6B` produces a bundle that is
   NOT FP32 and passes accuracy validation — without human intervention
2. The supervisor restarts the agent at least once (proving restart works)
3. The progress file shows multiple attempts with `verified: true`
4. The final bundle is FP16 or better (FP8/INT8/INT4)

## Files to Create

| File | Purpose |
|------|---------|
| `scripts/autopilot/optimize_supervisor.sh` | Supervisor loop (restarts agent until done) |
| `.claude/skills/optimize-model-precision.md` | Agent skill (tools + goal + invariants) |
| Modify `tools/diff_logits.py` | Add `--json` output flag |
| Modify `tools/perf_compare.py` | Ensure `--json` output is complete |
| Modify `quantization/formats.py` | Add `wrap_conv2d()`/`wrap_conv1d()` |
| Modify `quantization/scale_providers.py` | Implement AWQ extraction |

## Files NOT Created

No `optimize.py` orchestration script. No hardcoded search loop. The
agent IS the orchestrator. The supervisor just enforces persistence.
