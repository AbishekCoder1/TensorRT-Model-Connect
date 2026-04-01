---
description: Autonomously optimize any model's inference precision and quantization. Works for ALL modalities (text, speech, vision, diffusion). Validates via E2E pytest harness. If FP16 isn't threaded through a builder, the agent modifies the builder code following the FP16 skill guide.
---

# Optimize Model Precision

## Goal

Given a HuggingFace model ID, find the best low-precision configuration
that maximizes throughput while maintaining accuracy. Works for ALL
model types — text generation, speech-to-text, vision-language, diffusion,
segmentation, encoder-only.

## Inputs

You receive:
- `MODEL_ID`: HuggingFace model identifier
- `PROGRESS_FILE`: path to read/write persistent progress
- `ACCURACY_THRESHOLD`: minimum accuracy (default 0.95)
- `CONTAINER`: docker container name
- `REPO`: path to trt-transformers-cpp repo

At startup, READ the progress file if it exists. Resume from where the
previous agent left off. Do not repeat completed attempts.

## Available Tools

All commands run inside the container:
```
docker exec CONTAINER bash -c "cd REPO && COMMAND"
```

### Build
```
trtf-build build MODEL_ID -o OUTPUT.trtfb \
  --precision fp16 \
  [--quantize fp8|int8|int4|nvfp4|w4a8] \
  [--quant-scales SCALES.json] \
  [--max-cache-length 256]
```

### Validate (DETERMINISTIC — works for ALL modalities)
```
/opt/venv/bin/python -m pytest tests/test_e2e.py::test_e2e[MANIFEST_NAME] -v \
  --engine-dir /workspace/users/yifeif/trt-transformers/engines \
  --trtf-binary ./build/trtf \
  --hf-python /opt/venv/bin/python \
  --rebuild-engines
```
This is the **single source of truth**. It builds the bundle, runs TRT
inference, compares against the HF reference, and reports PASS/FAIL.
It works for text, speech, vision, diffusion — every modality has a
comparator in the E2E harness.

If the model already has an E2E manifest in `tests/e2e/models/`, use it.
If not, create one (see "Creating Manifests" below).

### Inspect bundle (verify FP16 actually took effect)
```
trtf-build inspect BUNDLE.trtfb
```
Check the engine size. An FP16 bundle should be roughly HALF the size
of FP32. If `--precision fp16` produces the same size as FP32, the
builder is NOT threading precision — you must fix the builder code.

### Quick inference sanity check
```
./build/trtf run BUNDLE.trtfb --prompt "test" --max-new-tokens 5 \
  --hf-python /opt/venv/bin/python
```

## Critical: Detecting Fake FP16

`--precision fp16` is accepted by ALL builders but only ACTUALLY
IMPLEMENTED in `standard_decoder_builder.py`. Custom builders (whisper,
mamba, mixtral, flux, sam, segformer, etc.) accept the parameter but
produce FP32 engines.

**How to detect:** After building with `--precision fp16`, compare bundle
size to an FP32 build. If sizes are similar (within 10%), FP16 is NOT
taking effect.

**How to fix:** Read `.claude/skills/fp16-trt-network.md` for the
complete guide. The pattern is:

1. In the builder function, compute work dtypes:
   ```python
   work_np_dtype = np.float16 if precision == "fp16" else np.float32
   work_trt_dtype = trt.float16 if precision == "fp16" else trt.float32
   ```

2. Set cache/state input dtypes to `work_trt_dtype` (not hardcoded `trt.float32`)

3. Pass `dtype=work_np_dtype` to ALL `graph_ops.add_constant()` and
   `graph_ops.add_matmul_rhs_constant()` calls

4. Keep attention mask as `trt.float32` at input, cast to work dtype

5. Cast logits/output to FP32 before `network.mark_output()`

6. Norm weights (gamma, beta, eps) stay FP32 inside precision boundaries
   (the norm functions in graph_ops.py handle this automatically when
   you pass `dtype=work_np_dtype`)

Look at `standard_decoder_builder.py` as the reference implementation.

## Creating E2E Manifests for Low-Precision Variants

For FP16 variants, create a new manifest alongside the FP32 one:

```json
{
  "name": "MODEL-fp16",
  "hf_id": "ORG/MODEL",
  "bundle": "MODEL-fp16.trtfb",
  "family": "FAMILY",
  "runtime_strategy": "STRATEGY",
  "precision": "fp16",
  "max_cache_length": 256,
  "prompt": "test prompt",
  "max_new_tokens": 20,
  "threshold_overrides": {
    "logit_cosine_p5": 0.0,
    "stable_top1_match_rate": 0.0,
    "token_agreement_rate": 0.0,
    "normalized_text_edit_distance": 1.0
  }
}
```

The relaxed thresholds are needed because FP16 TRT generates different
text than FP32 HF — the E2E test validates the model runs successfully
and produces non-degenerate output, not exact text match.

## Invariants (NEVER violate)

1. You are NOT DONE until the progress file contains a `best_passing`
   entry with `verified: true` that is NOT FP32.

2. After EVERY attempt (pass or fail), update the progress file.

3. If `--precision fp16` doesn't actually produce FP16 (same bundle
   size as FP32), you MUST fix the builder code. Read the FP16 skill
   guide at `.claude/skills/fp16-trt-network.md`.

4. Validation is the E2E pytest harness. PASS = the pytest exits 0.
   FAIL = pytest exits non-zero. You cannot override this.

5. You may try as many variants in parallel as you have compute.

## Progress File Format

Write to PROGRESS_FILE after every attempt:

```json
{
  "model": "MODEL_ID",
  "started": "ISO-8601 timestamp",
  "attempts": [
    {
      "id": 1,
      "precision": "fp16",
      "quantize": null,
      "status": "pass|build_failed|accuracy_failed|builder_fix_needed|error",
      "bundle_size_mb": 1557,
      "fp32_bundle_size_mb": 3111,
      "e2e_result": "PASSED|FAILED",
      "bundle": "/path/to/bundle.trtfb",
      "manifest": "model-fp16",
      "verified": true,
      "error": null,
      "code_changes": ["families/whisper.py: threaded precision through build_engine"]
    }
  ],
  "best_passing": {
    "precision": "fp16",
    "quantize": null,
    "bundle_size_mb": 1557,
    "bundle": "/path/to/bundle.trtfb",
    "verified": true
  }
}
```

## Strategy Suggestions (NOT mandatory)

- Build FP32 first to get baseline bundle size
- Build FP16 — compare sizes to detect if it took effect
- If FP16 didn't take effect: read the builder, apply the FP16 pattern, rebuild
- Run E2E to validate
- Then try quantization (FP8, INT8) if tools are available
