---
name: transform-model
description: Transform an HF model to torch-trt. Iterates build/validate/fix cycles until comparison output (HF vs torch-trt) is produced. Requires hf_model (HuggingFace ID) and branch arguments.
---

# Autonomous Torch-TRT Model Transform

Transform the given HF model to a torch-trt `.trtfb` bundle.

**Required arguments (pass after skill name):**
- `hf_model` — HuggingFace model ID (e.g. `Qwen/Qwen3-0.6B`)
- `branch` — Git branch to work on (created if it doesn't exist)
- `container` (optional) — Docker container name (default: `trtf-dev-torchtrt`)

**Container workspace:** `/workspace/trt-transformers-cpp/`

## Hard Rules

1. **NEVER stop until comparison output exists.** You must produce a side-by-side comparison of HF reference output vs torch-trt output before reporting to the user.
2. **Run everything in the container** via `docker exec <container>`. Never run GPU commands on the host.
3. **Read docs first.** Before writing ANY wrapper code:
   - Read `docs/torch-trt/TORCHTRT_TRANSFORM_GUIDE.md` (full playbook)
   - Read `docs/torch-trt/TORCHTRT_KNOWN_ISSUES.md` (critical pitfalls)
   - Read `docs/torch-trt/TORCHTRT_AGENT_GUIDE.md` (execution plan)
4. **Log all work** in `docs/torch-trt/TORCHTRT_WORKLOG.md` — what was done, what failed, decisions made.
5. **Log new issues** in `docs/torch-trt/TORCHTRT_KNOWN_ISSUES.md` if you discover any.
6. **Keep changes minimal and scoped** to this model. Do not refactor unrelated code.
7. **Do not relax thresholds** unless repeated evidence proves it is necessary.
8. **Commit on your branch only.** Never push to master.
9. **Max 10 fix iterations** before stopping and reporting the blocker.

## Workflow

### Phase 0: Confirm Model with User

Before doing any work, confirm the exact HuggingFace model to transform.
Present the user with:
1. The HF model ID you will use (e.g. `Qwen/Qwen3-0.6B`)
2. The model architecture type you identified (decoder, encoder-only, diffusion, etc.)
3. Whether a family plugin already exists for this model type
4. The branch name you will work on

**Wait for the user to confirm before proceeding.** If the user wants a
different model variant, size, or version, update accordingly.

### Phase 1: Setup

After user confirmation:
```
1. git checkout -b <branch> (or switch to it if it exists)
2. Read docs/torch-trt/TORCHTRT_TRANSFORM_GUIDE.md
3. Read docs/torch-trt/TORCHTRT_KNOWN_ISSUES.md
4. Identify model architecture type (decoder, encoder-only, diffusion, etc.)
5. Check if a family plugin already exists in ttrt_build/ttrt_build/families/
```

### Phase 2: Implement

Choose the right path based on model type:

**Decoder models (causal LM):**
- Strategy: `decoder` (in `ttrt_build/ttrt_build/strategies/decoder.py`)
- Wrapper: `StatelessCacheWrapper` handles KV cache I/O automatically
- You only need a family plugin in `ttrt_build/ttrt_build/families/` if one doesn't exist
- Reference: `ttrt_build/ttrt_build/families/qwen.py` (standard decoder)

**Encoder-only models:**
- Strategy: `encoder_only` (in `ttrt_build/ttrt_build/strategies/encoder_only.py`)
- Reference: `ttrt_build/ttrt_build/families/bert.py`

**Diffusion models (multi-engine):**
- Strategy: `diffusion` (in `ttrt_build/ttrt_build/strategies/diffusion.py`)
- Must implement `build_components()` in the family plugin
- Reference: `ttrt_build/ttrt_build/families/pixart.py`, `flux.py`, `wan_t2v.py`
- CRITICAL: Read Known Issue #1 (SDPA + masks = NaN) and #3 (multiplicative masking)

**Key files to know:**
| File | Purpose |
|------|---------|
| `ttrt_build/ttrt_build/strategies/__init__.py` | Strategy registry |
| `ttrt_build/ttrt_build/strategies/base.py` | BuildStrategy Protocol |
| `ttrt_build/ttrt_build/compiler.py` | Build orchestrator |
| `ttrt_build/ttrt_build/families/base.py` | FamilyPlugin protocol |
| `scripts/new_family.py` | Auto-scaffold a plugin from HF repo |

### Phase 3: Build

```bash
# Decoder / encoder-only models
docker exec <container> /opt/venv/bin/python -m ttrt_build build <hf_model> \
  -o /workspace/trt-transformers-cpp/engines/<model>.trtfb \
  --max-cache-length 256 --verbose

# Diffusion models
docker exec <container> /opt/venv/bin/python -m ttrt_build build <hf_model> \
  -o /workspace/trt-transformers-cpp/engines/<model>.trtfb --verbose
```

If build fails: read error, check KNOWN_ISSUES.md, fix, retry. Log new issues.

### Phase 4: Validate (THE CRITICAL LOOP)

```
REPEAT until correct output:
  1. Run torch-trt inference, save output
  2. Run HF reference inference, save output
  3. Compare outputs quantitatively
  4. If wrong: diagnose, fix, log issue, rebuild, retry
```

**For text models:**
```bash
# Torch-TRT inference
docker exec <container> ./build/trtf run \
  /workspace/trt-transformers-cpp/engines/<model>.trtfb \
  --prompt "The capital of France is" --max-new-tokens 20 \
  --hf-python /opt/venv/bin/python

# Quantitative comparison (diff_logits battery)
docker exec <container> /opt/venv/bin/python tools/diff_logits.py \
  --model <hf_model> --atol 1e-2 --battery --verbose
```

Pass criteria: `top1_match_rate >= 80%`, `mean_cosine_sim > 0.99`

**For diffusion models:**
```bash
# Run two semantically distinct prompts
docker exec <container> ./build/trtf run \
  /workspace/trt-transformers-cpp/engines/<model>.trtfb \
  --prompt "a golden retriever playing in snow" \
  --hf-python /opt/venv/bin/python \
  -o /workspace/trt-transformers-cpp/outputs/<model>/dog_v1.png

docker exec <container> ./build/trtf run \
  /workspace/trt-transformers-cpp/engines/<model>.trtfb \
  --prompt "a red sports car on a highway" \
  --hf-python /opt/venv/bin/python \
  -o /workspace/trt-transformers-cpp/outputs/<model>/car_v1.png
```

Pass criteria: images visually match prompts, outputs from different prompts are clearly distinct.

### Phase 5: E2E Manifest

Create `tests/e2e/models/<model-name>.json`:
```json
{
  "name": "<model-name>",
  "hf_id": "<hf_model>",
  "bundle": "<model-name>.trtfb",
  "family": "<family_name>",
  "runtime_strategy": "<strategy>",
  "max_cache_length": 256,
  "prompt": "<test prompt>",
  "max_new_tokens": 20
}
```

Run the E2E test:
```bash
docker exec <container> /opt/venv/bin/python -m pytest \
  tests/test_e2e.py::test_e2e[<model-name>] -v \
  --engine-dir /workspace/trt-transformers-cpp/engines \
  --trtf-binary ./build/trtf --hf-python /opt/venv/bin/python \
  --rebuild-engines
```

### Phase 6: Report and Await Feedback

Only after ALL of the following exist:
- Working `.trtfb` bundle that produces correct output
- At least 2 different test prompts validated
- Quantitative comparison metrics (diff_logits or visual comparison)
- E2E manifest created and test passing
- All work logged in TORCHTRT_WORKLOG.md
- Changes committed on the branch

**Then** report to the user with:
1. Model name and HF ID
2. What strategy/wrapper was used
3. Comparison metrics (cosine sim, top1 match, or visual assessment)
4. Sample outputs (text or image paths)
5. Any new issues logged in KNOWN_ISSUES.md
6. Branch name and commit hash

**Then ask the user to review the outputs and provide feedback.**

### Phase 7: Iterate on User Feedback

After the user reviews the comparison output, they may report discrepancies or
request changes. **Do not consider the task complete until the user explicitly
approves the results.**

When the user reports an issue:
1. Acknowledge the specific discrepancy they identified
2. Diagnose the root cause using the Diagnosis Protocol below
3. Fix the wrapper, family plugin, or pipeline code
4. Rebuild the bundle and re-run inference
5. Produce updated comparison output (save with incremented version: `_v2`, `_v3`, etc.)
6. Report the updated results back to the user
7. Repeat until the user approves

**Common user feedback and how to handle it:**

| User says | Action |
|-----------|--------|
| "Output text doesn't match HF" | Re-run diff_logits, check per-token divergence, fix wrapper I/O or cache logic |
| "Image doesn't follow the prompt" | Check attention masking (Issue #3), re-run with distinct prompts |
| "Image quality is degraded" | Check dtype handling, scaling factors, VAE decoder output range |
| "Metrics look off" | Re-examine thresholds, run more prompts, check for edge cases |
| "Try a different prompt" | Run the requested prompt, save output, report back |
| "Looks good" / "Approved" | Task is complete — push branch and clean up |

**The feedback loop continues until the user says the results are acceptable.**
There is no automatic exit — only the user can close the loop.

## Diagnosis Protocol (when output is wrong)

1. **Isolate the problem.** Test each engine component separately.
2. **Check quantitative metrics** before visual inspection.
3. **Reason through the math** — softmax, attention, masking, scaling.
4. **Version your bundles** — save as `<model>_v1.trtfb`, `<model>_v2.trtfb` for A/B testing.
5. **Check KNOWN_ISSUES.md** — your problem may already be solved.
6. Common failure modes:
   - NaN output → SDPA + attention mask (Issue #1). Use TrtSafeAttnProcessor.
   - Prompt ignored → multiplicative masking (Issue #3). Use additive masking.
   - Engine ignores input → uniform trace input constant-folded (Issue #4). Use mixed patterns.
   - All-zero logits → missing `use_explicit_typing=True` (Issue #10).
   - StaticCache breaks graph → in-place `index_copy_` (Issue #9). patch_static_cache_scatter() fixes this.
