# Torch-TRT Model Transform Guide

Step-by-step playbook for agents transforming a new HuggingFace model into a
torch-trt `.trtfb` bundle with validated C++ inference. Follow this document
in order. Do not skip steps.

**Before starting:** Read `docs/torch-trt/TORCHTRT_KNOWN_ISSUES.md` for known pitfalls.
Many hours of debugging are captured there — check it before writing any
wrapper code.

---

## 1. Determine model architecture type

Identify which category the model falls into. This determines the strategy,
wrapper pattern, and validation approach.

| Type | Examples | Strategy | Key challenge |
|------|----------|----------|---------------|
| Decoder (causal LM) | Qwen3, LLaMA, Phi | `decoder` | KV cache I/O, StaticCache |
| Encoder-only | BERT, DistilBERT | `encoder_only` | No cache, simple I/O |
| Diffusion (multi-engine) | PixArt, FLUX, Wan | `diffusion` | 3+ engines, attention masking, scheduler |
| Encoder-decoder | Whisper, T5 | (not yet supported) | Two-phase: encode then decode |

For decoder and encoder-only models, follow `docs/torch-trt/TORCHTRT_AGENT_GUIDE.md`
which has the detailed execution plan. This guide focuses on the transform
process, validation loop, and output requirements that apply to ALL model types.

---

## 2. Write the wrapper and family plugin

### General wrapper rules

Wrappers are thin adapters around unmodified HuggingFace models. They handle:
- **dtype casting** (int32↔int64, fp16↔fp32) at I/O boundaries
- **mask format conversion** (binary → additive)
- **output selection** (extract `.logits`, `.last_hidden_state`, `.sample`, etc.)

Wrappers must NOT:
- Modify model weights
- Change the model's layer structure
- Add trainable parameters

### Before writing attention code

**CHECK `docs/torch-trt/TORCHTRT_KNOWN_ISSUES.md` FIRST.** Key constraints:
- `F.scaled_dot_product_attention` with masks → NaN in TRT
- `torch.baddbmm` → compilation failure in TRT
- Multiplicative masking before softmax → broken conditioning
- Use `torch.matmul` + `add` + `softmax` for any masked attention

### Trace input rules

Every input that varies at runtime must have a **non-trivial pattern** during
tracing. Uniform inputs (all-ones, all-zeros) can be constant-folded by
`torch.export`, causing the engine to ignore that input at runtime.

```python
# BAD: all-ones mask may be constant-folded
mask = torch.ones(1, seq_len)

# GOOD: mixed pattern prevents folding
mask = torch.ones(1, seq_len)
mask[0, seq_len // 2:] = 0.0
```

---

## 3. Build the engine bundle

```bash
# Decoder models
docker exec <container> trtmc-build build --backend torchtrt <HF_ID> \
  -o /workspace/tensorrt-model-connect/engines/<model>.trtfb \
  --max-cache-length 256 --verbose

# Diffusion models (uses build_components path)
docker exec <container> trtmc-build build --backend torchtrt <HF_ID> \
  -o /workspace/tensorrt-model-connect/engines/<model>.trtfb --verbose
```

If the build fails, check `docs/torch-trt/TORCHTRT_KNOWN_ISSUES.md` for the error
message. If it's a new failure, debug it, fix it, and **add the issue to
the known issues file** before continuing.

---

## 4. Validation loop — DO NOT STOP UNTIL OUTPUT IS PRODUCED

This is the critical part. You must iterate until the model produces
correct output. Follow this loop:

```
┌─────────────────────────────────────┐
│ 1. Build engine bundle              │
│ 2. Run inference                    │
│ 3. Save output to accessible folder │
│ 4. Check output quality             │
│    ├── Output is correct → DONE     │
│    └── Output is wrong:             │
│        ├── Diagnose root cause      │
│        ├── Fix wrapper/pipeline     │
│        ├── Log issue in KNOWN_ISSUES│
│        └── Go to step 1             │
└─────────────────────────────────────┘
```

### 4a. Run inference and save output

**ALL output must go to the accessible output folder so the user can inspect it.**

```bash
# Text models — output printed to stdout
docker exec <container> ./build/trtmc run \
  /workspace/tensorrt-model-connect/engines/<model>.trtfb \
  --prompt "<test prompt>" --max-new-tokens 20 \
  --hf-python /opt/venv/bin/python

# Diffusion models — output saved as PNG
docker exec <container> ./build/trtmc run \
  /workspace/tensorrt-model-connect/engines/<model>.trtfb \
  --prompt "<test prompt>" \
  --hf-python /opt/venv/bin/python \
  -o /workspace/tensorrt-model-connect/outputs/<model_name>/<descriptive_name>.png
```

**Path convention:**
- Container path: `/workspace/tensorrt-model-connect/outputs/<model_name>/`
- Host path: `~/trt_repos/tensorrt-model-connect/outputs/<model_name>/`
- NEVER use host paths in `docker exec` commands — the container filesystem
  is different.

**File naming:** Use descriptive names, not `output.png`. Include the prompt
subject and version: `dog_v1.png`, `car_v2.png`, `cat_windowsill_v3.png`.

### 4b. Validate output quality

**For text models:**
- Output text should be coherent and relevant to the prompt
- Run at least 2 different prompts
- Run diff_logits to compare against HF reference:
  ```bash
  docker exec <container> python3 tools/diff_logits.py \
    --model <HF_ID> --atol 1e-2 --battery --verbose
  ```
- Metrics: `top1_match_rate >= 80%`, `mean_cosine_sim > 0.99`

**For diffusion models:**
- Output image should follow the text prompt (not just look good)
- Run at least 2 **semantically distinct** prompts (e.g., "a dog" and "a car")
- Verify the outputs are visibly different in the expected way
- If both prompts produce similar images, text conditioning is broken
- Track `|conditioned_pred - unconditioned_pred|` — if near zero, CFG is broken

### 4c. When output is wrong — diagnosis protocol

1. **Isolate the problem.** Write a Python script that loads engines from the
   bundle and runs them directly (bypass C++ pipeline). See
   `scripts/diagnose_loop.py` for an example.

2. **Check quantitative metrics** before looking at visual output. For diffusion:
   `|pred - uncond|` per step. For decoders: logit cosine similarity per token.

3. **Check the math.** If metrics look "fine" but output is wrong, reason through
   the mathematical operations (softmax, attention, masking) on paper.

4. **Version your bundles.** Save each attempt as `<model>_v1.trtfb`,
   `<model>_v2.trtfb`, etc. When something regresses, A/B test immediately.

5. **Log the issue.** Add it to `docs/torch-trt/TORCHTRT_KNOWN_ISSUES.md` with symptom,
   root cause, and fix — even if it's model-specific.

---

## 5. Final validation gates

Once output is correct, run the full validation suite:

```bash
# Unit tests
docker exec <container> python -m pytest tests/engine_defs/torch_trt/ -v

# C++ tests
docker exec <container> ctest --test-dir build --output-on-failure

# Builder tests
docker exec <container> python -m pytest tests/builder/ -v \
  --ignore=tests/builder/test_cli.py

# Tool tests
docker exec <container> python -m pytest tests/tools/ -v
```

For decoder models, also run:
```bash
docker exec <container> ./scripts/validate_family.sh <HF_ID>
```

---

## 5b. Performance comparison

When comparing Torch-TRT against HF baselines (eager, torch.compile), follow
these rules to get meaningful numbers.

### Fair timing: exclude startup, measure steady-state

**Load models/engines once, then loop inference.** Never spawn a new process
per iteration — process startup, engine deserialization, and model loading
dwarf actual inference time and produce misleading results.

| Cost | Typical time | When to pay |
|------|-------------|-------------|
| Process startup + Python imports | 3–4s | Once |
| TRT engine deserialization (3 engines) | 5–10s | Once |
| HF model load (`from_pretrained`) | 2–10s | Once |
| `torch.compile` compilation | 10–60s+ | Once (first few runs) |
| **Actual inference** | **0.5–3s** | **Every iteration** |

Report startup time separately (model load, compile, engine deser) so the
user can see cold-start cost vs steady-state throughput.

```python
# WRONG — subprocess per iteration includes engine deser every time
for i in range(iterations):
    subprocess.run(["./build/trtmc", "run", bundle, ...])  # 15s = 12s startup + 3s inference

# RIGHT — load once, loop inference
engine = deserialize(bundle)  # 6s one-time
for i in range(iterations):
    run_engine(engine, inputs)  # 1.5s pure inference
```

### Same precision across all backends

All backends in a comparison MUST use the same dtype. If the TRT engines are
fp16, the HF baseline must also be fp16:

```python
# HF eager — fp16 to match TRT
pipe = Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
```

Do not compare fp16 TRT against fp32 HF — the speedup will be inflated by
the precision difference, not the runtime.

### Use `pipe()` for HF baselines, not manual stage calls

Call the HF pipeline's `__call__` method directly instead of manually calling
individual stages (`pipe.transformer()`, `pipe.vae.decode()`, etc.). Manual
calls bypass internal dtype handling, projection layers, and scheduling logic,
causing dtype mismatches and incorrect results.

```python
# WRONG — manual calls bypass internal dtype handling
embeds = pipe.encode_prompt(prompt)
noise = pipe.transformer(latents, encoder_hidden_states=embeds, ...)  # dtype crash

# RIGHT — let the pipeline handle everything
output = pipe(prompt=prompt, num_inference_steps=20, ...)
```

### Reference: existing perf tools

- **Decoder models:** `tools/perf_compare.py` — uses `TrtRunner` (in-process)
- **Diffusion models:** `tools/perf_compare_diffusion.py` — loads TRT engines
  from bundle once, runs full pipeline in Python

---

## 6. Deliverables checklist

Before declaring the model done, ensure ALL of the following:

- [ ] Engine bundle built and saved to `engines/` folder
- [ ] Output images/text saved to `outputs/<model_name>/` with descriptive names
- [ ] At least 2 prompts validated with correct output
- [ ] All test suites pass (unit, C++, builder, tools)
- [ ] Entry added to `docs/torch-trt/TORCHTRT_WORKLOG.md` (what was done, what failed,
      decisions made)
- [ ] Any new issues added to `docs/torch-trt/TORCHTRT_KNOWN_ISSUES.md`
- [ ] E2E manifest added to `tests/e2e/models/<model>.json` (if applicable)
- [ ] One clean commit with clear message

---

## 7. Architecture reference

### How wrappers become engines

The wrapper's `forward()` method is traced by `torch.export.export()` into a
flat computation graph. `torch_tensorrt` converts that graph into a TRT engine.
Everything in `forward()` — dtype casts, mask conversions, model layers,
attention processors — is baked into the engine as TRT operations. There is no
"wrapper" vs "model" distinction in the final engine.

This means:
- Changes to the wrapper require rebuilding the engine
- The C++ pipeline doesn't need to know about wrapper internals
- I/O format is defined by the wrapper's input/output signatures

### Container path mapping

| Container path | Host path |
|---------------|-----------|
| `/workspace/tensorrt-model-connect/` | `~/trt_repos/tensorrt-model-connect/` |
| `/workspace/tensorrt-model-connect/engines/` | `~/trt_repos/tensorrt-model-connect/engines/` |
| `/workspace/tensorrt-model-connect/outputs/` | `~/trt_repos/tensorrt-model-connect/outputs/` |

### Model type → strategy mapping

| Model type | Strategy | Build method | Pipeline |
|-----------|----------|-------------|----------|
| Decoder (causal LM) | `decoder` | `wrap_model()` + `make_export_args()` | Single engine, KV cache |
| Encoder-only | `encoder_only` | `wrap_model()` + `make_export_args()` | Single engine, no cache |
| Diffusion | `diffusion` | `build_components()` | Multiple engines, scheduler |
