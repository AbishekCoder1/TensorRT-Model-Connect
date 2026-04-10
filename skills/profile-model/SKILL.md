---
name: profile-model
description: >-
  Profile a TRT model end-to-end: build bundle, measure TRT vs HF vs torch.compile
  latency, per-layer kernel timing, CPU phase breakdown, and HTML report. Use when
  the user wants to understand performance characteristics of a model, find bottlenecks,
  or measure optimization impact. Invoke via /profile-model.
---

# Profile Model: TRT Inference Performance Analysis

## Overview

You are a performance profiling agent. Run a structured profiling workflow for any
model supported by trt-transformers, producing actionable bottleneck analysis. All
commands run inside the dev container.

## Preconditions

- GPU + CUDA + TensorRT available (container environment)
- `trtf_build` installed in editable mode (`pip install --no-deps -e trtf_build/`)
- C++ binary built at `./build/trtf` (optional, for C++ pass)
- Model accessible via HuggingFace ID or local directory

## Workflow

### Step 0: Environment Check (ALWAYS run first)

Before doing anything else, verify the environment can run profiling:

```bash
# Check all preconditions in one shot
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null && echo "GPU: OK" || echo "GPU: MISSING"
python3 -c "import tensorrt as trt; print(f'TRT: {trt.__version__}')" 2>/dev/null || echo "TRT: MISSING"
which trtf-build 2>/dev/null && echo "trtf-build: OK" || echo "trtf-build: MISSING (run: pip install --no-deps -e trtf_build/)"
test -x ./build/trtf && echo "C++ binary: OK" || echo "C++ binary: MISSING (optional — run: cmake --build build -j)"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch: MISSING"
```

**If any required check fails (GPU, TRT, or trtf-build), auto-recover:**

**Case A: No GPU detected (not inside a container)**

Check if a team container already exists:
```bash
docker ps -a --filter "name=trtf-dev-gb300" --format "{{.Names}} {{.Status}}"
```

- **If a running container exists** (e.g., `trtf-dev-gb300-vivian`):
  Run all subsequent commands inside it via `docker exec`:
  ```bash
  docker exec trtf-dev-gb300-<team-id> <command>
  ```

- **If a stopped container exists**: Start it first:
  ```bash
  docker start trtf-dev-gb300-<team-id>
  ```

- **If no container exists**: Bootstrap a new isolated workspace:
  ```bash
  ./scripts/bootstrap_workspace.sh --id <team-id> --branch $(git branch --show-current) --detach
  ```
  This creates a new container `trtf-dev-gb300-<team-id>` with a full repo clone,
  runs `setup_container.sh` inside it (editable install + C++ build + tests), and
  leaves it running in the background. Then use `docker exec` for all commands.

- **If Docker image doesn't exist**: Build it first:
  ```bash
  ./scripts/docker_build_gb300.sh
  ```

**Case B: Inside container but trtf_build not installed**
```bash
pip install --no-deps -e trtf_build/
```

**Case C: Inside container but C++ binary missing (needed for C++/nsight passes)**
```bash
./scripts/setup_container.sh
```
Or manually:
```bash
cmake -S . -B build -G Ninja \
  -DTRTF_TRT_INCLUDE_DIR="${TRT_INC_DIR:-/usr/include/aarch64-linux-gnu}" \
  -DTRTF_TRT_LIBRARY="${TRT_LIB_DIR:-/opt/venv/lib/python3.12/site-packages/tensorrt_libs}/libnvinfer.so" \
  -DTRTF_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
  -DTRTF_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so
cmake --build build -j
```

**After auto-recovery:** Re-run the environment check to confirm everything passes.
Do NOT proceed to Step 1 until all required preconditions pass.

**Important:** When running commands via `docker exec`, prefix every command in
Steps 1-8 with `docker exec trtf-dev-gb300-<team-id>`. For multi-line commands,
wrap in `bash -c '...'`.

### Step 1: Determine what to profile

Ask the user (or infer from context):
- **Model**: HF repo ID (e.g., `Qwen/Qwen3-0.6B`) or local path
- **Bundle**: Pre-built `.trtfb` path, or build on the fly
- **Depth**: Quick (e2e only) or full (e2e + per-layer + CPU phases)

If the user just says "profile X", default to full depth.

**Detect model type** to select the right profiling path. Check the model's
`runtime_strategy` from the E2E manifest or bundle:
- `decoder_kv_cache`, `decoder_moe`, `ssm_recurrent`, `rwkv_recurrent` → **Decoder path** (Steps 3-4)
- `encoder_only`, `embedding`, `reranking` → **Encoder path** (single forward pass, Step 3 only)
- `diffusion_flux`, `diffusion_wan`, `diffusion_zimage` → **Diffusion** — not yet supported
  by the unified profiler. Use E2E test with `--e2e-artifacts-dir` for timing.
  Diffusion profiling support (per-component timing, per-step denoiser analysis,
  SOL estimate) will be added after MR 109 merges.
- `speech_to_text`, `text_to_audio_*`, `vision_language` → **Multi-stage** — use E2E test
  for now; dedicated profiling support planned.

```bash
# Check from E2E manifest:
cat tests/e2e/models/<model-name>.json | grep runtime_strategy

# Or from bundle:
./build/trtf inspect <bundle.trtfb> | grep "Runtime strategy"
```

### Step 2: Build the bundle (if needed)

Skip if the user provides `--bundle`.

```bash
trtf-build build <model> -o /tmp/<model-name>.trtfb --max-cache-length 256 --verbose
```

Verify the bundle:
```bash
trtf-build inspect /tmp/<model-name>.trtfb
```

### Step 3: Run the unified profiler

**Quick profile (e2e comparison only):**

```bash
python tools/trtf_profile.py \
  --model <model> \
  --bundle /tmp/<model-name>.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --warmup 3 --iterations 10 \
  --dtype float16 \
  --json --output-dir /tmp/<model-name>_profile
```

**Full profile (e2e + per-layer + CPU phases + C++ binary):**

```bash
python tools/trtf_profile.py \
  --model <model> \
  --bundle /tmp/<model-name>.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --warmup 3 --iterations 10 \
  --dtype float16 \
  --trtf-binary ./build/trtf \
  --hf-python /opt/venv/bin/python \
  --cpu-profile \
  --json --output-dir /tmp/<model-name>_profile
```

**Key flags:**
| Flag | Purpose | When to use |
|------|---------|-------------|
| `--no-compile` | Skip torch.compile pass | When inductor is broken or model unsupported |
| `--compile-mode max-autotune` | Thorough torch.compile | When comparing kernel quality |
| `--no-layer-profile` | Skip IProfiler pass | Quick e2e-only comparison |
| `--cpu-profile` | CPU phase breakdown | When investigating decode overhead |
| `--nsight` | Nsight Systems capture | Deep kernel analysis (needs --trtf-binary + --bundle) |
| `--trust-remote-code` | HF custom code | Phi-3, StarCoder2, etc. |

### Step 4: Run standalone CPU phase breakdown (decoder models only)

If the unified profiler's CPU phase data is insufficient, run the dedicated tool:

```bash
python tools/cpu_profile.py \
  --model <model> \
  --bundle /tmp/<model-name>.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 10 \
  --warmup 3 --iterations 20 \
  --json /tmp/<model-name>_profile/cpu_profile_detailed.json
```

For Mamba/SSM models:
```bash
python tools/cpu_profile.py \
  --model <model> \
  --bundle /tmp/<model-name>.trtfb \
  --runner mamba \
  --max-new-tokens 10 \
  --json /tmp/<model-name>_profile/cpu_profile_mamba.json
```

### Step 5: Run cross-strategy comparison (optional)

When comparing CPU bottlenecks across different model types:

```bash
python tools/cpu_profile_matrix.py \
  --engine-dir /workspace/users/yifeif/trt-transformers/engines \
  --strategies decoder_kv_cache ssm_recurrent \
  --json /tmp/matrix_profile.json \
  --html /tmp/matrix_profile.html
```

### Step 6: Generate HTML report

If `--json` was used with `trtf_profile.py`, the HTML report is auto-generated.
Otherwise, generate manually:

```bash
python tools/profile_report.py \
  --output-dir /tmp/<model-name>_profile \
  -o /tmp/<model-name>_profile/report.html
```

### Step 7: Interpret results and report

Read the JSON artifacts and provide analysis. Use this decision tree:

**Bottleneck Classification from CPU Phases:**

| Condition | Classification | Recommendation |
|-----------|---------------|----------------|
| `d2h + argmax > 15%` of step time | **Sync bottleneck** | GPU argmax (`TRTF_GPU_ARGMAX=1`) eliminates D2H + CPU argmax |
| `tensor_bind > 10%` of step time | **Launch overhead** | CUDA Graphs capture/replay eliminates per-step binding |
| `execute > 75%` of step time | **Compute-bound** | FP16/BF16 precision (`--precision fp16`) |
| `execute < 50%`, no single dominant | **Mixed overhead** | GPU argmax + CUDA Graphs + FP16 combined |

**Speedup Interpretation:**

| TRT vs HF speedup | What it means |
|-------------------|---------------|
| < 2x | TRT overhead is high relative to model compute; check CPU phases |
| 2-5x | Normal for small models; HF has Python overhead |
| 5-10x | Good; TRT kernel optimization contributing |
| > 10x | Excellent; check if HF is doing something suboptimal |

**TRT vs torch.compile interpretation:**

| TRT/compile ratio | What it means |
|-------------------|---------------|
| < 1.5x | Kernel quality similar; TRT win is from overhead reduction |
| 1.5-3x | Moderate kernel advantage from TRT |
| > 3x | Significant kernel advantage; TRT engine optimization is the main win |

**Per-layer bottleneck:**
- Top 1-3 layers consuming >30% of total → investigate those specific ops
- Even distribution across layers → systemic overhead, not layer-specific

### Step 8: Report to user

Provide a structured summary:

```
## Profile Results: <model-name>

### E2E Performance
- TRT (C++):     X tok/s (Y ms/token)
- TRT (Python):  X tok/s (Y ms/token)
- HF eager:      X tok/s (Y ms/token)
- torch.compile: X tok/s (Y ms/token)
- Speedup:       Xx vs HF, Xx vs compile

### Bottleneck Analysis
- Classification: <sync|compute|latency|mixed>
- Top CPU phase: <phase> at X% of step time
- Top TRT layer: <layer_name> at X% of kernel time

### Recommendations
1. <highest impact recommendation>
2. <second recommendation>
3. <third recommendation>

### Artifacts
- JSON: /tmp/<model>_profile/perf_compare.json
- HTML: /tmp/<model>_profile/report.html
```

## Comparing Before/After Optimization

When the user wants to measure the impact of an optimization:

```bash
# 1. Profile baseline
python tools/trtf_profile.py --model <model> --bundle <bundle> \
  --json --output-dir /tmp/before --cpu-profile

# 2. Apply optimization (e.g., rebuild with FP16, enable CUDA graphs)

# 3. Profile after
python tools/trtf_profile.py --model <model> --bundle <new_bundle> \
  --json --output-dir /tmp/after --cpu-profile

# 4. Compare JSONs manually
```

Read both `perf_compare.json` files and compute deltas for decode_ms, tps,
and per-phase breakdowns. Report as a before/after table.

## Standalone perf_compare.py (lighter weight)

When the unified profiler is overkill:

```bash
# TRT vs HF only
python tools/perf_compare.py \
  --model <model> \
  --bundle <bundle> \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --json /tmp/perf.json

# With C++ binary
python tools/perf_compare.py \
  --model <model> \
  --bundle <bundle> \
  --trtf-binary ./build/trtf \
  --hf-python /opt/venv/bin/python \
  --json /tmp/perf.json

# Record to perf database for regression tracking
python tools/perf_compare.py \
  --model <model> \
  --bundle <bundle> \
  --json /tmp/perf.json \
  --perf-db /tmp/perfdb.sqlite
```

## Real Output Examples

### trtf_profile.py console output (Qwen3-0.6B on GB300):

```
────────────────────────────────────────────────────────────────────────────────
  E2E Latency Comparison
────────────────────────────────────────────────────────────────────────────────
                          TRT (C++)        TRT (Python)         HF (eager)    HF (reduce-overhead)
  ──────────────────  ──────────────────  ──────────────────  ──────────────────  ──────────────────
  Prefill (ms)             1.2 ± 0.1          2.3 ± 0.2          5.1 ± 0.3          4.8 ± 0.2
  Decode (ms)             98.4 ± 1.2        144.9 ± 2.1       1096.0 ± 5.4        347.0 ± 3.2
  Throughput (t/s)            203                138                  18                 58
  Speedup vs TRT (C++)         —              0.68x              0.09x              0.29x

  Token match (TRT Python vs HF eager): True

────────────────────────────────────────────────────────────────────────────────
  Per-Layer TRT Kernel Timing  (total: 4.832 ms/step)  — top 15 slowest
────────────────────────────────────────────────────────────────────────────────
  Layer                                         Mean (ms)       Std      %
  ──────────────────────────────────────────  ─────────  ───────  ──────
  layer_23_mlp_gate_up_matmul                    0.2145   0.0012   4.4%
  layer_22_mlp_gate_up_matmul                    0.2134   0.0011   4.4%
  layer_23_attention_qkv_matmul                  0.1876   0.0009   3.9%
  ...
  Bottleneck: 'layer_23_mlp_gate_up_matmul'  (0.2145 ms, 4.4%)
```

### cpu_profile.py console output:

```
  CPU Phase Breakdown (decoder, 20 iterations)
  ─────────────────────────────────────────────
  Phase          Mean (ms)    Std      %
  ──────────     ─────────    ───    ─────
  mask_build       0.487     0.023   14.8%
  h2d              0.098     0.005    3.0%
  tensor_bind      0.312     0.018    9.5%
  execute          1.534     0.045   46.6%
  d2d_cache        0.102     0.003    3.1%
  d2h              0.412     0.012   12.5%
  argmax           0.347     0.009   10.5%
  ─────────────────────────────────────────────
  TOTAL            3.292 ms/step

  Bottleneck: execute (46.6%)
  Secondary: mask_build (14.8%), d2h (12.5%), argmax (10.5%)
```

## Environment Variables Reference

| Variable | Values | Description |
|----------|--------|-------------|
| `TRTF_GPU_ARGMAX` | `0` (default), `1` | GPU-side argmax; eliminates D2H logits transfer |
| `TRTF_DISABLE_CUDA_GRAPH` | `0` (default), `1` | Disable CUDA Graph capture/replay |
| `TRTF_TRT_LOG_STDERR` | `0`, `1` | TRT logger output to stderr |
| `TRTF_TRT_LOG_MIN_SEVERITY` | `0`-`4` | TRT log severity filter |

## Composability with Other Skills

- If profiling reveals **numerical issues** (TRT vs HF text mismatch), switch to
  `/debug-trt-mismatch` to investigate the divergence.
- After profiling a new family plugin, use `./scripts/validate_family.sh` for full
  correctness validation.
- For **regression tracking**, use `--perf-db` with `perf_compare.py` to store results
  in SQLite and detect regressions across commits.

## Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| TRT build fails | Missing op or unsupported config | Check `--verbose` output; review family plugin's `build_engine()` |
| torch.compile fails | CUDA graph conflict with RoPE | Profiler auto-retries `--compile-mode default`; use `--no-compile` if still fails |
| OOM on HF model load | Model too large for GPU | Use `--dtype float16` (default) or `--trt-only` on perf_compare |
| Mamba model detected | IProfiler not supported for SSM | Profiler auto-skips per-layer pass; CPU profile still works with `--runner mamba` |
| CPU profile timings noisy | Missing cuda-python sync | Ensure `cuda-python` >= 13 for proper `cudaStreamSynchronize` |
| C++ binary not found | Not built or wrong path | Run `cmake --build build -j` first, then pass `--trtf-binary ./build/trtf` |
| HTML report empty | No JSON artifacts generated | Ensure `--json` flag was passed to trtf_profile.py |
