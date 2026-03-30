# Flywheel Gap: Perf Instrumentation

## Problem

The performance flywheel cannot make informed optimization decisions because
it has no granular timing data. The E2E test reports total wall time (~18s
for Whisper) which is dominated by Python overhead and HF reference
inference — the actual TRT engine time is invisible.

Without per-stage timing, the agent cannot determine:
- Whether FP16 actually sped up TRT inference (vs just reducing memory)
- Which phase is the bottleneck (engine load? prefill? decode? post-process?)
- Whether quantization (FP8/INT8) would help (compute-bound vs memory-bound)

## What's Needed

### 1. C++ Binary Instrumented Timing

The `./build/trtf run` binary should output JSON timing when asked:

```bash
./build/trtf run bundle.trtfb --prompt "test" --max-new-tokens 20 \
  --hf-python /opt/venv/bin/python --perf-json /tmp/perf.json
```

Output:
```json
{
  "engine_load_ms": 245.3,
  "prefill_tokens": 5,
  "prefill_ms": 12.1,
  "decode_steps": 20,
  "decode_total_ms": 185.4,
  "decode_per_token_ms": 9.27,
  "tokens_per_second": 107.8,
  "peak_gpu_memory_mb": 1557,
  "cache_element_size": 2,
  "precision": "fp16"
}
```

This requires CUDA event timing in the decode loop:
- `cudaEventRecord` before/after each `enqueueV3()` call
- `cudaEventSynchronize` + `cudaEventElapsedTime` for each step
- Report min/max/avg/p50/p99 decode latency

### 2. Python Debug Runner Timing

The `TrtRunner` in `debug_runner.py` should also report per-step timing
so the flywheel can benchmark without the C++ binary:

```python
runner = TrtRunner(bundle, max_cache_length, num_layers)
runner.step(token_id)  # returns (logits, step_time_ms)
```

### 3. E2E Harness Stage Timing

The E2E orchestrator should record and persist per-stage timing in the
result JSON:

```json
{
  "stages": {
    "full_generation": {
      "runner_time_s": 0.185,
      "reference_time_s": 12.4,
      "compare_time_s": 0.02
    }
  }
}
```

This is partially there (the orchestrator has timing code) but the fields
aren't written to the result JSON.

### 4. nsight-systems Integration (Optional, Advanced)

For detailed GPU kernel analysis:

```bash
nsys profile --output /tmp/whisper_fp16.nsys-rep \
  ./build/trtf run bundle.trtfb --prompt "test" --max-new-tokens 20
```

This produces flame graphs showing exactly which TRT kernels run and
their GPU utilization. Not needed for the flywheel's automated decisions
but valuable for human debugging.

## Impact on Flywheel

With instrumented timing, the agent can:
1. Compare TRT-only latency (not polluted by Python/HF overhead)
2. Identify compute-bound vs memory-bound models
3. Make informed decisions about which quantization format to try
4. Report actual speedup numbers in the progress file

## Files to Modify

| File | Change |
|------|--------|
| `src/runtime/pipelines/text_generation_pipeline.cpp` | Add CUDA event timing around decode loop |
| `src/runtime/trt/core/device_kv_cache.cpp` | Add timing around `enqueueV3` |
| `src/cabi/api/trtf_c.cpp` | Expose timing via C API |
| `trtf_build/trtf_build/debug_runner.py` | Add per-step timing |
| `tests/e2e_harness/orchestrator.py` | Write stage timing to result JSON |
| CLI argument parsing in `src/` | Add `--perf-json` flag |
