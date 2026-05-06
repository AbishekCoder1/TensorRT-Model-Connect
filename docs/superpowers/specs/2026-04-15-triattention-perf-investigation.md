# TriAttention Performance Investigation

Date: 2026-04-15

## Question

Why does upstream TriAttention report throughput gains on long reasoning workloads,
while the native C++ TRT runtime in this repo shows a large slowdown once
compression activates?

## Upstream performance model

From the official TriAttention repo:

- the reported gains are on **long reasoning** workloads with contexts far larger
  than the retained KV budget
- the runtime path is designed around **lazy activation** and keeps the normal
  vLLM decode path untouched before the first compression boundary
- scoring is intended to run on **GPU/Triton**, with selection and gather on GPU
- the vLLM integration also includes **physical block reclaim** and scheduler
  support, so reduced KV usage feeds back into runtime throughput and capacity

In other words, upstream gets faster when:

1. the request spends many decode steps in the compressed regime
2. compression is relatively infrequent
3. compression work itself stays on GPU and is cheaper than the attention work it avoids

## Local benchmark setup

Model/bundles:

- `artifacts/triattention/qwen3-0.6b-dense256.trtfb`
- `artifacts/triattention/qwen3-0.6b-tri192-b128-r120-max.trtfb`

Prompt:

- `artifacts/triattention/marker_prompt_136tok.txt`

Key runs:

- dense baseline, 40 decode tokens:
  - `artifacts/triattention/bench_dense256_marker136_40.err`
- dense baseline, 80 decode tokens:
  - `artifacts/triattention/bench_dense256_marker136_seq.err`
- native TriAttention, 40 decode tokens:
  - `artifacts/triattention/bench_tri192_b128_marker136_40_profile.err`
- native TriAttention, 80 decode tokens with profiling:
  - `artifacts/triattention/bench_tri192_b128_marker136_profile.err`

## Observed behavior

### 1. Pre-compression decode is already fine

Before compaction fires, native TriAttention is basically identical to dense:

- dense256, 40 tokens: `606.05 tok/s`
- tri192-b128, 40 tokens: `608.32 tok/s`

This means the steady-state decode path is not the problem.

### 2. The slowdown appears exactly at the first compaction

When decode length reaches the first physical-full boundary:

- dense256, 80 tokens: `604.07 tok/s`
- tri192-b128, 80 tokens: `144.37 tok/s`

The profiled TriAttention run shows exactly one compaction event per generation:

- `compact abs_pos=192 old_rows=192 kept_rows=127`

If we extrapolate the no-compaction 40-token run to 80 tokens, native
TriAttention should be around `131-132 ms` of decode time. The actual profiled
80-token run is `554.13 ms`.

Missing time:

- `~422 ms`

That missing time matches the compaction profile almost exactly:

- `select_ms=397.30 ms`
- `repack_ms=25.45 ms`
- total compaction tax: `~422.75 ms`

## Where the time goes

From the native C++ compaction profile:

- `host_copy_ms=1.05 ms`
- `host_convert_ms=69.37 ms`
- `score_ms=301.42 ms`
- `combine_ms=0.19 ms`
- `select_ms=397.30 ms`
- `repack_ms=25.45 ms`
- `host_copy_mb=21.00`
- `repack_mb=27.34`
- `repack_calls=7000`

Interpretation:

1. D2H bandwidth is not the main bottleneck.
   The raw GPU->CPU copy is only about `1 ms`.

2. CPU-side data conversion is already expensive.
   Converting copied cache rows into host float arrays costs about `69 ms`.

3. CPU scoring is the dominant cost.
   The trigonometric scoring loop costs about `301 ms` for one compression.

4. Repacking is also inefficient.
   We spend about `25 ms` issuing `7000` tiny device-to-device row copies.

## Why upstream can be faster but ours is slower

### A. Upstream scoring stays on GPU

Upstream TriAttention is built around Triton scoring on GPU. Our native C++
runtime copies cache rows to host and scores them on CPU in
`src/runtime/core/triattention_kv_cache.cpp`.

That alone explains most of the delta:

- upstream fast path: GPU scoring
- local path: `~370 ms` of host conversion + CPU scoring per compression

### B. Our compaction repack path is row-by-row memcpy launch overhead

The native C++ repack loop does one `cudaMemcpyAsync` for K and one for V for
every moved row of every layer.

In the profiled Qwen3-0.6B case that becomes:

- `7000` D2D copy launches
- only `27.34 MiB` total moved

So the issue is not bandwidth; it is launch granularity.

### C. Our runtime does not inherit upstream vLLM system-level wins

Upstream’s throughput story is not only the selector itself. It also benefits
from:

- paged KV layout
- physical block reclaim
- scheduler/runtime integration around compressed effective length
- a decode-heavy long-context serving regime

Our TRT runtime currently compresses an internal dense cache object but does not
get those scheduler-side or block-manager-side benefits.

### D. The tested workload is too short to amortize a heavy compression event

In this local benchmark:

- there is only one compaction event
- only ~24 decode tokens remain after compaction

So even a correct compression policy cannot win if that single compression costs
hundreds of milliseconds.

### E. GQA models are especially unfriendly to the current local layout

Qwen3-0.6B has:

- `num_attention_heads = 16`
- `num_key_value_heads = 8`

But the local runtime cache dimension is derived from `attention_size`
(`num_heads * head_dim`) in `src/runtime/plugins/shared/plugin_helpers.cpp`,
not from `num_key_value_heads * head_dim`.

That means the local dense cache layout is wider than an upstream paged KV-head
layout for this GQA model, increasing compaction traffic and scoring footprint.

## Root cause summary

The missing speedup is explained by four concrete local costs:

1. CPU scoring instead of Triton/GPU scoring.
2. Host-side dtype conversion of full cache rows before scoring.
3. Thousands of tiny row-level D2D copies during repack.
4. No vLLM-style paged/block reclaim integration to convert smaller effective KV
   usage into runtime throughput gains.

## Highest-impact fixes

### 1. Replace host scoring with a GPU scoring path

Best option:

- port the selector to CUDA/Triton-style device scoring
- keep keys on device
- perform top-k on device

This is the biggest win. The current profile says the host selection path alone
costs about `397 ms` per compression.

### 2. Replace row-by-row repack with batched gather/scatter or block remap

Best option:

- compact with a single gather-style kernel per layer or per group
- ideally operate on paged/block metadata instead of copying individual rows

The current row-copy loop is launch-bound, not bandwidth-bound.

### 3. Move toward KV-head-native cache layout for GQA models

If the engine/runtime can avoid expanded query-head cache layout, compaction
traffic for Qwen-style models should drop materially.

### 4. Add runtime policy that only enables compression when enough future decode
tokens remain to amortize the cost

This is a secondary mitigation, not the main fix.

It would prevent obviously losing cases where:

- the first compression happens very late
- only a small number of decode tokens remain afterward

## Current conclusion

Upstream TriAttention can legitimately improve throughput on long reasoning
because its intended runtime path keeps compression work on GPU and amortizes
that work over many subsequent decode steps in the compressed regime.

The local native C++ port misses that performance because one compression event
currently costs about `422 ms`, mostly from host conversion + CPU scoring, and
that completely dominates the saved attention work on this benchmark.

## Fix outcome

The native C++ runtime now uses:

- GPU candidate scoring in `src/runtime/core/triattention_kernels.cu`
- host-side normalization / combine over much smaller score buffers
- GPU gather-style KV repack with per-layer scratch buffers instead of
  thousands of row-level `cudaMemcpyAsync` calls

Key post-fix runs:

- dense baseline, 80 decode tokens, 3 iterations:
  - `artifacts/triattention/bench_dense256_marker136_iter3.err`
- native TriAttention, 80 decode tokens, 3 iterations:
  - `artifacts/triattention/bench_tri192_b128_marker136_iter3.err`
- native TriAttention, 80 decode tokens, profiling enabled:
  - `artifacts/triattention/bench_tri192_b128_marker136_profile.err`

Observed post-fix behavior:

- dense256, 80 tokens, 3 iterations: `608.00 tok/s`
- tri192-b128, 80 tokens, 3 iterations: `601.23 tok/s`

So the native TriAttention path is now within about `1.1%` of the dense256
baseline on this benchmark, instead of falling to about `144 tok/s`.

The compaction profile now shows that the old hotspot is effectively gone:

- `trig_prep_ms=0.03`
- `score_ms=1.23`
- `combine_ms=0.02`
- `select_ms=1.32`
- `repack_ms=0.47`

Total compaction tax:

- about `1.8 ms`

That replaces the earlier:

- `select_ms=397.30`
- `repack_ms=25.45`

which totaled about `422.75 ms`.

## Correctness after the perf fix

The cache-compression benefit is still preserved on the conversational recall
prompt:

- `artifacts/triattention/run_dense128_qwen_chat_reasonable_short.out`
- `artifacts/triattention/run_dense256_qwen_chat_reasonable_short.out`
- `artifacts/triattention/run_tri192_b128_r120_max_qwen_chat_reasonable_short.out`

Current outputs:

- dense128: repeated garbage (`enderit ...`)
- dense256: `OTTER581`
- tri192-b128-r120-max: `OTTER581`

So native TriAttention still matches the larger dense baseline on the retained
context task while using the smaller KV budget.

## Secondary bug found while landing the fix

While enabling the CUDA-backed path, `TriAttentionKvCache` gained extra members
behind `#ifdef TRTMC_HAS_CUDA_KERNELS` in the public header. `trtmc_core` was
compiled with `TRTMC_HAS_CUDA_KERNELS=1`, but downstream test translation units
were not, so stack-allocated `TriAttentionKvCache` objects were compiled with a
smaller class layout than the library implementation expected.

That caused stack-smash failures in `test_triattention_kv_cache` even when the
GPU path was disabled at runtime.

The fix was to propagate `TRTMC_HAS_CUDA_KERNELS=1` as a public compile
definition from `trtmc_core` in `CMakeLists.txt`, so all consumers of the public
header see the same class layout.

## Matched-precision correction

One later dynamic-KV investigation initially appeared to show that the
single-engine dynamic path was still much slower than dense decode.

That conclusion was wrong because the compared artifacts were built at
different precisions:

- `artifacts/triattention/qwen3-0.6b-dense192.trtfb`: `fp16`
- `artifacts/triattention/qwen3-0.6b-dense256.trtfb`: `fp16`
- `artifacts/triattention/qwen3-0.6b-tri192-b128-r120-max-dynkv.trtfb`: `fp32`

So the apparent dynamic-shape penalty was mostly a precision mismatch, not a
runtime architecture problem.

The corrected matched-precision TriAttention bundle is:

- `artifacts/triattention/qwen3-0.6b-tri192-b128-r120-max-dynkv-fp16.trtfb`

Corrected GPU-sampler benchmark runs on
`artifacts/triattention/marker_prompt_136tok.txt`:

- no-compaction short decode, dense192:
  - `~709.67 tok/s`
- no-compaction short decode, tri dyn fp16:
  - `~727.56 tok/s`
- long decode, dense256:
  - `~677.30 tok/s`
- long decode, tri dyn fp16:
  - `~716.70 tok/s`

So with matched `fp16` builds, the single-engine dynamic-KV path is not slower.
It is slightly faster than the dense baselines even before the first
compaction, and it shows a clearer end-to-end gain once the decode is long
enough for compressed-KV execution to matter.

The profiled long run confirms that compression is active in the winning case:

- `compact#1 abs_pos=192 old_rows=192 kept_rows=127`
- `compact#2 abs_pos=257 old_rows=192 kept_rows=127`
- `select_ms≈1.3-1.5`
- `repack_ms≈0.47`

That gives an end-to-end result with both:

1. the retained-context benefit of TriAttention
2. a real throughput win over the larger dense baseline
