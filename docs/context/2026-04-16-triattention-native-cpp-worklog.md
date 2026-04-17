# TriAttention Native C++ Worklog

Date: 2026-04-16

## Purpose

This worklog records how TriAttention was brought from an upstream research
runtime into the native C++ TensorRT runtime in this repo, what design choices
were made along the way, which debugging hypotheses were wrong, and what
finally produced an end-to-end result with both accuracy and throughput
benefit.

This document is intentionally different from the design/spec notes in
`docs/superpowers/specs/`. Those documents capture point-in-time decisions and
investigations. This file captures the full feature narrative.

Relevant companion notes:

- `docs/superpowers/specs/2026-04-15-triattention-integration-design.md`
- `docs/superpowers/specs/2026-04-15-triattention-native-cpp-e2e-status.md`
- `docs/superpowers/specs/2026-04-15-triattention-perf-investigation.md`
- `docs/superpowers/specs/2026-04-15-triattention-overflow-fix.md`

## Final outcome

The final native implementation is a real C++/CUDA runtime path, not a Python
debug shortcut:

- bundle build embeds TriAttention metadata and stats
- runtime compaction is handled by `TriAttentionKvCache`
- selection and repack run natively in C++/CUDA
- the same TRT decoder engine continues after compaction
- runtime KV allocation can be set at launch time with `--kv-cache-size`

On the corrected Qwen3-8B AIME25 pilot slice used for parity checks, native
TriAttention now matches dense TRT on answer accuracy while remaining faster.

Matched 6-sample long-budget slice:

- dense answers:
  `tmp/qwen3_dense_completed6.jsonl`
- TriAttention answers:
  `tmp/qwen3_tri_v14_completed6_long.jsonl`
  `tmp/qwen3_tri_v14_sample5_completed6_long_gpu2.jsonl`
  `tmp/qwen3_tri_v14_sample6_completed6_long_gpu3.jsonl`

Answer parity on that slice:

- `aime25_1 -> 70`
- `aime25_2 -> 588`
- `aime25_3 -> 16`
- `aime25_4 -> 117`
- `aime25_5 -> 279`
- `aime25_6 -> 504`

Throughput on that same slice:

- dense: `64.16 tok/s`
- TriAttention: `86.12 tok/s`

That is about `1.34x` faster while preserving the dense answers on the matched
pilot.

Current reproducible MR-tip validation:

- TriAttention bundle:
  `artifacts/triattention/qwen3-8b-nonflash/qwen3-8b-tri12288-b3072-r128-dynkv-fp16-manual-current.trtfb`
- dense control bundle:
  `artifacts/triattention/qwen3-8b-nonflash/qwen3-8b-dense12288-dynkv-fp16-manual-samefam.trtfb`
- runtime overrides:
  `TRTF_TRIATTN_OVERRIDE_KV_BUDGET=6144`
  `TRTF_TRIATTN_OVERRIDE_DIVIDE_LENGTH=1024`
  `TRTF_TRIATTN_RUNTIME_BUCKET_ROWS=32`
- validation outputs:
  `artifacts/triattention/policy-sweeps/2026-04-17-current-bundle-override-validation/`

Checked results from that current-tree validation:

- `aime25_2`: TriAttention answer `588`, dense answer `588`
- `aime25_3`: TriAttention answer `16`
- `aime25_2` no-stop throughput:
  TriAttention `56.658344 tok/s`
  dense same-family control `42.614745 tok/s`

That is `1.3295x` on the fair same-family dense baseline while preserving the
checked hard answers.

## High-level design choices

### 1. Treat TriAttention as a runtime/cache policy, not an attention-kernel swap

The first important conclusion from upstream was that TriAttention is not
"replace TensorRT attention with a new attention op." It is a cache scoring,
selection, and compaction policy with runtime metadata and lifecycle handling.

That drove the integration point:

- builder persists TriAttention config and calibration stats into the bundle
- runtime owns cache state, position tracking, selection, and compaction
- the TRT engine stays the decoder, but the cache semantics change underneath it

This was the right choice. It kept the project feasible and matched how the
upstream runtime is actually structured.

### 2. Move to native C++ instead of a Python evaluation shim

The first repo-local enablement path used a Python debug runner to validate the
idea. That was useful for initial understanding, but the user explicitly did
not want a Python shortcut. The implementation was therefore moved fully into
the native runtime.

This forced the real project decisions:

- native state object under `IInferenceState`
- native bundle config parsing
- native selector and compaction logic
- native tests and native benchmarks

### 3. Prefer a single dynamic-KV engine over multi-engine switching

There were two plausible ways to reduce real attention work:

- bucketed engine switching
- a single engine with dynamic KV row count

After reviewing TensorRT local docs and the repo's existing dynamic-shape use,
the single-engine path was chosen. That proved sufficient as long as the decode
graph was built with dynamic KV input shapes and the runtime rebound the live
row count.

Why this choice was correct:

- less engine-management complexity
- no explicit engine migration cost
- cleaner runtime model
- consistent with TensorRT's intended dynamic-shape usage

### 4. Separate physical cache limit from logical TriAttention budget

One important quality issue was prompt-time overflow. Using a tiny physical
window and a tiny logical budget caused compaction to happen during prompt
prefill, which damaged the prompt-state before decode started.

The fix was architectural:

- physical limit large enough to hold the prompt cleanly
- logical TriAttention budget smaller and enforced during decode

This is why configurations like `tri256-b128` were materially better than
`tri192-b128` on overflow prompts.

### 5. Make runtime KV allocation configurable in bytes

Dynamic shapes remove the fixed exact KV length at runtime, but not the build
upper bound. The runtime therefore needed a user-facing way to choose the
actual allocation below that upper bound.

The implemented runtime model is:

- build with a large dynamic upper bound
- allocate only the requested runtime amount
- expose that through `--kv-cache-size`

That became a must-have feature because otherwise large build-time bounds would
force wasteful memory allocation at startup.

### 6. Keep GPU work on GPU

The first native implementation was functionally correct enough to prove the
idea, but it was slow because it copied cache rows to host, converted them on
CPU, scored on CPU, and repacked with many tiny D2D row copies.

The final runtime moved the hot pieces to GPU:

- GPU candidate scoring
- GPU gather-style compaction
- host only for small normalization/combine work where still acceptable

This was essential to recovering throughput.

## What was implemented

The final feature spans four layers.

### Bundle/build layer

- TriAttention stats export into bundle JSON sections
- TriAttention runtime config persisted in bundle config
- dynamic-KV-capable decoder engine build

### Runtime state layer

- `TriAttentionKvCache` owns:
  - cache tensors
  - per-head position tracking
  - compaction trigger logic
  - reserved/recent/prefill protection
  - keep-index selection
  - compaction and cache rebound

### CUDA helper layer

- GPU scoring kernel
- GPU KV gather/repack kernel

### CLI/runtime config layer

- runtime KV byte budget
- dynamic external KV binding without eager allocation of the full engine max

## Debugging chronology

### Phase 1: understand upstream correctly

The first upstream reading established two things:

1. TriAttention is a runtime policy, not a TRT graph replacement.
2. Upstream's best performance story is tied to long reasoning workloads,
   GPU scoring, and serving-runtime cooperation.

This prevented an early wrong turn: trying to "just add a new attention layer"
in TRT.

### Phase 2: native C++ port with a shared-row approximation

The first native version worked as a C++ cache compaction path and proved that:

- compaction fired in native code
- the runtime could continue generation after compaction
- retained-memory prompts behaved better than undersized dense baselines

But this version still used a simplified shared-row interpretation compared to
the stronger upstream per-head behavior.

### Phase 3: sanity checking with retrieval-style prompts

The first correctness checks used marker/retrieval prompts. These were useful
because they had deterministic expected answers, but they were not persuasive as
general quality evidence.

That led to a better evaluation discipline:

- use retrieval-style prompts only for deterministic cache sanity checks
- present normal conversational prompts when discussing visible output quality

### Phase 4: performance regression investigation

The first native C++ compaction path was dramatically slower than dense once
compaction engaged.

The key finding from profiling was simple:

- pre-compaction decode was already fine
- nearly all slowdown came from the compaction event itself

The cost breakdown pointed at:

- host copy + dtype conversion
- CPU scoring
- thousands of tiny D2D repack copies

That led directly to the GPU scoring and GPU gather/repack implementation.

### Phase 5: dynamic-KV path and runtime KV sizing

Reducing cache residency alone is not enough if the engine still binds and
processes the full physical window. The runtime was therefore extended to bind a
live KV row count into a single dynamic-shape engine.

This was paired with runtime byte-budget control so the actual allocation could
be chosen at launch time instead of being fully materialized at bundle max.

### Phase 6: overflow bug during prompt prefill

One of the harder early failures was long-context overflow on normal prompts.
The issue was not initially obvious because the selector appeared reasonable.

The actual problem was compaction during prefill in too-small a physical cache.

Fix:

- use physical slack during prompt fill
- decouple physical capacity from logical keep budget
- ensure compaction starts after decode crosses the physical threshold, not
  while prompt state is still being built

### Phase 7: evaluation recipe was wrong

A separate accuracy confusion came from the benchmark recipe itself. Early AIME
results looked poor for dense, TriAttention, and HF because the decode recipe
was throughput-oriented:

- greedy or deterministic decode
- poor stopping behavior
- simplistic extraction
- non-chat prompting in some earlier runs

After switching to the corrected Qwen reasoning recipe, dense TRT recovered to
HF parity on the pilot slice. This mattered because it separated "bad eval
recipe" from "TriAttention bug."

### Phase 8: remaining TriAttention accuracy gap after compaction

At that point dense and HF were aligned, but TriAttention still failed hard
reasoning samples once real compression activated.

Several debugging hypotheses were tested.

#### Hypothesis: sampled-head mapping was being handled incorrectly

This was partly true.

The runtime had to preserve actual `sampled_heads` mappings instead of assuming
uniform contiguous grouping. Fixing this materially improved results and made a
previously bad AIME sample return the correct answer.

But it was not sufficient by itself.

#### Hypothesis: the benchmark stats file was sparse sampled-head-only

This turned out to be wrong for the actual Qwen3-8B AIME25 calibration file.
Inspection showed that the file effectively carried full per-attention-head
stats for all `36 * 32 = 1152` attention heads.

That changed the interpretation of the runtime semantics:

- the selector should behave like grouped per-head reduction over dense
  attention-head stats
- sparse sampled-head fallback logic alone was not the real upstream match

#### Hypothesis: the remaining gap was purely in per-head compaction layout

This was plausible, but still not the first blocker.

The decisive remaining mismatch was the scorer formula.

### Phase 9: final root cause - native scorer used the wrong formulation

Upstream's current vLLM/Triton runtime does not score by unrotating keys using
cached token positions. It scores directly on stored `K_rot`, where key
position is already baked into the rotated key, and only the query-side phase
term remains.

Our native runtime was still using the older formulation:

- recover key position
- unrotate key
- reapply phase with cached positions

That was the wrong match for the upstream runtime being emulated.

The fix was to change both the host selector and the CUDA kernel to the direct
`K_rot` formulation:

- use stored rotated key components directly
- compute `Q_mean * conj(K_rot)` directly
- apply only the query-side trig term
- keep the additive MLR term on `|K_rot|`

After that fix, the previously bad hard samples flipped to the correct answers.

## Why the final fix worked

The direct-`K_rot` scorer mattered because it matched the runtime semantics of
the upstream implementation we were actually trying to reproduce.

It also removed two fragility points:

- dependence on exact cached position bookkeeping for scoring correctness
- extra numeric reconstruction work not present in the upstream runtime path

Once the native scorer matched upstream, the remaining native compaction path
was good enough to preserve quality on the corrected pilot.

## Evidence that the feature now works

### Native tests

- `ctest --test-dir build --output-on-failure -R test_triattention_kv_cache`

### Hard-sample recovery

Correct after scorer alignment:

- `tmp/qwen3_tri_v14_sample1_5000_gpu1.jsonl`
- `tmp/qwen3_tri_v14_sample2_5000_gpu1.jsonl`

### Corrected 6-sample pilot at shorter budget

- `tmp/qwen3_tri_v14_completed6.jsonl`

This run showed 6/6 correct with materially higher tok/s than dense.

### Corrected 6-sample pilot at matched long budget

Artifacts:

- aggregate partial:
  `tmp/qwen3_tri_v14_completed6_long.jsonl`
- per-sample completions:
  `tmp/qwen3_tri_v14_sample3_completed6_long_gpu1.jsonl`
  `tmp/qwen3_tri_v14_sample4_completed6_long_gpu1.jsonl`
  `tmp/qwen3_tri_v14_sample5_completed6_long_gpu2.jsonl`
  `tmp/qwen3_tri_v14_sample6_completed6_long_gpu3.jsonl`

Dense comparison:

- `tmp/qwen3_dense_completed6.jsonl`

Result:

- same extracted answers as dense on all 6 samples
- higher average throughput than dense

## Important non-obvious lessons

### 1. Accuracy debugging needed a matched evaluation recipe first

Until dense and HF agreed, TriAttention debugging was underdetermined. Fixing
the benchmark recipe first was mandatory.

### 2. Upstream drift matters

The upstream runtime semantics had moved to direct `K_rot` scoring. Matching an
older formulation would have kept the implementation "reasonable" but still
wrong.

### 3. Cache semantics and attention-work reduction are related but distinct

There were really two separate projects:

- make compaction correct
- make compaction useful for throughput

Both had to be solved independently.

### 4. Physical and logical cache sizes should not be conflated

That one design choice explained a large part of the early overflow and prompt
quality failures.

## Remaining limits

The current state is strong enough for native feature support, but there are
still follow-up opportunities:

- rerun a broader AIME25 sweep under the corrected recipe, not just the matched
  pilot slice
- revisit true KV-head-native cache layout for GQA models to eliminate the
  expanded query-head representation
- recover a more optimized long-window attention path where native TensorRT
  `IAttention` is unstable and the manual path is used instead
- consolidate the split pilot artifacts into one aggregate output file for
  easier future regression checking

## Files most directly responsible for the final result

- `src/runtime/core/triattention_kv_cache.cpp`
- `src/runtime/core/triattention_kernels.cu`
- `include/trtf/runtime/triattention_kv_cache.h`
- `tests/cpp/test_triattention_kv_cache.cpp`

## Short version

TriAttention only became accuracy-safe after the native scorer was aligned with
the upstream direct-`K_rot` semantics. Before that, the runtime was close in
structure but wrong in the core scoring math. Once that was fixed, the native
dynamic-KV C++ path achieved the intended state on the corrected pilot:

- dense-quality answers
- real native compaction
- real throughput gain
