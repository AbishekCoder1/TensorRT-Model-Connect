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

## Final apples-to-apples check

The final fair comparison used the upstream plain-prompt recipe and a true
full-KV dense baseline:

- dense full-KV bundle:
  `artifacts/triattention/qwen3-8b-nonflash/qwen3-8b-dense32768-dynkv-fp16-manual-fullkv.trtfb`
- TriAttention bundle:
  `artifacts/triattention/qwen3-8b-nonflash/qwen3-8b-tri12288-b3072-r128-dynkv-fp16-manual-current.trtfb`

Prompt recipe:

- plain upstream math prompt, no chat template
- `temperature=0.6`
- `top_k=20`
- `top_p=0.95`
- `min_p=0.0`
- `seed=1234`

Artifacts:

- copied probe inputs and outputs:
  `artifacts/triattention/apple-fullkv-plain/`
- compact summary:
  `artifacts/triattention/apple-fullkv-plain/summary.md`
  `artifacts/triattention/apple-fullkv-plain/summary.json`

Dense full-KV results:

- `aime25_2` no-stop:
  answer `588`, `38.89729 tok/s`
- `aime25_3` stop-on-answer:
  answer `16`, `53.068426 tok/s`

TriAttention results:

- `aime25_2` no-stop:
  answer `588`, `61.263541 tok/s`
- `aime25_3` stop-on-answer:
  answer `16`, `68.298592 tok/s`

Derived result:

- `aime25_2` no-stop speedup versus dense full-KV: `1.5750x`

### Why this is still fair even though the TriAttention bundle is `tri12288`

For these probes, the physical max length difference is inactive.

The prompt lengths are:

- `aime25_2`: `690` tokens
- `aime25_3`: `164` tokens

The runtime only begins compaction once
`cache_length_ >= compaction_trigger_length()`, and
`compaction_trigger_length()` is `kv_budget + divide_length = 3200` for this
bundle configuration; see
`src/runtime/core/triattention_kv_cache.cpp`.

Because both prompts are far below `3200`, the tested runs never depend on a
physical cache capacity larger than `12288`. After compaction starts, the live
cache stays near the logical TriAttention budget, again far below `12288`.
So on these probes a hypothetical `tri32768-b3072` bundle would follow the same
runtime path, and the comparison against `dense32768` is apples-to-apples for
the actual operating point being measured.

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

## 2026-04-17 Follow-up: full-benchmark regression and current root-cause status

After the earlier probe-level success, a full 30-sample AIME25 rerun exposed a
real remaining problem:

- HF eager:
  accuracy `0.700`, wall `15.96 tok/s`
- dense full-KV:
  accuracy `0.667`, wall `38.08 tok/s`
- TriAttention:
  accuracy `0.267`, wall `70.28 tok/s`

Recorded summary:

- `tmp/qwen3_8b_aime25_fullkv_plain_summary.md`
- `tmp/qwen3_8b_aime25_fullkv_plain_summary.json`

That result was too large a quality loss to wave away as sampling noise, so the
next debugging pass focused on whether the runtime selector/compactor was still
wrong on the bad long-window cases.

### Selector correctness is no longer the leading suspect

For `aime25_23`, native score-cache dumps were replayed through the local
upstream selector implementation in
`artifacts/triattention/upstream/triattention/vllm/runtime/selector_hf.py`.

For both compaction 1 and compaction 2:

- exact match on the selected rows for every sampled head
- mean Jaccard overlap `1.0`

This ruled out the remaining "wrong keep-set" theory for that sample.

### K-cache repack correctness is also no longer the leading suspect

For the same `aime25_23` investigation, the post-compaction K cache was checked
against the pre-compaction cache by directly gathering the kept indices. The
result was exact:

- `max_abs_diff = 0.0`
- no bad layers

So the native path is not silently scrambling K rows after selection on that
sample.

### One old long-window artifact was proven invalid as an apples-to-apples control

The older artifact
`artifacts/triattention/qwen3-8b-nonflash/qwen3-8b-tri32768-b3072-r128-dynkv-fp16-manual-apple.trtfb`
turned out not to be a valid same-family long-window baseline.

Even under conservative no-compaction settings and greedy decode, it diverged
from the dense full-KV bundle before compaction and also ran much faster than
dense in the pre-compaction region. That means it is not "the same engine path
plus TriAttention policy"; it is a materially different engine/runtime
combination. It must not be used for final parity claims.

This finding changed the next step of the investigation:

- stop trusting the stale `tri32768 ... manual-apple` artifact for full-benchmark
  conclusions
- rebuild a fresh same-family `dense32768` control from the current code
- rebuild the matching fresh `tri32768` bundle from the current code
- re-prove greedy no-compaction equivalence on the fresh pair before resuming
  full-benchmark tuning

At the time of writing this note, that fresh same-family rebuild is still in
progress and the full-benchmark parity question remains open.

### Same-bundle control proof: compaction is not the only remaining issue

After the stale-bundle baseline problem was identified, the next control was to
use the exact same `tri32768` bundle in two modes:

- `TRTF_TRIATTN_FORCE_ENABLE=0`
- `TRTF_TRIATTN_FORCE_ENABLE=1`

with compaction effectively disabled by runtime overrides:

- `TRTF_TRIATTN_OVERRIDE_KV_BUDGET=16384`
- `TRTF_TRIATTN_OVERRIDE_DIVIDE_LENGTH=2048`
- `TRTF_TRIATTN_OVERRIDE_RECENT_WINDOW=256`
- `TRTF_TRIATTN_RUNTIME_BUCKET_ROWS=32`

On `aime25_7`, greedy 3000-token generation from the same bundle was
byte-identical between the forced-off and forced-on modes:

- outputs identical
- extracted answer identical
- only a small throughput difference from extra TriAttention state allocation

This was an important control:

- when compaction is disabled, the native runtime no longer shows a separate
  "TriAttention corrupts output before compaction" problem
- the old pre-compaction divergence against dense was indeed caused by using a
  non-matched bundle pair

### But the same-bundle dense control still missed HF on long sampled decoding

The next focused test used that same `tri32768` bundle with
`TRTF_TRIATTN_FORCE_ENABLE=0` on the full sampled stop-on-answer recipe for the
known bad sample `aime25_2`.

Result:

- same-bundle forced-off control extracted `5`
- dense full-KV and HF reference both extracted `588`

That ruled out another attractive but incorrect explanation:

- the remaining full-benchmark gap is not only "bad compaction keep-sets"

Because `TRTF_TRIATTN_FORCE_ENABLE=0` takes the runtime back to the normal
`KvCache` state in `decoder_plugin.cpp`, this miss must come from the engine
family/build side, not from the TriAttention runtime state object.

### Dynamic-KV profile rows became the leading build-time suspect

The stale long-window bundles were then inspected directly. The dense full-KV
bundle and the `tri32768` bundle did not even share the same dynamic-KV
optimization profiles:

- dense full-KV:
  `[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]`
- `tri32768`:
  `[3072, 6144, 12288, 24576, 32768]`

Because the same-bundle force-off control already missed `aime25_2`, the
coarse TriAttention-specific profile schedule became the leading build-time
suspect for the remaining long-window drift. That led to the next experiment:

- add an explicit builder override for dynamic-KV profile rows
- rebuild a fresh `tri32768` bundle with the dense-style full profile ladder
- re-test sampled force-off control accuracy before returning to compaction
  tuning

### Dense-engine hybrid bundle proved the profile-row issue was real

Instead of waiting for another full engine rebuild, a direct control bundle was
constructed by taking the exact dense full-KV engine plan and tokenizer assets
from
`artifacts/triattention/qwen3-8b-nonflash/qwen3-8b-dense32768-dynkv-fp16-manual-fullkv.trtfb`
and grafting only the TriAttention metadata block plus
`triattention_stats.json` from the stale `tri32768` artifact.

The resulting hybrid bundle was:

- `artifacts/triattention/qwen3-8b-nonflash/qwen3-8b-tri32768-b3072-r128-dynkv-fp16-manual-denseengine-hybrid.trtfb`

This gave a same-family long-window comparison point with:

- the dense engine plan
- the dense dynamic-KV optimization profile ladder
- the current native TriAttention runtime path

That hybrid bundle immediately recovered the long sampled control behavior that
the stale `tri32768` bundle had lost.

On `aime25_2`, with `TRTF_TRIATTN_FORCE_ENABLE=0` and the full sampled
stop-on-answer recipe:

- hybrid force-off control extracted `588`
- generated `8592` tokens

So the dense-engine hybrid proved that the earlier long-window miss was not
intrinsic to the native runtime. It was tied to the stale engine-family build.

### Repaired-engine TriAttention then recovered the first hard cases

Using the same hybrid bundle with TriAttention enabled:

- default policy (`kv_budget=3072`, `divide_length=128`) also extracted `588`
  on `aime25_2`
- conservative policy
  (`TRTF_TRIATTN_OVERRIDE_KV_BUDGET=6144`,
  `TRTF_TRIATTN_OVERRIDE_DIVIDE_LENGTH=1024`,
  `TRTF_TRIATTN_RUNTIME_BUCKET_ROWS=32`) also extracted `588`

That meant the repaired engine family plus native TriAttention could preserve
the dense/HF answer on the previously bad sample `aime25_2`.

### Conservative policy fixed the three focused long-reasoning regressions

The next focused validation used the repaired hybrid bundle on the previously
bad slice `aime25_7`, `aime25_12`, and `aime25_23`.

With the repaired hybrid bundle and the conservative runtime policy:

- `aime25_7 -> 821`
- `aime25_12 -> 510`
- `aime25_23 -> 610`

Those are the same extracted answers as the dense/HF references for that
focused slice.

This is the first point in the long-window investigation where all three of the
known bad focus cases were simultaneously correct under the native runtime.

The exact conservative runtime recipe that achieved that state was:

- `TRTF_TRIATTN_FORCE_ENABLE=1`
- `TRTF_TRIATTN_OVERRIDE_KV_BUDGET=6144`
- `TRTF_TRIATTN_OVERRIDE_DIVIDE_LENGTH=1024`
- `TRTF_TRIATTN_RUNTIME_BUCKET_ROWS=32`

That repaired-engine conservative recipe is now the right candidate for the
next full 30-sample apples-to-apples benchmark:

- same hybrid bundle as dense control with `TRTF_TRIATTN_FORCE_ENABLE=0`
- same hybrid bundle as TriAttention run with the conservative policy above
- HF eager as the external reference

### A benchmark harness bug was then found in the seed schedule

The first full hybrid-bundle benchmark replay later showed a new regression on
`aime25_7`, but that run exposed a benchmark bug rather than a clean model
regression.

HF reference generation in
`tools/benchmark_qwen3_8b_aime25_vs_hf.py`
already reseeded per sample:

- `torch.manual_seed(seed + row_idx)` for the single-GPU case

But the TRT-side dataset benchmark binary was still reusing the exact same seed
for every row in a multi-sample run.

That meant the earlier "full benchmark" comparison was not even using the same
sampling schedule across HF and TRT. The benchmark runner was corrected so that
the TRT side now also advances the configured base seed by `sample_idx`.

### Seed fix changed the interpretation of the `aime25_7` regression

After rebuilding `trtf_dataset_benchmark` with the corrected per-row seed
schedule, the dense control was rechecked on `aime25_7` as a standalone sample:

- seed `1234` still produced `821`
- seed `1240` also produced `821`

So the seed fix did not break the previous single-sample proof.

Then the crucial sequence-dependent focused replay was repeated with the rebuilt
benchmark binary:

- dataset: `aime25_2`, then `aime25_7`
- dense control: same hybrid bundle with `TRTF_TRIATTN_FORCE_ENABLE=0`

Result:

- `aime25_2 -> 588`
- `aime25_7 -> 821`

And that same focused replay stayed correct even with CUDA Graphs disabled:

- `TRTF_DISABLE_CUDA_GRAPH=1`
- still `aime25_2 -> 588`
- still `aime25_7 -> 821`

This invalidated the earlier "CUDA Graph reuse is the remaining root cause"
hypothesis. Once the benchmark seed schedule was corrected, the focused dense
control reproduced the right answer path with and without CUDA Graphs.

### Conservative TriAttention also passed the corrected focused replay

The same corrected focused replay (`aime25_2`, then `aime25_7`) was then run on
the native TriAttention path using the conservative runtime policy:

- `TRTF_TRIATTN_FORCE_ENABLE=1`
- `TRTF_TRIATTN_OVERRIDE_KV_BUDGET=6144`
- `TRTF_TRIATTN_OVERRIDE_DIVIDE_LENGTH=1024`
- `TRTF_TRIATTN_RUNTIME_BUCKET_ROWS=32`

Result:

- `aime25_2 -> 588`
- `aime25_7 -> 821`

At that point, both dense and conservative TriAttention were again aligned on
the key focused pair under the corrected benchmark runner.

The next required step from that state is a fresh full 30-sample apples-to-
apples rerun using:

- the corrected TRT benchmark binary with per-row seed advancement
- dense control = hybrid bundle with `TRTF_TRIATTN_FORCE_ENABLE=0`
- TriAttention = same hybrid bundle with the conservative runtime policy
- HF eager as the external reference

### Prompt mismatch later invalidated part of the focused `aime25_7` story

While chasing the remaining full-benchmark miss on `aime25_7`, I discovered
that the old standalone sample file used for several earlier spot checks did
not contain the same prompt string as the current benchmark dataset row.

The old standalone prompt began with:

- `You are given a math problem.`

The current benchmark prompt begins directly with the problem statement and the
final-answer instruction:

- `The twelve letters ...`
- `Please reason step by step, and put your final answer within \boxed{}`

That means several earlier "standalone sample 7" checks were not strict
apples-to-apples comparisons against the current benchmark harness. The dense
and TriAttention results on those old prompt files are still useful as local
signals, but they cannot be treated as proof for the current benchmark recipe.

### Current-prompt standalone checks changed the interpretation again

After extracting `aime25_7` directly from the current benchmark prompt file, I
reran the same-family dense control and conservative TriAttention as strict
single-sample standalone runs.

Dense control, current prompt:

- seed `1234` -> `pred_answer = 1`
- seed `1240` -> `pred_answer = 41`

Artifacts:

- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_force0_sample7_currentprompt_seed1234_gpu2.jsonl`
- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_force0_sample7_currentprompt_seed1240_gpu1.jsonl`

This was an important correction. The dense current-prompt path is already
wrong on `aime25_7` for these seeds as a standalone sample, so the later
`aime25_6 -> aime25_7` replay that produced `41` is no longer evidence of a
sequence-state corruption by itself.

### Conservative TriAttention still diverges further on the same prompt

Using the exact same current benchmark prompt row and the same hybrid bundle,
the conservative TriAttention path produced:

- seed `1234` -> `pred_answer = 68`

Artifact:

- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_tri_sample7_currentprompt_seed1234_gpu2.jsonl`

So the current sharp parity statement is:

- dense current-prompt standalone at seed `1234` gives `1`
- TriAttention current-prompt standalone at the same seed gives `68`
- gold answer remains `821`

That means the remaining problem is no longer "TriAttention loses parity while
dense stays correct" on this case. On the current benchmark recipe, both paths
are already off the gold answer on `aime25_7`, and TriAttention departs even
further from the dense answer on the exact same prompt and seed.

The unfinished seed-`1240` TriAttention run only reached pipeline load and did
not produce a sample row, so it is not evidence either way:

- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_tri_sample7_currentprompt_seed1240_gpu3.log`
- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_tri_sample7_currentprompt_seed1240_gpu3.jsonl`

### Current working interpretation

At this point the investigation has two distinct questions:

- why the current-prompt dense long-window path misses `aime25_7` under these
  seeds
- why conservative TriAttention departs from that dense path on the exact same
  prompt and seed

The next valid parity step is therefore not another broad benchmark rerun. It
is a same-prompt, same-seed runtime diff between dense and TriAttention on the
current benchmark sample until the first real divergence is localized.

### Same-prompt boundary runs localized the divergence to the first compaction

Using the current benchmark prompt row for `aime25_7` and the same hybrid
bundle:

- dense force-off, seed `1234`, `max_new_tokens=7000`
- conservative TriAttention, same seed, `max_new_tokens=7000`

produced byte-identical text:

- same `pred_answer = 10`
- same `generated_tokens = 7000`
- exact string equality on the generated text

Artifacts:

- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_force0_sample7_currentprompt_seed1234_gpu1_max7000.jsonl`
- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_tri_sample7_currentprompt_seed1234_gpu2_max7000.jsonl`

This ruled out any "prefill mismatch" or "early decode mismatch" theory for
this sample. The dense and TriAttention paths are identical through 7000
generated tokens.

Then a first-compaction abort run with keep-dump enabled showed the exact first
compaction point:

- `planned_prompt_length = 158`
- `prompt_end_position = 158`
- first compaction at `absolute_position = 7168`
- `keep_count = 6144`

So the first compaction happens after exactly:

- `7168 - 158 = 7010` generated tokens

Artifacts:

- `artifacts/triattention/investigation-2026-04-17/sample7_currentprompt_seed1234_firstcomp`
- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_tri_sample7_currentprompt_seed1234_firstcomp_abort.log`

That means the `7000`-token equality checkpoint is only 10 generated tokens
before the first compaction.

### Divergence appears immediately after the first compaction window

At `max_new_tokens=7600`, dense and TriAttention already diverge on the same
prompt and seed:

- dense `pred_answer = 10`
- TriAttention `pred_answer = 10`
- but text is no longer equal
- first textual difference is still the same `char=24807` boundary observed in
  the full outputs

Since the exact `7000`-token text length is `24690`, the first difference
appears only:

- `24807 - 24690 = 117` characters

after the last pre-compaction-equal checkpoint.

Artifacts:

- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_force0_sample7_currentprompt_seed1234_gpu2_max7600_rerun.jsonl`
- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_tri_sample7_currentprompt_seed1234_gpu2_max7600.jsonl`

This sharply localizes the problem: the remaining divergence begins almost
immediately after the first compaction, not thousands of tokens later.

### GPU-specific selection/compaction is no longer a viable root cause

The same `7600`-token current-prompt run was repeated with the entire GPU
TriAttention fast path disabled:

- `TRTF_TRIATTN_DISABLE_GPU_SELECT=1`
- `TRTF_TRIATTN_DISABLE_GPU_COMPACT=1`
- `TRTF_TRIATTN_DISABLE_GPU_STATE=1`

Result:

- host-only TriAttention and normal GPU TriAttention were byte-identical
  at `7600` tokens

Artifacts:

- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_tri_sample7_currentprompt_seed1234_hostonly_max7600.jsonl`
- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_tri_sample7_currentprompt_seed1234_gpu2_max7600.jsonl`

So the remaining issue is not in:

- the CUDA selector kernel
- the CUDA compaction kernel
- GPU-only state bookkeeping

It is shared host/GPU TriAttention logic above that layer.

### The first-compaction keep-set disagrees with the upstream selector on this sample

The exact first-compaction K-cache snapshot for `aime25_7` was replayed through
the local upstream per-head selector implementation in:

- `artifacts/triattention/upstream/triattention/vllm/runtime/selector_hf.py`

using both:

- `artifacts/triattention/qwen3_8b_aime25.pt`
- `artifacts/triattention/upstream/triattention/calibration/for_aime25_experiment/qwen3_8b.pt`

Both produced the same result:

- exact head matches: `0 / 8`
- mean Jaccard overlap: `0.778429853926756`

The protected prefix and recent tail still agree, but the selected interior
rows differ materially.

This is the first decisive proof that, on the current sample and compaction
state, the native selector semantics are still not matching the upstream
selector semantics.

So the working diagnosis is now:

- the first real divergence happens in the first compaction window
- host/GPU native paths agree with each other
- native keep-set does **not** agree with upstream keep-set on this sample

The next debugging step from here is score-level comparison:

- dump native per-layer / aggregated score values for the first compaction
- compare them directly against the upstream selector scores on the same
  snapshot
- identify whether the mismatch is in raw scoring, per-head grouping, or
  cross-layer aggregation

### Later correction: selector and compaction are now cleared on the aggressive sample7 path

The previous diagnosis above was too pessimistic. After replaying the same
`aime25_7` first-compaction snapshot more carefully, the remaining bug is no
longer in the selector itself.

First, the attempted "reduce stats to runtime KV heads" patch was the wrong
direction and was reverted. On the exact `sample7` first-compaction snapshot,
the original native `32`-score-head semantics match the official upstream
selector exactly:

- simulated original native vs upstream official selector: exact head matches
  `8 / 8`
- mean Jaccard overlap: `1.0`

That proof used the real dumped pre-compaction cache from:

- `artifacts/triattention/investigation-2026-04-17/sample7_currentprompt_seed1234_debugprobe_revert4.json.layer00.bin`
  through
- `artifacts/triattention/investigation-2026-04-17/sample7_currentprompt_seed1234_debugprobe_revert4.json.layer35.bin`

and the real runtime keep dump:

- `artifacts/triattention/investigation-2026-04-17/sample7_currentprompt_seed1234_debugprobe_revert4.json`

Second, the real C++ runtime keep-set is also exact with respect to the native
math on that same dump:

- runtime keep-set vs direct native-math replay: exact head matches `8 / 8`
- mean Jaccard overlap: `1.0`

So by this point the aggressive `sample7` first compaction has:

- official upstream selector parity
- native runtime selector parity

Third, the post-compaction cache contents are also exact gathers of the
selected rows.

Using the first-compaction dump with explicit pre/post K/V cache snapshots:

- `artifacts/triattention/investigation-2026-04-17/sample7_currentprompt_seed1234_debugprobe_revert5.json`

all `36` layers passed exact gather checks for both:

- `K`
- `V`

That means:

- selected indices are correct
- `K` repack is correct
- `V` repack is correct

Fourth, the boundary localization still holds on the aggressive operating
point:

- dense force-off and TriAttention are byte-identical through
  `max_new_tokens=3000`
- at `max_new_tokens=3400`, they diverge
- first text difference is still at `char=11217`

Artifacts:

- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_force0_sample7_currentprompt_seed1234_gpu1_max3000_revert.jsonl`
- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_tri_sample7_currentprompt_seed1234_gpu2_max3000_revert.jsonl`
- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_force0_sample7_currentprompt_seed1234_gpu1_max3400_revert.jsonl`
- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_tri_sample7_currentprompt_seed1234_gpu2_max3400_revert.jsonl`

Finally, the current aggressive-path host-only replay still matches the normal
GPU TriAttention replay exactly at `3400` tokens:

- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_tri_sample7_currentprompt_seed1234_hostonly_max3400_revert.jsonl`
- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_tri_sample7_currentprompt_seed1234_gpu2_max3400_revert.jsonl`

So the updated diagnosis is:

- the remaining `sample7` divergence still begins in the first compaction
  window
- but it is **not** caused by selector mismatch
- and it is **not** caused by incorrect K/V gather
- and it is **not** a GPU-only kernel bug

At this point the remaining explanations are narrower:

- genuine quality loss from the aggressive compression operating point itself
- or a downstream runtime effect outside selector/repack that still has not
  been identified

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
