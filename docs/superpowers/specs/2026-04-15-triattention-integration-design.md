# TriAttention Integration Design

## Summary

Upstream [TriAttention](https://github.com/WeianMao/triattention) is not a TensorRT attention-kernel replacement. Its main contribution is a KV-cache compression policy plus the runtime machinery needed to score, compact, and recycle cached tokens during long-context decoding.

For this repo, that means the natural integration point is the runtime/cache-management layer, not the TensorRT graph builder itself.

## Upstream Findings

The upstream repo has two practical integration surfaces:

- Hugging Face monkeypatching under `triattention/integration/`.
- vLLM runtime patches under `triattention/vllm/`.

The critical runtime pieces are:

- Calibration stats for sampled heads.
- A scoring function that combines RoPE-aware trig terms with an additive magnitude term.
- A compaction policy that decides which cached rows to keep.
- Runtime metadata and scheduler hooks so the serving stack knows how many rows remain live after compaction.

Important consequence: the upstream implementation assumes it can change runtime cache semantics. It does not just swap one attention op for another.

## Fit With `trt-transformers-cpp`

The current repo splits into:

- Build-time graph construction in `trtf_build/`.
- Native runtime/cache management in C++ under `src/runtime/`.
- A Python debug/runtime path in `trtf_build/trtf_build/debug_runner.py`.

Today there is no generic attention-backend abstraction in either the builder or the C++ runtime. The dense decoder cache is treated as a standard fixed-row KV buffer.

That makes a full native TriAttention port a larger project involving:

- New runtime-state metadata for compacted cache rows.
- New cache-compaction behavior in the C++ inference state.
- Decoder plugin updates so attention masks and positions remain correct after compaction.
- Validation for per-head or grouped-head retention, which upstream uses for stronger quality/perf tradeoffs.

## Phase 1 Implemented Here

This change adds an experimental, opt-in path that is good enough for evaluation and bundle-level enablement:

- Builder-side CLI flags to embed TriAttention calibration stats and runtime knobs.
- Bundle export that converts upstream `.pt` calibration artifacts into a portable JSON bundle section.
- Bundle config injection under `config.json["triattention"]`.
- A Python `TriAttentionTrtRunner` in the debug path that keeps the TensorRT engine unchanged and applies shared-token cache compaction in software.

This MVP deliberately keeps one shared retained-token set across all heads and layers. That is simpler than the upstream per-head compaction model, but it is compatible with the repo's current dense cache layout.

## What This Enables

You can now build a bundle with embedded TriAttention metadata and select the experimental runtime path from the bundle itself. This is enough to:

- Reproduce calibration-driven cache pruning experiments in the Python debug runner.
- Compare dense-cache behavior against an opt-in compression policy without changing the engine format.
- Preserve a clean path toward a later C++ runtime implementation.

## Remaining Work For A Native Port

The follow-up work for production-grade support is still substantial:

- Add a native compacting KV-cache implementation under `IInferenceState`.
- Decide whether the repo wants shared-token compaction or upstream-style per-head compaction.
- Expose runtime metrics for retained rows, reclaimed rows, and compaction frequency.
- Validate numerical quality and latency against the upstream implementation on representative models.
- Add serving/runtime controls in the C++ binary, not only the Python debug path.

## Recommendation

Treat the current integration as an evaluation path, not final product support.

If the debug-path experiments are promising, the next real milestone should be a native C++ cache-state implementation behind a runtime option such as `--kv-cache-policy=dense|triattention`.
