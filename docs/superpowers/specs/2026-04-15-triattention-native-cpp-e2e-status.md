# TriAttention Native C++ E2E Status

Date: 2026-04-15

## What is implemented

- Native `TriAttentionKvCache` in C++.
- Bundle parsing for `config.json["triattention"]` and `triattention_stats.json`.
- Decoder plugin wiring that selects TriAttention natively from the bundle.
- Prompt-boundary hooks in `IInferenceState` / `TextGenerationPipeline`.
- Delayed compaction fix: compact only when the physical cache is full, then prune to `kv_budget`.

## Native tests

Passed:

- `test_triattention_kv_cache`
- `test_text_generation_pipeline`
- `test_kv_cache_new`
- `test_pipeline_api`
- `test_bundle_e2e`

## Working native C++ E2E case

Model: `Qwen3-0.6B`

Bundles:

- `artifacts/triattention/qwen3-0.6b-dense128.trtfb`
- `artifacts/triattention/qwen3-0.6b-dense256.trtfb`
- `artifacts/triattention/qwen3-0.6b-dense192.trtfb`
- `artifacts/triattention/qwen3-0.6b-tri192-b128-r120-max.trtfb`

Prompt:

- `artifacts/triattention/marker_prompt_136tok.txt`

Observed native outputs:

- `dense128` -> `filler filler filler filler filler`
- `dense256` -> `OTTER581`
- `tri192-b128-r120-max` -> `OTTER581`

Interpretation:

- Native TriAttention in C++ can preserve the early marker on this prompt when given a physical slack window (`max_cache_length=192`, `kv_budget=128`).
- This matches the larger dense baseline's answer while using a smaller physical cache than `dense256`.

## Working native C++ chat-format case

Prompt:

- `artifacts/triattention/qwen_chat_marker_prompt.txt`

This prompt is formatted as a native Qwen multi-turn chat transcript with
`<|im_start|>` / `<|im_end|>` role markers rather than a plain text sentence.
The conversational content is:

- user gives an early launch code: `OTTER581`
- later filler turns add long-context clutter
- final user asks: `What is my launch code? Answer with only the code.`

Observed native outputs:

- `dense128` -> repeated garbage (`网站地图 ...`)
- `dense256` -> `OTTER581`
- `tri192-b128-r120-max` -> `OTTER581`

Interpretation:

- The earlier retrieval-style test is not limited to a bare string-matching
  prompt; the same behavior appears on a real chat-formatted conversation.
- The reason to use this style of test is that it provides a deterministic
  expected answer, which makes cache-policy regressions easy to detect.

## Working native C++ natural conversation case

Prompt:

- `artifacts/triattention/qwen_chat_reasonable_prompt_short.txt`

This prompt is a normal travel-planning conversation rather than repeated
filler text:

- user shares a reservation confirmation code: `OTTER581`
- later user discusses groceries, packing, check-in time, quiet hours, and
  weather contingency plans
- final user asks for the reservation code again

Observed native outputs:

- `dense128` -> garbage repetition (`toward toward toward ...`)
- `dense256` -> `OTTER581`
- `tri192-b128-r120-max` -> `OTTER581`

Interpretation:

- Native C++ TriAttention also matches the larger dense baseline on a
  reasonable chat input, not only on the earlier synthetic marker prompt.

## Cases that are not yet proven

Same-physical-window recovery when compaction activates during prompt prefill is not yet reliable.

Examples:

- `dense192` on `marker_prompt_196tok.txt` fails, as expected.
- `tri192-b128-*` on `marker_prompt_196tok.txt` still fails across the tested runtime knobs.
- `tri128-*` on `completion_prompt_139tok.txt` and `completion_prompt_179tok.txt` also fails.
- Longer chat overflow case `qwen_chat_marker_prompt_220f.txt` is also not
  solved yet: `dense192` fails, while `dense256` and `tri192-b128-r120-max`
  are not reliable enough to claim a benefit.
- Harder natural conversation `qwen_chat_reasonable_prompt.txt` is also not
  solved yet: `dense256` keeps the code, but `dense192` and
  `tri192-b128-r120-max` both fail.

## Performance status

Sequential native benchmark on `marker_prompt_136tok.txt`, `--max-new-tokens 80`, `--benchmark 2`, `--warmup 1`:

- `dense256`: `prefill_ms=193.77`, `decode_ms=132.43`, `tokens_per_sec=604.07`
- `tri192-b128-r120-max`: `prefill_ms=192.74`, `decode_ms=540.82`, `tokens_per_sec=147.92`

Interpretation:

- Before compaction activates, native TriAttention behaves correctly.
- Once compaction activates, the current host-side scoring / compaction path is too slow to claim the upstream throughput benefit.

## Current conclusion

Native C++ integration exists and is functional.

What is proven:

- bundle option works in native C++
- C++ runtime can load and execute TriAttention bundles
- delayed physical-full compaction is required
- there is a working E2E slack-window regime (`tri192-b128`) for quality

What is not yet proven:

- robust same-window prompt-overflow recovery
- throughput benefit once compaction is active
