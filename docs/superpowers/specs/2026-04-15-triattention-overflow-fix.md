# TriAttention Prompt-Overflow Fix

## Question

Why does native C++ TriAttention fail on the longer prompt-overflow cases such as:

- `artifacts/triattention/marker_prompt_196tok.txt`
- `artifacts/triattention/qwen_chat_reasonable_prompt.txt`

And can we fix it without giving up KV-cache compression?

## What the failure actually is

The failure is not primarily a GPU-kernel bug.

On `tri192-b128-r120-max`, the runtime must compact during prompt prefill once
the physical cache reaches `192` rows. Even after the native runtime preserves
as much prompt slack as possible, the remaining prompt tokens are still computed
against a compressed history instead of the full dense prompt.

That is enough to corrupt the final prompt-state on the tested Qwen3-0.6B
prompt-overflow cases.

Concrete evidence:

- `artifacts/triattention/run_tri192_b128_r120_max_dynkv_fp16_manual_marker196.out`
  does not exist because `tri192-b128` still fails on that prompt after the
  runtime-side slack fix.
- Debug traces showed the first prompt-time compaction for
  `marker_prompt_196tok.txt` already happened at `abs_pos=192`.
- For `qwen_chat_reasonable_prompt.txt`, prompt processing also overflowed the
  `192`-row physical cache before the answer token was sampled.

## Runtime improvement that still landed

The native runtime now has a better prefill-overflow policy in
`src/runtime/core/triattention_kv_cache.cpp`:

- `compaction_keep_budget()` keeps the largest old-prefix that still allows the
  remainder of the prompt to fit physically.
- This is covered by `test_prefill_overflow_uses_physical_slack()` in
  `tests/cpp/test_triattention_kv_cache.cpp`.

That improves the correctness of near-threshold prompt overflow, but it does
not fully solve the real failing prompts above.

## What actually fixes the issue

Use a larger physical cache window than the logical TriAttention KV budget.

In practice:

- Physical cache: `256`
- Logical TriAttention budget: `128`

This keeps prompt prefill dense for prompts that fit within `256` rows, then
lets native C++ TriAttention compact later during decode.

That bundle was validated with:

- `artifacts/triattention/qwen3-0.6b-tri256-b128-r120-max-dynkv-fp16-manual.trtfb`

This artifact reuses the `dense256` engine and adds the TriAttention runtime
metadata and calibration stats. That is valid in this repo because TriAttention
is a runtime/cache policy, not a different TRT attention engine.

## Proof

Marker overflow prompt:

- Dense baseline: [run_dense256_marker196.out](../../../artifacts/triattention/run_dense256_marker196.out)
  returns `OTTER581`
- Fixed TriAttention bundle:
  [run_tri256_b128_r120_max_dynkv_fp16_manual_marker196.out](../../../artifacts/triattention/run_tri256_b128_r120_max_dynkv_fp16_manual_marker196.out)
  also returns `OTTER581`

Natural chat overflow prompt:

- Dense baseline:
  [run_dense256_qwen_chat_reasonable.out](../../../artifacts/triattention/run_dense256_qwen_chat_reasonable.out)
  returns `OTTER581`
- Fixed TriAttention bundle:
  [run_tri256_b128_r120_max_dynkv_fp16_manual_qwen_chat_reasonable.out](../../../artifacts/triattention/run_tri256_b128_r120_max_dynkv_fp16_manual_qwen_chat_reasonable.out)
  also returns `OTTER581`

And the debug log confirms compaction now happens only after the prompt fits and
decode crosses the physical `256`-row limit:

- [run_tri256_b128_r120_max_dynkv_fp16_manual_qwen_chat_reasonable.err](../../../artifacts/triattention/run_tri256_b128_r120_max_dynkv_fp16_manual_qwen_chat_reasonable.err)

That log shows:

- no prompt-time compaction
- first compaction at `abs_pos=256`
- correct output still produced

## Conclusion

The "long-context overflow issue" is a prompt-time physical-window problem.

For this repo, the robust fix is:

- keep TriAttention's logical KV budget small
- keep the physical TRT cache window large enough to absorb the full prompt
- start compressing only when decode actually grows beyond that physical window

Trying to make `tri192-b128` recover the tested prompt-overflow cases inside the
same `192` physical rows was not reliable enough, even after runtime-side
policy improvements.
