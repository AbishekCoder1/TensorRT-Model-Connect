# Apple-to-Apple Plain-Prompt Comparison

Bundle paths:

- dense full-KV: `artifacts/triattention/qwen3-8b-nonflash/qwen3-8b-dense32768-dynkv-fp16-manual-fullkv.trtfb`
- TriAttention: `artifacts/triattention/qwen3-8b-nonflash/qwen3-8b-tri12288-b3072-r128-dynkv-fp16-manual-current.trtfb`

Recipe:

- prompt style: upstream plain prompt, no chat template
- model: `Qwen/Qwen3-8B`
- decode: `temperature=0.6`, `top_k=20`, `top_p=0.95`, `min_p=0.0`, `seed=1234`

Fairness note:

- `aime25_2` prompt length: `690` tokens
- `aime25_3` prompt length: `164` tokens
- TriAttention compaction trigger: `kv_budget + divide_length = 3072 + 128 = 3200`
- Because both prompts are far below `3200`, the `tri12288` bundle is behaviorally equivalent to a `tri32768` bundle on these probes. The comparison below is therefore a valid full-KV-vs-TriAttention test for this operating point.

Results:

- dense full-KV `aime25_2` no-stop: answer `588`, `38.89729 tok/s`
- TriAttention `aime25_2` no-stop: answer `588`, `61.263541 tok/s`
- no-stop speedup on `aime25_2`: `1.5750x`
- dense full-KV `aime25_3` stop-on-answer: answer `16`, `53.068426 tok/s`
- TriAttention `aime25_3` stop-on-answer: answer `16`, `68.298592 tok/s`

Conclusion:

- On the upstream plain-prompt recipe, TriAttention preserves the checked dense full-KV answers on the hard `aime25_2` and `aime25_3` probes.
- On the fixed-length `aime25_2` no-stop comparison, TriAttention is `1.5750x` faster than dense full-KV.
