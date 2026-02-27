# Mutation Testing & Coverage Audit Plan

Red-team audit of the test suite: deliberately introduce small bugs, verify tests catch them.
Where tests pass despite the bug, we've found a coverage hole.

---

## Executive Summary

Across the C++ runtime, Python builder, and E2E harness, we identified **87 concrete mutation
candidates** and **32 E2E coverage gaps**. The highest-risk findings:

| Category | Mutations Found | Tests Would Catch | Tests Would MISS |
|----------|----------------|-------------------|-----------------|
| C++ runtime | 17 | 7 (41%) | **10 (59%)** |
| Python builder | 47 | 12 (26%) | **35 (74%)** |
| E2E harness | — | — | **32 structural gaps** |

**Key finding:** Only ~30% of high-risk mutations would be caught by existing unit tests.
E2E tests catch more, but are slow and expensive. The test suite has strong coverage for
bundle format and tokenizers, but weak coverage for the generate loop, weight transforms,
graph operations, and the E2E comparison logic itself.

---

## Part 1: C++ Runtime Mutations

### Category A: Well-Tested (tests WOULD catch — confidence HIGH)

| # | File | Line | Mutation | Why Caught |
|---|------|------|----------|-----------|
| A1 | device_kv_cache.cpp | 21 | Remove `-1` from position_limit | test_position_clamping checks exact values |
| A2 | trt_decode_runtime.cpp | 124 | Invert `include_current_slot` | test_mask_with_current_slot verifies mask.back() |
| A3 | bundle_format.cpp | 23-26 | Swap LE to BE byte order | test_read_valid_bundle checks round-trip |
| A4 | bundle_format.cpp | 143 | Change `+` to `-` in data_start | test_read_valid_bundle checks section data |
| A5 | json_helpers.cpp | 113 | Invert isdigit check | Bundle tests parse integers (vocab_size, etc.) |

### Category B: Under-Tested (tests MIGHT miss — confidence MEDIUM)

| # | File | Line | Mutation | Gap |
|---|------|------|----------|-----|
| B1 | device_kv_cache.cpp | 90 | `+1` → `+0` in cache_length | Tests check position_id output but not intermediate cache_length |
| B2 | trt_decode_runtime.cpp | 118 | Remove `std::max(0,...)` guard | No test with negative cache_length |
| B3 | json_helpers.cpp | 36 | Remove `-1` in substr length | Caught indirectly by bundle tests, no dedicated JSON unit test |
| B4 | text_parsers.cpp | 14 | `>=` → `>` in starts_with | Edge case: exact-length prefix match untested |
| B5 | vocab_tokenizer.cpp | 24 | Deref end() on missing `<unk>` | Tests assume `<unk>` always exists |
| B6 | image_preprocessor.cpp | 381 | Swap dst_h/dst_w (transpose patches) | Unit tests basic; E2E VL would catch |
| B7 | image_preprocessor.cpp | 74 | Invert near-zero std guard | No near-zero std test case |

### Category C: UNTESTED (tests would NOT catch — confidence LOW)

| # | File | Line | Mutation | Why Missed |
|---|------|------|----------|-----------|
| C1 | **device_kv_cache.cpp** | 72-76 | `mMaxCacheLength-1` → `mMaxCacheLength` in shift memcpy | **Tests don't verify cached KV data contents — only position_id and mask** |
| C2 | **trt_backend_shared.cpp** | 50 | `i+1 < input_ids.size()` → `i < input_ids.size()` (prefill off-by-one) | **No C++ unit test for generate() — only E2E** |
| C3 | **trt_backend_shared.cpp** | 74 | `==` → `!=` on EOS check | **No C++ unit test for EOS termination** |
| C4 | **trt_backend_shared.cpp** | 36 | `==` → `!=` on max_new_tokens check | **No C++ unit test for early return** |
| C5 | **fast_path_config.cpp** | 52 | `hidden_size / num_heads` → `hidden_size * num_heads` | **No C++ unit test for head_dim default calculation** |

**These 5 mutations are the most dangerous.** They would break inference completely but
have no C++ unit test coverage. Only E2E tests (which require GPU + engine) would catch them.

---

## Part 2: Python Builder Mutations

### Category A: Critical — No Test Coverage

| # | File | Function | Mutation | Impact |
|---|------|----------|----------|--------|
| P1 | **checkpoint_mapper.py** | `_repeat_head_norm` | `np.tile(norm, num_heads)` → `np.tile(norm, num_heads-1)` | Last head Q/K norm wrong. **No test exists for this function.** |
| P2 | **checkpoint_mapper.py** | bfloat16 numpy path | `<< 16` → `<< 8` (wrong bit shift) | **Silently corrupts ALL bfloat16 weights.** No test for numpy bf16 path. |
| P3 | **checkpoint_mapper.py** | bias expansion | Off-by-one in `qh // group_size` | **No test for bias GQA expansion at all.** |
| P4 | **checkpoint_mapper.py** | tied embeddings | Remove `.copy()` on embedding | Shared reference — mutation during inference corrupts embedding. **No independence test.** |
| P5 | **graph_ops.py** | `add_rms_norm` | `1 << 1` → `1 << 2` (wrong reduce axis) | RMSNorm over wrong dimension. Tests check vs reference but axis not validated in isolation. |
| P6 | **graph_ops.py** | rotation matrix | Swap signs in `matrix[d_odd, d_even]` | Rotation 90° off. **Test checks shape/diagonal but not per-element signs.** |
| P7 | **graph_ops.py** | `add_self_attention_block` softmax | `1 << 2` → `1 << 1` (wrong axis) | Softmax over wrong dimension. **No axis-specific unit test.** |
| P8 | **graph_ops.py** | odd-length rotary tail | `matrix[tail,tail]=1.0` → `0.0` | Odd dim zeroed. **Tests only use even head_dim.** |
| P9 | **graph_blocks.py** | `apply_norm` dispatch | Swap rmsnorm/layernorm conditional | Wrong norm type applied. **No dispatch test.** |
| P10 | **graph_blocks.py** | cache concatenation | Swap `[cache_k, k]` to `[k, cache_k]` | Cache order reversed. **Test checks shape not order.** |
| P11 | **graph_blocks.py** | concatenation axis | `axis=0` → `axis=1` | Concatenation over wrong dim. **No axis test.** |
| P12 | **standard_decoder_builder.py** | tensor naming | `"token_id"` → `"token_ids"` | C++ runtime can't find input. **No naming contract test.** |
| P13 | **debug_runner.py** | mask building | Invert `mask[:] = -1e9` vs `0.0` pattern | Attends to future, masks past. Caught by cache_state_machine test. |
| P14 | **debug_runner.py** | argmax | `np.argmax` → `np.argmin` | Selects worst token. **test_generate doesn't validate token selection.** |
| P15 | **families/qwen.py** | `matches()` | Remove "moe" exclusion check | MoE Qwen built with wrong plugin. **No plugin precedence test.** |
| P16 | **families/gemma.py** | weight transform | Remove `gamma + 1.0` | Gemma RMSNorm wrong. **No transform test for Gemma.** |

### Category B: Partially Tested — Edge Cases Missing

| # | File | Mutation | What's Tested | What's Missing |
|---|------|----------|---------------|----------------|
| P17 | config.py | Wrong fallback key order for hidden_size | Qwen3/LLaMA configs tested | GPT-2, Bloom, OPT non-standard keys untested |
| P18 | config.py | `float(rope_theta)` → `int(rope_theta)` | rope_theta=1000000.0 (passes as int) | Fractional rope_theta values never tested |
| P19 | config.py | Remove `or -1` on bos_token_id | bos=-1 tested | bos=0 (valid token ID) treated as falsy, becomes -1 |
| P20 | checkpoint_mapper.py | Wrong dimension in `q_hidden = q_raw.shape[0]` → `shape[1]` | Shape checked but not 0-vs-1 | Would swap hidden_size and num_features |
| P21 | graph_ops.py | YaRN `max(high-low, 1)` → `max(high-low, 0)` | Normal range tested | Boundary case high==low → div-by-zero untested |
| P22 | bundle_writer.py | `"<Q"` → `">Q"` (endian flip) | Round-trip on same machine | Cross-endian never tested |
| P23 | debug_runner.py | `cache_length += 1` (no clamping) | Clamping tested | Unbounded growth scenario missing |

---

## Part 3: E2E Harness Coverage Gaps

### Comparator Gaps (bugs that slip through E2E tests)

| # | Comparator | Gap | Bug That Slips Through |
|---|-----------|-----|----------------------|
| E1 | text.py | p5/p95 percentile allows 5% failure | 6% logit degradation on 90% of steps passes |
| E2 | text.py | No check for constant logits | KV cache corruption (all logits=same value) passes |
| E3 | text.py | NED skipped when token agreement is perfect | Whitespace/punctuation formatting bug passes |
| E4 | vision_language.py | Shape mismatch silently truncated to overlap | Vision stride bug produces fewer patches, comparator only checks subset |
| E5 | vision_language.py | `vision_embedding_cosine` threshold=0.5 | Vision encoder producing near-random features could still pass |
| E6 | text_to_audio.py | RMS bounds [0.001, 1.0] extremely loose | 50% amplitude regression passes |
| E7 | text_to_audio.py | No perceptual metric (PESQ/STOI) | Numerically close but perceptually poor audio passes |
| E8 | diffusion.py | Pixel distribution doesn't verify diversity | Same frame repeated could pass if internal std >= 0.05 |
| E9 | diffusion.py | PSNR threshold=20 very loose | Degraded output passes |
| E10 | segmentation.py | Shape mismatch uses NN resize silently | Resolution regression hidden by upsampling |

### Runner Gaps

| # | Runner | Gap | Bug That Slips Through |
|---|--------|-----|----------------------|
| E11 | text_generation.py | Logits .npy file not validated before return | Corrupted file passes runner, fails in comparator |
| E12 | text_generation.py | Debug runner exit code != 0 still returns metadata | OOM-incomplete output treated as valid |
| E13 | vision_language.py | diff_vl.py output parsing fragile | Format change breaks metric extraction silently |
| E14 | All runners | No NaN/Inf check on logits | Silent numerical instability not caught |

### Orchestrator Gaps

| # | Gap | Bug That Slips Through |
|---|-----|----------------------|
| E15 | Non-required stage failure is silent | Optional stage regression goes unnoticed |
| E16 | Determinism reruns not implemented | Non-deterministic inference not detected |
| E17 | Bundle build timeout=3600s with no progress | Builder infinite loop not diagnosed for 1 hour |

### Threshold Gaps

| # | Strategy | Gap | Bug That Slips Through |
|---|----------|-----|----------------------|
| E18 | text_generation_causal | token_agreement_rate=0.8 (20% mismatch allowed) | Significant token-level regression passes |
| E19 | vision_language | vision_embedding_cosine=0.5 (allows 50% angular distance) | Near-complete vision encoder failure passes |
| E20 | All | Thresholds not adaptive to model size | Large model regression hidden by small model thresholds |

---

## Part 4: Concrete Mutation Testing Protocol

### Phase 1: Automated Mutation Runs (Python)

Use `mutmut` for systematic Python mutation testing:

```bash
# Install mutmut
pip install mutmut

# Run on highest-risk files
mutmut run --paths-to-mutate=trtf_build/trtf_build/checkpoint_mapper.py \
  --tests-dir=tests/builder/ --runner="pytest tests/builder/test_checkpoint_mapper.py -x"

mutmut run --paths-to-mutate=trtf_build/trtf_build/graph_ops.py \
  --tests-dir=tests/builder/ --runner="pytest tests/builder/test_graph_ops.py -x"

mutmut run --paths-to-mutate=trtf_build/trtf_build/debug_runner.py \
  --tests-dir=tests/builder/ --runner="pytest tests/builder/test_cache_state_machine.py -x"

mutmut run --paths-to-mutate=trtf_build/trtf_build/bundle_writer.py \
  --tests-dir=tests/builder/ --runner="pytest tests/builder/test_bundle_writer.py -x"

# Check results
mutmut results
mutmut show <id>  # Show surviving mutants (test gaps)
```

**Target: <5% surviving mutants on critical files.**

### Phase 2: Manual C++ Mutation Injection (worktree-isolated)

For each C++ mutation in Category C (untested), create an isolated worktree,
apply the mutation, rebuild, run tests, verify failure:

```bash
# Example: Mutation C1 — KV cache shift size off-by-one
git worktree add /tmp/mutation-c1 -b mutation/c1-cache-shift HEAD

cd /tmp/mutation-c1
# Apply mutation to device_kv_cache.cpp line 72-76:
# Change (mMaxCacheLength - 1) to (mMaxCacheLength)
cmake -S . -B build -G Ninja && cmake --build build -j

# Run C++ unit tests — expect them to PASS (this is the problem!)
ctest --test-dir build -R test_device_kv_cache --output-on-failure
# If tests pass → COVERAGE HOLE CONFIRMED

# Run E2E smoke test — should catch it
pytest tests/test_e2e.py::test_e2e[qwen3-0.6b] -v \
  --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python

# Clean up
cd - && git worktree remove /tmp/mutation-c1
```

**Run for each mutation C1-C5. Document which tests catch it and which miss.**

### Phase 3: Targeted Test Additions

For every surviving mutant (test gap), write a targeted test:

#### C++ Tests to Add

| Test File | Test Function | Mutation It Catches |
|-----------|--------------|-------------------|
| test_device_kv_cache.cpp | `test_cache_shift_data_integrity` | C1: Verify actual KV data after shift |
| test_device_kv_cache.cpp | `test_cache_length_increment` | B1: Assert cache_length == expected after each step |
| test_trt_backend_mock.cpp | `test_prefill_loop_bounds` | C2: Mock engine, verify prefill processes N-1 tokens |
| test_trt_backend_mock.cpp | `test_eos_terminates` | C3: Mock engine returning EOS, verify loop stops |
| test_trt_backend_mock.cpp | `test_max_new_tokens_zero` | C4: max_new_tokens=0 returns input unchanged |
| test_fast_path_config.cpp | `test_head_dim_default_calculation` | C5: Verify head_dim = hidden/heads, not hidden*heads |
| test_json_helpers.cpp | `test_extract_json_string_no_trailing_quote` | B3: Verify extracted string has no trailing quote |
| test_text_parsers.cpp | `test_starts_with_exact_match` | B4: `starts_with("abc", "abc")` returns true |

#### Python Tests to Add

| Test File | Test Function | Mutation It Catches |
|-----------|--------------|-------------------|
| test_checkpoint_mapper.py | `test_repeat_head_norm` | P1: Verify tile produces correct num_heads copies |
| test_checkpoint_mapper.py | `test_bfloat16_numpy_bit_shift` | P2: Compare bf16→fp32 vs known reference |
| test_checkpoint_mapper.py | `test_bias_gqa_expansion` | P3: Verify bias expansion at GQA boundary |
| test_checkpoint_mapper.py | `test_tied_embeddings_independence` | P4: Modify w_out, verify embedding unchanged |
| test_graph_ops.py | `test_rms_norm_reduce_axis` | P5: Verify reduction is over feature dim, not batch |
| test_graph_ops.py | `test_rotation_matrix_signs` | P6: Verify exact sign pattern in rotation matrix |
| test_graph_ops.py | `test_attention_softmax_axis` | P7: Verify softmax axis is last dimension |
| test_graph_ops.py | `test_odd_head_dim_rope_tail` | P8: Use odd head_dim, verify tail is identity |
| test_graph_blocks.py | `test_norm_dispatch_rmsnorm_vs_layernorm` | P9: Both paths produce different outputs |
| test_graph_blocks.py | `test_cache_concatenation_order` | P10: Verify new tokens appended after cache |
| test_standard_decoder.py | `test_tensor_naming_contract` | P12: Verify input/output names match C++ expectations |
| test_families.py | `test_plugin_exclusion_checks` | P15: Verify MoE Qwen doesn't match qwen plugin |
| test_family_plugins.py | `test_gemma_gamma_plus_one` | P16: Verify Gemma transform applies +1.0 |

#### E2E Harness Tests to Add

| Test File | Test Function | Gap It Closes |
|-----------|--------------|--------------|
| test_text_comparator.py | `test_constant_logits_rejected` | E2: All-same logits should fail |
| test_text_comparator.py | `test_nan_logits_rejected` | E14: NaN/Inf logits should fail |
| test_vl_comparator.py | `test_shape_mismatch_detected` | E4: Different shapes should warn, not silently truncate |
| test_comparator_thresholds.py | `test_loose_threshold_sanity` | E18-E20: Verify thresholds reject known-bad inputs |

---

## Part 5: Priority Execution Order

### Tier 1: Immediate (blocks correctness confidence)

1. **Write C++ tests for Category C mutations** (C1-C5) — these are the highest-risk
   untested code paths in the runtime
2. **Write Python tests for P1-P2** (repeat_head_norm, bfloat16 bit shift) — silent
   weight corruption with zero test coverage
3. **Add constant/NaN logit check** to text comparator (E2, E14)

### Tier 2: High Priority (1-2 days)

4. Run `mutmut` on `checkpoint_mapper.py`, `graph_ops.py`, `debug_runner.py`
5. Write tests for all surviving mutants from mutmut run
6. Add tensor naming contract test (P12) — prevents C++/Python contract drift
7. Add plugin exclusion tests (P15) — prevents model family misrouting

### Tier 3: Medium Priority (1 week)

8. Run manual C++ mutation injection for Category B mutations (B1-B7)
9. Tighten E2E thresholds based on historical baseline data (E18-E20)
10. Implement determinism reruns in orchestrator (E16)
11. Add logits file validation in runners (E11)

### Tier 4: Ongoing

12. Integrate `mutmut` into CI — run on changed files, reject PRs with >5% surviving mutants
13. Add mutation testing to the "what to run when" table in CLAUDE.md
14. Track mutation kill rate per file as a coverage quality metric

---

## Appendix: Files by Test Coverage Quality

### Well-Tested (>80% mutation kill rate expected)
- `src/bundle/bundle_format.cpp` — strong round-trip tests
- `src/tokenizer/vocab_tokenizer.cpp` — encode/decode verified
- `src/runtime/trt/trt_decode_runtime.cpp` — mask/argmax tested
- `trtf_build/trtf_build/bundle_writer.py` — round-trip verified
- `trtf_build/trtf_build/config.py` — multiple config scenarios

### Moderately Tested (40-80%)
- `src/runtime/trt/device_kv_cache.cpp` — position/mask yes, data no
- `src/utils/json_helpers.cpp` — indirect via bundle tests
- `trtf_build/trtf_build/checkpoint_mapper.py` — basic paths yes, edge cases no
- `trtf_build/trtf_build/debug_runner.py` — state machine yes, generate loop no
- `trtf_build/trtf_build/graph_ops.py` — reference comparison yes, axis/sign no

### Poorly Tested (<40%)
- **`src/runtime/trt/trt_backend_shared.cpp`** — NO unit tests for generate()
- **`src/runtime/trt/mamba_backend.cpp`** — NO unit tests for generate()
- **`src/cabi/fast_path_config.cpp`** — minimal config parsing tests
- **`trtf_build/trtf_build/graph_blocks.py`** — shape tests only, no semantic tests
- **`trtf_build/trtf_build/standard_decoder_builder.py`** — builds but no contract tests
- **`trtf_build/trtf_build/families/*.py`** — only 1 of 40+ plugins tested
