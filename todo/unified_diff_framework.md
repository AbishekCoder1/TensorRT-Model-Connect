# Plan: Unified Diff Test Framework

## Context

The diff test tools grew organically — one per model type. Each has its own CLI, output format, and comparison logic:

| Tool | Model types | Input | What it compares |
|------|-------------|-------|-----------------|
| `diff_logits.py` | Decoder, Mamba | HF model ID | Multi-step logits |
| `diff_layers.py` | Decoder | HF model ID | Per-layer hidden states |
| `diff_vl.py` | VL (Qwen-VL) | Bundle + image | Vision features, generation, C++ parity |
| `debug_diffusion_pipeline.py` | Diffusion (Wan2.1) | Bundle | 9 components (T5, DiT, scheduler, etc.) |
| `test_runner_parity.py` | Any bundled model | Bundle | Python vs C++ exact token match |
| `perf_compare.py` | Decoder, Mamba | HF model or bundle | Latency/throughput |

**Problems:**
1. Agents must know which tool + CLI for which model type — no auto-dispatch
2. No unified result format — each tool prints different stdout
3. No auto-discovery — can't ask "what tests apply to this model?"
4. Not library-importable — CLI-only scripts, can't compose in Python

## Goal

A unified `diff.py` CLI that auto-detects model type, runs applicable tests, and returns structured JSON results. Each test is also library-importable for direct Python use.

```bash
# Agent just runs this — framework handles dispatch
python tools/diff.py run --model Qwen/Qwen3-0.6B --json result.json
python tools/diff.py run --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers --bundle /path/wan.trtfb --json result.json
python tools/diff.py run --model Qwen/Qwen2.5-VL-3B-Instruct --bundle /path/vl.trtfb --image test.jpg --json result.json

# Or pick specific tests
python tools/diff.py run --model Qwen/Qwen3-0.6B --test logit_diff --test layer_diff

# List available tests
python tools/diff.py list --model Qwen/Qwen3-0.6B
# → logit_diff, layer_diff, runner_parity, perf_benchmark
```

---

## Architecture

### Tools vs tests separation

Clear boundary:
- **`tools/`** — Agent-facing. What agents use to implement/debug models. CLIs, libraries, dev aids.
- **`tests/`** — CI-facing. Pytest/ctest scripts that detect regressions. Organized by scope:
  - `tests/cpp/` — C++ unit tests (ctest)
  - `tests/builder/` — Python builder unit tests (config, checkpoint, graph ops, families)
  - `tests/tools/` — Self-tests for tools and the diff framework (pure logic, no GPU)
  - `tests/e2e/` — E2E regression tests (build + infer + compare, requires GPU + bundles)

### Package structure

```
tools/
  diff_framework/                  # NEW — unified framework
    __init__.py                    # Auto-discovers checks, exports public API
    protocol.py                    # DiffResult, TestContext, DiffTest protocol
    registry.py                    # @register decorator, strategy-based lookup
    runner.py                      # detect_runtime_strategy(), run_tests(), list_tests()
    checks/                        # Check implementations (one per check type)
      __init__.py                  # (named "checks" not "tests" to avoid confusion with tests/)
      logit_diff.py                # From diff_logits.py
      layer_diff.py                # From diff_layers.py
      runner_parity.py             # From test_runner_parity.py
      perf_benchmark.py            # From perf_compare.py
      diffusion_components.py      # From debug_diffusion_pipeline.py (all 9 steps)
      vl_pipeline.py               # From diff_vl.py (all 4 sub-tests)

  diff.py                          # NEW — unified CLI entry point

  # Existing tools — preserved as backward-compat wrappers
  diff_logits.py                   # Adds run_as_diff_test(), main() unchanged
  diff_layers.py                   # Same pattern
  diff_vl.py                       # Same pattern
  debug_diffusion_pipeline.py      # Same pattern
  test_runner_parity.py            # Same pattern
  perf_compare.py                  # Same pattern
  tool_helpers.py                  # Unchanged
  diffusion_helpers.py             # Unchanged

tests/
  tools/
    test_diff_framework.py         # NEW — self-tests for the framework (see below)
    test_diff_logits.py            # Existing — tests compare_logits logic
    test_perf_compare.py           # Existing — tests _stats, _fmt, build_json_output
    ...                            # Other existing tool tests
```

### Core types: `protocol.py`

```python
@dataclass
class DiffResult:
    test_name: str                 # "logit_diff", "vl_pipeline", etc.
    model: str                     # HF repo ID or bundle path
    runtime_strategy: str          # "decoder_kv_cache", "diffusion", etc.
    passed: bool
    status: str                    # "PASS", "FAIL", "SKIP", "ERROR"
    message: str                   # 1-line summary
    metrics: dict                  # Test-specific: {"max_abs_diff": 0.003, ...}
    duration_s: float
    details: str                   # Multi-line detailed output (for debugging)

    def to_dict(self) -> dict: ...
    def to_json(self) -> str: ...

    @staticmethod
    def skip(...) -> DiffResult: ...
    @staticmethod
    def error(...) -> DiffResult: ...

@dataclass
class TestContext:
    model: str                     # HF repo ID or local path
    runtime_strategy: str          # Auto-detected or from manifest
    bundle_path: str | None        # Pre-built bundle (some tests require)
    binary_path: str | None        # C++ trtf binary path
    hf_python: str | None          # Python interpreter for tokenizer bridge
    image_path: str | None         # Test image (VL models)
    max_cache_length: int = 256
    max_new_tokens: int = 20
    atol: float = 1e-3
    layer_atol: float = 0.05
    trust_remote_code: bool = False
    verbose: bool = False
    num_inference_steps: int = 30  # Diffusion

@runtime_checkable
class DiffTest(Protocol):
    name: str
    description: str
    runtime_strategies: list[str]  # ["decoder_kv_cache", "decoder_moe"] or ["*"]
    requires_bundle: bool
    requires_gpu: bool

    def run(self, ctx: TestContext) -> DiffResult: ...
```

### Registry: `registry.py`

```python
_REGISTRY: list[type[DiffTest]] = []

def register(cls): ...              # Decorator — adds to registry
def get_tests_for_strategy(s): ...  # Filter by runtime_strategy
def get_test_by_name(name): ...     # Lookup by name
```

Auto-discovery: `__init__.py` uses `pkgutil.iter_modules` on `checks/` (same pattern as `trtf_build/families/`).

### Runner: `runner.py`

```python
def detect_runtime_strategy(model: str) -> str:
    """Auto-detect from HF config via family plugin, or from bundle config.json."""

def list_tests(runtime_strategy: str | None) -> list[dict]:
    """List available tests, optionally filtered."""

def run_tests(ctx: TestContext, test_names: list[str] | None = None) -> list[DiffResult]:
    """Run applicable tests. Auto-discovers if test_names is None."""
```

### Test-to-strategy mapping

| Test | Strategies | Bundle? | GPU? |
|------|-----------|---------|------|
| `logit_diff` | decoder_kv_cache, decoder_moe, ssm_recurrent | No | Yes |
| `layer_diff` | decoder_kv_cache, decoder_moe | No | Yes |
| `runner_parity` | decoder_kv_cache, decoder_moe, ssm_recurrent | Yes | Yes |
| `perf_benchmark` | decoder_kv_cache, decoder_moe, ssm_recurrent | No | Yes |
| `vl_pipeline` | vision_language | Yes | Yes |
| `diffusion_components` | diffusion | Yes | Yes |

---

## Migration pattern

Each existing tool gets a `run_as_diff_test(ctx: TestContext) -> DiffResult` function added at the bottom. The framework test class delegates to it. Existing `main()` and CLI are untouched.

**Example for diff_logits.py:**

```python
# --- Added at bottom of diff_logits.py ---

def run_as_diff_test(ctx):
    """Framework entry point. Returns DiffResult."""
    from diff_framework.protocol import DiffResult
    import time

    t0 = time.monotonic()
    engine_plan, config, model_dir = build_trt_engine(
        ctx.model, ctx.max_cache_length, ctx.verbose)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=ctx.trust_remote_code)

    worst_diff = 0.0
    all_passed = True
    for label, prompt in STANDARD_PROMPTS:
        input_ids = tokenizer.encode(prompt)
        trt_logits = run_trt(engine_plan, config, input_ids,
                              ctx.max_new_tokens, ctx.max_cache_length)
        hf_logits = run_hf(model_dir, input_ids, ctx.max_new_tokens,
                            trust_remote_code=ctx.trust_remote_code)
        max_diff, _ = compare_logits(trt_logits, hf_logits, ctx.atol)
        worst_diff = max(worst_diff, max_diff)
        if max_diff > ctx.atol:
            all_passed = False

    return DiffResult(
        test_name="logit_diff", model=ctx.model,
        runtime_strategy=ctx.runtime_strategy,
        passed=all_passed, status="PASS" if all_passed else "FAIL",
        message=f"max_abs_logit_diff={worst_diff:.6f} (atol={ctx.atol})",
        metrics={"max_abs_diff": worst_diff, "atol": ctx.atol},
        duration_s=time.monotonic() - t0, details="")
```

**Framework check class (in diff_framework/checks/logit_diff.py):**

```python
from diff_framework.registry import register

@register
class LogitDiffTest:
    name = "logit_diff"
    description = "Per-step logit comparison: TRT vs HF transformers"
    runtime_strategies = ["decoder_kv_cache", "decoder_moe", "ssm_recurrent"]
    requires_bundle = False
    requires_gpu = True

    def run(self, ctx):
        from diff_logits import run_as_diff_test
        return run_as_diff_test(ctx)
```

This pattern applies to all 6 tools. The framework classes are thin delegates — all logic stays in the original tool files.

### Special cases

**diff_vl.py** — Has 4 sub-tests (vision, embed, generation, cpp_parity). Becomes one `VLPipelineTest` that runs all 4 internally, returns a single `DiffResult` with per-sub-test metrics:
```python
metrics={"vision": "PASS", "embed": "PASS", "generation": "PASS", "cpp_parity": "PASS"}
```

**debug_diffusion_pipeline.py** — Has 9 steps sharing mutable state (`Context`). Becomes one `DiffusionComponentsTest` that runs all 9 steps internally, returns per-step results in metrics:
```python
metrics={"config": "PASS", "text_proj": "PASS", "t5": "PASS", ..., "steps_passed": 9, "steps_total": 9}
```

---

## Unified CLI: `diff.py`

```bash
# Subcommands
python tools/diff.py list [--model MODEL]       # List available tests
python tools/diff.py run --model MODEL [opts]    # Run tests

# Run options
  --model MODEL          HF repo ID or local path (required)
  --test TEST            Specific test(s) to run (repeatable, default: all applicable)
  --bundle PATH          Pre-built .trtfb bundle
  --binary PATH          C++ trtf binary (default: ./build/trtf)
  --hf-python PATH       Python for HF tokenizer bridge
  --image PATH           Test image (for VL models)
  --max-cache-length N   TRT cache length (default: 256)
  --max-new-tokens N     Generation length (default: 20)
  --atol FLOAT           Logit tolerance (default: 1e-3)
  --trust-remote-code    Allow custom model code
  --json PATH            Save structured JSON results
  --verbose              Enable verbose logging
```

### JSON output format

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "runtime_strategy": "decoder_kv_cache",
  "results": [
    {
      "test_name": "logit_diff",
      "passed": true,
      "status": "PASS",
      "message": "max_abs_logit_diff=0.000312 (atol=0.001)",
      "metrics": {"max_abs_diff": 0.000312, "atol": 0.001},
      "duration_s": 45.2
    },
    {
      "test_name": "layer_diff",
      "passed": true,
      "status": "PASS",
      "message": "max_overall_diff=0.042 (atol=0.05)",
      "metrics": {"max_overall_diff": 0.042},
      "duration_s": 32.1
    }
  ]
}
```

---

## Files summary

| File | Action |
|------|--------|
| `tools/diff_framework/__init__.py` | **NEW** — auto-discover checks, export API |
| `tools/diff_framework/protocol.py` | **NEW** — DiffResult, TestContext, DiffTest protocol |
| `tools/diff_framework/registry.py` | **NEW** — @register, lookup by strategy/name |
| `tools/diff_framework/runner.py` | **NEW** — detect_runtime_strategy, run_tests, list_tests |
| `tools/diff_framework/checks/__init__.py` | **NEW** — empty |
| `tools/diff_framework/checks/logit_diff.py` | **NEW** — delegates to diff_logits.run_as_diff_test |
| `tools/diff_framework/checks/layer_diff.py` | **NEW** — delegates to diff_layers.run_as_diff_test |
| `tools/diff_framework/checks/runner_parity.py` | **NEW** — delegates to test_runner_parity.run_as_diff_test |
| `tools/diff_framework/checks/perf_benchmark.py` | **NEW** — delegates to perf_compare.run_as_diff_test |
| `tools/diff_framework/checks/vl_pipeline.py` | **NEW** — delegates to diff_vl.run_as_diff_test |
| `tools/diff_framework/checks/diffusion_components.py` | **NEW** — delegates to debug_diffusion_pipeline.run_as_diff_test |
| `tools/diff.py` | **NEW** — unified CLI (list + run subcommands) |
| `tools/diff_logits.py` | **MODIFY** — add `run_as_diff_test()` at bottom |
| `tools/diff_layers.py` | **MODIFY** — add `run_as_diff_test()` at bottom |
| `tools/diff_vl.py` | **MODIFY** — add `run_as_diff_test()` at bottom |
| `tools/debug_diffusion_pipeline.py` | **MODIFY** — add `run_as_diff_test()` at bottom |
| `tools/test_runner_parity.py` | **MODIFY** — add `run_as_diff_test()` at bottom |
| `tools/perf_compare.py` | **MODIFY** — add `run_as_diff_test()` at bottom |
| `tests/tools/test_diff_framework.py` | **NEW** — framework self-tests (see below) |

**Unchanged:** tool_helpers.py, diffusion_helpers.py, validate_dit.py, validate_t5.py, diff_t5.py, test_graph_ops.py, existing tests in tests/

---

## Framework self-tests: `tests/tools/test_diff_framework.py`

Pure-Python tests (no GPU, no model loading) that verify the framework mechanics. Runs as part of Tier 1 (`pytest tests/tools/ -v`).

### TestDiffResult — serialization and constructors

```python
class TestDiffResult:
    def test_to_dict_roundtrip(self):
        r = DiffResult(test_name="logit_diff", model="test/model",
                       runtime_strategy="decoder_kv_cache",
                       passed=True, status="PASS", message="ok",
                       metrics={"max_abs_diff": 0.001}, duration_s=1.5, details="")
        d = r.to_dict()
        assert d["test_name"] == "logit_diff"
        assert d["passed"] is True
        assert d["metrics"]["max_abs_diff"] == 0.001

    def test_to_json_valid(self):
        r = DiffResult(...)
        parsed = json.loads(r.to_json())
        assert parsed["status"] == "PASS"

    def test_skip_constructor(self):
        r = DiffResult.skip("x", "m", "s", "no bundle")
        assert r.status == "SKIP"
        assert r.passed is True  # skip is not a failure

    def test_error_constructor(self):
        r = DiffResult.error("x", "m", "s", "crash")
        assert r.status == "ERROR"
        assert r.passed is False
```

### TestRegistry — registration and lookup

```python
class TestRegistry:
    def test_register_and_lookup_by_name(self):
        # Create a mock check, register it, verify get_test_by_name finds it

    def test_get_tests_for_strategy_filters(self):
        # Register checks with different strategies, verify filtering

    def test_unknown_test_returns_none(self):
        assert get_test_by_name("nonexistent") is None

    def test_wildcard_strategy_matches_all(self):
        # A check with runtime_strategies=["*"] matches any strategy
```

### TestRunner — orchestration logic

```python
class TestRunner:
    def test_detect_runtime_strategy_from_plugin(self, monkeypatch):
        # Mock _resolve_model + ModelConfig + find_plugin
        # Verify correct strategy returned

    def test_list_tests_returns_expected_fields(self):
        entries = list_tests()
        for e in entries:
            assert "name" in e and "description" in e
            assert "runtime_strategies" in e

    def test_run_tests_skips_bundle_required(self):
        # Create TestContext without bundle_path
        # Verify checks requiring bundle get status="SKIP"

    def test_run_tests_specific_names(self):
        # Pass test_names=["logit_diff"], verify only that check runs

    def test_run_tests_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown test"):
            run_tests(ctx, test_names=["nonexistent"])
```

### TestCLI — argument parsing (dry-run, no execution)

```python
class TestCLI:
    def test_list_subcommand_parses(self):
        # Verify argparse accepts: diff.py list --model X

    def test_run_subcommand_parses(self):
        # Verify argparse accepts: diff.py run --model X --test logit_diff --json out.json

    def test_run_requires_model(self):
        # Verify argparse errors without --model
```

---

## Implementation phases

### Phase 1: Framework skeleton + first check
1. Create `diff_framework/` package (protocol.py, registry.py, runner.py, `__init__.py`)
2. Create `diff_framework/checks/logit_diff.py`
3. Add `run_as_diff_test()` to `diff_logits.py`
4. Create `diff.py` CLI
5. Create `tests/tools/test_diff_framework.py` — self-tests for protocol + registry + runner
6. Verify: `pytest tests/tools/test_diff_framework.py -v`, `diff.py list`, `diff.py run --model Qwen/Qwen3-0.6B --test logit_diff`

### Phase 2: Remaining decoder checks
7. layer_diff, runner_parity, perf_benchmark (same pattern)
8. Add `run_as_diff_test()` to each existing tool

### Phase 3: VL + diffusion checks
9. vl_pipeline.py + run_as_diff_test in diff_vl.py
10. diffusion_components.py + run_as_diff_test in debug_diffusion_pipeline.py

---

## Verification

```bash
# 1. Framework self-tests (no GPU, Tier 1)
pytest tests/tools/test_diff_framework.py -v

# 2. All existing tool self-tests still pass (no GPU, Tier 1)
pytest tests/tools/ -v
pytest tests/builder/ -v --ignore=tests/builder/test_cli.py

# 3. Framework CLI basics (no GPU)
python tools/diff.py list
python tools/diff.py list --model Qwen/Qwen3-0.6B

# 4. Backward compat — old CLIs unchanged (no GPU)
python tools/diff_logits.py --help
python tools/debug_diffusion_pipeline.py --help

# 5. Framework single-check run (GPU, Tier 3)
python tools/diff.py run --model Qwen/Qwen3-0.6B --test logit_diff --json /tmp/result.json
cat /tmp/result.json | python -m json.tool

# 6. Framework all-checks run (GPU)
python tools/diff.py run --model Qwen/Qwen3-0.6B --json /tmp/all.json

# 7. Parallel agent simulation (GPU)
CUDA_VISIBLE_DEVICES=0 python tools/diff.py run --model Qwen/Qwen3-0.6B --test logit_diff --json /tmp/a.json &
CUDA_VISIBLE_DEVICES=1 python tools/diff.py run --model meta-llama/Llama-3.2-1B --test logit_diff --json /tmp/b.json &
wait

# 8. E2E regression unchanged (GPU, Tier 3-4)
pytest tests/e2e/test_full_pipeline.py -v -k qwen3-0.6b \
  --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python
```
