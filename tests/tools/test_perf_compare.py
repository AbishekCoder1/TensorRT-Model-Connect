"""Self-tests for tools/perf_compare.py — stats, formatting, JSON output, reporting."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest


def _import_perf_compare():
    import importlib
    return importlib.import_module("perf_compare")


# ---------------------------------------------------------------------------
# _stats
# ---------------------------------------------------------------------------

class TestStats:
    """Tests for _stats() — timing statistics helper."""

    def test_empty_list(self):
        mod = _import_perf_compare()
        result = mod._stats([])
        assert result["mean"] == 0.0
        assert result["std"] == 0.0
        assert result["values"] == []

    def test_single_value(self):
        mod = _import_perf_compare()
        result = mod._stats([42.0])
        assert result["mean"] == 42.0
        assert result["std"] == 0.0
        assert result["values"] == [42.0]

    def test_multiple_values(self):
        mod = _import_perf_compare()
        result = mod._stats([10.0, 20.0, 30.0])
        assert result["mean"] == pytest.approx(20.0)
        assert result["std"] > 0
        assert len(result["values"]) == 3

    def test_identical_values_zero_std(self):
        mod = _import_perf_compare()
        result = mod._stats([5.0, 5.0, 5.0])
        assert result["mean"] == 5.0
        assert result["std"] == 0.0


# ---------------------------------------------------------------------------
# _fmt / _speedup
# ---------------------------------------------------------------------------

class TestFormatting:
    """Tests for _fmt() and _speedup() — display helpers."""

    def test_fmt_basic(self):
        mod = _import_perf_compare()
        assert mod._fmt(12.3, 0.5) == "12.3 +/- 0.5"

    def test_fmt_zero(self):
        mod = _import_perf_compare()
        assert mod._fmt(0.0, 0.0) == "0.0 +/- 0.0"

    def test_speedup_normal(self):
        mod = _import_perf_compare()
        # HF=100ms, TRT=50ms → 2.00x
        assert mod._speedup(100.0, 50.0) == "2.00x"

    def test_speedup_slower(self):
        mod = _import_perf_compare()
        # HF=50ms, TRT=100ms → 0.50x
        assert mod._speedup(50.0, 100.0) == "0.50x"

    def test_speedup_zero_trt(self):
        mod = _import_perf_compare()
        assert mod._speedup(100.0, 0.0) == "N/A"

    def test_speedup_negative_trt(self):
        mod = _import_perf_compare()
        assert mod._speedup(100.0, -1.0) == "N/A"


# ---------------------------------------------------------------------------
# build_json_output
# ---------------------------------------------------------------------------

def _make_bench_result(prefill_ms, decode_ms, num_tokens, gen_ids):
    """Helper to build a synthetic bench result dict."""
    n = len(prefill_ms)
    return {
        "prefill_times": prefill_ms,
        "decode_times": decode_ms,
        "decode_token_counts": [num_tokens] * n,
        "gen_ids": gen_ids,
    }


class TestBuildJsonOutput:
    """Tests for build_json_output() — structured result dict."""

    def test_structure_has_required_keys(self):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0, 12.0], [40.0, 42.0], 20, [1, 2, 3])
        hf = _make_bench_result([8.0, 9.0], [80.0, 82.0], 20, [1, 2, 3])
        result = mod.build_json_output(
            "test-model", "Hello", 3, 20, 2, 1, "float16", trt, hf)

        assert "metadata" in result
        assert "trt" in result
        assert "hf" in result
        assert "speedup" in result
        assert "token_match" in result

    def test_metadata_fields(self):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0], [40.0], 20, [1])
        hf = _make_bench_result([8.0], [80.0], 20, [1])
        result = mod.build_json_output(
            "Qwen/Qwen3-0.6B", "Hello world", 5, 20, 3, 2, "float16", trt, hf)

        meta = result["metadata"]
        assert meta["model"] == "Qwen/Qwen3-0.6B"
        assert meta["prompt"] == "Hello world"
        assert meta["num_input_tokens"] == 5
        assert meta["max_new_tokens"] == 20
        assert meta["iterations"] == 3
        assert meta["warmup"] == 2
        assert meta["hf_dtype"] == "float16"
        assert "timestamp" in meta

    def test_speedup_computation(self):
        mod = _import_perf_compare()
        # TRT decode 2x faster than HF
        trt = _make_bench_result([10.0, 10.0], [50.0, 50.0], 20, [1, 2])
        hf = _make_bench_result([5.0, 5.0], [100.0, 100.0], 20, [1, 2])
        result = mod.build_json_output(
            "m", "p", 1, 20, 2, 0, "float16", trt, hf)

        assert result["speedup"]["decode"] == pytest.approx(2.0, abs=0.01)
        # Prefill: HF faster (5ms vs 10ms) → 0.5x
        assert result["speedup"]["prefill"] == pytest.approx(0.5, abs=0.01)

    def test_token_match_true(self):
        mod = _import_perf_compare()
        ids = [10, 20, 30]
        trt = _make_bench_result([10.0], [50.0], 3, ids)
        hf = _make_bench_result([8.0], [80.0], 3, ids)
        result = mod.build_json_output(
            "m", "p", 1, 3, 1, 0, "float16", trt, hf)
        assert result["token_match"] is True

    def test_token_match_false(self):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0], [50.0], 3, [10, 20, 30])
        hf = _make_bench_result([8.0], [80.0], 3, [10, 20, 99])
        result = mod.build_json_output(
            "m", "p", 1, 3, 1, 0, "float16", trt, hf)
        assert result["token_match"] is False

    def test_per_token_and_throughput(self):
        mod = _import_perf_compare()
        # 10 tokens in 100ms decode → 10ms/token, 100 t/s
        trt = _make_bench_result([5.0, 5.0], [100.0, 100.0], 10, list(range(10)))
        hf = _make_bench_result([5.0, 5.0], [200.0, 200.0], 10, list(range(10)))
        result = mod.build_json_output(
            "m", "p", 1, 10, 2, 0, "float16", trt, hf)

        assert result["trt"]["per_token_ms"]["mean"] == pytest.approx(10.0)
        assert result["trt"]["throughput_tps"]["mean"] == pytest.approx(100.0)
        assert result["hf"]["per_token_ms"]["mean"] == pytest.approx(20.0)
        assert result["hf"]["throughput_tps"]["mean"] == pytest.approx(50.0)

    def test_zero_decode_tokens(self):
        mod = _import_perf_compare()
        trt = _make_bench_result([5.0], [0.1], 0, [])
        hf = _make_bench_result([5.0], [0.1], 0, [])
        result = mod.build_json_output(
            "m", "p", 1, 0, 1, 0, "float16", trt, hf)

        assert result["trt"]["per_token_ms"]["mean"] == 0.0
        assert result["trt"]["throughput_tps"]["mean"] == 0.0

    def test_json_serializable(self):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0, 12.0], [40.0, 42.0], 5, [1, 2, 3])
        hf = _make_bench_result([8.0, 9.0], [80.0, 82.0], 5, [1, 2, 3])
        result = mod.build_json_output(
            "m", "p", 3, 5, 2, 1, "float16", trt, hf)
        # Must not raise
        serialized = json.dumps(result)
        parsed = json.loads(serialized)
        assert parsed["metadata"]["model"] == "m"


# ---------------------------------------------------------------------------
# print_report
# ---------------------------------------------------------------------------

class TestPrintReport:
    """Tests for print_report() — formatted output."""

    def test_report_contains_model_name(self, capsys):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0, 12.0], [40.0, 42.0], 20, [1, 2])
        hf = _make_bench_result([8.0, 9.0], [80.0, 82.0], 20, [1, 2])
        mod.print_report("TestModel", "Hello", 3, 20, 2, 1,
                         "float16", trt, hf)
        out = capsys.readouterr().out
        assert "TestModel" in out

    def test_report_contains_speedup(self, capsys):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0, 10.0], [50.0, 50.0], 20, [1])
        hf = _make_bench_result([5.0, 5.0], [100.0, 100.0], 20, [1])
        mod.print_report("m", "p", 1, 20, 2, 0, "float16", trt, hf)
        out = capsys.readouterr().out
        assert "Speedup" in out
        assert "2.00x" in out

    def test_report_token_match_true(self, capsys):
        mod = _import_perf_compare()
        ids = [1, 2, 3]
        trt = _make_bench_result([10.0], [50.0], 3, ids)
        hf = _make_bench_result([8.0], [80.0], 3, ids)
        mod.print_report("m", "p", 1, 3, 1, 0, "float16", trt, hf)
        out = capsys.readouterr().out
        assert "Token match: True" in out

    def test_report_token_match_false_shows_counts(self, capsys):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0], [50.0], 3, [1, 2, 3])
        hf = _make_bench_result([8.0], [80.0], 2, [1, 2])
        mod.print_report("m", "p", 1, 3, 1, 0, "float16", trt, hf)
        out = capsys.readouterr().out
        assert "Token match: False" in out
        assert "TRT=3" in out
        assert "HF=2" in out

    def test_report_kv_cache_footnote(self, capsys):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0], [50.0], 5, [1])
        hf = _make_bench_result([8.0], [80.0], 5, [1])
        mod.print_report("m", "p", 1, 5, 1, 0, "float16", trt, hf,
                         is_mamba=False)
        out = capsys.readouterr().out
        assert "KV cache" in out

    def test_report_mamba_footnote(self, capsys):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0], [50.0], 5, [1])
        hf = _make_bench_result([8.0], [80.0], 5, [1])
        mod.print_report("m", "p", 1, 5, 1, 0, "float16", trt, hf,
                         is_mamba=True)
        out = capsys.readouterr().out
        assert "recurrent state" in out
        assert "KV cache" not in out

    def test_report_shows_dtype(self, capsys):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0], [50.0], 5, [1])
        hf = _make_bench_result([8.0], [80.0], 5, [1])
        mod.print_report("m", "p", 1, 5, 1, 0, "bfloat16", trt, hf)
        out = capsys.readouterr().out
        assert "bfloat16" in out

    def test_report_long_prompt_truncated(self, capsys):
        mod = _import_perf_compare()
        long_prompt = "A" * 100
        trt = _make_bench_result([10.0], [50.0], 5, [1])
        hf = _make_bench_result([8.0], [80.0], 5, [1])
        mod.print_report("m", long_prompt, 50, 5, 1, 0, "float16", trt, hf)
        out = capsys.readouterr().out
        assert "..." in out
        # Should not print all 100 characters
        assert "A" * 100 not in out
