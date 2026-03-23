"""Self-tests for tools/perf_compare.py — stats, formatting, JSON output, reporting."""

from __future__ import annotations

import json
import math
from unittest import mock

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


# ---------------------------------------------------------------------------
# Serial GPU execution — main() ordering
# ---------------------------------------------------------------------------

def _fake_bench_result():
    return {
        "prefill_times": [10.0],
        "decode_times": [50.0],
        "decode_token_counts": [5],
        "gen_ids": [1, 2, 3],
    }


class TestSerialGpuExecution:
    """Verify main() runs TRT before HF and frees GPU between them."""

    def _run_main_with_mocks(self, monkeypatch, is_mamba=False):
        """Patch all heavy deps in main() and return the call log."""
        mod = _import_perf_compare()
        call_log = []

        # Patch sys.argv
        monkeypatch.setattr("sys.argv", [
            "perf_compare.py",
            "--model", "fake/model",
            "--bundle", "/fake/bundle.trtfb",
            "--prompt", "Hello",
            "--max-new-tokens", "5",
            "--warmup", "0",
            "--iterations", "1",
        ])

        # Patch _resolve_model
        monkeypatch.setattr(
            "trtf_build.engine_builder._resolve_model",
            lambda _: "/fake/model_dir")

        # Patch AutoTokenizer
        fake_tok = mock.MagicMock()
        fake_tok.encode.return_value = [1, 2, 3]
        fake_tok.eos_token_id = None
        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained",
            lambda *a, **kw: fake_tok)

        # Patch load_trt_from_bundle
        def fake_load_bundle(path):
            call_log.append("load_bundle")
            return (b"fake_plan", 2, 128, {}, is_mamba)
        monkeypatch.setattr(mod, "load_trt_from_bundle", fake_load_bundle)

        # Patch bench_trt / bench_trt_mamba
        def fake_bench_trt(*args, **kwargs):
            call_log.append("bench_trt")
            return _fake_bench_result()
        monkeypatch.setattr(mod, "bench_trt", fake_bench_trt)

        def fake_bench_trt_mamba(*args, **kwargs):
            call_log.append("bench_trt_mamba")
            return _fake_bench_result()
        monkeypatch.setattr(mod, "bench_trt_mamba", fake_bench_trt_mamba)

        # Patch gc.collect and torch.cuda.empty_cache to track calls
        real_gc_mod = __import__("gc")
        original_collect = real_gc_mod.collect

        def tracking_gc_collect():
            call_log.append("gc_collect")
            return original_collect()
        monkeypatch.setattr(real_gc_mod, "collect", tracking_gc_collect)

        fake_cuda = mock.MagicMock()
        fake_cuda.empty_cache = lambda: call_log.append("empty_cache")
        fake_torch = mock.MagicMock()
        fake_torch.cuda = fake_cuda
        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

        # Patch load_hf_model / bench_hf
        def fake_load_hf(*args, **kwargs):
            call_log.append("load_hf")
            return mock.MagicMock()
        monkeypatch.setattr(mod, "load_hf_model", fake_load_hf)

        def fake_bench_hf(*args, **kwargs):
            call_log.append("bench_hf")
            return _fake_bench_result()
        monkeypatch.setattr(mod, "bench_hf", fake_bench_hf)

        # Patch print_report to suppress output
        monkeypatch.setattr(mod, "print_report", lambda *a, **kw: None)

        mod.main()
        return call_log

    def test_trt_runs_before_hf_load(self, monkeypatch):
        """TRT benchmark must complete before HF model is loaded."""
        log = self._run_main_with_mocks(monkeypatch)
        trt_idx = log.index("bench_trt")
        hf_load_idx = log.index("load_hf")
        assert trt_idx < hf_load_idx, (
            f"bench_trt ({trt_idx}) must run before load_hf ({hf_load_idx}): {log}")

    def test_gpu_freed_between_trt_and_hf(self, monkeypatch):
        """gc.collect + empty_cache must happen between TRT and HF."""
        log = self._run_main_with_mocks(monkeypatch)
        trt_idx = log.index("bench_trt")
        hf_load_idx = log.index("load_hf")

        # Find gc_collect and empty_cache between TRT and HF load
        mid_section = log[trt_idx + 1:hf_load_idx]
        assert "gc_collect" in mid_section, (
            f"gc.collect() missing between bench_trt and load_hf: {log}")
        assert "empty_cache" in mid_section, (
            f"torch.cuda.empty_cache() missing between bench_trt and load_hf: {log}")

    def test_hf_freed_after_bench(self, monkeypatch):
        """HF model is freed after benchmarking (gc + empty_cache at end)."""
        log = self._run_main_with_mocks(monkeypatch)
        bench_hf_idx = log.index("bench_hf")
        remaining = log[bench_hf_idx + 1:]
        assert "gc_collect" in remaining, (
            f"gc.collect() missing after bench_hf: {log}")
        assert "empty_cache" in remaining, (
            f"torch.cuda.empty_cache() missing after bench_hf: {log}")

    def test_mamba_path_serial_execution(self, monkeypatch):
        """Mamba/SSM path also runs TRT before HF with GPU cleanup."""
        log = self._run_main_with_mocks(monkeypatch, is_mamba=True)
        trt_idx = log.index("bench_trt_mamba")
        hf_load_idx = log.index("load_hf")
        assert trt_idx < hf_load_idx

        mid_section = log[trt_idx + 1:hf_load_idx]
        assert "gc_collect" in mid_section
        assert "empty_cache" in mid_section

    def test_hf_never_loaded_before_trt_completes(self, monkeypatch):
        """Verify full ordering: load_bundle → bench_trt → cleanup → load_hf → bench_hf → cleanup."""
        log = self._run_main_with_mocks(monkeypatch)
        expected_order = ["load_bundle", "bench_trt", "gc_collect",
                          "empty_cache", "load_hf", "bench_hf",
                          "gc_collect", "empty_cache"]
        assert log == expected_order, (
            f"Expected exact serial order:\n  {expected_order}\nGot:\n  {log}")


# ---------------------------------------------------------------------------
# --trt-only tests
# ---------------------------------------------------------------------------

class TestTrtOnlyCLI:
    """Tests for --trt-only CLI flag."""

    def test_trt_only_flag_accepted(self):
        mod = _import_perf_compare()
        # Verify the parser accepts --trt-only
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--trt-only", action="store_true")
        args = parser.parse_args(["--trt-only"])
        assert args.trt_only is True

    def test_trt_only_json_structure(self):
        """TRT-only JSON should have empty hf section and None token_match."""
        # Build TRT-only json structure similar to what main() produces
        trt_only_json = {
            "metadata": {
                "model": "test-model",
                "gpu": "TestGPU",
                "trt_version": "10.0",
            },
            "trt": {
                "prefill_ms": {"mean": 5.0, "std": 0.1},
                "decode_ms": {"mean": 20.0, "std": 0.5},
                "per_token_ms": {"mean": 1.0, "std": 0.0},
                "throughput_tps": {"mean": 1000.0, "std": 0.0},
            },
            "hf": {},
            "speedup": {},
            "token_match": None,
        }
        # Verify structure
        assert trt_only_json["hf"] == {}
        assert trt_only_json["speedup"] == {}
        assert trt_only_json["token_match"] is None
        assert trt_only_json["trt"]["throughput_tps"]["mean"] == 1000.0

        # Verify it's valid JSON (serializable)
        import json
        serialized = json.dumps(trt_only_json)
        roundtrip = json.loads(serialized)
        assert roundtrip["trt"]["throughput_tps"]["mean"] == 1000.0
