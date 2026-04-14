"""Unit tests for perf_compare_torchtrt.py helper functions.

Tests the reporting, stats, and formatting helpers without GPU.
"""

from __future__ import annotations

import json
import sys
import os
import pytest

# Add tools/ to path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))

try:
    from perf_utils import (
        stats as _stats, fmt as _fmt, speedup as _speedup, build_json_output,
    )
except ImportError:
    pytest.skip("perf_utils not importable", allow_module_level=True)


class TestStats:
    def test_empty(self):
        s = _stats([])
        assert s["mean"] == 0.0
        assert s["std"] == 0.0
        assert s["values"] == []

    def test_single(self):
        s = _stats([5.0])
        assert s["mean"] == 5.0
        assert s["std"] == 0.0
        assert s["values"] == [5.0]

    def test_multiple(self):
        s = _stats([10.0, 20.0, 30.0])
        assert s["mean"] == 20.0
        assert s["std"] > 0

    def test_identical(self):
        s = _stats([5.0, 5.0, 5.0])
        assert s["mean"] == 5.0
        assert s["std"] == 0.0


class TestFmt:
    def test_basic(self):
        result = _fmt(10.0, 2.0)
        assert "10.0" in result
        assert "2.0" in result

    def test_zero(self):
        result = _fmt(0.0, 0.0)
        assert "0.0" in result


class TestSpeedup:
    def test_normal(self):
        assert _speedup(100.0, 50.0) == "2.00x"

    def test_equal(self):
        assert _speedup(100.0, 100.0) == "1.00x"

    def test_slower(self):
        assert _speedup(50.0, 100.0) == "0.50x"

    def test_zero_candidate(self):
        assert _speedup(100.0, 0.0) == "N/A"


class TestBuildJsonOutput:
    def _make_results(self):
        return {
            "hf_eager": {
                "prefill_times": [10.0, 12.0, 11.0],
                "decode_times": [100.0, 110.0, 105.0],
                "gen_ids": [1, 2, 3, 4, 5],
                "num_tokens": 5,
            },
            "torch_trt": {
                "prefill_times": [5.0, 6.0, 5.5],
                "decode_times": [50.0, 55.0, 52.0],
                "gen_ids": [1, 2, 3, 4, 5],
                "num_tokens": 5,
            },
        }

    def test_structure(self):
        results = self._make_results()
        out = build_json_output("test-model", "hello", 1, 5, 3, 1, results)
        assert "metadata" in out
        assert "backends" in out
        assert "hf_eager" in out["backends"]
        assert "torch_trt" in out["backends"]

    def test_metadata(self):
        results = self._make_results()
        out = build_json_output("test-model", "hello", 3, 5, 3, 1, results)
        assert out["metadata"]["model"] == "test-model"
        assert out["metadata"]["prompt"] == "hello"
        assert out["metadata"]["num_input_tokens"] == 3
        assert out["metadata"]["max_new_tokens"] == 5

    def test_speedup(self):
        results = self._make_results()
        out = build_json_output("test-model", "hello", 1, 5, 3, 1, results)
        assert "decode_speedup_vs_eager" in out
        assert "torch_trt" in out["decode_speedup_vs_eager"]
        speedup = out["decode_speedup_vs_eager"]["torch_trt"]
        assert speedup > 1.5  # TRT should be ~2x faster in our mock data

    def test_backend_fields(self):
        results = self._make_results()
        out = build_json_output("test-model", "hello", 1, 5, 3, 1, results)
        trt = out["backends"]["torch_trt"]
        assert "prefill_ms" in trt
        assert "decode_ms" in trt
        assert "total_ms" in trt
        assert "per_token_ms" in trt
        assert "throughput_tps" in trt

    def test_json_serializable(self):
        results = self._make_results()
        out = build_json_output("test-model", "hello", 1, 5, 3, 1, results)
        # Should serialize without error
        s = json.dumps(out)
        assert len(s) > 0

    def test_single_backend(self):
        results = {
            "hf_eager": {
                "prefill_times": [10.0],
                "decode_times": [100.0],
                "gen_ids": [1, 2, 3],
                "num_tokens": 3,
            },
        }
        out = build_json_output("test-model", "hello", 1, 3, 1, 0, results)
        assert "hf_eager" in out["backends"]
        assert "decode_speedup_vs_eager" not in out or out["decode_speedup_vs_eager"] == {}

    def test_three_backends(self):
        results = {
            "hf_eager": {
                "prefill_times": [10.0],
                "decode_times": [100.0],
                "gen_ids": [1, 2, 3],
                "num_tokens": 3,
            },
            "torch_trt": {
                "prefill_times": [5.0],
                "decode_times": [50.0],
                "gen_ids": [1, 2, 3],
                "num_tokens": 3,
            },
            "torch_compile": {
                "prefill_times": [8.0],
                "decode_times": [70.0],
                "gen_ids": [1, 2, 3],
                "num_tokens": 3,
                "compile_time_s": 15.0,
            },
        }
        out = build_json_output("test-model", "hello", 1, 3, 1, 0, results)
        assert len(out["backends"]) == 3
        assert "compile_time_s" in out["backends"]["torch_compile"]
        speedups = out["decode_speedup_vs_eager"]
        assert "torch_trt" in speedups
        assert "torch_compile" in speedups
