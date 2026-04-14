"""Tests for tools/diff_torchtrt.py — logit comparison harness.

Tests the comparison logic with synthetic data. Does not require GPU or models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add tools/ to path so we can import diff_torchtrt
TOOLS_DIR = str(Path(__file__).resolve().parents[2] / "tools")
sys.path.insert(0, TOOLS_DIR)

try:
    from diff_torchtrt import _compare_logits
except ImportError:
    pytest.skip("diff_torchtrt not importable", allow_module_level=True)


class TestCompareLogits:
    def test_identical_logits(self):
        logits = [np.random.randn(1000).astype(np.float32) for _ in range(5)]
        result = _compare_logits(logits, logits, atol=1e-6)
        assert result["num_steps"] == 5
        assert result["top1_match_rate"] == 1.0
        assert result["mean_cosine_sim"] > 0.999
        assert result["max_abs_diff"] < 1e-6
        assert result["all_within_atol"] is True

    def test_slightly_different_logits(self):
        base = [np.random.randn(1000).astype(np.float32) for _ in range(3)]
        noisy = [b + np.random.randn(1000).astype(np.float32) * 0.001
                 for b in base]
        result = _compare_logits(base, noisy, atol=0.01)
        # Should still have high cosine similarity and top-1 match
        assert result["mean_cosine_sim"] > 0.99
        assert result["top1_match_rate"] >= 0.5  # noisy but close

    def test_very_different_logits(self):
        a = [np.random.randn(1000).astype(np.float32) for _ in range(3)]
        b = [np.random.randn(1000).astype(np.float32) for _ in range(3)]
        result = _compare_logits(a, b, atol=0.01)
        assert result["all_within_atol"] is False
        assert result["max_abs_diff"] > 0.01

    def test_single_step(self):
        logits = [np.zeros(100, dtype=np.float32)]
        logits[0][42] = 10.0  # clear argmax
        result = _compare_logits(logits, logits, atol=1e-6)
        assert result["num_steps"] == 1
        assert result["top1_matches"] == 1

    def test_step_count_mismatch(self):
        a = [np.zeros(100, dtype=np.float32)]
        b = [np.zeros(100, dtype=np.float32)] * 2
        with pytest.raises(AssertionError, match="Step count mismatch"):
            _compare_logits(a, b, atol=1.0)

    def test_top5_overlap(self):
        logits = np.zeros(100, dtype=np.float32)
        logits[0:5] = [10.0, 9.0, 8.0, 7.0, 6.0]
        result = _compare_logits([logits], [logits], atol=1.0)
        assert result["mean_top5_overlap"] == 1.0

    def test_cosine_sim_orthogonal(self):
        a = np.zeros(100, dtype=np.float32)
        a[0] = 1.0
        b = np.zeros(100, dtype=np.float32)
        b[1] = 1.0
        result = _compare_logits([a], [b], atol=1.0)
        assert abs(result["mean_cosine_sim"]) < 0.01  # ~0 for orthogonal
