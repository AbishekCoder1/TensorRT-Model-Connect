"""Self-tests for tools/diff_logits.py — battery prompts, tolerance, compare_logits."""

from __future__ import annotations

import numpy as np
import pytest


def _import_diff_logits():
    import importlib
    return importlib.import_module("diff_logits")


class TestStandardPrompts:
    """Battery prompt list sanity checks."""

    def test_battery_has_entries(self):
        mod = _import_diff_logits()
        assert len(mod.STANDARD_PROMPTS) >= 3

    def test_battery_entries_are_tuples(self):
        mod = _import_diff_logits()
        for entry in mod.STANDARD_PROMPTS:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            label, prompt = entry
            assert isinstance(label, str) and len(label) > 0
            assert isinstance(prompt, str) and len(prompt) > 0

    def test_battery_labels_unique(self):
        mod = _import_diff_logits()
        labels = [label for label, _ in mod.STANDARD_PROMPTS]
        assert len(labels) == len(set(labels))


class TestCompareLogits:
    """Tests for compare_logits() — pure numpy, no GPU."""

    def test_identical_logits_zero_diff(self):
        mod = _import_diff_logits()
        logits = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
        max_diff, lines = mod.compare_logits(logits, logits, atol=1e-3)
        assert max_diff == 0.0
        assert len(lines) == 2

    def test_small_diff_within_tolerance(self):
        mod = _import_diff_logits()
        trt = [np.array([1.0, 2.0, 3.0])]
        hf = [np.array([1.0001, 2.0001, 3.0001])]
        max_diff, lines = mod.compare_logits(trt, hf, atol=1e-3)
        assert max_diff < 1e-3
        assert "argmax_match=Y" in lines[0]

    def test_large_diff_exceeds_tolerance(self):
        mod = _import_diff_logits()
        trt = [np.array([1.0, 2.0, 3.0])]
        hf = [np.array([1.0, 2.0, 5.0])]
        max_diff, lines = mod.compare_logits(trt, hf, atol=1e-3)
        assert max_diff > 1e-3
        assert "argmax_match=Y" in lines[0]  # argmax still 2 for both

    def test_argmax_mismatch(self):
        mod = _import_diff_logits()
        trt = [np.array([10.0, 1.0, 1.0])]
        hf = [np.array([1.0, 1.0, 10.0])]
        max_diff, lines = mod.compare_logits(trt, hf, atol=1e-3)
        assert "argmax_match=N" in lines[0]

    def test_shape_mismatch_reported(self):
        mod = _import_diff_logits()
        trt = [np.array([1.0, 2.0])]
        hf = [np.array([1.0, 2.0, 3.0])]
        max_diff, lines = mod.compare_logits(trt, hf, atol=1e-3)
        assert "shape mismatch" in lines[0]

    def test_top_k_overlap(self):
        mod = _import_diff_logits()
        vocab = 100
        logits = np.random.randn(vocab).astype(np.float32)
        trt = [logits]
        hf = [logits.copy()]
        max_diff, lines = mod.compare_logits(trt, hf, atol=1e-3, top_k=5)
        assert "top5_overlap=5/5" in lines[0]

    def test_different_length_sequences(self):
        mod = _import_diff_logits()
        trt = [np.array([1.0, 2.0])] * 3
        hf = [np.array([1.0, 2.0])] * 5
        max_diff, lines = mod.compare_logits(trt, hf, atol=1e-3)
        assert len(lines) == 3  # min(3, 5)
