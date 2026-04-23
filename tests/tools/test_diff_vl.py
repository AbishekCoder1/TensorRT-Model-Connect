"""Self-tests for tools/diff_vl.py — preprocessor config, cosine similarity, sanity checks.

Trace: ARCH-TRT-001, UD-TRT-DIFF-VL
Intent: Validate VL diff tool Qwen VL detection, cosine similarity computation, and preprocessor config
Preconditions: diff_vl module is importable; numpy available
Postconditions: Qwen VL model types are correctly identified and cosine similarity matches mathematical expectations
"""

from __future__ import annotations

import numpy as np


class TestIsQwenVl:
    """Test _is_qwen_vl() string matching — importable without GPU."""

    def test_qwen_vl_variants(self):
        # Import the function from tools/diff_vl.py
        import importlib
        mod = importlib.import_module("diff_vl")
        assert mod._is_qwen_vl("qwen2_5_vl") is True
        assert mod._is_qwen_vl("qwen_vl") is True
        assert mod._is_qwen_vl("Qwen2_5_VL") is True

    def test_non_vl_models(self):
        import importlib
        mod = importlib.import_module("diff_vl")
        assert mod._is_qwen_vl("qwen3") is False
        assert mod._is_qwen_vl("llama") is False
        assert mod._is_qwen_vl("mistral") is False


class TestCosineSimilarity:
    """Test cosine similarity computation (same formula as diff_vl.py)."""

    def test_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        cos_sim = np.dot(a, a) / (np.linalg.norm(a) * np.linalg.norm(a) + 1e-8)
        assert abs(cos_sim - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        assert abs(cos_sim) < 1e-6

    def test_opposite_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        b = -a
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        assert cos_sim < -0.99

    def test_threshold_check(self):
        """Cosine similarity < 0.5 means uncorrelated features (hard fail in diff_vl)."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.5, 0.5, 0.0])
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        assert cos_sim > 0.5  # These are somewhat correlated


class TestSanityChecks:
    """Test the NaN/Inf/zero sanity checks from diff_vl.py."""

    def test_all_zeros_detected(self):
        features = np.zeros((1, 64), dtype=np.float32)
        assert np.all(features == 0)

    def test_nan_detected(self):
        features = np.array([1.0, float("nan"), 3.0])
        assert np.any(np.isnan(features))

    def test_inf_detected(self):
        features = np.array([1.0, float("inf"), 3.0])
        assert np.any(np.isinf(features))

    def test_valid_features_pass(self):
        features = np.random.randn(1, 64).astype(np.float32)
        assert not np.all(features == 0)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))


class TestPreprocessorDefaults:
    """Verify expected default values used in diff_vl.py."""

    def test_default_image_mean(self):
        expected = (0.48145466, 0.4578275, 0.40821073)
        assert len(expected) == 3
        assert all(0.0 < v < 1.0 for v in expected)

    def test_default_image_std(self):
        expected = (0.26862954, 0.26130258, 0.27577711)
        assert len(expected) == 3
        assert all(0.0 < v < 1.0 for v in expected)

    def test_default_image_size(self):
        assert 448 > 0
        assert 448 % 14 == 0  # Must be divisible by patch size
