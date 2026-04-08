"""Unit tests for similarity helper functions (no GPU required).

Tests the mathematical primitives used by test_similarity.py:
cosine_sim, kl_divergence, top_k_overlap, softmax_entropy.
"""

from __future__ import annotations

import math
import pytest
import numpy as np

try:
    from scipy.special import softmax, log_softmax
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

requires_scipy = pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")


# Import helpers (inline to avoid GPU-dependent imports in test_similarity.py)
def cosine_sim(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


@requires_scipy
def kl_divergence(p_logits, q_logits):
    p = softmax(p_logits)
    log_p = log_softmax(p_logits)
    log_q = log_softmax(q_logits)
    kl = float(np.sum(p * (log_p - log_q)))
    return max(0.0, kl)


def top_k_overlap(a, b, k=5):
    top_a = set(np.argsort(a)[-k:])
    top_b = set(np.argsort(b)[-k:])
    return len(top_a & top_b) / k


@requires_scipy
def softmax_entropy(logits):
    p = softmax(logits)
    log_p = log_softmax(logits)
    return float(-np.sum(p * log_p))


class TestCosineSim:
    def test_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert cosine_sim(a, a) == pytest.approx(1.0)

    def test_opposite(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert cosine_sim(a, b) == pytest.approx(-1.0)

    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_sim(a, b) == pytest.approx(0.0)

    def test_similar(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.1, 2.1, 3.1])
        sim = cosine_sim(a, b)
        assert sim > 0.99

    def test_zero_vector(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 2.0, 3.0])
        assert cosine_sim(a, b) == 0.0

    def test_both_zero(self):
        a = np.array([0.0, 0.0])
        assert cosine_sim(a, a) == 0.0

    def test_high_dimensional(self):
        np.random.seed(42)
        a = np.random.randn(10000)
        b = a + np.random.randn(10000) * 0.01  # small noise
        sim = cosine_sim(a, b)
        assert sim > 0.99

    def test_scaled_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        b = a * 100.0
        assert cosine_sim(a, b) == pytest.approx(1.0)

    def test_negative_correlation(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([4.0, 3.0, 2.0, 1.0])
        sim = cosine_sim(a, b)
        assert sim > 0.0  # not orthogonal, still positive
        assert sim < 1.0

    def test_large_values(self):
        a = np.array([1e10, 2e10, 3e10])
        b = np.array([1e10, 2e10, 3e10])
        assert cosine_sim(a, b) == pytest.approx(1.0)


@requires_scipy
class TestKLDivergence:
    def test_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert kl_divergence(a, a) == pytest.approx(0.0, abs=1e-10)

    def test_nonnegative(self):
        np.random.seed(42)
        a = np.random.randn(100)
        b = np.random.randn(100)
        assert kl_divergence(a, b) >= 0.0

    def test_small_perturbation(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = a + np.random.randn(5) * 0.01
        kl = kl_divergence(a, b)
        assert kl < 0.01  # very small KL for small perturbation

    def test_large_divergence(self):
        a = np.array([10.0, -10.0, -10.0])  # very peaked at 0
        b = np.array([-10.0, 10.0, -10.0])  # very peaked at 1
        kl = kl_divergence(a, b)
        assert kl > 1.0  # should be large

    def test_uniform_vs_peaked(self):
        uniform = np.zeros(10)  # uniform distribution
        peaked = np.zeros(10)
        peaked[0] = 10.0  # peaked at 0
        kl = kl_divergence(peaked, uniform)
        # Peaked to uniform: moderate KL
        assert kl > 0.1

    def test_asymmetric(self):
        a = np.array([5.0, 1.0, 1.0])
        b = np.array([1.0, 5.0, 1.0])
        kl_ab = kl_divergence(a, b)
        kl_ba = kl_divergence(b, a)
        # KL is asymmetric but both should be positive
        assert kl_ab > 0
        assert kl_ba > 0


class TestTopKOverlap:
    def test_identical(self):
        a = np.array([1.0, 5.0, 3.0, 4.0, 2.0])
        assert top_k_overlap(a, a, k=3) == 1.0

    def test_no_overlap(self):
        a = np.array([10.0, 9.0, 8.0, 1.0, 0.0])
        b = np.array([0.0, 1.0, 2.0, 10.0, 9.0])
        overlap = top_k_overlap(a, b, k=2)
        assert overlap == 0.0

    def test_partial_overlap(self):
        a = np.array([10.0, 9.0, 1.0, 2.0, 3.0])
        b = np.array([10.0, 1.0, 9.0, 2.0, 3.0])
        # Top-2 of a: {0, 1}, top-2 of b: {0, 2} → overlap = {0} → 1/2
        overlap = top_k_overlap(a, b, k=2)
        assert overlap == 0.5

    def test_k_equals_length(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([3.0, 2.0, 1.0])
        assert top_k_overlap(a, b, k=3) == 1.0  # all indices overlap

    def test_k_one(self):
        a = np.array([1.0, 5.0, 3.0])
        b = np.array([1.0, 5.0, 3.0])
        assert top_k_overlap(a, b, k=1) == 1.0

    def test_noisy_but_similar(self):
        np.random.seed(42)
        a = np.random.randn(1000)
        b = a + np.random.randn(1000) * 0.01
        overlap = top_k_overlap(a, b, k=10)
        assert overlap >= 0.8  # small noise preserves top-k


@requires_scipy
class TestSoftmaxEntropy:
    def test_uniform(self):
        logits = np.zeros(10)
        ent = softmax_entropy(logits)
        expected = math.log(10)
        assert ent == pytest.approx(expected, rel=1e-5)

    def test_peaked(self):
        logits = np.zeros(10)
        logits[0] = 100.0
        ent = softmax_entropy(logits)
        assert ent < 0.01  # very low entropy

    def test_two_class_uniform(self):
        logits = np.array([0.0, 0.0])
        ent = softmax_entropy(logits)
        assert ent == pytest.approx(math.log(2), rel=1e-5)

    def test_nonnegative(self):
        np.random.seed(42)
        for _ in range(10):
            logits = np.random.randn(100)
            ent = softmax_entropy(logits)
            assert ent >= 0.0

    def test_monotone_with_peakedness(self):
        # More peaked → lower entropy
        logits_flat = np.array([1.0, 1.0, 1.0, 1.0])
        logits_peaked = np.array([10.0, 1.0, 1.0, 1.0])
        assert softmax_entropy(logits_peaked) < softmax_entropy(logits_flat)

    def test_large_vocab(self):
        logits = np.random.randn(151936)  # Qwen3 vocab size
        ent = softmax_entropy(logits)
        # Should be bounded by log(vocab_size)
        assert 0 < ent <= math.log(151936) + 0.01
