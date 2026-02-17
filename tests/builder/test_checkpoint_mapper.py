"""Tests for checkpoint_mapper.py — weight transforms and loading.

Uses synthetic safetensors files. No TRT needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("trtf_build", reason="trtf_build requires tensorrt")
from trtf_build.checkpoint_mapper import (
    _transpose_2d,
    _expand_kv_projection,
    _repeat_head_norm,
    load_standard_weights,
)
from trtf_build.config import ModelConfig


class TestTranspose2d:
    def test_basic(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = _transpose_2d(arr, "test")
        expected = arr.T.astype(np.float32)
        np.testing.assert_array_equal(result, expected)
        assert result.shape == (3, 2)

    def test_square(self):
        arr = np.eye(4, dtype=np.float32)
        result = _transpose_2d(arr, "identity")
        np.testing.assert_array_equal(result, arr)

    def test_rank1_raises(self):
        arr = np.array([1, 2, 3], dtype=np.float32)
        with pytest.raises(ValueError, match="Expected rank-2"):
            _transpose_2d(arr, "bad")

    def test_rank3_raises(self):
        arr = np.zeros((2, 3, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected rank-2"):
            _transpose_2d(arr, "bad")

    def test_contiguous_float32(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
        result = _transpose_2d(arr, "test")
        assert result.dtype == np.float32
        assert result.flags["C_CONTIGUOUS"]


class TestExpandKvProjection:
    def test_no_expansion_needed(self):
        """When kv_hidden == target_hidden, return unchanged."""
        arr = np.random.randn(64, 64).astype(np.float32)
        result = _expand_kv_projection(
            arr, hidden=64, kv_hidden=64, target_hidden=64,
            num_heads=8, num_kv_heads=8)
        np.testing.assert_array_equal(result, arr)

    def test_gqa_expansion_8q_2kv(self):
        """GQA: 8 query heads, 2 KV heads => group_size=4."""
        hidden = 64
        num_heads = 8
        num_kv_heads = 2
        head_dim = 8  # 64 / 8
        kv_hidden = num_kv_heads * head_dim  # 16
        target_hidden = num_heads * head_dim  # 64

        # Create KV projection [hidden, kv_hidden]
        rng = np.random.RandomState(42)
        transposed = rng.randn(hidden, kv_hidden).astype(np.float32)

        result = _expand_kv_projection(
            transposed, hidden, kv_hidden, target_hidden,
            num_heads, num_kv_heads)

        assert result.shape == (hidden, target_hidden)

        # Verify: heads 0-3 should share KV head 0, heads 4-7 share KV head 1
        for row in range(hidden):
            for qh in range(num_heads):
                kvh = qh // 4  # group_size = 4
                for d in range(head_dim):
                    assert result[row, qh * head_dim + d] == \
                        transposed[row, kvh * head_dim + d]

    def test_gqa_expansion_4q_4kv_noop(self):
        """When num_heads == num_kv_heads (MHA), no expansion."""
        hidden = 32
        num_heads = 4
        num_kv_heads = 4
        head_dim = 8
        kv_hidden = 32
        target_hidden = 32

        arr = np.random.randn(hidden, kv_hidden).astype(np.float32)
        result = _expand_kv_projection(
            arr, hidden, kv_hidden, target_hidden, num_heads, num_kv_heads)
        np.testing.assert_array_equal(result, arr)


class TestRepeatHeadNorm:
    def test_basic(self):
        norm = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = _repeat_head_norm(norm, num_heads=3)
        expected = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
                            dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.float32

    def test_single_head(self):
        norm = np.array([5.0, 6.0], dtype=np.float32)
        result = _repeat_head_norm(norm, num_heads=1)
        np.testing.assert_array_equal(result, norm)


class TestLoadStandardWeights:
    """Test full weight loading with synthetic safetensors."""

    def _create_model_dir(self, tmp_path: Path, config: dict,
                          num_layers: int = 2,
                          hidden: int = 16,
                          vocab: int = 32,
                          num_heads: int = 4,
                          num_kv_heads: int = 4,
                          mlp_size: int = 32) -> Path:
        """Create a minimal model directory with safetensors."""
        from safetensors.numpy import save_file

        (tmp_path / "config.json").write_text(json.dumps(config))

        head_dim = hidden // num_heads
        kv_hidden = num_kv_heads * head_dim
        tensors = {}

        # Embedding
        tensors["model.embed_tokens.weight"] = np.random.randn(
            vocab, hidden).astype(np.float32)

        for i in range(num_layers):
            prefix = f"model.layers.{i}"
            tensors[f"{prefix}.input_layernorm.weight"] = np.random.randn(
                hidden).astype(np.float32)
            tensors[f"{prefix}.post_attention_layernorm.weight"] = \
                np.random.randn(hidden).astype(np.float32)
            tensors[f"{prefix}.self_attn.q_proj.weight"] = np.random.randn(
                hidden, hidden).astype(np.float32)
            tensors[f"{prefix}.self_attn.k_proj.weight"] = np.random.randn(
                kv_hidden, hidden).astype(np.float32)
            tensors[f"{prefix}.self_attn.v_proj.weight"] = np.random.randn(
                kv_hidden, hidden).astype(np.float32)
            tensors[f"{prefix}.self_attn.o_proj.weight"] = np.random.randn(
                hidden, hidden).astype(np.float32)
            tensors[f"{prefix}.mlp.gate_proj.weight"] = np.random.randn(
                mlp_size, hidden).astype(np.float32)
            tensors[f"{prefix}.mlp.up_proj.weight"] = np.random.randn(
                mlp_size, hidden).astype(np.float32)
            tensors[f"{prefix}.mlp.down_proj.weight"] = np.random.randn(
                hidden, mlp_size).astype(np.float32)

        tensors["model.norm.weight"] = np.random.randn(hidden).astype(np.float32)
        tensors["lm_head.weight"] = np.random.randn(vocab, hidden).astype(np.float32)

        save_file(tensors, str(tmp_path / "model.safetensors"))
        return tmp_path

    def test_basic_loading(self, tmp_path):
        """Load standard weights from a minimal model dir."""
        config = {
            "model_type": "qwen3",
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
        }
        model_dir = self._create_model_dir(tmp_path, config)
        cfg = ModelConfig.from_dir(model_dir)
        weights = load_standard_weights(model_dir, cfg)

        # Check all expected keys exist
        assert "embedding" in weights
        assert weights["embedding"].shape == (32, 16)
        assert "layer.0.input_norm" in weights
        assert "layer.0.w_q" in weights
        assert "layer.0.w_k" in weights
        assert "layer.0.w_v" in weights
        assert "layer.0.w_o" in weights
        assert "layer.0.w_gate" in weights
        assert "layer.0.w_up" in weights
        assert "layer.0.w_down" in weights
        assert "layer.1.input_norm" in weights
        assert "final_norm" in weights
        assert "w_out" in weights

    def test_transpose_applied(self, tmp_path):
        """Verify projections are transposed from [out, in] to [in, out]."""
        config = {
            "model_type": "qwen3",
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
        }
        model_dir = self._create_model_dir(tmp_path, config, num_layers=1)
        cfg = ModelConfig.from_dir(model_dir)
        weights = load_standard_weights(model_dir, cfg)

        # w_q should be [hidden, attention_size] = [16, 16] (transposed from [16, 16])
        assert weights["layer.0.w_q"].shape == (16, 16)
        # w_o should be [attention_size, hidden] = [16, 16]
        assert weights["layer.0.w_o"].shape == (16, 16)
        # w_gate: [hidden, mlp_size] = [16, 32]
        assert weights["layer.0.w_gate"].shape == (16, 32)
        # w_out: [hidden, vocab] = [16, 32]
        assert weights["w_out"].shape == (16, 32)

    def test_gqa_kv_expansion(self, tmp_path):
        """Verify GQA K/V projections are expanded."""
        config = {
            "model_type": "qwen3",
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,  # GQA
        }
        model_dir = self._create_model_dir(
            tmp_path, config, num_layers=1, num_kv_heads=2)
        cfg = ModelConfig.from_dir(model_dir)
        weights = load_standard_weights(model_dir, cfg)

        # K and V should be expanded from kv_hidden=8 to attention_size=16
        assert weights["layer.0.w_k"].shape == (16, 16)
        assert weights["layer.0.w_v"].shape == (16, 16)

    def test_tied_embeddings_fallback(self, tmp_path):
        """When lm_head.weight is missing, w_out = transposed embedding."""
        from safetensors.numpy import save_file

        config = {
            "model_type": "qwen3",
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "tie_word_embeddings": True,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        tensors = {}
        embedding = np.random.randn(32, 16).astype(np.float32)
        tensors["model.embed_tokens.weight"] = embedding

        prefix = "model.layers.0"
        tensors[f"{prefix}.input_layernorm.weight"] = np.ones(16, dtype=np.float32)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = np.ones(16, dtype=np.float32)
        tensors[f"{prefix}.self_attn.q_proj.weight"] = np.eye(16, dtype=np.float32)
        tensors[f"{prefix}.self_attn.k_proj.weight"] = np.eye(16, dtype=np.float32)
        tensors[f"{prefix}.self_attn.v_proj.weight"] = np.eye(16, dtype=np.float32)
        tensors[f"{prefix}.self_attn.o_proj.weight"] = np.eye(16, dtype=np.float32)
        tensors[f"{prefix}.mlp.gate_proj.weight"] = np.random.randn(32, 16).astype(np.float32)
        tensors[f"{prefix}.mlp.up_proj.weight"] = np.random.randn(32, 16).astype(np.float32)
        tensors[f"{prefix}.mlp.down_proj.weight"] = np.random.randn(16, 32).astype(np.float32)
        tensors["model.norm.weight"] = np.ones(16, dtype=np.float32)
        # No lm_head.weight!

        save_file(tensors, str(tmp_path / "model.safetensors"))

        cfg = ModelConfig.from_dir(tmp_path)
        weights = load_standard_weights(tmp_path, cfg)

        # w_out should be transposed embedding: [16, 32]
        assert weights["w_out"].shape == (16, 32)
        np.testing.assert_allclose(
            weights["w_out"], embedding.T, atol=1e-6)

    def test_metadata_keys(self, tmp_path):
        """Verify _attention_size and _mlp_size metadata."""
        config = {
            "model_type": "qwen3",
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
        }
        model_dir = self._create_model_dir(tmp_path, config, num_layers=1)
        cfg = ModelConfig.from_dir(model_dir)
        weights = load_standard_weights(model_dir, cfg)

        assert weights["_attention_size"] == 16
        assert weights["_mlp_size"] == 32
