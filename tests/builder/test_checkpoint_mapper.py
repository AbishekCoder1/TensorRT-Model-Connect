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


class TestLoadStandardWeightsExtended:
    """Extended tests for load_standard_weights edge cases."""

    def _create_model_dir(self, tmp_path, config, tensors):
        """Create a model dir with config.json and custom tensors."""
        from safetensors.numpy import save_file

        (tmp_path / "config.json").write_text(json.dumps(config))
        save_file(tensors, str(tmp_path / "model.safetensors"))
        return tmp_path

    def _make_standard_tensors(self, num_layers, hidden, vocab,
                               num_heads, num_kv_heads, mlp_size,
                               include_lm_head=True,
                               include_biases=False):
        """Build the standard set of tensors for a model."""
        head_dim = hidden // num_heads
        kv_hidden = num_kv_heads * head_dim
        rng = np.random.RandomState(42)
        tensors = {}

        tensors["model.embed_tokens.weight"] = rng.randn(
            vocab, hidden).astype(np.float32)

        for i in range(num_layers):
            prefix = f"model.layers.{i}"
            tensors[f"{prefix}.input_layernorm.weight"] = rng.randn(
                hidden).astype(np.float32)
            tensors[f"{prefix}.post_attention_layernorm.weight"] = rng.randn(
                hidden).astype(np.float32)
            tensors[f"{prefix}.self_attn.q_proj.weight"] = rng.randn(
                hidden, hidden).astype(np.float32)
            tensors[f"{prefix}.self_attn.k_proj.weight"] = rng.randn(
                kv_hidden, hidden).astype(np.float32)
            tensors[f"{prefix}.self_attn.v_proj.weight"] = rng.randn(
                kv_hidden, hidden).astype(np.float32)
            tensors[f"{prefix}.self_attn.o_proj.weight"] = rng.randn(
                hidden, hidden).astype(np.float32)
            tensors[f"{prefix}.mlp.gate_proj.weight"] = rng.randn(
                mlp_size, hidden).astype(np.float32)
            tensors[f"{prefix}.mlp.up_proj.weight"] = rng.randn(
                mlp_size, hidden).astype(np.float32)
            tensors[f"{prefix}.mlp.down_proj.weight"] = rng.randn(
                hidden, mlp_size).astype(np.float32)

            if include_biases:
                tensors[f"{prefix}.self_attn.q_proj.bias"] = rng.randn(
                    hidden).astype(np.float32)
                tensors[f"{prefix}.self_attn.k_proj.bias"] = rng.randn(
                    kv_hidden).astype(np.float32)
                tensors[f"{prefix}.self_attn.v_proj.bias"] = rng.randn(
                    kv_hidden).astype(np.float32)

        tensors["model.norm.weight"] = rng.randn(hidden).astype(np.float32)
        if include_lm_head:
            tensors["lm_head.weight"] = rng.randn(
                vocab, hidden).astype(np.float32)

        return tensors

    # --- GQA expansion value correctness ---

    def test_gqa_expansion_values_correct(self, tmp_path):
        """Verify GQA K/V expansion produces correct repeated head values.

        Query heads in the same group should share identical K/V weight columns.
        group_size = num_heads / num_kv_heads = 4, so heads 0-3 share one KV
        head and heads 4-7 share the other.
        """
        hidden = 16
        num_heads = 8
        num_kv_heads = 2
        head_dim = hidden // num_heads  # 2

        config = {
            "model_type": "qwen3",
            "vocab_size": 32,
            "hidden_size": hidden,
            "num_hidden_layers": 1,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
        }
        tensors = self._make_standard_tensors(
            num_layers=1, hidden=hidden, vocab=32,
            num_heads=num_heads, num_kv_heads=num_kv_heads, mlp_size=32)

        model_dir = self._create_model_dir(tmp_path, config, tensors)
        cfg = ModelConfig.from_dir(model_dir)
        weights = load_standard_weights(model_dir, cfg)

        # K and V should be expanded from [hidden, kv_hidden=4] to [hidden, hidden=16]
        assert weights["layer.0.w_k"].shape == (hidden, hidden)
        assert weights["layer.0.w_v"].shape == (hidden, hidden)

        # Verify the expansion pattern: all query heads in the same group
        # share identical weight columns.
        k_weight = weights["layer.0.w_k"]
        group_size = num_heads // num_kv_heads  # 4
        for group in range(num_kv_heads):
            # First head in the group as reference
            ref_start = group * group_size * head_dim
            for member in range(1, group_size):
                member_start = (group * group_size + member) * head_dim
                for d in range(head_dim):
                    np.testing.assert_equal(
                        k_weight[:, member_start + d],
                        k_weight[:, ref_start + d])

    # --- Tied embeddings ---

    def test_tied_embeddings_no_lm_head(self, tmp_path):
        """When lm_head.weight is absent, w_out = transposed embedding."""
        hidden = 16
        vocab = 32
        config = {
            "model_type": "qwen3",
            "vocab_size": vocab,
            "hidden_size": hidden,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "tie_word_embeddings": True,
        }
        tensors = self._make_standard_tensors(
            num_layers=1, hidden=hidden, vocab=vocab,
            num_heads=4, num_kv_heads=4, mlp_size=32,
            include_lm_head=False)

        model_dir = self._create_model_dir(tmp_path, config, tensors)
        cfg = ModelConfig.from_dir(model_dir)
        weights = load_standard_weights(model_dir, cfg)

        # w_out should be transposed embedding: [hidden, vocab]
        assert weights["w_out"].shape == (hidden, vocab)
        embedding = tensors["model.embed_tokens.weight"]
        np.testing.assert_allclose(weights["w_out"], embedding.T, atol=1e-6)

    def test_lm_head_present_ignores_tie(self, tmp_path):
        """When lm_head.weight exists, it is used even if tie_word_embeddings=True."""
        hidden = 16
        vocab = 32
        config = {
            "model_type": "qwen3",
            "vocab_size": vocab,
            "hidden_size": hidden,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "tie_word_embeddings": True,
        }
        tensors = self._make_standard_tensors(
            num_layers=1, hidden=hidden, vocab=vocab,
            num_heads=4, num_kv_heads=4, mlp_size=32,
            include_lm_head=True)

        model_dir = self._create_model_dir(tmp_path, config, tensors)
        cfg = ModelConfig.from_dir(model_dir)
        weights = load_standard_weights(model_dir, cfg)

        # w_out should be transposed lm_head.weight, NOT embedding
        lm_head_raw = tensors["lm_head.weight"]
        np.testing.assert_allclose(weights["w_out"], lm_head_raw.T, atol=1e-6)

    # --- Optional biases ---

    def test_biases_loaded_when_present(self, tmp_path):
        """QKV biases are loaded when present in safetensors."""
        hidden = 16
        config = {
            "model_type": "qwen2",
            "vocab_size": 32,
            "hidden_size": hidden,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
        }
        tensors = self._make_standard_tensors(
            num_layers=1, hidden=hidden, vocab=32,
            num_heads=4, num_kv_heads=4, mlp_size=32,
            include_biases=True)

        model_dir = self._create_model_dir(tmp_path, config, tensors)
        cfg = ModelConfig.from_dir(model_dir)
        weights = load_standard_weights(model_dir, cfg)

        assert "layer.0.q_bias" in weights
        assert "layer.0.k_bias" in weights
        assert "layer.0.v_bias" in weights
        assert weights["layer.0.q_bias"].shape == (hidden,)
        assert weights["layer.0.k_bias"].shape == (hidden,)
        assert weights["layer.0.v_bias"].shape == (hidden,)

    def test_no_biases_when_absent(self, tmp_path):
        """QKV biases are absent from weights when not in safetensors."""
        hidden = 16
        config = {
            "model_type": "qwen3",
            "vocab_size": 32,
            "hidden_size": hidden,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
        }
        tensors = self._make_standard_tensors(
            num_layers=1, hidden=hidden, vocab=32,
            num_heads=4, num_kv_heads=4, mlp_size=32,
            include_biases=False)

        model_dir = self._create_model_dir(tmp_path, config, tensors)
        cfg = ModelConfig.from_dir(model_dir)
        weights = load_standard_weights(model_dir, cfg)

        assert "layer.0.q_bias" not in weights
        assert "layer.0.k_bias" not in weights
        assert "layer.0.v_bias" not in weights

    def test_gqa_bias_expansion(self, tmp_path):
        """K/V biases are GQA-expanded when num_kv_heads < num_heads."""
        hidden = 16
        num_heads = 8
        num_kv_heads = 2
        head_dim = hidden // num_heads  # 2
        kv_hidden = num_kv_heads * head_dim  # 4

        config = {
            "model_type": "qwen2",
            "vocab_size": 32,
            "hidden_size": hidden,
            "num_hidden_layers": 1,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
        }
        tensors = self._make_standard_tensors(
            num_layers=1, hidden=hidden, vocab=32,
            num_heads=num_heads, num_kv_heads=num_kv_heads, mlp_size=32,
            include_biases=True)

        model_dir = self._create_model_dir(tmp_path, config, tensors)
        cfg = ModelConfig.from_dir(model_dir)
        weights = load_standard_weights(model_dir, cfg)

        # q_bias stays at hidden size (no expansion for Q)
        assert weights["layer.0.q_bias"].shape == (hidden,)
        # k_bias and v_bias should be expanded from kv_hidden to hidden
        assert weights["layer.0.k_bias"].shape == (hidden,)
        assert weights["layer.0.v_bias"].shape == (hidden,)

        # Verify expansion pattern: each KV head bias repeated for its group
        k_bias_raw = tensors["model.layers.0.self_attn.k_proj.bias"]
        k_bias_expanded = weights["layer.0.k_bias"]
        group_size = num_heads // num_kv_heads
        for qh in range(num_heads):
            kvh = qh // group_size
            for d in range(head_dim):
                assert k_bias_expanded[qh * head_dim + d] == \
                    k_bias_raw[kvh * head_dim + d]

    # --- Error paths ---

    def test_missing_embed_tokens_raises(self, tmp_path):
        """Missing embed_tokens.weight raises KeyError with descriptive message."""
        from safetensors.numpy import save_file

        config = {
            "model_type": "qwen3",
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        # Create safetensors WITHOUT embed_tokens
        tensors = {
            "model.layers.0.input_layernorm.weight":
                np.ones(16, dtype=np.float32),
        }
        save_file(tensors, str(tmp_path / "model.safetensors"))

        cfg = ModelConfig.from_dir(tmp_path)
        with pytest.raises(KeyError, match="model.embed_tokens.weight"):
            load_standard_weights(tmp_path, cfg)

    def test_wrong_embedding_shape_raises(self, tmp_path):
        """Embedding with wrong shape raises AssertionError."""
        from safetensors.numpy import save_file

        config = {
            "model_type": "qwen3",
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 0,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        # Embedding with wrong vocab dimension
        tensors = {
            "model.embed_tokens.weight":
                np.random.randn(64, 16).astype(np.float32),  # vocab=64 != 32
            "model.norm.weight": np.ones(16, dtype=np.float32),
            "lm_head.weight": np.random.randn(32, 16).astype(np.float32),
        }
        save_file(tensors, str(tmp_path / "model.safetensors"))

        cfg = ModelConfig.from_dir(tmp_path)
        with pytest.raises(AssertionError, match="Embedding shape"):
            load_standard_weights(tmp_path, cfg)

    def test_no_safetensors_raises(self, tmp_path):
        """Model dir without any safetensors/bin files raises FileNotFoundError."""
        config = {
            "model_type": "qwen3",
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        # No safetensors file created

        cfg = ModelConfig.from_dir(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_standard_weights(tmp_path, cfg)

    def test_single_kv_head_expansion(self, tmp_path):
        """GQA with num_heads=8, num_kv_heads=1: KV weights repeated 8x.

        This is the extreme MQA (multi-query attention) case where a single
        KV head is shared across all query heads. The expansion should
        produce output where every query head slice is identical to the
        single source KV head.
        """
        hidden = 16
        num_heads = 8
        num_kv_heads = 1
        head_dim = hidden // num_heads  # 2
        kv_hidden = num_kv_heads * head_dim  # 2

        config = {
            "model_type": "qwen3",
            "vocab_size": 32,
            "hidden_size": hidden,
            "num_hidden_layers": 1,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
        }
        tensors = self._make_standard_tensors(
            num_layers=1, hidden=hidden, vocab=32,
            num_heads=num_heads, num_kv_heads=num_kv_heads, mlp_size=32)

        model_dir = self._create_model_dir(tmp_path, config, tensors)
        cfg = ModelConfig.from_dir(model_dir)
        weights = load_standard_weights(model_dir, cfg)

        # K and V should be expanded from [hidden, kv_hidden=2] to [hidden, hidden=16]
        assert weights["layer.0.w_k"].shape == (hidden, hidden)
        assert weights["layer.0.w_v"].shape == (hidden, hidden)

        # Verify the expansion pattern: all 8 query head slices must be
        # identical to the single source KV head.
        k_weight = weights["layer.0.w_k"]

        # Get the raw K projection before expansion.
        # Raw shape in safetensors is [kv_hidden, hidden] = [2, 16];
        # after transpose it's [hidden, kv_hidden] = [16, 2].
        k_raw_transposed = tensors[
            "model.layers.0.self_attn.k_proj.weight"].T.astype(np.float32)
        assert k_raw_transposed.shape == (hidden, kv_hidden)

        # Each of the 8 head slices (columns [qh*2 : qh*2+2]) should equal
        # the single source head (columns [0:2] of the raw transposed weight).
        for qh in range(num_heads):
            head_slice = k_weight[:, qh * head_dim:(qh + 1) * head_dim]
            np.testing.assert_array_equal(
                head_slice, k_raw_transposed,
                err_msg=f"Query head {qh} K slice differs from source KV head")

        # Same check for V weights.
        v_weight = weights["layer.0.w_v"]
        v_raw_transposed = tensors[
            "model.layers.0.self_attn.v_proj.weight"].T.astype(np.float32)
        for qh in range(num_heads):
            head_slice = v_weight[:, qh * head_dim:(qh + 1) * head_dim]
            np.testing.assert_array_equal(
                head_slice, v_raw_transposed,
                err_msg=f"Query head {qh} V slice differs from source KV head")
