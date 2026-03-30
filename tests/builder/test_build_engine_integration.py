"""Integration tests for family plugin build_engine methods.

These tests exercise the full build_engine pipeline for each plugin with
tiny synthetic models. Each test:
1. Creates synthetic safetensors + config.json
2. Calls plugin.load_weights()
3. Calls plugin.build_engine()
4. Verifies the returned engine plan is valid bytes

Requires TRT + GPU. Tests are marked with requires_trt.

Trace IDs: IT-BUILD-ENGINE
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

try:
    from safetensors.numpy import save_file
    from trtf_build.config import ModelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("trtf_build requires tensorrt", allow_module_level=True)


def _trt_available() -> bool:
    try:
        import tensorrt as trt  # noqa: F401
        try:
            from cuda.bindings import runtime as cudart  # noqa: F401
        except ImportError:
            from cuda import cudart  # type: ignore[no-redef]  # noqa: F401
        return True
    except ImportError:
        return False

requires_trt = pytest.mark.skipif(
    not _trt_available(), reason="TensorRT + CUDA not available"
)

RNG = np.random.RandomState(789)


def _rand(*shape: int) -> np.ndarray:
    return RNG.randn(*shape).astype(np.float32)


def _write_config(model_dir: Path, config: dict) -> None:
    (model_dir / "config.json").write_text(json.dumps(config))


def _write_safetensors(model_dir: Path, tensors: dict[str, np.ndarray],
                       filename: str = "model.safetensors") -> None:
    save_file(tensors, str(model_dir / filename))


# =========================================================================
# OLMo-2 build_engine — post-norm decoder with QK norm
# =========================================================================

@requires_trt
class TestOlmo2BuildEngine:
    """Test OLMo-2 build_engine produces valid TRT engine plan."""

    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, MLP = 32, 16, 1, 4, 4, 32

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, kv_heads, mlp):
        head_dim = hidden // heads
        kv_hidden = kv_heads * head_dim
        t = {}
        t["model.embed_tokens.weight"] = _rand(vocab, hidden)
        for i in range(layers):
            p = f"model.layers.{i}"
            t[f"{p}.post_attention_layernorm.weight"] = _rand(hidden)
            t[f"{p}.post_feedforward_layernorm.weight"] = _rand(hidden)
            t[f"{p}.self_attn.q_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.self_attn.k_proj.weight"] = _rand(kv_hidden, hidden)
            t[f"{p}.self_attn.v_proj.weight"] = _rand(kv_hidden, hidden)
            t[f"{p}.self_attn.o_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.self_attn.q_norm.weight"] = _rand(hidden)
            t[f"{p}.self_attn.k_norm.weight"] = _rand(kv_hidden)
            t[f"{p}.mlp.gate_proj.weight"] = _rand(mlp, hidden)
            t[f"{p}.mlp.up_proj.weight"] = _rand(mlp, hidden)
            t[f"{p}.mlp.down_proj.weight"] = _rand(hidden, mlp)
        t["model.norm.weight"] = _rand(hidden)
        t["lm_head.weight"] = _rand(vocab, hidden)
        return t

    def test_build_engine_returns_bytes(self, tmp_path):
        from trtf_build.families.olmo2 import plugin

        config = {
            "model_type": "olmo2",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "intermediate_size": self.MLP,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.KV_HEADS, self.MLP)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        engine = plugin.build_engine(cfg, weights, max_cache_length=32, verbose=False)

        assert isinstance(engine, bytes)
        assert len(engine) > 0


# =========================================================================
# ModernBERT build_engine — encoder-only with fused QKV and GeGLU
# =========================================================================

@requires_trt
class TestModernbertBuildEngine:
    """Test ModernBERT build_engine produces valid TRT engine plan."""

    VOCAB, HIDDEN, LAYERS, INTERMEDIATE = 32, 16, 2, 32

    @staticmethod
    def _make_tensors(vocab, hidden, layers, intermediate):
        t = {}
        t["model.embeddings.tok_embeddings.weight"] = _rand(vocab, hidden)
        t["model.embeddings.norm.weight"] = _rand(hidden)
        t["model.final_norm.weight"] = _rand(hidden)
        for i in range(layers):
            p = f"model.layers.{i}"
            if i > 0:
                t[f"{p}.attn_norm.weight"] = _rand(hidden)
            t[f"{p}.attn.Wqkv.weight"] = _rand(3 * hidden, hidden)
            t[f"{p}.attn.Wo.weight"] = _rand(hidden, hidden)
            t[f"{p}.mlp_norm.weight"] = _rand(hidden)
            t[f"{p}.mlp.Wi.weight"] = _rand(2 * intermediate, hidden)
            t[f"{p}.mlp.Wo.weight"] = _rand(hidden, intermediate)
        return t

    def test_build_engine_returns_bytes(self, tmp_path):
        from trtf_build.families.modernbert import plugin

        config = {
            "model_type": "modernbert",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": 4,
            "intermediate_size": self.INTERMEDIATE,
            "max_position_embeddings": 64,
        }
        tensors = self._make_tensors(self.VOCAB, self.HIDDEN, self.LAYERS, self.INTERMEDIATE)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        engine = plugin.build_engine(cfg, weights, max_cache_length=32, verbose=False)

        assert isinstance(engine, bytes)
        assert len(engine) > 0


# =========================================================================
# DeBERTa build_engine — encoder-only with disentangled attention
# =========================================================================

@requires_trt
class TestDebertaBuildEngine:
    """Test DeBERTa build_engine produces valid TRT engine plan."""

    VOCAB, HIDDEN, LAYERS, HEADS, INTERMEDIATE, MAX_POS = 32, 16, 1, 4, 32, 64

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, intermediate, max_pos):
        t = {}
        t["deberta.embeddings.word_embeddings.weight"] = _rand(vocab, hidden)
        t["deberta.embeddings.LayerNorm.weight"] = _rand(hidden)
        t["deberta.embeddings.LayerNorm.bias"] = _rand(hidden)
        t["deberta.encoder.rel_embeddings.weight"] = _rand(2 * max_pos, hidden)

        for i in range(layers):
            p = f"deberta.encoder.layer.{i}"
            t[f"{p}.attention.self.in_proj.weight"] = _rand(3 * hidden, hidden)
            t[f"{p}.attention.self.q_bias"] = _rand(hidden)
            t[f"{p}.attention.self.v_bias"] = _rand(hidden)
            t[f"{p}.attention.self.pos_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.self.pos_q_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.self.pos_q_proj.bias"] = _rand(hidden)
            t[f"{p}.attention.output.dense.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.output.dense.bias"] = _rand(hidden)
            t[f"{p}.attention.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.attention.output.LayerNorm.bias"] = _rand(hidden)
            t[f"{p}.intermediate.dense.weight"] = _rand(intermediate, hidden)
            t[f"{p}.intermediate.dense.bias"] = _rand(intermediate)
            t[f"{p}.output.dense.weight"] = _rand(hidden, intermediate)
            t[f"{p}.output.dense.bias"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.bias"] = _rand(hidden)

        return t

    def test_build_engine_returns_bytes(self, tmp_path):
        from trtf_build.families.deberta import plugin

        config = {
            "model_type": "deberta",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "intermediate_size": self.INTERMEDIATE,
            "max_position_embeddings": self.MAX_POS,
            "pos_att_type": "c2p|p2c",
            "max_relative_positions": self.MAX_POS,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.INTERMEDIATE, self.MAX_POS)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        engine = plugin.build_engine(cfg, weights, max_cache_length=32, verbose=False)

        assert isinstance(engine, bytes)
        assert len(engine) > 0


# =========================================================================
# Electra build_engine — encoder-only (BERT-like)
# =========================================================================

@requires_trt
class TestElectraBuildEngine:
    """Test Electra build_engine produces valid TRT engine plan."""

    VOCAB, HIDDEN, LAYERS, HEADS, INTERMEDIATE, MAX_POS = 32, 16, 1, 4, 32, 64

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, intermediate, max_pos):
        t = {}
        t["electra.embeddings.word_embeddings.weight"] = _rand(vocab, hidden)
        t["electra.embeddings.position_embeddings.weight"] = _rand(max_pos, hidden)
        t["electra.embeddings.token_type_embeddings.weight"] = _rand(2, hidden)
        t["electra.embeddings.LayerNorm.weight"] = _rand(hidden)
        t["electra.embeddings.LayerNorm.bias"] = _rand(hidden)

        for i in range(layers):
            p = f"electra.encoder.layer.{i}"
            t[f"{p}.attention.self.query.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.self.query.bias"] = _rand(hidden)
            t[f"{p}.attention.self.key.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.self.key.bias"] = _rand(hidden)
            t[f"{p}.attention.self.value.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.self.value.bias"] = _rand(hidden)
            t[f"{p}.attention.output.dense.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.output.dense.bias"] = _rand(hidden)
            t[f"{p}.attention.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.attention.output.LayerNorm.bias"] = _rand(hidden)
            t[f"{p}.intermediate.dense.weight"] = _rand(intermediate, hidden)
            t[f"{p}.intermediate.dense.bias"] = _rand(intermediate)
            t[f"{p}.output.dense.weight"] = _rand(hidden, intermediate)
            t[f"{p}.output.dense.bias"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.bias"] = _rand(hidden)
        return t

    def test_build_engine_returns_bytes(self, tmp_path):
        from trtf_build.families.electra import plugin

        config = {
            "model_type": "electra",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "intermediate_size": self.INTERMEDIATE,
            "max_position_embeddings": self.MAX_POS,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.INTERMEDIATE, self.MAX_POS)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        engine = plugin.build_engine(cfg, weights, max_cache_length=32, verbose=False)

        assert isinstance(engine, bytes)
        assert len(engine) > 0


# =========================================================================
# FNet build_engine — encoder-only with Fourier transform
# =========================================================================

@requires_trt
class TestFNetBuildEngine:
    """Test FNet build_engine produces valid TRT engine plan."""

    VOCAB, HIDDEN, LAYERS, INTERMEDIATE, MAX_POS = 32, 16, 1, 32, 64

    @staticmethod
    def _make_tensors(vocab, hidden, layers, intermediate, max_pos):
        t = {}
        t["fnet.embeddings.word_embeddings.weight"] = _rand(vocab, hidden)
        t["fnet.embeddings.position_embeddings.weight"] = _rand(max_pos, hidden)
        t["fnet.embeddings.token_type_embeddings.weight"] = _rand(4, hidden)
        t["fnet.embeddings.LayerNorm.weight"] = _rand(hidden)
        t["fnet.embeddings.LayerNorm.bias"] = _rand(hidden)
        t["fnet.embeddings.projection.weight"] = _rand(hidden, hidden)
        t["fnet.embeddings.projection.bias"] = _rand(hidden)

        for i in range(layers):
            p = f"fnet.encoder.layer.{i}"
            t[f"{p}.fourier.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.fourier.output.LayerNorm.bias"] = _rand(hidden)
            t[f"{p}.intermediate.dense.weight"] = _rand(intermediate, hidden)
            t[f"{p}.intermediate.dense.bias"] = _rand(intermediate)
            t[f"{p}.output.dense.weight"] = _rand(hidden, intermediate)
            t[f"{p}.output.dense.bias"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.bias"] = _rand(hidden)
        return t

    def test_build_engine_returns_bytes(self, tmp_path):
        from trtf_build.families.fnet import plugin

        config = {
            "model_type": "fnet",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": 4,
            "intermediate_size": self.INTERMEDIATE,
            "max_position_embeddings": self.MAX_POS,
            "type_vocab_size": 4,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.INTERMEDIATE, self.MAX_POS)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        engine = plugin.build_engine(cfg, weights, max_cache_length=32, verbose=False)

        assert isinstance(engine, bytes)
        assert len(engine) > 0


# =========================================================================
# Albert build_engine — encoder with shared parameters
# =========================================================================

@requires_trt
class TestAlbertBuildEngine:
    """Test Albert build_engine produces valid TRT engine plan."""

    VOCAB, HIDDEN, LAYERS, HEADS, INTERMEDIATE, MAX_POS = 32, 16, 2, 4, 32, 64
    EMBEDDING_SIZE = 8

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, intermediate, max_pos, embed_size):
        t = {}
        t["albert.embeddings.word_embeddings.weight"] = _rand(vocab, embed_size)
        t["albert.embeddings.position_embeddings.weight"] = _rand(max_pos, embed_size)
        t["albert.embeddings.token_type_embeddings.weight"] = _rand(2, embed_size)
        t["albert.embeddings.LayerNorm.weight"] = _rand(embed_size)
        t["albert.embeddings.LayerNorm.bias"] = _rand(embed_size)
        t["albert.encoder.embedding_hidden_mapping_in.weight"] = _rand(hidden, embed_size)
        t["albert.encoder.embedding_hidden_mapping_in.bias"] = _rand(hidden)

        group_prefix = "albert.encoder.albert_layer_groups.0.albert_layers.0"
        t[f"{group_prefix}.attention.query.weight"] = _rand(hidden, hidden)
        t[f"{group_prefix}.attention.query.bias"] = _rand(hidden)
        t[f"{group_prefix}.attention.key.weight"] = _rand(hidden, hidden)
        t[f"{group_prefix}.attention.key.bias"] = _rand(hidden)
        t[f"{group_prefix}.attention.value.weight"] = _rand(hidden, hidden)
        t[f"{group_prefix}.attention.value.bias"] = _rand(hidden)
        t[f"{group_prefix}.attention.dense.weight"] = _rand(hidden, hidden)
        t[f"{group_prefix}.attention.dense.bias"] = _rand(hidden)
        t[f"{group_prefix}.attention.LayerNorm.weight"] = _rand(hidden)
        t[f"{group_prefix}.attention.LayerNorm.bias"] = _rand(hidden)
        t[f"{group_prefix}.ffn.weight"] = _rand(intermediate, hidden)
        t[f"{group_prefix}.ffn.bias"] = _rand(intermediate)
        t[f"{group_prefix}.ffn_output.weight"] = _rand(hidden, intermediate)
        t[f"{group_prefix}.ffn_output.bias"] = _rand(hidden)
        t[f"{group_prefix}.full_layer_layer_norm.weight"] = _rand(hidden)
        t[f"{group_prefix}.full_layer_layer_norm.bias"] = _rand(hidden)
        t["albert.pooler.weight"] = _rand(hidden, hidden)
        t["albert.pooler.bias"] = _rand(hidden)

        return t

    def test_build_engine_returns_bytes(self, tmp_path):
        from trtf_build.families.albert import plugin

        config = {
            "model_type": "albert",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "embedding_size": self.EMBEDDING_SIZE,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "intermediate_size": self.INTERMEDIATE,
            "max_position_embeddings": self.MAX_POS,
            "num_hidden_groups": 1,
            "inner_group_num": 1,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS,
            self.INTERMEDIATE, self.MAX_POS, self.EMBEDDING_SIZE)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        engine = plugin.build_engine(cfg, weights, max_cache_length=32, verbose=False)

        assert isinstance(engine, bytes)
        assert len(engine) > 0


# =========================================================================
# XLNet build_engine — skipped, requires specific weight dim constraints
# that are hard to satisfy with synthetic data
# =========================================================================

@requires_trt
@pytest.mark.skip(reason="XLNet builder has complex weight dim requirements")
class TestXLNetBuildEngine:
    """Test XLNet build_engine produces valid TRT engine plan."""

    VOCAB, HIDDEN, LAYERS, HEADS, INTERMEDIATE = 32, 16, 1, 4, 32

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, intermediate):
        head_dim = hidden // heads
        t = {}
        t["transformer.word_embedding.weight"] = _rand(vocab, hidden)
        for i in range(layers):
            p = f"transformer.layer.{i}"
            for proj in ("q", "k", "v", "o", "r"):
                t[f"{p}.rel_attn.{proj}"] = _rand(hidden, heads, head_dim)
            t[f"{p}.rel_attn.r_w_bias"] = _rand(heads, head_dim)
            t[f"{p}.rel_attn.r_r_bias"] = _rand(heads, head_dim)
            t[f"{p}.rel_attn.r_s_bias"] = _rand(heads, head_dim)
            t[f"{p}.rel_attn.seg_embed"] = _rand(2, heads, head_dim)
            t[f"{p}.rel_attn.layer_norm.weight"] = _rand(hidden)
            t[f"{p}.rel_attn.layer_norm.bias"] = _rand(hidden)
            t[f"{p}.ff.layer_1.weight"] = _rand(intermediate, hidden)
            t[f"{p}.ff.layer_1.bias"] = _rand(intermediate)
            t[f"{p}.ff.layer_2.weight"] = _rand(hidden, intermediate)
            t[f"{p}.ff.layer_2.bias"] = _rand(hidden)
            t[f"{p}.ff.layer_norm.weight"] = _rand(hidden)
            t[f"{p}.ff.layer_norm.bias"] = _rand(hidden)
        return t

    def test_build_engine_returns_bytes(self, tmp_path):
        from trtf_build.families.xlnet import plugin

        config = {
            "model_type": "xlnet",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "d_model": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "d_inner": self.INTERMEDIATE,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.INTERMEDIATE)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        engine = plugin.build_engine(cfg, weights, max_cache_length=32, verbose=False)

        assert isinstance(engine, bytes)
        assert len(engine) > 0
