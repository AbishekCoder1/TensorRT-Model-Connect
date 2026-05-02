"""Extended tests batch 2 — M2M-100/NLLB, Marian, and sinusoidal utilities.

Tests load_weights for encoder-decoder translation models with shared embeddings,
sinusoidal positional encodings, and cross-attention layers.

No GPU or TRT needed.

Trace: ARCH-FAM-001, UD-FAM-EXTENDED-2
Intent: Validate M2M-100/NLLB and Marian encoder-decoder plugins with sinusoidal position embeddings
Preconditions: Synthetic safetensors with translation model weight naming and shared embeddings are available
Postconditions: Plugins produce correct sinusoidal embeddings and cross-attention weight keys
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

RNG = np.random.RandomState(456)


def _rand(*shape: int) -> np.ndarray:
    return RNG.randn(*shape).astype(np.float32)


def _write_config(model_dir: Path, config: dict) -> None:
    (model_dir / "config.json").write_text(json.dumps(config))


def _write_safetensors(model_dir: Path, tensors: dict[str, np.ndarray],
                       filename: str = "model.safetensors") -> None:
    save_file(tensors, str(model_dir / filename))


# =========================================================================
# M2M-100 sinusoidal positional embedding utility
# =========================================================================

class TestM2M100SinusoidalPosEmbed:
    """Test the _make_sinusoidal_pos_embed utility function."""

    def test_output_shape(self):
        from trtf_build.families.m2m_100 import _make_sinusoidal_pos_embed
        result = _make_sinusoidal_pos_embed(10, 16)
        assert result.shape == (10, 16)
        assert result.dtype == np.float32

    def test_padding_idx_zeroed(self):
        from trtf_build.families.m2m_100 import _make_sinusoidal_pos_embed
        result = _make_sinusoidal_pos_embed(10, 16, padding_idx=1)
        np.testing.assert_array_equal(result[1], np.zeros(16))

    def test_first_position_pattern(self):
        from trtf_build.families.m2m_100 import _make_sinusoidal_pos_embed
        result = _make_sinusoidal_pos_embed(10, 16, padding_idx=None)
        # Position 0 should have specific sin/cos pattern
        assert result[0, 0] == pytest.approx(0.0, abs=1e-6)  # sin(0) = 0

    def test_odd_dimension(self):
        from trtf_build.families.m2m_100 import _make_sinusoidal_pos_embed
        result = _make_sinusoidal_pos_embed(10, 17)
        assert result.shape == (10, 17)


# =========================================================================
# M2M-100 — encoder-decoder multilingual translation
# =========================================================================

class TestM2M100Plugin:
    """M2M-100 plugin: shared embedding, sinusoidal pos, encoder+decoder."""

    VOCAB, HIDDEN, LAYERS, HEADS, FFN, MAX_POS = 32, 16, 2, 4, 32, 64

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, ffn):
        t = {}
        t["model.shared.weight"] = _rand(vocab, hidden)

        for i in range(layers):
            pfx = f"model.encoder.layers.{i}"
            for proj in ("q", "k", "v"):
                t[f"{pfx}.self_attn.{proj}_proj.weight"] = _rand(hidden, hidden)
                t[f"{pfx}.self_attn.{proj}_proj.bias"] = _rand(hidden)
            t[f"{pfx}.self_attn.out_proj.weight"] = _rand(hidden, hidden)
            t[f"{pfx}.self_attn.out_proj.bias"] = _rand(hidden)
            t[f"{pfx}.self_attn_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.self_attn_layer_norm.bias"] = _rand(hidden)
            t[f"{pfx}.fc1.weight"] = _rand(ffn, hidden)
            t[f"{pfx}.fc1.bias"] = _rand(ffn)
            t[f"{pfx}.fc2.weight"] = _rand(hidden, ffn)
            t[f"{pfx}.fc2.bias"] = _rand(hidden)
            t[f"{pfx}.final_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.final_layer_norm.bias"] = _rand(hidden)

        t["model.encoder.layer_norm.weight"] = _rand(hidden)
        t["model.encoder.layer_norm.bias"] = _rand(hidden)

        for i in range(layers):
            pfx = f"model.decoder.layers.{i}"
            for proj in ("q", "k", "v"):
                t[f"{pfx}.self_attn.{proj}_proj.weight"] = _rand(hidden, hidden)
                t[f"{pfx}.self_attn.{proj}_proj.bias"] = _rand(hidden)
            t[f"{pfx}.self_attn.out_proj.weight"] = _rand(hidden, hidden)
            t[f"{pfx}.self_attn.out_proj.bias"] = _rand(hidden)
            t[f"{pfx}.self_attn_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.self_attn_layer_norm.bias"] = _rand(hidden)
            for proj in ("q", "k", "v"):
                t[f"{pfx}.encoder_attn.{proj}_proj.weight"] = _rand(hidden, hidden)
                t[f"{pfx}.encoder_attn.{proj}_proj.bias"] = _rand(hidden)
            t[f"{pfx}.encoder_attn.out_proj.weight"] = _rand(hidden, hidden)
            t[f"{pfx}.encoder_attn.out_proj.bias"] = _rand(hidden)
            t[f"{pfx}.encoder_attn_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.encoder_attn_layer_norm.bias"] = _rand(hidden)
            t[f"{pfx}.fc1.weight"] = _rand(ffn, hidden)
            t[f"{pfx}.fc1.bias"] = _rand(ffn)
            t[f"{pfx}.fc2.weight"] = _rand(hidden, ffn)
            t[f"{pfx}.fc2.bias"] = _rand(hidden)
            t[f"{pfx}.final_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.final_layer_norm.bias"] = _rand(hidden)

        t["model.decoder.layer_norm.weight"] = _rand(hidden)
        t["model.decoder.layer_norm.bias"] = _rand(hidden)

        return t

    def test_load_weights_keys(self, tmp_path):
        from trtf_build.families.m2m_100 import plugin

        config = {
            "model_type": "m2m_100",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "d_model": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "encoder_layers": self.LAYERS,
            "decoder_layers": self.LAYERS,
            "encoder_attention_heads": self.HEADS,
            "decoder_attention_heads": self.HEADS,
            "encoder_ffn_dim": self.FFN,
            "decoder_ffn_dim": self.FFN,
            "max_position_embeddings": self.MAX_POS,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.FFN)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "shared_embedding" in weights
        assert "sinusoidal_pos_embed" in weights
        assert "enc_final_norm" in weights
        assert weights["sinusoidal_pos_embed"].shape[1] == self.HIDDEN

        for i in range(self.LAYERS):
            for key in ("w_q", "w_k", "w_v", "w_o", "b_q", "b_k", "b_v", "b_o",
                        "attn_norm", "attn_norm_beta", "w_fc1", "b_fc1", "w_fc2",
                        "b_fc2", "ffn_norm", "ffn_norm_beta"):
                assert f"enc_layer.{i}.{key}" in weights, f"Missing enc_layer.{i}.{key}"

    def test_matches(self):
        from trtf_build.families.m2m_100 import plugin
        assert plugin.matches("m2m_100")
        assert plugin.matches("nllb")
        assert not plugin.matches("bart")


# =========================================================================
# Marian — encoder-decoder neural MT
# =========================================================================

class TestMarianPlugin:
    """Marian plugin: encoder-decoder for machine translation."""

    VOCAB, HIDDEN, LAYERS, HEADS, FFN, MAX_POS = 32, 16, 2, 4, 32, 64

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, ffn, max_pos):
        t = {}
        # Marian uses separate encoder/decoder embeddings and position embeddings
        t["model.encoder.embed_tokens.weight"] = _rand(vocab, hidden)
        t["model.encoder.embed_positions.weight"] = _rand(max_pos, hidden)
        t["model.decoder.embed_tokens.weight"] = _rand(vocab, hidden)
        t["model.decoder.embed_positions.weight"] = _rand(max_pos, hidden)

        # Encoder
        for i in range(layers):
            pfx = f"model.encoder.layers.{i}"
            for proj in ("q", "k", "v"):
                t[f"{pfx}.self_attn.{proj}_proj.weight"] = _rand(hidden, hidden)
                t[f"{pfx}.self_attn.{proj}_proj.bias"] = _rand(hidden)
            t[f"{pfx}.self_attn.out_proj.weight"] = _rand(hidden, hidden)
            t[f"{pfx}.self_attn.out_proj.bias"] = _rand(hidden)
            t[f"{pfx}.self_attn_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.self_attn_layer_norm.bias"] = _rand(hidden)
            t[f"{pfx}.fc1.weight"] = _rand(ffn, hidden)
            t[f"{pfx}.fc1.bias"] = _rand(ffn)
            t[f"{pfx}.fc2.weight"] = _rand(hidden, ffn)
            t[f"{pfx}.fc2.bias"] = _rand(hidden)
            t[f"{pfx}.final_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.final_layer_norm.bias"] = _rand(hidden)

        # Decoder
        for i in range(layers):
            pfx = f"model.decoder.layers.{i}"
            for proj in ("q", "k", "v"):
                t[f"{pfx}.self_attn.{proj}_proj.weight"] = _rand(hidden, hidden)
                t[f"{pfx}.self_attn.{proj}_proj.bias"] = _rand(hidden)
            t[f"{pfx}.self_attn.out_proj.weight"] = _rand(hidden, hidden)
            t[f"{pfx}.self_attn.out_proj.bias"] = _rand(hidden)
            t[f"{pfx}.self_attn_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.self_attn_layer_norm.bias"] = _rand(hidden)
            for proj in ("q", "k", "v"):
                t[f"{pfx}.encoder_attn.{proj}_proj.weight"] = _rand(hidden, hidden)
                t[f"{pfx}.encoder_attn.{proj}_proj.bias"] = _rand(hidden)
            t[f"{pfx}.encoder_attn.out_proj.weight"] = _rand(hidden, hidden)
            t[f"{pfx}.encoder_attn.out_proj.bias"] = _rand(hidden)
            t[f"{pfx}.encoder_attn_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.encoder_attn_layer_norm.bias"] = _rand(hidden)
            t[f"{pfx}.fc1.weight"] = _rand(ffn, hidden)
            t[f"{pfx}.fc1.bias"] = _rand(ffn)
            t[f"{pfx}.fc2.weight"] = _rand(hidden, ffn)
            t[f"{pfx}.fc2.bias"] = _rand(hidden)
            t[f"{pfx}.final_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.final_layer_norm.bias"] = _rand(hidden)

        t["final_logits_bias"] = _rand(1, vocab)

        return t

    def test_load_weights_keys(self, tmp_path):
        from trtf_build.families.marian import plugin

        config = {
            "model_type": "marian",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "d_model": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "encoder_layers": self.LAYERS,
            "decoder_layers": self.LAYERS,
            "encoder_attention_heads": self.HEADS,
            "decoder_attention_heads": self.HEADS,
            "encoder_ffn_dim": self.FFN,
            "decoder_ffn_dim": self.FFN,
            "max_position_embeddings": self.MAX_POS,
            "scale_embedding": True,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.FFN, self.MAX_POS)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "enc_embedding" in weights
        assert "enc_pos_embedding" in weights

    def test_matches(self):
        from trtf_build.families.marian import plugin
        assert plugin.matches("marian")
        assert not plugin.matches("bart")


# =========================================================================
# OLMo (v1) — standard decoder
# =========================================================================

class TestOlmoPlugin:
    """OLMo (v1) plugin: standard decoder."""

    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, MLP = 32, 16, 2, 4, 4, 32

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, kv_heads, mlp):
        head_dim = hidden // heads
        kv_hidden = kv_heads * head_dim
        t = {}
        t["model.embed_tokens.weight"] = _rand(vocab, hidden)
        for i in range(layers):
            p = f"model.layers.{i}"
            t[f"{p}.input_layernorm.weight"] = _rand(hidden)
            t[f"{p}.post_attention_layernorm.weight"] = _rand(hidden)
            t[f"{p}.self_attn.q_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.self_attn.k_proj.weight"] = _rand(kv_hidden, hidden)
            t[f"{p}.self_attn.v_proj.weight"] = _rand(kv_hidden, hidden)
            t[f"{p}.self_attn.o_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.mlp.gate_proj.weight"] = _rand(mlp, hidden)
            t[f"{p}.mlp.up_proj.weight"] = _rand(mlp, hidden)
            t[f"{p}.mlp.down_proj.weight"] = _rand(hidden, mlp)
        t["model.norm.weight"] = _rand(hidden)
        t["lm_head.weight"] = _rand(vocab, hidden)
        return t

    def test_load_weights_keys(self, tmp_path):
        from trtf_build.families.olmo import plugin

        config = {
            "model_type": "olmo",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.KV_HEADS, self.MLP)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        assert "final_norm" in weights
        assert "w_out" in weights

    def test_matches(self):
        from trtf_build.families.olmo import plugin
        assert plugin.matches("olmo")
        assert not plugin.matches("olmo2")


# =========================================================================
# StableLM — decoder with QK norm
# =========================================================================

class TestStablelmPlugin:
    """StableLM plugin: standard decoder with partial_rotary_factor."""

    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, MLP = 32, 16, 2, 4, 4, 32

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, kv_heads, mlp):
        head_dim = hidden // heads
        kv_hidden = kv_heads * head_dim
        t = {}
        t["model.embed_tokens.weight"] = _rand(vocab, hidden)
        for i in range(layers):
            p = f"model.layers.{i}"
            t[f"{p}.input_layernorm.weight"] = _rand(hidden)
            t[f"{p}.input_layernorm.bias"] = _rand(hidden)
            t[f"{p}.post_attention_layernorm.weight"] = _rand(hidden)
            t[f"{p}.post_attention_layernorm.bias"] = _rand(hidden)
            t[f"{p}.self_attn.q_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.self_attn.q_proj.bias"] = _rand(hidden)
            t[f"{p}.self_attn.k_proj.weight"] = _rand(kv_hidden, hidden)
            t[f"{p}.self_attn.k_proj.bias"] = _rand(kv_hidden)
            t[f"{p}.self_attn.v_proj.weight"] = _rand(kv_hidden, hidden)
            t[f"{p}.self_attn.v_proj.bias"] = _rand(kv_hidden)
            t[f"{p}.self_attn.o_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.mlp.gate_proj.weight"] = _rand(mlp, hidden)
            t[f"{p}.mlp.up_proj.weight"] = _rand(mlp, hidden)
            t[f"{p}.mlp.down_proj.weight"] = _rand(hidden, mlp)
        t["model.norm.weight"] = _rand(hidden)
        t["model.norm.bias"] = _rand(hidden)
        t["lm_head.weight"] = _rand(vocab, hidden)
        return t

    def test_load_weights_keys(self, tmp_path):
        from trtf_build.families.stablelm import plugin

        config = {
            "model_type": "stablelm",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.KV_HEADS, self.MLP)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        assert "final_norm" in weights

    def test_matches(self):
        from trtf_build.families.stablelm import plugin
        assert plugin.matches("stablelm")
        assert not plugin.matches("llama")


# =========================================================================
# Starcoder2 — decoder
# =========================================================================

class TestStarcoder2Plugin:
    """Starcoder2 plugin: standard decoder for code generation."""

    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, MLP = 32, 16, 2, 4, 2, 32

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, kv_heads, mlp):
        head_dim = hidden // heads
        kv_hidden = kv_heads * head_dim
        t = {}
        t["model.embed_tokens.weight"] = _rand(vocab, hidden)
        for i in range(layers):
            p = f"model.layers.{i}"
            t[f"{p}.input_layernorm.weight"] = _rand(hidden)
            t[f"{p}.input_layernorm.bias"] = _rand(hidden)
            t[f"{p}.post_attention_layernorm.weight"] = _rand(hidden)
            t[f"{p}.post_attention_layernorm.bias"] = _rand(hidden)
            t[f"{p}.self_attn.q_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.self_attn.q_proj.bias"] = _rand(hidden)
            t[f"{p}.self_attn.k_proj.weight"] = _rand(kv_hidden, hidden)
            t[f"{p}.self_attn.k_proj.bias"] = _rand(kv_hidden)
            t[f"{p}.self_attn.v_proj.weight"] = _rand(kv_hidden, hidden)
            t[f"{p}.self_attn.v_proj.bias"] = _rand(kv_hidden)
            t[f"{p}.self_attn.o_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.self_attn.o_proj.bias"] = _rand(hidden)
            t[f"{p}.mlp.c_fc.weight"] = _rand(mlp, hidden)
            t[f"{p}.mlp.c_fc.bias"] = _rand(mlp)
            t[f"{p}.mlp.c_proj.weight"] = _rand(hidden, mlp)
            t[f"{p}.mlp.c_proj.bias"] = _rand(hidden)
        t["model.norm.weight"] = _rand(hidden)
        t["model.norm.bias"] = _rand(hidden)
        t["lm_head.weight"] = _rand(vocab, hidden)
        return t

    def test_load_weights_keys(self, tmp_path):
        from trtf_build.families.starcoder2 import plugin

        config = {
            "model_type": "starcoder2",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.KV_HEADS, self.MLP)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        assert "final_norm" in weights
        kv_hidden = self.KV_HEADS * (self.HIDDEN // self.HEADS)
        assert weights["_kv_attention_size"] == kv_hidden
        assert weights["layer.0.w_k"].shape == (self.HIDDEN, kv_hidden)
        assert weights["layer.0.w_v"].shape == (self.HIDDEN, kv_hidden)
        assert weights["layer.0.k_bias"].shape == (kv_hidden,)
        assert weights["layer.0.v_bias"].shape == (kv_hidden,)

    def test_matches(self):
        from trtf_build.families.starcoder2 import plugin
        assert plugin.matches("starcoder2")
        assert not plugin.matches("gpt2")


# =========================================================================
# Granite — decoder with attention multiplier
# =========================================================================

class TestGranitePlugin:
    """Granite plugin: decoder with attention_multiplier."""

    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, MLP = 32, 16, 2, 4, 4, 32

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, kv_heads, mlp):
        head_dim = hidden // heads
        kv_hidden = kv_heads * head_dim
        t = {}
        t["model.embed_tokens.weight"] = _rand(vocab, hidden)
        for i in range(layers):
            p = f"model.layers.{i}"
            t[f"{p}.input_layernorm.weight"] = _rand(hidden)
            t[f"{p}.post_attention_layernorm.weight"] = _rand(hidden)
            t[f"{p}.self_attn.q_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.self_attn.k_proj.weight"] = _rand(kv_hidden, hidden)
            t[f"{p}.self_attn.v_proj.weight"] = _rand(kv_hidden, hidden)
            t[f"{p}.self_attn.o_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.mlp.gate_proj.weight"] = _rand(mlp, hidden)
            t[f"{p}.mlp.up_proj.weight"] = _rand(mlp, hidden)
            t[f"{p}.mlp.down_proj.weight"] = _rand(hidden, mlp)
        t["model.norm.weight"] = _rand(hidden)
        t["lm_head.weight"] = _rand(vocab, hidden)
        return t

    def test_load_weights_keys(self, tmp_path):
        from trtf_build.families.granite import plugin

        config = {
            "model_type": "granite",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.KV_HEADS, self.MLP)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights

    def test_matches(self):
        from trtf_build.families.granite import plugin
        assert plugin.matches("granite")
        assert plugin.matches("granitemoeshared")
        assert not plugin.matches("llama")


# =========================================================================
# XGLM — decoder with sinusoidal pos
# =========================================================================

class TestXglmPlugin:
    """XGLM plugin: multilingual decoder."""

    VOCAB, HIDDEN, LAYERS, HEADS, MLP = 32, 16, 2, 4, 32

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, mlp):
        t = {}
        t["model.embed_tokens.weight"] = _rand(vocab, hidden)
        for i in range(layers):
            p = f"model.layers.{i}"
            t[f"{p}.self_attn.q_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.self_attn.q_proj.bias"] = _rand(hidden)
            t[f"{p}.self_attn.k_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.self_attn.k_proj.bias"] = _rand(hidden)
            t[f"{p}.self_attn.v_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.self_attn.v_proj.bias"] = _rand(hidden)
            t[f"{p}.self_attn.out_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.self_attn.out_proj.bias"] = _rand(hidden)
            t[f"{p}.self_attn_layer_norm.weight"] = _rand(hidden)
            t[f"{p}.self_attn_layer_norm.bias"] = _rand(hidden)
            t[f"{p}.fc1.weight"] = _rand(mlp, hidden)
            t[f"{p}.fc1.bias"] = _rand(mlp)
            t[f"{p}.fc2.weight"] = _rand(hidden, mlp)
            t[f"{p}.fc2.bias"] = _rand(hidden)
            t[f"{p}.final_layer_norm.weight"] = _rand(hidden)
            t[f"{p}.final_layer_norm.bias"] = _rand(hidden)
        t["model.layer_norm.weight"] = _rand(hidden)
        t["model.layer_norm.bias"] = _rand(hidden)
        t["lm_head.weight"] = _rand(vocab, hidden)
        return t

    def test_load_weights_keys(self, tmp_path):
        from trtf_build.families.xglm import plugin

        config = {
            "model_type": "xglm",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "ffn_dim": self.MLP,
            "max_position_embeddings": 64,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.MLP)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights

    def test_matches(self):
        from trtf_build.families.xglm import plugin
        assert plugin.matches("xglm")
        assert not plugin.matches("gpt2")
