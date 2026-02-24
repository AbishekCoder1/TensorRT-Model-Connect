"""Tests for Phi-4-multimodal family plugin — weight loading and VL config.

Creates synthetic model directories with the Phi-4-multimodal weight naming
convention (fused QKV + gate_up, vision encoder, image projection), then
verifies the plugin correctly splits/transposes weights and produces valid
VL configuration.

No GPU or TRT needed.
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
    pytest.skip("trtf_build requires safetensors", allow_module_level=True)

# ---- helpers ----

RNG = np.random.RandomState(42)


def _rand(*shape: int) -> np.ndarray:
    return RNG.randn(*shape).astype(np.float32)


def _write_config(model_dir: Path, config: dict) -> None:
    (model_dir / "config.json").write_text(json.dumps(config))


def _write_safetensors(model_dir: Path, tensors: dict[str, np.ndarray],
                       filename: str = "model.safetensors") -> None:
    save_file(tensors, str(model_dir / filename))


# =========================================================================
# Phi-4-multimodal text decoder weights
# =========================================================================

class TestPhi4MultimodalPlugin:
    """Phi-4-multimodal plugin: fused QKV split, gate_up split, VL config."""

    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, MLP = 64, 32, 2, 4, 4, 64

    @classmethod
    def _make_text_tensors(cls):
        """Create synthetic text decoder tensors with fused QKV/gate_up."""
        vocab = cls.VOCAB
        hidden = cls.HIDDEN
        layers = cls.LAYERS
        heads = cls.HEADS
        kv_heads = cls.KV_HEADS
        mlp = cls.MLP
        head_dim = hidden // heads
        q_dim = heads * head_dim
        kv_dim = kv_heads * head_dim

        t = {}
        t["model.embed_tokens.weight"] = _rand(vocab, hidden)

        for i in range(layers):
            p = f"model.layers.{i}"
            t[f"{p}.input_layernorm.weight"] = _rand(hidden)
            t[f"{p}.post_attention_layernorm.weight"] = _rand(hidden)

            # Fused QKV: [q_dim + 2*kv_dim, hidden]
            qkv = _rand(q_dim + 2 * kv_dim, hidden)
            t[f"{p}.self_attn.qkv_proj.weight"] = qkv

            t[f"{p}.self_attn.o_proj.weight"] = _rand(hidden, hidden)

            # Fused gate_up: [2 * mlp, hidden]
            t[f"{p}.mlp.gate_up_proj.weight"] = _rand(2 * mlp, hidden)
            t[f"{p}.mlp.down_proj.weight"] = _rand(hidden, mlp)

        t["model.norm.weight"] = _rand(hidden)
        t["lm_head.weight"] = _rand(vocab, hidden)
        return t

    @classmethod
    def _make_config(cls):
        return {
            "model_type": "phi4mm",
            "vocab_size": cls.VOCAB,
            "hidden_size": cls.HIDDEN,
            "num_hidden_layers": cls.LAYERS,
            "num_attention_heads": cls.HEADS,
            "num_key_value_heads": cls.KV_HEADS,
            "intermediate_size": cls.MLP,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "img_processor": {
                "image_size": 336,
                "patch_size": 14,
                "hidden_size": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "image_token_id": 200011,
            },
        }

    def test_matches(self):
        from trtf_build.families.phi4_multimodal import plugin

        assert plugin.matches("phi4mm")
        assert plugin.matches("phi4_multimodal")
        assert not plugin.matches("phi")
        assert not plugin.matches("phimoe")
        assert not plugin.matches("qwen3")

    def test_load_weights_keys(self, tmp_path):
        from trtf_build.families.phi4_multimodal import plugin

        config = self._make_config()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, self._make_text_tensors())

        mc = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), mc)

        # Check that all expected keys exist
        expected_keys = {"embedding", "final_norm", "w_out",
                         "_attention_size", "_mlp_size"}
        for i in range(self.LAYERS):
            expected_keys.update({
                f"layer.{i}.input_norm",
                f"layer.{i}.post_attn_norm",
                f"layer.{i}.w_q",
                f"layer.{i}.w_k",
                f"layer.{i}.w_v",
                f"layer.{i}.w_o",
                f"layer.{i}.w_gate",
                f"layer.{i}.w_up",
                f"layer.{i}.w_down",
            })

        for key in expected_keys:
            assert key in weights, f"Missing weight key: {key}"

    def test_fused_qkv_split(self, tmp_path):
        """Verify fused QKV is correctly split into Q, K, V."""
        from trtf_build.families.phi4_multimodal import plugin

        config = self._make_config()
        _write_config(tmp_path, config)
        tensors = self._make_text_tensors()
        _write_safetensors(tmp_path, tensors)

        mc = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), mc)

        hidden = self.HIDDEN
        heads = self.HEADS
        head_dim = hidden // heads
        q_dim = heads * head_dim

        # Q should be [hidden, q_dim] after transpose
        assert weights["layer.0.w_q"].shape == (hidden, q_dim)
        # K and V should be [hidden, q_dim] after GQA expansion
        assert weights["layer.0.w_k"].shape == (hidden, q_dim)
        assert weights["layer.0.w_v"].shape == (hidden, q_dim)

    def test_fused_gate_up_split(self, tmp_path):
        """Verify fused gate_up is correctly split into gate and up."""
        from trtf_build.families.phi4_multimodal import plugin

        config = self._make_config()
        _write_config(tmp_path, config)
        tensors = self._make_text_tensors()
        _write_safetensors(tmp_path, tensors)

        mc = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), mc)

        hidden = self.HIDDEN
        mlp = self.MLP

        # gate: [hidden, mlp], up: [hidden, mlp] (transposed)
        assert weights["layer.0.w_gate"].shape == (hidden, mlp)
        assert weights["layer.0.w_up"].shape == (hidden, mlp)
        assert weights["layer.0.w_down"].shape == (mlp, hidden)

    def test_embedding_shape(self, tmp_path):
        from trtf_build.families.phi4_multimodal import plugin

        config = self._make_config()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, self._make_text_tensors())

        mc = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), mc)

        assert weights["embedding"].shape == (self.VOCAB, self.HIDDEN)
        assert weights["w_out"].shape == (self.HIDDEN, self.VOCAB)

    def test_tied_embeddings(self, tmp_path):
        """When lm_head.weight is missing, w_out should be tied to embedding."""
        from trtf_build.families.phi4_multimodal import plugin

        config = self._make_config()
        _write_config(tmp_path, config)
        tensors = self._make_text_tensors()
        del tensors["lm_head.weight"]
        _write_safetensors(tmp_path, tensors)

        mc = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), mc)

        # w_out should be the transpose of the embedding
        embedding = weights["embedding"]
        w_out = weights["w_out"]
        np.testing.assert_allclose(w_out, embedding.T, atol=1e-6)

    def test_runtime_strategy(self):
        from trtf_build.families.phi4_multimodal import plugin
        assert plugin.runtime_strategy == "vision_language"

    def test_embed_input(self):
        from trtf_build.families.phi4_multimodal import plugin
        assert plugin.embed_input is True


# =========================================================================
# VL config
# =========================================================================

class TestPhi4MultimodalVLConfig:
    """Test VL config generation."""

    def test_get_vl_config(self):
        from trtf_build.families.phi4_multimodal import plugin

        config = ModelConfig(
            model_type="phi4mm",
            hidden_size=3072,
            vocab_size=100352,
            raw={
                "model_type": "phi4mm",
                "img_processor": {
                    "image_size": 336,
                    "patch_size": 14,
                    "hidden_size": 1024,
                    "num_attention_heads": 16,
                    "num_hidden_layers": 24,
                    "intermediate_size": 4096,
                    "image_token_id": 200011,
                },
            },
        )

        vl_cfg = plugin.get_vl_config(config)
        assert vl_cfg is not None
        assert vl_cfg["image_token_id"] == 200011
        assert vl_cfg["fixed_image_size"] == 336
        # 336/14 = 24, 24*24 = 576 patches
        assert vl_cfg["num_image_pad_tokens"] == 576
        assert vl_cfg["vision_output_dim"] == 3072
        assert vl_cfg["preprocessor_type"] == "simple_chw"
        assert "{image_pads}" in vl_cfg["vl_prompt_template"]
        assert "{prompt}" in vl_cfg["vl_prompt_template"]

    def test_get_vl_config_no_vision(self):
        from trtf_build.families.phi4_multimodal import plugin

        config = ModelConfig(
            model_type="phi4mm",
            hidden_size=3072,
            raw={"model_type": "phi4mm"},
        )

        vl_cfg = plugin.get_vl_config(config)
        assert vl_cfg is None


# =========================================================================
# GQA expansion
# =========================================================================

class TestPhi4MultimodalGQA:
    """Test GQA expansion when num_kv_heads != num_heads."""

    VOCAB, HIDDEN, LAYERS = 64, 32, 1
    HEADS, KV_HEADS, MLP = 8, 4, 64

    def test_gqa_kv_expansion(self, tmp_path):
        from trtf_build.families.phi4_multimodal import plugin

        hidden = self.HIDDEN
        heads = self.HEADS
        kv_heads = self.KV_HEADS
        head_dim = hidden // heads
        q_dim = heads * head_dim
        kv_dim = kv_heads * head_dim

        config = {
            "model_type": "phi4mm",
            "vocab_size": self.VOCAB,
            "hidden_size": hidden,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "intermediate_size": self.MLP,
        }
        _write_config(tmp_path, config)

        t = {}
        t["model.embed_tokens.weight"] = _rand(self.VOCAB, hidden)
        p = "model.layers.0"
        t[f"{p}.input_layernorm.weight"] = _rand(hidden)
        t[f"{p}.post_attention_layernorm.weight"] = _rand(hidden)
        t[f"{p}.self_attn.qkv_proj.weight"] = _rand(
            q_dim + 2 * kv_dim, hidden)
        t[f"{p}.self_attn.o_proj.weight"] = _rand(hidden, hidden)
        t[f"{p}.mlp.gate_up_proj.weight"] = _rand(
            2 * self.MLP, hidden)
        t[f"{p}.mlp.down_proj.weight"] = _rand(hidden, self.MLP)
        t["model.norm.weight"] = _rand(hidden)
        t["lm_head.weight"] = _rand(self.VOCAB, hidden)

        _write_safetensors(tmp_path, t)

        mc = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), mc)

        # After GQA expansion, K and V should have q_dim columns
        assert weights["layer.0.w_k"].shape == (hidden, q_dim)
        assert weights["layer.0.w_v"].shape == (hidden, q_dim)


# =========================================================================
# Plugin auto-discovery
# =========================================================================

class TestPhi4MultimodalDiscovery:
    """Verify the plugin is auto-discovered by the families package."""

    def test_find_plugin(self):
        from trtf_build.families import find_plugin
        p = find_plugin("phi4mm")
        assert p is not None
        assert p.name == "phi4_multimodal"

    def test_find_plugin_alternate_name(self):
        from trtf_build.families import find_plugin
        p = find_plugin("phi4_multimodal")
        assert p is not None
        assert p.name == "phi4_multimodal"

    def test_no_conflict_with_phi(self):
        """phi4mm should not match the regular phi plugin."""
        from trtf_build.families import find_plugin
        p = find_plugin("phi4mm")
        assert p is not None
        assert p.name == "phi4_multimodal"
