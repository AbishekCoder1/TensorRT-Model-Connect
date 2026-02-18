"""Tests for config.py — ModelConfig parsing from HF config.json."""

from __future__ import annotations

import json
import pytest

pytest.importorskip("trtf_build", reason="trtf_build requires tensorrt")
from trtf_build.config import ModelConfig


class TestModelConfigFromJson:
    def test_qwen3(self):
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "vocab_size": 151936,
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
        }))
        assert cfg.model_type == "qwen3"
        assert cfg.vocab_size == 151936
        assert cfg.hidden_size == 1024
        assert cfg.intermediate_size == 3072
        assert cfg.num_hidden_layers == 28
        assert cfg.num_attention_heads == 16
        assert cfg.num_key_value_heads == 4
        assert cfg.rms_norm_eps == 1e-6
        assert cfg.rope_theta == 1000000.0

    def test_llama(self):
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 32000,
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "num_hidden_layers": 22,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-5,
        }))
        assert cfg.model_type == "llama"
        assert cfg.hidden_size == 2048
        assert cfg.num_key_value_heads == 4
        assert cfg.rms_norm_eps == 1e-5

    def test_gpt2_nonstandard_keys(self):
        """GPT-2 uses n_embd, n_head, n_layer, n_inner."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "gpt2",
            "n_embd": 768,
            "n_head": 12,
            "n_layer": 12,
            "n_inner": 3072,
            "vocab_size": 50257,
            "layer_norm_epsilon": 1e-5,
        }))
        assert cfg.hidden_size == 768
        assert cfg.num_attention_heads == 12
        assert cfg.num_hidden_layers == 12
        assert cfg.intermediate_size == 3072
        assert cfg.rms_norm_eps == 1e-5

    def test_bloom_nonstandard_keys(self):
        """BLOOM uses d_model, attention_heads, num_layers."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "bloom",
            "d_model": 2560,
            "attention_heads": 32,
            "num_layers": 30,
            "vocab_size": 250880,
            "layer_norm_eps": 1e-5,
        }))
        assert cfg.hidden_size == 2560
        assert cfg.num_attention_heads == 32
        assert cfg.num_hidden_layers == 30
        assert cfg.rms_norm_eps == 1e-5
        # intermediate_size fallback: hidden * 4
        assert cfg.intermediate_size == 2560 * 4

    def test_opt_layer_norm_eps(self):
        """OPT uses layer_norm_eps (no 'ilon' suffix)."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "opt",
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "ffn_dim": 3072,
            "vocab_size": 50272,
            "layer_norm_eps": 1e-5,
        }))
        assert cfg.rms_norm_eps == 1e-5
        assert cfg.intermediate_size == 3072

    def test_falcon_norm_epsilon(self):
        """Falcon uses norm_epsilon."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "falcon",
            "hidden_size": 4544,
            "num_attention_heads": 71,
            "num_hidden_layers": 32,
            "vocab_size": 65024,
            "norm_epsilon": 1e-5,
        }))
        assert cfg.rms_norm_eps == 1e-5


class TestHeadDim:
    def test_computed(self):
        cfg = ModelConfig(hidden_size=1024, num_attention_heads=16)
        assert cfg.head_dim == 64

    def test_explicit_override(self):
        cfg = ModelConfig(hidden_size=1024, num_attention_heads=16, _head_dim=128)
        assert cfg.head_dim == 128

    def test_zero_heads(self):
        cfg = ModelConfig(hidden_size=1024, num_attention_heads=0)
        assert cfg.head_dim == 0


class TestAttentionSize:
    def test_basic(self):
        cfg = ModelConfig(hidden_size=1024, num_attention_heads=16)
        assert cfg.attention_size == 1024  # 16 * 64

    def test_with_explicit_head_dim(self):
        cfg = ModelConfig(
            hidden_size=1024, num_attention_heads=16, _head_dim=128)
        assert cfg.attention_size == 2048  # 16 * 128


class TestFromDir:
    def test_from_dir(self, tmp_path):
        config = {
            "model_type": "test",
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_hidden_layers": 6,
            "vocab_size": 10000,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        cfg = ModelConfig.from_dir(tmp_path)
        assert cfg.model_type == "test"
        assert cfg.hidden_size == 512


class TestEdgeCases:
    def test_missing_keys_defaults(self):
        cfg = ModelConfig.from_json("{}")
        assert cfg.model_type == ""
        assert cfg.hidden_size == 0
        assert cfg.num_attention_heads == 1  # fallback
        assert cfg.rms_norm_eps == 1e-5  # fallback

    def test_tie_word_embeddings(self):
        cfg = ModelConfig.from_json(json.dumps({
            "tie_word_embeddings": True,
            "hidden_size": 768,
            "num_attention_heads": 12,
        }))
        assert cfg.tie_word_embeddings is True

    def test_max_position_embeddings_n_positions(self):
        """Some models use n_positions instead of max_position_embeddings."""
        cfg = ModelConfig.from_json(json.dumps({
            "hidden_size": 768,
            "num_attention_heads": 12,
            "n_positions": 1024,
        }))
        assert cfg.max_position_embeddings == 1024

    def test_hidden_act(self):
        cfg = ModelConfig.from_json(json.dumps({
            "hidden_size": 768,
            "num_attention_heads": 12,
            "hidden_act": "silu",
        }))
        assert cfg.hidden_act == "silu"

    def test_activation_function_fallback(self):
        """Some models use activation_function instead of hidden_act."""
        cfg = ModelConfig.from_json(json.dumps({
            "hidden_size": 768,
            "num_attention_heads": 12,
            "activation_function": "gelu_new",
        }))
        assert cfg.hidden_act == "gelu_new"

    def test_rope_theta_from_rope_parameters(self):
        """Llama-3.1 variants store rope_theta inside rope_parameters."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "llama",
            "hidden_size": 3072,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 128256,
            "rope_parameters": {
                "rope_type": "llama3",
                "rope_theta": 500000.0,
            },
        }))
        assert cfg.rope_theta == 500000.0

    def test_rope_theta_top_level_takes_precedence(self):
        """Top-level rope_theta takes precedence over rope_parameters."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "llama",
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "rope_theta": 1000000.0,
            "rope_parameters": {
                "rope_theta": 500000.0,
            },
        }))
        assert cfg.rope_theta == 1000000.0

    def test_rope_theta_default_no_rope_parameters(self):
        """Default rope_theta when neither top-level nor rope_parameters present."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "llama",
            "hidden_size": 1024,
            "num_attention_heads": 16,
        }))
        assert cfg.rope_theta == 10000.0

    def test_raw_dict_preserved(self):
        raw = {
            "model_type": "test",
            "hidden_size": 768,
            "num_attention_heads": 12,
            "custom_field": "custom_value",
        }
        cfg = ModelConfig.from_json(json.dumps(raw))
        assert cfg.raw["custom_field"] == "custom_value"
