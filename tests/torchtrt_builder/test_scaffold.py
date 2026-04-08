"""Tests for scripts/new_torchtrt_family.py — plugin scaffolding."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add scripts/ to path
SCRIPTS_DIR = str(Path(__file__).resolve().parents[2] / "scripts")
sys.path.insert(0, SCRIPTS_DIR)

try:
    from new_torchtrt_family import detect_features, generate_plugin
except ImportError:
    pytest.skip("new_torchtrt_family not importable", allow_module_level=True)


class TestDetectFeatures:
    def test_basic_decoder(self):
        cfg = {
            "model_type": "llama",
            "hidden_size": 2048,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 22,
            "vocab_size": 32000,
            "rms_norm_eps": 1e-5,
        }
        f = detect_features(cfg)
        assert f["model_type"] == "llama"
        assert f["is_gqa"] is True
        assert f["is_moe"] is False
        assert f["has_rms_norm_eps"] is True

    def test_moe_detection(self):
        cfg = {
            "model_type": "mixtral",
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_local_experts": 8,
        }
        f = detect_features(cfg)
        assert f["is_moe"] is True

    def test_no_gqa(self):
        cfg = {
            "model_type": "gpt2",
            "num_attention_heads": 12,
        }
        f = detect_features(cfg)
        assert f["is_gqa"] is False

    def test_sliding_window(self):
        cfg = {
            "model_type": "mistral",
            "sliding_window": 4096,
            "num_attention_heads": 32,
        }
        f = detect_features(cfg)
        assert f["sliding_window"] is True

    def test_explicit_head_dim(self):
        cfg = {
            "model_type": "qwen3",
            "head_dim": 128,
            "num_attention_heads": 16,
        }
        f = detect_features(cfg)
        assert f["explicit_head_dim"] is True


class TestGeneratePlugin:
    def test_basic_plugin_structure(self):
        features = {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "is_gqa": True,
            "num_key_value_heads": 8,
            "num_attention_heads": 32,
        }
        source = generate_plugin("llama", "llama", features,
                                  "meta-llama/Llama-3.2-1B")
        assert "class LlamaTorchTrtPlugin:" in source
        assert 'name = "llama"' in source
        assert "def matches(self" in source
        assert "def load_model(" in source
        assert "def get_export_args(" in source
        assert "plugin = LlamaTorchTrtPlugin()" in source
        assert "AutoModelForCausalLM" in source

    def test_match_expression(self):
        features = {"model_type": "phi3"}
        source = generate_plugin("phi", "phi3", features,
                                  "microsoft/Phi-3-mini")
        assert 'startswith("phi3")' in source

    def test_gqa_note(self):
        features = {
            "model_type": "llama",
            "is_gqa": True,
            "num_key_value_heads": 8,
            "num_attention_heads": 32,
        }
        source = generate_plugin("llama", "llama", features, "repo")
        assert "GQA" in source

    def test_moe_note(self):
        features = {"model_type": "mixtral", "is_moe": True}
        source = generate_plugin("mixtral", "mixtral", features, "repo")
        assert "MoE" in source

    def test_no_notes_for_simple(self):
        features = {"model_type": "gpt2"}
        source = generate_plugin("gpt2", "gpt2", features, "repo")
        # Should not have architecture warnings
        assert "NOTE" not in source
