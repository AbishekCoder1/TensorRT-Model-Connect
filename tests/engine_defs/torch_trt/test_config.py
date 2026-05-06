"""Tests for tensorrt_model_connect.config — ModelConfig parsing."""

from __future__ import annotations

import json
import pytest

try:
    from tensorrt_model_connect.engine_defs.torch_trt.config import ModelConfig
except ImportError:
    pytest.skip("tensorrt_model_connect not importable", allow_module_level=True)


class TestModelConfig:
    def test_from_json_qwen3(self):
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "qwen3",
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "intermediate_size": 2816,
            "vocab_size": 151936,
            "head_dim": 64,
        }))
        assert cfg.model_type == "qwen3"
        assert cfg.hidden_size == 1024
        assert cfg.num_hidden_layers == 28
        assert cfg.num_attention_heads == 16
        assert cfg.num_key_value_heads == 8
        assert cfg.vocab_size == 151936
        assert cfg.head_dim == 64

    def test_from_json_defaults(self):
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "test",
            "hidden_size": 256,
            "num_attention_heads": 4,
        }))
        assert cfg.num_key_value_heads == 4  # defaults to num_attention_heads
        assert cfg.head_dim == 64  # 256 / 4

    def test_from_dir(self, tmp_path):
        config = {
            "model_type": "qwen3",
            "hidden_size": 512,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "vocab_size": 32000,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        cfg = ModelConfig.from_dir(tmp_path)
        assert cfg.model_type == "qwen3"
        assert cfg.hidden_size == 512

    def test_missing_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ModelConfig.from_dir(tmp_path)

    def test_raw_dict_preserved(self):
        raw = {"model_type": "test", "hidden_size": 128, "custom_field": "hello"}
        cfg = ModelConfig(raw=raw)
        assert cfg.raw["custom_field"] == "hello"
