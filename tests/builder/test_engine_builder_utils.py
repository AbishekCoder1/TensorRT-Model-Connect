"""Tests for engine_builder.py utility functions — pure Python, no TRT needed.

Tests _is_hf_model_dir, _detect_tokenizer_add_special_tokens, _resolve_model
(local path), _ensure_tokenizer_json (skips), and build() public API error paths.

Trace: ARCH-ENG-001, UD-ENG-05
Intent: Validate engine builder utility functions including HF model directory detection, tokenizer special-token detection, model resolution, and public API error paths.
Preconditions: trtf_build is importable; no TRT or GPU required.
Postconditions: HF directories are correctly identified, tokenizer special-token flags match HF config, local paths resolve as expected, and invalid inputs raise appropriate errors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from trtf_build.engine_builder import (
        _is_hf_model_dir,
        _detect_tokenizer_add_special_tokens,
        _resolve_model,
        _get_trt_version,
        _get_gpu_name,
        _HF_ALLOW_PATTERNS,
        _ensure_tokenizer_json,
        build_bundle,
    )
    from trtf_build.config import ModelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("trtf_build not importable", allow_module_level=True)


class TestIsHfModelDir:
    """Test _is_hf_model_dir detection."""

    def test_with_config_json(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        assert _is_hf_model_dir(tmp_path)

    def test_with_model_index_json(self, tmp_path):
        (tmp_path / "model_index.json").write_text("{}")
        assert _is_hf_model_dir(tmp_path)

    def test_empty_dir(self, tmp_path):
        assert not _is_hf_model_dir(tmp_path)

    def test_with_other_files(self, tmp_path):
        (tmp_path / "random.txt").write_text("hello")
        assert not _is_hf_model_dir(tmp_path)


class TestDetectTokenizerAddSpecialTokens:
    """Test _detect_tokenizer_add_special_tokens from tokenizer_config.json."""

    def test_explicit_add_bos_token_true(self, tmp_path):
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"add_bos_token": True}))
        assert _detect_tokenizer_add_special_tokens(tmp_path) is True

    def test_explicit_add_bos_token_false(self, tmp_path):
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"add_bos_token": False}))
        assert _detect_tokenizer_add_special_tokens(tmp_path) is False

    def test_no_tokenizer_config(self, tmp_path):
        # No tokenizer_config.json and no transformers — should return False
        result = _detect_tokenizer_add_special_tokens(tmp_path)
        assert result is False

    def test_invalid_json(self, tmp_path):
        (tmp_path / "tokenizer_config.json").write_text("not json{{{")
        # Should not crash, falls through to fallback
        result = _detect_tokenizer_add_special_tokens(tmp_path)
        assert isinstance(result, bool)


class TestResolveModel:
    """Test _resolve_model with local directories."""

    def test_local_dir_with_config(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        assert _resolve_model(str(tmp_path)) == str(tmp_path)

    def test_local_dir_with_model_index(self, tmp_path):
        (tmp_path / "model_index.json").write_text("{}")
        assert _resolve_model(str(tmp_path)) == str(tmp_path)


class TestGetTrtVersion:
    """Test _get_trt_version helper."""

    def test_returns_string(self):
        result = _get_trt_version()
        assert isinstance(result, str)
        # Should either be a version string or "unknown"
        assert result == "unknown" or "." in result


class TestGetGpuName:
    """Test _get_gpu_name helper."""

    def test_returns_string(self):
        result = _get_gpu_name()
        assert isinstance(result, str)


class TestHfAllowPatterns:
    """Test that _HF_ALLOW_PATTERNS contains essential patterns."""

    def test_contains_config(self):
        assert "config.json" in _HF_ALLOW_PATTERNS

    def test_contains_safetensors(self):
        assert "model.safetensors" in _HF_ALLOW_PATTERNS

    def test_contains_tokenizer(self):
        assert "tokenizer.json" in _HF_ALLOW_PATTERNS

    def test_contains_sharded(self):
        assert "model-*.safetensors" in _HF_ALLOW_PATTERNS


class TestEnsureTokenizerJson:
    """Test _ensure_tokenizer_json skips when tokenizer.json exists."""

    def test_skips_if_exists(self, tmp_path):
        (tmp_path / "tokenizer.json").write_text("{}")
        # Should not raise or modify
        _ensure_tokenizer_json(tmp_path)
        assert (tmp_path / "tokenizer.json").read_text() == "{}"


class TestBuildBundleErrors:
    """Test build_bundle error handling."""

    def test_unsupported_model_type(self, tmp_path):
        config = {
            "model_type": "totally_unsupported_model_xyz",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "vocab_size": 32,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        with pytest.raises(ValueError, match="No family plugin"):
            build_bundle(str(tmp_path), str(tmp_path / "out.trtfb"))
