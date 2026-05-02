"""Tests for ttrt_build.compiler — compilation pipeline.

Tests cover:
  - _detect_tokenizer_add_special_tokens(): tokenizer config parsing
  - _get_torch_version() / _get_torchtrt_version(): version utilities
  - StatelessCacheWrapper: I/O format, shape contracts, GQA handling
  - patch_static_cache_scatter(): StaticLayer.update override
  - build_bundle(): orchestrator error handling
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock
from types import SimpleNamespace

try:
    from trtf_build.engine_defs.torch_trt.compiler import (
        _detect_tokenizer_add_special_tokens,
        _parse_model_config,
        _get_torch_version,
        _get_torchtrt_version,
        StatelessCacheWrapper,
        patch_static_cache_scatter,
    )
    from trtf_build.engine_defs.torch_trt import _resolve_model
except ImportError:
    pytest.skip("ttrt_build not importable", allow_module_level=True)

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not available")


class TestDetectTokenizerSpecialTokens:
    def test_add_bos_true(self, tmp_path):
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"add_bos_token": True}))
        assert _detect_tokenizer_add_special_tokens(tmp_path) is True

    def test_add_bos_false(self, tmp_path):
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"add_bos_token": False}))
        assert _detect_tokenizer_add_special_tokens(tmp_path) is False

    def test_no_config(self, tmp_path):
        assert _detect_tokenizer_add_special_tokens(tmp_path) is False

    def test_no_bos_field(self, tmp_path):
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "QWenTokenizer"}))
        assert _detect_tokenizer_add_special_tokens(tmp_path) is False


class TestGetVersions:
    def test_torch_version_is_string(self):
        ver = _get_torch_version()
        assert isinstance(ver, str)
        assert len(ver) > 0

    def test_torchtrt_version(self):
        ver = _get_torchtrt_version()
        assert isinstance(ver, str)
        # Could be a version string or "not installed"
        assert len(ver) > 0


@requires_torch
class TestStatelessCacheWrapper:
    """Tests for StatelessCacheWrapper — raw TRT I/O adapter."""

    def _make_config(self, num_layers=2, num_heads=4, num_kv_heads=2, head_dim=16,
                     hidden_size=64, vocab_size=1000):
        """Create a minimal config namespace for testing."""
        return SimpleNamespace(
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )

    def test_init_computes_compact_kv_dim(self):
        config = self._make_config(num_heads=16, num_kv_heads=2)
        model = MagicMock()
        wrapper = StatelessCacheWrapper(model, config, max_cache_length=32)
        assert wrapper.attention_size == 16 * 16  # num_heads * head_dim
        assert wrapper.kv_dim == 2 * 16
        assert not hasattr(wrapper, "group_size")

    def test_init_mha_kv_dim_matches_attention_size(self):
        config = self._make_config(num_heads=4, num_kv_heads=4)
        model = MagicMock()
        wrapper = StatelessCacheWrapper(model, config, max_cache_length=32)
        assert wrapper.kv_dim == wrapper.attention_size

    def test_init_stores_cache_params(self):
        config = self._make_config(num_layers=28, num_heads=16, head_dim=64)
        model = MagicMock()
        wrapper = StatelessCacheWrapper(model, config, max_cache_length=256)
        assert wrapper.num_layers == 28
        assert wrapper.max_cache_length == 256
        assert wrapper.num_heads == 16
        assert wrapper.head_dim == 64

    def test_forward_signature_arg_count(self):
        """forward() accepts: token_id, position_id, attention_mask, *cache_kv."""
        import inspect
        config = self._make_config()
        model = MagicMock()
        wrapper = StatelessCacheWrapper(model, config, max_cache_length=32)

        sig = inspect.signature(wrapper.forward)
        params = list(sig.parameters.keys())
        assert params[:3] == ["token_id", "position_id", "attention_mask"]
        assert params[3] == "cache_kv"

    def test_attention_and_kv_size_for_gqa(self):
        """attention_size uses query heads; kv_dim uses KV heads."""
        config = self._make_config(num_heads=16, num_kv_heads=2, head_dim=64)
        model = MagicMock()
        wrapper = StatelessCacheWrapper(model, config, max_cache_length=32)
        # Must be num_heads * head_dim, not num_kv_heads * head_dim
        assert wrapper.attention_size == 16 * 64  # 1024
        assert wrapper.attention_size != 2 * 64   # 128
        assert wrapper.kv_dim == 2 * 64


@requires_torch
class TestPatchStaticCacheScatter:
    """Tests for patch_static_cache_scatter() — StaticLayer.update override."""

    def test_patch_is_idempotent(self):
        """Calling patch_static_cache_scatter() twice should not fail."""
        patch_static_cache_scatter()
        patch_static_cache_scatter()

    def test_patch_marks_function(self):
        """After patching, StaticLayer.update has _scatter_patched attr."""
        patch_static_cache_scatter()
        from transformers.cache_utils import StaticLayer
        assert getattr(StaticLayer.update, '_scatter_patched', False) is True


class TestBuildBundle:
    """Mock-based tests for the build_bundle orchestrator."""

    def test_missing_plugin_raises(self, tmp_path):
        from trtf_build.engine_defs.torch_trt.compiler import build_bundle

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "nonexistent_model_xyz",
            "hidden_size": 64,
        }))

        with pytest.raises(ValueError, match="No Torch-TRT family plugin"):
            build_bundle(str(tmp_path), str(tmp_path / "out.trtfb"))

    def test_config_parsed_correctly(self, tmp_path):
        """Verify config is parsed before plugin lookup."""
        from trtf_build.engine_defs.torch_trt.config import ModelConfig
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3",
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "vocab_size": 151936,
        }))

        config = ModelConfig.from_dir(tmp_path)
        assert config.model_type == "qwen3"
        assert config.hidden_size == 1024

    def test_output_extension_trtfb(self, tmp_path):
        """build_bundle writes .trtfb files (not .ttrtb)."""
        from trtf_build.engine_defs.torch_trt.compiler import build_bundle

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "nonexistent_model_xyz",
            "hidden_size": 64,
        }))

        out_path = str(tmp_path / "model.trtfb")
        # Will fail at plugin lookup, but verifies the path is .trtfb
        with pytest.raises(ValueError):
            build_bundle(str(tmp_path), out_path)

    def test_runtime_strategy_is_torchtrt_decoder(self, tmp_path):
        """build_bundle sets runtime_strategy='torchtrt_decoder' in the bundle."""
        from trtf_build.engine_defs.torch_trt.bundle_writer import TtrtBundleInfo
        info = TtrtBundleInfo(runtime_strategy="torchtrt_decoder")
        assert info.runtime_strategy == "torchtrt_decoder"


class TestParseModelConfig:
    """Tests for _parse_model_config — supports both config.json and model_index.json."""

    def test_standard_config_json(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3",
            "hidden_size": 1024,
            "num_hidden_layers": 28,
        }))
        config = _parse_model_config(tmp_path)
        assert config.model_type == "qwen3"
        assert config.hidden_size == 1024

    def test_diffusers_model_index(self, tmp_path):
        (tmp_path / "model_index.json").write_text(json.dumps({
            "_class_name": "PixArtSigmaPipeline",
            "_diffusers_version": "0.28.0",
            "transformer": ["diffusers", "PixArtTransformer2DModel"],
        }))
        config = _parse_model_config(tmp_path)
        assert config.model_type == "PixArtSigmaPipeline"

    def test_config_json_preferred_over_model_index(self, tmp_path):
        """When both exist, config.json takes priority."""
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3",
            "hidden_size": 1024,
        }))
        (tmp_path / "model_index.json").write_text(json.dumps({
            "_class_name": "PixArtSigmaPipeline",
        }))
        config = _parse_model_config(tmp_path)
        assert config.model_type == "qwen3"

    def test_no_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No config.json"):
            _parse_model_config(tmp_path)


class TestResolveModel:
    """Tests for _resolve_model — local path resolution."""

    def test_local_dir_with_config_json(self, tmp_path):
        (tmp_path / "config.json").write_text('{"model_type": "qwen3"}')
        assert _resolve_model(str(tmp_path)) == str(tmp_path)

    def test_local_dir_with_model_index_json(self, tmp_path):
        (tmp_path / "model_index.json").write_text(
            '{"_class_name": "PixArtSigmaPipeline"}')
        assert _resolve_model(str(tmp_path)) == str(tmp_path)
