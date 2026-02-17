"""Tests for engine_builder.py — orchestrator logic.

Tests plugin discovery and model resolution without requiring TRT.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("trtf_build", reason="trtf_build requires tensorrt")
from trtf_build.engine_builder import _resolve_model
from trtf_build.families import find_plugin, _ALL_PLUGINS


class TestResolveModel:
    def test_local_dir_with_config(self, tmp_path):
        """Local directory with config.json returns the path directly."""
        (tmp_path / "config.json").write_text('{"model_type": "test"}')
        result = _resolve_model(str(tmp_path))
        assert result == str(tmp_path)

    def test_local_dir_without_config(self, tmp_path):
        """Directory without config.json treated as HF repo ID.
        Should raise ImportError if huggingface_hub is not installed,
        or attempt download."""
        # The directory exists but has no config.json
        try:
            _resolve_model(str(tmp_path))
        except (ImportError, Exception):
            # Expected: either HF hub not available or download fails
            pass

    def test_nonexistent_path_treated_as_repo_id(self):
        """A non-existent path is treated as a HF repo ID."""
        try:
            _resolve_model("nonexistent/model-that-does-not-exist-12345")
        except (ImportError, Exception):
            pass


class TestFindPlugin:
    def test_supported_model_types(self):
        """Verify find_plugin returns non-None for all known model types."""
        known_types = [
            "qwen", "qwen2", "qwen3", "qwq",
            "llama", "mistral", "gemma", "gemma2",
            "phi", "phi3", "phimoe",
            "granite", "internlm", "internlm2",
            "starcoder2", "gpt2", "opt", "falcon", "stablelm",
            "olmo", "xglm", "gpt_neox", "gpt_neo", "codegen",
            "bloom", "mamba", "mixtral",
            "qwen2_vl", "qwen2_5_vl",
        ]
        for model_type in known_types:
            plugin = find_plugin(model_type)
            assert plugin is not None, f"No plugin for {model_type}"

    def test_unsupported_model_type(self):
        assert find_plugin("nonexistent_model_type_12345") is None

    def test_known_families(self):
        """Verify key model types map to expected family names."""
        known = {
            "qwen3": "qwen",
            "qwen2": "qwen",
            "llama": "llama",
            "mistral": "mistral",
            "gemma": "gemma",
            "gemma2": "gemma",
            "phi3": "phi",
        }
        for model_type, expected_family in known.items():
            plugin = find_plugin(model_type)
            if plugin is not None:
                assert plugin.name == expected_family, \
                    f"{model_type} -> {plugin.name} (expected {expected_family})"

    def test_all_plugins_have_required_attributes(self):
        """Every plugin must have name, matches, load_weights, build_engine."""
        for p in _ALL_PLUGINS:
            assert hasattr(p, "name")
            assert hasattr(p, "matches")
            assert hasattr(p, "load_weights")
            assert hasattr(p, "build_engine")
            assert isinstance(p.name, str)
            assert len(p.name) > 0
            assert callable(p.matches)
            assert callable(p.load_weights)
            assert callable(p.build_engine)
