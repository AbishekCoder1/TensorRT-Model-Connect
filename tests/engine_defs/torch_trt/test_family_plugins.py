"""Tests for Torch-TRT family plugin auto-discovery and contracts."""

from __future__ import annotations

import pytest

try:
    from tensorrt_model_connect.engine_defs.torch_trt.families import find_plugin, ALL_PLUGINS
except ImportError:
    pytest.skip("tensorrt_model_connect not importable", allow_module_level=True)


class TestPluginDiscovery:
    def test_at_least_one_plugin(self):
        assert len(ALL_PLUGINS) >= 1, "Expected at least the qwen plugin"

    def test_all_plugins_have_name(self):
        for p in ALL_PLUGINS:
            assert hasattr(p, "name"), f"Plugin {p} missing 'name' attribute"
            assert isinstance(p.name, str)
            assert len(p.name) > 0

    def test_all_plugins_have_matches(self):
        for p in ALL_PLUGINS:
            assert callable(getattr(p, "matches", None)), \
                f"Plugin {p.name} missing 'matches()' method"

    def test_all_plugins_have_load_model_or_build_components(self):
        for p in ALL_PLUGINS:
            has_load = callable(getattr(p, "load_model", None))
            has_build = callable(getattr(p, "build_components", None))
            assert has_load or has_build, \
                f"Plugin {p.name} needs 'load_model()' or 'build_components()'"

    def test_all_plugins_have_get_export_args_or_build_components(self):
        for p in ALL_PLUGINS:
            has_args = callable(getattr(p, "get_export_args", None))
            has_build = callable(getattr(p, "build_components", None))
            assert has_args or has_build, \
                f"Plugin {p.name} needs 'get_export_args()' or 'build_components()'"


class TestQwenPlugin:
    def test_find_qwen3(self):
        p = find_plugin("qwen3")
        assert p is not None
        assert p.name == "qwen"

    def test_find_qwen2(self):
        p = find_plugin("qwen2")
        assert p is not None
        assert p.name == "qwen"

    def test_find_qwq(self):
        p = find_plugin("qwq")
        assert p is not None
        assert p.name == "qwen"

    def test_reject_qwen_vl(self):
        assert find_plugin("qwen2_vl") is None

    def test_reject_qwen_moe(self):
        assert find_plugin("qwen2_moe") is None

    def test_reject_unknown(self):
        assert find_plugin("totally_unknown_model") is None


class TestPixArtPlugin:
    def test_find_pixart(self):
        p = find_plugin("pixart")
        assert p is not None
        assert p.name == "pixart"

    def test_find_pixart_sigma_pipeline(self):
        p = find_plugin("PixArtSigmaPipeline")
        assert p is not None
        assert p.name == "pixart"

    def test_find_pixart_alpha_pipeline(self):
        p = find_plugin("PixArtAlphaPipeline")
        assert p is not None
        assert p.name == "pixart"

    def test_pixart_runtime_strategy(self):
        p = find_plugin("pixart")
        assert p.runtime_strategy == "diffusion"
        assert p.bundle_runtime_strategy == "torchtrt_diffusion"

    def test_pixart_has_build_components(self):
        p = find_plugin("pixart")
        assert callable(getattr(p, "build_components", None))

    def test_pixart_load_model_raises(self):
        """PixArt uses build_components, not load_model."""
        p = find_plugin("pixart")
        with pytest.raises(NotImplementedError):
            p.load_model("dummy", None, 256)

    def test_pixart_get_export_args_raises(self):
        """PixArt uses build_components, not get_export_args."""
        p = find_plugin("pixart")
        with pytest.raises(NotImplementedError):
            p.get_export_args(None, None, 256)
