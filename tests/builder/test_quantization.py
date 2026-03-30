"""Tests for quantization framework abstractions."""
import json
import numpy as np
import pytest

from trtf_build.quantization import get_format, list_formats, QuantScaleMap, LayerScales
from trtf_build.quantization.formats import QuantFormat


class TestFormatRegistry:
    def test_all_formats_registered(self):
        names = list_formats()
        assert "fp8" in names
        assert "int8_sq" in names
        assert "int4_awq" in names
        assert "nvfp4" in names
        assert "w4a8" in names

    def test_get_format_returns_protocol(self):
        fmt = get_format("fp8")
        assert isinstance(fmt, QuantFormat)
        assert fmt.name == "fp8"

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_format("nonexistent")


class TestScaleMapJsonRoundTrip:
    def test_roundtrip(self):
        original = QuantScaleMap(scales={
            "layer.0.w_q": LayerScales(input_scale=0.042, weight_scale=0.051),
            "layer.1.w_k": LayerScales(input_scale=0.1, weight_scale=0.2, block_size=128),
        })
        restored = QuantScaleMap.from_json(original.to_json())
        assert len(restored.scales) == 2
        assert abs(restored.scales["layer.0.w_q"].input_scale - 0.042) < 1e-6
        assert restored.scales["layer.1.w_k"].block_size == 128

    def test_dynamic_flag(self):
        m = QuantScaleMap(scales={}, dynamic=True)
        restored = QuantScaleMap.from_json(m.to_json())
        assert restored.dynamic is True


class TestQuantFormatProtocol:
    def test_all_formats_have_wrap_matmul(self):
        for name in list_formats():
            fmt = get_format(name)
            assert hasattr(fmt, "wrap_matmul"), f"{name} missing wrap_matmul"

    def test_all_formats_have_wrap_conv2d(self):
        for name in list_formats():
            fmt = get_format(name)
            assert hasattr(fmt, "wrap_conv2d"), f"{name} missing wrap_conv2d"


class TestPreQuantizedCheckpointProvider:
    def test_detect_awq_format(self, tmp_path):
        """AWQ path reached (not NotImplementedError)."""
        from trtf_build.quantization.scale_providers import PreQuantizedCheckpointProvider
        from trtf_build.config import ModelConfig

        config = ModelConfig.from_json(json.dumps({
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
                "zero_point": True,
            }
        }))
        provider = PreQuantizedCheckpointProvider()
        # Should reach AWQ extraction (not NotImplementedError)
        # Will return empty scales since tmp_path has no safetensors,
        # but must NOT raise NotImplementedError
        result = provider.acquire_scales(str(tmp_path), config, get_format("int4_awq"), [])
        assert isinstance(result, QuantScaleMap)
        assert len(result.scales) == 0  # no safetensors files present
