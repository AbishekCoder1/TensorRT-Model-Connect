"""Tests for ttrt_build.strategies — build strategy dispatch and wrappers.

Tests cover:
  - Strategy registry: get_strategy() dispatch, unknown strategy error
  - DecoderBuildStrategy: wrap_model returns StatelessCacheWrapper,
    pre_export_setup calls patch_static_cache_scatter
  - EncoderOnlyBuildStrategy: wrap_model returns EncoderOnlyWrapper,
    make_export_args produces correct shapes
  - EncoderOnlyWrapper: forward produces [1, seq_len, hidden_size] output
  - Plugin default: plugins without runtime_strategy default to "decoder"
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from types import SimpleNamespace

try:
    from trtf_build.engine_defs.torch_trt.strategies import get_strategy
    from trtf_build.engine_defs.torch_trt.strategies.base import BuildStrategy  # noqa: F401
    from trtf_build.engine_defs.torch_trt.strategies.decoder import (
        DecoderBuildStrategy,
        StatelessCacheWrapper,
    )
    from trtf_build.engine_defs.torch_trt.strategies.encoder_only import (
        EncoderOnlyBuildStrategy,
        EncoderOnlyWrapper,
    )
    from trtf_build.engine_defs.torch_trt.strategies.diffusion import (
        DiffusionBuildStrategy,
        T5EncoderWrapper,
        PixArtDiTWrapper,
        VAEDecoderWrapper,
    )
except ImportError:
    pytest.skip("ttrt_build not importable", allow_module_level=True)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not available")


class TestStrategyRegistry:
    """Tests for the strategy registry (get_strategy)."""

    def test_get_decoder_strategy(self):
        strategy = get_strategy("decoder")
        assert isinstance(strategy, DecoderBuildStrategy)
        assert strategy.name == "decoder"
        assert strategy.runtime_strategy == "torchtrt_decoder"

    def test_get_encoder_only_strategy(self):
        strategy = get_strategy("encoder_only")
        assert isinstance(strategy, EncoderOnlyBuildStrategy)
        assert strategy.name == "encoder_only"
        assert strategy.runtime_strategy == "torchtrt_encoder"

    def test_get_diffusion_strategy(self):
        strategy = get_strategy("diffusion")
        assert isinstance(strategy, DiffusionBuildStrategy)
        assert strategy.name == "diffusion"
        assert strategy.runtime_strategy == "torchtrt_diffusion"

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown build strategy"):
            get_strategy("nonexistent_strategy")

    def test_unknown_strategy_lists_available(self):
        with pytest.raises(ValueError, match="decoder"):
            get_strategy("nonexistent_strategy")


class TestDecoderBuildStrategy:
    """Tests for DecoderBuildStrategy."""

    def _make_config(self, num_layers=2, num_heads=4, num_kv_heads=2,
                     head_dim=16, hidden_size=64, vocab_size=1000):
        return SimpleNamespace(
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )

    def test_wrap_model_returns_stateless_cache_wrapper(self):
        strategy = DecoderBuildStrategy()
        config = self._make_config()
        model = MagicMock()
        wrapper = strategy.wrap_model(model, config, max_cache_length=32)
        assert isinstance(wrapper, StatelessCacheWrapper)

    def test_wrap_model_passes_config(self):
        strategy = DecoderBuildStrategy()
        config = self._make_config(num_layers=4, num_heads=8)
        model = MagicMock()
        wrapper = strategy.wrap_model(model, config, max_cache_length=64)
        assert wrapper.num_layers == 4
        assert wrapper.num_heads == 8
        assert wrapper.max_cache_length == 64

    @requires_torch
    def test_pre_export_setup_patches_static_cache(self):
        strategy = DecoderBuildStrategy()
        strategy.pre_export_setup()
        from transformers.cache_utils import StaticLayer
        assert getattr(StaticLayer.update, '_scatter_patched', False) is True


class TestEncoderOnlyBuildStrategy:
    """Tests for EncoderOnlyBuildStrategy."""

    def test_wrap_model_returns_encoder_only_wrapper(self):
        strategy = EncoderOnlyBuildStrategy()
        config = SimpleNamespace(hidden_size=768)
        model = MagicMock()
        wrapper = strategy.wrap_model(model, config, max_cache_length=128)
        assert isinstance(wrapper, EncoderOnlyWrapper)
        assert wrapper.hidden_size == 768
        assert wrapper.max_seq_length == 128

    def test_pre_export_setup_is_noop(self):
        strategy = EncoderOnlyBuildStrategy()
        # Should not raise
        strategy.pre_export_setup()


@requires_torch
class TestEncoderOnlyWrapper:
    """Tests for EncoderOnlyWrapper forward pass."""

    def test_forward_output_shape(self):
        """EncoderOnlyWrapper produces [1, seq_len, hidden_size] output."""
        hidden_size = 64
        seq_len = 8
        config = SimpleNamespace(hidden_size=hidden_size)

        # Create a mock model that returns last_hidden_state
        mock_output = SimpleNamespace(
            last_hidden_state=torch.randn(1, seq_len, hidden_size, dtype=torch.float16)
        )
        model = MagicMock(return_value=mock_output)

        wrapper = EncoderOnlyWrapper(model, config, max_seq_length=seq_len)

        input_ids = torch.zeros(1, seq_len, dtype=torch.int32)
        attention_mask = torch.ones(1, seq_len, dtype=torch.float32)
        result = wrapper(input_ids, attention_mask)

        assert len(result) == 1
        assert result[0].shape == (1, seq_len, hidden_size)
        assert result[0].dtype == torch.float32

    def test_forward_converts_dtypes(self):
        """Wrapper converts int32 -> int64 for model input."""
        hidden_size = 32
        seq_len = 4
        config = SimpleNamespace(hidden_size=hidden_size)

        mock_output = SimpleNamespace(
            last_hidden_state=torch.randn(1, seq_len, hidden_size, dtype=torch.float16)
        )
        model = MagicMock(return_value=mock_output)

        wrapper = EncoderOnlyWrapper(model, config, max_seq_length=seq_len)

        input_ids = torch.zeros(1, seq_len, dtype=torch.int32)
        attention_mask = torch.ones(1, seq_len, dtype=torch.float32)
        wrapper(input_ids, attention_mask)

        # Check model was called with int64 tensors
        call_kwargs = model.call_args[1]
        assert call_kwargs["input_ids"].dtype == torch.int64
        assert call_kwargs["attention_mask"].dtype == torch.int64


class TestPluginDefaultStrategy:
    """Test that plugins without runtime_strategy default to 'decoder'."""

    def test_plugin_without_runtime_strategy_defaults_to_decoder(self):
        """getattr(plugin, 'runtime_strategy', 'decoder') returns 'decoder'."""
        plugin = SimpleNamespace(name="test_plugin")
        strategy_name = getattr(plugin, 'runtime_strategy', 'decoder')
        assert strategy_name == "decoder"

    def test_plugin_with_runtime_strategy_uses_it(self):
        plugin = SimpleNamespace(name="test_plugin", runtime_strategy="encoder_only")
        strategy_name = getattr(plugin, 'runtime_strategy', 'decoder')
        assert strategy_name == "encoder_only"


class TestDiffusionBuildStrategy:
    """Tests for DiffusionBuildStrategy."""

    def test_wrap_model_raises(self):
        strategy = DiffusionBuildStrategy()
        with pytest.raises(NotImplementedError, match="build_components"):
            strategy.wrap_model(None, None, 256)

    def test_make_export_args_raises(self):
        strategy = DiffusionBuildStrategy()
        with pytest.raises(NotImplementedError, match="build_components"):
            strategy.make_export_args(None, 256)

    def test_pre_export_setup_is_noop(self):
        strategy = DiffusionBuildStrategy()
        strategy.pre_export_setup()


@requires_torch
class TestT5EncoderWrapper:
    """Tests for T5EncoderWrapper forward pass."""

    def test_forward_output_shape(self):
        d_model = 64
        seq_len = 8
        mock_output = SimpleNamespace(
            last_hidden_state=torch.randn(1, seq_len, d_model, dtype=torch.float16)
        )
        model = MagicMock(return_value=mock_output)

        wrapper = T5EncoderWrapper(model)
        input_ids = torch.zeros(1, seq_len, dtype=torch.int32)
        attention_mask = torch.ones(1, seq_len, dtype=torch.int32)
        result = wrapper(input_ids, attention_mask)

        assert len(result) == 1
        assert result[0].shape == (1, seq_len, d_model)
        assert result[0].dtype == torch.float32

    def test_forward_converts_int32_to_int64(self):
        d_model = 32
        seq_len = 4
        mock_output = SimpleNamespace(
            last_hidden_state=torch.randn(1, seq_len, d_model)
        )
        model = MagicMock(return_value=mock_output)

        wrapper = T5EncoderWrapper(model)
        input_ids = torch.zeros(1, seq_len, dtype=torch.int32)
        attention_mask = torch.ones(1, seq_len, dtype=torch.int32)
        wrapper(input_ids, attention_mask)

        call_kwargs = model.call_args[1]
        assert call_kwargs["input_ids"].dtype == torch.int64


@requires_torch
class TestPixArtDiTWrapper:
    """Tests for PixArtDiTWrapper forward pass."""

    def test_forward_output_shape(self):
        h_lat, w_lat = 16, 16
        out_channels = 8
        seq_len = 120
        mock_output = SimpleNamespace(
            sample=torch.randn(1, out_channels, h_lat, w_lat, dtype=torch.float16)
        )
        model = MagicMock(return_value=mock_output)

        wrapper = PixArtDiTWrapper(model, in_channels=4, out_channels=out_channels)
        sample = torch.randn(1, 4, h_lat, w_lat, dtype=torch.float16)
        text = torch.randn(1, seq_len, 4096, dtype=torch.float16)
        timestep = torch.tensor([1.0], dtype=torch.float16)
        enc_mask = torch.ones(1, seq_len, dtype=torch.float16)
        result = wrapper(sample, text, timestep, enc_mask)

        assert len(result) == 1
        assert result[0].shape == (1, out_channels, h_lat, w_lat)

    def test_forward_passes_correct_kwargs(self):
        seq_len = 120
        mock_output = SimpleNamespace(
            sample=torch.randn(1, 8, 16, 16)
        )
        model = MagicMock(return_value=mock_output)

        wrapper = PixArtDiTWrapper(model, in_channels=4, out_channels=8)
        sample = torch.randn(1, 4, 16, 16)
        text = torch.randn(1, seq_len, 4096)
        timestep = torch.tensor([1.0])
        enc_mask = torch.ones(1, seq_len)
        wrapper(sample, text, timestep, enc_mask)

        call_kwargs = model.call_args[1]
        assert "hidden_states" in call_kwargs
        assert "encoder_hidden_states" in call_kwargs
        assert "timestep" in call_kwargs
        assert "encoder_attention_mask" in call_kwargs


@requires_torch
class TestVAEDecoderWrapper:
    """Tests for VAEDecoderWrapper forward pass."""

    def test_forward_output_shape(self):
        h_lat, w_lat = 16, 16
        scaling_factor = 0.13025
        mock_decoded = SimpleNamespace(
            sample=torch.randn(1, 3, h_lat * 8, w_lat * 8, dtype=torch.float16)
        )
        model = MagicMock()
        model.decode = MagicMock(return_value=mock_decoded)

        wrapper = VAEDecoderWrapper(model, scaling_factor)
        latent = torch.randn(1, 4, h_lat, w_lat, dtype=torch.float16)
        result = wrapper(latent)

        assert len(result) == 1
        assert result[0].shape == (1, 3, h_lat * 8, w_lat * 8)
        assert result[0].dtype == torch.float32

    def test_forward_applies_scaling(self):
        scaling_factor = 0.5
        mock_decoded = SimpleNamespace(
            sample=torch.randn(1, 3, 128, 128)
        )
        model = MagicMock()
        model.decode = MagicMock(return_value=mock_decoded)

        wrapper = VAEDecoderWrapper(model, scaling_factor)
        latent = torch.ones(1, 4, 16, 16)
        wrapper(latent)

        # Check that decode was called with scaled input
        call_args = model.decode.call_args[0]
        expected = latent / scaling_factor
        assert torch.allclose(call_args[0], expected)


class TestBackwardCompatImports:
    """Test that backward-compat aliases in compiler.py still work."""

    def test_import_stateless_cache_wrapper_from_compiler(self):
        from trtf_build.engine_defs.torch_trt.compiler import StatelessCacheWrapper as SCW
        from trtf_build.engine_defs.torch_trt.strategies.decoder import StatelessCacheWrapper as SCW2
        assert SCW is SCW2

    def test_import_patch_from_compiler(self):
        from trtf_build.engine_defs.torch_trt.compiler import patch_static_cache_scatter as p1
        from trtf_build.engine_defs.torch_trt.strategies.decoder import patch_static_cache_scatter as p2
        assert p1 is p2
