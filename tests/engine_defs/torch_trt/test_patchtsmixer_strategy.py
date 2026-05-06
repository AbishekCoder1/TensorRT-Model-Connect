"""Tests for PatchTSMixer Torch-TRT strategy and family plugin."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from tensorrt_model_connect.engine_defs.torch_trt.families.patchtsmixer import (
    PatchTSMixerTorchTrtPlugin,
)
from tensorrt_model_connect.engine_defs.torch_trt.strategies.patchtsmixer import (
    PatchTSMixerBuildStrategy,
    PatchTSMixerWrapper,
    infer_patchtsmixer_task_kind,
)


class TestPatchTSMixerStrategy:
    def test_task_kind_inference_uses_architecture(self):
        config = SimpleNamespace(raw={"architectures": ["PatchTSMixerForRegression"]})
        assert infer_patchtsmixer_task_kind(config) == "regression"

    def test_wrap_model_returns_patchtsmixer_wrapper(self):
        strategy = PatchTSMixerBuildStrategy()
        config = SimpleNamespace(
            raw={
                "architectures": ["PatchTSMixerForPrediction"],
                "context_length": 64,
                "num_input_channels": 3,
                "prediction_length": 12,
            },
            context_length=64,
            num_input_channels=3,
            prediction_length=12,
        )
        model = MagicMock()

        wrapper = strategy.wrap_model(model, config, max_cache_length=64)
        assert isinstance(wrapper, PatchTSMixerWrapper)
        assert wrapper.context_length == 64
        assert wrapper.num_input_channels == 3
        assert wrapper.task_kind == "prediction"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
    def test_make_export_args_shapes_numeric_inputs(self):
        strategy = PatchTSMixerBuildStrategy()
        config = SimpleNamespace(
            raw={"context_length": 48, "num_input_channels": 5},
            context_length=48,
            num_input_channels=5,
        )

        past_values, observed_mask = strategy.make_export_args(config, 48)

        assert past_values.shape == (1, 48, 5)
        assert observed_mask.shape == (1, 48, 5)
        assert past_values.dtype == torch.float32
        assert observed_mask.dtype == torch.float32
        assert torch.all(observed_mask == 1)

    def test_wrapper_uses_prediction_outputs(self):
        config = SimpleNamespace(raw={"prediction_length": 8, "num_input_channels": 2})
        mock_output = SimpleNamespace(
            prediction_outputs=torch.randn(1, 8, 2, dtype=torch.float16)
        )
        model = MagicMock(return_value=mock_output)
        wrapper = PatchTSMixerWrapper(
            model,
            config,
            context_length=16,
            num_input_channels=2,
            compute_dtype=torch.float16,
            task_kind="prediction",
        )

        past_values = torch.randn(1, 16, 2, dtype=torch.float32)
        observed_mask = torch.ones(1, 16, 2, dtype=torch.float32)
        result = wrapper(past_values, observed_mask)

        assert len(result) == 1
        assert result[0].shape == (1, 8, 2)
        assert result[0].dtype == torch.float32

        kwargs = model.call_args.kwargs
        assert kwargs["past_values"].dtype == torch.float16
        assert kwargs["observed_mask"].dtype == torch.float16
        assert kwargs["return_dict"] is True
        assert kwargs["return_loss"] is False

    def test_wrapper_uses_regression_outputs_without_forward_mask_kwarg(self):
        config = SimpleNamespace(raw={"num_targets": 4, "num_input_channels": 3})
        mock_output = SimpleNamespace(
            regression_outputs=torch.randn(1, 4, dtype=torch.float16)
        )
        model = MagicMock(return_value=mock_output)
        wrapper = PatchTSMixerWrapper(
            model,
            config,
            context_length=8,
            num_input_channels=3,
            compute_dtype=torch.float16,
            task_kind="regression",
        )

        past_values = torch.randn(1, 8, 3, dtype=torch.float32)
        observed_mask = torch.ones(1, 8, 3, dtype=torch.float32)
        result = wrapper(past_values, observed_mask)

        assert len(result) == 1
        assert result[0].shape == (1, 4)

        kwargs = model.call_args.kwargs
        assert "observed_mask" not in kwargs
        assert kwargs["past_values"].dtype == torch.float16

    def test_family_plugin_matches_patchtsmixer(self):
        plugin = PatchTSMixerTorchTrtPlugin()
        assert plugin.runtime_strategy == "patchtsmixer"
        assert plugin.matches("PatchTSMixer")
        assert plugin.matches("granite-timeseries-patchtsmixer")
