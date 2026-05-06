"""Tests for the PatchTST Torch-TRT family plugin and build strategy."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

try:
    import torch
except ImportError:
    pytest.skip("torch not available", allow_module_level=True)

try:
    from tensorrt_model_connect.engine_defs.torch_trt.config import ModelConfig
    from tensorrt_model_connect.engine_defs.torch_trt.families.patchtst import plugin as patchtst_plugin
    from tensorrt_model_connect.engine_defs.torch_trt.strategies import get_strategy
    from tensorrt_model_connect.engine_defs.torch_trt.strategies.patchtst import (
        PatchTSTBuildStrategy,
        PatchTSTWrapper,
    )
except ImportError:
    pytest.skip("tensorrt_model_connect not importable", allow_module_level=True)


class TestPatchTSTRegistry:
    def test_family_import_registers_patchtst_strategy(self):
        strategy = get_strategy("patchtst")
        assert isinstance(strategy, PatchTSTBuildStrategy)
        assert strategy.runtime_strategy == "patchtst_torchtrt"


class TestPatchTSTBuildStrategy:
    def test_make_export_args_uses_numeric_float32_inputs(self):
        strategy = PatchTSTBuildStrategy()
        config = SimpleNamespace(context_length=16, num_input_channels=5)

        past_values, past_observed_mask = strategy.make_export_args(
            config, max_cache_length=16, precision="bf16")

        assert past_values.shape == (1, 16, 5)
        assert past_observed_mask.shape == (1, 16, 5)
        assert past_values.dtype == torch.float32
        assert past_observed_mask.dtype == torch.float32

    @pytest.mark.parametrize(
        "raw_overrides, expected_task",
        [
            ({"problem_type": "single_label_classification"}, "classification"),
            ({"problem_type": "regression"}, "regression"),
            ({"problem_type": "forecasting"}, "forecast"),
            ({"architectures": ["PatchTSTForClassification"]}, "classification"),
            ({"architectures": ["PatchTSTForRegression"]}, "regression"),
            ({"architectures": ["PatchTSTForPrediction"]}, "forecast"),
            ({}, "forecast"),
        ],
    )
    def test_load_model_selects_correct_hf_class(self, monkeypatch, raw_overrides,
                                                 expected_task):
        calls: dict[str, list[dict[str, object]]] = {
            "PatchTSTForPrediction": [],
            "PatchTSTForClassification": [],
            "PatchTSTForRegression": [],
        }

        def _make_model_class(class_name: str):
            class _FakeModel:
                @classmethod
                def from_pretrained(cls, model_dir, dtype=None, device_map=None):
                    calls[class_name].append(
                        {"model_dir": model_dir, "dtype": dtype, "device_map": device_map}
                    )
                    model = SimpleNamespace(config=SimpleNamespace())
                    model.eval = lambda: model
                    model.loaded_class = class_name
                    return model

            _FakeModel.__name__ = class_name
            return _FakeModel

        fake_transformers = SimpleNamespace(
            PatchTSTForPrediction=_make_model_class("PatchTSTForPrediction"),
            PatchTSTForClassification=_make_model_class("PatchTSTForClassification"),
            PatchTSTForRegression=_make_model_class("PatchTSTForRegression"),
        )
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        raw = {"model_type": "patchtst", **raw_overrides}
        config = ModelConfig(raw=raw)

        model = patchtst_plugin.load_model(
            "/tmp/patchtst-model", config, 32, dtype=torch.float16)

        assert model.loaded_class == {
            "classification": "PatchTSTForClassification",
            "regression": "PatchTSTForRegression",
            "forecast": "PatchTSTForPrediction",
        }[expected_task]
        assert calls[model.loaded_class][-1]["dtype"] == torch.float16
        assert calls[model.loaded_class][-1]["device_map"] == "cuda"


class TestPatchTSTWrapper:
    @pytest.mark.parametrize(
        "task_type, output_attr, output_shape",
        [
            ("forecast", "prediction_outputs", (1, 12, 4)),
            ("classification", "prediction_logits", (1, 3)),
            ("regression", "regression_outputs", (1, 2)),
        ],
    )
    def test_forward_returns_float32_tensor_and_casts_inputs(
        self, task_type, output_attr, output_shape
    ):
        output_tensor = torch.randn(*output_shape, dtype=torch.float16)
        dummy_model = MagicMock(return_value=SimpleNamespace(**{output_attr: output_tensor}))
        wrapper = PatchTSTWrapper(
            dummy_model, task_type=task_type, compute_dtype=torch.float16
        )

        past_values = torch.randn(1, 8, 4, dtype=torch.float32)
        observed_mask = torch.ones(1, 8, 4, dtype=torch.float32)
        result = wrapper(past_values, observed_mask)

        assert len(result) == 1
        assert result[0].dtype == torch.float32
        assert torch.allclose(result[0], output_tensor.to(torch.float32))

        call_kwargs = dummy_model.call_args.kwargs
        assert call_kwargs["past_values"].dtype == torch.float16
        assert call_kwargs["past_observed_mask"].dtype == torch.bool

    def test_regression_distribution_tuple_is_stacked_into_tensor(self):
        dummy_model = MagicMock(
            return_value=SimpleNamespace(
                regression_outputs=(
                    torch.tensor([0.25], dtype=torch.float16),
                    torch.tensor([0.75], dtype=torch.float16),
                )
            )
        )
        wrapper = PatchTSTWrapper(
            dummy_model, task_type="regression", compute_dtype=torch.float16
        )

        past_values = torch.randn(1, 8, 1, dtype=torch.float32)
        observed_mask = torch.ones(1, 8, 1, dtype=torch.float32)
        result = wrapper(past_values, observed_mask)[0]

        assert result.dtype == torch.float32
        assert result.shape == (1, 2)
        assert torch.allclose(
            result,
            torch.tensor([[0.25, 0.75]], dtype=torch.float32),
            rtol=0.0,
            atol=0.0,
        )
