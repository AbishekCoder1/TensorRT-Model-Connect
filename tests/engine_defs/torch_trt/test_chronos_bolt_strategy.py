"""Tests for the Chronos-Bolt Torch-TRT strategy and wrapper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from trtf_build.engine_defs.torch_trt.families.chronos_bolt import (
        ChronosBoltTorchTrtPlugin,
    )
    from trtf_build.engine_defs.torch_trt.strategies.chronos_bolt import (
        ChronosBoltBuildStrategy,
        ChronosBoltForecastWrapper,
    )
except ImportError:
    pytest.skip("chronos_bolt strategy modules not importable", allow_module_level=True)


requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not available")


class TestChronosBoltFamilyPlugin:
    def test_matches_expected_aliases(self):
        plugin = ChronosBoltTorchTrtPlugin()
        assert plugin.name == "chronos_bolt"
        assert plugin.runtime_strategy == "chronos_bolt"
        assert plugin.matches("chronos_bolt")
        assert plugin.matches("chronos-bolt")
        assert plugin.matches("chronos-bolt-small")
        assert not plugin.matches("t5")

    def test_matches_official_t5_config_with_chronos_metadata(self):
        plugin = ChronosBoltTorchTrtPlugin()
        config = SimpleNamespace(
            model_type="t5",
            raw={
                "architectures": ["ChronosBoltModelForForecasting"],
                "chronos_config": {"context_length": 2048, "prediction_length": 64},
            },
        )
        assert plugin.matches_config(config)


class TestChronosBoltBuildStrategy:
    def test_runtime_strategy_is_chronos_bolt_torchtrt(self):
        strategy = ChronosBoltBuildStrategy()
        assert strategy.name == "chronos_bolt"
        assert strategy.runtime_strategy == "chronos_bolt_torchtrt"

    @requires_torch
    def test_make_export_args_uses_numeric_tensors(self):
        strategy = ChronosBoltBuildStrategy()
        config = SimpleNamespace(
            raw={
                "chronos_config": {
                    "context_length": 32,
                    "prediction_length": 4,
                    "quantiles": [0.1, 0.5, 0.9],
                }
            }
        )
        (context,) = strategy.make_export_args(config, max_cache_length=16)
        assert context.shape == (1, 32)
        assert context.dtype == torch.float32
        assert torch.isnan(context[0, 0])
        assert not torch.isnan(context[0, -1])

    @requires_torch
    def test_wrap_model_returns_numeric_wrapper(self):
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))
                self.config = SimpleNamespace(d_model=32)

            def forward(self, context):
                return SimpleNamespace(
                    quantile_preds=torch.randn(1, 3, 4, dtype=torch.float32, device=context.device)
                )

        model = DummyModel()
        config = SimpleNamespace(raw={"chronos_config": {"prediction_length": 4, "context_length": 8}})
        wrapper = ChronosBoltBuildStrategy().wrap_model(model, config, max_cache_length=8)
        assert isinstance(wrapper, ChronosBoltForecastWrapper)
        assert wrapper.max_context_length == 8


@requires_torch
class TestChronosBoltForecastWrapper:
    def test_forward_produces_float32_forecast_tensor(self):
        seq_len = 8
        prediction_length = 5
        num_quantiles = 3

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))
                self.config = SimpleNamespace(d_model=16)

            def forward(self, context):
                assert context.shape == (1, seq_len)
                return SimpleNamespace(
                    quantile_preds=torch.randn(
                        1, num_quantiles, prediction_length, dtype=torch.float32, device=context.device
                    )
                )

        wrapper = ChronosBoltForecastWrapper(
            DummyModel(),
            max_context_length=seq_len,
        )

        context = torch.linspace(0.0, 1.0, seq_len, dtype=torch.float32).unsqueeze(0)
        result = wrapper(context)

        assert len(result) == 1
        assert result[0].shape == (1, num_quantiles, prediction_length)
        assert result[0].dtype == torch.float32
