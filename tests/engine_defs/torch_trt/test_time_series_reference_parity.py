"""Reference parity tests for time-series Torch-TRT wrappers.

These tests validate wrapper correctness against the direct Python reference
implementation using tiny in-memory models. They are intended to run in CI:
no checkpoint downloads, no TRT engine build, no GPU requirement.
"""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

PatchTSTWrapper = importlib.import_module(
    "trtf_build.engine_defs.torch_trt.strategies.patchtst"
).PatchTSTWrapper
PatchTSMixerWrapper = importlib.import_module(
    "trtf_build.engine_defs.torch_trt.strategies.patchtsmixer"
).PatchTSMixerWrapper
ChronosBoltForecastWrapper = importlib.import_module(
    "trtf_build.engine_defs.torch_trt.strategies.chronos_bolt"
).ChronosBoltForecastWrapper
TimesFmWrapper = importlib.import_module(
    "trtf_build.engine_defs.torch_trt.strategies.timesfm"
).TimesFmWrapper


@pytest.mark.unit
def test_patchtst_wrapper_matches_reference_forward_exactly():
    torch.manual_seed(0)

    config = transformers.PatchTSTConfig(
        num_input_channels=2,
        num_targets=3,
        context_length=8,
        prediction_length=4,
        patch_length=2,
        stride=2,
        d_model=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        use_cls_token=True,
    )
    model = transformers.PatchTSTForClassification(config).eval()
    wrapper = PatchTSTWrapper(model, task_type="classification", compute_dtype=torch.float32)

    past_values = torch.randn(1, 8, 2, dtype=torch.float32)
    past_observed_mask = torch.ones(1, 8, 2, dtype=torch.float32)

    wrapped = wrapper(past_values, past_observed_mask)[0]
    reference = model(
        past_values=past_values.to(torch.float32),
        past_observed_mask=past_observed_mask.to(torch.bool),
        return_dict=True,
    ).prediction_logits.to(torch.float32)

    torch.testing.assert_close(wrapped, reference, rtol=0.0, atol=0.0)


@pytest.mark.unit
def test_patchtsmixer_wrapper_matches_reference_forward_exactly():
    torch.manual_seed(0)

    config = transformers.PatchTSMixerConfig(
        context_length=8,
        prediction_length=4,
        num_input_channels=2,
        patch_length=2,
        patch_stride=2,
        d_model=8,
        num_layers=1,
        expansion_factor=2,
        dropout=0.0,
    )
    model = transformers.PatchTSMixerForPrediction(config).eval()
    wrapper = PatchTSMixerWrapper(
        model,
        config,
        context_length=8,
        num_input_channels=2,
        compute_dtype=torch.float32,
        task_kind="prediction",
    )

    past_values = torch.randn(1, 8, 2, dtype=torch.float32)
    observed_mask = torch.ones(1, 8, 2, dtype=torch.float32)

    wrapped = wrapper(past_values, observed_mask)[0]
    reference = model(
        past_values=past_values.to(torch.float32) * observed_mask.to(torch.float32),
        observed_mask=observed_mask.to(torch.float32),
        return_loss=False,
        return_dict=True,
    ).prediction_outputs.to(torch.float32)

    torch.testing.assert_close(wrapped, reference, rtol=0.0, atol=0.0)


@pytest.mark.unit
def test_timesfm_wrapper_matches_reference_forward_exactly():
    torch.manual_seed(0)

    config = transformers.TimesFmConfig(
        patch_length=2,
        context_length=8,
        horizon_length=4,
        freq_size=3,
        num_hidden_layers=1,
        hidden_size=16,
        intermediate_size=32,
        head_dim=8,
        num_attention_heads=2,
        quantiles=(0.1, 0.5, 0.9),
        attention_dropout=0.0,
    )
    model = transformers.TimesFmModelForPrediction(config).eval()
    wrapper = TimesFmWrapper(model, context_length=8, compute_dtype=torch.float32)

    past_values = torch.linspace(0.0, 1.0, steps=8, dtype=torch.float32).unsqueeze(0)
    past_values_padding = torch.zeros((1, 8), dtype=torch.int32)
    freq = torch.tensor([2], dtype=torch.int32)

    wrapped_mean, wrapped_full = wrapper(past_values, past_values_padding, freq)

    masked_series = past_values[0]
    reference = model(
        past_values=[masked_series.contiguous()],
        freq=[freq.reshape(-1)[0].to(torch.int64)],
        return_dict=True,
    )

    torch.testing.assert_close(
        wrapped_mean, reference.mean_predictions.to(torch.float32), rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        wrapped_full, reference.full_predictions.to(torch.float32), rtol=0.0, atol=0.0)


@pytest.mark.unit
def test_chronos_bolt_wrapper_matches_reference_forward_exactly():
    chronos = pytest.importorskip("chronos", reason="chronos-forecasting not installed")
    torch.manual_seed(0)

    config = transformers.T5Config(
        d_model=16,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        dropout_rate=0.0,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    config.architectures = ["ChronosBoltModelForForecasting"]
    config.chronos_config = {
        "context_length": 16,
        "prediction_length": 4,
        "input_patch_size": 4,
        "input_patch_stride": 4,
        "quantiles": [0.1, 0.5, 0.9],
        "use_reg_token": True,
    }
    model = chronos.chronos_bolt.ChronosBoltModelForForecasting(config).eval()
    wrapper = ChronosBoltForecastWrapper(model, max_context_length=16)

    context = torch.full((1, 16), float("nan"), dtype=torch.float32)
    context[0, -8:] = torch.linspace(0.0, 1.0, steps=8, dtype=torch.float32)

    wrapped = wrapper(context)[0]
    reference = model(context=context).quantile_preds.to(torch.float32)

    torch.testing.assert_close(wrapped, reference, rtol=0.0, atol=0.0)
