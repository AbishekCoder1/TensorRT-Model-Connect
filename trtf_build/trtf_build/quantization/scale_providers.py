"""Scale acquisition strategies.

Each ScaleProvider knows how to obtain quantization scales for a model.
Implementations range from running ModelOpt calibration to loading
pre-computed JSON files or extracting from pre-quantized checkpoints.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .formats import QuantFormat
from .scales import LayerScales, QuantScaleMap

if TYPE_CHECKING:
    from ..config import ModelConfig

logger = logging.getLogger(__name__)


class ScaleProvider(Protocol):
    """Strategy for obtaining quantization scales."""

    def acquire_scales(
        self,
        model_dir: str,
        config: ModelConfig,
        quant_format: QuantFormat,
        exclude_patterns: list[str],
    ) -> QuantScaleMap:
        """Return per-layer scales for the given format."""
        ...


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class PrecomputedJsonProvider:
    """Load pre-computed scales from a JSON file."""

    def __init__(self, json_path: str) -> None:
        self.json_path = json_path

    def acquire_scales(
        self,
        model_dir: str,
        config: ModelConfig,
        quant_format: QuantFormat,
        exclude_patterns: list[str],
    ) -> QuantScaleMap:
        logger.info("Loading pre-computed scales from %s", self.json_path)
        return QuantScaleMap.load(self.json_path)


class ModelOptCalibrationProvider:
    """Run ModelOpt PTQ calibration to compute scales.

    Works for ANY model — ModelOpt operates at the nn.Linear level and
    does not require model-specific recipes. Generic configs like
    FP8_DEFAULT_CFG, INT8_SMOOTHQUANT_CFG, INT4_AWQ_CFG work universally.
    """

    # Map our format names to ModelOpt config names
    _MTQ_CONFIG_NAMES: dict[str, str] = {
        "fp8": "FP8_DEFAULT_CFG",
        "int8_sq": "INT8_SMOOTHQUANT_CFG",
        "int4_awq": "INT4_AWQ_CFG",
        "nvfp4": "NVFP4_DEFAULT_CFG",
        "w4a8": "W4A8_AWQ_BETA_CFG",
    }

    # Maxbound values for scale computation: scale = amax / maxbound
    _MAXBOUND: dict[str, float] = {
        "fp8": 448.0,       # FP8 E4M3
        "int8_sq": 127.0,   # INT8
        "int4_awq": 7.0,    # INT4
        "nvfp4": 6.0,       # NVFP4
        "w4a8": 7.0,        # W4A8 weight
    }

    def __init__(
        self,
        num_samples: int = 512,
        calibration_prompts: list[str] | None = None,
    ) -> None:
        self.num_samples = num_samples
        self.calibration_prompts = calibration_prompts

    def acquire_scales(
        self,
        model_dir: str,
        config: ModelConfig,
        quant_format: QuantFormat,
        exclude_patterns: list[str],
    ) -> QuantScaleMap:
        try:
            import modelopt.torch.quantization as mtq
        except ImportError:
            raise RuntimeError(
                "nvidia-modelopt is required for auto-calibration. "
                "Install with: pip install nvidia-modelopt"
            )

        mtq_config_name = self._MTQ_CONFIG_NAMES.get(quant_format.name)
        if mtq_config_name is None:
            raise ValueError(
                f"No ModelOpt config for format {quant_format.name!r}")
        mtq_cfg = getattr(mtq, mtq_config_name)

        logger.info(
            "Running ModelOpt calibration (%s, %d samples) ...",
            mtq_config_name, self.num_samples)

        import re
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Build calibration data
        prompts = self.calibration_prompts or self._default_prompts()
        calib_ids = [
            tokenizer(p, return_tensors="pt", truncation=True,
                      max_length=256).input_ids.to(model.device)
            for p in prompts[:self.num_samples]
        ]

        def forward_loop(m):
            for ids in calib_ids:
                m(ids)

        # Disable quantization on excluded layers
        exclude_re = re.compile("|".join(
            f"({p.replace('*', '.*')})" for p in exclude_patterns
        )) if exclude_patterns else None

        quantized = mtq.quantize(model, mtq_cfg, forward_loop)

        if exclude_re:
            for name, module in quantized.named_modules():
                if exclude_re.search(name):
                    mtq.disable_quantizer(module, "input_quantizer")
                    mtq.disable_quantizer(module, "weight_quantizer")

        # Extract scales from state dict
        maxbound = self._MAXBOUND.get(quant_format.name, 448.0)
        return self._extract_scales(
            quantized.state_dict(), exclude_re, maxbound)

    def _extract_scales(
        self,
        state_dict: dict,
        exclude_re,
        maxbound: float,
    ) -> QuantScaleMap:
        """Convert ModelOpt amax values to scales."""
        import re

        amax_entries: dict[str, dict[str, float]] = {}
        for key, val in state_dict.items():
            if "_quantizer._amax" not in key:
                continue
            # e.g., "model.layers.0.self_attn.q_proj.input_quantizer._amax"
            layer_name = key.rsplit(".", 2)[0]  # strip ".XXX_quantizer._amax"
            quant_type = "input" if "input_quantizer" in key else "weight"
            amax = float(val.max())
            if layer_name not in amax_entries:
                amax_entries[layer_name] = {}
            amax_entries[layer_name][f"{quant_type}_scale"] = amax / maxbound

        # Convert to our weight naming convention
        scales: dict[str, LayerScales] = {}
        for layer_name, scale_dict in amax_entries.items():
            if exclude_re and exclude_re.search(layer_name):
                continue
            if "input_scale" in scale_dict and "weight_scale" in scale_dict:
                scales[layer_name] = LayerScales(
                    input_scale=scale_dict["input_scale"],
                    weight_scale=scale_dict["weight_scale"],
                )

        logger.info("Extracted scales for %d layers", len(scales))
        return QuantScaleMap(scales=scales)

    @staticmethod
    def _default_prompts() -> list[str]:
        """Small default calibration set for decoder models."""
        return [
            "The quick brown fox jumps over the lazy dog.",
            "In a recent study, researchers found that",
            "The capital of France is Paris, which is known for",
            "Machine learning models can be trained using",
            "Once upon a time in a land far away,",
        ] * 100  # repeat to get 500 samples


class DynamicQuantizationProvider:
    """No calibration — scales computed at runtime by TRT.

    Used for NVFP4 and MXFP8 where TRT's IDynamicQuantizeLayer computes
    per-block scales during inference.
    """

    def acquire_scales(
        self,
        model_dir: str,
        config: ModelConfig,
        quant_format: QuantFormat,
        exclude_patterns: list[str],
    ) -> QuantScaleMap:
        logger.info("Using dynamic quantization (runtime scales)")
        return QuantScaleMap(scales={}, dynamic=True)


class PreQuantizedCheckpointProvider:
    """Extract scales from pre-quantized HF checkpoints (GPTQ, AWQ).

    Detects quantization format from the model's config.json
    quantization_config field.
    """

    def acquire_scales(
        self,
        model_dir: str,
        config: ModelConfig,
        quant_format: QuantFormat,
        exclude_patterns: list[str],
    ) -> QuantScaleMap:
        quant_config = config.raw.get("quantization_config", {})
        quant_method = quant_config.get("quant_method", "")

        if quant_method == "gptq":
            return self._extract_gptq(model_dir, config, exclude_patterns)
        elif quant_method == "awq":
            return self._extract_awq(model_dir, config, exclude_patterns)
        else:
            raise ValueError(
                f"Unsupported pre-quantized format: {quant_method!r}. "
                "Expected 'gptq' or 'awq'.")

    def _extract_gptq(self, model_dir, config, exclude_patterns):
        """Extract scales from GPTQ checkpoint."""
        import re
        from safetensors import safe_open

        exclude_re = re.compile("|".join(
            f"({p.replace('*', '.*')})" for p in exclude_patterns
        )) if exclude_patterns else None

        scales: dict[str, LayerScales] = {}
        model_path = Path(model_dir)
        for sf_path in model_path.glob("*.safetensors"):
            with safe_open(str(sf_path), framework="numpy") as f:
                for key in f.keys():
                    if key.endswith(".g_idx") or key.endswith(".qzeros"):
                        continue
                    if key.endswith(".scales"):
                        layer_name = key.rsplit(".scales", 1)[0]
                        if exclude_re and exclude_re.search(layer_name):
                            continue
                        scale_array = f.get_tensor(key)
                        scales[layer_name] = LayerScales(
                            weight_scale=scale_array,
                            input_scale=1.0,
                            block_size=config.raw.get(
                                "quantization_config", {}).get(
                                "group_size", 128),
                        )

        logger.info("Extracted GPTQ scales for %d layers", len(scales))
        return QuantScaleMap(scales=scales)

    def _extract_awq(self, model_dir, config, exclude_patterns):
        """Extract scales from AWQ checkpoint."""
        import re
        from safetensors import safe_open

        exclude_re = re.compile("|".join(
            f"({p.replace('*', '.*')})" for p in exclude_patterns
        )) if exclude_patterns else None

        quant_config = config.raw.get("quantization_config", {})
        group_size = quant_config.get("group_size", 128)

        scales: dict[str, LayerScales] = {}
        model_path = Path(model_dir)
        for sf_path in model_path.glob("*.safetensors"):
            with safe_open(str(sf_path), framework="numpy") as f:
                for key in f.keys():
                    if not key.endswith(".scales"):
                        continue
                    layer_name = key.rsplit(".scales", 1)[0]
                    if exclude_re and exclude_re.search(layer_name):
                        continue
                    scale_array = f.get_tensor(key)
                    scales[layer_name] = LayerScales(
                        weight_scale=scale_array,
                        input_scale=1.0,
                        block_size=group_size,
                    )

        logger.info("Extracted AWQ scales for %d layers", len(scales))
        return QuantScaleMap(scales=scales)
