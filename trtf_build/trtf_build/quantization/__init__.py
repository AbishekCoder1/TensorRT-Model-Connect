"""Quantization framework — extensible low-precision support.

Public API:
    build_quant_context() — construct a QuantContext from CLI args
    get_format() — look up a registered format by name
    list_formats() — list available format names
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .context import QuantContext
from .formats import QuantFormat
from .profile import QuantProfile
from .registry import get_format, list_formats, register_format
from .scales import LayerScales, QuantScaleMap
from .scale_providers import (
    DynamicQuantizationProvider,
    ModelOptCalibrationProvider,
    PrecomputedJsonProvider,
    PreQuantizedCheckpointProvider,
)

if TYPE_CHECKING:
    from ..config import ModelConfig

logger = logging.getLogger(__name__)

__all__ = [
    "QuantContext",
    "QuantFormat",
    "QuantProfile",
    "QuantScaleMap",
    "LayerScales",
    "build_quant_context",
    "get_format",
    "list_formats",
    "register_format",
]


def build_quant_context(
    format_name: str,
    model_dir: str,
    config: ModelConfig,
    exclude_patterns: list[str] | None = None,
    *,
    scales_json: str | None = None,
    num_calibration_samples: int = 512,
    calibration_prompts: list[str] | None = None,
) -> QuantContext:
    """Construct a QuantContext from high-level parameters.

    This is the main entry point called by engine_builder.py.

    Args:
        format_name: Quantization format ('fp8', 'int8_sq', 'int4_awq', etc.)
        model_dir: Path to HuggingFace model directory.
        config: Parsed ModelConfig.
        exclude_patterns: Weight name patterns to skip (norms, embeddings).
        scales_json: Path to pre-computed scales JSON. If provided, skips
            calibration entirely.
        num_calibration_samples: Number of calibration samples for PTQ.
        calibration_prompts: Custom calibration prompts. None = default.

    Returns:
        QuantContext ready to thread through graph_blocks.
    """
    fmt = get_format(format_name)

    if exclude_patterns is None:
        exclude_patterns = _default_exclude_patterns()

    # Select scale provider
    if scales_json:
        provider = PrecomputedJsonProvider(scales_json)
    elif format_name == "nvfp4":
        # NVFP4 uses dynamic quantization (runtime scales)
        provider = DynamicQuantizationProvider()
    else:
        # Auto-calibrate with ModelOpt
        provider = ModelOptCalibrationProvider(
            num_samples=num_calibration_samples,
            calibration_prompts=calibration_prompts,
        )

    scale_map = provider.acquire_scales(
        model_dir, config, fmt, exclude_patterns)

    profile = QuantProfile(
        format=fmt,
        scale_map=scale_map,
        exclude_patterns=exclude_patterns,
    )

    logger.info(
        "Built quantization context: format=%s, %d layers quantized, "
        "%d excluded patterns",
        format_name, len(scale_map.scales), len(exclude_patterns))

    return QuantContext(profile=profile)


def _default_exclude_patterns() -> list[str]:
    """Default weight name patterns to exclude from quantization.

    Norms, embeddings, and output heads are kept in full precision.
    """
    return [
        "embedding",
        "final_norm",
        "w_out",
        "lm_head",
        "*.input_norm",
        "*.post_attn_norm",
        "*_norm*",
    ]
