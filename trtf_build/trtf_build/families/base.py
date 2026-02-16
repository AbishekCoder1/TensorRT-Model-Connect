"""FamilyPlugin protocol — defines the interface for model family plugins."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..config import ModelConfig
from ..checkpoint_mapper import WeightDict


class FamilyPlugin(Protocol):
    """Interface for a model family plugin.

    Required attributes:
        name: Human-readable family name (e.g. "qwen", "llama").

    Optional attributes:
        runtime_strategy: Backend dispatch key for C++ runtime.
            "decoder_kv_cache" (default), "decoder_moe", "ssm_recurrent",
            "vision_language".
        embed_input: If True, the text decoder takes input_embed instead of
            token_id during VL prefill. Only meaningful for VL models.
    """

    name: str

    def matches(self, model_type: str) -> bool:
        """Return True if this plugin handles the given model_type."""
        ...

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        """Load and preprocess weights for this family."""
        ...

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, verbose: bool = False,
    ) -> bytes:
        """Build TRT engine plan bytes."""
        ...

    # ------------------------------------------------------------------
    # Optional: Vision-Language support
    # ------------------------------------------------------------------

    def build_vision_engine(
        self, model_dir: str, config: ModelConfig, weights: WeightDict,
        *, verbose: bool = False,
    ) -> bytes | None:
        """Build TRT engine plan bytes for the vision encoder.

        Return None (default) if this is not a VL model. Plugins that
        support vision should override this to return serialized engine bytes,
        either via ONNX tracing (Strategy A) or manual graph_ops (Strategy B).
        """
        return None

    def get_vl_config(self, config: ModelConfig) -> dict | None:
        """Return VL config dict to inject into the bundle's config.json.

        Return None (default) if this is not a VL model. VL plugins should
        return a dict with keys like:
            image_token_id, fixed_image_size, num_image_pad_tokens,
            vision_output_dim, vl_prompt_template, image_token_str
        """
        return None
