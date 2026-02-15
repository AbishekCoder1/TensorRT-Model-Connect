"""FamilyPlugin protocol — defines the interface for model family plugins."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..config import ModelConfig
from ..checkpoint_mapper import WeightDict


class FamilyPlugin(Protocol):
    """Interface for a model family plugin."""

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
