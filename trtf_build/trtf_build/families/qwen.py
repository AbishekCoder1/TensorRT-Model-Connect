"""Qwen family plugin — Qwen, Qwen2, Qwen3, QwQ."""

from __future__ import annotations

from ..config import ModelConfig
from ..checkpoint_mapper import WeightDict, load_standard_weights
from ..standard_decoder_builder import build_standard_decoder_engine


class QwenPlugin:
    name = "qwen"

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        return mt.startswith("qwen") or mt.startswith("qwq")

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        return load_standard_weights(model_dir, config)

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, verbose: bool = False,
    ) -> bytes:
        return build_standard_decoder_engine(
            config, weights, max_cache_length, verbose=verbose)


plugin = QwenPlugin()
