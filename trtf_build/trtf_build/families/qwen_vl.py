"""Qwen2.5-VL family plugin — text decoder for vision-language model.

Qwen2.5-VL is a two-engine vision-language model:
  1. Vision encoder (ViT with windowed attention) -- NOT YET IMPLEMENTED
  2. Text decoder (standard Qwen2.5 architecture with QKV biases)

This plugin handles the TEXT DECODER only. The text decoder is architecturally
identical to Qwen2 (RMSNorm + SwiGLU + RoPE + GQA with QKV biases), so we
reuse load_standard_weights() and build_standard_decoder_engine().

Vision weights (prefixed "visual.*") are skipped by load_standard_weights()
since it only loads keys matching model.embed_tokens, model.layers.*, etc.

The model config has a nested "vision_config" dict which is stored in
config.raw for future use when we add the vision encoder.
"""

from __future__ import annotations

from ..config import ModelConfig
from ..checkpoint_mapper import WeightDict, load_standard_weights
from ..standard_decoder_builder import build_standard_decoder_engine


class QwenVLPlugin:
    name = "qwen_vl"

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        # Match qwen2_5_vl, qwen2_vl, qwen_vl, etc.
        return "qwen" in mt and "vl" in mt

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        # The text decoder weights follow standard Qwen2 layout:
        #   model.embed_tokens.weight
        #   model.layers.{i}.self_attn.{q,k,v,o}_proj.{weight,bias}
        #   model.layers.{i}.mlp.{gate,up,down}_proj.weight
        #   model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight
        #   model.norm.weight
        # Vision weights (visual.*) are naturally ignored.
        return load_standard_weights(model_dir, config)

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, verbose: bool = False,
        debug_layer_outputs: bool = False,
    ) -> bytes:
        return build_standard_decoder_engine(
            config, weights, max_cache_length, verbose=verbose,
            debug_layer_outputs=debug_layer_outputs)


plugin = QwenVLPlugin()
