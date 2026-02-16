"""Qwen2.5-VL family plugin — vision-language model.

Qwen2.5-VL is a two-engine vision-language model:
  1. Vision encoder (ViT with 3D RoPE + spatial merge) — built via manual graph_ops
  2. Text decoder (standard Qwen2.5 architecture with QKV biases)

The text decoder uses embed_input mode: during VL prefill, it accepts
pre-computed embeddings (fused text + vision features) instead of token IDs.

The model config has a nested "vision_config" dict which is used to build
the vision encoder.
"""

from __future__ import annotations

from ..config import ModelConfig
from ..checkpoint_mapper import WeightDict, load_standard_weights
from ..standard_decoder_builder import build_standard_decoder_engine


# Default fixed image size for the vision encoder (matches Qwen2.5-VL default)
_DEFAULT_FIXED_IMAGE_SIZE = 448


class QwenVLPlugin:
    name = "qwen_vl"
    runtime_strategy = "vision_language"
    embed_input = True

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        # Match qwen2_5_vl, qwen2_vl, qwen_vl, etc.
        return "qwen" in mt and "vl" in mt

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        return load_standard_weights(model_dir, config)

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, verbose: bool = False,
        debug_layer_outputs: bool = False,
    ) -> bytes:
        return build_standard_decoder_engine(
            config, weights, max_cache_length, verbose=verbose,
            embed_input=True,
            debug_layer_outputs=debug_layer_outputs)

    def build_vision_engine(
        self, model_dir: str, config: ModelConfig, weights: WeightDict,
        *, verbose: bool = False,
    ) -> bytes | None:
        """Build complete vision encoder TRT engine (patch_embed -> ViT -> merge -> projection)."""
        vision_config = config.raw.get("vision_config")
        if vision_config is None:
            return None

        from ..vision_encoder_builder import build_qwen_vl_vision_engine

        # Load vision-specific weights from safetensors.
        vision_weights = _load_vision_weights(model_dir, config)
        return build_qwen_vl_vision_engine(
            vision_config, vision_weights,
            fixed_image_size=_DEFAULT_FIXED_IMAGE_SIZE,
            verbose=verbose)

    def get_vl_config(self, config: ModelConfig) -> dict | None:
        """Return VL-specific config fields to inject into bundle's config.json."""
        vision_config = config.raw.get("vision_config")
        if vision_config is None:
            return None

        patch_size = vision_config.get("patch_size", 14)
        merge_size = vision_config.get("spatial_merge_size", 2)
        fixed_image_size = _DEFAULT_FIXED_IMAGE_SIZE

        grid_h = fixed_image_size // patch_size
        grid_w = fixed_image_size // patch_size
        num_patches = grid_h * grid_w
        num_merged = num_patches // (merge_size * merge_size)

        return {
            "image_token_id": 151655,  # <|image_pad|> in Qwen2.5-VL
            "fixed_image_size": fixed_image_size,
            "num_image_pad_tokens": num_merged,
            "vision_output_dim": config.hidden_size,
            "vl_prompt_template": (
                "<|im_start|>user\n"
                "<|vision_start|>{image_pads}<|vision_end|>\n"
                "{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
            "image_token_str": "<|image_pad|>",
        }


def _load_vision_weights(model_dir: str, config: ModelConfig) -> WeightDict:
    """Load vision encoder weights (visual.* prefix) from safetensors."""
    from pathlib import Path
    from ..checkpoint_mapper import WeightDict, _open_safetensors, _load_tensor

    model_dir_path = Path(model_dir)
    readers = _open_safetensors(model_dir_path)

    weights = WeightDict()
    for reader in readers:
        for key in reader.keys():
            if key.startswith("visual."):
                weights[key] = _load_tensor([reader], key)

    return weights


plugin = QwenVLPlugin()
