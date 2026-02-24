"""FLUX family plugin.

Supports FLUX.1-dev, FLUX.1-schnell, FLUX.2-klein, and similar models
using the FluxPipeline from diffusers.

Components:
    - CLIP text encoder (pooled output for conditioning)
    - T5 text encoder (sequence output for cross-attention)
    - FLUX DiT denoiser (joint + single transformer blocks)
    - AutoencoderKL VAE decoder (2D, via subprocess)
"""

from __future__ import annotations

import sys

from ..config import ModelConfig
from ..checkpoint_mapper import WeightDict


class FluxPlugin:
    name = "flux"
    runtime_strategy = "diffusion"
    pipeline_classes = ["FluxPipeline", "Flux2Pipeline"]

    # Default FLUX.1-dev architecture params
    _CLIP_HIDDEN = 768
    _CLIP_HEADS = 12
    _CLIP_INTERMEDIATE = 3072
    _CLIP_LAYERS = 12
    _CLIP_VOCAB = 49408
    _CLIP_MAX_SEQ = 77

    _T5_D_MODEL = 4096
    _T5_NUM_HEADS = 64
    _T5_D_KV = 64
    _T5_D_FF = 10240
    _T5_NUM_LAYERS = 24
    _T5_VOCAB_SIZE = 32128
    _T5_MAX_SEQ_LEN = 512

    _DIT_DIM = 3072  # 24 heads * 128 head_dim
    _DIT_NUM_HEADS = 24
    _DIT_HEAD_DIM = 128
    _DIT_NUM_LAYERS = 19
    _DIT_NUM_SINGLE_LAYERS = 38
    _DIT_MLP_RATIO = 4.0
    _DIT_IN_CHANNELS = 64
    _DIT_PATCH_SIZE = 1
    _DIT_JOINT_ATTN_DIM = 4096
    _DIT_POOLED_PROJ_DIM = 768

    _AXES_DIMS_ROPE = (16, 56, 56)

    _VAE_LATENT_CHANNELS = 16
    _VAE_SCALING_FACTOR = 0.3611
    _VAE_SHIFT_FACTOR = 0.1159

    _IMAGE_HEIGHT = 1024
    _IMAGE_WIDTH = 1024

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        return mt in ("flux", "flux.1", "flux.2", "flux_t2i")

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        """Load weight paths from diffusers-format directory."""
        from pathlib import Path

        model_path = Path(model_dir)
        weights = WeightDict()

        if (model_path / "model_index.json").exists():
            weights["_model_format"] = "diffusers"
            if (model_path / "text_encoder").exists():
                weights["_text_encoder_dir"] = str(model_path / "text_encoder")
            if (model_path / "text_encoder_2").exists():
                weights["_text_encoder_2_dir"] = str(model_path / "text_encoder_2")
            weights["_transformer_dir"] = str(model_path / "transformer")
            weights["_vae_dir"] = str(model_path / "vae")
        else:
            raise ValueError(
                f"Expected diffusers format with model_index.json in {model_dir}")

        # Read transformer config to get exact architecture params
        import json
        transformer_config_path = model_path / "transformer" / "config.json"
        if transformer_config_path.exists():
            tc = json.loads(transformer_config_path.read_text())
            weights["_transformer_config"] = tc

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, verbose: bool = False,
    ) -> bytes:
        raise NotImplementedError(
            "FLUX uses build_components(), not build_engine()")

    def build_components(
        self, model_dir: str, config: ModelConfig, weights: WeightDict,
        *, verbose: bool = False,
    ) -> dict:
        """Build all component engines."""
        from ..t5_encoder_builder import build_t5_encoder_engine, load_t5_weights
        from ..clip_encoder_builder import build_clip_encoder_engine, load_clip_weights
        from ..flux_dit_builder import build_flux_dit_engine, load_flux_dit_weights
        import json
        from pathlib import Path

        transformer_dir = weights["_transformer_dir"]
        vae_dir = weights["_vae_dir"]

        # Read transformer config for exact params
        tc = weights.get("_transformer_config", {})
        dit_dim = tc.get("num_attention_heads", self._DIT_NUM_HEADS) * \
                  tc.get("attention_head_dim", self._DIT_HEAD_DIM)
        num_heads = tc.get("num_attention_heads", self._DIT_NUM_HEADS)
        num_layers = tc.get("num_layers", self._DIT_NUM_LAYERS)
        num_single_layers = tc.get("num_single_layers", self._DIT_NUM_SINGLE_LAYERS)
        in_channels = tc.get("in_channels", self._DIT_IN_CHANNELS)
        patch_size = tc.get("patch_size", self._DIT_PATCH_SIZE)
        joint_attn_dim = tc.get("joint_attention_dim", self._DIT_JOINT_ATTN_DIM)
        pooled_proj_dim = tc.get("pooled_projection_dim", self._DIT_POOLED_PROJ_DIM)
        guidance_embeds = tc.get("guidance_embeds", False)
        axes_dims_rope = tuple(tc.get("axes_dims_rope", self._AXES_DIMS_ROPE))

        # Image dimensions
        img_h = config.raw.get("image_height", self._IMAGE_HEIGHT)
        img_w = config.raw.get("image_width", self._IMAGE_WIDTH)
        h_lat = img_h // 8  # VAE spatial downscale = 8 (2^3 from 3 downsampling blocks)
        w_lat = img_w // 8
        # FLUX applies 2x2 packing to latents before the transformer:
        # [B, 16, H, W] -> [B, H//2 * W//2, 64]
        # So the effective patch size is 2 regardless of the config's patch_size=1
        pack_size = 2
        num_img_tokens = (h_lat // pack_size) * (w_lat // pack_size)

        text_encoders = []

        # 1. CLIP text encoder (produces pooled_output for timestep conditioning)
        clip_dir = weights.get("_text_encoder_dir")
        if clip_dir:
            # Check if this is CLIP or T5 by looking at config
            clip_config_path = Path(clip_dir) / "config.json"
            if clip_config_path.exists():
                clip_cfg = json.loads(clip_config_path.read_text())
                arch = clip_cfg.get("architectures", [""])[0]
                if "CLIP" in arch or clip_cfg.get("model_type") == "clip_text_model":
                    print("[flux] Loading CLIP encoder weights ...", file=sys.stderr)
                    clip_weights = load_clip_weights(
                        clip_dir,
                        hidden_size=clip_cfg.get("hidden_size", self._CLIP_HIDDEN),
                        num_layers=clip_cfg.get("num_hidden_layers", self._CLIP_LAYERS),
                    )
                    clip_plan = build_clip_encoder_engine(
                        clip_weights,
                        hidden_size=clip_cfg.get("hidden_size", self._CLIP_HIDDEN),
                        num_heads=clip_cfg.get("num_attention_heads", self._CLIP_HEADS),
                        intermediate_size=clip_cfg.get("intermediate_size", self._CLIP_INTERMEDIATE),
                        num_layers=clip_cfg.get("num_hidden_layers", self._CLIP_LAYERS),
                        vocab_size=clip_cfg.get("vocab_size", self._CLIP_VOCAB),
                        max_seq_len=clip_cfg.get("max_position_embeddings", self._CLIP_MAX_SEQ),
                        verbose=verbose,
                    )
                    text_encoders.append(("clip", clip_plan))
                else:
                    # text_encoder is T5 (FLUX.2-klein only has T5)
                    print("[flux] text_encoder is T5, loading ...", file=sys.stderr)
                    t5_weights = load_t5_weights(
                        clip_dir,
                        d_model=clip_cfg.get("d_model", self._T5_D_MODEL),
                        num_heads=clip_cfg.get("num_heads", self._T5_NUM_HEADS),
                        d_kv=clip_cfg.get("d_kv", self._T5_D_KV),
                        d_ff=clip_cfg.get("d_ff", self._T5_D_FF),
                        num_layers=clip_cfg.get("num_layers", self._T5_NUM_LAYERS),
                        vocab_size=clip_cfg.get("vocab_size", self._T5_VOCAB_SIZE),
                    )
                    t5_plan = build_t5_encoder_engine(
                        t5_weights,
                        d_model=clip_cfg.get("d_model", self._T5_D_MODEL),
                        num_heads=clip_cfg.get("num_heads", self._T5_NUM_HEADS),
                        d_kv=clip_cfg.get("d_kv", self._T5_D_KV),
                        d_ff=clip_cfg.get("d_ff", self._T5_D_FF),
                        num_layers=clip_cfg.get("num_layers", self._T5_NUM_LAYERS),
                        vocab_size=clip_cfg.get("vocab_size", self._T5_VOCAB_SIZE),
                        max_seq_len=self._T5_MAX_SEQ_LEN,
                        verbose=verbose,
                    )
                    text_encoders.append(("t5", t5_plan))

        # 2. T5 text encoder (sequence output for cross-attention)
        t5_dir = weights.get("_text_encoder_2_dir")
        if t5_dir:
            t5_config_path = Path(t5_dir) / "config.json"
            t5_cfg = {}
            if t5_config_path.exists():
                t5_cfg = json.loads(t5_config_path.read_text())

            print("[flux] Loading T5 encoder weights ...", file=sys.stderr)
            t5_d_model = t5_cfg.get("d_model", self._T5_D_MODEL)
            t5_num_heads = t5_cfg.get("num_heads", self._T5_NUM_HEADS)
            t5_d_kv = t5_cfg.get("d_kv", self._T5_D_KV)
            t5_d_ff = t5_cfg.get("d_ff", self._T5_D_FF)
            t5_num_layers = t5_cfg.get("num_layers", self._T5_NUM_LAYERS)
            t5_vocab_size = t5_cfg.get("vocab_size", self._T5_VOCAB_SIZE)

            t5_weights = load_t5_weights(
                t5_dir,
                d_model=t5_d_model,
                num_heads=t5_num_heads,
                d_kv=t5_d_kv,
                d_ff=t5_d_ff,
                num_layers=t5_num_layers,
                vocab_size=t5_vocab_size,
            )
            t5_plan = build_t5_encoder_engine(
                t5_weights,
                d_model=t5_d_model,
                num_heads=t5_num_heads,
                d_kv=t5_d_kv,
                d_ff=t5_d_ff,
                num_layers=t5_num_layers,
                vocab_size=t5_vocab_size,
                max_seq_len=self._T5_MAX_SEQ_LEN,
                verbose=verbose,
            )
            text_encoders.append(("t5", t5_plan))

        # 3. FLUX DiT denoiser
        print("[flux] Loading FLUX DiT weights ...", file=sys.stderr)
        dit_weights = load_flux_dit_weights(
            transformer_dir,
            dim=dit_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
        )

        dit_plan = build_flux_dit_engine(
            dit_weights,
            dim=dit_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            num_img_tokens=num_img_tokens,
            text_seq_len=self._T5_MAX_SEQ_LEN,
            verbose=verbose,
        )

        # 4. VAE decoder - native TRT engine via ONNX export
        from ..flux_vae_builder import build_flux_vae_decoder_engine
        vae_plan = build_flux_vae_decoder_engine(
            vae_dir,
            latent_channels=self._VAE_LATENT_CHANNELS,
            h_lat=h_lat,
            w_lat=w_lat,
            scaling_factor=self._VAE_SCALING_FACTOR,
            shift_factor=self._VAE_SHIFT_FACTOR,
            verbose=verbose,
        )

        # 5. Serialize preprocessor weights
        preprocessor_weights = _serialize_flux_preprocessor(dit_weights, guidance_embeds)

        return {
            "text_encoders": text_encoders,
            "denoiser": dit_plan,
            "vae_decoder": vae_plan,
            "preprocessor_weights": preprocessor_weights,
        }

    def get_diffusion_config(self, config: ModelConfig) -> dict:
        """Return diffusion pipeline configuration."""
        tc = config.raw.get("_transformer_config", {})
        guidance_embeds = tc.get("guidance_embeds", False)

        img_h = config.raw.get("image_height", self._IMAGE_HEIGHT)
        img_w = config.raw.get("image_width", self._IMAGE_WIDTH)

        return {
            "diffusion_backend_type": "flux_2d",
            "scheduler": "flow_match_euler",
            "num_inference_steps": 28 if guidance_embeds else 4,
            "guidance_scale": 3.5 if guidance_embeds else 0.0,
            "flow_shift": 3.0,
            "use_dynamic_shifting": 1,
            "base_shift": 0.5,
            "max_shift": 1.15,
            "image_height": img_h,
            "image_width": img_w,
            "video_height": img_h,
            "video_width": img_w,
            "video_num_frames": 1,
            "dit_dim": tc.get("num_attention_heads", self._DIT_NUM_HEADS) * \
                       tc.get("attention_head_dim", self._DIT_HEAD_DIM),
            "dit_num_heads": tc.get("num_attention_heads", self._DIT_NUM_HEADS),
            "dit_num_layers": tc.get("num_layers", self._DIT_NUM_LAYERS),
            "patch_size": [1, 2, 2],  # FLUX packs 2x2 latent patches into tokens
            "z_dim": self._VAE_LATENT_CHANNELS,
            "scale_factor_temporal": 1,
            "scale_factor_spatial": 8,
            "freq_dim": 256,
            "text_seq_len": self._T5_MAX_SEQ_LEN,
            "text_encoder_dim": self._T5_D_MODEL,
            "vae_scaling_factor": self._VAE_SCALING_FACTOR,
            "vae_shift_factor": self._VAE_SHIFT_FACTOR,
            "guidance_embeds": 1 if guidance_embeds else 0,
            "axes_dims_rope": list(tc.get("axes_dims_rope", self._AXES_DIMS_ROPE)),
            "num_vae_caches": 0,
            "vae_model_id": "sayakpaul/FLUX.1-merged",
        }


def _build_vae_placeholder(latent_channels, h_lat, w_lat, verbose):
    """Build a minimal VAE placeholder engine.

    Actual VAE decoding for FLUX is done via Python subprocess
    using diffusers AutoencoderKL. This placeholder engine exists
    only to satisfy the bundle format requirement for a vae_decoder_plan.
    """
    import tensorrt as trt
    import numpy as np

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)

    # Simple identity: input [C, H, W] -> output [C, H, W]
    inp = network.add_input("latents", trt.float32, (latent_channels, h_lat, w_lat))
    identity = network.add_identity(inp)
    out = identity.get_output(0)
    out.name = "output"
    network.mark_output(out)
    out.dtype = trt.float32

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("VAE placeholder engine build failed")
    return bytes(plan)


def _serialize_flux_preprocessor(dit_weights: dict, guidance_embeds: bool) -> bytes:
    """Serialize FLUX preprocessor weights.

    Uses Wan-compatible key names so the C++ parse_preprocessor_weights()
    can load them directly, plus FLUX-specific keys for x_embedder and
    context_embedder.
    """
    import json
    import struct
    import numpy as np

    # Map FLUX keys to Wan-compatible keys for the C++ parser
    key_map = {
        # x_embedder -> patch_embedding (C++ parser expects this)
        "x_embedder.weight": "patch_embedding.weight",
        "x_embedder.bias": "patch_embedding.bias",
        # Also write with original names for FLUX-specific parsing
        "context_embedder.weight": "context_embedder.weight",
        "context_embedder.bias": "context_embedder.bias",
        # timestep embedder -> condition_embedder.time_embedding
        "time_text_embed.timestep_embedder.linear_1.weight": "condition_embedder.time_embedding.0.weight",
        "time_text_embed.timestep_embedder.linear_1.bias": "condition_embedder.time_embedding.0.bias",
        "time_text_embed.timestep_embedder.linear_2.weight": "condition_embedder.time_embedding.2.weight",
        "time_text_embed.timestep_embedder.linear_2.bias": "condition_embedder.time_embedding.2.bias",
        # text embedder -> condition_embedder.text_embedding
        "time_text_embed.text_embedder.linear_1.weight": "condition_embedder.text_embedding.weight",
        "time_text_embed.text_embedder.linear_1.bias": "condition_embedder.text_embedding.bias",
        "time_text_embed.text_embedder.linear_2.weight": "condition_embedder.text_embedding_2.weight",
        "time_text_embed.text_embedder.linear_2.bias": "condition_embedder.text_embedding_2.bias",
    }

    guidance_keys = {}
    if guidance_embeds:
        guidance_keys = {
            "time_text_embed.guidance_embedder.linear_1.weight": "condition_embedder.guidance_embedding.0.weight",
            "time_text_embed.guidance_embedder.linear_1.bias": "condition_embedder.guidance_embedding.0.bias",
            "time_text_embed.guidance_embedder.linear_2.weight": "condition_embedder.guidance_embedding.2.weight",
            "time_text_embed.guidance_embedder.linear_2.bias": "condition_embedder.guidance_embedding.2.bias",
        }

    all_maps = {**key_map, **guidance_keys}

    index = {}
    data_parts = []
    offset = 0

    for src_key, dst_key in all_maps.items():
        if src_key not in dit_weights:
            continue
        w = dit_weights[src_key].astype(np.float32)
        w = np.ascontiguousarray(w)
        nbytes = w.nbytes
        index[dst_key] = {"offset": offset, "shape": list(w.shape)}
        data_parts.append(w.tobytes())
        offset += nbytes

    index_json = json.dumps(index).encode("utf-8")
    result = struct.pack("<I", len(index_json)) + index_json
    for part in data_parts:
        result += part

    return result


plugin = FluxPlugin()
