"""Shared Causal 3D VAE decoder engine builder.

Builds a TensorRT engine for a causal 3D VAE decoder that processes one
latent frame at a time, using temporal caches for causal convolutions.

Reusable by: Wan2.1, Hunyuan Video (same causal 3D VAE architecture).

For Wan2.1:
  base_dim=96, dim_mult=[1,2,4,4], z_dim=16
  Decoder channels (reversed): [384, 384, 192, 96]
  up_blocks: 4 levels, each with 3 resnets (num_res_blocks+1)
  Temporal upsample at blocks 0, 1 (reversed from encoder's [False, True, True])
  Spatial upsample at blocks 0, 1, 2 (not at last block)

Weight naming (diffusers WanDecoder3d):
  - Norms use `.gamma` (shape [C,1,1,1]) — WanRMS_norm, not GroupNorm
  - Resnets: `decoder.up_blocks.{i}.resnets.{j}.{conv1|conv2|norm1|norm2}`
  - Channel change shortcut: `resnets.{j}.conv_shortcut`
  - Spatial upsample: `upsamplers.0.resample.1` (2D Conv after nearest-upsample)
  - Temporal upsample: `upsamplers.0.time_conv` (CausalConv3D for pixel-shuffle-in-time)
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict


def load_vae_weights(
    model_dir: str,
    *,
    z_dim: int = 16,
    base_dim: int = 96,
    dim_mult: tuple[int, ...] = (1, 2, 4, 4),
    num_res_blocks: int = 2,
) -> "WeightDict":
    """Load VAE decoder weights from a diffusers-format vae directory.

    Only loads decoder weights (not encoder). Returns raw weight arrays
    (no transposition — conv weights are used as-is).
    """
    from pathlib import Path
    from .checkpoint_mapper import WeightDict, _open_safetensors, _load_tensor, _has_tensor

    model_path = Path(model_dir)
    readers = _open_safetensors(model_path)
    weights = WeightDict()

    def _w(name: str) -> np.ndarray:
        return _load_tensor(readers, name).astype(np.float32)

    def _maybe(name: str) -> np.ndarray | None:
        if _has_tensor(readers, name):
            return _w(name)
        return None

    # post_quant_conv [z_dim, z_dim, 1, 1, 1]
    weights["post_quant_conv.weight"] = _w("post_quant_conv.weight")
    weights["post_quant_conv.bias"] = _w("post_quant_conv.bias")

    # conv_in [384, 16, 3, 3, 3]
    weights["decoder.conv_in.weight"] = _w("decoder.conv_in.weight")
    weights["decoder.conv_in.bias"] = _w("decoder.conv_in.bias")

    # mid_block: 2 resnets + 1 attention
    for i in range(2):
        p = f"decoder.mid_block.resnets.{i}"
        weights[f"{p}.norm1.gamma"] = _w(f"{p}.norm1.gamma")
        weights[f"{p}.norm2.gamma"] = _w(f"{p}.norm2.gamma")
        weights[f"{p}.conv1.weight"] = _w(f"{p}.conv1.weight")
        weights[f"{p}.conv1.bias"] = _w(f"{p}.conv1.bias")
        weights[f"{p}.conv2.weight"] = _w(f"{p}.conv2.weight")
        weights[f"{p}.conv2.bias"] = _w(f"{p}.conv2.bias")

    # mid_block attention
    weights["decoder.mid_block.attentions.0.norm.gamma"] = _w(
        "decoder.mid_block.attentions.0.norm.gamma")
    weights["decoder.mid_block.attentions.0.to_qkv.weight"] = _w(
        "decoder.mid_block.attentions.0.to_qkv.weight")
    weights["decoder.mid_block.attentions.0.to_qkv.bias"] = _w(
        "decoder.mid_block.attentions.0.to_qkv.bias")
    weights["decoder.mid_block.attentions.0.proj.weight"] = _w(
        "decoder.mid_block.attentions.0.proj.weight")
    weights["decoder.mid_block.attentions.0.proj.bias"] = _w(
        "decoder.mid_block.attentions.0.proj.bias")

    # up_blocks
    num_levels = len(dim_mult)
    for level in range(num_levels):
        for blk in range(num_res_blocks + 1):
            p = f"decoder.up_blocks.{level}.resnets.{blk}"
            weights[f"{p}.norm1.gamma"] = _w(f"{p}.norm1.gamma")
            weights[f"{p}.norm2.gamma"] = _w(f"{p}.norm2.gamma")
            weights[f"{p}.conv1.weight"] = _w(f"{p}.conv1.weight")
            weights[f"{p}.conv1.bias"] = _w(f"{p}.conv1.bias")
            weights[f"{p}.conv2.weight"] = _w(f"{p}.conv2.weight")
            weights[f"{p}.conv2.bias"] = _w(f"{p}.conv2.bias")
            # Channel shortcut
            sc_w = _maybe(f"{p}.conv_shortcut.weight")
            if sc_w is not None:
                weights[f"{p}.conv_shortcut.weight"] = sc_w
                weights[f"{p}.conv_shortcut.bias"] = _w(f"{p}.conv_shortcut.bias")

        # Upsampler (spatial 2D conv + optional temporal)
        sp_w = _maybe(f"decoder.up_blocks.{level}.upsamplers.0.resample.1.weight")
        if sp_w is not None:
            weights[f"decoder.up_blocks.{level}.upsamplers.0.resample.1.weight"] = sp_w
            weights[f"decoder.up_blocks.{level}.upsamplers.0.resample.1.bias"] = _w(
                f"decoder.up_blocks.{level}.upsamplers.0.resample.1.bias")
        tc_w = _maybe(f"decoder.up_blocks.{level}.upsamplers.0.time_conv.weight")
        if tc_w is not None:
            weights[f"decoder.up_blocks.{level}.upsamplers.0.time_conv.weight"] = tc_w
            weights[f"decoder.up_blocks.{level}.upsamplers.0.time_conv.bias"] = _w(
                f"decoder.up_blocks.{level}.upsamplers.0.time_conv.bias")

    # Output norm + conv
    weights["decoder.norm_out.gamma"] = _w("decoder.norm_out.gamma")
    weights["decoder.conv_out.weight"] = _w("decoder.conv_out.weight")
    weights["decoder.conv_out.bias"] = _w("decoder.conv_out.bias")

    return weights


def count_vae_caches(
    dim_mult: tuple[int, ...] = (1, 2, 4, 4),
    num_res_blocks: int = 2,
    temporal_upsample: tuple[bool, ...] = (False, True, True),
) -> int:
    """Count causal conv cache slots needed for the VAE decoder.

    Each CausalConv3D with temporal kernel > 1 needs one cache.
    """
    count = 0
    num_levels = len(dim_mult)
    temp_up = list(reversed(temporal_upsample))

    # conv_in: CausalConv3d(kt=3) -> 1 cache
    count += 1

    # mid_block: 2 resnets × 2 causal convs each = 4 caches
    count += 4

    # up_blocks: each level has (num_res_blocks+1) resnets × 2 caches
    for level in range(num_levels):
        count += (num_res_blocks + 1) * 2

        # Spatial-only levels still have spatial upsample convs but those
        # use 2D conv (no temporal cache needed)
        # Temporal upsample: time_conv is CausalConv3d -> 1 cache
        if level < num_levels - 1:
            if level < len(temp_up) and temp_up[level]:
                count += 1

    # conv_out: CausalConv3d(kt=3) -> 1 cache
    count += 1

    return count


def build_causal_vae_3d_engine(
    weights: "WeightDict",
    *,
    z_dim: int = 16,
    base_dim: int = 96,
    dim_mult: tuple[int, ...] = (1, 2, 4, 4),
    num_res_blocks: int = 2,
    temporal_upsample: tuple[bool, ...] = (False, True, True),
    h_lat: int = 60,
    w_lat: int = 104,
    out_channels: int = 3,
    eps: float = 1e-6,
    verbose: bool = False,
) -> bytes:
    """Build causal 3D VAE decoder TRT engine plan.

    For now, this builds a placeholder engine that just passes through
    the latent frame. The full causal conv implementation requires
    significant graph complexity due to:
    - Temporal pixel shuffle in upsampler (time_conv → reshape → interleave)
    - 2D spatial upsample (nearest + conv2d) mixed with 3D temporal ops
    - Mid-block spatial attention
    - WanRMS_norm (not standard GroupNorm)

    A full implementation will be done incrementally. The placeholder
    allows the bundle to be built and the pipeline to execute end-to-end.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.clear_flag(trt.BuilderFlag.TF32)

    # Compute output spatial dims (8x upsample from latent)
    h_out = h_lat * 8
    w_out = w_lat * 8

    # Input: single latent frame [1, z_dim, 1, h_lat, w_lat]
    latent = network.add_input(
        "latent_frame", trt.float32, (1, z_dim, 1, h_lat, w_lat))

    # For the placeholder: just resize to output dims and project channels
    # This produces garbage output but validates the pipeline I/O
    from . import graph_ops

    # Reshape to [1, z_dim, h_lat, w_lat] for 2D operations
    reshape_2d = network.add_shuffle(latent)
    reshape_2d.reshape_dims = (1, z_dim, h_lat, w_lat)

    # Resize to output spatial dims
    resize = network.add_resize(reshape_2d.get_output(0))
    resize.resize_mode = trt.InterpolationMode.NEAREST
    resize.shape = (1, z_dim, h_out, w_out)

    # 1x1 conv to project z_dim -> out_channels
    proj_w = np.zeros((out_channels, z_dim, 1, 1), dtype=np.float32)
    # Initialize with simple channel averaging
    for c in range(out_channels):
        proj_w[c, :, 0, 0] = 1.0 / z_dim
    conv_w = trt.Weights(np.ascontiguousarray(proj_w))
    conv = network.add_convolution_nd(
        resize.get_output(0),
        num_output_maps=out_channels,
        kernel_shape=(1, 1),
        kernel=conv_w,
        bias=trt.Weights(),
    )

    # Reshape to [1, out_channels, 1, h_out, w_out]
    reshape_5d = network.add_shuffle(conv.get_output(0))
    reshape_5d.reshape_dims = (1, out_channels, 1, h_out, w_out)

    out_t = reshape_5d.get_output(0)
    out_t.name = "video_frame"
    network.mark_output(out_t)
    out_t.dtype = trt.float32

    # Build
    print(f"[vae-3d] Building placeholder TRT engine "
          f"(z_dim={z_dim}, latent={h_lat}x{w_lat} -> {h_out}x{w_out}) ...",
          file=sys.stderr)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for 3D VAE")
    return bytes(plan)
