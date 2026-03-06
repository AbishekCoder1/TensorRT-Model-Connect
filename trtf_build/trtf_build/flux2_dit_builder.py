"""FLUX.2-dev DiT (Diffusion Transformer) engine builder.

Builds a TensorRT engine for the FLUX.2-dev transformer denoiser,
which has two types of blocks:
  1. Joint transformer blocks (double-stream): image and text attend jointly
  2. Single transformer blocks (single-stream): operate on concatenated tokens

Key differences from FLUX.1:
  - Inner dim = 6144 (48 heads x 128), joint = 8, single = 48
  - MLP ratio = 3.0 (vs 4.0)
  - 4D RoPE (32,32,32,32) vs 3D (16,56,56)
  - Global modulation inputs instead of per-block norm1.linear
  - Fused to_qkv_mlp_proj in single blocks
  - FFN uses .linear_in / .linear_out naming
  - No pooled projections; joint attn dim matches dit_dim

Engine I/O:
    Inputs:
        hidden_states [num_img_tokens, dim] float32
        encoder_hidden_states [text_seq_len, dim] float32
        temb [dim] float32
        rotary_cos [total_seq_len, head_dim] float32
        rotary_sin [total_seq_len, head_dim] float32
    Outputs:
        output [num_img_tokens, out_channels] float32

Global modulation weights (double_stream_modulation_img/txt, single_stream_modulation)
are baked into the engine as constant linear projections from temb.

Preprocessor weights (timestep MLP, x_embedder, context_embedder, RoPE,
global modulation tables) are handled externally by the runtime.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt

from . import graph_ops

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict


def build_flux2_dit_engine(
    weights: "WeightDict",
    *,
    dim: int = 6144,
    num_heads: int = 48,
    num_layers: int = 8,
    num_single_layers: int = 48,
    num_img_tokens: int,
    text_seq_len: int = 512,
    mlp_ratio: float = 3.0,
    eps: float = 1e-6,
    verbose: bool = False,
) -> bytes:
    """Build FLUX.2-dev DiT denoiser TRT engine plan."""
    head_dim = dim // num_heads
    attn_scale = 1.0 / np.sqrt(head_dim)
    ffn_dim = int(dim * mlp_ratio)
    total_seq = text_seq_len + num_img_tokens

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 128 << 30)
    config.clear_flag(trt.BuilderFlag.TF32)

    network = builder.create_network()

    # --- Inputs ---
    hidden_inp = network.add_input(
        "hidden_states", trt.float32, (num_img_tokens, dim))
    encoder_inp = network.add_input(
        "encoder_hidden_states", trt.float32, (text_seq_len, dim))
    temb_inp = network.add_input(
        "temb", trt.float32, (dim,))
    rotary_cos = network.add_input(
        "rotary_cos", trt.float32, (total_seq, head_dim))
    rotary_sin = network.add_input(
        "rotary_sin", trt.float32, (total_seq, head_dim))

    # Constants
    eps_np = np.array([eps], dtype=np.float32)
    eps_t = graph_ops.add_constant(network, (1, 1), eps_np)
    scale_const = graph_ops.add_constant(
        network, (1, 1, 1), np.array([attn_scale], dtype=np.float32))

    # Build rotate-half matrix for RoPE
    rot_half_np = graph_ops.make_rotate_half_matrix(
        head_dim * num_heads, num_heads, interleaved=True)
    rot_half_const = graph_ops.add_constant(
        network, (head_dim * num_heads, head_dim * num_heads), rot_half_np)

    # Expand RoPE cos/sin from [total_seq, head_dim] to [total_seq, dim]
    cos_expand = _tile_rope_for_heads(network, rotary_cos, num_heads, total_seq, head_dim)
    sin_expand = _tile_rope_for_heads(network, rotary_sin, num_heads, total_seq, head_dim)

    # Split RoPE for text and image
    txt_cos = network.add_slice(cos_expand, (0, 0), (text_seq_len, dim), (1, 1)).get_output(0)
    txt_sin = network.add_slice(sin_expand, (0, 0), (text_seq_len, dim), (1, 1)).get_output(0)
    img_cos = network.add_slice(cos_expand, (text_seq_len, 0), (num_img_tokens, dim), (1, 1)).get_output(0)
    img_sin = network.add_slice(sin_expand, (text_seq_len, 0), (num_img_tokens, dim), (1, 1)).get_output(0)

    hidden = hidden_inp
    encoder_hidden = encoder_inp

    # --- Compute SiLU(temb) once for all modulation ---
    temb_silu = network.add_activation(temb_inp, trt.ActivationType.SIGMOID)
    temb_silu_out = network.add_elementwise(
        temb_inp, temb_silu.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)

    # --- Global modulation weights as constants ---
    # These are shared across all blocks: temb @ mod_weight -> [6*dim] or [3*dim]
    mod_img_w = weights.get("double_stream_modulation_img")
    mod_txt_w = weights.get("double_stream_modulation_txt")
    mod_single_w = weights.get("single_stream_modulation")

    # ===================== Joint Transformer Blocks =====================
    for layer_idx in range(num_layers):
        p = f"transformer_blocks.{layer_idx}"
        print(f"  [flux2-dit] Joint block {layer_idx}/{num_layers}", file=sys.stderr)

        # --- Global modulation: SiLU(temb) @ mod_weight ---
        # mod_img: [dim] @ [dim, 6*dim] -> [6*dim]
        mod_img_proj = _matmul_bias_1d_opt(network, temb_silu_out, dim, 6 * dim, mod_img_w)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = _chunk_6(
            network, mod_img_proj, dim)

        mod_txt_proj = _matmul_bias_1d_opt(network, temb_silu_out, dim, 6 * dim, mod_txt_w)
        c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = _chunk_6(
            network, mod_txt_proj, dim)

        # --- AdaLN-Zero for image ---
        normed_hidden = _adaln_modulate(
            network, hidden, scale_msa, shift_msa, dim, eps_t, num_img_tokens)

        # --- AdaLN-Zero for text ---
        normed_encoder = _adaln_modulate(
            network, encoder_hidden, c_scale_msa, c_shift_msa, dim, eps_t, text_seq_len)

        # --- Joint Attention ---
        # Image QKV
        q_img = _linear(network, normed_hidden, dim, dim, weights, f"{p}.attn.to_q")
        k_img = _linear(network, normed_hidden, dim, dim, weights, f"{p}.attn.to_k")
        v_img = _linear(network, normed_hidden, dim, dim, weights, f"{p}.attn.to_v")

        # Text QKV (added projections)
        q_txt = _linear(network, normed_encoder, dim, dim, weights, f"{p}.attn.add_q_proj")
        k_txt = _linear(network, normed_encoder, dim, dim, weights, f"{p}.attn.add_k_proj")
        v_txt = _linear(network, normed_encoder, dim, dim, weights, f"{p}.attn.add_v_proj")

        # QK norm
        q_img = _rms_norm_per_head_seq(network, q_img, num_heads, head_dim, weights[f"{p}.attn.norm_q.weight"], eps_t, num_img_tokens)
        k_img = _rms_norm_per_head_seq(network, k_img, num_heads, head_dim, weights[f"{p}.attn.norm_k.weight"], eps_t, num_img_tokens)
        q_txt = _rms_norm_per_head_seq(network, q_txt, num_heads, head_dim, weights[f"{p}.attn.norm_added_q.weight"], eps_t, text_seq_len)
        k_txt = _rms_norm_per_head_seq(network, k_txt, num_heads, head_dim, weights[f"{p}.attn.norm_added_k.weight"], eps_t, text_seq_len)

        # Apply RoPE to image Q, K
        q_img = _apply_rope(network, q_img, img_cos, img_sin, rot_half_const)
        k_img = _apply_rope(network, k_img, img_cos, img_sin, rot_half_const)

        # Apply RoPE to text Q, K
        q_txt = _apply_rope(network, q_txt, txt_cos, txt_sin, rot_half_const)
        k_txt = _apply_rope(network, k_txt, txt_cos, txt_sin, rot_half_const)

        # Concatenate: [text, image] for joint attention
        q_cat = network.add_concatenation([q_txt, q_img])
        q_cat.axis = 0  # [total_seq, dim]
        k_cat = network.add_concatenation([k_txt, k_img])
        k_cat.axis = 0
        v_cat = network.add_concatenation([v_txt, v_img])
        v_cat.axis = 0

        # Multi-head attention
        attn_out = _mha(network, q_cat.get_output(0), k_cat.get_output(0),
                       v_cat.get_output(0), num_heads, head_dim, total_seq,
                       scale_const)

        # Split attention output back into text and image
        txt_attn = network.add_slice(attn_out, (0, 0), (text_seq_len, dim), (1, 1)).get_output(0)
        img_attn = network.add_slice(attn_out, (text_seq_len, 0), (num_img_tokens, dim), (1, 1)).get_output(0)

        # Image output projection + gate + residual
        img_attn_proj = _linear(network, img_attn, dim, dim, weights, f"{p}.attn.to_out.0")
        img_attn_gated = _gate_1d(network, img_attn_proj, gate_msa, num_img_tokens)
        hidden = network.add_elementwise(
            hidden, img_attn_gated,
            trt.ElementWiseOperation.SUM).get_output(0)

        # Text output projection + gate + residual
        txt_attn_proj = _linear(network, txt_attn, dim, dim, weights, f"{p}.attn.to_add_out")
        txt_attn_gated = _gate_1d(network, txt_attn_proj, c_gate_msa, text_seq_len)
        encoder_hidden = network.add_elementwise(
            encoder_hidden, txt_attn_gated,
            trt.ElementWiseOperation.SUM).get_output(0)

        # --- Image FFN (linear_in / linear_out naming) ---
        normed_ff = _adaln_modulate(
            network, hidden, scale_mlp, shift_mlp, dim, eps_t, num_img_tokens)
        ff_out = _swiglu_ffn(network, normed_ff, dim, weights, f"{p}.ff")
        ff_gated = _gate_1d(network, ff_out, gate_mlp, num_img_tokens)
        hidden = network.add_elementwise(
            hidden, ff_gated,
            trt.ElementWiseOperation.SUM).get_output(0)

        # --- Text FFN (linear_in / linear_out naming) ---
        normed_ctx_ff = _adaln_modulate(
            network, encoder_hidden, c_scale_mlp, c_shift_mlp, dim, eps_t, text_seq_len)
        ctx_ff_out = _swiglu_ffn(network, normed_ctx_ff, dim, weights, f"{p}.ff_context")
        ctx_ff_gated = _gate_1d(network, ctx_ff_out, c_gate_mlp, text_seq_len)
        encoder_hidden = network.add_elementwise(
            encoder_hidden, ctx_ff_gated,
            trt.ElementWiseOperation.SUM).get_output(0)

    # ===================== Single Transformer Blocks =====================
    for layer_idx in range(num_single_layers):
        p = f"single_transformer_blocks.{layer_idx}"
        if layer_idx % 8 == 0:
            print(f"  [flux2-dit] Single block {layer_idx}/{num_single_layers}", file=sys.stderr)

        # Concatenate text + image
        cat_hidden = network.add_concatenation([encoder_hidden, hidden])
        cat_hidden.axis = 0  # [total_seq, dim]
        residual = cat_hidden.get_output(0)

        # --- Global modulation: SiLU(temb) @ mod_single_weight ---
        mod_single_proj = _matmul_bias_1d_opt(network, temb_silu_out, dim, 3 * dim, mod_single_w)
        shift_msa_s, scale_msa_s, gate_msa_s = _chunk_3(
            network, mod_single_proj, dim)

        # AdaLN-Zero modulation
        normed_cat = _adaln_modulate(
            network, residual, scale_msa_s, shift_msa_s, dim, eps_t, total_seq)

        # --- Fused QKV + MLP projection ---
        # to_qkv_mlp_proj: [dim, 3*dim + 2*ffn_dim]  (gated MLP: gate + value)
        fused_out_dim = 3 * dim + 2 * ffn_dim
        fused_w = weights[f"{p}.attn.to_qkv_mlp_proj.weight"]
        fused = graph_ops.add_matmul_rhs_constant(
            network, normed_cat, dim, fused_out_dim, fused_w)
        fused_b = weights.get(f"{p}.attn.to_qkv_mlp_proj.bias")
        if fused_b is not None:
            fused = graph_ops.add_bias_sum(network, fused, fused_out_dim, fused_b)

        # Slice: Q [dim], K [dim], V [dim], MLP_gate [ffn_dim], MLP_value [ffn_dim]
        q_s = network.add_slice(fused, (0, 0), (total_seq, dim), (1, 1)).get_output(0)
        k_s = network.add_slice(fused, (0, dim), (total_seq, dim), (1, 1)).get_output(0)
        v_s = network.add_slice(fused, (0, 2 * dim), (total_seq, dim), (1, 1)).get_output(0)
        mlp_gate = network.add_slice(fused, (0, 3 * dim), (total_seq, ffn_dim), (1, 1)).get_output(0)
        mlp_value = network.add_slice(fused, (0, 3 * dim + ffn_dim), (total_seq, ffn_dim), (1, 1)).get_output(0)

        # SwiGLU on MLP branch: silu(x1) * x2
        mlp_gate_act = graph_ops.add_activation(network, mlp_gate, "silu")
        mlp_hidden = network.add_elementwise(
            mlp_gate_act, mlp_value, trt.ElementWiseOperation.PROD).get_output(0)

        # QK norm
        q_s = _rms_norm_per_head_seq(network, q_s, num_heads, head_dim, weights[f"{p}.attn.norm_q.weight"], eps_t, total_seq)
        k_s = _rms_norm_per_head_seq(network, k_s, num_heads, head_dim, weights[f"{p}.attn.norm_k.weight"], eps_t, total_seq)

        # Apply RoPE (full sequence: text + image cos/sin)
        q_s = _apply_rope(network, q_s, cos_expand, sin_expand, rot_half_const)
        k_s = _apply_rope(network, k_s, cos_expand, sin_expand, rot_half_const)

        attn_out_s = _mha(network, q_s, k_s, v_s, num_heads, head_dim,
                         total_seq, scale_const)

        # Concatenate attn + mlp -> to_out projection
        cat_attn_mlp = network.add_concatenation([attn_out_s, mlp_hidden])
        cat_attn_mlp.axis = 1  # [total_seq, dim + ffn_dim]

        to_out_w = weights[f"{p}.attn.to_out.weight"]
        in_features = dim + ffn_dim
        combined = graph_ops.add_matmul_rhs_constant(
            network, cat_attn_mlp.get_output(0), in_features, dim, to_out_w)
        to_out_b = weights.get(f"{p}.attn.to_out.bias")
        if to_out_b is not None:
            combined = graph_ops.add_bias_sum(network, combined, dim, to_out_b)

        # Gate + residual
        gated_s = _gate_1d(network, combined, gate_msa_s, total_seq)
        cat_hidden_out = network.add_elementwise(
            residual, gated_s,
            trt.ElementWiseOperation.SUM).get_output(0)

        # Split back
        encoder_hidden = network.add_slice(
            cat_hidden_out, (0, 0), (text_seq_len, dim), (1, 1)).get_output(0)
        hidden = network.add_slice(
            cat_hidden_out, (text_seq_len, 0), (num_img_tokens, dim), (1, 1)).get_output(0)

    # ===================== Final Output =====================
    # AdaLayerNormContinuous: SiLU(temb) -> Linear(dim, 2*dim) -> chunk(2) -> scale, shift
    final_norm_w = weights["norm_out.linear.weight"]
    final_norm_b = weights.get("norm_out.linear.bias")

    temb_silu_f = network.add_activation(temb_inp, trt.ActivationType.SIGMOID)
    temb_silu_f_out = network.add_elementwise(
        temb_inp, temb_silu_f.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)

    final_proj = _matmul_bias_1d_opt(network, temb_silu_f_out, dim, 2 * dim, final_norm_w, final_norm_b)
    final_scale = network.add_slice(final_proj, (0,), (dim,), (1,)).get_output(0)
    final_shift = network.add_slice(final_proj, (dim,), (dim,), (1,)).get_output(0)

    output = _adaln_modulate(
        network, hidden, final_scale, final_shift, dim, eps_t, num_img_tokens)

    # proj_out: [num_img_tokens, dim] -> [num_img_tokens, out_channels]
    proj_out_w = weights["proj_out.weight"]
    out_channels = proj_out_w.shape[1]
    output = graph_ops.add_matmul_rhs_constant(
        network, output, dim, out_channels, proj_out_w)
    proj_out_b = weights.get("proj_out.bias")
    if proj_out_b is not None:
        output = graph_ops.add_bias_sum(network, output, out_channels, proj_out_b)

    output.name = "output"
    network.mark_output(output)
    output.dtype = trt.float32

    print(f"[flux2-dit] Building TRT engine "
          f"(dim={dim}, joint={num_layers}, single={num_single_layers}, "
          f"img_tokens={num_img_tokens}, text_seq={text_seq_len}) ...",
          file=sys.stderr)
    # Use verbose logging for this large model to capture TRT errors
    logger.min_severity = trt.Logger.INFO
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for FLUX.2 DiT")
    return bytes(plan)


# ============================================================================
# Helper functions
# ============================================================================

def _tile_rope_for_heads(network, rope, num_heads, seq_len, head_dim):
    """Tile [seq, head_dim] -> [seq, num_heads * head_dim]."""
    if num_heads == 1:
        return rope
    parts = [rope] * num_heads
    concat = network.add_concatenation(parts)
    concat.axis = 1
    return concat.get_output(0)


def _matmul_bias_1d(network, inp, in_dim, out_dim, weight, bias):
    """Matmul + bias for 1D input: [in_dim] -> [out_dim]."""
    inp_2d = network.add_shuffle(inp)
    inp_2d.reshape_dims = (1, in_dim)
    out = graph_ops.add_matmul_rhs_constant(
        network, inp_2d.get_output(0), in_dim, out_dim, weight)
    out = graph_ops.add_bias_sum(network, out, out_dim, bias)
    flat = network.add_shuffle(out)
    flat.reshape_dims = (out_dim,)
    return flat.get_output(0)


def _matmul_bias_1d_opt(network, inp, in_dim, out_dim, weight, bias=None):
    """Matmul + optional bias for 1D input: [in_dim] -> [out_dim]."""
    inp_2d = network.add_shuffle(inp)
    inp_2d.reshape_dims = (1, in_dim)
    out = graph_ops.add_matmul_rhs_constant(
        network, inp_2d.get_output(0), in_dim, out_dim, weight)
    if bias is not None:
        out = graph_ops.add_bias_sum(network, out, out_dim, bias)
    flat = network.add_shuffle(out)
    flat.reshape_dims = (out_dim,)
    return flat.get_output(0)


def _chunk_6(network, tensor, dim):
    """Split [6*dim] into 6 x [dim]."""
    chunks = []
    for i in range(6):
        s = network.add_slice(tensor, (i * dim,), (dim,), (1,))
        chunks.append(s.get_output(0))
    return chunks


def _chunk_3(network, tensor, dim):
    """Split [3*dim] into 3 x [dim]."""
    chunks = []
    for i in range(3):
        s = network.add_slice(tensor, (i * dim,), (dim,), (1,))
        chunks.append(s.get_output(0))
    return chunks


def _adaln_modulate(network, x, scale, shift, dim, eps_t, seq_len):
    """AdaLN: LayerNorm(x) * (1 + scale) + shift.
    x: [seq_len, dim], scale/shift: [dim] (1D from chunk)."""
    normed = graph_ops.add_layer_norm_no_affine(network, x, dim, eps_t)

    # Reshape scale/shift from [dim] to [1, dim] for broadcast with [seq_len, dim]
    scale_2d = network.add_shuffle(scale)
    scale_2d.reshape_dims = (1, dim)
    shift_2d = network.add_shuffle(shift)
    shift_2d.reshape_dims = (1, dim)

    one_const = graph_ops.add_constant(network, (1, 1), np.array([1.0], dtype=np.float32))
    scale_plus_1 = network.add_elementwise(
        one_const, scale_2d.get_output(0), trt.ElementWiseOperation.SUM).get_output(0)

    scaled = network.add_elementwise(
        normed, scale_plus_1, trt.ElementWiseOperation.PROD)
    shifted = network.add_elementwise(
        scaled.get_output(0), shift_2d.get_output(0), trt.ElementWiseOperation.SUM)
    return shifted.get_output(0)


def _linear(network, inp, in_dim, out_dim, weights, prefix):
    """Linear projection with optional bias."""
    out = graph_ops.add_matmul_rhs_constant(
        network, inp, in_dim, out_dim, weights[f"{prefix}.weight"])
    b = weights.get(f"{prefix}.bias")
    if b is not None:
        out = graph_ops.add_bias_sum(network, out, out_dim, b)
    return out


def _rms_norm_per_head_seq(network, x, num_heads, head_dim, weight, eps_t, seq_len):
    """Per-head RMS norm for [seq_len, dim] tensors with [head_dim] weights.

    Reshapes to [seq_len, num_heads, head_dim], applies RMS norm on head_dim axis,
    then reshapes back to [seq_len, dim].
    """
    dim = num_heads * head_dim

    # Reshape [seq_len, dim] -> [seq_len * num_heads, head_dim]
    reshaped = network.add_shuffle(x)
    reshaped.reshape_dims = (seq_len * num_heads, head_dim)

    # RMS norm on last axis (head_dim)
    reshaped_out = reshaped.get_output(0)
    sq = network.add_elementwise(reshaped_out, reshaped_out, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    denom_in = network.add_elementwise(
        mean.get_output(0), eps_t, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        reshaped_out, recip.get_output(0), trt.ElementWiseOperation.PROD)

    # Apply per-head gamma [1, head_dim]
    gamma_t = graph_ops.add_constant(network, (1, head_dim), weight)
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD)

    # Reshape back to [seq_len, dim]
    reshape_back = network.add_shuffle(scaled.get_output(0))
    reshape_back.reshape_dims = (seq_len, dim)
    return reshape_back.get_output(0)


def _apply_rope(network, x, cos_vals, sin_vals, rot_half_const):
    """Apply rotary position embedding."""
    x_rot = network.add_matrix_multiply(
        x, trt.MatrixOperation.NONE,
        rot_half_const, trt.MatrixOperation.NONE)
    x_cos = network.add_elementwise(
        x, cos_vals, trt.ElementWiseOperation.PROD)
    x_sin = network.add_elementwise(
        x_rot.get_output(0), sin_vals, trt.ElementWiseOperation.PROD)
    return network.add_elementwise(
        x_cos.get_output(0), x_sin.get_output(0),
        trt.ElementWiseOperation.SUM).get_output(0)


def _mha(network, q, k, v, num_heads, head_dim, seq_len, scale_const):
    """Multi-head attention: returns [seq_len, dim]."""
    dim = num_heads * head_dim

    q_h = network.add_shuffle(q)
    q_h.reshape_dims = (seq_len, num_heads, head_dim)
    q_h.second_transpose = trt.Permutation([1, 0, 2])

    k_h = network.add_shuffle(k)
    k_h.reshape_dims = (seq_len, num_heads, head_dim)
    k_h.second_transpose = trt.Permutation([1, 0, 2])

    v_h = network.add_shuffle(v)
    v_h.reshape_dims = (seq_len, num_heads, head_dim)
    v_h.second_transpose = trt.Permutation([1, 0, 2])

    score = network.add_matrix_multiply(
        q_h.get_output(0), trt.MatrixOperation.NONE,
        k_h.get_output(0), trt.MatrixOperation.TRANSPOSE)
    scaled = network.add_elementwise(
        score.get_output(0), scale_const,
        trt.ElementWiseOperation.PROD)
    softmax = network.add_softmax(scaled.get_output(0))
    softmax.axes = 1 << 2
    context = network.add_matrix_multiply(
        softmax.get_output(0), trt.MatrixOperation.NONE,
        v_h.get_output(0), trt.MatrixOperation.NONE)

    flat = network.add_shuffle(context.get_output(0))
    flat.first_transpose = trt.Permutation([1, 0, 2])
    flat.reshape_dims = (seq_len, dim)
    return flat.get_output(0)


def _gate_1d(network, x, gate, seq_len):
    """Gate: x * gate (broadcast gate [dim] over [seq_len, dim]).
    gate is [dim] (1D), x is [seq_len, dim] (2D). Reshape gate for broadcast."""
    # Reshape gate from [dim] to [1, dim] for TRT broadcast
    gate_2d = network.add_shuffle(gate)
    gate_2d.reshape_dims = (1, -1)
    return network.add_elementwise(
        x, gate_2d.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)


def _swiglu_ffn(network, inp, dim, weights, prefix):
    """SwiGLU FFN (FLUX.2 convention: Flux2SwiGLU).

    linear_in: [dim, 2*ffn_dim] -> split into x1 [ffn_dim] and x2 [ffn_dim]
    output = linear_out(silu(x1) * x2)

    Uses .linear_in / .linear_out naming.
    """
    fc1_w = weights[f"{prefix}.linear_in.weight"]
    double_ffn_dim = fc1_w.shape[1]  # 2 * ffn_dim
    ffn_dim = double_ffn_dim // 2

    fc1 = graph_ops.add_matmul_rhs_constant(network, inp, dim, double_ffn_dim, fc1_w)
    fc1_b = weights.get(f"{prefix}.linear_in.bias")
    if fc1_b is not None:
        fc1 = graph_ops.add_bias_sum(network, fc1, double_ffn_dim, fc1_b)

    # Split into x1 and x2 (SwiGLU: silu(x1) * x2)
    seq_len = inp.shape[0]
    x1 = network.add_slice(fc1, (0, 0), (seq_len, ffn_dim), (1, 1)).get_output(0)
    x2 = network.add_slice(fc1, (0, ffn_dim), (seq_len, ffn_dim), (1, 1)).get_output(0)

    # SiLU on x1, then multiply with x2
    gate_act = graph_ops.add_activation(network, x1, "silu")
    gated = network.add_elementwise(
        gate_act, x2, trt.ElementWiseOperation.PROD).get_output(0)

    fc2_w = weights[f"{prefix}.linear_out.weight"]
    fc2 = graph_ops.add_matmul_rhs_constant(network, gated, ffn_dim, dim, fc2_w)
    fc2_b = weights.get(f"{prefix}.linear_out.bias")
    if fc2_b is not None:
        fc2 = graph_ops.add_bias_sum(network, fc2, dim, fc2_b)
    return fc2


def load_flux2_dit_weights(
    model_dir: str,
    *,
    dim: int = 6144,
    num_heads: int = 48,
    num_layers: int = 8,
    num_single_layers: int = 48,
) -> "WeightDict":
    """Load FLUX.2-dev DiT weights from diffusers-format transformer directory."""
    from pathlib import Path
    from .checkpoint_mapper import WeightDict, _open_safetensors, _load_tensor, _has_tensor

    readers = _open_safetensors(Path(model_dir))
    weights = WeightDict()

    def _t(name):
        w = _load_tensor(readers, name)
        return np.ascontiguousarray(w.T, dtype=np.float32)

    def _f(name):
        return _load_tensor(readers, name).astype(np.float32)

    def _maybe_f(name):
        if _has_tensor(readers, name):
            return _f(name)
        return None

    def _maybe_t(name):
        if _has_tensor(readers, name):
            return _t(name)
        return None

    # --- Joint transformer blocks ---
    for i in range(num_layers):
        p = f"transformer_blocks.{i}"
        print(f"  [flux2-dit] Loading joint block {i} weights ...", file=sys.stderr)

        # Attention projections (image)
        for proj in ("to_q", "to_k", "to_v"):
            weights[f"{p}.attn.{proj}.weight"] = _t(f"{p}.attn.{proj}.weight")
            b = _maybe_f(f"{p}.attn.{proj}.bias")
            if b is not None:
                weights[f"{p}.attn.{proj}.bias"] = b
        weights[f"{p}.attn.to_out.0.weight"] = _t(f"{p}.attn.to_out.0.weight")
        b = _maybe_f(f"{p}.attn.to_out.0.bias")
        if b is not None:
            weights[f"{p}.attn.to_out.0.bias"] = b

        # Attention projections (text "added")
        for proj in ("add_q_proj", "add_k_proj", "add_v_proj"):
            weights[f"{p}.attn.{proj}.weight"] = _t(f"{p}.attn.{proj}.weight")
            b = _maybe_f(f"{p}.attn.{proj}.bias")
            if b is not None:
                weights[f"{p}.attn.{proj}.bias"] = b
        weights[f"{p}.attn.to_add_out.weight"] = _t(f"{p}.attn.to_add_out.weight")
        b = _maybe_f(f"{p}.attn.to_add_out.bias")
        if b is not None:
            weights[f"{p}.attn.to_add_out.bias"] = b

        # QK norms
        for norm in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
            w = _maybe_f(f"{p}.attn.{norm}.weight")
            if w is not None:
                weights[f"{p}.attn.{norm}.weight"] = w

        # FFN (image) — linear_in / linear_out naming
        weights[f"{p}.ff.linear_in.weight"] = _t(f"{p}.ff.linear_in.weight")
        b = _maybe_f(f"{p}.ff.linear_in.bias")
        if b is not None:
            weights[f"{p}.ff.linear_in.bias"] = b
        weights[f"{p}.ff.linear_out.weight"] = _t(f"{p}.ff.linear_out.weight")
        b = _maybe_f(f"{p}.ff.linear_out.bias")
        if b is not None:
            weights[f"{p}.ff.linear_out.bias"] = b

        # FFN (text context) — linear_in / linear_out naming
        weights[f"{p}.ff_context.linear_in.weight"] = _t(f"{p}.ff_context.linear_in.weight")
        b = _maybe_f(f"{p}.ff_context.linear_in.bias")
        if b is not None:
            weights[f"{p}.ff_context.linear_in.bias"] = b
        weights[f"{p}.ff_context.linear_out.weight"] = _t(f"{p}.ff_context.linear_out.weight")
        b = _maybe_f(f"{p}.ff_context.linear_out.bias")
        if b is not None:
            weights[f"{p}.ff_context.linear_out.bias"] = b

    # --- Single transformer blocks ---
    for i in range(num_single_layers):
        p = f"single_transformer_blocks.{i}"
        if i % 8 == 0:
            print(f"  [flux2-dit] Loading single block {i} weights ...", file=sys.stderr)

        # Fused QKV + MLP projection
        weights[f"{p}.attn.to_qkv_mlp_proj.weight"] = _t(f"{p}.attn.to_qkv_mlp_proj.weight")
        b = _maybe_f(f"{p}.attn.to_qkv_mlp_proj.bias")
        if b is not None:
            weights[f"{p}.attn.to_qkv_mlp_proj.bias"] = b

        # QK norms
        for norm in ("norm_q", "norm_k"):
            w = _maybe_f(f"{p}.attn.{norm}.weight")
            if w is not None:
                weights[f"{p}.attn.{norm}.weight"] = w

        # attn.to_out: projects concatenated [attn, mlp] back to dim
        weights[f"{p}.attn.to_out.weight"] = _t(f"{p}.attn.to_out.weight")
        b = _maybe_f(f"{p}.attn.to_out.bias")
        if b is not None:
            weights[f"{p}.attn.to_out.bias"] = b

    # --- Global ---
    weights["norm_out.linear.weight"] = _t("norm_out.linear.weight")
    b = _maybe_f("norm_out.linear.bias")
    if b is not None:
        weights["norm_out.linear.bias"] = b
    weights["proj_out.weight"] = _t("proj_out.weight")
    b = _maybe_f("proj_out.bias")
    if b is not None:
        weights["proj_out.bias"] = b

    # Preprocessor weights (external to TRT engine)
    weights["x_embedder.weight"] = _t("x_embedder.weight")
    b = _maybe_f("x_embedder.bias")
    if b is not None:
        weights["x_embedder.bias"] = b
    weights["context_embedder.weight"] = _t("context_embedder.weight")
    b = _maybe_f("context_embedder.bias")
    if b is not None:
        weights["context_embedder.bias"] = b

    # Timestep embedder MLPs — try both FLUX.2 naming conventions
    for prefix in ("time_guidance_embed", "time_text_embed"):
        for comp in ("timestep_embedder", "guidance_embedder"):
            for layer in ("linear_1", "linear_2"):
                key = f"{prefix}.{comp}.{layer}"
                if _has_tensor(readers, f"{key}.weight"):
                    # Store with canonical time_text_embed prefix for C++ runtime
                    canonical = f"time_text_embed.{comp}.{layer}"
                    weights[f"{canonical}.weight"] = _t(f"{key}.weight")
                    b = _maybe_f(f"{key}.bias")
                    if b is not None:
                        weights[f"{canonical}.bias"] = b

    # Global modulation weights (stored as preprocessor weights)
    # FLUX.2 uses {name}.linear.weight format
    for mod_key in ("double_stream_modulation_img",
                    "double_stream_modulation_txt",
                    "single_stream_modulation"):
        # Try both with and without .linear suffix
        if _has_tensor(readers, f"{mod_key}.linear.weight"):
            weights[mod_key] = _t(f"{mod_key}.linear.weight")
        elif _has_tensor(readers, mod_key):
            weights[mod_key] = _f(mod_key)

    return weights
