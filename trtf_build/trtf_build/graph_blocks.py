"""Composable architectural building blocks for TRT engine construction.

Layer 2 in the three-layer builder stack:

    graph_ops.py        Layer 1: Atomic TRT operations (tensor-in/tensor-out)
        |
    graph_blocks.py     Layer 2: Composable blocks (weight-aware)  <- THIS FILE
        |
    builders / plugins  Layer 3: Full engine assembly

Each block composes multiple graph_ops into a reusable sub-structure
(full attention block, SwiGLU MLP, GELU MLP, norm dispatch). Functions
accept a ``weights`` dict + ``prefix`` string to resolve weight names.

Blocks do NOT apply residual connections. Callers compose the residual
pattern, which is what varies across architectures (sequential vs parallel
residual, DeepStack injection, MoE routing, etc.).
"""

from __future__ import annotations

import os

from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt

from . import graph_ops

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from .quantization.context import QuantContext


# ---------------------------------------------------------------------------
# Precision boundary helpers (used by standard_decoder_builder, not inside
# blocks themselves).
# ---------------------------------------------------------------------------

def _make_matmul_fn(network, dtype, quant_ctx):
    """Create a matmul callable that routes through quant_ctx if present.

    Returns a function: (lhs, lhs_w, rhs_w, rhs_weights, weight_name) -> ITensor
    """
    if quant_ctx is None:
        def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
            return graph_ops.add_matmul_rhs_constant(
                network, lhs, lhs_w, rhs_w, rhs_weights, dtype=dtype)
        return matmul
    else:
        def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
            return quant_ctx.maybe_quantized_matmul(
                network, lhs, lhs_w, rhs_w, rhs_weights, weight_name,
                dtype=dtype)
        return matmul


def cast_to_fp32(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
) -> trt.ITensor:
    """Cast tensor to FP32 for numerically sensitive ops."""
    if tensor.dtype == trt.float32:
        return tensor
    return network.add_cast(tensor, trt.float32).get_output(0)


def cast_to_dtype(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    target_dtype: trt.DataType,
) -> trt.ITensor:
    """Cast tensor to target dtype (no-op if already matching)."""
    if tensor.dtype == target_dtype:
        return tensor
    return network.add_cast(tensor, target_dtype).get_output(0)


def apply_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray | None,
    eps_tensor: trt.ITensor,
    norm_type: str,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Dispatch to RMSNorm or LayerNorm based on norm_type."""
    if norm_type == "layernorm":
        if beta is None:
            beta = np.zeros(hidden_size, dtype=np.float32)
        return graph_ops.add_layer_norm(
            network, inp, hidden_size, gamma, beta, eps_tensor, dtype=dtype)
    else:
        return graph_ops.add_rms_norm(
            network, inp, hidden_size, gamma, eps_tensor, dtype=dtype)


def add_attention_block(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    position_id: trt.ITensor,
    *,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    attention_size: int,
    num_heads: int,
    head_dim: int,
    max_cache_length: int,
    cos_tensor: trt.ITensor | None,
    sin_tensor: trt.ITensor | None,
    rotate_half_tensor: trt.ITensor | None,
    attn_scale_tensor: trt.ITensor,
    eps_tensor: trt.ITensor,
    norm_type: str = "rmsnorm",
    position_type: str = "rope",
    alibi_slopes_tensor: trt.ITensor | None = None,
    alibi_indices_tensor: trt.ITensor | None = None,
    dtype: np.dtype = np.float32,
    quant_ctx: QuantContext | None = None,
    layer_prefix: str = "",
    # TRT 10 native API tensors (optional — when provided, replaces manual
    # rotate-half RoPE and the score/softmax/V chain with fused kernels).
    cos_half_tensor: trt.ITensor | None = None,
    sin_half_tensor: trt.ITensor | None = None,
    rotary_embedding_dim: int = 0,
    interleaved_rope: bool = False,
    ffi_attention_kernel: str | None = None,
    dynamic_kv_cache: bool = False,
) -> dict[str, trt.ITensor]:
    """Pre-norm -> QKV -> RoPE -> cache concat -> attention -> output proj.

    Returns {"normed": ..., "attn_out": ..., "present_k": ..., "present_v": ...}.
    Does NOT apply residual -- callers compose the residual pattern.

    When ``cos_half_tensor`` / ``sin_half_tensor`` are provided the function
    uses TRT 10 native APIs:
      - IRotaryEmbeddingLayer  (replaces the rotate-half matmul+elementwise chain)
      - IAttention             (replaces the Q@K^T → scale → mask → softmax → @V chain)
    Otherwise it falls back to the manual primitive implementation, which is
    used for ALiBi and other callers that have not yet been updated.
    """
    matmul = _make_matmul_fn(network, dtype, quant_ctx)
    attention_window = max_cache_length + 1

    # Weight name for quant scale lookup — use layer_prefix if provided,
    # otherwise fall back to the weights-dict prefix.
    _lp = layer_prefix or prefix

    # Pre-attention norm
    normed = apply_norm(
        network, hidden, hidden_size,
        weights[f"{prefix}.input_norm"],
        weights.get(f"{prefix}.input_norm_beta"),
        eps_tensor, norm_type, dtype=dtype)

    # QKV projections
    q = matmul(normed, hidden_size, attention_size,
               weights[f"{prefix}.w_q"], f"{_lp}.w_q")
    k = matmul(normed, hidden_size, attention_size,
               weights[f"{prefix}.w_k"], f"{_lp}.w_k")
    v = matmul(normed, hidden_size, attention_size,
               weights[f"{prefix}.w_v"], f"{_lp}.w_v")

    # Optional QKV biases
    q_bias = weights.get(f"{prefix}.q_bias")
    if q_bias is not None:
        q = graph_ops.add_bias_sum(network, q, attention_size, q_bias, dtype=dtype)
    k_bias = weights.get(f"{prefix}.k_bias")
    if k_bias is not None:
        k = graph_ops.add_bias_sum(network, k, attention_size, k_bias, dtype=dtype)
    v_bias = weights.get(f"{prefix}.v_bias")
    if v_bias is not None:
        v = graph_ops.add_bias_sum(network, v, attention_size, v_bias, dtype=dtype)

    # Optional per-head q/k norm
    q_norm = weights.get(f"{prefix}.q_norm")
    if q_norm is not None:
        q = graph_ops.add_rms_norm_per_head(
            network, q, num_heads, head_dim, q_norm, eps_tensor, dtype=dtype)
    k_norm = weights.get(f"{prefix}.k_norm")
    if k_norm is not None:
        k = graph_ops.add_rms_norm_per_head(
            network, k, num_heads, head_dim, k_norm, eps_tensor, dtype=dtype)

    # ------------------------------------------------------------------ #
    # RoPE — native IRotaryEmbeddingLayer or manual rotate-half fallback  #
    # ------------------------------------------------------------------ #
    use_native_rope = (
        position_type == "rope"
        and cos_half_tensor is not None
        and sin_half_tensor is not None
        and alibi_slopes_tensor is None  # ALiBi still uses manual path
    )
    force_manual_attention = os.getenv("TRTF_FORCE_MANUAL_DECODER_ATTENTION") == "1"
    use_native_attention = use_native_rope and not force_manual_attention

    if use_native_rope:
        q = graph_ops.add_apply_rope_native(
            network, q, num_heads, head_dim,
            cos_half_tensor, sin_half_tensor, position_id,
            rotary_embedding_dim or head_dim, interleaved_rope)
        k = graph_ops.add_apply_rope_native(
            network, k, num_heads, head_dim,
            cos_half_tensor, sin_half_tensor, position_id,
            rotary_embedding_dim or head_dim, interleaved_rope)
    elif position_type == "rope" and cos_tensor is not None:
        q = graph_ops.add_apply_rope(
            network, q, position_id, cos_tensor, sin_tensor,
            rotate_half_tensor)
        k = graph_ops.add_apply_rope(
            network, k, position_id, cos_tensor, sin_tensor,
            rotate_half_tensor)

    # Save present K/V (before concatenation, this is the raw projection output)
    present_k = k
    present_v = v

    # Reshape current K, V for concatenation
    k_reshape = network.add_shuffle(k)
    k_reshape.reshape_dims = (1, attention_size)
    v_reshape = network.add_shuffle(v)
    v_reshape.reshape_dims = (1, attention_size)

    # Concatenate with cache
    all_k = network.add_concatenation(
        [cache_k, k_reshape.get_output(0)])
    all_k.axis = 0
    all_v = network.add_concatenation(
        [cache_v, v_reshape.get_output(0)])
    all_v.axis = 0

    # ------------------------------------------------------------------ #
    # Attention core — native IAttention, FFI kernel, or manual fallback  #
    # ------------------------------------------------------------------ #
    if use_native_attention:
        # Reshape Q/K/V to [B=1, H, S, D] for IAttention.
        #
        # Q layout: [1, H*D] — heads are contiguous, so reshape directly.
        q_4d = network.add_shuffle(q)
        q_4d.reshape_dims = (1, num_heads, 1, head_dim)

        # K/V layout: [attention_window, H*D] — each row has all heads interleaved.
        # Direct reshape to [1, H, S, D] would mis-map heads and positions.
        # Correct: reshape to [S, H, D], transpose to [H, S, D], add batch dim.
        kv_3d_k = network.add_shuffle(all_k.get_output(0))
        kv_3d_k.reshape_dims = (
            (-1, num_heads, head_dim)
            if dynamic_kv_cache
            else (attention_window, num_heads, head_dim)
        )
        kv_3d_k.second_transpose = trt.Permutation([1, 0, 2])
        all_k_4d = network.add_shuffle(kv_3d_k.get_output(0))
        all_k_4d.reshape_dims = (
            (1, num_heads, -1, head_dim)
            if dynamic_kv_cache
            else (1, num_heads, attention_window, head_dim)
        )

        kv_3d_v = network.add_shuffle(all_v.get_output(0))
        kv_3d_v.reshape_dims = (
            (-1, num_heads, head_dim)
            if dynamic_kv_cache
            else (attention_window, num_heads, head_dim)
        )
        kv_3d_v.second_transpose = trt.Permutation([1, 0, 2])
        all_v_4d = network.add_shuffle(kv_3d_v.get_output(0))
        all_v_4d.reshape_dims = (
            (1, num_heads, -1, head_dim)
            if dynamic_kv_cache
            else (1, num_heads, attention_window, head_dim)
        )

        # Reshape mask [1, attention_window] → [1, 1, 1, attention_window]
        # for broadcast across the head dimension.
        mask_4d = network.add_shuffle(attention_mask)
        mask_4d.reshape_dims = (
            (1, 1, 1, -1)
            if dynamic_kv_cache
            else (1, 1, 1, attention_window)
        )

        ctx_4d = graph_ops._add_attention_core(
            network,
            q_4d.get_output(0),
            all_k_4d.get_output(0),
            all_v_4d.get_output(0),
            causal=False,
            mask=mask_4d.get_output(0),
        )

        # Reshape [1, num_heads, 1, head_dim] → [1, attention_size]
        context_flat = network.add_shuffle(ctx_4d)
        context_flat.reshape_dims = (1, attention_size)
        context = context_flat.get_output(0)
    elif ffi_attention_kernel is not None:
        # Fused attention kernel via TVM-FFI plugin
        context = graph_ops.add_decoder_attention_ffi(
            network, q, all_k.get_output(0), all_v.get_output(0),
            kernel_name=ffi_attention_kernel,
            num_heads=num_heads, head_dim=head_dim,
            attention_window=attention_window)
    elif not dynamic_kv_cache:
        # Standard decomposed attention path (master's extracted helper).
        context = graph_ops.add_decoder_attention_decomposed(
            network, q, all_k.get_output(0), all_v.get_output(0),
            attention_mask,
            num_heads=num_heads, head_dim=head_dim,
            attention_window=attention_window,
            attn_scale_tensor=attn_scale_tensor,
            alibi_slopes_tensor=alibi_slopes_tensor,
            alibi_indices_tensor=alibi_indices_tensor,
            position_id=position_id)
    else:
        # TriAttention dynamic-KV path: identical decomposition but the K/V
        # and mask reshapes use -1 so the engine accepts runtime-variable
        # cache lengths. Keep inline until the extracted helper also
        # supports a dynamic profile.
        q_heads = network.add_shuffle(q)
        q_heads.reshape_dims = (num_heads, 1, head_dim)

        k_heads = network.add_shuffle(all_k.get_output(0))
        k_heads.reshape_dims = (-1, num_heads, head_dim)
        v_heads = network.add_shuffle(all_v.get_output(0))
        v_heads.reshape_dims = (-1, num_heads, head_dim)

        k_heads.second_transpose = trt.Permutation([1, 0, 2])
        v_heads.second_transpose = trt.Permutation([1, 0, 2])

        score_q = q_heads.get_output(0)
        score_k = k_heads.get_output(0)
        attn_scale_value = attn_scale_tensor
        if score_q.dtype != trt.float32:
            score_q = network.add_cast(score_q, trt.float32).get_output(0)
            score_k = network.add_cast(score_k, trt.float32).get_output(0)
            attn_scale_value = network.add_cast(attn_scale_value, trt.float32).get_output(0)

        score = network.add_matrix_multiply(
            score_q, trt.MatrixOperation.NONE,
            score_k, trt.MatrixOperation.TRANSPOSE)

        scaled = network.add_elementwise(
            score.get_output(0), attn_scale_value,
            trt.ElementWiseOperation.PROD)

        if alibi_slopes_tensor is not None and alibi_indices_tensor is not None:
            pos_float = network.add_cast(position_id, trt.float32)
            pos_1d = network.add_shuffle(pos_float.get_output(0))
            pos_1d.reshape_dims = (1,)
            full_indices = network.add_concatenation(
                [alibi_indices_tensor, pos_1d.get_output(0)])
            full_indices.axis = 0
            idx_3d = network.add_shuffle(full_indices.get_output(0))
            idx_3d.reshape_dims = (1, 1, attention_window)
            pos_reshaped = network.add_shuffle(pos_float.get_output(0))
            pos_reshaped.reshape_dims = (1, 1, 1)
            rel_pos = network.add_elementwise(
                idx_3d.get_output(0), pos_reshaped.get_output(0),
                trt.ElementWiseOperation.SUB)
            alibi_slopes_value = alibi_slopes_tensor
            if alibi_slopes_value.dtype != rel_pos.get_output(0).dtype:
                alibi_slopes_value = network.add_cast(
                    alibi_slopes_value, rel_pos.get_output(0).dtype).get_output(0)
            alibi_bias = network.add_elementwise(
                alibi_slopes_value, rel_pos.get_output(0),
                trt.ElementWiseOperation.PROD)
            scaled = network.add_elementwise(
                scaled.get_output(0), alibi_bias.get_output(0),
                trt.ElementWiseOperation.SUM)

        mask3d = network.add_shuffle(attention_mask)
        mask3d.reshape_dims = (1, 1, -1)

        mask_value = mask3d.get_output(0)
        if mask_value.dtype != scaled.get_output(0).dtype:
            mask_value = network.add_cast(mask_value, scaled.get_output(0).dtype).get_output(0)

        mask_value = mask3d.get_output(0)
        if mask_value.dtype != scaled.get_output(0).dtype:
            mask_value = network.add_cast(mask_value, scaled.get_output(0).dtype).get_output(0)

        masked = network.add_elementwise(
            scaled.get_output(0), mask_value,
            trt.ElementWiseOperation.SUM)

        softmax = network.add_softmax(masked.get_output(0))
        softmax.axes = 1 << 2

        softmax_out = softmax.get_output(0)
        v_value = v_heads.get_output(0)
        if softmax_out.dtype != v_value.dtype:
            softmax_out = network.add_cast(softmax_out, v_value.dtype).get_output(0)

        context_heads = network.add_matrix_multiply(
            softmax_out, trt.MatrixOperation.NONE,
            v_value, trt.MatrixOperation.NONE)

        context_flat = network.add_shuffle(context_heads.get_output(0))
        context_flat.reshape_dims = (1, attention_size)
        context = context_flat.get_output(0)

    # Output projection
    attn_out = matmul(context,
                      attention_size, hidden_size,
                      weights[f"{prefix}.w_o"], f"{_lp}.w_o")

    # Optional output projection bias
    o_bias = weights.get(f"{prefix}.o_bias")
    if o_bias is not None:
        attn_out = graph_ops.add_bias_sum(network, attn_out, hidden_size, o_bias, dtype=dtype)

    return {
        "normed": normed,
        "attn_out": attn_out,
        "present_k": present_k,
        "present_v": present_v,
    }


def add_swiglu_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    mlp_size: int,
    dtype: np.dtype = np.float32,
    quant_ctx: QuantContext | None = None,
    layer_prefix: str = "",
) -> trt.ITensor:
    """Gate/up/down SwiGLU MLP. Returns output tensor."""
    matmul = _make_matmul_fn(network, dtype, quant_ctx)
    _lp = layer_prefix or prefix

    gate = matmul(inp, hidden_size, mlp_size,
                  weights[f"{prefix}.w_gate"], f"{_lp}.w_gate")
    up = matmul(inp, hidden_size, mlp_size,
                weights[f"{prefix}.w_up"], f"{_lp}.w_up")

    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(
        gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(
        swish.get_output(0), up, trt.ElementWiseOperation.PROD)

    mlp_out = matmul(gated.get_output(0), mlp_size, hidden_size,
                     weights[f"{prefix}.w_down"], f"{_lp}.w_down")
    return mlp_out


def add_gelu_fc_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    mlp_size: int,
    activation: str = "gelu_new",
    dtype: np.dtype = np.float32,
    quant_ctx: QuantContext | None = None,
    layer_prefix: str = "",
) -> trt.ITensor:
    """fc1 -> activation -> fc2 MLP. Returns output tensor."""
    matmul = _make_matmul_fn(network, dtype, quant_ctx)
    _lp = layer_prefix or prefix

    fc1 = matmul(inp, hidden_size, mlp_size,
                 weights[f"{prefix}.w_fc1"], f"{_lp}.w_fc1")
    fc1_bias = weights.get(f"{prefix}.fc1_bias")
    if fc1_bias is not None:
        fc1 = graph_ops.add_bias_sum(network, fc1, mlp_size, fc1_bias, dtype=dtype)

    activated = graph_ops.add_activation(network, fc1, activation, dtype=dtype)

    fc2 = matmul(activated, mlp_size, hidden_size,
                 weights[f"{prefix}.w_fc2"], f"{_lp}.w_fc2")
    fc2_bias = weights.get(f"{prefix}.fc2_bias")
    if fc2_bias is not None:
        fc2 = graph_ops.add_bias_sum(network, fc2, hidden_size, fc2_bias, dtype=dtype)

    return fc2


# ---------------------------------------------------------------------------
# Diffusion building blocks
# ---------------------------------------------------------------------------

def add_gated_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    mlp_size: int,
    activation: str = "gelu_new",
    dtype: np.dtype = np.float32,
    quant_ctx: QuantContext | None = None,
    layer_prefix: str = "",
) -> trt.ITensor:
    """Gated MLP: activation(fc1(x)) * fc1_gate(x), then fc2.

    Used by T5 encoder (gated GELU) and DiT FFN. Two parallel projections
    where one is gated by activation.

    Weight keys: {prefix}.w_fc1, {prefix}.w_fc1_gate, {prefix}.w_fc2
    Optional: {prefix}.fc1_bias, {prefix}.fc1_gate_bias, {prefix}.fc2_bias
    """
    matmul = _make_matmul_fn(network, dtype, quant_ctx)
    _lp = layer_prefix or prefix

    # Two parallel projections
    fc1 = matmul(inp, hidden_size, mlp_size,
                 weights[f"{prefix}.w_fc1"], f"{_lp}.w_fc1")
    fc1_bias = weights.get(f"{prefix}.fc1_bias")
    if fc1_bias is not None:
        fc1 = graph_ops.add_bias_sum(network, fc1, mlp_size, fc1_bias, dtype=dtype)

    fc1_gate = matmul(inp, hidden_size, mlp_size,
                      weights[f"{prefix}.w_fc1_gate"], f"{_lp}.w_fc1_gate")
    fc1_gate_bias = weights.get(f"{prefix}.fc1_gate_bias")
    if fc1_gate_bias is not None:
        fc1_gate = graph_ops.add_bias_sum(
            network, fc1_gate, mlp_size, fc1_gate_bias, dtype=dtype)

    # Gate: activation(fc1) * fc1_gate
    activated = graph_ops.add_activation(network, fc1, activation, dtype=dtype)
    gated = network.add_elementwise(
        activated, fc1_gate, trt.ElementWiseOperation.PROD)

    # Output projection
    fc2 = matmul(gated.get_output(0), mlp_size, hidden_size,
                 weights[f"{prefix}.w_fc2"], f"{_lp}.w_fc2")
    fc2_bias = weights.get(f"{prefix}.fc2_bias")
    if fc2_bias is not None:
        fc2 = graph_ops.add_bias_sum(network, fc2, hidden_size, fc2_bias, dtype=dtype)

    return fc2


def add_dit_block(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    context: trt.ITensor,
    adaln_params: trt.ITensor,
    *,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    context_dim: int,
    num_heads: int,
    q_seq_len: int,
    kv_seq_len: int,
    mlp_size: int,
    eps: float = 1e-5,
    dtype: np.dtype = np.float32,
    quant_ctx: QuantContext | None = None,
    layer_prefix: str = "",
) -> trt.ITensor:
    """DiT block: AdaLN self-attention + cross-attention + AdaLN FFN.

    adaln_params: [1, 6 * hidden_size] — from timestep MLP, split into
        (scale1, shift1, gate1, scale2, shift2, gate2) for self-attn and FFN.

    Weight keys expected:
        {prefix}.self_attn.w_q/w_k/w_v/w_o
        {prefix}.cross_attn.w_q/w_k/w_v/w_o
        {prefix}.ffn.w_fc1/w_fc1_gate/w_fc2
        {prefix}.norm_cross.gamma  (for cross-attn norm)

    Returns updated hidden state.
    """
    # Split AdaLN params: 6 chunks of hidden_size
    # [scale1, shift1, gate1, scale2, shift2, gate2]
    chunks = []
    for i in range(6):
        s = network.add_slice(
            adaln_params,
            start=(0, i * hidden_size),
            shape=(1, hidden_size),
            stride=(1, 1),
        )
        chunks.append(s.get_output(0))
    scale1, shift1, gate1, scale2, shift2, gate2 = chunks

    # --- Self-attention with AdaLN ---
    normed = graph_ops.add_adaptive_layernorm(
        network, hidden, scale1, shift1, hidden_size, eps, dtype=dtype)

    self_attn_out = graph_ops.add_self_attention_block(
        network, normed,
        w_q=weights[f"{prefix}.self_attn.w_q"],
        w_k=weights[f"{prefix}.self_attn.w_k"],
        w_v=weights[f"{prefix}.self_attn.w_v"],
        w_o=weights[f"{prefix}.self_attn.w_o"],
        hidden_size=hidden_size,
        num_heads=num_heads,
        seq_length=q_seq_len,
        dtype=dtype,
    )

    # Gate and residual
    gated_self_attn = network.add_elementwise(
        self_attn_out, gate1, trt.ElementWiseOperation.PROD)
    hidden = network.add_elementwise(
        hidden, gated_self_attn.get_output(0),
        trt.ElementWiseOperation.SUM).get_output(0)

    # --- Cross-attention (no AdaLN, uses standard LayerNorm) ---
    cross_norm_gamma = weights.get(f"{prefix}.norm_cross.gamma")
    if cross_norm_gamma is not None:
        eps_t = graph_ops.add_constant(
            network, (1, 1), np.array([eps], dtype=np.float32), dtype=dtype)
        cross_normed = graph_ops.add_layer_norm(
            network, hidden, hidden_size,
            cross_norm_gamma,
            weights.get(f"{prefix}.norm_cross.beta",
                        np.zeros(hidden_size, dtype=np.float32)),
            eps_t, dtype=dtype)
    else:
        cross_normed = hidden

    cross_attn_out = graph_ops.add_cross_attention(
        network, cross_normed, context,
        w_q=weights[f"{prefix}.cross_attn.w_q"],
        w_k=weights[f"{prefix}.cross_attn.w_k"],
        w_v=weights[f"{prefix}.cross_attn.w_v"],
        w_o=weights[f"{prefix}.cross_attn.w_o"],
        hidden_size=hidden_size,
        context_dim=context_dim,
        num_heads=num_heads,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        dtype=dtype,
    )

    hidden = network.add_elementwise(
        hidden, cross_attn_out,
        trt.ElementWiseOperation.SUM).get_output(0)

    # --- FFN with AdaLN ---
    ffn_normed = graph_ops.add_adaptive_layernorm(
        network, hidden, scale2, shift2, hidden_size, eps, dtype=dtype)

    ffn_out = add_gated_mlp(
        network, ffn_normed,
        weights=weights,
        prefix=f"{prefix}.ffn",
        hidden_size=hidden_size,
        mlp_size=mlp_size,
        activation="silu",
        dtype=dtype,
        quant_ctx=quant_ctx,
        layer_prefix=layer_prefix,
    )

    # Gate and residual
    gated_ffn = network.add_elementwise(
        ffn_out, gate2, trt.ElementWiseOperation.PROD)
    hidden = network.add_elementwise(
        hidden, gated_ffn.get_output(0),
        trt.ElementWiseOperation.SUM).get_output(0)

    return hidden


def add_vae_resblock_3d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    cache_in1: trt.ITensor,
    cache_in2: trt.ITensor,
    *,
    weights: WeightDict,
    prefix: str,
    in_channels: int,
    out_channels: int,
    norm_type: str = "group_norm",
    num_groups: int = 32,
    temporal_kernel: int = 3,
    eps: float = 1e-6,
    dtype: np.dtype = np.float32,
) -> tuple[trt.ITensor, trt.ITensor, trt.ITensor]:
    """3D VAE residual block with causal temporal convolutions.

    Input: [B, C_in, T, H, W] (T >= 1)
    cache_in1, cache_in2: temporal caches for the two causal convs

    Args:
        norm_type: "group_norm" uses GroupNorm with weight/bias keys,
                   "l2_channel_norm" uses L2 channel norm with gamma key.

    Returns: (output, updated_cache1, updated_cache2)

    Structure: Norm -> SiLU -> CausalConv3D -> Norm -> SiLU -> CausalConv3D + shortcut
    """
    def _apply_vae_norm(x, channels, norm_idx):
        if norm_type == "l2_channel_norm":
            return graph_ops.add_l2_channel_norm(
                network, x, channels,
                weights[f"{prefix}.norm{norm_idx}.gamma"], eps,
                dtype=dtype)
        else:
            return graph_ops.add_group_norm(
                network, x, channels, num_groups,
                weights[f"{prefix}.norm{norm_idx}.weight"],
                weights[f"{prefix}.norm{norm_idx}.bias"], eps,
                dtype=dtype)

    # First Norm + SiLU + CausalConv3D
    normed1 = _apply_vae_norm(inp, in_channels, 1)
    act1 = graph_ops.add_silu(network, normed1)
    conv1_out, cache_out1 = graph_ops.add_causal_conv3d(
        network, act1, cache_in1,
        weight=weights[f"{prefix}.conv1.weight"],
        bias=weights.get(f"{prefix}.conv1.bias"),
        out_channels=out_channels,
        kernel_size=(temporal_kernel, 3, 3),
        padding_hw=(1, 1),
        dtype=dtype,
    )

    # Second Norm + SiLU + CausalConv3D
    normed2 = _apply_vae_norm(conv1_out, out_channels, 2)
    act2 = graph_ops.add_silu(network, normed2)
    conv2_out, cache_out2 = graph_ops.add_causal_conv3d(
        network, act2, cache_in2,
        weight=weights[f"{prefix}.conv2.weight"],
        bias=weights.get(f"{prefix}.conv2.bias"),
        out_channels=out_channels,
        kernel_size=(temporal_kernel, 3, 3),
        padding_hw=(1, 1),
        dtype=dtype,
    )

    # Shortcut (1x1 conv if channel mismatch)
    # Weight key differs: l2_channel_norm models use "conv_shortcut", group_norm use "shortcut"
    if in_channels != out_channels:
        sc_key = f"{prefix}.conv_shortcut" if norm_type == "l2_channel_norm" else f"{prefix}.shortcut"
        shortcut = graph_ops.add_conv3d_as_conv2d(
            network, inp,
            weight=weights[f"{sc_key}.weight"],
            bias=weights.get(f"{sc_key}.bias"),
            out_channels=out_channels,
            kernel_size=(1, 1, 1),
            dtype=dtype,
        )
    else:
        shortcut = inp

    # Residual connection
    out = network.add_elementwise(
        conv2_out, shortcut, trt.ElementWiseOperation.SUM)

    return out.get_output(0), cache_out1, cache_out2


def add_vae_spatial_attention(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    weights: WeightDict,
    prefix: str,
    channels: int,
    norm_type: str = "l2_channel_norm",
    num_groups: int = 32,
    eps: float = 1e-6,
    dtype: np.dtype = np.float32,
    quant_ctx: QuantContext | None = None,
    layer_prefix: str = "",
) -> trt.ITensor:
    """VAE mid-block spatial self-attention with configurable norm.

    Single-head attention over spatial positions (H*W) per frame.

    Input: [B, C, T, H, W]
    Weight keys:
        {prefix}.norm.gamma           [C, 1, 1, 1]  (l2_channel_norm)
        {prefix}.norm.weight/.bias    [C]            (group_norm)
        {prefix}.to_qkv.weight        [3C, C, 1, 1, 1]
        {prefix}.to_qkv.bias          [3C]
        {prefix}.proj.weight           [C, C, 1, 1, 1]
        {prefix}.proj.bias             [C]

    Output: [B, C, T, H, W] (residual connection applied)
    """
    matmul = _make_matmul_fn(network, dtype, quant_ctx)
    _lp = layer_prefix or prefix

    b, c, t, h, w = inp.shape
    bt = b * t
    hw = h * w
    attn_scale = 1.0 / np.sqrt(max(c, 1))

    identity = inp

    # Configurable norm
    if norm_type == "l2_channel_norm":
        normed = graph_ops.add_l2_channel_norm(
            network, inp, channels,
            weights[f"{prefix}.norm.gamma"], eps, dtype=dtype)
    else:
        normed = graph_ops.add_group_norm(
            network, inp, channels, num_groups,
            weights[f"{prefix}.norm.weight"],
            weights[f"{prefix}.norm.bias"], eps, dtype=dtype)

    # Reshape [B, C, T, H, W] -> [B*T*H*W, C]  (2D for matmul compat)
    flatten = network.add_shuffle(normed)
    flatten.first_transpose = trt.Permutation([0, 2, 3, 4, 1])  # [B,T,H,W,C]
    flatten.reshape_dims = (bt * hw, c)

    # QKV projection: [BT*HW, C] @ [C, 3C] -> [BT*HW, 3C]
    qkv_w = weights[f"{prefix}.to_qkv.weight"]
    qkv_w_2d = qkv_w.reshape(3 * c, c).T.copy()
    qkv = matmul(flatten.get_output(0), c, 3 * c, qkv_w_2d,
                 f"{_lp}.to_qkv.weight")
    qkv_bias = weights.get(f"{prefix}.to_qkv.bias")
    if qkv_bias is not None:
        qkv = graph_ops.add_bias_sum(network, qkv, 3 * c, qkv_bias, dtype=dtype)

    # Reshape to [BT, HW, 3C] then split Q, K, V
    qkv_3d = network.add_shuffle(qkv)
    qkv_3d.reshape_dims = (bt, hw, 3 * c)

    q_slice = network.add_slice(
        qkv_3d.get_output(0),
        start=(0, 0, 0), shape=(bt, hw, c), stride=(1, 1, 1))
    k_slice = network.add_slice(
        qkv_3d.get_output(0),
        start=(0, 0, c), shape=(bt, hw, c), stride=(1, 1, 1))
    v_slice = network.add_slice(
        qkv_3d.get_output(0),
        start=(0, 0, 2 * c), shape=(bt, hw, c), stride=(1, 1, 1))

    q = q_slice.get_output(0)  # [BT, HW, C]
    k = k_slice.get_output(0)
    v = v_slice.get_output(0)

    # Attention: score = Q @ K^T / sqrt(C)  -> [BT, HW, HW]
    score = network.add_matrix_multiply(
        q, trt.MatrixOperation.NONE,
        k, trt.MatrixOperation.TRANSPOSE)

    scale_const = graph_ops.add_constant(
        network, (1, 1, 1), np.array([attn_scale], dtype=np.float32), dtype=dtype)
    scaled = network.add_elementwise(
        score.get_output(0), scale_const, trt.ElementWiseOperation.PROD)

    softmax = network.add_softmax(scaled.get_output(0))
    softmax.axes = 1 << 2  # last dim

    # Context = softmax @ V  -> [BT, HW, C]
    context = network.add_matrix_multiply(
        softmax.get_output(0), trt.MatrixOperation.NONE,
        v, trt.MatrixOperation.NONE)

    # Flatten context to 2D for output projection: [BT*HW, C]
    ctx_flat = network.add_shuffle(context.get_output(0))
    ctx_flat.reshape_dims = (bt * hw, c)

    # Output projection: [BT*HW, C] @ [C, C] -> [BT*HW, C]
    proj_w = weights[f"{prefix}.proj.weight"]
    proj_w_2d = proj_w.reshape(c, c).T.copy()
    proj_out = matmul(ctx_flat.get_output(0), c, c, proj_w_2d,
                      f"{_lp}.proj.weight")
    proj_bias = weights.get(f"{prefix}.proj.bias")
    if proj_bias is not None:
        proj_out = graph_ops.add_bias_sum(network, proj_out, c, proj_bias, dtype=dtype)

    # Reshape back to [B, C, T, H, W]
    reshape_out = network.add_shuffle(proj_out)
    reshape_out.reshape_dims = (b, t, h, w, c)
    reshape_out.second_transpose = trt.Permutation([0, 4, 1, 2, 3])  # [B,C,T,H,W]

    # Residual
    result = network.add_elementwise(
        reshape_out.get_output(0), identity, trt.ElementWiseOperation.SUM)

    return result.get_output(0)
