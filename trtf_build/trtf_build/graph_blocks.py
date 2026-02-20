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

from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt

from . import graph_ops

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict


def apply_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray | None,
    eps_tensor: trt.ITensor,
    norm_type: str,
) -> trt.ITensor:
    """Dispatch to RMSNorm or LayerNorm based on norm_type."""
    if norm_type == "layernorm":
        if beta is None:
            beta = np.zeros(hidden_size, dtype=np.float32)
        return graph_ops.add_layer_norm(
            network, inp, hidden_size, gamma, beta, eps_tensor)
    else:
        return graph_ops.add_rms_norm(
            network, inp, hidden_size, gamma, eps_tensor)


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
) -> dict[str, trt.ITensor]:
    """Pre-norm -> QKV -> RoPE -> cache concat -> attention -> output proj.

    Returns {"normed": ..., "attn_out": ..., "present_k": ..., "present_v": ...}.
    Does NOT apply residual -- callers compose the residual pattern.
    """
    attention_window = max_cache_length + 1

    # Pre-attention norm
    normed = apply_norm(
        network, hidden, hidden_size,
        weights[f"{prefix}.input_norm"],
        weights.get(f"{prefix}.input_norm_beta"),
        eps_tensor, norm_type)

    # QKV projections
    q = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, attention_size,
        weights[f"{prefix}.w_q"])
    k = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, attention_size,
        weights[f"{prefix}.w_k"])
    v = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, attention_size,
        weights[f"{prefix}.w_v"])

    # Optional QKV biases
    q_bias = weights.get(f"{prefix}.q_bias")
    if q_bias is not None:
        q = graph_ops.add_bias_sum(network, q, attention_size, q_bias)
    k_bias = weights.get(f"{prefix}.k_bias")
    if k_bias is not None:
        k = graph_ops.add_bias_sum(network, k, attention_size, k_bias)
    v_bias = weights.get(f"{prefix}.v_bias")
    if v_bias is not None:
        v = graph_ops.add_bias_sum(network, v, attention_size, v_bias)

    # Optional per-head q/k norm
    q_norm = weights.get(f"{prefix}.q_norm")
    if q_norm is not None:
        q = graph_ops.add_rms_norm_per_head(
            network, q, num_heads, head_dim, q_norm, eps_tensor)
    k_norm = weights.get(f"{prefix}.k_norm")
    if k_norm is not None:
        k = graph_ops.add_rms_norm_per_head(
            network, k, num_heads, head_dim, k_norm, eps_tensor)

    # Apply RoPE if using rotary positions
    if position_type == "rope" and cos_tensor is not None:
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

    # Reshape for multi-head attention
    q_heads = network.add_shuffle(q)
    q_heads.reshape_dims = (num_heads, 1, head_dim)

    k_heads = network.add_shuffle(all_k.get_output(0))
    k_heads.reshape_dims = (attention_window, num_heads, head_dim)
    v_heads = network.add_shuffle(all_v.get_output(0))
    v_heads.reshape_dims = (attention_window, num_heads, head_dim)

    # Transpose K, V to [num_heads, seq_len, head_dim]
    k_heads.second_transpose = trt.Permutation([1, 0, 2])
    v_heads.second_transpose = trt.Permutation([1, 0, 2])

    # Attention scores: Q @ K^T
    score = network.add_matrix_multiply(
        q_heads.get_output(0), trt.MatrixOperation.NONE,
        k_heads.get_output(0), trt.MatrixOperation.TRANSPOSE)

    # Scale
    scaled = network.add_elementwise(
        score.get_output(0), attn_scale_tensor,
        trt.ElementWiseOperation.PROD)

    # ALiBi bias
    if alibi_slopes_tensor is not None and alibi_indices_tensor is not None:
        pos_float = network.add_identity(position_id)
        pos_float.set_output_type(0, trt.float32)
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
        alibi_bias = network.add_elementwise(
            alibi_slopes_tensor, rel_pos.get_output(0),
            trt.ElementWiseOperation.PROD)
        scaled = network.add_elementwise(
            scaled.get_output(0), alibi_bias.get_output(0),
            trt.ElementWiseOperation.SUM)

    # Mask (reshape to [1, 1, attention_window])
    mask3d = network.add_shuffle(attention_mask)
    mask3d.reshape_dims = (1, 1, attention_window)

    masked = network.add_elementwise(
        scaled.get_output(0), mask3d.get_output(0),
        trt.ElementWiseOperation.SUM)

    # Softmax
    softmax = network.add_softmax(masked.get_output(0))
    softmax.axes = 1 << 2

    # Context: softmax @ V
    context_heads = network.add_matrix_multiply(
        softmax.get_output(0), trt.MatrixOperation.NONE,
        v_heads.get_output(0), trt.MatrixOperation.NONE)

    # Reshape back to [1, attention_size]
    context_flat = network.add_shuffle(context_heads.get_output(0))
    context_flat.reshape_dims = (1, attention_size)

    # Output projection
    attn_out = graph_ops.add_matmul_rhs_constant(
        network, context_flat.get_output(0),
        attention_size, hidden_size,
        weights[f"{prefix}.w_o"])

    # Optional output projection bias
    o_bias = weights.get(f"{prefix}.o_bias")
    if o_bias is not None:
        attn_out = graph_ops.add_bias_sum(network, attn_out, hidden_size, o_bias)

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
) -> trt.ITensor:
    """Gate/up/down SwiGLU MLP. Returns output tensor."""
    gate = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, mlp_size,
        weights[f"{prefix}.w_gate"])
    up = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, mlp_size,
        weights[f"{prefix}.w_up"])

    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(
        gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(
        swish.get_output(0), up, trt.ElementWiseOperation.PROD)

    mlp_out = graph_ops.add_matmul_rhs_constant(
        network, gated.get_output(0), mlp_size, hidden_size,
        weights[f"{prefix}.w_down"])
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
) -> trt.ITensor:
    """fc1 -> activation -> fc2 MLP. Returns output tensor."""
    fc1 = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, mlp_size,
        weights[f"{prefix}.w_fc1"])
    fc1_bias = weights.get(f"{prefix}.fc1_bias")
    if fc1_bias is not None:
        fc1 = graph_ops.add_bias_sum(network, fc1, mlp_size, fc1_bias)

    activated = graph_ops.add_activation(network, fc1, activation)

    fc2 = graph_ops.add_matmul_rhs_constant(
        network, activated, mlp_size, hidden_size,
        weights[f"{prefix}.w_fc2"])
    fc2_bias = weights.get(f"{prefix}.fc2_bias")
    if fc2_bias is not None:
        fc2 = graph_ops.add_bias_sum(network, fc2, hidden_size, fc2_bias)

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
) -> trt.ITensor:
    """Gated MLP: activation(fc1(x)) * fc1_gate(x), then fc2.

    Used by T5 encoder (gated GELU) and DiT FFN. Two parallel projections
    where one is gated by activation.

    Weight keys: {prefix}.w_fc1, {prefix}.w_fc1_gate, {prefix}.w_fc2
    Optional: {prefix}.fc1_bias, {prefix}.fc1_gate_bias, {prefix}.fc2_bias
    """
    # Two parallel projections
    fc1 = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, mlp_size,
        weights[f"{prefix}.w_fc1"])
    fc1_bias = weights.get(f"{prefix}.fc1_bias")
    if fc1_bias is not None:
        fc1 = graph_ops.add_bias_sum(network, fc1, mlp_size, fc1_bias)

    fc1_gate = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, mlp_size,
        weights[f"{prefix}.w_fc1_gate"])
    fc1_gate_bias = weights.get(f"{prefix}.fc1_gate_bias")
    if fc1_gate_bias is not None:
        fc1_gate = graph_ops.add_bias_sum(
            network, fc1_gate, mlp_size, fc1_gate_bias)

    # Gate: activation(fc1) * fc1_gate
    activated = graph_ops.add_activation(network, fc1, activation)
    gated = network.add_elementwise(
        activated, fc1_gate, trt.ElementWiseOperation.PROD)

    # Output projection
    fc2 = graph_ops.add_matmul_rhs_constant(
        network, gated.get_output(0), mlp_size, hidden_size,
        weights[f"{prefix}.w_fc2"])
    fc2_bias = weights.get(f"{prefix}.fc2_bias")
    if fc2_bias is not None:
        fc2 = graph_ops.add_bias_sum(network, fc2, hidden_size, fc2_bias)

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
        network, hidden, scale1, shift1, hidden_size, eps)

    self_attn_out = graph_ops.add_self_attention_block(
        network, normed,
        w_q=weights[f"{prefix}.self_attn.w_q"],
        w_k=weights[f"{prefix}.self_attn.w_k"],
        w_v=weights[f"{prefix}.self_attn.w_v"],
        w_o=weights[f"{prefix}.self_attn.w_o"],
        hidden_size=hidden_size,
        num_heads=num_heads,
        seq_length=q_seq_len,
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
            network, (1, 1), np.array([eps], dtype=np.float32))
        cross_normed = graph_ops.add_layer_norm(
            network, hidden, hidden_size,
            cross_norm_gamma,
            weights.get(f"{prefix}.norm_cross.beta",
                        np.zeros(hidden_size, dtype=np.float32)),
            eps_t)
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
    )

    hidden = network.add_elementwise(
        hidden, cross_attn_out,
        trt.ElementWiseOperation.SUM).get_output(0)

    # --- FFN with AdaLN ---
    ffn_normed = graph_ops.add_adaptive_layernorm(
        network, hidden, scale2, shift2, hidden_size, eps)

    ffn_out = add_gated_mlp(
        network, ffn_normed,
        weights=weights,
        prefix=f"{prefix}.ffn",
        hidden_size=hidden_size,
        mlp_size=mlp_size,
        activation="silu",
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
    num_groups: int = 32,
    temporal_kernel: int = 3,
    eps: float = 1e-6,
) -> tuple[trt.ITensor, trt.ITensor, trt.ITensor]:
    """3D VAE residual block with causal temporal convolutions.

    Input: [B, C_in, 1, H, W] (single frame)
    cache_in1, cache_in2: temporal caches for the two causal convs

    Returns: (output, updated_cache1, updated_cache2)

    Structure: GroupNorm -> SiLU -> CausalConv3D -> GroupNorm -> SiLU -> CausalConv3D + shortcut
    """
    # First GroupNorm + SiLU + CausalConv3D
    normed1 = graph_ops.add_group_norm(
        network, inp, in_channels, num_groups,
        weights[f"{prefix}.norm1.weight"],
        weights[f"{prefix}.norm1.bias"],
        eps)
    act1 = graph_ops.add_silu(network, normed1)
    conv1_out, cache_out1 = graph_ops.add_causal_conv3d(
        network, act1, cache_in1,
        weight=weights[f"{prefix}.conv1.weight"],
        bias=weights.get(f"{prefix}.conv1.bias"),
        out_channels=out_channels,
        kernel_size=(temporal_kernel, 3, 3),
        padding_hw=(1, 1),
    )

    # Second GroupNorm + SiLU + CausalConv3D
    normed2 = graph_ops.add_group_norm(
        network, conv1_out, out_channels, num_groups,
        weights[f"{prefix}.norm2.weight"],
        weights[f"{prefix}.norm2.bias"],
        eps)
    act2 = graph_ops.add_silu(network, normed2)
    conv2_out, cache_out2 = graph_ops.add_causal_conv3d(
        network, act2, cache_in2,
        weight=weights[f"{prefix}.conv2.weight"],
        bias=weights.get(f"{prefix}.conv2.bias"),
        out_channels=out_channels,
        kernel_size=(temporal_kernel, 3, 3),
        padding_hw=(1, 1),
    )

    # Shortcut (1x1 conv if channel mismatch)
    if in_channels != out_channels:
        shortcut = graph_ops.add_conv3d_as_conv2d(
            network, inp,
            weight=weights[f"{prefix}.shortcut.weight"],
            bias=weights.get(f"{prefix}.shortcut.bias"),
            out_channels=out_channels,
            kernel_size=(1, 1, 1),
        )
    else:
        shortcut = inp

    # Residual connection
    out = network.add_elementwise(
        conv2_out, shortcut, trt.ElementWiseOperation.SUM)

    return out.get_output(0), cache_out1, cache_out2
