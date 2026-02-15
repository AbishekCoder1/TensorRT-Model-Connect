"""1:1 port of src/runtime/trt/trt_graph_ops.cpp to Python TensorRT API.

Every function matches the C++ signature semantics exactly. Tensor names and
shapes must be identical so the C++ runtime can consume the built engine.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def layer_tensor_name(stem: str, layer: int) -> str:
    return f"{stem}_{layer}"


def add_constant(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    values: np.ndarray,
) -> trt.ITensor:
    """Add a constant tensor (float32)."""
    weights = trt.Weights(np.ascontiguousarray(values, dtype=np.float32))
    layer = network.add_constant(shape, weights)
    return layer.get_output(0)


def add_matmul_rhs_constant(
    network: trt.INetworkDefinition,
    lhs: trt.ITensor,
    lhs_width: int,
    rhs_width: int,
    rhs_weights: np.ndarray,
) -> trt.ITensor:
    """Matrix multiply: lhs @ rhs_constant.  rhs is [lhs_width, rhs_width]."""
    rhs = add_constant(network, (lhs_width, rhs_width), rhs_weights)
    mm = network.add_matrix_multiply(
        lhs, trt.MatrixOperation.NONE,
        rhs, trt.MatrixOperation.NONE,
    )
    return mm.get_output(0)


def add_bias_sum(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    width: int,
    bias: np.ndarray,
) -> trt.ITensor:
    """Element-wise add a [1, width] bias."""
    bias_t = add_constant(network, (1, width), bias)
    s = network.add_elementwise(inp, bias_t, trt.ElementWiseOperation.SUM)
    return s.get_output(0)


def add_rms_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    eps_tensor: trt.ITensor,
) -> trt.ITensor:
    """RMSNorm: gamma * (x / sqrt(mean(x^2) + eps))."""
    sq = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    denom_in = network.add_elementwise(
        mean.get_output(0), eps_tensor, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        inp, recip.get_output(0), trt.ElementWiseOperation.PROD)
    gamma_t = add_constant(network, (1, hidden_size), gamma)
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD)
    return scaled.get_output(0)


def add_rms_norm_per_head(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_heads: int,
    head_dim: int,
    gamma: np.ndarray,
    eps_tensor: trt.ITensor,
) -> trt.ITensor:
    """Per-head RMSNorm: reshape to [num_heads, head_dim], norm, reshape back."""
    reshape_in = network.add_shuffle(inp)
    reshape_in.reshape_dims = (num_heads, head_dim)

    reshaped = reshape_in.get_output(0)
    sq = network.add_elementwise(reshaped, reshaped, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    denom_in = network.add_elementwise(
        mean.get_output(0), eps_tensor, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        reshaped, recip.get_output(0), trt.ElementWiseOperation.PROD)
    gamma_t = add_constant(network, (num_heads, head_dim), gamma)
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD)

    reshape_out = network.add_shuffle(scaled.get_output(0))
    reshape_out.reshape_dims = (1, num_heads * head_dim)
    return reshape_out.get_output(0)


# ---------------------------------------------------------------------------
# RoPE tables — pure NumPy, no TRT dependency
# ---------------------------------------------------------------------------

def make_rope_table(
    max_cache_length: int,
    hidden_size: int,
    num_attention_heads: int,
    rope_theta: float,
    cosine: bool,
) -> np.ndarray:
    """Build cos or sin RoPE table of shape [max_cache_length, hidden_size]."""
    table = np.full(
        (max_cache_length, hidden_size),
        1.0 if cosine else 0.0,
        dtype=np.float32,
    )
    if (max_cache_length <= 0 or hidden_size <= 0
            or num_attention_heads <= 0
            or hidden_size % num_attention_heads != 0):
        return table

    head_dim = hidden_size // num_attention_heads
    half_head_dim = head_dim // 2
    if half_head_dim <= 0 or rope_theta <= 0.0:
        return table

    for pos in range(max_cache_length):
        for head in range(num_attention_heads):
            rope_dims = half_head_dim * 2
            for dim in range(rope_dims):
                freq_idx = dim % half_head_dim
                exponent = (2.0 * freq_idx) / head_dim
                inv_freq = rope_theta ** (-exponent)
                angle = pos * inv_freq
                value = np.cos(angle) if cosine else np.sin(angle)
                offset = head * head_dim + dim
                table[pos, offset] = value

    return table


def make_rotate_half_matrix(
    hidden_size: int,
    num_attention_heads: int,
) -> np.ndarray:
    """Build the rotate-half permutation matrix [hidden_size, hidden_size]."""
    matrix = np.zeros((hidden_size, hidden_size), dtype=np.float32)
    if (hidden_size <= 0 or num_attention_heads <= 0
            or hidden_size % num_attention_heads != 0):
        np.fill_diagonal(matrix, 1.0)
        return matrix

    head_dim = hidden_size // num_attention_heads
    half_head_dim = head_dim // 2

    for head in range(num_attention_heads):
        base = head * head_dim
        for i in range(half_head_dim):
            out_left = base + i
            out_right = base + half_head_dim + i
            matrix[out_left, out_right] = 1.0
            matrix[out_right, out_left] = -1.0

        if head_dim % 2 != 0:
            tail = base + 2 * half_head_dim
            matrix[tail, tail] = 1.0

    return matrix


def add_apply_rope(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    position_id: trt.ITensor,
    cos_table: trt.ITensor,
    sin_table: trt.ITensor,
    rotate_half_matrix: trt.ITensor,
) -> trt.ITensor:
    """Apply rotary position embedding: x*cos + rotate_half(x)*sin."""
    cos_gather = network.add_gather(cos_table, position_id, 0)
    sin_gather = network.add_gather(sin_table, position_id, 0)

    rotated = network.add_matrix_multiply(
        inp, trt.MatrixOperation.NONE,
        rotate_half_matrix, trt.MatrixOperation.NONE,
    )

    x_cos = network.add_elementwise(
        inp, cos_gather.get_output(0), trt.ElementWiseOperation.PROD)
    rot_sin = network.add_elementwise(
        rotated.get_output(0), sin_gather.get_output(0),
        trt.ElementWiseOperation.PROD)
    result = network.add_elementwise(
        x_cos.get_output(0), rot_sin.get_output(0),
        trt.ElementWiseOperation.SUM)
    return result.get_output(0)
