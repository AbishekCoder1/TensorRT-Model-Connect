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
    partial_rotary_factor: float = 1.0,
) -> np.ndarray:
    """Build cos or sin RoPE table of shape [max_cache_length, hidden_size].

    Args:
        partial_rotary_factor: Fraction of head dimensions that get RoPE
            (e.g. 0.25 for StableLM-2). Default 1.0 = full RoPE.
    """
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
    rotary_ndims = int(head_dim * partial_rotary_factor)
    half_rotary = rotary_ndims // 2
    if half_rotary <= 0 or rope_theta <= 0.0:
        return table

    for pos in range(max_cache_length):
        for head in range(num_attention_heads):
            for dim in range(rotary_ndims):
                freq_idx = dim % half_rotary
                exponent = (2.0 * freq_idx) / rotary_ndims
                inv_freq = rope_theta ** (-exponent)
                angle = pos * inv_freq
                value = np.cos(angle) if cosine else np.sin(angle)
                offset = head * head_dim + dim
                table[pos, offset] = value

    return table


def make_rotate_half_matrix(
    hidden_size: int,
    num_attention_heads: int,
    partial_rotary_factor: float = 1.0,
) -> np.ndarray:
    """Build the rotate-half permutation matrix [hidden_size, hidden_size].

    For partial rotary (factor < 1.0), non-rotary dims pass through unchanged
    via the sin=0 multiplication (the matrix zeros here don't matter since
    they're multiplied by sin=0). But we still set identity for safety.
    """
    matrix = np.zeros((hidden_size, hidden_size), dtype=np.float32)
    if (hidden_size <= 0 or num_attention_heads <= 0
            or hidden_size % num_attention_heads != 0):
        np.fill_diagonal(matrix, 1.0)
        return matrix

    head_dim = hidden_size // num_attention_heads
    rotary_ndims = int(head_dim * partial_rotary_factor)
    half_rotary = rotary_ndims // 2

    for head in range(num_attention_heads):
        base = head * head_dim
        for i in range(half_rotary):
            out_left = base + i
            out_right = base + half_rotary + i
            matrix[out_left, out_right] = 1.0
            matrix[out_right, out_left] = -1.0

        if rotary_ndims % 2 != 0:
            tail = base + 2 * half_rotary
            matrix[tail, tail] = 1.0

        # Non-rotary dims: identity (pass through, since sin=0 for these)
        for d in range(rotary_ndims, head_dim):
            matrix[base + d, base + d] = 1.0

    return matrix


def add_layer_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps_tensor: trt.ITensor,
) -> trt.ITensor:
    """LayerNorm: gamma * ((x - mean) / sqrt(var + eps)) + beta."""
    # mean = reduce_mean(x)
    mean = network.add_reduce(
        inp, trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    # x - mean
    centered = network.add_elementwise(
        inp, mean.get_output(0), trt.ElementWiseOperation.SUB)
    # variance = mean((x - mean)^2)
    sq = network.add_elementwise(
        centered.get_output(0), centered.get_output(0),
        trt.ElementWiseOperation.PROD)
    var = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    # sqrt(var + eps)
    denom_in = network.add_elementwise(
        var.get_output(0), eps_tensor, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    # normalized = (x - mean) / sqrt(var + eps)
    normalized = network.add_elementwise(
        centered.get_output(0), recip.get_output(0),
        trt.ElementWiseOperation.PROD)
    # gamma * normalized + beta
    gamma_t = add_constant(network, (1, hidden_size), gamma)
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD)
    beta_t = add_constant(network, (1, hidden_size), beta)
    result = network.add_elementwise(
        scaled.get_output(0), beta_t, trt.ElementWiseOperation.SUM)
    return result.get_output(0)


def add_gelu_new(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
) -> trt.ITensor:
    """GELU (tanh approximation): 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))."""
    # x^3
    x_sq = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD)
    x_cu = network.add_elementwise(
        x_sq.get_output(0), inp, trt.ElementWiseOperation.PROD)
    # 0.044715 * x^3
    coeff = add_constant(network, (1, 1), np.array([0.044715], dtype=np.float32))
    scaled_cube = network.add_elementwise(
        x_cu.get_output(0), coeff, trt.ElementWiseOperation.PROD)
    # x + 0.044715 * x^3
    inner_sum = network.add_elementwise(
        inp, scaled_cube.get_output(0), trt.ElementWiseOperation.SUM)
    # sqrt(2/pi) * (x + 0.044715 * x^3)
    sqrt_2_over_pi = add_constant(
        network, (1, 1),
        np.array([np.sqrt(2.0 / np.pi)], dtype=np.float32))
    tanh_arg = network.add_elementwise(
        sqrt_2_over_pi, inner_sum.get_output(0),
        trt.ElementWiseOperation.PROD)
    # tanh(...)
    tanh_l = network.add_activation(
        tanh_arg.get_output(0), trt.ActivationType.TANH)
    # 1 + tanh(...)
    one = add_constant(network, (1, 1), np.array([1.0], dtype=np.float32))
    one_plus_tanh = network.add_elementwise(
        one, tanh_l.get_output(0), trt.ElementWiseOperation.SUM)
    # 0.5 * x
    half = add_constant(network, (1, 1), np.array([0.5], dtype=np.float32))
    half_x = network.add_elementwise(
        half, inp, trt.ElementWiseOperation.PROD)
    # 0.5 * x * (1 + tanh(...))
    result = network.add_elementwise(
        half_x.get_output(0), one_plus_tanh.get_output(0),
        trt.ElementWiseOperation.PROD)
    return result.get_output(0)


def add_activation(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    activation_type: str,
) -> trt.ITensor:
    """Dispatch activation by name: 'silu', 'gelu_new', 'gelu', 'relu'."""
    if activation_type in ("gelu_new", "gelu"):
        return add_gelu_new(network, inp)
    elif activation_type == "relu":
        act = network.add_activation(inp, trt.ActivationType.RELU)
        return act.get_output(0)
    elif activation_type == "silu":
        sigmoid = network.add_activation(inp, trt.ActivationType.SIGMOID)
        swish = network.add_elementwise(
            inp, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
        return swish.get_output(0)
    else:
        raise ValueError(f"Unsupported activation: {activation_type}")


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
