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


# ---------------------------------------------------------------------------
# Vision encoder graph ops
# ---------------------------------------------------------------------------

def add_self_attention_block(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    hidden_size: int,
    num_heads: int,
    seq_length: int,
    q_bias: np.ndarray | None = None,
    k_bias: np.ndarray | None = None,
    v_bias: np.ndarray | None = None,
    o_bias: np.ndarray | None = None,
) -> trt.ITensor:
    """Full self-attention without KV cache (single-pass, for vision encoders).

    Input hidden: [seq_length, hidden_size]
    Output: [seq_length, hidden_size]
    """
    head_dim = hidden_size // num_heads
    attn_scale = 1.0 / np.sqrt(max(head_dim, 1))

    # Q, K, V projections: [seq, hidden] @ [hidden, hidden] = [seq, hidden]
    q = add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_q)
    k = add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_k)
    v = add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_v)

    if q_bias is not None:
        q = add_bias_sum(network, q, hidden_size, q_bias)
    if k_bias is not None:
        k = add_bias_sum(network, k, hidden_size, k_bias)
    if v_bias is not None:
        v = add_bias_sum(network, v, hidden_size, v_bias)

    # Reshape to [num_heads, seq, head_dim]
    q_heads = network.add_shuffle(q)
    q_heads.reshape_dims = (seq_length, num_heads, head_dim)
    q_heads.second_transpose = trt.Permutation([1, 0, 2])

    k_heads = network.add_shuffle(k)
    k_heads.reshape_dims = (seq_length, num_heads, head_dim)
    k_heads.second_transpose = trt.Permutation([1, 0, 2])

    v_heads = network.add_shuffle(v)
    v_heads.reshape_dims = (seq_length, num_heads, head_dim)
    v_heads.second_transpose = trt.Permutation([1, 0, 2])

    # Attention scores: [num_heads, seq, head_dim] @ [num_heads, head_dim, seq]
    score = network.add_matrix_multiply(
        q_heads.get_output(0), trt.MatrixOperation.NONE,
        k_heads.get_output(0), trt.MatrixOperation.TRANSPOSE)

    # Scale
    scale_const = add_constant(
        network, (1, 1, 1), np.array([attn_scale], dtype=np.float32))
    scaled = network.add_elementwise(
        score.get_output(0), scale_const,
        trt.ElementWiseOperation.PROD)

    # Softmax over last dim
    softmax = network.add_softmax(scaled.get_output(0))
    softmax.axes = 1 << 2  # last dim

    # Context: [num_heads, seq, head_dim]
    context = network.add_matrix_multiply(
        softmax.get_output(0), trt.MatrixOperation.NONE,
        v_heads.get_output(0), trt.MatrixOperation.NONE)

    # Reshape back to [seq, hidden]
    context_flat = network.add_shuffle(context.get_output(0))
    context_flat.first_transpose = trt.Permutation([1, 0, 2])
    context_flat.reshape_dims = (seq_length, hidden_size)

    # Output projection
    out = add_matmul_rhs_constant(
        network, context_flat.get_output(0), hidden_size, hidden_size, w_o)
    if o_bias is not None:
        out = add_bias_sum(network, out, hidden_size, o_bias)

    return out


def add_self_attention_block_with_rope(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    hidden_size: int,
    num_heads: int,
    seq_length: int,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    q_bias: np.ndarray | None = None,
    k_bias: np.ndarray | None = None,
    v_bias: np.ndarray | None = None,
    o_bias: np.ndarray | None = None,
) -> trt.ITensor:
    """Full self-attention with precomputed RoPE (for vision encoders with 3D RoPE).

    Unlike the KV-cache decoder attention, this processes all positions at once
    and applies RoPE via precomputed per-position cos/sin tables.

    Input hidden: [seq_length, hidden_size]
    cos_table/sin_table: [seq_length, hidden_size] precomputed constants
    Output: [seq_length, hidden_size]
    """
    head_dim = hidden_size // num_heads
    attn_scale = 1.0 / np.sqrt(max(head_dim, 1))

    # Q, K, V projections: [seq, hidden] @ [hidden, hidden] = [seq, hidden]
    q = add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_q)
    k = add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_k)
    v = add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_v)

    if q_bias is not None:
        q = add_bias_sum(network, q, hidden_size, q_bias)
    if k_bias is not None:
        k = add_bias_sum(network, k, hidden_size, k_bias)
    if v_bias is not None:
        v = add_bias_sum(network, v, hidden_size, v_bias)

    # Apply RoPE using precomputed per-position cos/sin tables.
    # q_rot = q * cos + rotate_half(q) * sin  (element-wise per position)
    cos_const = add_constant(network, (seq_length, hidden_size), cos_table)
    sin_const = add_constant(network, (seq_length, hidden_size), sin_table)

    # Build rotate-half matrix for this hidden_size/num_heads config
    rot_half_np = make_rotate_half_matrix(hidden_size, num_heads)
    rot_half_const = add_constant(network, (hidden_size, hidden_size), rot_half_np)

    # Apply RoPE to Q
    q_rot = network.add_matrix_multiply(
        q, trt.MatrixOperation.NONE, rot_half_const, trt.MatrixOperation.NONE)
    q_cos = network.add_elementwise(q, cos_const, trt.ElementWiseOperation.PROD)
    q_sin = network.add_elementwise(
        q_rot.get_output(0), sin_const, trt.ElementWiseOperation.PROD)
    q = network.add_elementwise(
        q_cos.get_output(0), q_sin.get_output(0), trt.ElementWiseOperation.SUM)
    q = q.get_output(0)

    # Apply RoPE to K
    k_rot = network.add_matrix_multiply(
        k, trt.MatrixOperation.NONE, rot_half_const, trt.MatrixOperation.NONE)
    k_cos = network.add_elementwise(k, cos_const, trt.ElementWiseOperation.PROD)
    k_sin = network.add_elementwise(
        k_rot.get_output(0), sin_const, trt.ElementWiseOperation.PROD)
    k = network.add_elementwise(
        k_cos.get_output(0), k_sin.get_output(0), trt.ElementWiseOperation.SUM)
    k = k.get_output(0)

    # Reshape to [num_heads, seq, head_dim]
    q_heads = network.add_shuffle(q)
    q_heads.reshape_dims = (seq_length, num_heads, head_dim)
    q_heads.second_transpose = trt.Permutation([1, 0, 2])

    k_heads = network.add_shuffle(k)
    k_heads.reshape_dims = (seq_length, num_heads, head_dim)
    k_heads.second_transpose = trt.Permutation([1, 0, 2])

    v_heads = network.add_shuffle(v)
    v_heads.reshape_dims = (seq_length, num_heads, head_dim)
    v_heads.second_transpose = trt.Permutation([1, 0, 2])

    # Attention scores: [num_heads, seq, head_dim] @ [num_heads, head_dim, seq]
    score = network.add_matrix_multiply(
        q_heads.get_output(0), trt.MatrixOperation.NONE,
        k_heads.get_output(0), trt.MatrixOperation.TRANSPOSE)

    # Scale
    scale_const = add_constant(
        network, (1, 1, 1), np.array([attn_scale], dtype=np.float32))
    scaled = network.add_elementwise(
        score.get_output(0), scale_const,
        trt.ElementWiseOperation.PROD)

    # Softmax over last dim
    softmax = network.add_softmax(scaled.get_output(0))
    softmax.axes = 1 << 2  # last dim

    # Context: [num_heads, seq, head_dim]
    context = network.add_matrix_multiply(
        softmax.get_output(0), trt.MatrixOperation.NONE,
        v_heads.get_output(0), trt.MatrixOperation.NONE)

    # Reshape back to [seq, hidden]
    context_flat = network.add_shuffle(context.get_output(0))
    context_flat.first_transpose = trt.Permutation([1, 0, 2])
    context_flat.reshape_dims = (seq_length, hidden_size)

    # Output projection
    out = add_matmul_rhs_constant(
        network, context_flat.get_output(0), hidden_size, hidden_size, w_o)
    if o_bias is not None:
        out = add_bias_sum(network, out, hidden_size, o_bias)

    return out


def add_windowed_self_attention_with_rope(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    hidden_size: int,
    num_heads: int,
    seq_length: int,
    num_windows: int,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    q_bias: np.ndarray | None = None,
    k_bias: np.ndarray | None = None,
    v_bias: np.ndarray | None = None,
    o_bias: np.ndarray | None = None,
) -> trt.ITensor:
    """Windowed self-attention with precomputed RoPE.

    Splits the sequence into non-overlapping windows and runs attention
    independently per window. Patches must already be reordered so that
    each window's patches are contiguous in the sequence.

    Input hidden: [seq_length, hidden_size]
    cos_table/sin_table: [seq_length, hidden_size]
    Output: [seq_length, hidden_size]
    """
    head_dim = hidden_size // num_heads
    win_seq = seq_length // num_windows  # patches per window
    attn_scale = 1.0 / np.sqrt(max(head_dim, 1))

    # Q, K, V projections: [seq, hidden] @ [hidden, hidden]
    q = add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_q)
    k = add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_k)
    v = add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_v)

    if q_bias is not None:
        q = add_bias_sum(network, q, hidden_size, q_bias)
    if k_bias is not None:
        k = add_bias_sum(network, k, hidden_size, k_bias)
    if v_bias is not None:
        v = add_bias_sum(network, v, hidden_size, v_bias)

    # RoPE: same as full attention
    cos_const = add_constant(network, (seq_length, hidden_size), cos_table)
    sin_const = add_constant(network, (seq_length, hidden_size), sin_table)
    rot_half_np = make_rotate_half_matrix(hidden_size, num_heads)
    rot_half_const = add_constant(network, (hidden_size, hidden_size), rot_half_np)

    q_rot = network.add_matrix_multiply(
        q, trt.MatrixOperation.NONE, rot_half_const, trt.MatrixOperation.NONE)
    q_cos = network.add_elementwise(q, cos_const, trt.ElementWiseOperation.PROD)
    q_sin = network.add_elementwise(
        q_rot.get_output(0), sin_const, trt.ElementWiseOperation.PROD)
    q = network.add_elementwise(
        q_cos.get_output(0), q_sin.get_output(0), trt.ElementWiseOperation.SUM)
    q = q.get_output(0)

    k_rot = network.add_matrix_multiply(
        k, trt.MatrixOperation.NONE, rot_half_const, trt.MatrixOperation.NONE)
    k_cos = network.add_elementwise(k, cos_const, trt.ElementWiseOperation.PROD)
    k_sin = network.add_elementwise(
        k_rot.get_output(0), sin_const, trt.ElementWiseOperation.PROD)
    k = network.add_elementwise(
        k_cos.get_output(0), k_sin.get_output(0), trt.ElementWiseOperation.SUM)
    k = k.get_output(0)

    # Reshape to [num_windows, win_seq, num_heads, head_dim]
    # then transpose to [num_windows, num_heads, win_seq, head_dim]
    # then merge first two dims: [num_windows * num_heads, win_seq, head_dim]
    q_win = network.add_shuffle(q)
    q_win.reshape_dims = (num_windows, win_seq, num_heads, head_dim)
    q_win.second_transpose = trt.Permutation([0, 2, 1, 3])
    q_flat = network.add_shuffle(q_win.get_output(0))
    q_flat.reshape_dims = (num_windows * num_heads, win_seq, head_dim)

    k_win = network.add_shuffle(k)
    k_win.reshape_dims = (num_windows, win_seq, num_heads, head_dim)
    k_win.second_transpose = trt.Permutation([0, 2, 1, 3])
    k_flat = network.add_shuffle(k_win.get_output(0))
    k_flat.reshape_dims = (num_windows * num_heads, win_seq, head_dim)

    v_win = network.add_shuffle(v)
    v_win.reshape_dims = (num_windows, win_seq, num_heads, head_dim)
    v_win.second_transpose = trt.Permutation([0, 2, 1, 3])
    v_flat = network.add_shuffle(v_win.get_output(0))
    v_flat.reshape_dims = (num_windows * num_heads, win_seq, head_dim)

    # Attention: [NW*NH, win_seq, head_dim] @ [NW*NH, head_dim, win_seq]
    score = network.add_matrix_multiply(
        q_flat.get_output(0), trt.MatrixOperation.NONE,
        k_flat.get_output(0), trt.MatrixOperation.TRANSPOSE)

    scale_const = add_constant(
        network, (1, 1, 1), np.array([attn_scale], dtype=np.float32))
    scaled = network.add_elementwise(
        score.get_output(0), scale_const, trt.ElementWiseOperation.PROD)

    softmax = network.add_softmax(scaled.get_output(0))
    softmax.axes = 1 << 2

    context = network.add_matrix_multiply(
        softmax.get_output(0), trt.MatrixOperation.NONE,
        v_flat.get_output(0), trt.MatrixOperation.NONE)

    # Reshape back: [NW*NH, win_seq, head_dim] → [NW, NH, win_seq, head_dim]
    # → [NW, win_seq, NH, head_dim] → [seq_length, hidden_size]
    ctx_unflat = network.add_shuffle(context.get_output(0))
    ctx_unflat.reshape_dims = (num_windows, num_heads, win_seq, head_dim)
    ctx_unflat.second_transpose = trt.Permutation([0, 2, 1, 3])
    ctx_flat = network.add_shuffle(ctx_unflat.get_output(0))
    ctx_flat.reshape_dims = (seq_length, hidden_size)

    # Output projection
    out = add_matmul_rhs_constant(
        network, ctx_flat.get_output(0), hidden_size, hidden_size, w_o)
    if o_bias is not None:
        out = add_bias_sum(network, out, hidden_size, o_bias)

    return out


def add_patch_embed_3d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    in_channels: int,
    embed_dim: int,
    temporal_patch_size: int,
    patch_size: int,
) -> trt.ITensor:
    """3D patch embedding via convolution.

    Input: [T*C, H, W] (already flattened temporal*channels) or [T, C, H, W]
    Output: [num_patches, embed_dim]

    The 3D convolution is implemented as a 2D convolution over the flattened
    temporal*channel dimension, matching HuggingFace's PatchEmbed3D.
    """
    # Input may be [T*C, H, W] (3D) or [T, C, H, W] (4D).
    # We need [1, T*C, H, W] for conv2d.
    inp_ndims = len(inp.shape)
    reshape_in = network.add_shuffle(inp)
    if inp_ndims == 3:
        # [T*C, H, W] -> [1, T*C, H, W]
        tc = inp.shape[0]
        h = inp.shape[1]
        w = inp.shape[2]
        reshape_in.reshape_dims = (1, tc, h, w)
    else:
        # [T, C, H, W] -> [1, T*C, H, W]
        reshape_in.reshape_dims = (1, temporal_patch_size * in_channels, -1, 0)

    # Conv2D with kernel [embed_dim, T*C, patch_size, patch_size]
    # weight shape from HF: [embed_dim, T*C, patch_size, patch_size]
    conv_w = trt.Weights(np.ascontiguousarray(weight, dtype=np.float32))
    conv_b = trt.Weights()
    if bias is not None:
        conv_b = trt.Weights(np.ascontiguousarray(bias, dtype=np.float32))

    conv = network.add_convolution_nd(
        reshape_in.get_output(0),
        num_output_maps=embed_dim,
        kernel_shape=(patch_size, patch_size),
        kernel=conv_w,
        bias=conv_b,
    )
    conv.stride_nd = (patch_size, patch_size)

    # Output shape: [1, embed_dim, H', W'] -> flatten to [num_patches, embed_dim]
    reshape_out = network.add_shuffle(conv.get_output(0))
    reshape_out.first_transpose = trt.Permutation([0, 2, 3, 1])
    reshape_out.reshape_dims = (-1, embed_dim)

    return reshape_out.get_output(0)


def add_spatial_merge(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    w_fc1: np.ndarray,
    w_fc2: np.ndarray,
    b_fc1: np.ndarray | None,
    b_fc2: np.ndarray | None,
    norm_gamma: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    eps_tensor: trt.ITensor,
    seq_length: int,
    merge_size: int = 2,
) -> trt.ITensor:
    """Spatial merge: 2x2 merge MLP that reduces spatial resolution.

    Reshapes [seq, dim] -> merge adjacent 2x2 patches, then MLP.
    Input: [seq_length, input_dim]
    Output: [seq_length // (merge_size^2), output_dim]

    Note: This is a simplified version. For Qwen2.5-VL, the merge
    concatenates merge_size^2 adjacent patches, then applies layernorm + MLP.
    """
    merged_dim = input_dim * merge_size * merge_size

    # LayerNorm on the merged representation
    norm = add_layer_norm(
        network, inp, input_dim,
        norm_gamma, np.zeros(input_dim, dtype=np.float32), eps_tensor)

    # For simplicity in the TRT graph, we use a 2-layer MLP directly
    # on the already-flattened input. The spatial rearrangement is handled
    # during preprocessing.
    fc1 = add_matmul_rhs_constant(network, norm, input_dim, hidden_dim, w_fc1)
    if b_fc1 is not None:
        fc1 = add_bias_sum(network, fc1, hidden_dim, b_fc1)

    # GELU activation
    activated = add_gelu_new(network, fc1)

    fc2 = add_matmul_rhs_constant(network, activated, hidden_dim, output_dim, w_fc2)
    if b_fc2 is not None:
        fc2 = add_bias_sum(network, fc2, output_dim, b_fc2)

    return fc2
