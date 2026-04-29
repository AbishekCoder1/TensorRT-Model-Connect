"""Dual-profile decoder engine builder — single engine, two optimization profiles.

Produces one TensorRT engine that handles both prefill (multi-token) and
decode (single-token) phases by switching between two optimization profiles
at runtime:
  * Profile 0 (prefill): Sq ranges over [1, opt=opt_prefill_length, max=max_prefill_length].
    TensorRT picks batched MHA kernels (e.g. ``_gemm_mha_v2``) at opt Sq.
  * Profile 1 (decode): Sq fixed to 1. TensorRT picks the GEMV fast-path
    (``_gemv_mha_v1``).

Both profiles use the same graph and weights — only the optimization
profile differs, so the engine's weights live once in GPU memory and the
C++ runtime creates two ``IExecutionContext``s (one per profile) that
share the engine.

Scope: covers the same architectural variants the legacy
``standard_decoder_builder`` supports — RMSNorm or LayerNorm; SwiGLU or
GeluFC MLP; RoPE (full / partial / interleaved), learned absolute, or
ALiBi position; sequential or parallel residual; optional q/k_norm,
QKV/output/MLP biases, and a Bloom-style embedding LayerNorm. Quantized
builds (fp8 / int8 ``quant_ctx``) thread Q/DQ insertion through every
projection matmul via ``QuantContext.maybe_quantized_matmul``. Per-layer
debug outputs, hidden-state outputs, and the VL ``embed_input`` path stay
on ``standard_decoder_builder`` for now and are dispatched there from
inside ``build_standard_decoder_engine``.

Tensor contract (matches the C++ runtime KvCache naming):
  Inputs (dynamic shapes — Sq varies by profile)
    token_id        int32   (-1,)
    position_id     int32   (-1,)
    attention_mask  float32 (-1, -1)                 # (Sq, max_cache + Sq)
    cache_k_i       fp16/f32 (max_cache, attn_size)  # static
    cache_v_i       fp16/f32 (max_cache, attn_size)  # static
  Outputs
    logits          float32 (1, vocab)               # last-row sliced inside the engine
    present_k_i     fp16/f32 (-1, attn_size)         # (Sq, attn_size)
    present_v_i     fp16/f32 (-1, attn_size)         # (Sq, attn_size)
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt

from . import graph_ops

if TYPE_CHECKING:
    from .config import ModelConfig
    from .checkpoint_mapper import WeightDict
    from .quantization.context import QuantContext


def _const_in_work_dtype(
    network: trt.INetworkDefinition,
    shape: tuple,
    values: np.ndarray,
    work_np_dtype: np.dtype,
    work_trt_dtype: trt.DataType,
) -> trt.ITensor:
    """Create a constant in work_np_dtype storage and cast it to work_trt_dtype.

    Needed for bf16 builds: the dual-profile builder stores bf16 weights
    on disk as fp16 (work_np_dtype = np.float16), but the runtime tensor
    must be bfloat16 to match the rest of the graph. ``add_constant``
    alone produces an fp16 constant — we need an explicit cast to
    bfloat16 so layers like IRotaryEmbeddingLayer (which require all
    inputs to share a dtype) accept it. fp16 / fp32 builds are no-ops
    because work_np_dtype maps directly to work_trt_dtype.
    """
    const = graph_ops.add_constant(network, shape, values, dtype=work_np_dtype)
    if const.dtype != work_trt_dtype:
        const = network.add_cast(const, work_trt_dtype).get_output(0)
    return const


def _make_matmul_fn(
    network: trt.INetworkDefinition,
    dtype: np.dtype,
    quant_ctx: "QuantContext | None",
):
    """Mirror of ``graph_blocks._make_matmul_fn`` for the dual-profile path.

    Returns a callable ``(lhs, lhs_w, rhs_w, rhs_weights, weight_name) -> ITensor``
    that routes through ``QuantContext.maybe_quantized_matmul`` when present
    and falls back to a plain ``add_matmul_rhs_constant`` otherwise. The
    ``weight_name`` is the dotted weight key (e.g. ``layer.0.w_q``) used by
    the quantization profile to look up scales and the per-layer exclude
    pattern.
    """
    if quant_ctx is None:
        def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
            return graph_ops.add_matmul_rhs_constant(
                network, lhs, lhs_w, rhs_w, rhs_weights, dtype=dtype)
        return matmul

    def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
        return quant_ctx.maybe_quantized_matmul(
            network, lhs, lhs_w, rhs_w, rhs_weights, weight_name,
            dtype=dtype)
    return matmul


# ---------------------------------------------------------------------------
# Norm helpers — multi-row Sq variants of RMSNorm / LayerNorm.
# ---------------------------------------------------------------------------


def _rms_norm_multi(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden: int,
    gamma: np.ndarray,
    eps: float,
    dtype: np.dtype,
) -> trt.ITensor:
    """RMSNorm on (Sq, hidden) — reduces along the hidden dim.

    FP32 precision boundary: when ``inp`` isn't already fp32 we cast to fp32
    for numerical stability and then cast back to the original input dtype
    (recovered from ``inp.dtype`` so bf16 stays bf16 — the ``dtype`` numpy
    type stores both fp16 and bf16 weights as np.float16, so it can't be
    used to disambiguate the runtime tensor type).
    """
    work_trt_dtype = inp.dtype
    need_cast = work_trt_dtype != trt.float32
    x = inp
    if need_cast:
        x = network.add_cast(x, trt.float32).get_output(0)
    sq = network.add_elementwise(x, x, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    eps_t = graph_ops.add_constant(
        network, (1, 1), np.array([[eps]], dtype=np.float32), dtype=np.float32)
    denom = network.add_elementwise(
        mean.get_output(0), eps_t, trt.ElementWiseOperation.SUM)
    rsqrt = network.add_unary(
        network.add_unary(denom.get_output(0), trt.UnaryOperation.SQRT).get_output(0),
        trt.UnaryOperation.RECIP)
    normed = network.add_elementwise(
        x, rsqrt.get_output(0), trt.ElementWiseOperation.PROD)
    gamma_t = graph_ops.add_constant(
        network, (1, hidden), gamma.reshape(1, hidden), dtype=np.float32)
    scaled = network.add_elementwise(
        normed.get_output(0), gamma_t, trt.ElementWiseOperation.PROD)
    out = scaled.get_output(0)
    if need_cast:
        out = network.add_cast(out, work_trt_dtype).get_output(0)
    return out


def _layer_norm_multi(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float,
    dtype: np.dtype,
) -> trt.ITensor:
    """LayerNorm on (Sq, hidden): gamma * (x - mean) / sqrt(var + eps) + beta.

    Same fp32 precision boundary as ``_rms_norm_multi``; cast-back uses
    ``inp.dtype`` so bf16 inputs stay bf16.
    """
    work_trt_dtype = inp.dtype
    need_cast = work_trt_dtype != trt.float32
    x = inp
    if need_cast:
        x = network.add_cast(x, trt.float32).get_output(0)
    mean = network.add_reduce(
        x, trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    centered = network.add_elementwise(
        x, mean.get_output(0), trt.ElementWiseOperation.SUB)
    sq = network.add_elementwise(
        centered.get_output(0), centered.get_output(0),
        trt.ElementWiseOperation.PROD)
    var = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    eps_t = graph_ops.add_constant(
        network, (1, 1), np.array([[eps]], dtype=np.float32), dtype=np.float32)
    denom_in = network.add_elementwise(
        var.get_output(0), eps_t, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        centered.get_output(0), recip.get_output(0),
        trt.ElementWiseOperation.PROD)
    gamma_t = graph_ops.add_constant(
        network, (1, hidden), gamma.reshape(1, hidden), dtype=np.float32)
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD)
    beta_t = graph_ops.add_constant(
        network, (1, hidden), beta.reshape(1, hidden), dtype=np.float32)
    summed = network.add_elementwise(
        scaled.get_output(0), beta_t, trt.ElementWiseOperation.SUM)
    out = summed.get_output(0)
    if need_cast:
        out = network.add_cast(out, work_trt_dtype).get_output(0)
    return out


def _norm_multi(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden: int,
    gamma: np.ndarray,
    beta: np.ndarray | None,
    eps: float,
    norm_type: str,
    dtype: np.dtype,
) -> trt.ITensor:
    if norm_type == "layernorm":
        if beta is None:
            beta = np.zeros(hidden, dtype=np.float32)
        return _layer_norm_multi(network, inp, hidden, gamma, beta, eps, dtype)
    return _rms_norm_multi(network, inp, hidden, gamma, eps, dtype)


def _rms_norm_per_head_multi(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_heads: int,
    head_dim: int,
    gamma: np.ndarray,
    eps: float,
    dtype: np.dtype,
) -> trt.ITensor:
    """Per-head RMSNorm for (Sq, num_heads * head_dim) input."""
    work_trt_dtype = inp.dtype
    reshape_in = network.add_shuffle(inp)
    reshape_in.reshape_dims = (-1, num_heads, head_dim)
    x = reshape_in.get_output(0)

    need_cast = work_trt_dtype != trt.float32
    if need_cast:
        x = network.add_cast(x, trt.float32).get_output(0)

    sq = network.add_elementwise(x, x, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 2, keep_dims=True)
    eps_t = graph_ops.add_constant(
        network, (1, 1, 1), np.array([[[eps]]], dtype=np.float32), dtype=np.float32)
    denom = network.add_elementwise(
        mean.get_output(0), eps_t, trt.ElementWiseOperation.SUM)
    rsqrt = network.add_unary(
        network.add_unary(denom.get_output(0), trt.UnaryOperation.SQRT).get_output(0),
        trt.UnaryOperation.RECIP)
    normed = network.add_elementwise(
        x, rsqrt.get_output(0), trt.ElementWiseOperation.PROD)
    gamma_t = graph_ops.add_constant(
        network, (1, num_heads, head_dim),
        gamma.reshape(num_heads, head_dim), dtype=np.float32)
    scaled = network.add_elementwise(
        normed.get_output(0), gamma_t, trt.ElementWiseOperation.PROD)
    out = scaled.get_output(0)
    if need_cast:
        out = network.add_cast(out, work_trt_dtype).get_output(0)
    flat = network.add_shuffle(out)
    flat.reshape_dims = (-1, num_heads * head_dim)
    return flat.get_output(0)


# ---------------------------------------------------------------------------
# Reshape helpers — convert between (Sq, num_heads * head_dim) and
# (1, num_heads, Sq, head_dim) for IAttention.
# ---------------------------------------------------------------------------


def _to_heads_4d(
    network: trt.INetworkDefinition,
    x: trt.ITensor,
    num_heads: int,
    head_dim: int,
    tag: str,
) -> trt.ITensor:
    """(S, num_heads * head_dim) -> (1, num_heads, S, head_dim).

    Collapses the original two-shuffle chain (reshape -> transpose ->
    reshape) into a single IShuffleLayer that does reshape + permute in
    one step. For Sq=1 (decode profile) the transpose between two
    singleton dims is a no-op, so TRT folds the shuffle out entirely
    and the per-step decode kernel-launch overhead matches the legacy
    single-profile graph.
    """
    s = network.add_shuffle(x)
    s.name = tag + "_to_4d"
    s.reshape_dims = (1, -1, num_heads, head_dim)
    s.second_transpose = trt.Permutation([0, 2, 1, 3])
    return s.get_output(0)


def _apply_rope_native_multi(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_heads: int,
    head_dim: int,
    cos_half_2d: trt.ITensor,
    sin_half_2d: trt.ITensor,
    position_id: trt.ITensor,
    rotary_embedding_dim: int,
    interleaved: bool,
) -> trt.ITensor:
    """Apply RoPE via TRT's IRotaryEmbeddingLayer for dynamic Sq.

    Reshapes (Sq, num_heads * head_dim) -> (1, num_heads, Sq, head_dim),
    runs the fused rotary layer, then reshapes back. position_id is
    (Sq,) -> (1, Sq). This picks up the fused RoPE kernel under both
    optimization profiles (Sq=1 decode and Sq=opt prefill).
    """
    attention_size = num_heads * head_dim

    # (Sq, H * D) -> (1, Sq, H, D) -> transpose to (1, H, Sq, D) in ONE shuffle.
    # For Sq=1 the transpose is between singleton dims and TRT folds the
    # whole shuffle out, so the decode profile pays no extra kernel.
    pre = network.add_shuffle(inp)
    pre.reshape_dims = (1, -1, num_heads, head_dim)
    pre.second_transpose = trt.Permutation([0, 2, 1, 3])

    # position_id (Sq,) -> (1, Sq)
    pos_2d = network.add_shuffle(position_id)
    pos_2d.reshape_dims = (1, -1)

    rope = network.add_rotary_embedding(
        pre.get_output(0),
        cos_half_2d,
        sin_half_2d,
        interleaved,
        rotary_embedding_dim,
    )
    rope.set_input(3, pos_2d.get_output(0))

    # (1, H, Sq, D) -> (Sq, H * D) in ONE shuffle (transpose + reshape).
    post = network.add_shuffle(rope.get_output(0))
    post.first_transpose = trt.Permutation([0, 2, 1, 3])
    post.reshape_dims = (-1, attention_size)
    return post.get_output(0)


def _from_heads_4d(
    network: trt.INetworkDefinition,
    ctx_4d: trt.ITensor,
    attention_size: int,
    tag: str,
) -> trt.ITensor:
    """(1, num_heads, S, head_dim) -> (S, num_heads * head_dim)."""
    squeeze = network.add_shuffle(ctx_4d)
    squeeze.name = tag + "_h_s_d"
    squeeze.first_transpose = trt.Permutation([0, 2, 1, 3])  # (1, S, H, D)
    squeeze.reshape_dims = (-1, attention_size)
    return squeeze.get_output(0)


# ---------------------------------------------------------------------------
# Mask helpers.
# ---------------------------------------------------------------------------


def _mask_to_4d(
    network: trt.INetworkDefinition,
    mask_2d: trt.ITensor,
) -> trt.ITensor:
    """(Sq, K) -> (1, 1, Sq, K) via shape-tensor-driven shuffle."""
    shp = network.add_shape(mask_2d).get_output(0)  # [2] int64
    ones = graph_ops.add_constant(
        network, (2,), np.array([1, 1], dtype=np.int64), dtype=np.int64)
    concat = network.add_concatenation([ones, shp])
    concat.axis = 0
    target = concat.get_output(0)
    shuffle = network.add_shuffle(mask_2d)
    shuffle.set_input(1, target)
    return shuffle.get_output(0)


def _alibi_mask_4d(
    network: trt.INetworkDefinition,
    mask_2d: trt.ITensor,
    position_id: trt.ITensor,
    alibi_slopes_tensor: trt.ITensor,
    cache_position_indices_fp32: trt.ITensor,
    num_heads: int,
    work_trt_dtype: trt.DataType,
) -> trt.ITensor:
    """Build a (1, num_heads, Sq, K) ALiBi-augmented attention mask.

    Adds ``slope[h] * (key_pos[k] - query_pos[q])`` to the 2D additive
    mask so softmax sees both the causal/padding mask AND the ALiBi
    linear bias in one tensor. Dynamic Sq and K are recovered from the
    mask's shape at runtime.

    The KV cache is laid out as ``[max_cache_length, attention_size]``
    where slot ``k`` (for ``k < max_cache_length``) holds the K/V at
    position ``k``. The current-step K/V live in slots
    ``[max_cache_length, max_cache_length + Sq)`` and their positions
    come from ``position_id``. So the per-key position vector is::

        key_pos = concat([0, 1, ..., max_cache_length - 1], position_id)

    which has shape ``(max_cache_length + Sq,)`` — exactly the K
    dimension of the attention mask.
    """
    pos_float = network.add_cast(position_id, trt.float32).get_output(0)  # (Sq,)

    # Per-key positions: (max_cache + Sq,) fp32.
    key_pos_concat = network.add_concatenation([cache_position_indices_fp32, pos_float])
    key_pos_concat.axis = 0
    key_pos_t = key_pos_concat.get_output(0)

    # mask shape [2] int64 -> Sq (axis 0) and K (axis 1).
    mask_shape = network.add_shape(mask_2d).get_output(0)  # [2]
    one_const = graph_ops.add_constant(
        network, (1,), np.array([1], dtype=np.int64), dtype=np.int64)
    sq_size = network.add_slice(mask_shape, start=(0,), shape=(1,), stride=(1,))
    sq_size_t = sq_size.get_output(0)  # [1] = Sq
    k_size = network.add_slice(mask_shape, start=(1,), shape=(1,), stride=(1,))
    k_size_t = k_size.get_output(0)  # [1] = K

    # Reshape key_pos (K,) -> (1, K), and pos_float (Sq,) -> (Sq, 1) for
    # broadcast subtraction.
    one_k_shape = network.add_concatenation([one_const, k_size_t])
    one_k_shape.axis = 0
    key_pos_2d = network.add_shuffle(key_pos_t)
    key_pos_2d.set_input(1, one_k_shape.get_output(0))

    sq_one_shape = network.add_concatenation([sq_size_t, one_const])
    sq_one_shape.axis = 0
    pos_2d = network.add_shuffle(pos_float)
    pos_2d.set_input(1, sq_one_shape.get_output(0))

    # rel_pos[q, k] = key_pos[k] - position_id[q] (float32).
    rel_pos = network.add_elementwise(
        key_pos_2d.get_output(0), pos_2d.get_output(0),
        trt.ElementWiseOperation.SUB)

    # Reshape to (1, 1, Sq, K) for broadcast with slopes (1, H, 1, 1).
    one_const2 = graph_ops.add_constant(
        network, (1,), np.array([1], dtype=np.int64), dtype=np.int64)
    rel_4d_shape = network.add_concatenation([one_const, one_const2, sq_size_t, k_size_t])
    rel_4d_shape.axis = 0
    rel_4d = network.add_shuffle(rel_pos.get_output(0))
    rel_4d.set_input(1, rel_4d_shape.get_output(0))

    # alibi_slopes_tensor is (num_heads, 1, 1). Reshape to (1, H, 1, 1).
    slopes_4d = network.add_shuffle(alibi_slopes_tensor)
    slopes_4d.reshape_dims = (1, num_heads, 1, 1)

    # bias = slopes * rel_pos -> (1, H, Sq, K).
    alibi_bias = network.add_elementwise(
        slopes_4d.get_output(0), rel_4d.get_output(0),
        trt.ElementWiseOperation.PROD)
    alibi_bias_t = alibi_bias.get_output(0)
    if work_trt_dtype != trt.float32:
        alibi_bias_t = network.add_cast(alibi_bias_t, work_trt_dtype).get_output(0)

    # Broadcast 2D mask (Sq, K) to (1, 1, Sq, K) via shape-tensor shuffle.
    mask_4d = _mask_to_4d(network, mask_2d)

    # Add ALiBi bias to the additive mask.
    summed = network.add_elementwise(
        mask_4d, alibi_bias_t, trt.ElementWiseOperation.SUM)
    return summed.get_output(0)


# ---------------------------------------------------------------------------
# Attention core — IAttention with explicit (additive) mask.
# ---------------------------------------------------------------------------


def _attention_core(
    network: trt.INetworkDefinition,
    q_4d: trt.ITensor,
    k_4d: trt.ITensor,
    v_4d: trt.ITensor,
    mask_4d: trt.ITensor,
    *,
    head_dim: int,
    scale: float,
) -> trt.ITensor:
    """IAttention with explicit mask (no causal flag — mask carries causality)."""
    if q_4d.dtype == trt.float16:
        scale_t = graph_ops.add_constant(
            network, (1, 1, 1, 1),
            np.array([[[[scale]]]], dtype=np.float16), dtype=np.float16)
    else:
        scale_t = graph_ops.add_constant(
            network, (1, 1, 1, 1),
            np.array([[[[scale]]]], dtype=np.float32), dtype=np.float32)
        if q_4d.dtype == trt.bfloat16:
            scale_t = network.add_cast(scale_t, trt.bfloat16).get_output(0)
    q_scaled = network.add_elementwise(q_4d, scale_t, trt.ElementWiseOperation.PROD)

    attn = network.add_attention(
        q_scaled.get_output(0), k_4d, v_4d,
        trt.AttentionNormalizationOp.SOFTMAX, False)
    attn.mask = mask_4d
    # Allow TRT to decompose into primitives when no fused MHA kernel is
    # available for this (dtype, head_dim, seq) combination. On real
    # targets (fp16, head_dim=128, opt Sq=64) TRT still picks the named
    # _mha_v2 kernel from profile 0 — verified via
    # ``trtexec --useProfile=0 --dumpProfile`` on Qwen3-0.6B. The
    # fallback path keeps tiny unit-test configs (head_dim=4, fp32)
    # buildable since TRT has no fused kernel for those shapes.
    attn.decomposable = True
    return attn.get_output(0)


# ---------------------------------------------------------------------------
# MLP helpers.
# ---------------------------------------------------------------------------


def _swiglu_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    matmul,
    weights: "WeightDict",
    prefix: str,
    hidden: int,
    mlp_size: int,
) -> trt.ITensor:
    gate = matmul(inp, hidden, mlp_size,
                  weights[f"{prefix}.w_gate"], f"{prefix}.w_gate")
    up = matmul(inp, hidden, mlp_size,
                weights[f"{prefix}.w_up"], f"{prefix}.w_up")
    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(
        gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(
        swish.get_output(0), up, trt.ElementWiseOperation.PROD)
    mlp_out = matmul(gated.get_output(0), mlp_size, hidden,
                     weights[f"{prefix}.w_down"], f"{prefix}.w_down")
    return mlp_out


def _gelu_fc_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    matmul,
    weights: "WeightDict",
    prefix: str,
    hidden: int,
    mlp_size: int,
    activation: str,
    work_np_dtype: np.dtype,
) -> trt.ITensor:
    fc1 = matmul(inp, hidden, mlp_size,
                 weights[f"{prefix}.w_fc1"], f"{prefix}.w_fc1")
    fc1_bias = weights.get(f"{prefix}.fc1_bias")
    if fc1_bias is not None:
        fc1 = graph_ops.add_bias_sum(network, fc1, mlp_size, fc1_bias, dtype=work_np_dtype)
    activated = graph_ops.add_activation(network, fc1, activation, dtype=work_np_dtype)
    fc2 = matmul(activated, mlp_size, hidden,
                 weights[f"{prefix}.w_fc2"], f"{prefix}.w_fc2")
    fc2_bias = weights.get(f"{prefix}.fc2_bias")
    if fc2_bias is not None:
        fc2 = graph_ops.add_bias_sum(network, fc2, hidden, fc2_bias, dtype=work_np_dtype)
    return fc2


# ---------------------------------------------------------------------------
# Config guard.
# ---------------------------------------------------------------------------


def _supports_config(config: "ModelConfig", weights: "WeightDict") -> None:
    """Reject configs the dual-profile builder cannot handle."""
    model_type = getattr(config, "model_type", "").lower()
    if "moe" in model_type or "mamba" in model_type or "rwkv" in model_type:
        raise NotImplementedError(
            f"dual_profile_decoder_builder does not support model_type={model_type!r}")
    if "embedding" not in weights:
        raise NotImplementedError("missing embedding weight")
    if "final_norm" not in weights:
        raise NotImplementedError("missing final_norm weight")


# ---------------------------------------------------------------------------
# Main builder.
# ---------------------------------------------------------------------------


def build_dual_profile_decoder_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "fp16",
    opt_prefill_length: int = 64,
    max_prefill_length: int | None = None,
    quant_ctx: "QuantContext | None" = None,
    norm_type: str = "rmsnorm",
    mlp_type: str = "swiglu",
    position_type: str = "rope",
    activation: str = "silu",
    partial_rotary_factor: float = 1.0,
    interleaved_rope: bool = False,
    parallel_residual: bool = False,
    scale_attn_weights: bool = True,
    verbose: bool = False,
    dynamic_kv_profile_rows: list[int] | None = None,
) -> bytes:
    """Build a dual-profile prefill+decode engine with two optimization profiles.

    ``norm_type`` / ``mlp_type`` / ``position_type`` / ``activation`` /
    ``partial_rotary_factor`` / ``interleaved_rope`` / ``parallel_residual`` /
    ``scale_attn_weights`` mirror the same parameters on
    ``build_standard_decoder_engine``.

    ``quant_ctx`` (optional) routes every projection matmul through
    ``QuantContext.maybe_quantized_matmul`` for fp8 / int8 Q/DQ insertion;
    when ``None`` the matmuls are plain fp16 / bf16 / fp32.

    When ``dynamic_kv_profile_rows`` is provided, the engine carries one
    prefill profile (profile 0) followed by N decode profiles (profile 1..N),
    one per bucket — letting TriAttention pick the smallest active KV cache
    bucket at runtime while still benefitting from batched prefill on the
    prompt. The cache_k/cache_v inputs are declared dynamic so each profile
    can constrain their row count independently.
    """
    _supports_config(config, weights)

    if max_prefill_length is None:
        max_prefill_length = max_cache_length
    max_prefill_length = max(1, min(max_prefill_length, max_cache_length))
    opt_prefill_length = max(1, min(opt_prefill_length, max_prefill_length))

    multi_bucket_decode = bool(dynamic_kv_profile_rows)
    if multi_bucket_decode:
        decode_buckets: list[int] = []
        seen = set()
        for raw in dynamic_kv_profile_rows or []:
            clamped = max(1, min(int(raw), max_cache_length))
            if clamped not in seen:
                seen.add(clamped)
                decode_buckets.append(clamped)
        decode_buckets.sort()
        if not decode_buckets:
            decode_buckets = [max_cache_length]
            multi_bucket_decode = False

    attention_size = weights.get("_attention_size", config.attention_size)
    mlp_size = weights.get("_mlp_size", config.intermediate_size)
    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = attention_size // num_heads
    rotary_embedding_dim = int(head_dim * partial_rotary_factor)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "bf16":
        work_np_dtype, work_trt_dtype = np.float16, trt.bfloat16
    else:
        work_np_dtype, work_trt_dtype = np.float32, trt.float32

    # ---- Inputs (dynamic Sq) ---------------------------------------------
    token_id = network.add_input("token_id", trt.int32, (-1,))
    position_id = network.add_input("position_id", trt.int32, (-1,))
    attention_mask = network.add_input("attention_mask", trt.float32, (-1, -1))

    cache_shape: tuple[int, int]
    if multi_bucket_decode:
        cache_shape = (-1, attention_size)
    else:
        cache_shape = (max_cache_length, attention_size)
    cache_k_inputs: list[trt.ITensor] = []
    cache_v_inputs: list[trt.ITensor] = []
    for i in range(num_layers):
        ck = network.add_input(
            graph_ops.layer_tensor_name("cache_k", i),
            work_trt_dtype, cache_shape)
        cv = network.add_input(
            graph_ops.layer_tensor_name("cache_v", i),
            work_trt_dtype, cache_shape)
        cache_k_inputs.append(ck)
        cache_v_inputs.append(cv)

    # Cast mask to compute dtype for elementwise broadcast.
    if work_trt_dtype != trt.float32:
        attention_mask_work = network.add_cast(
            attention_mask, work_trt_dtype).get_output(0)
    else:
        attention_mask_work = attention_mask

    # Two (or 1+N) optimization profiles — same graph, different Sq / cache.
    def _add_profile(opt_sq: int, max_sq: int, *, fixed: bool = False,
                     cache_rows_min: int | None = None,
                     cache_rows_opt: int | None = None,
                     cache_rows_max: int | None = None):
        prof = builder.create_optimization_profile()
        min_sq = opt_sq if fixed else 1
        prof.set_shape("token_id", (min_sq,), (opt_sq,), (max_sq,))
        prof.set_shape("position_id", (min_sq,), (opt_sq,), (max_sq,))
        prof.set_shape(
            "attention_mask",
            (min_sq, max_cache_length + min_sq),
            (opt_sq, max_cache_length + opt_sq),
            (max_sq, max_cache_length + max_sq))
        if multi_bucket_decode:
            cmn = cache_rows_min if cache_rows_min is not None else 1
            cop = cache_rows_opt if cache_rows_opt is not None else max_cache_length
            cmx = cache_rows_max if cache_rows_max is not None else max_cache_length
            for i in range(num_layers):
                for name in (graph_ops.layer_tensor_name("cache_k", i),
                             graph_ops.layer_tensor_name("cache_v", i)):
                    prof.set_shape(
                        name,
                        (cmn, attention_size),
                        (cop, attention_size),
                        (cmx, attention_size))
        trt_config.add_optimization_profile(prof)

    _add_profile(opt_prefill_length, max_prefill_length, fixed=False,
                 cache_rows_min=1, cache_rows_opt=max_cache_length,
                 cache_rows_max=max_cache_length)
    if multi_bucket_decode:
        for bucket in decode_buckets:
            _add_profile(1, 1, fixed=True,
                         cache_rows_min=1, cache_rows_opt=bucket,
                         cache_rows_max=bucket)
    else:
        _add_profile(1, 1, fixed=True)

    # ---- Shared constants ------------------------------------------------
    embedding_table = _const_in_work_dtype(
        network, (vocab, hidden), weights["embedding"],
        work_np_dtype, work_trt_dtype)

    # RoPE tables (only when position_type == "rope"). Built for the worst
    # case key length max_cache_length + max_prefill_length, since RoPE is
    # gathered by position_id at runtime.
    #
    # Two table representations: the half-dim (max_S, rotary_dim // 2) tables
    # feed TRT's IRotaryEmbeddingLayer (fused kernel — both profiles pick it
    # up); the full-attention-size tables and rotate_half matrix back the
    # manual rotate-half fallback for tiny/odd configs that the native layer
    # cannot accept.
    cos_table: trt.ITensor | None = None
    sin_table: trt.ITensor | None = None
    rotate_half: trt.ITensor | None = None
    cos_half_table: trt.ITensor | None = None
    sin_half_table: trt.ITensor | None = None
    use_native_rope = False
    if position_type == "rope":
        kmax = max_cache_length + max_prefill_length
        cos_np = graph_ops.make_rope_table(
            kmax, attention_size, num_heads,
            config.rope_theta, True, partial_rotary_factor,
            interleaved=interleaved_rope)
        sin_np = graph_ops.make_rope_table(
            kmax, attention_size, num_heads,
            config.rope_theta, False, partial_rotary_factor,
            interleaved=interleaved_rope)
        rotate_half_np = graph_ops.make_rotate_half_matrix(
            attention_size, num_heads, partial_rotary_factor,
            interleaved=interleaved_rope)
        cos_table = _const_in_work_dtype(
            network, cos_np.shape, cos_np, work_np_dtype, work_trt_dtype)
        sin_table = _const_in_work_dtype(
            network, sin_np.shape, sin_np, work_np_dtype, work_trt_dtype)
        rotate_half = _const_in_work_dtype(
            network, (attention_size, attention_size), rotate_half_np,
            work_np_dtype, work_trt_dtype)

        # IRotaryEmbeddingLayer requires rotary_embedding_dim >= 2 and even.
        # Tiny synthetic configs (head_dim=4 with partial rotary producing an
        # odd dim) still fall back to the manual rotate-half path.
        if rotary_embedding_dim >= 2 and rotary_embedding_dim % 2 == 0:
            cos_half_np = graph_ops.make_rope_table_half_dim(
                kmax, head_dim, config.rope_theta, True,
                partial_rotary_factor, interleaved=interleaved_rope)
            sin_half_np = graph_ops.make_rope_table_half_dim(
                kmax, head_dim, config.rope_theta, False,
                partial_rotary_factor, interleaved=interleaved_rope)
            cos_half_table = _const_in_work_dtype(
                network, cos_half_np.shape, cos_half_np,
                work_np_dtype, work_trt_dtype)
            sin_half_table = _const_in_work_dtype(
                network, sin_half_np.shape, sin_half_np,
                work_np_dtype, work_trt_dtype)
            use_native_rope = True

    # Learned position embedding (GPT-2 / OPT / GPT-Neo / XGLM).
    position_embed_table: trt.ITensor | None = None
    if position_type == "learned":
        pos_embed_np = weights["position_embedding"]
        position_embed_table = _const_in_work_dtype(
            network, pos_embed_np.shape, pos_embed_np,
            work_np_dtype, work_trt_dtype)

    # ALiBi slopes + cache-slot positions for multi-row mask augmentation.
    alibi_slopes_tensor: trt.ITensor | None = None
    alibi_cache_positions_fp32: trt.ITensor | None = None
    if position_type == "alibi":
        alibi_slopes_np = graph_ops.compute_alibi_slopes(num_heads)
        # Slopes live as fp32 so the (key_pos - q_pos) math stays in fp32;
        # _alibi_mask_4d casts the final bias to work_trt_dtype before adding
        # to the additive mask.
        alibi_slopes_tensor = graph_ops.add_constant(
            network, (num_heads, 1, 1),
            alibi_slopes_np.reshape(num_heads, 1, 1), dtype=np.float32)
        # Cache slot k (for k in [0, max_cache_length)) holds the K/V at
        # position k. The current step's K/V live in slots
        # [max_cache_length, max_cache_length + Sq) and their positions come
        # from position_id at runtime, so we only pre-build the cache half.
        alibi_cache_positions_fp32 = graph_ops.add_constant(
            network, (max_cache_length,),
            np.arange(max_cache_length, dtype=np.float32), dtype=np.float32)

    # Attention scale.
    attn_scale = (1.0 / np.sqrt(max(head_dim, 1))) if scale_attn_weights else 1.0

    # Quantization-aware matmul (passes weight_name through to QuantContext).
    matmul = _make_matmul_fn(network, work_np_dtype, quant_ctx)

    # ---- Embedding -------------------------------------------------------
    emb = network.add_gather(embedding_table, token_id, 0)
    hidden_state = emb.get_output(0)  # (Sq, hidden)

    if position_type == "learned" and position_embed_table is not None:
        pos_gather = network.add_gather(position_embed_table, position_id, 0)
        pos_add = network.add_elementwise(
            hidden_state, pos_gather.get_output(0),
            trt.ElementWiseOperation.SUM)
        hidden_state = pos_add.get_output(0)

    # Make sure the main hidden stream is in the requested runtime dtype
    # before entering the layer stack (BF16 mode stores fp16 constants).
    if hidden_state.dtype != work_trt_dtype:
        hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)

    # Optional embedding LayerNorm (Bloom).
    embed_norm = weights.get("embedding_norm")
    if embed_norm is not None:
        embed_norm_beta = weights.get(
            "embedding_norm_beta", np.zeros(hidden, dtype=np.float32))
        hidden_state = _layer_norm_multi(
            network, hidden_state, hidden, embed_norm, embed_norm_beta,
            config.rms_norm_eps, work_np_dtype)

    # Build the 4D additive mask once — shared across layers. ALiBi
    # variants augment the mask with per-head linear bias.
    if position_type == "alibi":
        mask_4d = _alibi_mask_4d(
            network, attention_mask_work, position_id,
            alibi_slopes_tensor, alibi_cache_positions_fp32,
            num_heads, work_trt_dtype)
    else:
        mask_4d = _mask_to_4d(network, attention_mask_work)

    present_k_outs: list[trt.ITensor] = []
    present_v_outs: list[trt.ITensor] = []

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"

        # Pre-attention norm.
        normed = _norm_multi(
            network, hidden_state, hidden,
            weights[f"{prefix}.input_norm"],
            weights.get(f"{prefix}.input_norm_beta"),
            config.rms_norm_eps, norm_type, work_np_dtype)

        # Q / K / V projections.
        q = matmul(normed, hidden, attention_size,
                   weights[f"{prefix}.w_q"], f"{prefix}.w_q")
        k = matmul(normed, hidden, attention_size,
                   weights[f"{prefix}.w_k"], f"{prefix}.w_k")
        v = matmul(normed, hidden, attention_size,
                   weights[f"{prefix}.w_v"], f"{prefix}.w_v")

        # Optional QKV biases (Qwen2 / GPT-2 / OPT / Bloom / Falcon / etc.).
        q_bias = weights.get(f"{prefix}.q_bias")
        if q_bias is not None:
            q = graph_ops.add_bias_sum(
                network, q, attention_size, q_bias, dtype=work_np_dtype)
        k_bias = weights.get(f"{prefix}.k_bias")
        if k_bias is not None:
            k = graph_ops.add_bias_sum(
                network, k, attention_size, k_bias, dtype=work_np_dtype)
        v_bias = weights.get(f"{prefix}.v_bias")
        if v_bias is not None:
            v = graph_ops.add_bias_sum(
                network, v, attention_size, v_bias, dtype=work_np_dtype)

        # Optional per-head q/k norm (Qwen3).
        q_norm = weights.get(f"{prefix}.q_norm")
        if q_norm is not None:
            q = _rms_norm_per_head_multi(
                network, q, num_heads, head_dim, q_norm,
                config.rms_norm_eps, work_np_dtype)
        k_norm = weights.get(f"{prefix}.k_norm")
        if k_norm is not None:
            k = _rms_norm_per_head_multi(
                network, k, num_heads, head_dim, k_norm,
                config.rms_norm_eps, work_np_dtype)

        # Position embedding (RoPE only — learned was applied above; ALiBi
        # is added into the attention mask). Prefer the fused
        # IRotaryEmbeddingLayer when the rotary dim is even and >= 2 so
        # both profiles pick up the optimized RoPE kernel.
        if position_type == "rope":
            if use_native_rope:
                q = _apply_rope_native_multi(
                    network, q, num_heads, head_dim,
                    cos_half_table, sin_half_table, position_id,
                    rotary_embedding_dim, interleaved_rope)
                k = _apply_rope_native_multi(
                    network, k, num_heads, head_dim,
                    cos_half_table, sin_half_table, position_id,
                    rotary_embedding_dim, interleaved_rope)
            else:
                q = graph_ops.add_apply_rope(
                    network, q, position_id, cos_table, sin_table, rotate_half)
                k = graph_ops.add_apply_rope(
                    network, k, position_id, cos_table, sin_table, rotate_half)

        # Present K / V (this step's raw K / V), shape (Sq, attn_size).
        present_k_outs.append(k)
        present_v_outs.append(v)

        # Concatenate cached + current K / V along the sequence dim.
        all_k_cat = network.add_concatenation([cache_k_inputs[layer_idx], k])
        all_k_cat.axis = 0
        all_v_cat = network.add_concatenation([cache_v_inputs[layer_idx], v])
        all_v_cat.axis = 0

        q_4d = _to_heads_4d(network, q, num_heads, head_dim, f"{prefix}.q")
        k_4d = _to_heads_4d(network, all_k_cat.get_output(0), num_heads, head_dim,
                            f"{prefix}.k")
        v_4d = _to_heads_4d(network, all_v_cat.get_output(0), num_heads, head_dim,
                            f"{prefix}.v")

        ctx_4d = _attention_core(
            network, q_4d, k_4d, v_4d, mask_4d,
            head_dim=head_dim, scale=attn_scale)
        context = _from_heads_4d(network, ctx_4d, attention_size, f"{prefix}.ctx")

        attn_out = matmul(context, attention_size, hidden,
                          weights[f"{prefix}.w_o"], f"{prefix}.w_o")
        o_bias = weights.get(f"{prefix}.o_bias")
        if o_bias is not None:
            attn_out = graph_ops.add_bias_sum(
                network, attn_out, hidden, o_bias, dtype=work_np_dtype)

        # Residual structure: parallel (GPT-NeoX / CodeGen / Falcon-3) vs
        # sequential (everything else).
        if parallel_residual:
            post_attn_norm_w = weights.get(f"{prefix}.post_attn_norm")
            if post_attn_norm_w is not None:
                norm2 = _norm_multi(
                    network, hidden_state, hidden,
                    post_attn_norm_w,
                    weights.get(f"{prefix}.post_attn_norm_beta"),
                    config.rms_norm_eps, norm_type, work_np_dtype)
            else:
                norm2 = normed
        else:
            residual1 = network.add_elementwise(
                hidden_state, attn_out, trt.ElementWiseOperation.SUM)
            norm2 = _norm_multi(
                network, residual1.get_output(0), hidden,
                weights[f"{prefix}.post_attn_norm"],
                weights.get(f"{prefix}.post_attn_norm_beta"),
                config.rms_norm_eps, norm_type, work_np_dtype)

        # MLP — SwiGLU (Llama-style) or GeluFC (GPT-2-style).
        if mlp_type == "gelu_fc":
            mlp_out = _gelu_fc_mlp(
                network, norm2,
                matmul=matmul, weights=weights, prefix=prefix,
                hidden=hidden, mlp_size=mlp_size,
                activation=activation, work_np_dtype=work_np_dtype)
        else:
            mlp_out = _swiglu_mlp(
                network, norm2,
                matmul=matmul, weights=weights, prefix=prefix,
                hidden=hidden, mlp_size=mlp_size)

        # Final residual.
        if parallel_residual:
            sum_attn = network.add_elementwise(
                hidden_state, attn_out, trt.ElementWiseOperation.SUM)
            residual2 = network.add_elementwise(
                sum_attn.get_output(0), mlp_out, trt.ElementWiseOperation.SUM)
        else:
            residual2 = network.add_elementwise(
                residual1.get_output(0), mlp_out, trt.ElementWiseOperation.SUM)
        hidden_state = residual2.get_output(0)

    # ---- Final norm + LM head -------------------------------------------
    final_norm = weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = _norm_multi(
            network, hidden_state, hidden, final_norm,
            weights.get("final_norm_beta"),
            config.rms_norm_eps, norm_type, work_np_dtype)

    # Only the LAST prompt token's logits matter for the next-token sample,
    # so slice hidden_state from (Sq, hidden) to (1, hidden) before the LM
    # head. This keeps the output contract identical to the single-token
    # engine (logits shape = (1, vocab)) under both profiles and avoids
    # computing (Sq - 1) redundant vocab-sized matmul rows during prefill.
    shape_t = network.add_shape(hidden_state).get_output(0)  # [2] int64
    one_hidden = graph_ops.add_constant(
        network, (2,), np.array([1, hidden], dtype=np.int64), dtype=np.int64)
    start_sub = network.add_elementwise(
        shape_t, one_hidden, trt.ElementWiseOperation.SUB)
    start_t = start_sub.get_output(0)  # [Sq - 1, 0]
    size_t = graph_ops.add_constant(
        network, (2,), np.array([1, hidden], dtype=np.int64), dtype=np.int64)
    slicer = network.add_slice(hidden_state, start=(0, 0), shape=(0, 0), stride=(1, 1))
    slicer.set_input(1, start_t)
    slicer.set_input(2, size_t)
    last_hidden = slicer.get_output(0)

    out_vocab = (weights["w_out"].shape[1]
                 if isinstance(weights["w_out"], np.ndarray) else vocab)
    logits = graph_ops.add_matmul_rhs_constant(
        network, last_hidden, hidden, out_vocab, weights["w_out"],
        dtype=work_np_dtype)
    lm_bias = weights.get("lm_head_bias")
    if lm_bias is not None:
        logits = graph_ops.add_bias_sum(
            network, logits, out_vocab, lm_bias, dtype=work_np_dtype)
    else:
        zero_bias = np.zeros(out_vocab, dtype=work_np_dtype)
        logits = graph_ops.add_bias_sum(
            network, logits, out_vocab, zero_bias, dtype=work_np_dtype)

    if work_trt_dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)

    for i in range(num_layers):
        pk = present_k_outs[i]
        pv = present_v_outs[i]
        pk.name = graph_ops.layer_tensor_name("present_k", i)
        pv.name = graph_ops.layer_tensor_name("present_v", i)
        network.mark_output(pk)
        network.mark_output(pv)

    if verbose:
        print(f"[trtf-build] Building dual-profile engine "
              f"(layers={num_layers}, hidden={hidden}, attn={attention_size}, "
              f"mlp={mlp_size}, cache={max_cache_length}, "
              f"opt_prefill={opt_prefill_length}, max_prefill={max_prefill_length}, "
              f"norm={norm_type}, mlp_type={mlp_type}, pos={position_type}, "
              f"precision={precision}) ...",
              file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("dual-profile decoder engine build failed")
    return bytes(plan)
