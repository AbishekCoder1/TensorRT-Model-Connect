"""Dual-profile decoder engine builder — single engine, two optimization profiles.

Produces one TensorRT engine that handles both prefill (multi-token) and
decode (single-token) phases by switching between two optimization profiles
at runtime:
  * Profile 0 (prefill): Sq ranges over [1, opt=opt_prefill_length, max=max_prefill_length].
    TensorRT picks batched MHA kernels (e.g. `_gemm_mha_v2`) at opt Sq.
  * Profile 1 (decode): Sq fixed to 1. TensorRT picks the GEMV fast-path
    (`_gemv_mha_v1`).

Both profiles use the same graph and weights — only the optimization
profile differs, so the engine's weights live once in GPU memory and
the C++ runtime creates two IExecutionContexts (one per profile) that
share the engine.

Scope: Qwen / LLaMA-style configs only (rmsnorm + swiglu + rope, optional
q_norm/k_norm, no biases, GQA weights pre-expanded to
attention_size = num_heads * head_dim). Other variants (ALiBi, learned
position, LayerNorm, GeluFC MLP, parallel residual, VL embed_input)
remain on the legacy `standard_decoder_builder` path.

Tensor contract (matches the existing C++ runtime KvCache naming):
  Inputs (dynamic shapes — Sq varies by profile)
    token_id        int32   (-1,)
    position_id     int32   (-1,)
    attention_mask  float32 (-1, -1)                 # (Sq, max_cache + Sq)
    cache_k_i       fp16/f32 (max_cache, attn_size)  # static
    cache_v_i       fp16/f32 (max_cache, attn_size)  # static
  Outputs
    logits          float32 (-1, vocab)              # only last row used at runtime
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


def _rms_norm_multi(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden: int,
    gamma: np.ndarray,
    eps: float,
    dtype: np.dtype,
) -> trt.ITensor:
    """RMSNorm on (Sq, hidden) — reduces along the hidden dim."""
    need_cast = dtype != np.float32
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
        trt_dtype = trt.float16 if dtype == np.float16 else trt.float32
        out = network.add_cast(out, trt_dtype).get_output(0)
    return out


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
    reshape_in = network.add_shuffle(inp)
    reshape_in.reshape_dims = (-1, num_heads, head_dim)
    x = reshape_in.get_output(0)

    need_cast = dtype != np.float32
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
        trt_dtype = trt.float16 if dtype == np.float16 else trt.float32
        out = network.add_cast(out, trt_dtype).get_output(0)
    flat = network.add_shuffle(out)
    flat.reshape_dims = (-1, num_heads * head_dim)
    return flat.get_output(0)


def _to_heads_4d(
    network: trt.INetworkDefinition,
    x: trt.ITensor,
    num_heads: int,
    head_dim: int,
    tag: str,
) -> trt.ITensor:
    """(S, num_heads * head_dim) -> (1, num_heads, S, head_dim).

    We need the actual transpose between S and H (row-major reshape alone
    would mis-interleave heads and positions). Two shuffles: reshape to
    (S, H, D) with transpose [1, 0, 2] to produce (H, S, D), then reshape
    to (1, H, S, D).
    """
    r1 = network.add_shuffle(x)
    r1.name = tag + "_s_h_d"
    r1.reshape_dims = (-1, num_heads, head_dim)
    r1.second_transpose = trt.Permutation([1, 0, 2])

    r2 = network.add_shuffle(r1.get_output(0))
    r2.name = tag + "_1_h_s_d"
    r2.reshape_dims = (1, num_heads, -1, head_dim)
    return r2.get_output(0)


def _from_heads_4d(
    network: trt.INetworkDefinition,
    ctx_4d: trt.ITensor,
    attention_size: int,
    tag: str,
) -> trt.ITensor:
    """(1, num_heads, S, head_dim) -> (S, num_heads * head_dim)."""
    # Drop batch + transpose back: (H, S, D) -> (S, H, D)
    squeeze = network.add_shuffle(ctx_4d)
    squeeze.name = tag + "_h_s_d"
    squeeze.first_transpose = trt.Permutation([0, 2, 1, 3])  # (1, S, H, D)
    squeeze.reshape_dims = (-1, attention_size)              # flatten batch+S and H*D
    return squeeze.get_output(0)


def _mask_to_4d(
    network: trt.INetworkDefinition,
    mask_2d: trt.ITensor,
) -> trt.ITensor:
    """(Sq, K) -> (1, 1, Sq, K) via shape-tensor-driven shuffle.

    The mask has TWO dynamic dimensions, so reshape_dims with multiple -1
    is ambiguous. Build the target shape explicitly from IShapeLayer.
    """
    shp = network.add_shape(mask_2d).get_output(0)  # [2] int64
    ones = graph_ops.add_constant(
        network, (2,), np.array([1, 1], dtype=np.int64), dtype=np.int64)
    concat = network.add_concatenation([ones, shp])
    concat.axis = 0
    target = concat.get_output(0)
    shuffle = network.add_shuffle(mask_2d)
    shuffle.set_input(1, target)
    return shuffle.get_output(0)


def _attention_core(
    network: trt.INetworkDefinition,
    q_4d: trt.ITensor,
    k_4d: trt.ITensor,
    v_4d: trt.ITensor,
    mask_4d: trt.ITensor,
) -> trt.ITensor:
    """IAttention with explicit mask (no causal flag — mask carries causality)."""
    head_dim = q_4d.shape[-1]
    scale = float(1.0 / np.sqrt(max(head_dim, 1)))
    if q_4d.dtype == trt.float16:
        scale_np = np.array([[[[scale]]]], dtype=np.float16)
        scale_t = graph_ops.add_constant(network, (1, 1, 1, 1), scale_np, dtype=np.float16)
    else:
        scale_np = np.array([[[[scale]]]], dtype=np.float32)
        scale_t = graph_ops.add_constant(network, (1, 1, 1, 1), scale_np, dtype=np.float32)
        if q_4d.dtype == trt.bfloat16:
            scale_t = network.add_cast(scale_t, trt.bfloat16).get_output(0)
    q_scaled = network.add_elementwise(q_4d, scale_t, trt.ElementWiseOperation.PROD)

    attn = network.add_attention(
        q_scaled.get_output(0), k_4d, v_4d,
        trt.AttentionNormalizationOp.SOFTMAX, False)
    attn.mask = mask_4d
    # Allow TRT to decompose into primitives when no fused MHA kernel
    # is available for this (dtype, head_dim, seq) combination. On
    # real targets (fp16, head_dim=128, opt Sq=64) TRT still picks the
    # named _mha_v2 kernel from profile 0 — verified via
    # `trtexec --useProfile=0 --dumpProfile` on Qwen3-0.6B. The
    # fallback path keeps tiny unit-test configs (head_dim=4, fp32)
    # buildable since TRT has no fused kernel for those shapes.
    attn.decomposable = True
    return attn.get_output(0)


def _supports_config(config: "ModelConfig", weights: "WeightDict") -> None:
    """Guard: this builder only supports the Qwen-style standard decoder."""
    model_type = getattr(config, "model_type", "").lower()
    if "moe" in model_type or "mamba" in model_type or "rwkv" in model_type:
        raise NotImplementedError(
            f"dual_profile_decoder_builder does not support model_type={model_type!r}")
    if "embedding" not in weights:
        raise NotImplementedError("missing embedding weight")
    if "final_norm" not in weights:
        raise NotImplementedError("missing final_norm weight")


def build_dual_profile_decoder_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "fp16",
    opt_prefill_length: int = 64,
    max_prefill_length: int | None = None,
    interleaved_rope: bool = False,
    verbose: bool = False,
    dynamic_kv_profile_rows: list[int] | None = None,
) -> bytes:
    """Build a dual-profile prefill+decode engine with two optimization profiles.

    When ``dynamic_kv_profile_rows`` is provided, the engine carries one
    prefill profile (profile 0) followed by N decode profiles (profile 1..N),
    one per bucket — letting TriAttention pick the smallest active KV cache
    bucket at runtime while still benefitting from batched-prefill on the
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
        # Sanitize: clamp, dedupe, sort. Always include max_cache_length so
        # the runtime can fall back to the full bucket once compaction
        # restores the cache to <= max_cache_length rows.
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
    cache_k_inputs = []
    cache_v_inputs = []
    for i in range(num_layers):
        ck = network.add_input(
            graph_ops.layer_tensor_name("cache_k", i),
            work_trt_dtype, cache_shape)
        cv = network.add_input(
            graph_ops.layer_tensor_name("cache_v", i),
            work_trt_dtype, cache_shape)
        cache_k_inputs.append(ck)
        cache_v_inputs.append(cv)

    # Cast mask to compute dtype for elementwise broadcast
    if work_trt_dtype != trt.float32:
        attention_mask_work = network.add_cast(attention_mask, work_trt_dtype).get_output(0)
    else:
        attention_mask_work = attention_mask

    # Two optimization profiles (same graph, different Sq)
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

    # Profile 0: prefill (TRT picks batched MHA kernel at opt=opt_prefill_length)
    _add_profile(opt_prefill_length, max_prefill_length, fixed=False,
                 cache_rows_min=1, cache_rows_opt=max_cache_length,
                 cache_rows_max=max_cache_length)
    if multi_bucket_decode:
        # Profiles 1..N: decode (Sq=1) at increasing cache buckets.
        for bucket in decode_buckets:
            _add_profile(1, 1, fixed=True,
                         cache_rows_min=1, cache_rows_opt=bucket,
                         cache_rows_max=bucket)
    else:
        # Profile 1: decode (Sq=1 everywhere; GEMV fast-path)
        _add_profile(1, 1, fixed=True)

    # ---- Shared constants -----------------------------------------------
    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype)

    cos_np = graph_ops.make_rope_table(
        max_cache_length + max_prefill_length, attention_size, num_heads,
        config.rope_theta, True, 1.0, interleaved=interleaved_rope)
    sin_np = graph_ops.make_rope_table(
        max_cache_length + max_prefill_length, attention_size, num_heads,
        config.rope_theta, False, 1.0, interleaved=interleaved_rope)
    rotate_half_np = graph_ops.make_rotate_half_matrix(
        attention_size, num_heads, 1.0, interleaved=interleaved_rope)
    cos_table = graph_ops.add_constant(
        network, cos_np.shape, cos_np, dtype=work_np_dtype)
    sin_table = graph_ops.add_constant(
        network, sin_np.shape, sin_np, dtype=work_np_dtype)
    rotate_half = graph_ops.add_constant(
        network, (attention_size, attention_size), rotate_half_np,
        dtype=work_np_dtype)

    # ---- Embedding -------------------------------------------------------
    emb = network.add_gather(embedding_table, token_id, 0)
    hidden_state = emb.get_output(0)  # (Sq, hidden)

    # Broadcast mask to 4D once — shared across layers.
    mask_4d = _mask_to_4d(network, attention_mask_work)

    present_k_outs: list[trt.ITensor] = []
    present_v_outs: list[trt.ITensor] = []

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"

        normed = _rms_norm_multi(
            network, hidden_state, hidden,
            weights[f"{prefix}.input_norm"], config.rms_norm_eps, work_np_dtype)

        q = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden, attention_size,
            weights[f"{prefix}.w_q"], dtype=work_np_dtype)
        k = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden, attention_size,
            weights[f"{prefix}.w_k"], dtype=work_np_dtype)
        v = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden, attention_size,
            weights[f"{prefix}.w_v"], dtype=work_np_dtype)

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

        # Rotary position embedding via per-position gather (works for any Sq)
        q = graph_ops.add_apply_rope(
            network, q, position_id, cos_table, sin_table, rotate_half)
        k = graph_ops.add_apply_rope(
            network, k, position_id, cos_table, sin_table, rotate_half)

        # Present K/V (this step's raw K/V) — shape (Sq, attn_size)
        present_k_outs.append(k)
        present_v_outs.append(v)

        # Concatenate cached K/V with current K/V along the sequence dim:
        # (max_cache, attn_size) cat (Sq, attn_size) = (max_cache + Sq, attn_size)
        all_k_cat = network.add_concatenation([cache_k_inputs[layer_idx], k])
        all_k_cat.axis = 0
        all_v_cat = network.add_concatenation([cache_v_inputs[layer_idx], v])
        all_v_cat.axis = 0

        q_4d = _to_heads_4d(network, q, num_heads, head_dim, f"{prefix}.q")
        k_4d = _to_heads_4d(network, all_k_cat.get_output(0), num_heads, head_dim,
                            f"{prefix}.k")
        v_4d = _to_heads_4d(network, all_v_cat.get_output(0), num_heads, head_dim,
                            f"{prefix}.v")

        ctx_4d = _attention_core(network, q_4d, k_4d, v_4d, mask_4d)
        context = _from_heads_4d(network, ctx_4d, attention_size, f"{prefix}.ctx")

        attn_out = graph_ops.add_matmul_rhs_constant(
            network, context, attention_size, hidden,
            weights[f"{prefix}.w_o"], dtype=work_np_dtype)

        residual1 = network.add_elementwise(
            hidden_state, attn_out, trt.ElementWiseOperation.SUM)
        norm2 = _rms_norm_multi(
            network, residual1.get_output(0), hidden,
            weights[f"{prefix}.post_attn_norm"],
            config.rms_norm_eps, work_np_dtype)

        gate = graph_ops.add_matmul_rhs_constant(
            network, norm2, hidden, mlp_size,
            weights[f"{prefix}.w_gate"], dtype=work_np_dtype)
        up = graph_ops.add_matmul_rhs_constant(
            network, norm2, hidden, mlp_size,
            weights[f"{prefix}.w_up"], dtype=work_np_dtype)
        silu = network.add_activation(gate, trt.ActivationType.SIGMOID)
        silu_gate = network.add_elementwise(
            gate, silu.get_output(0), trt.ElementWiseOperation.PROD)
        gate_up = network.add_elementwise(
            silu_gate.get_output(0), up, trt.ElementWiseOperation.PROD)
        mlp_out = graph_ops.add_matmul_rhs_constant(
            network, gate_up.get_output(0), mlp_size, hidden,
            weights[f"{prefix}.w_down"], dtype=work_np_dtype)

        residual2 = network.add_elementwise(
            residual1.get_output(0), mlp_out, trt.ElementWiseOperation.SUM)
        hidden_state = residual2.get_output(0)

    # ---- Final norm + LM head -------------------------------------------
    hidden_state = _rms_norm_multi(
        network, hidden_state, hidden,
        weights["final_norm"], config.rms_norm_eps, work_np_dtype)

    # Only the LAST prompt token's logits matter for the next-token
    # sample, so slice hidden_state from (Sq, hidden) to (1, hidden)
    # before the LM head. This keeps the output contract identical to
    # the single-token engine (logits shape = (1, vocab)) under both
    # profiles and avoids computing (Sq-1) redundant vocab-sized
    # matmul rows during prefill.
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
              f"precision={precision}) ...",
              file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("dual-profile decoder engine build failed")
    return bytes(plan)
