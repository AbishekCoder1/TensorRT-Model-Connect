"""1:1 port of standard_decoder_graph_builder.cpp (multi-layer path only).

Builds a TensorRT engine for a standard pre-RMSNorm + GQA + RoPE + SwiGLU
decoder. Tensor names MUST match what the C++ runtime expects:
  Inputs:  token_id, position_id, attention_mask, cache_k_0..N, cache_v_0..N
  Outputs: logits, present_k_0..N, present_v_0..N
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt

from . import graph_ops
from .config import ModelConfig

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict


def build_standard_decoder_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    verbose: bool = False,
) -> bytes:
    """Build a TRT engine plan (serialized bytes) for a standard decoder.

    Args:
        config: Model architecture from config.json.
        weights: Loaded weight dict from checkpoint_mapper.
        max_cache_length: KV cache length (engine is compiled for this value).
        verbose: Print TRT builder logs.

    Returns:
        Serialized engine plan bytes.
    """
    attention_size: int = weights.get("_attention_size", config.attention_size)
    mlp_size: int = weights.get("_mlp_size", config.intermediate_size)
    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = attention_size // num_heads
    attention_window = max_cache_length + 1

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    trt_config.clear_flag(trt.BuilderFlag.TF32)

    # ---------------------------------------------------------------
    # Inputs
    # ---------------------------------------------------------------
    token_id = network.add_input("token_id", trt.int32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))
    attention_mask = network.add_input(
        "attention_mask", trt.float32, (1, attention_window))

    cache_k_inputs = []
    cache_v_inputs = []
    for i in range(num_layers):
        ck = network.add_input(
            graph_ops.layer_tensor_name("cache_k", i),
            trt.float32, (max_cache_length, attention_size))
        cv = network.add_input(
            graph_ops.layer_tensor_name("cache_v", i),
            trt.float32, (max_cache_length, attention_size))
        cache_k_inputs.append(ck)
        cache_v_inputs.append(cv)

    # ---------------------------------------------------------------
    # Shared constants
    # ---------------------------------------------------------------
    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), weights["embedding"])

    cos_table_np = graph_ops.make_rope_table(
        attention_window, attention_size, num_heads, config.rope_theta, True)
    sin_table_np = graph_ops.make_rope_table(
        attention_window, attention_size, num_heads, config.rope_theta, False)
    rotate_half_np = graph_ops.make_rotate_half_matrix(
        attention_size, num_heads)

    cos_tensor = graph_ops.add_constant(
        network, (attention_window, attention_size), cos_table_np)
    sin_tensor = graph_ops.add_constant(
        network, (attention_window, attention_size), sin_table_np)
    rotate_half_tensor = graph_ops.add_constant(
        network, (attention_size, attention_size), rotate_half_np)
    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=np.float32))
    attn_scale = 1.0 / np.sqrt(max(head_dim, 1))
    attn_scale_tensor = graph_ops.add_constant(
        network, (1, 1, 1), np.array([attn_scale], dtype=np.float32))

    # ---------------------------------------------------------------
    # Embedding lookup
    # ---------------------------------------------------------------
    gather = network.add_gather(embedding_table, token_id, 0)
    hidden_state = gather.get_output(0)

    # ---------------------------------------------------------------
    # Decoder layers
    # ---------------------------------------------------------------
    present_k_outputs = []
    present_v_outputs = []

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"

        result = _add_decoder_layer(
            network=network,
            hidden=hidden_state,
            cache_k=cache_k_inputs[layer_idx],
            cache_v=cache_v_inputs[layer_idx],
            attention_mask=attention_mask,
            position_id=position_id,
            cos_tensor=cos_tensor,
            sin_tensor=sin_tensor,
            rotate_half_tensor=rotate_half_tensor,
            attn_scale_tensor=attn_scale_tensor,
            eps_tensor=eps_tensor,
            weights=weights,
            prefix=prefix,
            hidden_size=hidden,
            attention_size=attention_size,
            mlp_size=mlp_size,
            num_heads=num_heads,
            head_dim=head_dim,
            max_cache_length=max_cache_length,
        )

        hidden_state = result["hidden"]
        present_k_outputs.append(result["present_k"])
        present_v_outputs.append(result["present_v"])

    # ---------------------------------------------------------------
    # Final norm
    # ---------------------------------------------------------------
    final_norm = weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = graph_ops.add_rms_norm(
            network, hidden_state, hidden, final_norm, eps_tensor)

    # ---------------------------------------------------------------
    # LM head (logits)
    # ---------------------------------------------------------------
    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden, vocab, weights["w_out"])
    # Zero bias — match C++ behavior
    b_out = np.zeros(vocab, dtype=np.float32)
    logits = graph_ops.add_bias_sum(network, logits, vocab, b_out)

    logits.name = "logits"
    network.mark_output(logits)

    # ---------------------------------------------------------------
    # Present K/V outputs
    # ---------------------------------------------------------------
    for i in range(num_layers):
        pk = present_k_outputs[i]
        pv = present_v_outputs[i]
        pk.name = graph_ops.layer_tensor_name("present_k", i)
        pv.name = graph_ops.layer_tensor_name("present_v", i)
        network.mark_output(pk)
        network.mark_output(pv)

    # ---------------------------------------------------------------
    # Build engine
    # ---------------------------------------------------------------
    if verbose:
        print(f"[trtf-build] Building TRT engine ({num_layers} layers, "
              f"hidden={hidden}, attn={attention_size}, mlp={mlp_size}, "
              f"cache={max_cache_length}) ...", file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")

    return bytes(plan)


def _add_decoder_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    position_id: trt.ITensor,
    cos_tensor: trt.ITensor,
    sin_tensor: trt.ITensor,
    rotate_half_tensor: trt.ITensor,
    attn_scale_tensor: trt.ITensor,
    eps_tensor: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    attention_size: int,
    mlp_size: int,
    num_heads: int,
    head_dim: int,
    max_cache_length: int,
) -> dict[str, trt.ITensor]:
    """Add one standard decoder layer block. Returns hidden, present_k, present_v."""
    attention_window = max_cache_length + 1

    # Pre-attention RMSNorm
    norm1 = graph_ops.add_rms_norm(
        network, hidden, hidden_size,
        weights[f"{prefix}.input_norm"], eps_tensor)

    # QKV projections
    q = graph_ops.add_matmul_rhs_constant(
        network, norm1, hidden_size, attention_size,
        weights[f"{prefix}.w_q"])
    k = graph_ops.add_matmul_rhs_constant(
        network, norm1, hidden_size, attention_size,
        weights[f"{prefix}.w_k"])
    v = graph_ops.add_matmul_rhs_constant(
        network, norm1, hidden_size, attention_size,
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

    # RoPE
    q = graph_ops.add_apply_rope(
        network, q, position_id, cos_tensor, sin_tensor, rotate_half_tensor)
    k = graph_ops.add_apply_rope(
        network, k, position_id, cos_tensor, sin_tensor, rotate_half_tensor)

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

    # Residual connection
    residual1 = network.add_elementwise(
        hidden, attn_out, trt.ElementWiseOperation.SUM)

    # Post-attention RMSNorm
    norm2 = graph_ops.add_rms_norm(
        network, residual1.get_output(0), hidden_size,
        weights[f"{prefix}.post_attn_norm"], eps_tensor)

    # SwiGLU MLP
    gate = graph_ops.add_matmul_rhs_constant(
        network, norm2, hidden_size, mlp_size,
        weights[f"{prefix}.w_gate"])
    up = graph_ops.add_matmul_rhs_constant(
        network, norm2, hidden_size, mlp_size,
        weights[f"{prefix}.w_up"])

    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(
        gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(
        swish.get_output(0), up, trt.ElementWiseOperation.PROD)

    down = graph_ops.add_matmul_rhs_constant(
        network, gated.get_output(0), mlp_size, hidden_size,
        weights[f"{prefix}.w_down"])

    # Residual connection
    residual2 = network.add_elementwise(
        residual1.get_output(0), down, trt.ElementWiseOperation.SUM)

    return {
        "hidden": residual2.get_output(0),
        "present_k": present_k,
        "present_v": present_v,
    }
