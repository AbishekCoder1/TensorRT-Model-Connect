"""Qwen3 non-autoregressive text encoder builder.

Builds a TRT engine for the Qwen3 model used as a text encoder in Z-Image.
Unlike the standard decoder builder, this:
  - Processes the entire sequence at once (no KV cache)
  - Uses bidirectional causal attention (full attention mask)
  - Returns hidden_states from a configurable layer (default: layer -2)

Engine I/O:
    Inputs:  input_ids [seq_len] int32, attention_mask [seq_len] float32
    Outputs: text_embeddings [seq_len, hidden_size] float32
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorrt as trt

from . import graph_ops
from .checkpoint_mapper import WeightDict, _open_safetensors, _load_tensor, _has_tensor


def load_qwen3_encoder_weights(
    model_dir: str,
    *,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    intermediate_size: int,
    vocab_size: int,
) -> WeightDict:
    """Load Qwen3 encoder weights from HF safetensors."""
    model_path = Path(model_dir)
    readers = _open_safetensors(model_path)
    weights = WeightDict()

    def _t(name: str) -> np.ndarray:
        w = _load_tensor(readers, name)
        return np.ascontiguousarray(w.T, dtype=np.float32)

    def _f(name: str) -> np.ndarray:
        return _load_tensor(readers, name).astype(np.float32)

    # Embedding
    weights["embed_tokens"] = _f("model.embed_tokens.weight")

    for i in range(num_layers):
        p = f"model.layers.{i}"

        # Self-attention projections (transposed for matmul)
        weights[f"layer.{i}.q_proj"] = _t(f"{p}.self_attn.q_proj.weight")
        weights[f"layer.{i}.k_proj"] = _t(f"{p}.self_attn.k_proj.weight")
        weights[f"layer.{i}.v_proj"] = _t(f"{p}.self_attn.v_proj.weight")
        weights[f"layer.{i}.o_proj"] = _t(f"{p}.self_attn.o_proj.weight")

        # QK norms
        weights[f"layer.{i}.q_norm"] = _f(f"{p}.self_attn.q_norm.weight")
        weights[f"layer.{i}.k_norm"] = _f(f"{p}.self_attn.k_norm.weight")

        # RMSNorm
        weights[f"layer.{i}.input_layernorm"] = _f(f"{p}.input_layernorm.weight")
        weights[f"layer.{i}.post_attn_norm"] = _f(f"{p}.post_attention_layernorm.weight")

        # SwiGLU MLP
        weights[f"layer.{i}.gate_proj"] = _t(f"{p}.mlp.gate_proj.weight")
        weights[f"layer.{i}.up_proj"] = _t(f"{p}.mlp.up_proj.weight")
        weights[f"layer.{i}.down_proj"] = _t(f"{p}.mlp.down_proj.weight")

    # Final norm (only needed if output_layer < num_layers)
    if _has_tensor(readers, "model.norm.weight"):
        weights["final_norm"] = _f("model.norm.weight")

    return weights


def build_qwen3_encoder_engine(
    weights: WeightDict,
    *,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    intermediate_size: int,
    vocab_size: int,
    max_seq_len: int,
    rope_theta: float = 1000000.0,
    eps: float = 1e-6,
    output_layer: int = -2,
    verbose: bool = False,
) -> bytes:
    """Build Qwen3 text encoder TRT engine.

    Args:
        output_layer: Which layer's output to return. -2 means second-to-last.
        All other args describe the Qwen3 architecture.
    """
    if output_layer < 0:
        output_layer = num_layers + output_layer  # e.g., 36 + (-2) = 34

    kv_dim = num_kv_heads * head_dim
    attn_scale = 1.0 / np.sqrt(max(head_dim, 1))

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 30)
    config.clear_flag(trt.BuilderFlag.TF32)

    network = builder.create_network()

    # Inputs
    input_ids = network.add_input("input_ids", trt.int32, (max_seq_len,))
    attn_mask = network.add_input("attention_mask", trt.float32, (max_seq_len,))

    # Constants
    eps_t = graph_ops.add_constant(network, (1, 1), np.array([eps], dtype=np.float32))
    scale_t = graph_ops.add_constant(
        network, (1, 1, 1), np.array([attn_scale], dtype=np.float32))

    # Embedding
    embed_table = graph_ops.add_constant(
        network, (vocab_size, hidden_size), weights["embed_tokens"])
    hidden = network.add_gather(embed_table, input_ids, 0).get_output(0)

    # Build RoPE cos/sin tables
    rope_cos_np = _build_rope_table(max_seq_len, num_heads, head_dim, rope_theta, cosine=True)
    rope_sin_np = _build_rope_table(max_seq_len, num_heads, head_dim, rope_theta, cosine=False)
    rope_cos = graph_ops.add_constant(
        network, (max_seq_len, num_heads * head_dim), rope_cos_np)
    rope_sin = graph_ops.add_constant(
        network, (max_seq_len, num_heads * head_dim), rope_sin_np)

    # Build rotate-half matrix
    rot_half_np = graph_ops.make_rotate_half_matrix(
        num_heads * head_dim, num_heads, interleaved=False)
    rot_half = graph_ops.add_constant(
        network, (num_heads * head_dim, num_heads * head_dim), rot_half_np)

    # KV-head RoPE tables (if GQA)
    if num_kv_heads != num_heads:
        kv_rope_cos_np = _build_rope_table(max_seq_len, num_kv_heads, head_dim, rope_theta, cosine=True)
        kv_rope_sin_np = _build_rope_table(max_seq_len, num_kv_heads, head_dim, rope_theta, cosine=False)
        kv_rope_cos = graph_ops.add_constant(
            network, (max_seq_len, num_kv_heads * head_dim), kv_rope_cos_np)
        kv_rope_sin = graph_ops.add_constant(
            network, (max_seq_len, num_kv_heads * head_dim), kv_rope_sin_np)
        kv_rot_half_np = graph_ops.make_rotate_half_matrix(
            num_kv_heads * head_dim, num_kv_heads, interleaved=False)
        kv_rot_half = graph_ops.add_constant(
            network, (num_kv_heads * head_dim, num_kv_heads * head_dim), kv_rot_half_np)
    else:
        kv_rope_cos = rope_cos
        kv_rope_sin = rope_sin
        kv_rot_half = rot_half

    # Build attention mask: [seq_len] -> [1, seq_len] for broadcast in softmax
    # We mask padded positions with -1e9
    # attn_mask input: 0.0 for valid tokens, -1e9 for padding
    # Reshape for broadcast: [1, 1, seq_len] for [num_heads, seq_len, seq_len]
    mask_reshape = network.add_shuffle(attn_mask)
    mask_reshape.reshape_dims = (1, 1, max_seq_len)

    for layer_idx in range(num_layers):
        if layer_idx == output_layer:
            # Save this hidden state as output
            output_hidden = hidden

        # RMSNorm
        normed = graph_ops.add_rms_norm(
            network, hidden, hidden_size,
            weights[f"layer.{layer_idx}.input_layernorm"], eps_t)

        # QKV projections
        q = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, num_heads * head_dim,
            weights[f"layer.{layer_idx}.q_proj"])
        k = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, kv_dim,
            weights[f"layer.{layer_idx}.k_proj"])
        v = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, kv_dim,
            weights[f"layer.{layer_idx}.v_proj"])

        # QK norms (per-head RMSNorm)
        q_norm_w = weights[f"layer.{layer_idx}.q_norm"]
        k_norm_w = weights[f"layer.{layer_idx}.k_norm"]
        # Tile per-head norm weights for all heads
        q_norm_tiled = np.tile(q_norm_w.reshape(1, head_dim), (num_heads, 1))
        k_norm_tiled = np.tile(k_norm_w.reshape(1, head_dim), (num_kv_heads, 1))

        q = _add_per_head_rms_norm(network, q, num_heads, head_dim, q_norm_tiled, eps_t, max_seq_len)
        k = _add_per_head_rms_norm(network, k, num_kv_heads, head_dim, k_norm_tiled, eps_t, max_seq_len)

        # Apply RoPE
        q = _apply_rope(network, q, rope_cos, rope_sin, rot_half)
        k = _apply_rope(network, k, kv_rope_cos, kv_rope_sin, kv_rot_half)

        # GQA: expand K,V from num_kv_heads to num_heads
        if num_kv_heads != num_heads:
            repeat = num_heads // num_kv_heads
            k = _expand_kv_heads(network, k, num_kv_heads, head_dim, repeat, max_seq_len)
            v = _expand_kv_heads(network, v, num_kv_heads, head_dim, repeat, max_seq_len)

        # Multi-head attention: [seq, dim] -> [heads, seq, head_dim]
        q_h = network.add_shuffle(q)
        q_h.reshape_dims = (max_seq_len, num_heads, head_dim)
        q_h.second_transpose = trt.Permutation([1, 0, 2])

        k_h = network.add_shuffle(k)
        k_h.reshape_dims = (max_seq_len, num_heads, head_dim)
        k_h.second_transpose = trt.Permutation([1, 0, 2])

        v_h = network.add_shuffle(v)
        v_h.reshape_dims = (max_seq_len, num_heads, head_dim)
        v_h.second_transpose = trt.Permutation([1, 0, 2])

        # Scores: Q @ K^T -> [heads, seq, seq]
        score = network.add_matrix_multiply(
            q_h.get_output(0), trt.MatrixOperation.NONE,
            k_h.get_output(0), trt.MatrixOperation.TRANSPOSE)
        scaled_score = network.add_elementwise(
            score.get_output(0), scale_t, trt.ElementWiseOperation.PROD)

        # Apply padding mask (broadcast from [1, 1, seq] to [heads, seq, seq])
        masked = network.add_elementwise(
            scaled_score.get_output(0), mask_reshape.get_output(0),
            trt.ElementWiseOperation.SUM)

        softmax = network.add_softmax(masked.get_output(0))
        softmax.axes = 1 << 2

        # Context
        ctx = network.add_matrix_multiply(
            softmax.get_output(0), trt.MatrixOperation.NONE,
            v_h.get_output(0), trt.MatrixOperation.NONE)

        ctx_flat = network.add_shuffle(ctx.get_output(0))
        ctx_flat.first_transpose = trt.Permutation([1, 0, 2])
        ctx_flat.reshape_dims = (max_seq_len, num_heads * head_dim)

        # Output projection
        attn_out = graph_ops.add_matmul_rhs_constant(
            network, ctx_flat.get_output(0), num_heads * head_dim, hidden_size,
            weights[f"layer.{layer_idx}.o_proj"])

        # Residual
        hidden = network.add_elementwise(
            hidden, attn_out, trt.ElementWiseOperation.SUM).get_output(0)

        # Post-attention RMSNorm
        normed2 = graph_ops.add_rms_norm(
            network, hidden, hidden_size,
            weights[f"layer.{layer_idx}.post_attn_norm"], eps_t)

        # SwiGLU MLP
        gate = graph_ops.add_matmul_rhs_constant(
            network, normed2, hidden_size, intermediate_size,
            weights[f"layer.{layer_idx}.gate_proj"])
        up = graph_ops.add_matmul_rhs_constant(
            network, normed2, hidden_size, intermediate_size,
            weights[f"layer.{layer_idx}.up_proj"])

        # SiLU(gate) * up
        sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
        silu = network.add_elementwise(
            gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
        gated = network.add_elementwise(
            silu.get_output(0), up, trt.ElementWiseOperation.PROD)

        down = graph_ops.add_matmul_rhs_constant(
            network, gated.get_output(0), intermediate_size, hidden_size,
            weights[f"layer.{layer_idx}.down_proj"])

        # Residual
        hidden = network.add_elementwise(
            hidden, down, trt.ElementWiseOperation.SUM).get_output(0)

    # Use the output from the target layer
    if output_layer >= num_layers:
        output_hidden = hidden
    elif output_layer < 0:
        output_hidden = hidden

    output_hidden.name = "text_embeddings"
    output_hidden.dtype = trt.float32
    network.mark_output(output_hidden)

    print(f"[qwen3-encoder] Building TRT engine "
          f"(layers={num_layers}, hidden={hidden_size}, output_layer={output_layer}, "
          f"seq_len={max_seq_len}) ...", file=sys.stderr)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("Qwen3 encoder TRT engine build failed")
    return bytes(plan)


def _build_rope_table(
    max_seq_len: int,
    num_heads: int,
    head_dim: int,
    theta: float,
    cosine: bool,
) -> np.ndarray:
    """Build RoPE cos/sin table [max_seq_len, num_heads * head_dim]."""
    total_dim = num_heads * head_dim
    half = head_dim // 2
    table = np.full((max_seq_len, total_dim), 1.0 if cosine else 0.0, dtype=np.float32)

    for pos in range(max_seq_len):
        for head in range(num_heads):
            for d in range(head_dim):
                freq_idx = d % half
                exponent = (2.0 * freq_idx) / head_dim
                inv_freq = theta ** (-exponent)
                angle = pos * inv_freq
                val = np.cos(angle) if cosine else np.sin(angle)
                table[pos, head * head_dim + d] = val

    return table


def _add_per_head_rms_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_heads: int,
    head_dim: int,
    gamma: np.ndarray,
    eps_t: trt.ITensor,
    seq_len: int,
) -> trt.ITensor:
    """Per-head RMSNorm for sequence input [seq_len, num_heads * head_dim]."""
    total_dim = num_heads * head_dim

    # Reshape: [seq_len, total_dim] -> [seq_len * num_heads, head_dim]
    r1 = network.add_shuffle(inp)
    r1.reshape_dims = (seq_len * num_heads, head_dim)

    reshaped = r1.get_output(0)
    sq = network.add_elementwise(reshaped, reshaped, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    denom = network.add_elementwise(
        mean.get_output(0), eps_t, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        reshaped, recip.get_output(0), trt.ElementWiseOperation.PROD)

    # Apply gamma [num_heads, head_dim] -> tile for [seq_len * num_heads, head_dim]
    gamma_t = graph_ops.add_constant(network, (1, head_dim), gamma[0:1])  # same for all heads
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD)

    # Reshape back: [seq_len * num_heads, head_dim] -> [seq_len, total_dim]
    r2 = network.add_shuffle(scaled.get_output(0))
    r2.reshape_dims = (seq_len, total_dim)
    return r2.get_output(0)


def _apply_rope(
    network: trt.INetworkDefinition,
    x: trt.ITensor,
    cos_table: trt.ITensor,
    sin_table: trt.ITensor,
    rot_half: trt.ITensor,
) -> trt.ITensor:
    """Apply RoPE: x * cos + rotate_half(x) * sin."""
    x_rot = network.add_matrix_multiply(
        x, trt.MatrixOperation.NONE,
        rot_half, trt.MatrixOperation.NONE)
    x_cos = network.add_elementwise(
        x, cos_table, trt.ElementWiseOperation.PROD)
    x_sin = network.add_elementwise(
        x_rot.get_output(0), sin_table, trt.ElementWiseOperation.PROD)
    result = network.add_elementwise(
        x_cos.get_output(0), x_sin.get_output(0),
        trt.ElementWiseOperation.SUM)
    return result.get_output(0)


def _expand_kv_heads(
    network: trt.INetworkDefinition,
    kv: trt.ITensor,
    num_kv_heads: int,
    head_dim: int,
    repeat: int,
    seq_len: int,
) -> trt.ITensor:
    """Expand KV from [seq, kv_heads * head_dim] to [seq, num_heads * head_dim]."""
    # Reshape to [seq, kv_heads, head_dim]
    r1 = network.add_shuffle(kv)
    r1.reshape_dims = (seq_len, num_kv_heads, 1, head_dim)

    # Tile along new axis: [seq, kv_heads, repeat, head_dim]
    parts = [r1.get_output(0)] * repeat
    concat = network.add_concatenation(parts)
    concat.axis = 2

    # Reshape to [seq, num_heads * head_dim]
    r2 = network.add_shuffle(concat.get_output(0))
    r2.reshape_dims = (seq_len, num_kv_heads * repeat * head_dim)
    return r2.get_output(0)
