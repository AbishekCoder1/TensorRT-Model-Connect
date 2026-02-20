"""Shared T5 encoder engine builder.

Builds a TensorRT engine for a T5-style text encoder (UMT5, mT5, T5).
Reusable by: Wan2.1, FLUX, SD3, Hunyuan Video, CogVideoX.

Engine I/O:
    Input:  input_ids [1, max_seq_len] int32
    Output: text_embeddings [1, max_seq_len, d_model] float32

Single forward pass (no cache), like vision encoder.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt

from . import graph_ops
from . import graph_blocks

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict


def build_t5_encoder_engine(
    weights: WeightDict,
    *,
    d_model: int = 4096,
    num_heads: int = 64,
    d_kv: int = 64,
    d_ff: int = 10240,
    num_layers: int = 24,
    vocab_size: int = 250112,
    max_seq_len: int = 512,
    relative_attention_num_buckets: int = 32,
    relative_attention_max_distance: int = 128,
    eps: float = 1e-6,
    verbose: bool = False,
) -> bytes:
    """Build T5 encoder TRT engine plan.

    Args:
        weights: Weight dict with T5 encoder weights. Expected keys:
            - shared.weight: [vocab_size, d_model] embedding
            - encoder.block.{i}.layer.0.SelfAttention.q/k/v/o.weight
            - encoder.block.{i}.layer.0.layer_norm.weight
            - encoder.block.{i}.layer.1.DenseReluDense.wi_0/wi_1/wo.weight
            - encoder.block.{i}.layer.1.layer_norm.weight
            - encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight
            - encoder.final_layer_norm.weight
        d_model: Hidden dimension.
        num_heads: Number of attention heads.
        d_kv: Key/value dimension per head.
        d_ff: Feed-forward inner dimension.
        num_layers: Number of encoder layers.
        vocab_size: Vocabulary size.
        max_seq_len: Maximum sequence length.
        relative_attention_num_buckets: T5 relative position bias buckets.
        relative_attention_max_distance: Max distance for relative position.
        eps: RMSNorm epsilon.
        verbose: Enable TRT builder verbose logging.

    Returns:
        Serialized TRT engine plan bytes.
    """
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.clear_flag(trt.BuilderFlag.TF32)

    # --- Inputs ---
    input_ids = network.add_input(
        "input_ids", trt.int32, (1, max_seq_len))
    # Attention mask: [1, max_seq_len] float32, 0.0 for valid, -1e9 for padding
    attention_mask_input = network.add_input(
        "attention_mask", trt.float32, (1, max_seq_len))

    # --- Constants ---
    eps_t = graph_ops.add_constant(
        network, (1, 1), np.array([eps], dtype=np.float32))

    # Embedding table [vocab_size, d_model]
    embed_table = graph_ops.add_constant(
        network, (vocab_size, d_model), weights["shared.weight"])

    # --- Embedding lookup ---
    # Flatten input_ids to [max_seq_len] for gather
    flatten_ids = network.add_shuffle(input_ids)
    flatten_ids.reshape_dims = (max_seq_len,)

    gather = network.add_gather(embed_table, flatten_ids.get_output(0), 0)
    hidden = gather.get_output(0)  # [max_seq_len, d_model]

    # --- Relative position bias ---
    # Precompute bucket indices [max_seq_len, max_seq_len]
    bucket_indices = graph_ops.make_t5_relative_position_bias(
        num_heads, max_seq_len,
        num_buckets=relative_attention_num_buckets,
        max_distance=relative_attention_max_distance,
    )

    # Reshape attention mask: [1, max_seq_len] -> [1, 1, max_seq_len]
    attn_mask_3d = network.add_shuffle(attention_mask_input)
    attn_mask_3d.reshape_dims = (1, 1, max_seq_len)

    # UMT5: each layer has its own relative_attention_bias (unlike T5 which shares layer 0's)
    # Precompute per-layer bias tables as constants
    per_layer_bias = []
    for layer_idx in range(num_layers):
        bias_key = f"encoder.block.{layer_idx}.layer.0.SelfAttention.relative_attention_bias.weight"
        if bias_key in weights:
            rel_bias_weight = weights[bias_key]
        else:
            # Fallback: use layer 0's bias (vanilla T5 behavior)
            rel_bias_weight = weights[
                "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"]

        bias_table = np.zeros(
            (num_heads, max_seq_len, max_seq_len), dtype=np.float32)
        for q_pos in range(max_seq_len):
            for k_pos in range(max_seq_len):
                bucket = bucket_indices[q_pos, k_pos]
                for h in range(num_heads):
                    bias_table[h, q_pos, k_pos] = rel_bias_weight[bucket, h]

        rel_bias_const = graph_ops.add_constant(
            network, (num_heads, max_seq_len, max_seq_len), bias_table)

        # Combine position bias + attention mask
        position_bias_masked = network.add_elementwise(
            rel_bias_const, attn_mask_3d.get_output(0),
            trt.ElementWiseOperation.SUM)
        per_layer_bias.append(position_bias_masked.get_output(0))

    # --- Encoder layers ---
    attention_size = num_heads * d_kv

    for layer_idx in range(num_layers):
        prefix = f"encoder.block.{layer_idx}"

        # === Self-attention sub-layer ===
        # Pre-norm (RMSNorm)
        norm1_gamma = weights[f"{prefix}.layer.0.layer_norm.weight"]
        normed = graph_ops.add_rms_norm(
            network, hidden, d_model, norm1_gamma, eps_t)

        # Q, K, V projections
        # T5 uses no bias on Q/K/V/O projections
        w_q = weights[f"{prefix}.layer.0.SelfAttention.q.weight"]
        w_k = weights[f"{prefix}.layer.0.SelfAttention.k.weight"]
        w_v = weights[f"{prefix}.layer.0.SelfAttention.v.weight"]
        w_o = weights[f"{prefix}.layer.0.SelfAttention.o.weight"]

        # Self-attention with relative position bias
        q = graph_ops.add_matmul_rhs_constant(
            network, normed, d_model, attention_size, w_q)
        k = graph_ops.add_matmul_rhs_constant(
            network, normed, d_model, attention_size, w_k)
        v = graph_ops.add_matmul_rhs_constant(
            network, normed, d_model, attention_size, w_v)

        # Reshape to [num_heads, seq, d_kv]
        q_heads = network.add_shuffle(q)
        q_heads.reshape_dims = (max_seq_len, num_heads, d_kv)
        q_heads.second_transpose = trt.Permutation([1, 0, 2])

        k_heads = network.add_shuffle(k)
        k_heads.reshape_dims = (max_seq_len, num_heads, d_kv)
        k_heads.second_transpose = trt.Permutation([1, 0, 2])

        v_heads = network.add_shuffle(v)
        v_heads.reshape_dims = (max_seq_len, num_heads, d_kv)
        v_heads.second_transpose = trt.Permutation([1, 0, 2])

        # Attention: Q @ K^T / sqrt(d_kv) + rel_bias
        score = network.add_matrix_multiply(
            q_heads.get_output(0), trt.MatrixOperation.NONE,
            k_heads.get_output(0), trt.MatrixOperation.TRANSPOSE)

        # Note: T5 does NOT scale by 1/sqrt(d_kv) — the scale is baked into
        # the initialization. We add it for numerical stability matching HF.
        # Actually, HF T5 does NOT scale. We follow HF behavior: no scaling.

        # Add per-layer relative position bias + attention mask
        biased = network.add_elementwise(
            score.get_output(0), per_layer_bias[layer_idx],
            trt.ElementWiseOperation.SUM)

        softmax = network.add_softmax(biased.get_output(0))
        softmax.axes = 1 << 2

        context = network.add_matrix_multiply(
            softmax.get_output(0), trt.MatrixOperation.NONE,
            v_heads.get_output(0), trt.MatrixOperation.NONE)

        # Reshape back: [num_heads, seq, d_kv] -> [seq, attention_size]
        context_flat = network.add_shuffle(context.get_output(0))
        context_flat.first_transpose = trt.Permutation([1, 0, 2])
        context_flat.reshape_dims = (max_seq_len, attention_size)

        # Output projection
        attn_out = graph_ops.add_matmul_rhs_constant(
            network, context_flat.get_output(0),
            attention_size, d_model, w_o)

        # Residual
        hidden = network.add_elementwise(
            hidden, attn_out,
            trt.ElementWiseOperation.SUM).get_output(0)

        # === FFN sub-layer ===
        # Pre-norm (RMSNorm)
        norm2_gamma = weights[f"{prefix}.layer.1.layer_norm.weight"]
        ffn_normed = graph_ops.add_rms_norm(
            network, hidden, d_model, norm2_gamma, eps_t)

        # T5 gated GELU FFN: gelu(wi_0(x)) * wi_1(x), then wo
        w_fc1 = weights[f"{prefix}.layer.1.DenseReluDense.wi_0.weight"]
        w_fc1_gate = weights[f"{prefix}.layer.1.DenseReluDense.wi_1.weight"]
        w_fc2 = weights[f"{prefix}.layer.1.DenseReluDense.wo.weight"]

        fc1 = graph_ops.add_matmul_rhs_constant(
            network, ffn_normed, d_model, d_ff, w_fc1)
        fc1_gate = graph_ops.add_matmul_rhs_constant(
            network, ffn_normed, d_model, d_ff, w_fc1_gate)

        # GELU activation on fc1, multiply with gate
        activated = graph_ops.add_gelu_new(network, fc1)
        gated = network.add_elementwise(
            activated, fc1_gate,
            trt.ElementWiseOperation.PROD)

        # Output projection
        ffn_out = graph_ops.add_matmul_rhs_constant(
            network, gated.get_output(0), d_ff, d_model, w_fc2)

        # Residual
        hidden = network.add_elementwise(
            hidden, ffn_out,
            trt.ElementWiseOperation.SUM).get_output(0)

    # --- Final norm ---
    final_norm_gamma = weights["encoder.final_layer_norm.weight"]
    hidden = graph_ops.add_rms_norm(
        network, hidden, d_model, final_norm_gamma, eps_t)

    # --- Output ---
    # Reshape to [1, max_seq_len, d_model]
    out_reshape = network.add_shuffle(hidden)
    out_reshape.reshape_dims = (1, max_seq_len, d_model)
    out_tensor = out_reshape.get_output(0)
    out_tensor.name = "text_embeddings"
    network.mark_output(out_tensor)
    out_tensor.dtype = trt.float32

    # --- Build ---
    print(f"[t5-encoder] Building TRT engine "
          f"(d_model={d_model}, layers={num_layers}, seq={max_seq_len}) ...",
          file=sys.stderr)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for T5 encoder")
    return bytes(plan)


def load_t5_weights(
    model_dir: str,
    *,
    d_model: int = 4096,
    num_heads: int = 64,
    d_kv: int = 64,
    d_ff: int = 10240,
    num_layers: int = 24,
    vocab_size: int = 250112,
) -> WeightDict:
    """Load T5 encoder weights from a diffusers-format text_encoder directory.

    Expects: model_dir/model.safetensors (or sharded) with HF T5 weight keys.
    Returns WeightDict with transposed projections for TRT matmul.
    """
    from pathlib import Path
    from .checkpoint_mapper import WeightDict, _open_safetensors, _load_tensor, _has_tensor

    model_path = Path(model_dir)
    readers = _open_safetensors(model_path)

    weights = WeightDict()

    # Embedding
    embed = _load_tensor(readers, "shared.weight")
    weights["shared.weight"] = embed.astype(np.float32)

    # Relative attention bias — UMT5 has per-layer bias, T5 has only layer 0
    for i in range(num_layers):
        bias_key = f"encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight"
        if _has_tensor(readers, bias_key):
            weights[bias_key] = _load_tensor(readers, bias_key).astype(np.float32)

    for i in range(num_layers):
        prefix = f"encoder.block.{i}"

        # Self-attention weights (transpose for TRT matmul)
        for proj in ("q", "k", "v", "o"):
            key = f"{prefix}.layer.0.SelfAttention.{proj}.weight"
            w = _load_tensor(readers, key)
            # HF shape: [out, in] -> transpose to [in, out]
            weights[key] = np.ascontiguousarray(w.T, dtype=np.float32)

        # Self-attention layer norm
        norm_key = f"{prefix}.layer.0.layer_norm.weight"
        weights[norm_key] = _load_tensor(readers, norm_key).astype(np.float32)

        # FFN weights (transpose for TRT matmul)
        for proj in ("wi_0", "wi_1", "wo"):
            key = f"{prefix}.layer.1.DenseReluDense.{proj}.weight"
            w = _load_tensor(readers, key)
            weights[key] = np.ascontiguousarray(w.T, dtype=np.float32)

        # FFN layer norm
        norm_key = f"{prefix}.layer.1.layer_norm.weight"
        weights[norm_key] = _load_tensor(readers, norm_key).astype(np.float32)

    # Final layer norm
    weights["encoder.final_layer_norm.weight"] = _load_tensor(
        readers, "encoder.final_layer_norm.weight").astype(np.float32)

    return weights
