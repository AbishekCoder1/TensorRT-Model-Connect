"""Mistral 3 text encoder engine builder for FLUX.2-dev.

Builds a TensorRT engine for a Mistral-3/LLaMA-style decoder architecture
run as an encoder-only model (single forward pass, no KV cache, no
autoregressive generation).  Multi-layer hidden state extraction produces
a concatenated output suitable for conditioning diffusion models.

Engine I/O:
    Inputs:
        input_ids [1, max_seq_len] int32
        attention_mask [1, max_seq_len] float32 (0.0 for valid, -1e9 for padding)
    Outputs:
        text_embeddings [1, max_seq_len, concat_dim] float32
            where concat_dim = len(extract_layers) * hidden_size
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt

from . import graph_ops

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict


def load_mistral_encoder_weights(
    model_dir: str,
    *,
    hidden_size: int = 5120,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    intermediate_size: int = 32768,
    num_layers: int = 40,
    vocab_size: int = 131072,
) -> WeightDict:
    """Load Mistral 3 encoder weights from a diffusers-format text_encoder directory.

    Expects: model_dir/model.safetensors (or sharded) with HF Mistral weight keys.
    Returns WeightDict with transposed projections for TRT matmul.

    Auto-detects the weight prefix: either ``model.`` (standalone Mistral) or
    ``language_model.model.`` (Mistral3ForConditionalGeneration multimodal wrapper
    as used in FLUX.2-dev).  Weights are stored under the canonical ``model.``
    prefix regardless of the source prefix.
    """
    from pathlib import Path
    from .checkpoint_mapper import (
        WeightDict, _open_safetensors, _load_tensor, _has_tensor)

    model_path = Path(model_dir)
    readers = _open_safetensors(model_path)

    # Auto-detect weight prefix
    if _has_tensor(readers, "model.embed_tokens.weight"):
        src_prefix = "model."
    elif _has_tensor(readers, "language_model.model.embed_tokens.weight"):
        src_prefix = "language_model.model."
    else:
        raise KeyError(
            "Cannot find embedding weights under 'model.' or "
            "'language_model.model.' prefix")

    dst_prefix = "model."

    print(f"[mistral-encoder] Weight prefix: {src_prefix!r}", file=sys.stderr)

    def _src(canonical: str) -> str:
        """Map canonical 'model.X' key to the actual source key."""
        if canonical.startswith(dst_prefix):
            return src_prefix + canonical[len(dst_prefix):]
        return canonical

    weights = WeightDict()

    # Embedding table
    embed = _load_tensor(readers, _src("model.embed_tokens.weight"))
    weights["model.embed_tokens.weight"] = embed.astype(np.float32)

    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    for i in range(num_layers):
        prefix = f"model.layers.{i}"

        # RMSNorm weights (no transpose needed — 1-D)
        weights[f"{prefix}.input_layernorm.weight"] = _load_tensor(
            readers, _src(f"{prefix}.input_layernorm.weight")).astype(np.float32)
        weights[f"{prefix}.post_attention_layernorm.weight"] = _load_tensor(
            readers, _src(f"{prefix}.post_attention_layernorm.weight")).astype(np.float32)

        # Self-attention projections — transpose [out, in] -> [in, out]
        for proj, out_size in [
            ("q_proj", q_size),
            ("k_proj", kv_size),
            ("v_proj", kv_size),
        ]:
            key = f"{prefix}.self_attn.{proj}.weight"
            w = _load_tensor(readers, _src(key))
            weights[key] = np.ascontiguousarray(w.T, dtype=np.float32)

        o_key = f"{prefix}.self_attn.o_proj.weight"
        w_o = _load_tensor(readers, _src(o_key))
        weights[o_key] = np.ascontiguousarray(w_o.T, dtype=np.float32)

        # MLP projections — transpose [out, in] -> [in, out]
        for proj in ("gate_proj", "up_proj", "down_proj"):
            key = f"{prefix}.mlp.{proj}.weight"
            w = _load_tensor(readers, _src(key))
            weights[key] = np.ascontiguousarray(w.T, dtype=np.float32)

    # Final RMSNorm
    weights["model.norm.weight"] = _load_tensor(
        readers, _src("model.norm.weight")).astype(np.float32)

    return weights


def build_mistral_encoder_engine(
    weights: WeightDict,
    *,
    hidden_size: int = 5120,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    intermediate_size: int = 32768,
    num_layers: int = 40,
    vocab_size: int = 131072,
    max_seq_len: int = 512,
    extract_layers: list[int] | tuple[int, ...] = (10, 20, 30),
    eps: float = 1e-5,
    rope_theta: float = 1000000000.0,
    verbose: bool = False,
) -> bytes:
    """Build Mistral 3 encoder TRT engine plan.

    Runs all layers as a single forward pass with bidirectional attention
    (no causal mask).  Extracts hidden states from specified layers and
    concatenates them along the feature dimension.

    Args:
        weights: Weight dict from load_mistral_encoder_weights.
        hidden_size: Model hidden dimension.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value attention heads (GQA).
        head_dim: Dimension per attention head.
        intermediate_size: SwiGLU FFN inner dimension.
        num_layers: Number of transformer layers.
        vocab_size: Vocabulary size.
        max_seq_len: Maximum input sequence length.
        extract_layers: Layer indices (0-based) whose hidden states are
            extracted and concatenated for the output.
        eps: RMSNorm epsilon.
        verbose: Enable TRT builder verbose logging.

    Returns:
        Serialized TRT engine plan bytes.
    """
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    gqa_ratio = num_heads // num_kv_heads
    concat_dim = len(extract_layers) * hidden_size

    # Precompute RoPE cos/sin tables [max_seq_len, head_dim]
    rope_cos_np, rope_sin_np = _make_rope_table(
        head_dim, max_seq_len, theta=rope_theta)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 128 << 30)
    config.clear_flag(trt.BuilderFlag.TF32)

    # --- Inputs ---
    input_ids = network.add_input(
        "input_ids", trt.int32, (1, max_seq_len))
    attention_mask_input = network.add_input(
        "attention_mask", trt.float32, (1, max_seq_len))

    # --- Constants ---
    eps_t = graph_ops.add_constant(
        network, (1, 1), np.array([eps], dtype=np.float32))
    attn_scale_val = 1.0 / np.sqrt(head_dim)
    attn_scale = graph_ops.add_constant(
        network, (1, 1, 1), np.array([attn_scale_val], dtype=np.float32))

    # Embedding table [vocab_size, hidden_size]
    embed_table = graph_ops.add_constant(
        network, (vocab_size, hidden_size),
        weights["model.embed_tokens.weight"])

    # --- Embedding lookup ---
    flatten_ids = network.add_shuffle(input_ids)
    flatten_ids.reshape_dims = (max_seq_len,)

    gather = network.add_gather(embed_table, flatten_ids.get_output(0), 0)
    hidden = gather.get_output(0)  # [max_seq_len, hidden_size]

    # --- Attention mask: causal + padding ---
    # Build causal mask: upper triangular with -1e9 above diagonal
    causal_mask_np = np.zeros((max_seq_len, max_seq_len), dtype=np.float32)
    for i in range(max_seq_len):
        for j in range(i + 1, max_seq_len):
            causal_mask_np[i, j] = -1e9
    causal_mask = graph_ops.add_constant(
        network, (1, max_seq_len, max_seq_len), causal_mask_np)

    # Combine with padding mask: reshape [1, max_seq_len] -> [1, 1, max_seq_len]
    attn_mask_3d = network.add_shuffle(attention_mask_input)
    attn_mask_3d.reshape_dims = (1, 1, max_seq_len)

    # Combined mask = causal_mask + padding_mask (both add -1e9 for masked positions)
    combined_mask = network.add_elementwise(
        causal_mask, attn_mask_3d.get_output(0),
        trt.ElementWiseOperation.SUM).get_output(0)

    # --- RoPE constants ---
    # Tile cos/sin from [seq, head_dim] to [seq, q_size] for Q
    # and [seq, kv_size] for K (different num_heads)
    rope_cos_const = graph_ops.add_constant(
        network, (max_seq_len, head_dim), rope_cos_np)
    rope_sin_const = graph_ops.add_constant(
        network, (max_seq_len, head_dim), rope_sin_np)

    # Tile for Q heads: [seq, head_dim] -> [seq, q_size] via repeat
    q_rope_cos_parts = [rope_cos_const] * num_heads
    q_rope_cos = network.add_concatenation(q_rope_cos_parts)
    q_rope_cos.axis = 1
    q_rope_sin_parts = [rope_sin_const] * num_heads
    q_rope_sin = network.add_concatenation(q_rope_sin_parts)
    q_rope_sin.axis = 1

    # Tile for K heads: [seq, head_dim] -> [seq, kv_size] via repeat
    k_rope_cos_parts = [rope_cos_const] * num_kv_heads
    k_rope_cos = network.add_concatenation(k_rope_cos_parts)
    k_rope_cos.axis = 1
    k_rope_sin_parts = [rope_sin_const] * num_kv_heads
    k_rope_sin = network.add_concatenation(k_rope_sin_parts)
    k_rope_sin.axis = 1

    # Rotate-half matrices for RoPE application
    q_rot_half_np = graph_ops.make_rotate_half_matrix(
        q_size, num_heads, interleaved=True)
    q_rot_half = graph_ops.add_constant(
        network, (q_size, q_size), q_rot_half_np)
    k_rot_half_np = graph_ops.make_rotate_half_matrix(
        kv_size, num_kv_heads, interleaved=True)
    k_rot_half = graph_ops.add_constant(
        network, (kv_size, kv_size), k_rot_half_np)

    # Collected hidden states for extraction
    extracted = []

    # --- Encoder layers ---
    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}"

        # === Self-attention sub-layer ===

        # Pre-norm (RMSNorm)
        norm1_gamma = weights[f"{prefix}.input_layernorm.weight"]
        normed = graph_ops.add_rms_norm(
            network, hidden, hidden_size, norm1_gamma, eps_t)

        # Q/K/V projections
        w_q = weights[f"{prefix}.self_attn.q_proj.weight"]
        w_k = weights[f"{prefix}.self_attn.k_proj.weight"]
        w_v = weights[f"{prefix}.self_attn.v_proj.weight"]
        w_o = weights[f"{prefix}.self_attn.o_proj.weight"]

        q = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, q_size, w_q)
        k = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, kv_size, w_k)
        v = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, kv_size, w_v)

        # Apply RoPE to Q: q = q * cos + rotate_half(q) * sin
        q_rot = network.add_matrix_multiply(
            q, trt.MatrixOperation.NONE,
            q_rot_half, trt.MatrixOperation.NONE)
        q_cos = network.add_elementwise(
            q, q_rope_cos.get_output(0), trt.ElementWiseOperation.PROD)
        q_sin = network.add_elementwise(
            q_rot.get_output(0), q_rope_sin.get_output(0),
            trt.ElementWiseOperation.PROD)
        q = network.add_elementwise(
            q_cos.get_output(0), q_sin.get_output(0),
            trt.ElementWiseOperation.SUM).get_output(0)

        # Apply RoPE to K
        k_rot = network.add_matrix_multiply(
            k, trt.MatrixOperation.NONE,
            k_rot_half, trt.MatrixOperation.NONE)
        k_cos = network.add_elementwise(
            k, k_rope_cos.get_output(0), trt.ElementWiseOperation.PROD)
        k_sin = network.add_elementwise(
            k_rot.get_output(0), k_rope_sin.get_output(0),
            trt.ElementWiseOperation.PROD)
        k = network.add_elementwise(
            k_cos.get_output(0), k_sin.get_output(0),
            trt.ElementWiseOperation.SUM).get_output(0)

        # Reshape Q to [num_heads, max_seq_len, head_dim]
        q_heads = network.add_shuffle(q)
        q_heads.reshape_dims = (max_seq_len, num_heads, head_dim)
        q_heads.second_transpose = trt.Permutation([1, 0, 2])

        # Reshape K to [num_kv_heads, max_seq_len, head_dim]
        k_heads = network.add_shuffle(k)
        k_heads.reshape_dims = (max_seq_len, num_kv_heads, head_dim)
        k_heads.second_transpose = trt.Permutation([1, 0, 2])

        # Reshape V to [num_kv_heads, max_seq_len, head_dim]
        v_heads = network.add_shuffle(v)
        v_heads.reshape_dims = (max_seq_len, num_kv_heads, head_dim)
        v_heads.second_transpose = trt.Permutation([1, 0, 2])

        # GQA: tile K and V from [num_kv_heads, seq, head_dim] to
        # [num_heads, seq, head_dim] by repeating each KV head gqa_ratio times
        if gqa_ratio > 1:
            k_expanded = _tile_kv_heads(
                network, k_heads.get_output(0),
                num_kv_heads, gqa_ratio, max_seq_len, head_dim)
            v_expanded = _tile_kv_heads(
                network, v_heads.get_output(0),
                num_kv_heads, gqa_ratio, max_seq_len, head_dim)
        else:
            k_expanded = k_heads.get_output(0)
            v_expanded = v_heads.get_output(0)

        # Attention: Q @ K^T / sqrt(head_dim)
        score = network.add_matrix_multiply(
            q_heads.get_output(0), trt.MatrixOperation.NONE,
            k_expanded, trt.MatrixOperation.TRANSPOSE)

        # Scale by 1/sqrt(head_dim)
        scaled = network.add_elementwise(
            score.get_output(0), attn_scale,
            trt.ElementWiseOperation.PROD)

        # Add combined causal + padding mask
        biased = network.add_elementwise(
            scaled.get_output(0), combined_mask,
            trt.ElementWiseOperation.SUM)

        # Softmax over key dimension
        softmax = network.add_softmax(biased.get_output(0))
        softmax.axes = 1 << 2  # last dim

        # Context: [num_heads, max_seq_len, head_dim]
        context = network.add_matrix_multiply(
            softmax.get_output(0), trt.MatrixOperation.NONE,
            v_expanded, trt.MatrixOperation.NONE)

        # Reshape back to [max_seq_len, q_size]
        context_flat = network.add_shuffle(context.get_output(0))
        context_flat.first_transpose = trt.Permutation([1, 0, 2])
        context_flat.reshape_dims = (max_seq_len, q_size)

        # O projection
        attn_out = graph_ops.add_matmul_rhs_constant(
            network, context_flat.get_output(0), q_size, hidden_size, w_o)

        # Residual
        hidden = network.add_elementwise(
            hidden, attn_out,
            trt.ElementWiseOperation.SUM).get_output(0)

        # === SwiGLU FFN sub-layer ===

        # Pre-norm (RMSNorm)
        norm2_gamma = weights[f"{prefix}.post_attention_layernorm.weight"]
        ffn_normed = graph_ops.add_rms_norm(
            network, hidden, hidden_size, norm2_gamma, eps_t)

        # SwiGLU: silu(gate_proj(x)) * up_proj(x), then down_proj
        w_gate = weights[f"{prefix}.mlp.gate_proj.weight"]
        w_up = weights[f"{prefix}.mlp.up_proj.weight"]
        w_down = weights[f"{prefix}.mlp.down_proj.weight"]

        gate = graph_ops.add_matmul_rhs_constant(
            network, ffn_normed, hidden_size, intermediate_size, w_gate)
        up = graph_ops.add_matmul_rhs_constant(
            network, ffn_normed, hidden_size, intermediate_size, w_up)

        # SiLU activation on gate
        gate_activated = graph_ops.add_activation(network, gate, "silu")

        # gate * up
        gated = network.add_elementwise(
            gate_activated, up,
            trt.ElementWiseOperation.PROD)

        # down_proj
        ffn_out = graph_ops.add_matmul_rhs_constant(
            network, gated.get_output(0),
            intermediate_size, hidden_size, w_down)

        # Residual
        hidden = network.add_elementwise(
            hidden, ffn_out,
            trt.ElementWiseOperation.SUM).get_output(0)

        # Extract hidden state if this layer is in the extraction list
        if layer_idx in extract_layers:
            extracted.append(hidden)

    # --- Final RMSNorm ---
    final_norm_gamma = weights["model.norm.weight"]
    hidden = graph_ops.add_rms_norm(
        network, hidden, hidden_size, final_norm_gamma, eps_t)

    # --- Multi-layer concatenation ---
    # Concatenate extracted hidden states along the feature dimension:
    # each is [max_seq_len, hidden_size] -> result is [max_seq_len, concat_dim]
    if len(extracted) == 0:
        raise ValueError(
            f"No layers matched extract_layers={extract_layers} "
            f"(num_layers={num_layers})")

    if len(extracted) == 1:
        concat_out = extracted[0]
    else:
        concat_layer = network.add_concatenation(extracted)
        concat_layer.axis = 1  # feature dimension
        concat_out = concat_layer.get_output(0)

    # --- Output ---
    # Reshape to [1, max_seq_len, concat_dim]
    out_reshape = network.add_shuffle(concat_out)
    out_reshape.reshape_dims = (1, max_seq_len, concat_dim)
    out_tensor = out_reshape.get_output(0)
    out_tensor.name = "text_embeddings"
    network.mark_output(out_tensor)
    out_tensor.dtype = trt.float32

    # --- Build ---
    print(
        f"[mistral-encoder] Building TRT engine "
        f"(hidden={hidden_size}, layers={num_layers}, "
        f"heads={num_heads}/{num_kv_heads}, head_dim={head_dim}, "
        f"seq={max_seq_len}, extract={list(extract_layers)}, "
        f"concat_dim={concat_dim}) ...",
        file=sys.stderr)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(
            "TRT engine serialization failed for Mistral encoder")
    return bytes(plan)


def _make_rope_table(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute RoPE cos/sin tables [max_seq_len, head_dim].

    Uses interleaved format: cos[pos, 2i] = cos[pos, 2i+1] = cos(pos * freq_i).
    This matches HuggingFace's repeat_interleave RoPE convention.
    """
    half = head_dim // 2
    freqs = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
    positions = np.arange(max_seq_len, dtype=np.float64)
    angles = np.outer(positions, freqs)  # [seq, half]
    cos_vals = np.cos(angles)  # [seq, half]
    sin_vals = np.sin(angles)  # [seq, half]
    # Interleave: [seq, half] -> [seq, head_dim] via repeat_interleave
    cos_out = np.repeat(cos_vals, 2, axis=1).astype(np.float32)  # [seq, head_dim]
    sin_out = np.repeat(sin_vals, 2, axis=1).astype(np.float32)  # [seq, head_dim]
    return cos_out, sin_out


def _tile_kv_heads(
    network: trt.INetworkDefinition,
    kv_tensor: trt.ITensor,
    num_kv_heads: int,
    gqa_ratio: int,
    seq_len: int,
    head_dim: int,
) -> trt.ITensor:
    """Tile KV heads from [num_kv_heads, seq, head_dim] to [num_heads, seq, head_dim].

    Each KV head is repeated gqa_ratio times via concatenation so that the
    result has num_kv_heads * gqa_ratio = num_heads total heads.
    """
    # Slice each KV head and repeat it gqa_ratio times
    slices = []
    for kv_idx in range(num_kv_heads):
        head_slice = network.add_slice(
            kv_tensor,
            start=(kv_idx, 0, 0),
            shape=(1, seq_len, head_dim),
            stride=(1, 1, 1),
        )
        for _ in range(gqa_ratio):
            slices.append(head_slice.get_output(0))

    concat = network.add_concatenation(slices)
    concat.axis = 0  # head dimension
    return concat.get_output(0)
