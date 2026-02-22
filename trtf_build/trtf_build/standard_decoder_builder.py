"""Standard decoder engine builder (parameterized).

Builds a TensorRT engine for a decoder-only transformer. Supports multiple
norm, MLP, and position embedding strategies via parameters:

  norm_type:     "rmsnorm" | "layernorm"
  mlp_type:      "swiglu"  | "gelu_fc"
  position_type: "rope"    | "learned" | "alibi"
  activation:    "silu" | "gelu_new" | "gelu" | "relu" | "relu2" (used by gelu_fc MLP)

Tensor names MUST match what the C++ runtime expects:
  Inputs:  token_id, position_id, attention_mask, cache_k_0..N, cache_v_0..N
  Outputs: logits, present_k_0..N, present_v_0..N
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt

from . import graph_ops
from . import graph_blocks
from .config import ModelConfig

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict


def _mark_debug_output(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    name: str,
) -> None:
    """Mark a tensor as a network output for debug inspection."""
    # Use an identity layer to avoid aliasing issues with existing outputs.
    identity = network.add_identity(tensor)
    out = identity.get_output(0)
    out.name = name
    network.mark_output(out)
    out.dtype = trt.float32


def build_standard_decoder_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    norm_type: str = "rmsnorm",
    mlp_type: str = "swiglu",
    position_type: str = "rope",
    activation: str = "silu",
    partial_rotary_factor: float = 1.0,
    interleaved_rope: bool = False,
    parallel_residual: bool = False,
    scale_attn_weights: bool = True,
    embed_input: bool = False,
    verbose: bool = False,
    debug_layer_outputs: bool = False,
) -> bytes:
    """Build a TRT engine plan (serialized bytes) for a standard decoder.

    Args:
        config: Model architecture from config.json.
        weights: Loaded weight dict from checkpoint_mapper.
        max_cache_length: KV cache length (engine is compiled for this value).
        norm_type: "rmsnorm" or "layernorm".
        mlp_type: "swiglu" (3 projections: gate/up/down) or
                  "gelu_fc" (2 projections: fc1/fc2 with activation).
        position_type: "rope" (rotary), "learned" (absolute position embeddings),
            or "alibi" (attention with linear biases, no position embeddings).
        activation: Activation function for gelu_fc MLP ("gelu_new", "gelu", "relu", "relu2").
        partial_rotary_factor: Fraction of head dims that get RoPE (default 1.0).
        interleaved_rope: If True, use interleaved RoPE (CodeGen/GPT-J) where
            adjacent dims (d, d+1) share frequencies. Default False uses
            rotated-half (LLaMA/Qwen) where (d, d+half) share frequencies.
        scale_attn_weights: Whether to scale attention scores by 1/sqrt(head_dim).
            Most models use this (True, default). GPT-Neo does NOT scale (False).
        embed_input: If True, add input_embed [1, hidden] and use_input_embed [1]
            engine inputs. When use_input_embed==1, the decoder uses input_embed
            directly instead of the embedding lookup. Used for VL models where
            the vision encoder provides fused embeddings during prefill.
        verbose: Print TRT builder logs.
        debug_layer_outputs: If True, mark per-layer hidden states as network
            outputs for diff testing.

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

    # Optional VL inputs: when embed_input=True, the decoder can accept
    # a pre-computed embedding vector instead of a token ID.
    input_embed_tensor = None
    use_input_embed_tensor = None
    if embed_input:
        input_embed_tensor = network.add_input(
            "input_embed", trt.float32, (1, hidden))
        use_input_embed_tensor = network.add_input(
            "use_input_embed", trt.float32, (1,))

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

    # RoPE tables (only needed when position_type == "rope")
    cos_tensor = None
    sin_tensor = None
    rotate_half_tensor = None
    position_embed_table = None
    alibi_slopes_tensor = None
    alibi_indices_tensor = None

    if position_type == "rope":
        cos_table_np = graph_ops.make_rope_table(
            attention_window, attention_size, num_heads, config.rope_theta, True,
            partial_rotary_factor, interleaved=interleaved_rope)
        sin_table_np = graph_ops.make_rope_table(
            attention_window, attention_size, num_heads, config.rope_theta, False,
            partial_rotary_factor, interleaved=interleaved_rope)
        rotate_half_np = graph_ops.make_rotate_half_matrix(
            attention_size, num_heads, partial_rotary_factor,
            interleaved=interleaved_rope)

        cos_tensor = graph_ops.add_constant(
            network, (attention_window, attention_size), cos_table_np)
        sin_tensor = graph_ops.add_constant(
            network, (attention_window, attention_size), sin_table_np)
        rotate_half_tensor = graph_ops.add_constant(
            network, (attention_size, attention_size), rotate_half_np)
    elif position_type == "learned":
        pos_embed_np = weights["position_embedding"]
        position_embed_table = graph_ops.add_constant(
            network, pos_embed_np.shape, pos_embed_np)
    elif position_type == "alibi":
        alibi_slopes_np = graph_ops.compute_alibi_slopes(num_heads)
        alibi_slopes_tensor = graph_ops.add_constant(
            network, (num_heads, 1, 1),
            alibi_slopes_np.reshape(num_heads, 1, 1))
        # Cache position indices [0, 1, ..., max_cache_length-1].
        # The current token's position (position_id) is appended at runtime.
        alibi_indices_tensor = graph_ops.add_constant(
            network, (max_cache_length,),
            np.arange(max_cache_length, dtype=np.float32))

    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=np.float32))
    attn_scale = (1.0 / np.sqrt(max(head_dim, 1))) if scale_attn_weights else 1.0
    attn_scale_tensor = graph_ops.add_constant(
        network, (1, 1, 1), np.array([attn_scale], dtype=np.float32))

    # ---------------------------------------------------------------
    # Embedding lookup (with optional embed_input override for VL)
    # ---------------------------------------------------------------
    gather = network.add_gather(embedding_table, token_id, 0)
    token_embed = gather.get_output(0)

    if embed_input and input_embed_tensor is not None and use_input_embed_tensor is not None:
        # Conditional embedding: (1 - flag) * token_embed + flag * input_embed
        # use_input_embed is [1] scalar, broadcast to [1, hidden]
        flag_broadcast = network.add_shuffle(use_input_embed_tensor)
        flag_broadcast.reshape_dims = (1, 1)
        one_const = graph_ops.add_constant(
            network, (1, 1), np.array([1.0], dtype=np.float32))
        inv_flag = network.add_elementwise(
            one_const, flag_broadcast.get_output(0),
            trt.ElementWiseOperation.SUB)
        # (1 - flag) * token_embed
        tok_part = network.add_elementwise(
            inv_flag.get_output(0), token_embed,
            trt.ElementWiseOperation.PROD)
        # flag * input_embed
        embed_part = network.add_elementwise(
            flag_broadcast.get_output(0), input_embed_tensor,
            trt.ElementWiseOperation.PROD)
        # sum
        hidden_state_sum = network.add_elementwise(
            tok_part.get_output(0), embed_part.get_output(0),
            trt.ElementWiseOperation.SUM)
        hidden_state = hidden_state_sum.get_output(0)
    else:
        hidden_state = token_embed

    # Add learned position embedding if applicable
    if position_type == "learned" and position_embed_table is not None:
        pos_gather = network.add_gather(position_embed_table, position_id, 0)
        pos_add = network.add_elementwise(
            hidden_state, pos_gather.get_output(0),
            trt.ElementWiseOperation.SUM)
        hidden_state = pos_add.get_output(0)

    # Optional embedding LayerNorm (e.g. BLOOM)
    embed_norm = weights.get("embedding_norm")
    if embed_norm is not None:
        embed_norm_beta = weights.get("embedding_norm_beta")
        if embed_norm_beta is None:
            embed_norm_beta = np.zeros(hidden, dtype=np.float32)
        hidden_state = graph_ops.add_layer_norm(
            network, hidden_state, hidden, embed_norm, embed_norm_beta,
            eps_tensor)

    if debug_layer_outputs:
        _mark_debug_output(network, hidden_state, "debug_embed")

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
            norm_type=norm_type,
            mlp_type=mlp_type,
            position_type=position_type,
            activation=activation,
            parallel_residual=parallel_residual,
            alibi_slopes_tensor=alibi_slopes_tensor,
            alibi_indices_tensor=alibi_indices_tensor,
        )

        hidden_state = result["hidden"]
        present_k_outputs.append(result["present_k"])
        present_v_outputs.append(result["present_v"])

        if debug_layer_outputs:
            _mark_debug_output(network, result["post_attn"], f"debug_post_attn_{layer_idx}")
            _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")

    # ---------------------------------------------------------------
    # Final norm
    # ---------------------------------------------------------------
    final_norm = weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = _apply_norm(
            network, hidden_state, hidden, final_norm,
            weights.get("final_norm_beta"), eps_tensor, norm_type)

    # ---------------------------------------------------------------
    # LM head (logits)
    # ---------------------------------------------------------------
    # Output vocab may differ from input vocab (e.g. Bark semantic: 129600 in, 10048 out).
    # Derive from w_out shape if available.
    out_vocab = weights["w_out"].shape[1] if isinstance(weights["w_out"], np.ndarray) else vocab
    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden, out_vocab, weights["w_out"])
    # LM head bias (if present, e.g. CodeGen) or zero bias for C++ parity
    lm_bias = weights.get("lm_head_bias")
    if lm_bias is not None:
        logits = graph_ops.add_bias_sum(network, logits, out_vocab, lm_bias)
    else:
        b_out = np.zeros(out_vocab, dtype=np.float32)
        logits = graph_ops.add_bias_sum(network, logits, out_vocab, b_out)

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


def _apply_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray | None,
    eps_tensor: trt.ITensor,
    norm_type: str,
) -> trt.ITensor:
    """Dispatch to RMSNorm or LayerNorm based on norm_type."""
    return graph_blocks.apply_norm(
        network, inp, hidden_size, gamma, beta, eps_tensor, norm_type)


def _add_decoder_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    position_id: trt.ITensor,
    cos_tensor: trt.ITensor | None,
    sin_tensor: trt.ITensor | None,
    rotate_half_tensor: trt.ITensor | None,
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
    norm_type: str = "rmsnorm",
    mlp_type: str = "swiglu",
    position_type: str = "rope",
    activation: str = "silu",
    parallel_residual: bool = False,
    alibi_slopes_tensor: trt.ITensor | None = None,
    alibi_indices_tensor: trt.ITensor | None = None,
) -> dict[str, trt.ITensor]:
    """Add one standard decoder layer block. Returns hidden, present_k, present_v."""

    # Attention block (pre-norm -> QKV -> RoPE -> cache -> attn -> out proj)
    attn = graph_blocks.add_attention_block(
        network, hidden, cache_k, cache_v, attention_mask, position_id,
        weights=weights, prefix=prefix,
        hidden_size=hidden_size, attention_size=attention_size,
        num_heads=num_heads, head_dim=head_dim,
        max_cache_length=max_cache_length,
        cos_tensor=cos_tensor, sin_tensor=sin_tensor,
        rotate_half_tensor=rotate_half_tensor,
        attn_scale_tensor=attn_scale_tensor, eps_tensor=eps_tensor,
        norm_type=norm_type, position_type=position_type,
        alibi_slopes_tensor=alibi_slopes_tensor,
        alibi_indices_tensor=alibi_indices_tensor,
    )
    attn_out = attn["attn_out"]
    present_k = attn["present_k"]
    present_v = attn["present_v"]

    # --- Parallel vs sequential residual ---
    if parallel_residual:
        post_attn_norm_w = weights.get(f"{prefix}.post_attn_norm")
        if post_attn_norm_w is not None:
            norm2 = _apply_norm(
                network, hidden, hidden_size,
                post_attn_norm_w,
                weights.get(f"{prefix}.post_attn_norm_beta"),
                eps_tensor, norm_type)
        else:
            norm2 = attn["normed"]
    else:
        residual1 = network.add_elementwise(
            hidden, attn_out, trt.ElementWiseOperation.SUM)
        norm2 = _apply_norm(
            network, residual1.get_output(0), hidden_size,
            weights[f"{prefix}.post_attn_norm"],
            weights.get(f"{prefix}.post_attn_norm_beta"),
            eps_tensor, norm_type)

    # MLP
    if mlp_type == "gelu_fc":
        mlp_out = graph_blocks.add_gelu_fc_mlp(
            network, norm2, weights=weights, prefix=prefix,
            hidden_size=hidden_size, mlp_size=mlp_size,
            activation=activation)
    else:
        mlp_out = graph_blocks.add_swiglu_mlp(
            network, norm2, weights=weights, prefix=prefix,
            hidden_size=hidden_size, mlp_size=mlp_size)

    # Final residual connection
    if parallel_residual:
        sum_attn = network.add_elementwise(
            hidden, attn_out, trt.ElementWiseOperation.SUM)
        residual2 = network.add_elementwise(
            sum_attn.get_output(0), mlp_out, trt.ElementWiseOperation.SUM)
        post_attn_tensor = sum_attn.get_output(0)
    else:
        residual2 = network.add_elementwise(
            residual1.get_output(0), mlp_out, trt.ElementWiseOperation.SUM)
        post_attn_tensor = residual1.get_output(0)

    return {
        "hidden": residual2.get_output(0),
        "post_attn": post_attn_tensor,
        "present_k": present_k,
        "present_v": present_v,
    }
