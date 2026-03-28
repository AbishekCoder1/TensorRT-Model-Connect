"""ModernBERT family plugin -- encoder-only transformer with modern design.

ModernBERT differs significantly from classic BERT:
  - PRE-norm with LayerNorm (no bias) -- NOT RMSNorm despite weight naming
  - Fused QKV projection (Wqkv) -- split into Q/K/V
  - GeGLU MLP (fused Wi gate+up, Wo down) -- split Wi into gate/up
  - RoPE position encoding with per-layer theta (full_attention=160000, sliding=10000)
  - No token type embeddings
  - No attention bias, no MLP bias
  - Layer 0 has no attn_norm (identity)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ..config import ModelConfig
from ..checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
)
from .. import graph_ops

try:
    import tensorrt as trt
except ImportError:
    trt = None  # type: ignore[assignment]


def _add_layernorm_no_bias(network, inp, hidden_size, gamma, eps):
    """LayerNorm without bias: gamma * (x - mean) / sqrt(var + eps).

    ModernBERT uses nn.LayerNorm(bias=False) which still mean-centers,
    unlike RMSNorm which does not.
    """
    eps_t = graph_ops.add_constant(network, (1, 1), np.array([eps], dtype=np.float32))
    gamma_t = graph_ops.add_constant(network, (1, hidden_size), gamma)

    mean = network.add_reduce(inp, trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    centered = network.add_elementwise(
        inp, mean.get_output(0), trt.ElementWiseOperation.SUB)
    sq = network.add_elementwise(
        centered.get_output(0), centered.get_output(0), trt.ElementWiseOperation.PROD)
    var = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    denom = network.add_elementwise(
        var.get_output(0), eps_t, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        centered.get_output(0), recip.get_output(0), trt.ElementWiseOperation.PROD)
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD)
    return scaled.get_output(0)


class ModernbertPlugin:
    name = "modernbert"
    runtime_strategy = "encoder_only"

    def matches(self, model_type: str) -> bool:
        return model_type.lower().startswith("modernbert")

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        num_layers = config.num_hidden_layers
        intermediate = config.intermediate_size

        weights = WeightDict()

        # Word embedding
        embedding = _load_tensor(readers, "model.embeddings.tok_embeddings.weight")
        assert embedding.shape == (config.vocab_size, hidden)
        weights["embedding"] = embedding.astype(np.float32)

        # Embedding LayerNorm (no bias)
        weights["embed_norm"] = _load_tensor(
            readers, "model.embeddings.norm.weight").astype(np.float32)

        # Final LayerNorm
        weights["final_norm"] = _load_tensor(
            readers, "model.final_norm.weight").astype(np.float32)

        # MLM head weights (optional)
        if _has_tensor(readers, "head.dense.weight"):
            weights["head_dense_w"] = np.ascontiguousarray(
                _load_tensor(readers, "head.dense.weight").T.astype(np.float32))
        if _has_tensor(readers, "head.norm.weight"):
            weights["head_norm"] = _load_tensor(
                readers, "head.norm.weight").astype(np.float32)
        if _has_tensor(readers, "decoder.bias"):
            weights["decoder_bias"] = _load_tensor(
                readers, "decoder.bias").astype(np.float32)

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"model.layers.{layer_idx}"

            # Attention LayerNorm (layer 0 has no attn_norm)
            attn_norm_key = f"{hf_prefix}.attn_norm.weight"
            if _has_tensor(readers, attn_norm_key):
                weights[f"{prefix}.attn_norm"] = _load_tensor(
                    readers, attn_norm_key).astype(np.float32)

            # Fused QKV: [3*hidden, hidden] -> split into Q, K, V
            wqkv = _load_tensor(readers, f"{hf_prefix}.attn.Wqkv.weight")
            assert wqkv.shape == (3 * hidden, hidden)
            q_w, k_w, v_w = np.split(wqkv, 3, axis=0)
            weights[f"{prefix}.w_q"] = np.ascontiguousarray(q_w.T.astype(np.float32))
            weights[f"{prefix}.w_k"] = np.ascontiguousarray(k_w.T.astype(np.float32))
            weights[f"{prefix}.w_v"] = np.ascontiguousarray(v_w.T.astype(np.float32))

            # Output projection
            wo = _load_tensor(readers, f"{hf_prefix}.attn.Wo.weight")
            weights[f"{prefix}.w_o"] = np.ascontiguousarray(wo.T.astype(np.float32))

            # MLP LayerNorm
            weights[f"{prefix}.mlp_norm"] = _load_tensor(
                readers, f"{hf_prefix}.mlp_norm.weight").astype(np.float32)

            # GeGLU MLP: Wi [2*intermediate, hidden] -> split into input, gate
            wi = _load_tensor(readers, f"{hf_prefix}.mlp.Wi.weight")
            assert wi.shape == (2 * intermediate, hidden)
            input_w, gate_w = np.split(wi, 2, axis=0)
            weights[f"{prefix}.w_mlp_input"] = np.ascontiguousarray(input_w.T.astype(np.float32))
            weights[f"{prefix}.w_mlp_gate"] = np.ascontiguousarray(gate_w.T.astype(np.float32))

            # Down projection
            mlp_wo = _load_tensor(readers, f"{hf_prefix}.mlp.Wo.weight")
            weights[f"{prefix}.w_down"] = np.ascontiguousarray(mlp_wo.T.astype(np.float32))

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, verbose: bool = False,
    ) -> bytes:
        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        head_dim = hidden // num_heads
        intermediate = config.intermediate_size
        eps = config.raw.get("norm_eps", config.rms_norm_eps)
        max_seq = max_cache_length

        # Per-layer RoPE theta from layer_types
        layer_types = config.raw.get("layer_types", [])
        rope_params = config.raw.get("rope_parameters", {})
        # Default theta values
        full_theta = 160000.0
        sliding_theta = 10000.0
        if rope_params:
            if "full_attention" in rope_params and rope_params["full_attention"]:
                full_theta = rope_params["full_attention"].get("rope_theta", 160000.0)
            if "sliding_attention" in rope_params and rope_params["sliding_attention"]:
                sliding_theta = rope_params["sliding_attention"].get("rope_theta", 10000.0)

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network()
        trt_config = builder.create_builder_config()
        trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        trt_config.clear_flag(trt.BuilderFlag.TF32)

        # Inputs
        input_ids = network.add_input("input_ids", trt.int32, (max_seq,))
        attention_mask_input = network.add_input("attention_mask", trt.int32, (max_seq,))

        # Attention scale constant
        attn_scale = graph_ops.add_constant(
            network, (1, 1, 1),
            np.array([1.0 / np.sqrt(max(head_dim, 1))], dtype=np.float32))

        # Attention mask: [seq] int -> [1, 1, seq] additive float mask
        mask_float = network.add_cast(attention_mask_input, trt.float32)
        ones_c = graph_ops.add_constant(network, (1,), np.array([1.0], dtype=np.float32))
        neg_large = graph_ops.add_constant(network, (1,), np.array([-1e10], dtype=np.float32))
        inv_mask = network.add_elementwise(
            ones_c, mask_float.get_output(0), trt.ElementWiseOperation.SUB)
        pad_penalty = network.add_elementwise(
            inv_mask.get_output(0), neg_large, trt.ElementWiseOperation.PROD)
        pad_mask_3d = network.add_shuffle(pad_penalty.get_output(0))
        pad_mask_3d.reshape_dims = (1, 1, max_seq)

        # Pre-compute RoPE tables for both theta values
        rope_tables = {}
        for theta in set([full_theta, sliding_theta]):
            cos = graph_ops.add_constant(
                network, (max_seq, hidden),
                graph_ops.make_rope_table(max_seq, hidden, num_heads, theta, cosine=True))
            sin = graph_ops.add_constant(
                network, (max_seq, hidden),
                graph_ops.make_rope_table(max_seq, hidden, num_heads, theta, cosine=False))
            rope_tables[theta] = (cos, sin)

        rotate_half = graph_ops.add_constant(
            network, (hidden, hidden),
            graph_ops.make_rotate_half_matrix(hidden, num_heads))

        pos_indices = graph_ops.add_constant(
            network, (max_seq,), np.arange(max_seq, dtype=np.float32))
        pos_int = network.add_identity(pos_indices)
        pos_int.set_output_type(0, trt.int32)

        # Embedding
        embed_table = graph_ops.add_constant(network, (vocab, hidden), weights["embedding"])
        word_embed = network.add_gather(embed_table, input_ids, 0)
        hidden_state = _add_layernorm_no_bias(
            network, word_embed.get_output(0), hidden, weights["embed_norm"], eps)

        # Encoder layers
        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"

            # Determine RoPE theta for this layer
            if layer_idx < len(layer_types):
                lt = layer_types[layer_idx]
                if lt in ("full_attention", "global_attention"):
                    theta = full_theta
                else:
                    theta = sliding_theta
            else:
                theta = full_theta
            cos_table, sin_table = rope_tables[theta]

            # Pre-norm attention
            has_attn_norm = f"{prefix}.attn_norm" in weights
            if has_attn_norm:
                attn_input = _add_layernorm_no_bias(
                    network, hidden_state, hidden,
                    weights[f"{prefix}.attn_norm"], eps)
            else:
                attn_input = hidden_state

            # QKV projections
            q = graph_ops.add_matmul_rhs_constant(network, attn_input, hidden, hidden, weights[f"{prefix}.w_q"])
            k = graph_ops.add_matmul_rhs_constant(network, attn_input, hidden, hidden, weights[f"{prefix}.w_k"])
            v = graph_ops.add_matmul_rhs_constant(network, attn_input, hidden, hidden, weights[f"{prefix}.w_v"])

            # RoPE
            q = graph_ops.add_apply_rope(network, q, pos_int.get_output(0), cos_table, sin_table, rotate_half)
            k = graph_ops.add_apply_rope(network, k, pos_int.get_output(0), cos_table, sin_table, rotate_half)

            # Multi-head reshape
            q_heads = network.add_shuffle(q)
            q_heads.reshape_dims = (max_seq, num_heads, head_dim)
            q_heads.second_transpose = trt.Permutation([1, 0, 2])

            k_heads = network.add_shuffle(k)
            k_heads.reshape_dims = (max_seq, num_heads, head_dim)
            k_heads.second_transpose = trt.Permutation([1, 0, 2])

            v_heads = network.add_shuffle(v)
            v_heads.reshape_dims = (max_seq, num_heads, head_dim)
            v_heads.second_transpose = trt.Permutation([1, 0, 2])

            # Attention
            score = network.add_matrix_multiply(
                q_heads.get_output(0), trt.MatrixOperation.NONE,
                k_heads.get_output(0), trt.MatrixOperation.TRANSPOSE)
            scaled = network.add_elementwise(
                score.get_output(0), attn_scale, trt.ElementWiseOperation.PROD)
            masked = network.add_elementwise(
                scaled.get_output(0), pad_mask_3d.get_output(0), trt.ElementWiseOperation.SUM)
            softmax = network.add_softmax(masked.get_output(0))
            softmax.axes = 1 << 2

            context = network.add_matrix_multiply(
                softmax.get_output(0), trt.MatrixOperation.NONE,
                v_heads.get_output(0), trt.MatrixOperation.NONE)
            context_flat = network.add_shuffle(context.get_output(0))
            context_flat.first_transpose = trt.Permutation([1, 0, 2])
            context_flat.reshape_dims = (max_seq, hidden)

            attn_out = graph_ops.add_matmul_rhs_constant(
                network, context_flat.get_output(0), hidden, hidden, weights[f"{prefix}.w_o"])

            # Residual
            res1 = network.add_elementwise(hidden_state, attn_out, trt.ElementWiseOperation.SUM)
            hidden_state = res1.get_output(0)

            # Pre-norm GeGLU MLP
            mlp_input = _add_layernorm_no_bias(
                network, hidden_state, hidden, weights[f"{prefix}.mlp_norm"], eps)

            # GeGLU: act(input) * gate
            inp_proj = graph_ops.add_matmul_rhs_constant(
                network, mlp_input, hidden, intermediate, weights[f"{prefix}.w_mlp_input"])
            gate_proj = graph_ops.add_matmul_rhs_constant(
                network, mlp_input, hidden, intermediate, weights[f"{prefix}.w_mlp_gate"])
            inp_act = graph_ops.add_gelu_erf(network, inp_proj)
            gated = network.add_elementwise(inp_act, gate_proj, trt.ElementWiseOperation.PROD)

            down = graph_ops.add_matmul_rhs_constant(
                network, gated.get_output(0), intermediate, hidden, weights[f"{prefix}.w_down"])

            res2 = network.add_elementwise(hidden_state, down, trt.ElementWiseOperation.SUM)
            hidden_state = res2.get_output(0)

        # Final norm
        hidden_state = _add_layernorm_no_bias(
            network, hidden_state, hidden, weights["final_norm"], eps)

        hidden_state.name = "hidden_states"
        network.mark_output(hidden_state)

        if verbose:
            print(f"[trtf-build] Building ModernBERT encoder TRT engine "
                  f"({num_layers} layers, hidden={hidden}, seq_len={max_seq}) ...",
                  file=sys.stderr)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT engine build failed")
        return bytes(plan)


plugin = ModernbertPlugin()
