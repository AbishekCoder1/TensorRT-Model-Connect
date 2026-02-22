"""Bark family plugin -- text-to-audio generation.

Bark is a multi-stage text-to-audio model:
  1. Semantic: text tokens -> semantic tokens (GPT decoder with learned positions)
  2. Coarse: semantic tokens -> coarse audio codes (GPT decoder)
  3. Fine: coarse codes -> fine audio codes (iterative refinement, no KV cache)
  4. Codec (EnCodec): audio codes -> waveform

HF Bark uses `layers.{i}.attn.att_proj.weight` in [3H, H] format (fused QKV).
This is NOT GPT-2 Conv1D -- the weights are already in standard linear format.
att_proj splits into Q, K, V each of dim H.

Weight key mapping:
  HF: semantic/coarse/fine.model.transformer.wte.weight
  HF: semantic/coarse/fine.model.transformer.wpe.weight
  HF: semantic/coarse/fine.model.transformer.h.{i}.layernorm_1/2.weight/bias
  HF: semantic/coarse/fine.model.transformer.h.{i}.attn.att_proj.weight
  HF: semantic/coarse/fine.model.transformer.h.{i}.attn.out_proj.weight/bias
  HF: semantic/coarse/fine.model.transformer.h.{i}.mlp.in_proj.weight/bias
  HF: semantic/coarse/fine.model.transformer.h.{i}.mlp.out_proj.weight/bias
  HF: semantic/coarse/fine.model.lm_head.weight
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorrt as trt

from ..config import ModelConfig
from ..checkpoint_mapper import WeightDict
from .. import graph_ops
from ..standard_decoder_builder import build_standard_decoder


def _load_bark_state_dict(model_dir: str) -> dict:
    """Load Bark state dict from pytorch_model.bin."""
    import torch
    model_path = Path(model_dir) / "pytorch_model.bin"
    if not model_path.exists():
        # Try safetensors
        from ..checkpoint_mapper import _open_safetensors, _load_tensor
        readers = _open_safetensors(Path(model_dir))
        state_dict = {}
        for reader in readers:
            for key in reader.keys():
                state_dict[key] = reader.get_tensor(key).numpy()
        return state_dict
    return torch.load(str(model_path), map_location="cpu", weights_only=True)


def _detect_sub_model_config(state_dict: dict, prefix: str) -> dict:
    """Auto-detect dimensions from state dict for a sub-model."""
    # Find embedding weight to get vocab_size, hidden_size
    wte_key = f"{prefix}.model.transformer.wte.weight"
    if wte_key not in state_dict:
        return {}
    wte = state_dict[wte_key]
    if hasattr(wte, 'numpy'):
        wte = wte.numpy()
    vocab_size, hidden_size = wte.shape

    # Count layers
    num_layers = 0
    while f"{prefix}.model.transformer.h.{num_layers}.layernorm_1.weight" in state_dict:
        num_layers += 1

    # Detect num_heads from att_proj shape
    att_key = f"{prefix}.model.transformer.h.0.attn.att_proj.weight"
    num_heads = hidden_size // 64  # Default guess
    if att_key in state_dict:
        att_w = state_dict[att_key]
        if hasattr(att_w, 'numpy'):
            att_w = att_w.numpy()
        # att_proj is [3*H, H], head_dim = H / num_heads
        # Try common head dims: 64, 128
        for hd in [64, 128, 96]:
            if hidden_size % hd == 0:
                num_heads = hidden_size // hd
                break

    # Detect position embedding max length
    wpe_key = f"{prefix}.model.transformer.wpe.weight"
    max_position = 256
    if wpe_key in state_dict:
        wpe = state_dict[wpe_key]
        if hasattr(wpe, 'numpy'):
            wpe = wpe.numpy()
        max_position = wpe.shape[0]

    return {
        "vocab_size": int(vocab_size),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "max_position": int(max_position),
    }


def _map_bark_decoder_weights(
    state_dict: dict,
    prefix: str,
    sub_config: dict,
) -> WeightDict:
    """Map HF Bark decoder weights to standard decoder format."""
    weights = WeightDict()
    hidden = sub_config["hidden_size"]
    vocab = sub_config["vocab_size"]
    num_layers = sub_config["num_layers"]

    def _to_np(t):
        if hasattr(t, 'numpy'):
            return t.numpy().astype(np.float32)
        return np.asarray(t, dtype=np.float32)

    def _t2d(w):
        """Transpose [out, in] -> [in, out] for matmul."""
        a = _to_np(w)
        if a.ndim == 2:
            return np.ascontiguousarray(a.T)
        return a

    # Embedding
    weights["embedding"] = _to_np(state_dict[f"{prefix}.model.transformer.wte.weight"])

    # Position embedding
    weights["position_embedding"] = _to_np(state_dict[f"{prefix}.model.transformer.wpe.weight"])

    for i in range(num_layers):
        hf_layer = f"{prefix}.model.transformer.h.{i}"
        layer = f"layer.{i}"

        # Layer norms
        weights[f"{layer}.norm1_gamma"] = _to_np(state_dict[f"{hf_layer}.layernorm_1.weight"])
        weights[f"{layer}.norm1_beta"] = _to_np(state_dict[f"{hf_layer}.layernorm_1.bias"])
        weights[f"{layer}.norm2_gamma"] = _to_np(state_dict[f"{hf_layer}.layernorm_2.weight"])
        weights[f"{layer}.norm2_beta"] = _to_np(state_dict[f"{hf_layer}.layernorm_2.bias"])

        # att_proj: [3*H, H] -> split into Q [H,H], K [H,H], V [H,H]
        att_w = _to_np(state_dict[f"{hf_layer}.attn.att_proj.weight"])  # [3H, H]
        w_q = att_w[:hidden, :]     # [H, H]
        w_k = att_w[hidden:2*hidden, :]  # [H, H]
        w_v = att_w[2*hidden:, :]   # [H, H]

        # Transpose each: [H, H] -> [H, H] for matmul (lhs @ rhs)
        weights[f"{layer}.w_q"] = _t2d(w_q)
        weights[f"{layer}.w_k"] = _t2d(w_k)
        weights[f"{layer}.w_v"] = _t2d(w_v)

        # Output projection
        weights[f"{layer}.w_o"] = _t2d(state_dict[f"{hf_layer}.attn.out_proj.weight"])
        weights[f"{layer}.b_o"] = _to_np(state_dict[f"{hf_layer}.attn.out_proj.bias"])

        # MLP
        weights[f"{layer}.w_fc1"] = _t2d(state_dict[f"{hf_layer}.mlp.in_proj.weight"])
        weights[f"{layer}.b_fc1"] = _to_np(state_dict[f"{hf_layer}.mlp.in_proj.bias"])
        weights[f"{layer}.w_fc2"] = _t2d(state_dict[f"{hf_layer}.mlp.out_proj.weight"])
        weights[f"{layer}.b_fc2"] = _to_np(state_dict[f"{hf_layer}.mlp.out_proj.bias"])

    # Final layer norm
    ln_f_prefix = f"{prefix}.model.transformer.layernorm_final"
    if f"{ln_f_prefix}.weight" in state_dict:
        weights["final_norm_gamma"] = _to_np(state_dict[f"{ln_f_prefix}.weight"])
        weights["final_norm_beta"] = _to_np(state_dict[f"{ln_f_prefix}.bias"])
    else:
        weights["final_norm_gamma"] = np.ones(hidden, dtype=np.float32)
        weights["final_norm_beta"] = np.zeros(hidden, dtype=np.float32)

    # LM head
    lm_key = f"{prefix}.model.lm_head.weight"
    if lm_key in state_dict:
        weights["w_lm_head"] = _t2d(state_dict[lm_key])
    else:
        weights["w_lm_head"] = _t2d(state_dict[f"{prefix}.model.transformer.wte.weight"])

    return weights


class BarkPlugin:
    name = "bark"
    runtime_strategy = "text_to_audio"

    def matches(self, model_type: str) -> bool:
        return model_type.lower() in ("bark",)

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        """Load Bark weights from pytorch_model.bin."""
        state_dict = _load_bark_state_dict(model_dir)

        # Detect sub-model configs
        semantic_cfg = _detect_sub_model_config(state_dict, "semantic")
        coarse_cfg = _detect_sub_model_config(state_dict, "coarse_acoustics")
        fine_cfg = _detect_sub_model_config(state_dict, "fine_acoustics")

        weights = WeightDict()

        # Map each sub-model's weights
        sem_w = _map_bark_decoder_weights(state_dict, "semantic", semantic_cfg)
        for k, v in sem_w.items():
            weights[f"semantic.{k}"] = v

        coarse_w = _map_bark_decoder_weights(state_dict, "coarse_acoustics", coarse_cfg)
        for k, v in coarse_w.items():
            weights[f"coarse.{k}"] = v

        fine_w = _map_bark_decoder_weights(state_dict, "fine_acoustics", fine_cfg)
        for k, v in fine_w.items():
            weights[f"fine.{k}"] = v

        # Store sub-model configs
        weights["_semantic_cfg"] = semantic_cfg
        weights["_coarse_cfg"] = coarse_cfg
        weights["_fine_cfg"] = fine_cfg

        # Store codec info
        weights["_codec_model_id"] = "facebook/encodec_24khz"

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, verbose: bool = False,
    ) -> bytes:
        """Build TRT engine for semantic decoder (primary engine)."""
        sem_cfg = weights["_semantic_cfg"]
        return _build_bark_decoder_engine(
            weights, "semantic", sem_cfg, max_cache_length,
            embed_input=True, verbose=verbose)

    def build_extra_engines(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, verbose: bool = False,
    ) -> dict:
        """Build coarse, fine, and codec engines."""
        coarse_cfg = weights["_coarse_cfg"]
        fine_cfg = weights["_fine_cfg"]

        coarse_plan = _build_bark_decoder_engine(
            weights, "coarse", coarse_cfg, max_cache_length,
            embed_input=True, verbose=verbose)

        fine_plan = _build_bark_fine_engine(
            weights, fine_cfg, verbose=verbose)

        # Codec engine is built separately from EnCodec weights
        codec_plan = None  # Built from encodec_builder at runtime

        result = {
            "coarse_engine_plan": coarse_plan,
            "fine_engine_plan": fine_plan,
        }
        if codec_plan is not None:
            result["codec_engine_plan"] = codec_plan
        return result

    def get_audio_config(self, config: ModelConfig) -> dict:
        """Return audio config for bundle config.json."""
        return {
            "sample_rate": 24000,
            "semantic_vocab_size": 10000,
            "coarse_vocab_size": 1024,
            "fine_vocab_size": 1024,
            "n_coarse_codebooks": 2,
            "n_fine_codebooks": 8,
            "semantic_pad_token": 10000,
            "semantic_infer_token": 10001,
            "coarse_semantic_pad_token": 12048,
            "coarse_infer_token": 12050,
        }


def _build_bark_decoder_engine(
    weights: WeightDict,
    sub_model: str,
    sub_cfg: dict,
    max_cache_length: int,
    embed_input: bool = True,
    verbose: bool = False,
) -> bytes:
    """Build a standard decoder engine for semantic or coarse sub-model."""
    hidden = sub_cfg["hidden_size"]
    vocab = sub_cfg["vocab_size"]
    num_layers = sub_cfg["num_layers"]
    num_heads = sub_cfg["num_heads"]
    head_dim = hidden // num_heads

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    trt_config.clear_flag(trt.BuilderFlag.TF32)

    attention_window = max_cache_length + 1
    attention_size = num_heads * head_dim
    prefix = f"{sub_model}."

    # Inputs
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

    # Embedding tables
    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), weights[f"{prefix}embedding"])
    position_table = graph_ops.add_constant(
        network, (sub_cfg["max_position"], hidden),
        weights[f"{prefix}position_embedding"])

    # Token + position embedding
    tok_embed = network.add_gather(embedding_table, token_id, 0)
    pos_embed = network.add_gather(position_table, position_id, 0)
    hidden_state = network.add_elementwise(
        tok_embed.get_output(0), pos_embed.get_output(0),
        trt.ElementWiseOperation.SUM).get_output(0)

    eps_t = graph_ops.add_constant(
        network, (1, 1), np.array([1e-5], dtype=np.float32))

    # RoPE tables not needed (learned positions)
    # cos/sin tables for apply_rope
    cos_table = None
    sin_table = None
    rotate_half_matrix = None

    present_k_outputs = []
    present_v_outputs = []

    for layer_idx in range(num_layers):
        lp = f"{prefix}layer.{layer_idx}"

        # Pre-attention LayerNorm
        normed = graph_ops.add_layer_norm(
            network, hidden_state, hidden,
            weights[f"{lp}.norm1_gamma"],
            weights[f"{lp}.norm1_beta"], eps_t)

        # Q, K, V projections
        q = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden, attention_size,
            weights[f"{lp}.w_q"])
        k = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden, attention_size,
            weights[f"{lp}.w_k"])
        v = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden, attention_size,
            weights[f"{lp}.w_v"])

        # Attention with KV cache
        # K -> present_k: [1, attention_size]
        # Gather from cache
        attn_scale = 1.0 / np.sqrt(head_dim)
        scale_c = graph_ops.add_constant(
            network, (1, 1), np.array([attn_scale], dtype=np.float32))

        # present_k = k (single new row)
        present_k = k
        present_v = v

        # Full K = cache_k concat present_k for attention
        # [max_cache, attn] + [1, attn] -> [max_cache+1, attn]
        full_k = network.add_concatenation(
            [cache_k_inputs[layer_idx], present_k])
        full_k.axis = 0

        full_v = network.add_concatenation(
            [cache_v_inputs[layer_idx], present_v])
        full_v.axis = 0

        # Multi-head: q [1, attn] -> [num_heads, 1, head_dim]
        q_h = network.add_shuffle(q)
        q_h.reshape_dims = (1, num_heads, head_dim)
        q_h.second_transpose = trt.Permutation([1, 0, 2])

        # full_k [attn_window, attn] -> [num_heads, attn_window, head_dim]
        k_h = network.add_shuffle(full_k.get_output(0))
        k_h.reshape_dims = (attention_window, num_heads, head_dim)
        k_h.second_transpose = trt.Permutation([1, 0, 2])

        v_h = network.add_shuffle(full_v.get_output(0))
        v_h.reshape_dims = (attention_window, num_heads, head_dim)
        v_h.second_transpose = trt.Permutation([1, 0, 2])

        # Score: [heads, 1, head_dim] @ [heads, head_dim, window] -> [heads, 1, window]
        score = network.add_matrix_multiply(
            q_h.get_output(0), trt.MatrixOperation.NONE,
            k_h.get_output(0), trt.MatrixOperation.TRANSPOSE)
        scaled = network.add_elementwise(
            score.get_output(0), graph_ops.add_constant(
                network, (1, 1, 1), np.array([attn_scale], dtype=np.float32)),
            trt.ElementWiseOperation.PROD)

        # Add attention mask: [1, attention_window] -> broadcast to [heads, 1, window]
        mask_3d = network.add_shuffle(attention_mask)
        mask_3d.reshape_dims = (1, 1, attention_window)
        masked = network.add_elementwise(
            scaled.get_output(0), mask_3d.get_output(0),
            trt.ElementWiseOperation.SUM)

        softmax = network.add_softmax(masked.get_output(0))
        softmax.axes = 1 << 2

        ctx = network.add_matrix_multiply(
            softmax.get_output(0), trt.MatrixOperation.NONE,
            v_h.get_output(0), trt.MatrixOperation.NONE)

        # Reshape: [heads, 1, head_dim] -> [1, attention_size]
        ctx_flat = network.add_shuffle(ctx.get_output(0))
        ctx_flat.first_transpose = trt.Permutation([1, 0, 2])
        ctx_flat.reshape_dims = (1, attention_size)

        # Output projection
        attn_out = graph_ops.add_matmul_rhs_constant(
            network, ctx_flat.get_output(0), attention_size, hidden,
            weights[f"{lp}.w_o"])
        attn_out = graph_ops.add_bias_sum(
            network, attn_out, hidden, weights[f"{lp}.b_o"])

        # Residual
        res1 = network.add_elementwise(
            hidden_state, attn_out, trt.ElementWiseOperation.SUM)
        hidden_state = res1.get_output(0)

        # Pre-FFN LayerNorm
        normed2 = graph_ops.add_layer_norm(
            network, hidden_state, hidden,
            weights[f"{lp}.norm2_gamma"],
            weights[f"{lp}.norm2_beta"], eps_t)

        # MLP: FC1 -> GELU -> FC2
        ffn_hidden = weights[f"{lp}.w_fc1"].shape[1]
        fc1 = graph_ops.add_matmul_rhs_constant(
            network, normed2, hidden, ffn_hidden,
            weights[f"{lp}.w_fc1"])
        fc1 = graph_ops.add_bias_sum(network, fc1, ffn_hidden,
            weights[f"{lp}.b_fc1"])
        gelu = graph_ops.add_gelu_new(network, fc1)
        fc2 = graph_ops.add_matmul_rhs_constant(
            network, gelu, ffn_hidden, hidden,
            weights[f"{lp}.w_fc2"])
        fc2 = graph_ops.add_bias_sum(network, fc2, hidden,
            weights[f"{lp}.b_fc2"])

        # Residual
        res2 = network.add_elementwise(
            hidden_state, fc2, trt.ElementWiseOperation.SUM)
        hidden_state = res2.get_output(0)

        present_k_outputs.append(present_k)
        present_v_outputs.append(present_v)

    # Final LayerNorm
    hidden_state = graph_ops.add_layer_norm(
        network, hidden_state, hidden,
        weights[f"{prefix}final_norm_gamma"],
        weights[f"{prefix}final_norm_beta"], eps_t)

    # LM head
    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden, vocab,
        weights[f"{prefix}w_lm_head"])

    logits.name = "logits"
    network.mark_output(logits)

    for i in range(num_layers):
        pk = present_k_outputs[i]
        pv = present_v_outputs[i]
        pk.name = graph_ops.layer_tensor_name("present_k", i)
        pv.name = graph_ops.layer_tensor_name("present_v", i)
        network.mark_output(pk)
        network.mark_output(pv)

    if verbose:
        print(f"[trtf-build] Building Bark {sub_model} engine "
              f"(layers={num_layers}, hidden={hidden}, vocab={vocab}) ...",
              file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError(f"TensorRT engine build failed for Bark {sub_model}")
    return bytes(plan)


def _build_bark_fine_engine(
    weights: WeightDict,
    fine_cfg: dict,
    verbose: bool = False,
) -> bytes:
    """Build a non-autoregressive fine decoder engine.

    Input: input_ids [1, seq_len] (coarse+fine codes)
    Output: logits [1, vocab_size]
    """
    hidden = fine_cfg["hidden_size"]
    vocab = fine_cfg["vocab_size"]
    num_layers = fine_cfg["num_layers"]
    num_heads = fine_cfg["num_heads"]
    head_dim = hidden // num_heads
    max_seq = 1024  # fixed sequence length for fine model
    prefix = "fine."

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    trt_config.clear_flag(trt.BuilderFlag.TF32)

    # Input: single token for iterative refinement
    token_id = network.add_input("token_id", trt.int32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))

    # Embedding
    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), weights[f"{prefix}embedding"])
    position_table = graph_ops.add_constant(
        network, (fine_cfg["max_position"], hidden),
        weights[f"{prefix}position_embedding"])

    tok_embed = network.add_gather(embedding_table, token_id, 0)
    pos_embed = network.add_gather(position_table, position_id, 0)
    hidden_state = network.add_elementwise(
        tok_embed.get_output(0), pos_embed.get_output(0),
        trt.ElementWiseOperation.SUM).get_output(0)

    eps_t = graph_ops.add_constant(
        network, (1, 1), np.array([1e-5], dtype=np.float32))

    for layer_idx in range(num_layers):
        lp = f"{prefix}layer.{layer_idx}"

        normed = graph_ops.add_layer_norm(
            network, hidden_state, hidden,
            weights[f"{lp}.norm1_gamma"],
            weights[f"{lp}.norm1_beta"], eps_t)

        # Simple self-attention (no cache for fine model -- single pass)
        q = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden, hidden, weights[f"{lp}.w_q"])
        k = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden, hidden, weights[f"{lp}.w_k"])
        v = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden, hidden, weights[f"{lp}.w_v"])

        attn_scale = 1.0 / np.sqrt(head_dim)
        score = network.add_matrix_multiply(
            q, trt.MatrixOperation.NONE,
            k, trt.MatrixOperation.TRANSPOSE)
        scale_c = graph_ops.add_constant(
            network, (1, 1), np.array([attn_scale], dtype=np.float32))
        scaled = network.add_elementwise(
            score.get_output(0), scale_c, trt.ElementWiseOperation.PROD)

        softmax = network.add_softmax(scaled.get_output(0))
        softmax.axes = 1 << 1

        ctx = network.add_matrix_multiply(
            softmax.get_output(0), trt.MatrixOperation.NONE,
            v, trt.MatrixOperation.NONE)

        attn_out = graph_ops.add_matmul_rhs_constant(
            network, ctx.get_output(0), hidden, hidden, weights[f"{lp}.w_o"])
        attn_out = graph_ops.add_bias_sum(
            network, attn_out, hidden, weights[f"{lp}.b_o"])

        res1 = network.add_elementwise(
            hidden_state, attn_out, trt.ElementWiseOperation.SUM)
        hidden_state = res1.get_output(0)

        normed2 = graph_ops.add_layer_norm(
            network, hidden_state, hidden,
            weights[f"{lp}.norm2_gamma"],
            weights[f"{lp}.norm2_beta"], eps_t)

        ffn_hidden = weights[f"{lp}.w_fc1"].shape[1]
        fc1 = graph_ops.add_matmul_rhs_constant(
            network, normed2, hidden, ffn_hidden, weights[f"{lp}.w_fc1"])
        fc1 = graph_ops.add_bias_sum(network, fc1, ffn_hidden, weights[f"{lp}.b_fc1"])
        gelu = graph_ops.add_gelu_new(network, fc1)
        fc2 = graph_ops.add_matmul_rhs_constant(
            network, gelu, ffn_hidden, hidden, weights[f"{lp}.w_fc2"])
        fc2 = graph_ops.add_bias_sum(network, fc2, hidden, weights[f"{lp}.b_fc2"])

        res2 = network.add_elementwise(
            hidden_state, fc2, trt.ElementWiseOperation.SUM)
        hidden_state = res2.get_output(0)

    hidden_state = graph_ops.add_layer_norm(
        network, hidden_state, hidden,
        weights[f"{prefix}final_norm_gamma"],
        weights[f"{prefix}final_norm_beta"], eps_t)

    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden, vocab, weights[f"{prefix}w_lm_head"])
    logits.name = "logits"
    network.mark_output(logits)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed for Bark fine model")
    return bytes(plan)


plugin = BarkPlugin()
