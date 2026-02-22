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
from ..standard_decoder_builder import build_standard_decoder_engine


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
    """Auto-detect dimensions from state dict for a sub-model.

    HF Bark uses keys like:
      {prefix}.input_embeds_layer.weight  [vocab, hidden]
      {prefix}.position_embeds_layer.weight  [max_pos, hidden]
      {prefix}.layers.{i}.layernorm_1.weight
      {prefix}.layers.{i}.attn.att_proj.weight  [3*H, H]
      {prefix}.lm_head.weight  [output_vocab, hidden]  (semantic/coarse only)
    Fine model uses:
      {prefix}.input_embeds_layers.{i}.weight  [vocab, hidden]
      {prefix}.lm_heads.{i}.weight  [output_vocab, hidden]
    """
    # Find embedding weight to get vocab_size, hidden_size
    wte_key = f"{prefix}.input_embeds_layer.weight"
    if wte_key not in state_dict:
        # Fine model has multiple embedding tables
        wte_key = f"{prefix}.input_embeds_layers.0.weight"
    if wte_key not in state_dict:
        return {}
    wte = state_dict[wte_key]
    if hasattr(wte, 'numpy'):
        wte = wte.numpy()
    vocab_size, hidden_size = wte.shape

    # Count layers
    num_layers = 0
    while f"{prefix}.layers.{num_layers}.layernorm_1.weight" in state_dict:
        num_layers += 1

    # Detect num_heads from att_proj shape
    num_heads = hidden_size // 64  # Default guess
    att_key = f"{prefix}.layers.0.attn.att_proj.weight"
    if att_key in state_dict:
        for hd in [64, 128, 96]:
            if hidden_size % hd == 0:
                num_heads = hidden_size // hd
                break

    # Detect position embedding max length
    wpe_key = f"{prefix}.position_embeds_layer.weight"
    max_position = 1024
    if wpe_key in state_dict:
        wpe = state_dict[wpe_key]
        if hasattr(wpe, 'numpy'):
            wpe = wpe.numpy()
        max_position = wpe.shape[0]

    # Output vocab (from lm_head)
    lm_key = f"{prefix}.lm_head.weight"
    output_vocab = vocab_size
    if lm_key in state_dict:
        lm_w = state_dict[lm_key]
        if hasattr(lm_w, 'numpy'):
            lm_w = lm_w.numpy()
        output_vocab = lm_w.shape[0]

    return {
        "vocab_size": int(vocab_size),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "max_position": int(max_position),
        "output_vocab": int(output_vocab),
        "intermediate_size": int(hidden_size * 4),
    }


def _map_bark_decoder_weights(
    state_dict: dict,
    prefix: str,
    sub_config: dict,
) -> WeightDict:
    """Map HF Bark decoder weights to standard decoder format.

    HF Bark key patterns:
      {prefix}.input_embeds_layer.weight  [vocab, hidden]
      {prefix}.position_embeds_layer.weight  [max_pos, hidden]
      {prefix}.layers.{i}.layernorm_1.weight  [hidden]  (no bias in bark-small)
      {prefix}.layers.{i}.attn.att_proj.weight  [3*H, H]  (fused QKV)
      {prefix}.layers.{i}.attn.out_proj.weight  [H, H]
      {prefix}.layers.{i}.mlp.in_proj.weight  [4*H, H]
      {prefix}.layers.{i}.mlp.out_proj.weight  [H, 4*H]
      {prefix}.layernorm_final.weight  [hidden]
      {prefix}.lm_head.weight  [output_vocab, hidden]

    Standard decoder builder expects:
      embedding [vocab, hidden], position_embedding [max_pos, hidden]
      layer.{i}.input_norm [hidden], layer.{i}.input_norm_beta [hidden]
      layer.{i}.w_q/w_k/w_v [hidden, hidden] (transposed: [in, out])
      layer.{i}.w_o [hidden, hidden]
      layer.{i}.post_attn_norm [hidden], layer.{i}.post_attn_norm_beta [hidden]
      layer.{i}.w_fc1 [hidden, 4*hidden], layer.{i}.w_fc2 [4*hidden, hidden]
      final_norm [hidden], final_norm_beta [hidden]
      w_out [hidden, output_vocab]
    """
    weights = WeightDict()
    hidden = sub_config["hidden_size"]
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
    weights["embedding"] = _to_np(state_dict[f"{prefix}.input_embeds_layer.weight"])

    # Position embedding
    weights["position_embedding"] = _to_np(state_dict[f"{prefix}.position_embeds_layer.weight"])

    for i in range(num_layers):
        hf = f"{prefix}.layers.{i}"
        layer = f"layer.{i}"

        # Layer norms (bark-small has weight only, no bias)
        weights[f"{layer}.input_norm"] = _to_np(state_dict[f"{hf}.layernorm_1.weight"])
        if f"{hf}.layernorm_1.bias" in state_dict:
            weights[f"{layer}.input_norm_beta"] = _to_np(state_dict[f"{hf}.layernorm_1.bias"])
        else:
            weights[f"{layer}.input_norm_beta"] = np.zeros(hidden, dtype=np.float32)

        weights[f"{layer}.post_attn_norm"] = _to_np(state_dict[f"{hf}.layernorm_2.weight"])
        if f"{hf}.layernorm_2.bias" in state_dict:
            weights[f"{layer}.post_attn_norm_beta"] = _to_np(state_dict[f"{hf}.layernorm_2.bias"])
        else:
            weights[f"{layer}.post_attn_norm_beta"] = np.zeros(hidden, dtype=np.float32)

        # att_proj: [3*H, H] -> split into Q [H,H], K [H,H], V [H,H], transpose each
        att_w = _to_np(state_dict[f"{hf}.attn.att_proj.weight"])  # [3H, H]
        w_q = att_w[:hidden, :]           # [H, H] in [out, in]
        w_k = att_w[hidden:2*hidden, :]   # [H, H]
        w_v = att_w[2*hidden:, :]         # [H, H]
        weights[f"{layer}.w_q"] = _t2d(w_q)  # [in, out] = [H, H]
        weights[f"{layer}.w_k"] = _t2d(w_k)
        weights[f"{layer}.w_v"] = _t2d(w_v)

        # Output projection
        weights[f"{layer}.w_o"] = _t2d(state_dict[f"{hf}.attn.out_proj.weight"])

        # MLP: fc1 = in_proj, fc2 = out_proj
        weights[f"{layer}.w_fc1"] = _t2d(state_dict[f"{hf}.mlp.in_proj.weight"])
        weights[f"{layer}.w_fc2"] = _t2d(state_dict[f"{hf}.mlp.out_proj.weight"])

        # Biases (bark-small typically has no biases on attn/mlp, but handle if present)
        for bkey, wkey in [
            (f"{layer}.q_bias", f"{hf}.attn.att_proj.bias"),
            (f"{layer}.o_bias", f"{hf}.attn.out_proj.bias"),
            (f"{layer}.fc1_bias", f"{hf}.mlp.in_proj.bias"),
            (f"{layer}.fc2_bias", f"{hf}.mlp.out_proj.bias"),
        ]:
            if wkey in state_dict:
                b = _to_np(state_dict[wkey])
                if "q_bias" in bkey and len(b) == 3 * hidden:
                    # Fused QKV bias, split
                    weights[f"{layer}.q_bias"] = b[:hidden]
                    weights[f"{layer}.k_bias"] = b[hidden:2*hidden]
                    weights[f"{layer}.v_bias"] = b[2*hidden:]
                else:
                    weights[bkey] = b

    # Final layer norm
    ln_key = f"{prefix}.layernorm_final.weight"
    if ln_key in state_dict:
        weights["final_norm"] = _to_np(state_dict[ln_key])
    else:
        weights["final_norm"] = np.ones(hidden, dtype=np.float32)
    ln_bias_key = f"{prefix}.layernorm_final.bias"
    if ln_bias_key in state_dict:
        weights["final_norm_beta"] = _to_np(state_dict[ln_bias_key])
    else:
        weights["final_norm_beta"] = np.zeros(hidden, dtype=np.float32)

    # LM head: standard_decoder_builder uses "w_out"
    lm_key = f"{prefix}.lm_head.weight"
    if lm_key in state_dict:
        weights["w_out"] = _t2d(state_dict[lm_key])  # [hidden, output_vocab]
    else:
        # Tied embeddings
        weights["w_out"] = _t2d(state_dict[f"{prefix}.input_embeds_layer.weight"])

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

        # Fine model has different structure (8 embed tables, 7 LM heads) — store raw state_dict keys
        weights["_state_dict"] = state_dict  # Needed for fine + codec engine builds

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
        return _build_bark_standard_engine(
            weights, "semantic", sem_cfg, max_cache_length,
            embed_input=True, verbose=verbose)

    def build_extra_engines(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, verbose: bool = False,
    ) -> dict:
        """Build coarse, fine, and codec engines."""
        coarse_cfg = weights["_coarse_cfg"]
        fine_cfg = weights["_fine_cfg"]

        coarse_plan = _build_bark_standard_engine(
            weights, "coarse", coarse_cfg, max_cache_length,
            embed_input=True, verbose=verbose)

        # Fine and codec engines require additional work (different architectures).
        # For now, only build semantic + coarse TRT engines.
        # Fine stage and codec will use HF models at runtime for full pipeline,
        # or the codec can be built separately via encodec_builder.

        return {
            "coarse_engine": coarse_plan,
        }

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


def _build_bark_standard_engine(
    weights: WeightDict,
    sub_model: str,
    sub_cfg: dict,
    max_cache_length: int,
    embed_input: bool = True,
    verbose: bool = False,
) -> bytes:
    """Build a standard decoder engine for semantic or coarse using build_standard_decoder_engine."""
    from ..standard_decoder_builder import build_standard_decoder_engine

    # Extract sub-model weights (strip prefix)
    prefix = f"{sub_model}."
    sub_weights = WeightDict()
    for k, v in weights.items():
        if k.startswith(prefix) and not k.startswith("_"):
            sub_weights[k[len(prefix):]] = v

    # Also need _attention_size and _mlp_size for the builder
    hidden = sub_cfg["hidden_size"]
    num_heads = sub_cfg["num_heads"]
    head_dim = hidden // num_heads
    sub_weights["_attention_size"] = num_heads * head_dim
    sub_weights["_mlp_size"] = sub_cfg.get("intermediate_size", hidden * 4)

    # Create a ModelConfig for this sub-model
    sub_mc = ModelConfig(
        model_type="bark",
        vocab_size=sub_cfg["vocab_size"],
        hidden_size=hidden,
        intermediate_size=sub_cfg.get("intermediate_size", hidden * 4),
        num_hidden_layers=sub_cfg["num_layers"],
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        max_position_embeddings=sub_cfg.get("max_position", 1024),
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        raw={},
    )

    if verbose:
        print(f"[trtf-build]   Building {sub_model} engine: "
              f"layers={sub_cfg['num_layers']}, hidden={hidden}, "
              f"vocab={sub_cfg['vocab_size']}, output_vocab={sub_cfg.get('output_vocab', sub_cfg['vocab_size'])}",
              file=sys.stderr)

    return build_standard_decoder_engine(
        sub_mc, sub_weights, max_cache_length,
        norm_type="layernorm",
        mlp_type="gelu_fc",
        position_type="learned",
        activation="gelu_new",
        embed_input=embed_input,
        verbose=verbose,
    )


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
