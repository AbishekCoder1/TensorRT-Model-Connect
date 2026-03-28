"""BART family plugin -- encoder-decoder seq2seq model.

BART is an encoder-decoder transformer for text generation (summarization,
translation, etc.):
  - Encoder: token embeddings + learned positional embeddings + LayerNorm
             -> N self-attention layers -> encoder output [seq_len, d_model]
  - Decoder: autoregressive text generation with causal self-attention (KV cache)
             + cross-attention to encoder output + GELU MLP
  - Uses LayerNorm, GELU activation, learned positional embeddings
  - model_type: "bart", architectures: ["BartModel", "BartForConditionalGeneration"]
  - Shared embedding between encoder and decoder
  - Position embeddings have offset=2 (first 2 positions are reserved)
  - Post-norm (normalize_before=False): norm AFTER residual connection

Cross-attention design:
  Same as Whisper -- cross_k/cross_v inputs to the decoder engine are the RAW
  encoder output. Per-layer K/V projections are baked into the decoder TRT graph.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorrt as trt

from ..config import ModelConfig
from ..checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from .. import graph_ops
from .. import graph_blocks


class BartPlugin:
    name = "bart"
    runtime_strategy = "seq2seq_encoder_decoder"

    def matches(self, model_type: str) -> bool:
        return model_type.lower() in ("bart", "mbart")

    def load_weights(self, model_dir: str, config: ModelConfig) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)
        raw = config.raw
        hidden = config.hidden_size
        enc_layers = raw.get("encoder_layers", config.num_hidden_layers)
        dec_layers = raw.get("decoder_layers", config.num_hidden_layers)
        enc_heads = raw.get("encoder_attention_heads", config.num_attention_heads)
        dec_heads = raw.get("decoder_attention_heads", config.num_attention_heads)
        enc_ffn = raw.get("encoder_ffn_dim", config.intermediate_size)
        dec_ffn = raw.get("decoder_ffn_dim", config.intermediate_size)
        max_position_embeddings = raw.get("max_position_embeddings", 1024)
        normalize_embedding = raw.get("normalize_embedding", True)

        weights = WeightDict()
        weights["_enc_layers"] = enc_layers
        weights["_dec_layers"] = dec_layers
        weights["_enc_heads"] = enc_heads
        weights["_dec_heads"] = dec_heads
        weights["_enc_ffn"] = enc_ffn
        weights["_dec_ffn"] = dec_ffn
        weights["_max_position_embeddings"] = max_position_embeddings
        weights["_normalize_embedding"] = normalize_embedding

        # Shared embedding (used by both encoder and decoder)
        if _has_tensor(readers, "shared.weight"):
            shared_embed = _load_tensor(readers, "shared.weight").astype(np.float32)
        elif _has_tensor(readers, "model.shared.weight"):
            shared_embed = _load_tensor(readers, "model.shared.weight").astype(np.float32)
        else:
            raise RuntimeError("BART: cannot find shared embedding weight")
        weights["shared_embedding"] = shared_embed

        # Encoder position embeddings (shape [max_pos+2, hidden] due to offset=2)
        for key in ("encoder.embed_positions.weight", "model.encoder.embed_positions.weight"):
            if _has_tensor(readers, key):
                weights["enc_pos_embedding"] = _load_tensor(readers, key).astype(np.float32)
                break
        if "enc_pos_embedding" not in weights:
            raise RuntimeError("BART: cannot find encoder position embeddings")

        # Encoder layernorm_embedding
        if normalize_embedding:
            for prefix in ("encoder", "model.encoder"):
                if _has_tensor(readers, f"{prefix}.layernorm_embedding.weight"):
                    weights["enc_embed_norm"] = _load_tensor(readers, f"{prefix}.layernorm_embedding.weight").astype(np.float32)
                    weights["enc_embed_norm_beta"] = _load_tensor(readers, f"{prefix}.layernorm_embedding.bias").astype(np.float32)
                    break

        # Encoder layers
        for i in range(enc_layers):
            hf = f"encoder.layers.{i}"
            if not _has_tensor(readers, f"{hf}.self_attn.q_proj.weight"):
                hf = f"model.encoder.layers.{i}"
            pfx = f"enc_layer.{i}"
            for proj in ("q", "k", "v"):
                weights[f"{pfx}.w_{proj}"] = _transpose_2d(_load_tensor(readers, f"{hf}.self_attn.{proj}_proj.weight"), f"enc_{proj}")
                weights[f"{pfx}.b_{proj}"] = _load_tensor(readers, f"{hf}.self_attn.{proj}_proj.bias").astype(np.float32)
            weights[f"{pfx}.w_o"] = _transpose_2d(_load_tensor(readers, f"{hf}.self_attn.out_proj.weight"), "enc_o")
            weights[f"{pfx}.b_o"] = _load_tensor(readers, f"{hf}.self_attn.out_proj.bias").astype(np.float32)
            weights[f"{pfx}.attn_norm"] = _load_tensor(readers, f"{hf}.self_attn_layer_norm.weight").astype(np.float32)
            weights[f"{pfx}.attn_norm_beta"] = _load_tensor(readers, f"{hf}.self_attn_layer_norm.bias").astype(np.float32)
            weights[f"{pfx}.w_fc1"] = _transpose_2d(_load_tensor(readers, f"{hf}.fc1.weight"), "enc_fc1")
            weights[f"{pfx}.b_fc1"] = _load_tensor(readers, f"{hf}.fc1.bias").astype(np.float32)
            weights[f"{pfx}.w_fc2"] = _transpose_2d(_load_tensor(readers, f"{hf}.fc2.weight"), "enc_fc2")
            weights[f"{pfx}.b_fc2"] = _load_tensor(readers, f"{hf}.fc2.bias").astype(np.float32)
            weights[f"{pfx}.ffn_norm"] = _load_tensor(readers, f"{hf}.final_layer_norm.weight").astype(np.float32)
            weights[f"{pfx}.ffn_norm_beta"] = _load_tensor(readers, f"{hf}.final_layer_norm.bias").astype(np.float32)

        # Decoder position embeddings
        for key in ("decoder.embed_positions.weight", "model.decoder.embed_positions.weight"):
            if _has_tensor(readers, key):
                weights["dec_pos_embedding"] = _load_tensor(readers, key).astype(np.float32)
                break
        if "dec_pos_embedding" not in weights:
            raise RuntimeError("BART: cannot find decoder position embeddings")

        # Decoder layernorm_embedding
        if normalize_embedding:
            for prefix in ("decoder", "model.decoder"):
                if _has_tensor(readers, f"{prefix}.layernorm_embedding.weight"):
                    weights["dec_embed_norm"] = _load_tensor(readers, f"{prefix}.layernorm_embedding.weight").astype(np.float32)
                    weights["dec_embed_norm_beta"] = _load_tensor(readers, f"{prefix}.layernorm_embedding.bias").astype(np.float32)
                    break

        # Decoder layers
        for i in range(dec_layers):
            hf = f"decoder.layers.{i}"
            if not _has_tensor(readers, f"{hf}.self_attn.q_proj.weight"):
                hf = f"model.decoder.layers.{i}"
            pfx = f"layer.{i}"
            for proj in ("q", "k", "v"):
                weights[f"{pfx}.w_{proj}"] = _transpose_2d(_load_tensor(readers, f"{hf}.self_attn.{proj}_proj.weight"), f"dec_{proj}")
                weights[f"{pfx}.{proj}_bias"] = _load_tensor(readers, f"{hf}.self_attn.{proj}_proj.bias").astype(np.float32)
            weights[f"{pfx}.w_o"] = _transpose_2d(_load_tensor(readers, f"{hf}.self_attn.out_proj.weight"), "dec_o")
            weights[f"{pfx}.o_bias"] = _load_tensor(readers, f"{hf}.self_attn.out_proj.bias").astype(np.float32)
            weights[f"{pfx}.input_norm"] = _load_tensor(readers, f"{hf}.self_attn_layer_norm.weight").astype(np.float32)
            weights[f"{pfx}.input_norm_beta"] = _load_tensor(readers, f"{hf}.self_attn_layer_norm.bias").astype(np.float32)
            for proj in ("q", "k", "v"):
                weights[f"{pfx}.cross_w_{proj}"] = _transpose_2d(_load_tensor(readers, f"{hf}.encoder_attn.{proj}_proj.weight"), f"xattn_{proj}")
                weights[f"{pfx}.cross_b_{proj}"] = _load_tensor(readers, f"{hf}.encoder_attn.{proj}_proj.bias").astype(np.float32)
            weights[f"{pfx}.cross_w_o"] = _transpose_2d(_load_tensor(readers, f"{hf}.encoder_attn.out_proj.weight"), "xattn_o")
            weights[f"{pfx}.cross_b_o"] = _load_tensor(readers, f"{hf}.encoder_attn.out_proj.bias").astype(np.float32)
            weights[f"{pfx}.cross_attn_norm"] = _load_tensor(readers, f"{hf}.encoder_attn_layer_norm.weight").astype(np.float32)
            weights[f"{pfx}.cross_attn_norm_beta"] = _load_tensor(readers, f"{hf}.encoder_attn_layer_norm.bias").astype(np.float32)
            weights[f"{pfx}.w_fc1"] = _transpose_2d(_load_tensor(readers, f"{hf}.fc1.weight"), "dec_fc1")
            weights[f"{pfx}.fc1_bias"] = _load_tensor(readers, f"{hf}.fc1.bias").astype(np.float32)
            weights[f"{pfx}.w_fc2"] = _transpose_2d(_load_tensor(readers, f"{hf}.fc2.weight"), "dec_fc2")
            weights[f"{pfx}.fc2_bias"] = _load_tensor(readers, f"{hf}.fc2.bias").astype(np.float32)
            weights[f"{pfx}.post_attn_norm"] = _load_tensor(readers, f"{hf}.final_layer_norm.weight").astype(np.float32)
            weights[f"{pfx}.post_attn_norm_beta"] = _load_tensor(readers, f"{hf}.final_layer_norm.bias").astype(np.float32)

        # LM head
        if _has_tensor(readers, "lm_head.weight"):
            weights["w_out"] = _transpose_2d(_load_tensor(readers, "lm_head.weight"), "lm_head")
        else:
            weights["w_out"] = _transpose_2d(shared_embed.copy(), "embedding_tied")

        return weights

    def build_engine(self, config, weights, max_cache_length, *, verbose=False, debug_layer_outputs=False):
        self._max_cache_length = max_cache_length
        dec_layers = weights["_dec_layers"]; dec_heads = weights["_dec_heads"]
        dec_ffn = weights["_dec_ffn"]; normalize_embedding = weights["_normalize_embedding"]
        hidden = config.hidden_size; vocab = config.vocab_size
        head_dim = hidden // dec_heads; attention_window = max_cache_length + 1
        max_enc_seq = max_cache_length

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger); network = builder.create_network()
        trt_config = builder.create_builder_config()
        trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        trt_config.clear_flag(trt.BuilderFlag.TF32)

        token_id = network.add_input("token_id", trt.int32, (1,))
        position_id = network.add_input("position_id", trt.int32, (1,))
        attention_mask = network.add_input("attention_mask", trt.float32, (attention_window,))

        cache_k_inputs, cache_v_inputs = [], []
        for i in range(dec_layers):
            cache_k_inputs.append(network.add_input(graph_ops.layer_tensor_name("cache_k", i), trt.float32, (max_cache_length, hidden)))
            cache_v_inputs.append(network.add_input(graph_ops.layer_tensor_name("cache_v", i), trt.float32, (max_cache_length, hidden)))

        cross_k_inputs, cross_v_inputs = [], []
        for i in range(dec_layers):
            cross_k_inputs.append(network.add_input(graph_ops.layer_tensor_name("cross_k", i), trt.float32, (max_enc_seq, hidden)))
            cross_v_inputs.append(network.add_input(graph_ops.layer_tensor_name("cross_v", i), trt.float32, (max_enc_seq, hidden)))

        embedding_table = graph_ops.add_constant(network, (vocab, hidden), weights["shared_embedding"])
        pos_embed_np = weights["dec_pos_embedding"]
        pos_embedding_table = graph_ops.add_constant(network, pos_embed_np.shape, pos_embed_np)
        eps_tensor = graph_ops.add_constant(network, (1, 1), np.array([config.rms_norm_eps], dtype=np.float32))
        attn_scale_tensor = graph_ops.add_constant(network, (1, 1, 1), np.array([1.0 / np.sqrt(max(head_dim, 1))], dtype=np.float32))

        tok_embed = network.add_gather(embedding_table, token_id, 0).get_output(0)
        # Position offset=2 for BART
        offset_weights = trt.Weights(np.array([2], dtype=np.int32))
        offset_layer = network.add_constant((1,), offset_weights)
        offset_layer.get_output(0).dtype = trt.int32
        offset_const = offset_layer.get_output(0)
        offset_pos = network.add_elementwise(position_id, offset_const, trt.ElementWiseOperation.SUM).get_output(0)
        pos_embed = network.add_gather(pos_embedding_table, offset_pos, 0).get_output(0)
        hidden_state = network.add_elementwise(tok_embed, pos_embed, trt.ElementWiseOperation.SUM).get_output(0)

        if normalize_embedding:
            hidden_state = graph_ops.add_layer_norm(network, hidden_state, hidden, weights["dec_embed_norm"], weights["dec_embed_norm_beta"], eps_tensor)

        if debug_layer_outputs:
            _mark_debug_output(network, hidden_state, "debug_embed")

        present_k_outputs, present_v_outputs = [], []
        for layer_idx in range(dec_layers):
            prefix = f"layer.{layer_idx}"
            result = _add_bart_decoder_layer(
                network=network, hidden=hidden_state,
                cache_k=cache_k_inputs[layer_idx], cache_v=cache_v_inputs[layer_idx],
                cross_k=cross_k_inputs[layer_idx], cross_v=cross_v_inputs[layer_idx],
                attention_mask=attention_mask, attn_scale_tensor=attn_scale_tensor,
                eps_tensor=eps_tensor, weights=weights, prefix=prefix,
                hidden_size=hidden, num_heads=dec_heads, head_dim=head_dim,
                ffn_dim=dec_ffn, max_cache_length=max_cache_length, max_enc_seq=max_enc_seq)
            hidden_state = result["hidden"]
            present_k_outputs.append(result["present_k"])
            present_v_outputs.append(result["present_v"])
            if debug_layer_outputs:
                _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")

        logits = graph_ops.add_matmul_rhs_constant(network, hidden_state, hidden, vocab, weights["w_out"])
        logits = graph_ops.add_bias_sum(network, logits, vocab, np.zeros(vocab, dtype=np.float32))
        logits.name = "logits"; network.mark_output(logits)

        for i in range(dec_layers):
            present_k_outputs[i].name = graph_ops.layer_tensor_name("present_k", i)
            present_v_outputs[i].name = graph_ops.layer_tensor_name("present_v", i)
            network.mark_output(present_k_outputs[i]); network.mark_output(present_v_outputs[i])

        if verbose:
            print(f"[trtf-build] Building BART decoder ({dec_layers}L, h={hidden}, heads={dec_heads}, ffn={dec_ffn}, cache={max_cache_length})", file=sys.stderr)
        plan = builder.build_serialized_network(network, trt_config)
        if plan is None: raise RuntimeError("TensorRT decoder engine build failed")
        return bytes(plan)

    def build_vision_engine(self, model_dir, config, weights, *, verbose=False):
        mcl = getattr(self, '_max_cache_length', 256)
        return _build_bart_encoder(config, weights, max_cache_length=mcl, verbose=verbose)

    def get_vl_config(self, config):
        raw = config.raw
        return {
            "encoder_layers": raw.get("encoder_layers", config.num_hidden_layers),
            "decoder_layers": raw.get("decoder_layers", config.num_hidden_layers),
            "encoder_attention_heads": raw.get("encoder_attention_heads", config.num_attention_heads),
            "decoder_attention_heads": raw.get("decoder_attention_heads", config.num_attention_heads),
            "encoder_ffn_dim": raw.get("encoder_ffn_dim", config.intermediate_size),
            "decoder_ffn_dim": raw.get("decoder_ffn_dim", config.intermediate_size),
            "max_position_embeddings": raw.get("max_position_embeddings", 1024),
            "has_vision_engine": True,
            "is_encoder_decoder": True,
            "decoder_start_token_id": raw.get("decoder_start_token_id", 2),
            "forced_bos_token_id": raw.get("forced_bos_token_id", 0),
            "position_embedding_offset": 2,
        }


def _build_bart_encoder(config, weights, *, max_cache_length=256, verbose=False):
    enc_layers = weights["_enc_layers"]; enc_heads = weights["_enc_heads"]
    enc_ffn = weights["_enc_ffn"]; max_pos = weights["_max_position_embeddings"]
    normalize_embedding = weights["_normalize_embedding"]
    hidden = config.hidden_size; vocab = config.vocab_size
    max_enc_seq = max_cache_length

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger); network = builder.create_network()
    tc = builder.create_builder_config()
    tc.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    tc.clear_flag(trt.BuilderFlag.TF32)

    eps_tensor = graph_ops.add_constant(network, (1, 1), np.array([config.rms_norm_eps], dtype=np.float32))
    input_ids = network.add_input("input_ids", trt.int32, (max_enc_seq,))
    attention_mask = network.add_input("attention_mask", trt.float32, (max_enc_seq,))

    embedding_table = graph_ops.add_constant(network, (vocab, hidden), weights["shared_embedding"])
    enc_pos_np = weights["enc_pos_embedding"]
    pos_embedding_table = graph_ops.add_constant(network, enc_pos_np.shape, enc_pos_np)

    tok_embed = network.add_gather(embedding_table, input_ids, 0).get_output(0)
    # Position indices [2, 3, ..., max_enc_seq+1] for offset=2
    pos_indices = np.arange(2, max_enc_seq + 2, dtype=np.int32)
    pos_idx_layer = network.add_constant((max_enc_seq,), trt.Weights(pos_indices))
    pos_idx_layer.get_output(0).dtype = trt.int32
    pos_indices_const = pos_idx_layer.get_output(0)
    pos_embed = network.add_gather(pos_embedding_table, pos_indices_const, 0).get_output(0)

    hs = network.add_elementwise(tok_embed, pos_embed, trt.ElementWiseOperation.SUM).get_output(0)

    if normalize_embedding:
        hs = graph_ops.add_layer_norm(network, hs, hidden, weights["enc_embed_norm"], weights["enc_embed_norm_beta"], eps_tensor)

    # Reshape attention mask [max_enc_seq] -> [1, 1, max_enc_seq] for broadcasting
    enc_mask_3d = network.add_shuffle(attention_mask)
    enc_mask_3d.reshape_dims = (1, 1, max_enc_seq)
    head_dim = hidden // enc_heads
    attn_scale_np = np.array([1.0 / np.sqrt(max(head_dim, 1))], dtype=np.float32)
    attn_scale_c = graph_ops.add_constant(network, (1, 1, 1), attn_scale_np)

    for li in range(enc_layers):
        pfx = f"enc_layer.{li}"
        # Post-norm BART encoder: self-attention with padding mask
        q = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(network, hs, hidden, hidden, weights[f"{pfx}.w_q"]), hidden, weights[f"{pfx}.b_q"])
        k = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(network, hs, hidden, hidden, weights[f"{pfx}.w_k"]), hidden, weights[f"{pfx}.b_k"])
        v = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(network, hs, hidden, hidden, weights[f"{pfx}.w_v"]), hidden, weights[f"{pfx}.b_v"])
        q_h = network.add_shuffle(q); q_h.reshape_dims = (max_enc_seq, enc_heads, head_dim); q_h.second_transpose = trt.Permutation([1, 0, 2])
        k_h = network.add_shuffle(k); k_h.reshape_dims = (max_enc_seq, enc_heads, head_dim); k_h.second_transpose = trt.Permutation([1, 0, 2])
        v_h = network.add_shuffle(v); v_h.reshape_dims = (max_enc_seq, enc_heads, head_dim); v_h.second_transpose = trt.Permutation([1, 0, 2])
        score = network.add_matrix_multiply(q_h.get_output(0), trt.MatrixOperation.NONE, k_h.get_output(0), trt.MatrixOperation.TRANSPOSE)
        scaled = network.add_elementwise(score.get_output(0), attn_scale_c, trt.ElementWiseOperation.PROD)
        masked = network.add_elementwise(scaled.get_output(0), enc_mask_3d.get_output(0), trt.ElementWiseOperation.SUM)
        sm = network.add_softmax(masked.get_output(0)); sm.axes = 1 << 2
        ctx = network.add_matrix_multiply(sm.get_output(0), trt.MatrixOperation.NONE, v_h.get_output(0), trt.MatrixOperation.NONE)
        ctx_flat = network.add_shuffle(ctx.get_output(0)); ctx_flat.first_transpose = trt.Permutation([1, 0, 2]); ctx_flat.reshape_dims = (max_enc_seq, hidden)
        attn = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(network, ctx_flat.get_output(0), hidden, hidden, weights[f"{pfx}.w_o"]), hidden, weights[f"{pfx}.b_o"])
        hs = network.add_elementwise(hs, attn, trt.ElementWiseOperation.SUM).get_output(0)
        hs = graph_ops.add_layer_norm(network, hs, hidden, weights[f"{pfx}.attn_norm"], weights[f"{pfx}.attn_norm_beta"], eps_tensor)

        fc1 = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(network, hs, hidden, enc_ffn, weights[f"{pfx}.w_fc1"]), enc_ffn, weights[f"{pfx}.b_fc1"])
        act = graph_ops.add_activation(network, fc1, "gelu_new")
        fc2 = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(network, act, enc_ffn, hidden, weights[f"{pfx}.w_fc2"]), hidden, weights[f"{pfx}.b_fc2"])
        hs = network.add_elementwise(hs, fc2, trt.ElementWiseOperation.SUM).get_output(0)
        hs = graph_ops.add_layer_norm(network, hs, hidden, weights[f"{pfx}.ffn_norm"], weights[f"{pfx}.ffn_norm_beta"], eps_tensor)

    hs.name = "encoder_output"; network.mark_output(hs)
    if verbose:
        print(f"[trtf-build] Building BART encoder ({enc_layers}L, h={hidden}, heads={enc_heads}, seq={max_enc_seq})", file=sys.stderr)
    plan = builder.build_serialized_network(network, tc)
    if plan is None: raise RuntimeError("TensorRT encoder engine build failed")
    return bytes(plan)


def _add_bart_decoder_layer(*, network, hidden, cache_k, cache_v, cross_k, cross_v,
    attention_mask, attn_scale_tensor, eps_tensor, weights, prefix,
    hidden_size, num_heads, head_dim, ffn_dim, max_cache_length, max_enc_seq):
    attention_size = hidden_size; attention_window = max_cache_length + 1

    # Self-attention (no pre-norm for post-LN BART)
    q = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(network, hidden, hidden_size, attention_size, weights[f"{prefix}.w_q"]), attention_size, weights[f"{prefix}.q_bias"])
    k = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(network, hidden, hidden_size, attention_size, weights[f"{prefix}.w_k"]), attention_size, weights[f"{prefix}.k_bias"])
    v = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(network, hidden, hidden_size, attention_size, weights[f"{prefix}.w_v"]), attention_size, weights[f"{prefix}.v_bias"])
    present_k, present_v = k, v

    kr = network.add_shuffle(k); kr.reshape_dims = (1, attention_size)
    vr = network.add_shuffle(v); vr.reshape_dims = (1, attention_size)
    ak = network.add_concatenation([cache_k, kr.get_output(0)]); ak.axis = 0
    av = network.add_concatenation([cache_v, vr.get_output(0)]); av.axis = 0

    qh = network.add_shuffle(q); qh.reshape_dims = (num_heads, 1, head_dim)
    kh = network.add_shuffle(ak.get_output(0)); kh.reshape_dims = (attention_window, num_heads, head_dim); kh.second_transpose = trt.Permutation([1, 0, 2])
    vh = network.add_shuffle(av.get_output(0)); vh.reshape_dims = (attention_window, num_heads, head_dim); vh.second_transpose = trt.Permutation([1, 0, 2])

    sc = network.add_elementwise(network.add_matrix_multiply(qh.get_output(0), trt.MatrixOperation.NONE, kh.get_output(0), trt.MatrixOperation.TRANSPOSE).get_output(0), attn_scale_tensor, trt.ElementWiseOperation.PROD)
    m3 = network.add_shuffle(attention_mask); m3.reshape_dims = (1, 1, attention_window)
    ma = network.add_elementwise(sc.get_output(0), m3.get_output(0), trt.ElementWiseOperation.SUM)
    sm = network.add_softmax(ma.get_output(0)); sm.axes = 1 << 2
    ct = network.add_matrix_multiply(sm.get_output(0), trt.MatrixOperation.NONE, vh.get_output(0), trt.MatrixOperation.NONE)
    cf = network.add_shuffle(ct.get_output(0)); cf.reshape_dims = (1, attention_size)
    sa = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(network, cf.get_output(0), attention_size, hidden_size, weights[f"{prefix}.w_o"]), hidden_size, weights[f"{prefix}.o_bias"])
    # Residual + post-norm
    psa = network.add_elementwise(hidden, sa, trt.ElementWiseOperation.SUM).get_output(0)
    psa = graph_ops.add_layer_norm(network, psa, hidden_size, weights[f"{prefix}.input_norm"], weights[f"{prefix}.input_norm_beta"], eps_tensor)

    # Cross-attention (no pre-norm)
    cq = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(network, psa, hidden_size, attention_size, weights[f"{prefix}.cross_w_q"]), attention_size, weights[f"{prefix}.cross_b_q"])
    ck_proj = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(network, cross_k, hidden_size, attention_size, weights[f"{prefix}.cross_w_k"]), attention_size, weights[f"{prefix}.cross_b_k"])
    cv_proj = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(network, cross_v, hidden_size, attention_size, weights[f"{prefix}.cross_w_v"]), attention_size, weights[f"{prefix}.cross_b_v"])

    cqh = network.add_shuffle(cq); cqh.reshape_dims = (num_heads, 1, head_dim)
    ckh = network.add_shuffle(ck_proj); ckh.reshape_dims = (max_enc_seq, num_heads, head_dim); ckh.second_transpose = trt.Permutation([1, 0, 2])
    cvh = network.add_shuffle(cv_proj); cvh.reshape_dims = (max_enc_seq, num_heads, head_dim); cvh.second_transpose = trt.Permutation([1, 0, 2])

    cs = network.add_elementwise(network.add_matrix_multiply(cqh.get_output(0), trt.MatrixOperation.NONE, ckh.get_output(0), trt.MatrixOperation.TRANSPOSE).get_output(0), attn_scale_tensor, trt.ElementWiseOperation.PROD)
    csm = network.add_softmax(cs.get_output(0)); csm.axes = 1 << 2
    cc = network.add_matrix_multiply(csm.get_output(0), trt.MatrixOperation.NONE, cvh.get_output(0), trt.MatrixOperation.NONE)
    ccf = network.add_shuffle(cc.get_output(0)); ccf.reshape_dims = (1, attention_size)
    ca = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(network, ccf.get_output(0), attention_size, hidden_size, weights[f"{prefix}.cross_w_o"]), hidden_size, weights[f"{prefix}.cross_b_o"])
    # Residual + post-norm
    pca = network.add_elementwise(psa, ca, trt.ElementWiseOperation.SUM).get_output(0)
    pca = graph_ops.add_layer_norm(network, pca, hidden_size, weights[f"{prefix}.cross_attn_norm"], weights[f"{prefix}.cross_attn_norm_beta"], eps_tensor)

    # MLP (no pre-norm, GELU)
    fc1 = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(network, pca, hidden_size, ffn_dim, weights[f"{prefix}.w_fc1"]), ffn_dim, weights[f"{prefix}.fc1_bias"])
    act = graph_ops.add_activation(network, fc1, "gelu_new")
    fc2 = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(network, act, ffn_dim, hidden_size, weights[f"{prefix}.w_fc2"]), hidden_size, weights[f"{prefix}.fc2_bias"])
    # Residual + post-norm
    out = network.add_elementwise(pca, fc2, trt.ElementWiseOperation.SUM).get_output(0)
    out = graph_ops.add_layer_norm(network, out, hidden_size, weights[f"{prefix}.post_attn_norm"], weights[f"{prefix}.post_attn_norm_beta"], eps_tensor)

    return {"hidden": out, "present_k": present_k, "present_v": present_v}


def _mark_debug_output(network, tensor, name):
    identity = network.add_identity(tensor); out = identity.get_output(0)
    out.name = name; network.mark_output(out); out.dtype = trt.float32


plugin = BartPlugin()
