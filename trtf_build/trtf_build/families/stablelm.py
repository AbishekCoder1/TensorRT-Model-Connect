"""StableLM-2 family plugin — LayerNorm + SwiGLU + RoPE.

StableLM-2 (Stability AI) uses:
  - LayerNorm (with beta) instead of RMSNorm
  - SwiGLU MLP (gate/up/down) — same as LLaMA
  - RoPE for positional encoding
  - GQA (grouped query attention)
  - QKV biases
  - Standard LLaMA-like weight key patterns (model.layers.N.*)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import ModelConfig
from ..checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
    _expand_kv_projection,
)
from ..standard_decoder_builder import build_standard_decoder_engine


class StableLMPlugin:
    name = "stablelm"

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        return mt == "stablelm" or mt.startswith("stablelm")

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim

        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        weights = WeightDict()

        # Embedding
        embedding = _load_tensor(readers, "model.embed_tokens.weight")
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})")
        weights["embedding"] = embedding.astype(np.float32)

        mlp_size = 0
        attention_size = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"model.layers.{layer_idx}"

            # LayerNorm weights + biases
            input_norm = _load_tensor(
                readers, f"{hf_prefix}.input_layernorm.weight")
            input_norm_beta = _load_tensor(
                readers, f"{hf_prefix}.input_layernorm.bias")
            post_norm = _load_tensor(
                readers, f"{hf_prefix}.post_attention_layernorm.weight")
            post_norm_beta = _load_tensor(
                readers, f"{hf_prefix}.post_attention_layernorm.bias")

            weights[f"{prefix}.input_norm"] = input_norm.astype(np.float32)
            weights[f"{prefix}.input_norm_beta"] = input_norm_beta.astype(np.float32)
            weights[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)
            weights[f"{prefix}.post_attn_norm_beta"] = post_norm_beta.astype(np.float32)

            # Q/K/V projections (separate)
            q_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.q_proj.weight")
            k_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.k_proj.weight")
            v_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.v_proj.weight")
            o_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.o_proj.weight")

            if attention_size == 0:
                attention_size = q_raw.shape[0]

            q_t = _transpose_2d(q_raw, "q_proj")
            k_t = _transpose_2d(k_raw, "k_proj")
            v_t = _transpose_2d(v_raw, "v_proj")
            o_t = _transpose_2d(o_raw, "o_proj")

            # GQA expansion for K, V
            k_expanded = _expand_kv_projection(
                k_t, hidden, kv_dim, q_dim, num_heads, num_kv_heads)
            v_expanded = _expand_kv_projection(
                v_t, hidden, kv_dim, q_dim, num_heads, num_kv_heads)

            weights[f"{prefix}.w_q"] = q_t
            weights[f"{prefix}.w_k"] = k_expanded
            weights[f"{prefix}.w_v"] = v_expanded
            weights[f"{prefix}.w_o"] = o_t

            # Optional QKV biases
            q_bias_key = f"{hf_prefix}.self_attn.q_proj.bias"
            k_bias_key = f"{hf_prefix}.self_attn.k_proj.bias"
            v_bias_key = f"{hf_prefix}.self_attn.v_proj.bias"
            if _has_tensor(readers, q_bias_key):
                weights[f"{prefix}.q_bias"] = _load_tensor(
                    readers, q_bias_key).astype(np.float32)
            if _has_tensor(readers, k_bias_key):
                raw = _load_tensor(readers, k_bias_key).astype(np.float32)
                if raw.shape[0] == kv_dim and kv_dim != q_dim:
                    expanded = np.zeros(q_dim, dtype=np.float32)
                    for qh in range(num_heads):
                        kvh = min(num_kv_heads - 1,
                                  qh // (num_heads // num_kv_heads))
                        expanded[qh * head_dim:(qh + 1) * head_dim] = \
                            raw[kvh * head_dim:(kvh + 1) * head_dim]
                    weights[f"{prefix}.k_bias"] = expanded
                else:
                    weights[f"{prefix}.k_bias"] = raw
            if _has_tensor(readers, v_bias_key):
                raw = _load_tensor(readers, v_bias_key).astype(np.float32)
                if raw.shape[0] == kv_dim and kv_dim != q_dim:
                    expanded = np.zeros(q_dim, dtype=np.float32)
                    for qh in range(num_heads):
                        kvh = min(num_kv_heads - 1,
                                  qh // (num_heads // num_kv_heads))
                        expanded[qh * head_dim:(qh + 1) * head_dim] = \
                            raw[kvh * head_dim:(kvh + 1) * head_dim]
                    weights[f"{prefix}.v_bias"] = expanded
                else:
                    weights[f"{prefix}.v_bias"] = raw

            # SwiGLU MLP (gate/up/down)
            gate_raw = _load_tensor(
                readers, f"{hf_prefix}.mlp.gate_proj.weight")
            up_raw = _load_tensor(
                readers, f"{hf_prefix}.mlp.up_proj.weight")
            down_raw = _load_tensor(
                readers, f"{hf_prefix}.mlp.down_proj.weight")
            if mlp_size == 0:
                mlp_size = gate_raw.shape[0]

            weights[f"{prefix}.w_gate"] = _transpose_2d(gate_raw, "gate_proj")
            weights[f"{prefix}.w_up"] = _transpose_2d(up_raw, "up_proj")
            weights[f"{prefix}.w_down"] = _transpose_2d(down_raw, "down_proj")

        # Final LayerNorm
        final_norm_key = "model.norm.weight"
        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = _load_tensor(
                readers, final_norm_key).astype(np.float32)
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        final_norm_beta_key = "model.norm.bias"
        if _has_tensor(readers, final_norm_beta_key):
            weights["final_norm_beta"] = _load_tensor(
                readers, final_norm_beta_key).astype(np.float32)

        # LM head
        lm_head_key = "lm_head.weight"
        if _has_tensor(readers, lm_head_key):
            weights["w_out"] = _transpose_2d(
                _load_tensor(readers, lm_head_key), "lm_head")
        else:
            weights["w_out"] = _transpose_2d(embedding.copy(), "embedding_tied")

        weights["_attention_size"] = attention_size  # type: ignore[assignment]
        weights["_mlp_size"] = mlp_size  # type: ignore[assignment]

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, verbose: bool = False,
        debug_layer_outputs: bool = False,
    ) -> bytes:
        partial_rotary = config.raw.get("partial_rotary_factor", 1.0)
        return build_standard_decoder_engine(
            config, weights, max_cache_length,
            norm_type="layernorm",
            mlp_type="swiglu",
            position_type="rope",
            partial_rotary_factor=partial_rotary,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs)


plugin = StableLMPlugin()
