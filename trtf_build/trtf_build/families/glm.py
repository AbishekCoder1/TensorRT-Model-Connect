"""GLM-4 family plugin — handles fused gate_up_proj splitting."""

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


class GlmPlugin:
    name = "glm"

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "glm"

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        """Load GLM-4 weights, splitting fused gate_up_proj."""
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

            # Norms (1D, no transpose)
            input_norm = _load_tensor(
                readers, f"{hf_prefix}.input_layernorm.weight")
            post_norm = _load_tensor(
                readers, f"{hf_prefix}.post_attention_layernorm.weight")
            weights[f"{prefix}.input_norm"] = input_norm.astype(np.float32)
            weights[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)

            # ---- Separate Q/K/V projections ----
            q_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.q_proj.weight")
            k_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.k_proj.weight")
            v_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.v_proj.weight")

            if attention_size == 0:
                attention_size = q_raw.shape[0]

            # Transpose [out, in] -> [in, out]
            q_t = _transpose_2d(q_raw, "q_proj")
            k_t = _transpose_2d(k_raw, "k_proj")
            v_t = _transpose_2d(v_raw, "v_proj")

            # GQA expansion for K, V
            k_expanded = _expand_kv_projection(
                k_t, hidden, kv_dim, q_dim, num_heads, num_kv_heads)
            v_expanded = _expand_kv_projection(
                v_t, hidden, kv_dim, q_dim, num_heads, num_kv_heads)

            weights[f"{prefix}.w_q"] = q_t
            weights[f"{prefix}.w_k"] = k_expanded
            weights[f"{prefix}.w_v"] = v_expanded

            # Q/K/V biases (GLM-4 has biases on Q, K, V but NOT O)
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
                    group_size = num_heads // num_kv_heads
                    for qh in range(num_heads):
                        kvh = min(num_kv_heads - 1, qh // group_size)
                        expanded[qh * head_dim:(qh + 1) * head_dim] = \
                            raw[kvh * head_dim:(kvh + 1) * head_dim]
                    weights[f"{prefix}.k_bias"] = expanded
                else:
                    weights[f"{prefix}.k_bias"] = raw
            if _has_tensor(readers, v_bias_key):
                raw = _load_tensor(readers, v_bias_key).astype(np.float32)
                if raw.shape[0] == kv_dim and kv_dim != q_dim:
                    expanded = np.zeros(q_dim, dtype=np.float32)
                    group_size = num_heads // num_kv_heads
                    for qh in range(num_heads):
                        kvh = min(num_kv_heads - 1, qh // group_size)
                        expanded[qh * head_dim:(qh + 1) * head_dim] = \
                            raw[kvh * head_dim:(kvh + 1) * head_dim]
                    weights[f"{prefix}.v_bias"] = expanded
                else:
                    weights[f"{prefix}.v_bias"] = raw

            # Output projection (no bias in GLM-4)
            o_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.o_proj.weight")
            weights[f"{prefix}.w_o"] = _transpose_2d(o_raw, "o_proj")

            # ---- Fused gate_up projection ----
            # Shape: [2 * intermediate_size, hidden]
            gate_up_raw = _load_tensor(
                readers, f"{hf_prefix}.mlp.gate_up_proj.weight")
            intermediate = gate_up_raw.shape[0] // 2
            if mlp_size == 0:
                mlp_size = intermediate

            gate_raw = gate_up_raw[:intermediate, :]
            up_raw = gate_up_raw[intermediate:, :]
            del gate_up_raw

            weights[f"{prefix}.w_gate"] = _transpose_2d(gate_raw, "gate_proj")
            weights[f"{prefix}.w_up"] = _transpose_2d(up_raw, "up_proj")
            del gate_raw, up_raw

            # Down projection
            down_raw = _load_tensor(
                readers, f"{hf_prefix}.mlp.down_proj.weight")
            weights[f"{prefix}.w_down"] = _transpose_2d(down_raw, "down_proj")

        # Final norm
        final_norm_key = "model.norm.weight"
        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = _load_tensor(
                readers, final_norm_key).astype(np.float32)
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

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
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
        debug_layer_outputs: bool = False,
    ) -> bytes:
        # GLM-4 uses partial RoPE (default 0.5) with interleaved layout.
        partial_rotary_factor = config.raw.get("partial_rotary_factor", 0.5)
        return build_standard_decoder_engine(
            config, weights, max_cache_length,
            precision=precision, quant_ctx=quant_ctx,
            partial_rotary_factor=partial_rotary_factor,
            interleaved_rope=True,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs)


plugin = GlmPlugin()
