"""Tests for standard_decoder_builder.py — tensor naming contract.

Builds tiny engines and verifies all I/O tensor names match C++ expectations.
Requires TRT + GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("trtf_build", reason="trtf_build requires tensorrt")
from tests.builder.conftest import requires_trt


def _make_weights(hidden: int, vocab: int, num_layers: int,
                  attention_size: int, mlp_size: int,
                  *, mlp_type: str = "swiglu",
                  position_type: str = "rope",
                  has_bias: bool = False) -> dict:
    """Create a minimal synthetic weight dict for the standard decoder builder."""
    from trtf_build.checkpoint_mapper import WeightDict
    rng = np.random.RandomState(42)
    w = WeightDict()
    w["embedding"] = rng.randn(vocab, hidden).astype(np.float32)

    for i in range(num_layers):
        p = f"layer.{i}"
        w[f"{p}.input_norm"] = rng.randn(hidden).astype(np.float32)
        w[f"{p}.post_attn_norm"] = rng.randn(hidden).astype(np.float32)
        w[f"{p}.w_q"] = rng.randn(hidden, attention_size).astype(np.float32)
        w[f"{p}.w_k"] = rng.randn(hidden, attention_size).astype(np.float32)
        w[f"{p}.w_v"] = rng.randn(hidden, attention_size).astype(np.float32)
        w[f"{p}.w_o"] = rng.randn(attention_size, hidden).astype(np.float32)

        if mlp_type == "swiglu":
            w[f"{p}.w_gate"] = rng.randn(hidden, mlp_size).astype(np.float32)
            w[f"{p}.w_up"] = rng.randn(hidden, mlp_size).astype(np.float32)
            w[f"{p}.w_down"] = rng.randn(mlp_size, hidden).astype(np.float32)
        else:  # gelu_fc
            w[f"{p}.w_fc1"] = rng.randn(hidden, mlp_size).astype(np.float32)
            w[f"{p}.w_fc2"] = rng.randn(mlp_size, hidden).astype(np.float32)

    w["final_norm"] = rng.randn(hidden).astype(np.float32)
    w["w_out"] = rng.randn(hidden, vocab).astype(np.float32)
    w["_attention_size"] = attention_size
    w["_mlp_size"] = mlp_size

    if position_type == "learned":
        max_pos = 64
        w["position_embedding"] = rng.randn(max_pos, hidden).astype(np.float32)

    return w


def _get_io_names(engine_plan: bytes) -> tuple[list[str], list[str]]:
    """Deserialize engine plan and return (input_names, output_names)."""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_plan)
    inputs, outputs = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            inputs.append(name)
        else:
            outputs.append(name)
    return inputs, outputs


@requires_trt
class TestTensorNamingContract:
    """Verify that built engines have the exact I/O tensor names the C++ runtime expects."""

    def _build_engine(self, **kwargs):
        from trtf_build.config import ModelConfig
        from trtf_build.standard_decoder_builder import build_standard_decoder_engine

        hidden, vocab, num_layers = 16, 32, 2
        num_heads = 4
        attention_size = hidden
        mlp_size = 32
        max_cache = 4

        config = ModelConfig(
            hidden_size=hidden,
            vocab_size=vocab,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
        )
        mlp_type = kwargs.get("mlp_type", "swiglu")
        position_type = kwargs.get("position_type", "rope")
        weights = _make_weights(
            hidden, vocab, num_layers, attention_size, mlp_size,
            mlp_type=mlp_type, position_type=position_type)

        return build_standard_decoder_engine(
            config, weights, max_cache, **kwargs)

    def test_default_rope_swiglu(self):
        """Default: RoPE + SwiGLU, standard I/O names."""
        plan = self._build_engine()
        inputs, outputs = _get_io_names(plan)

        assert "token_id" in inputs
        assert "position_id" in inputs
        assert "attention_mask" in inputs
        assert "cache_k_0" in inputs
        assert "cache_k_1" in inputs
        assert "cache_v_0" in inputs
        assert "cache_v_1" in inputs

        assert "logits" in outputs
        assert "present_k_0" in outputs
        assert "present_k_1" in outputs
        assert "present_v_0" in outputs
        assert "present_v_1" in outputs

    def test_layernorm_gelu_fc(self):
        """LayerNorm + gelu_fc MLP, same I/O names."""
        plan = self._build_engine(
            norm_type="layernorm", mlp_type="gelu_fc", activation="gelu_new")
        inputs, outputs = _get_io_names(plan)

        assert "token_id" in inputs
        assert "logits" in outputs
        assert "present_k_0" in outputs

    def test_learned_positions(self):
        """Learned position embeddings, same I/O names."""
        plan = self._build_engine(position_type="learned")
        inputs, outputs = _get_io_names(plan)

        assert "token_id" in inputs
        assert "position_id" in inputs
        assert "logits" in outputs

    def test_alibi_positions(self):
        """ALiBi positions, same I/O names."""
        plan = self._build_engine(position_type="alibi")
        inputs, outputs = _get_io_names(plan)

        assert "token_id" in inputs
        assert "position_id" in inputs
        assert "logits" in outputs

    def test_embed_input(self):
        """With embed_input=True, extra VL inputs appear."""
        plan = self._build_engine(embed_input=True)
        inputs, outputs = _get_io_names(plan)

        assert "input_embed" in inputs
        assert "use_input_embed" in inputs
        assert "token_id" in inputs
        assert "logits" in outputs

    def test_debug_layer_outputs(self):
        """With debug_layer_outputs=True, per-layer debug outputs appear."""
        plan = self._build_engine(debug_layer_outputs=True)
        inputs, outputs = _get_io_names(plan)

        assert "debug_embed" in outputs
        assert "debug_hidden_0" in outputs
        assert "debug_hidden_1" in outputs
        assert "debug_post_attn_0" in outputs
        assert "debug_post_attn_1" in outputs
        assert "logits" in outputs

    def test_interleaved_rope(self):
        plan = self._build_engine(interleaved_rope=True)
        inputs, outputs = _get_io_names(plan)
        assert "logits" in outputs

    def test_partial_rotary(self):
        plan = self._build_engine(partial_rotary_factor=0.5)
        inputs, outputs = _get_io_names(plan)
        assert "logits" in outputs
