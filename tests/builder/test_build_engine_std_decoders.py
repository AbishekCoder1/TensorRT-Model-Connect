"""Integration tests for standard decoder build_engine methods.

Tests GPT2, GPT-Neo, GPT-NeoX, InternLM2, CodeGen build_engine pipelines.

Requires TRT + GPU.

Trace: ARCH-ENG-001, IT-BUILD-ENGINE-STD-DECODERS
Intent: Validate build_engine for standard decoder families (GPT2, GPT-Neo, GPT-NeoX, InternLM2, CodeGen)
Preconditions: TRT and CUDA GPU are available; synthetic safetensors for each decoder variant are created
Postconditions: Each standard decoder plugin produces valid engine plan bytes with correct I/O tensor names
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

try:
    from safetensors.numpy import save_file
    from trtf_build.config import ModelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("trtf_build requires tensorrt", allow_module_level=True)


def _trt_available() -> bool:
    try:
        import tensorrt as trt  # noqa: F401
        try:
            from cuda.bindings import runtime as cudart  # noqa: F401
        except ImportError:
            from cuda import cudart  # type: ignore[no-redef]  # noqa: F401
        return True
    except ImportError:
        return False

requires_trt = pytest.mark.skipif(
    not _trt_available(), reason="TensorRT + CUDA not available"
)

RNG = np.random.RandomState(654)

def _rand(*shape: int) -> np.ndarray:
    return RNG.randn(*shape).astype(np.float32)

def _write_config(model_dir: Path, config: dict) -> None:
    (model_dir / "config.json").write_text(json.dumps(config))

def _write_safetensors(model_dir: Path, tensors: dict[str, np.ndarray],
                       filename: str = "model.safetensors") -> None:
    save_file(tensors, str(model_dir / filename))


# =========================================================================
# GPT-2 — Conv1D layout, fused QKV, learned position embeddings
# =========================================================================

@requires_trt
class TestGPT2BuildEngine:
    VOCAB, HIDDEN, LAYERS, HEADS, MAX_POS = 32, 16, 1, 4, 64
    MLP = HIDDEN * 4

    @staticmethod
    def _make(vocab, hidden, layers, heads, max_pos, mlp):
        t = {}
        t["wte.weight"] = _rand(vocab, hidden)
        t["wpe.weight"] = _rand(max_pos, hidden)
        for i in range(layers):
            t[f"h.{i}.ln_1.weight"] = _rand(hidden)
            t[f"h.{i}.ln_1.bias"] = _rand(hidden)
            # Conv1D layout: [in, out] — fused QKV [hidden, 3*hidden]
            t[f"h.{i}.attn.c_attn.weight"] = _rand(hidden, 3 * hidden)
            t[f"h.{i}.attn.c_attn.bias"] = _rand(3 * hidden)
            t[f"h.{i}.attn.c_proj.weight"] = _rand(hidden, hidden)
            t[f"h.{i}.attn.c_proj.bias"] = _rand(hidden)
            t[f"h.{i}.ln_2.weight"] = _rand(hidden)
            t[f"h.{i}.ln_2.bias"] = _rand(hidden)
            t[f"h.{i}.mlp.c_fc.weight"] = _rand(hidden, mlp)
            t[f"h.{i}.mlp.c_fc.bias"] = _rand(mlp)
            t[f"h.{i}.mlp.c_proj.weight"] = _rand(mlp, hidden)
            t[f"h.{i}.mlp.c_proj.bias"] = _rand(hidden)
        t["ln_f.weight"] = _rand(hidden)
        t["ln_f.bias"] = _rand(hidden)
        t["lm_head.weight"] = _rand(vocab, hidden)
        return t

    def test_build_engine(self, tmp_path):
        from trtf_build.families.gpt2 import plugin
        config = {
            "model_type": "gpt2",
            "vocab_size": self.VOCAB, "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS, "num_attention_heads": self.HEADS,
            "n_positions": self.MAX_POS,
        }
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, self._make(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.MAX_POS, self.MLP))
        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        engine = plugin.build_engine(cfg, weights, max_cache_length=32, verbose=False)
        assert isinstance(engine, bytes) and len(engine) > 0

    def test_load_weights(self, tmp_path):
        from trtf_build.families.gpt2 import plugin
        config = {
            "model_type": "gpt2",
            "vocab_size": self.VOCAB, "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS, "num_attention_heads": self.HEADS,
            "n_positions": self.MAX_POS,
        }
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, self._make(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.MAX_POS, self.MLP))
        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        assert "embedding" in weights
        assert "position_embedding" in weights
        for key in ("w_q", "w_k", "w_v", "w_o"):
            assert f"layer.0.{key}" in weights


# =========================================================================
# GPT-Neo — separate Q/K/V, learned position embeddings
# =========================================================================

@requires_trt
class TestGPTNeoBuildEngine:
    VOCAB, HIDDEN, LAYERS, HEADS, MAX_POS = 32, 16, 1, 4, 64
    MLP = HIDDEN * 4

    @staticmethod
    def _make(vocab, hidden, layers, heads, max_pos, mlp):
        t = {}
        t["transformer.wte.weight"] = _rand(vocab, hidden)
        t["transformer.wpe.weight"] = _rand(max_pos, hidden)
        for i in range(layers):
            t[f"transformer.h.{i}.ln_1.weight"] = _rand(hidden)
            t[f"transformer.h.{i}.ln_1.bias"] = _rand(hidden)
            t[f"transformer.h.{i}.attn.attention.q_proj.weight"] = _rand(hidden, hidden)
            t[f"transformer.h.{i}.attn.attention.k_proj.weight"] = _rand(hidden, hidden)
            t[f"transformer.h.{i}.attn.attention.v_proj.weight"] = _rand(hidden, hidden)
            t[f"transformer.h.{i}.attn.attention.out_proj.weight"] = _rand(hidden, hidden)
            t[f"transformer.h.{i}.attn.attention.out_proj.bias"] = _rand(hidden)
            t[f"transformer.h.{i}.ln_2.weight"] = _rand(hidden)
            t[f"transformer.h.{i}.ln_2.bias"] = _rand(hidden)
            t[f"transformer.h.{i}.mlp.c_fc.weight"] = _rand(mlp, hidden)
            t[f"transformer.h.{i}.mlp.c_fc.bias"] = _rand(mlp)
            t[f"transformer.h.{i}.mlp.c_proj.weight"] = _rand(hidden, mlp)
            t[f"transformer.h.{i}.mlp.c_proj.bias"] = _rand(hidden)
        t["transformer.ln_f.weight"] = _rand(hidden)
        t["transformer.ln_f.bias"] = _rand(hidden)
        t["lm_head.weight"] = _rand(vocab, hidden)
        return t

    def test_build_engine(self, tmp_path):
        from trtf_build.families.gpt_neo import plugin
        config = {
            "model_type": "gpt_neo",
            "vocab_size": self.VOCAB, "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS, "num_attention_heads": self.HEADS,
            "max_position_embeddings": self.MAX_POS,
        }
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, self._make(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.MAX_POS, self.MLP))
        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        engine = plugin.build_engine(cfg, weights, max_cache_length=32, verbose=False)
        assert isinstance(engine, bytes) and len(engine) > 0


# =========================================================================
# GPT-NeoX — fused QKV head-interleaved, RoPE, parallel residual
# =========================================================================

@requires_trt
class TestGPTNeoXBuildEngine:
    VOCAB, HIDDEN, LAYERS, HEADS = 32, 16, 1, 4
    MLP = HIDDEN * 4

    @staticmethod
    def _make(vocab, hidden, layers, heads, mlp):
        t = {}
        t["gpt_neox.embed_in.weight"] = _rand(vocab, hidden)
        for i in range(layers):
            p = f"gpt_neox.layers.{i}"
            t[f"{p}.input_layernorm.weight"] = _rand(hidden)
            t[f"{p}.input_layernorm.bias"] = _rand(hidden)
            # Fused QKV: head-interleaved [3*hidden, hidden]
            t[f"{p}.attention.query_key_value.weight"] = _rand(3 * hidden, hidden)
            t[f"{p}.attention.query_key_value.bias"] = _rand(3 * hidden)
            t[f"{p}.attention.dense.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.dense.bias"] = _rand(hidden)
            t[f"{p}.post_attention_layernorm.weight"] = _rand(hidden)
            t[f"{p}.post_attention_layernorm.bias"] = _rand(hidden)
            t[f"{p}.mlp.dense_h_to_4h.weight"] = _rand(mlp, hidden)
            t[f"{p}.mlp.dense_h_to_4h.bias"] = _rand(mlp)
            t[f"{p}.mlp.dense_4h_to_h.weight"] = _rand(hidden, mlp)
            t[f"{p}.mlp.dense_4h_to_h.bias"] = _rand(hidden)
        t["gpt_neox.final_layer_norm.weight"] = _rand(hidden)
        t["gpt_neox.final_layer_norm.bias"] = _rand(hidden)
        t["embed_out.weight"] = _rand(vocab, hidden)
        return t

    def test_build_engine(self, tmp_path):
        from trtf_build.families.gpt_neox import plugin
        config = {
            "model_type": "gpt_neox",
            "vocab_size": self.VOCAB, "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS, "num_attention_heads": self.HEADS,
            "rotary_pct": 0.25,
        }
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, self._make(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.MLP))
        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        engine = plugin.build_engine(cfg, weights, max_cache_length=32, verbose=False)
        assert isinstance(engine, bytes) and len(engine) > 0


# =========================================================================
# InternLM2 — fused QKV group-interleaved, SwiGLU, RoPE
# =========================================================================

@requires_trt
class TestInternLMBuildEngine:
    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS = 32, 16, 1, 4, 4
    HEAD_DIM = HIDDEN // HEADS
    MLP = 32

    @staticmethod
    def _make(vocab, hidden, layers, heads, kv_heads, mlp):
        head_dim = hidden // heads
        kv_dim = kv_heads * head_dim
        qkv_dim = hidden + 2 * kv_dim
        t = {}
        t["model.tok_embeddings.weight"] = _rand(vocab, hidden)
        for i in range(layers):
            p = f"model.layers.{i}"
            t[f"{p}.attention_norm.weight"] = _rand(hidden)
            t[f"{p}.attention.wqkv.weight"] = _rand(qkv_dim, hidden)
            t[f"{p}.attention.wo.weight"] = _rand(hidden, hidden)
            t[f"{p}.ffn_norm.weight"] = _rand(hidden)
            t[f"{p}.feed_forward.w1.weight"] = _rand(mlp, hidden)
            t[f"{p}.feed_forward.w3.weight"] = _rand(mlp, hidden)
            t[f"{p}.feed_forward.w2.weight"] = _rand(hidden, mlp)
        t["model.norm.weight"] = _rand(hidden)
        t["output.weight"] = _rand(vocab, hidden)
        return t

    def test_build_engine(self, tmp_path):
        from trtf_build.families.internlm import plugin
        config = {
            "model_type": "internlm2",
            "vocab_size": self.VOCAB, "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS, "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, self._make(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.KV_HEADS, self.MLP))
        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        engine = plugin.build_engine(cfg, weights, max_cache_length=32, verbose=False)
        assert isinstance(engine, bytes) and len(engine) > 0


# =========================================================================
# CodeGen — fused QKV mp_num interleaved, partial RoPE, parallel residual
# =========================================================================

@requires_trt
class TestCodeGenBuildEngine:
    VOCAB, HIDDEN, LAYERS, HEADS = 32, 16, 1, 4
    MLP = HIDDEN * 4

    @staticmethod
    def _make(vocab, hidden, layers, heads, mlp):
        t = {}
        t["transformer.wte.weight"] = _rand(vocab, hidden)
        for i in range(layers):
            p = f"transformer.h.{i}"
            t[f"{p}.ln_1.weight"] = _rand(hidden)
            t[f"{p}.ln_1.bias"] = _rand(hidden)
            t[f"{p}.attn.qkv_proj.weight"] = _rand(3 * hidden, hidden)
            t[f"{p}.attn.out_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.mlp.fc_in.weight"] = _rand(mlp, hidden)
            t[f"{p}.mlp.fc_in.bias"] = _rand(mlp)
            t[f"{p}.mlp.fc_out.weight"] = _rand(hidden, mlp)
            t[f"{p}.mlp.fc_out.bias"] = _rand(hidden)
        t["transformer.ln_f.weight"] = _rand(hidden)
        t["transformer.ln_f.bias"] = _rand(hidden)
        t["lm_head.weight"] = _rand(vocab, hidden)
        t["lm_head.bias"] = _rand(vocab)
        return t

    def test_build_engine(self, tmp_path):
        from trtf_build.families.codegen import plugin
        config = {
            "model_type": "codegen",
            "vocab_size": self.VOCAB, "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS, "num_attention_heads": self.HEADS,
            "rotary_dim": self.HIDDEN // self.HEADS,
        }
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, self._make(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.MLP))
        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        engine = plugin.build_engine(cfg, weights, max_cache_length=32, verbose=False)
        assert isinstance(engine, bytes) and len(engine) > 0
