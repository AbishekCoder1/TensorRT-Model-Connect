"""1:1 port of standard_checkpoint_mapper.cpp + tensor_math.cpp to Python.

Loads HF safetensors and maps keys to the flat weight dict expected by
standard_decoder_builder.py. All projections are transposed from HF
[out, in] layout to [in, out] for TRT matmul.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Register bfloat16 dtype with numpy (needed for safetensors without torch).
try:
    import ml_dtypes  # noqa: F401
except ImportError:
    pass

from safetensors import safe_open

from .config import ModelConfig


def _layer_key(layer_idx: int, suffix: str) -> str:
    return f"model.layers.{layer_idx}.{suffix}"


def _transpose_2d(arr: np.ndarray, name: str) -> np.ndarray:
    """Transpose [rows, cols] -> [cols, rows] in C-contiguous float32."""
    if arr.ndim != 2:
        raise ValueError(f"Expected rank-2 tensor for transpose: {name}")
    return np.ascontiguousarray(arr.T, dtype=np.float32)


def _expand_kv_projection(
    transposed: np.ndarray,
    hidden: int,
    kv_hidden: int,
    target_hidden: int,
    num_heads: int,
    num_kv_heads: int,
) -> np.ndarray:
    """Expand GQA K/V projection from [hidden, kv_hidden] to [hidden, target_hidden]."""
    if kv_hidden == target_hidden:
        return transposed

    head_dim = target_hidden // num_heads
    group_size = num_heads // num_kv_heads

    out = np.zeros((hidden, target_hidden), dtype=np.float32)
    for row in range(hidden):
        for col in range(target_hidden):
            q_head = col // head_dim
            offset = col % head_dim
            kv_head = min(num_kv_heads - 1, q_head // group_size)
            src_col = kv_head * head_dim + offset
            out[row, col] = transposed[row, src_col]
    return out


def _repeat_head_norm(norm: np.ndarray, num_heads: int) -> np.ndarray:
    """Repeat per-head norm [head_dim] -> [num_heads * head_dim]."""
    return np.tile(norm, num_heads).astype(np.float32)


class WeightDict(dict):
    """A dict mapping logical weight names to flat float32 arrays.

    Keys follow the convention used by standard_decoder_builder.py:
      - embedding: [vocab, hidden]
      - layer.{i}.input_norm: [hidden]
      - layer.{i}.w_q: [hidden, attention_size]  (transposed, GQA-expanded)
      - layer.{i}.w_k: [hidden, attention_size]
      - layer.{i}.w_v: [hidden, attention_size]
      - layer.{i}.q_bias: [attention_size]       (optional)
      - layer.{i}.k_bias: [attention_size]       (optional, GQA-expanded)
      - layer.{i}.v_bias: [attention_size]        (optional, GQA-expanded)
      - layer.{i}.q_norm: [attention_size]        (optional)
      - layer.{i}.k_norm: [attention_size]        (optional)
      - layer.{i}.w_o: [attention_size, hidden]
      - layer.{i}.post_attn_norm: [hidden]
      - layer.{i}.w_gate: [hidden, mlp_size]
      - layer.{i}.w_up: [hidden, mlp_size]
      - layer.{i}.w_down: [mlp_size, hidden]
      - final_norm: [hidden]
      - w_out: [hidden, vocab]
    """


def load_standard_weights(
    model_dir: str | Path,
    config: ModelConfig,
) -> WeightDict:
    """Load HF safetensors and map to standard weight dict."""
    model_dir = Path(model_dir)
    readers = _open_safetensors(model_dir)

    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads

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

        # Norms
        input_norm = _load_tensor(
            readers, _layer_key(layer_idx, "input_layernorm.weight"))
        post_norm = _load_tensor(
            readers, _layer_key(layer_idx, "post_attention_layernorm.weight"))
        weights[f"{prefix}.input_norm"] = input_norm.astype(np.float32)
        weights[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)

        # Q/K/V/O projections
        q_raw = _load_tensor(
            readers, _layer_key(layer_idx, "self_attn.q_proj.weight"))
        k_raw = _load_tensor(
            readers, _layer_key(layer_idx, "self_attn.k_proj.weight"))
        v_raw = _load_tensor(
            readers, _layer_key(layer_idx, "self_attn.v_proj.weight"))
        o_raw = _load_tensor(
            readers, _layer_key(layer_idx, "self_attn.o_proj.weight"))

        q_hidden = q_raw.shape[0]
        kv_hidden = k_raw.shape[0]
        if attention_size == 0:
            attention_size = q_hidden
        if mlp_size == 0:
            gate_raw = _load_tensor(
                readers, _layer_key(layer_idx, "mlp.gate_proj.weight"))
            mlp_size = gate_raw.shape[0]
        else:
            gate_raw = _load_tensor(
                readers, _layer_key(layer_idx, "mlp.gate_proj.weight"))

        head_dim = q_hidden // num_heads

        # Transpose all projections [out, in] -> [in, out]
        q_t = _transpose_2d(q_raw, "q_proj")
        k_t = _transpose_2d(k_raw, "k_proj")
        v_t = _transpose_2d(v_raw, "v_proj")
        o_t = _transpose_2d(o_raw, "o_proj")

        # GQA expansion for K, V
        k_expanded = _expand_kv_projection(
            k_t, hidden, kv_hidden, q_hidden, num_heads, num_kv_heads)
        v_expanded = _expand_kv_projection(
            v_t, hidden, kv_hidden, q_hidden, num_heads, num_kv_heads)

        weights[f"{prefix}.w_q"] = q_t
        weights[f"{prefix}.w_k"] = k_expanded
        weights[f"{prefix}.w_v"] = v_expanded
        weights[f"{prefix}.w_o"] = o_t

        # Optional QKV biases (Qwen2 style)
        q_bias_key = _layer_key(layer_idx, "self_attn.q_proj.bias")
        k_bias_key = _layer_key(layer_idx, "self_attn.k_proj.bias")
        v_bias_key = _layer_key(layer_idx, "self_attn.v_proj.bias")
        if _has_tensor(readers, q_bias_key):
            weights[f"{prefix}.q_bias"] = _load_tensor(
                readers, q_bias_key).astype(np.float32)
        if _has_tensor(readers, k_bias_key):
            raw = _load_tensor(readers, k_bias_key).astype(np.float32)
            if raw.shape[0] == kv_hidden and kv_hidden != q_hidden:
                expanded = np.zeros(q_hidden, dtype=np.float32)
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
            if raw.shape[0] == kv_hidden and kv_hidden != q_hidden:
                expanded = np.zeros(q_hidden, dtype=np.float32)
                for qh in range(num_heads):
                    kvh = min(num_kv_heads - 1,
                              qh // (num_heads // num_kv_heads))
                    expanded[qh * head_dim:(qh + 1) * head_dim] = \
                        raw[kvh * head_dim:(kvh + 1) * head_dim]
                weights[f"{prefix}.v_bias"] = expanded
            else:
                weights[f"{prefix}.v_bias"] = raw

        # Optional per-head q/k norm (Qwen3 style)
        q_norm_key = _layer_key(layer_idx, "self_attn.q_norm.weight")
        k_norm_key = _layer_key(layer_idx, "self_attn.k_norm.weight")
        if _has_tensor(readers, q_norm_key):
            weights[f"{prefix}.q_norm"] = _repeat_head_norm(
                _load_tensor(readers, q_norm_key).astype(np.float32),
                num_heads)
        if _has_tensor(readers, k_norm_key):
            weights[f"{prefix}.k_norm"] = _repeat_head_norm(
                _load_tensor(readers, k_norm_key).astype(np.float32),
                num_heads)

        # MLP projections
        up_raw = _load_tensor(
            readers, _layer_key(layer_idx, "mlp.up_proj.weight"))
        down_raw = _load_tensor(
            readers, _layer_key(layer_idx, "mlp.down_proj.weight"))

        weights[f"{prefix}.w_gate"] = _transpose_2d(gate_raw, "gate_proj")
        weights[f"{prefix}.w_up"] = _transpose_2d(up_raw, "up_proj")
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
        # Tied embeddings
        weights["w_out"] = _transpose_2d(embedding.copy(), "embedding_tied")

    weights["_attention_size"] = attention_size  # type: ignore[assignment]
    weights["_mlp_size"] = mlp_size  # type: ignore[assignment]

    return weights


# ---------------------------------------------------------------------------
# Safetensors I/O helpers
# ---------------------------------------------------------------------------

def _detect_framework() -> str:
    """Use 'torch' if available (handles BF16 natively), else 'numpy'."""
    try:
        import torch  # noqa: F401
        return "torch"
    except ImportError:
        return "numpy"


class _TorchBinReader:
    """Adapter that wraps a pytorch .bin state dict with the safetensors reader
    interface (keys() / get_tensor())."""

    def __init__(self, path: Path):
        import torch
        self._state = torch.load(str(path), map_location="cpu", weights_only=True)

    def keys(self) -> list[str]:
        return list(self._state.keys())

    def get_tensor(self, name: str):
        return self._state[name]


def _open_safetensors(model_dir: Path) -> list:
    """Open all safetensor shards (or pytorch .bin) in a model directory."""
    fw = _detect_framework()
    single = model_dir / "model.safetensors"
    if single.exists():
        return [safe_open(str(single), framework=fw)]

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        import json
        index = json.loads(index_path.read_text())
        shard_files = sorted(set(index.get("weight_map", {}).values()))
        return [
            safe_open(str(model_dir / f), framework=fw)
            for f in shard_files
        ]

    # Diffusers format: diffusion_pytorch_model.safetensors
    diff_single = model_dir / "diffusion_pytorch_model.safetensors"
    if diff_single.exists():
        return [safe_open(str(diff_single), framework=fw)]

    diff_index = model_dir / "diffusion_pytorch_model.safetensors.index.json"
    if diff_index.exists():
        import json
        index = json.loads(diff_index.read_text())
        shard_files = sorted(set(index.get("weight_map", {}).values()))
        return [
            safe_open(str(model_dir / f), framework=fw)
            for f in shard_files
        ]

    # Fallback: pytorch_model.bin (older HF models)
    bin_single = model_dir / "pytorch_model.bin"
    if bin_single.exists():
        return [_TorchBinReader(bin_single)]

    raise FileNotFoundError(
        f"No model.safetensors, index.json, or pytorch_model.bin in {model_dir}")


def _has_tensor(readers: list, name: str) -> bool:
    for r in readers:
        if name in r.keys():
            return True
    return False


def _load_tensor(readers: list, name: str) -> np.ndarray:
    for r in readers:
        if name in r.keys():
            t = r.get_tensor(name)
            # If loaded via torch framework, convert to numpy float32
            if hasattr(t, "numpy"):
                # torch.Tensor — handles BF16 natively
                return t.float().numpy()
            # numpy path — handle BF16 via uint16 bit-shift
            dtype_str = str(t.dtype)
            if t.dtype == np.uint16 or dtype_str == "bfloat16":
                t = t.view(np.uint16).astype(np.uint32) << 16
                t = t.view(np.float32)
            elif dtype_str == "float16":
                t = t.astype(np.float32)
            return t.astype(np.float32)
    raise KeyError(f"Tensor not found: {name}")
