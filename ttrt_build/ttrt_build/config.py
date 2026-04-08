"""Model configuration — wraps HF config.json for Torch-TRT builder.

Reuses trtf_build.config.ModelConfig when available, otherwise provides
a minimal standalone implementation so ttrt_build can work independently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


try:
    from trtf_build.config import ModelConfig
except ImportError:
    class ModelConfig:
        """Minimal standalone config when trtf_build is not installed."""

        def __init__(self, *, model_type: str = "", raw: dict[str, Any] | None = None):
            raw = raw or {}
            self.model_type = model_type or raw.get("model_type", "")
            self.raw = raw
            self.hidden_size: int = raw.get("hidden_size", 0)
            self.num_hidden_layers: int = raw.get("num_hidden_layers", 0)
            self.num_attention_heads: int = raw.get("num_attention_heads", 0)
            self.num_key_value_heads: int = raw.get(
                "num_key_value_heads", self.num_attention_heads)
            self.intermediate_size: int = raw.get("intermediate_size", 0)
            self.vocab_size: int = raw.get("vocab_size", 0)
            self.head_dim: int = raw.get(
                "head_dim", self.hidden_size // max(self.num_attention_heads, 1))
            self.rms_norm_eps: float = raw.get("rms_norm_eps", 1e-6)
            self.rope_theta: float = raw.get("rope_theta", 10000.0)
            self.max_position_embeddings: int = raw.get("max_position_embeddings", 2048)

        @classmethod
        def from_dir(cls, model_dir: str | Path) -> "ModelConfig":
            """Load from a directory containing config.json."""
            cfg_path = Path(model_dir) / "config.json"
            with open(cfg_path) as f:
                raw = json.load(f)
            return cls(raw=raw)

        @classmethod
        def from_json(cls, json_str: str) -> "ModelConfig":
            """Load from a JSON string."""
            raw = json.loads(json_str)
            return cls(raw=raw)

        def to_hf_config(self):
            """Convert to a HuggingFace PretrainedConfig for model loading."""
            from transformers import AutoConfig
            return AutoConfig.for_model(self.model_type, **self.raw)
