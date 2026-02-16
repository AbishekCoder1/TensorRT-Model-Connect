"""ModelConfig — parse HF config.json into a typed dataclass."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Parsed model architecture from HF config.json."""

    model_type: str = ""
    architectures: list[str] = field(default_factory=list)
    vocab_size: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 1
    num_key_value_heads: int = 1
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    bos_token_id: int = -1
    eos_token_id: int = -1
    pad_token_id: int = -1
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 8192

    # Explicit head_dim from config.json (0 = not set, fall back to computed).
    _head_dim: int = 0

    # Raw JSON dict for family-specific fields
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def head_dim(self) -> int:
        if self._head_dim > 0:
            return self._head_dim
        if self.num_attention_heads <= 0:
            return 0
        return self.hidden_size // self.num_attention_heads

    @property
    def attention_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @staticmethod
    def from_json(text: str) -> ModelConfig:
        d = json.loads(text)
        return ModelConfig(
            model_type=d.get("model_type", ""),
            architectures=d.get("architectures", []),
            vocab_size=d.get("vocab_size", 0),
            hidden_size=d.get("hidden_size", 0),
            intermediate_size=d.get("intermediate_size", 0),
            num_hidden_layers=d.get("num_hidden_layers", 0),
            num_attention_heads=d.get("num_attention_heads", 1),
            num_key_value_heads=d.get("num_key_value_heads",
                                     d.get("num_attention_heads", 1)),
            rms_norm_eps=d.get("rms_norm_eps", 1e-5),
            rope_theta=d.get("rope_theta", 10000.0),
            bos_token_id=d.get("bos_token_id", -1) or -1,
            eos_token_id=d.get("eos_token_id", -1) or -1,
            pad_token_id=d.get("pad_token_id", -1) or -1,
            tie_word_embeddings=d.get("tie_word_embeddings", False),
            max_position_embeddings=d.get("max_position_embeddings", 8192),
            _head_dim=d.get("head_dim", 0),
            raw=d,
        )

    @staticmethod
    def from_dir(model_dir: str | Path) -> ModelConfig:
        config_path = Path(model_dir) / "config.json"
        return ModelConfig.from_json(config_path.read_text())
