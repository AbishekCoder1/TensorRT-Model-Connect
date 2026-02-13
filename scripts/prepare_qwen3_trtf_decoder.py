#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a trtf_decoder directory for a local upstream Qwen3-style HF checkpoint. "
            "This writes decoder config/vocab/transitions that point to model.safetensors."
        )
    )
    parser.add_argument("--hf-model-dir", required=True, help="Path to local HF model directory")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output decoder directory (default: <hf-model-dir>/trtf_decoder)",
    )
    parser.add_argument(
        "--weights-file",
        default="../model.safetensors",
        help="weights_file path written into decoder config.json",
    )
    parser.add_argument(
        "--max-cache-length",
        type=int,
        default=4096,
        help="max_cache_length written into decoder config.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_placeholder_vocab(vocab_size: int) -> list[str]:
    if vocab_size <= 0:
        raise RuntimeError(f"Invalid vocab_size={vocab_size}")
    vocab = [f"token_{i}" for i in range(vocab_size)]
    if vocab_size > 0:
        vocab[0] = "<unk>"
    if vocab_size > 1:
        vocab[1] = "<bos>"
    if vocab_size > 2:
        vocab[2] = "<eos>"
    if vocab_size > 3:
        vocab[3] = "<pad>"
    return vocab


def main() -> int:
    args = parse_args()
    hf_model_dir = Path(args.hf_model_dir).resolve()
    if not hf_model_dir.exists() or not hf_model_dir.is_dir():
        raise RuntimeError(f"HF model directory does not exist: {hf_model_dir}")

    hf_config_path = hf_model_dir / "config.json"
    if not hf_config_path.exists():
        raise RuntimeError(f"Missing config.json in {hf_model_dir}")
    hf_cfg = load_json(hf_config_path)

    model_type = str(hf_cfg.get("model_type", ""))
    if not model_type.lower().startswith("qwen"):
        raise RuntimeError(
            f"Expected Qwen-style model_type, got {model_type!r}. "
            "This script is intended for upstream Qwen3-like checkpoints."
        )

    vocab_size = int(hf_cfg.get("vocab_size", 0))
    hidden_size = int(hf_cfg.get("hidden_size", 0))
    num_hidden_layers = int(hf_cfg.get("num_hidden_layers", 1))
    num_attention_heads = int(hf_cfg.get("num_attention_heads", 1))
    num_key_value_heads = int(hf_cfg.get("num_key_value_heads", num_attention_heads))
    rms_norm_eps = float(hf_cfg.get("rms_norm_eps", 1.0e-5))
    rope_theta = float(hf_cfg.get("rope_theta", 10000.0))

    if vocab_size <= 0 or hidden_size <= 0:
        raise RuntimeError(
            f"Invalid upstream config values: vocab_size={vocab_size}, hidden_size={hidden_size}"
        )

    output_dir = Path(args.output_dir).resolve() if args.output_dir else hf_model_dir / "trtf_decoder"
    output_dir.mkdir(parents=True, exist_ok=True)

    decoder_cfg = {
        "model_type": "qwen3_decoder_block",
        "architecture_family": "qwen3",
        "default_next_token": "<eos>",
        "max_cache_length": int(max(1, args.max_cache_length)),
        "weights_file": args.weights_file,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": max(1, num_hidden_layers),
        "num_attention_heads": max(1, num_attention_heads),
        "num_key_value_heads": max(1, num_key_value_heads),
        "rms_norm_eps": rms_norm_eps,
        "rope_theta": rope_theta,
        "source_model_type": model_type,
        "mapping_mode": "qwen3-full-stack-v2",
    }
    (output_dir / "config.json").write_text(json.dumps(decoder_cfg, indent=2) + "\n", encoding="utf-8")

    vocab = build_placeholder_vocab(vocab_size)
    (output_dir / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    (output_dir / "transitions.txt").write_text("<eos> <eos>\n", encoding="utf-8")

    print(f"Wrote: {output_dir / 'config.json'}")
    print(f"Wrote: {output_dir / 'vocab.txt'}")
    print(f"Wrote: {output_dir / 'transitions.txt'}")
    print("Next step:")
    print(f"  ./build/trtf_load_model --force-trt {hf_model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
