#!/usr/bin/env python3
"""MagpieTTS IPA tokenizer bridge for trtf C++ runtime.

Same CLI protocol as hf_tokenizer.py. Loads a MagpieTTS .nemo archive
via NeMo's model restore, extracts the IPATokenizer for the selected
language (default: english_phoneme), and runs encode/decode.

Requires NeMo to be installed in the Python environment.

The tokenizer is an aggregated multi-language IPA tokenizer. Each language
has its own sub-tokenizer with a vocab offset. For English, the offset is 0
so token IDs are used directly. EOS = text_vocab_size + 1 = 2379.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import warnings

# Suppress noisy NeMo/PyTorch logs and avoid network requests
os.environ.setdefault("NEMO_LOG_LEVEL", "ERROR")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MagpieTTS IPA tokenizer bridge for trtf C++ runtime")
    parser.add_argument("--nemo-path", required=True,
                        help="Path to MagpieTTS .nemo archive or directory containing one")
    parser.add_argument("--check", action="store_true",
                        help="Validate tokenizer can be loaded")
    parser.add_argument("--op", choices=["encode", "decode"],
                        default="", help="Tokenizer operation")
    parser.add_argument("--text-file", default="",
                        help="Input text file for encode")
    parser.add_argument("--ids", default="",
                        help="Comma-separated token IDs for decode")
    parser.add_argument("--lang", default="english_phoneme",
                        help="Language key from text_tokenizers (default: english_phoneme)")
    return parser.parse_args()


def _resolve_nemo_path(path: str) -> pathlib.Path:
    """Resolve .nemo archive path (file or directory containing .nemo)."""
    p = pathlib.Path(path)
    if p.is_dir():
        nemo_files = sorted(p.glob("*.nemo"))
        if nemo_files:
            return nemo_files[0]
        raise FileNotFoundError(f"No .nemo file found in {path}")
    return p


def _extract_nemo_assets(nemo_path: pathlib.Path, extract_dir: pathlib.Path) -> None:
    """Extract phoneme dict / heteronym files from .nemo archive.

    The .nemo archive contains the phoneme dict and heteronym files that
    IpaG2p needs (e.g. ipa_cmudict-0.7b_nv23.01.txt, heteronyms-052722).
    We extract all non-checkpoint, non-config files to a persistent cache dir.
    """
    import tarfile
    with tarfile.open(str(nemo_path), "r") as tar:
        for member in tar.getmembers():
            basename = pathlib.Path(member.name).name
            # Skip model weights and config (we only want asset files)
            if basename in ("model_weights.ckpt", "model_config.yaml"):
                continue
            if member.isfile():
                dest = extract_dir / basename
                if not dest.exists():
                    f = tar.extractfile(member)
                    if f is not None:
                        dest.write_bytes(f.read())


def load_tokenizer(nemo_path: pathlib.Path, lang_key: str = "english_phoneme"):
    """Load NeMo IPATokenizer from .nemo archive config.

    Extracts the text_tokenizers config from model_config.yaml and
    instantiates the IPA tokenizer via Hydra. Phoneme dict / heteronym
    files referenced by nemo: URIs are extracted directly from the .nemo
    archive (they are bundled inside it as registered NeMo artifacts).

    Returns (sub_tokenizer, text_vocab_size).
    """
    import logging
    import tarfile
    logging.disable(logging.WARNING)

    import yaml
    from omegaconf import OmegaConf

    # Persistent cache dir for extracted assets (survives across calls)
    asset_dir = pathlib.Path.home() / ".cache" / "trtf_nemo_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    # Extract model_config.yaml and asset files from the archive
    nemo_cfg = None
    with tarfile.open(str(nemo_path), "r") as tar:
        for member in tar.getmembers():
            if pathlib.Path(member.name).name == "model_config.yaml":
                f = tar.extractfile(member)
                if f is not None:
                    nemo_cfg = yaml.safe_load(f.read())
                break
    if nemo_cfg is None:
        raise FileNotFoundError(f"model_config.yaml not found in {nemo_path}")

    # Extract phoneme dict / heteronym asset files
    _extract_nemo_assets(nemo_path, asset_dir)

    text_vocab_size = int(nemo_cfg.get("text_vocab_size", 2378))
    text_tokenizers = nemo_cfg.get("text_tokenizers", {})

    if lang_key not in text_tokenizers:
        available = list(text_tokenizers.keys())
        raise ValueError(
            f"Language '{lang_key}' not found. Available: {available}")

    tok_cfg = dict(text_tokenizers[lang_key])

    # Override phoneme_probability for deterministic inference
    if "g2p" in tok_cfg and "phoneme_probability" in tok_cfg["g2p"]:
        tok_cfg["g2p"]["phoneme_probability"] = 1.0

    # Resolve nemo: URIs to local paths extracted from the archive
    if "g2p" in tok_cfg:
        g2p_cfg = tok_cfg["g2p"]
        for key in ("phoneme_dict", "heteronyms"):
            val = g2p_cfg.get(key)
            if isinstance(val, str) and val.startswith("nemo:"):
                filename = val.split(":")[-1]
                local_path = asset_dir / filename
                if local_path.exists():
                    g2p_cfg[key] = str(local_path)

    # Instantiate via Hydra
    from hydra.utils import instantiate
    oc = OmegaConf.create(tok_cfg)
    sub_tok = instantiate(oc)

    return sub_tok, text_vocab_size


def parse_ids_csv(ids_text: str) -> list[int]:
    ids_text = ids_text.strip()
    if not ids_text:
        return []
    out: list[int] = []
    for part in ids_text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def main() -> int:
    args = parse_args()

    nemo_path = _resolve_nemo_path(args.nemo_path)
    if not nemo_path.exists():
        print(f".nemo archive does not exist: {nemo_path}", file=sys.stderr)
        return 2

    try:
        tokenizer, text_vocab_size = load_tokenizer(nemo_path, args.lang)
    except Exception as exc:
        print(f"Failed to load MagpieTTS tokenizer: {exc}", file=sys.stderr)
        return 3

    if args.check:
        return 0

    # EOS token ID = text_vocab_size + 1 (NeMo convention)
    eos_id = text_vocab_size + 1

    if args.op == "encode":
        if not args.text_file:
            print("--text-file is required for encode", file=sys.stderr)
            return 4
        text = pathlib.Path(args.text_file).read_text(encoding="utf-8")
        # Tokenize via NeMo IPA tokenizer
        ids = tokenizer.encode(text)
        # Append EOS
        ids.append(eos_id)
        print(" ".join(str(i) for i in ids))
        return 0

    if args.op == "decode":
        # Decode is best-effort for IPA tokenizer (not round-trippable)
        ids = parse_ids_csv(args.ids)
        try:
            decoded = tokenizer.decode(ids)
        except Exception:
            decoded = " ".join(str(i) for i in ids)
        print(decoded)
        return 0

    print("--op is required unless --check is set", file=sys.stderr)
    return 6


if __name__ == "__main__":
    raise SystemExit(main())
