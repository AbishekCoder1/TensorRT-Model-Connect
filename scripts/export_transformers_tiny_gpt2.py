#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel


def strip_inline_comment(line: str) -> str:
    hash_pos = line.find("#")
    if hash_pos >= 0:
        line = line[:hash_pos]
    return line.strip()


def load_vocab(path: Path) -> List[str]:
    if not path.exists():
        return [
            "<unk>",
            "<bos>",
            "<eos>",
            "the",
            "secret",
            "to",
            "baking",
            "a",
            "really",
            "good",
            "cake",
            "is",
            "use",
            "fresh",
            "butter",
            "and",
            "measure",
            "carefully",
            "follow",
            "recipe",
            "exactly",
            ".",
            ",",
            "?",
            "!",
            "<pad>",
        ]

    vocab: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = strip_inline_comment(raw)
        if line:
            vocab.append(line)
    if not vocab:
        raise RuntimeError(f"Vocabulary is empty: {path}")
    return vocab


def load_transitions(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        return [
            ("is", "to"),
            ("to", "use"),
            ("use", "fresh"),
            ("fresh", "butter"),
            ("butter", "and"),
            ("and", "measure"),
            ("measure", "carefully"),
            ("carefully", "follow"),
            ("follow", "recipe"),
            ("recipe", "exactly"),
            ("exactly", "."),
            (".", "<eos>"),
            ("<eos>", "<eos>"),
        ]

    transitions: List[Tuple[str, str]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = strip_inline_comment(raw)
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise RuntimeError(f"Invalid transition at {path}:{line_no}: {raw}")
        transitions.append((parts[0], parts[1]))
    if not transitions:
        raise RuntimeError(f"Transitions are empty: {path}")
    return transitions


def build_transition_logits(
    vocab: Sequence[str], transitions: Sequence[Tuple[str, str]], default_next_token: str
) -> np.ndarray:
    token_to_id: Dict[str, int] = {token: i for i, token in enumerate(vocab)}
    if default_next_token not in token_to_id:
        raise RuntimeError(f"default_next_token {default_next_token!r} not found in vocab")

    vocab_size = len(vocab)
    logits = np.full((vocab_size, vocab_size), -1.0, dtype=np.float32)
    default_id = token_to_id[default_next_token]
    logits[:, default_id] = 8.0

    for from_token, to_token in transitions:
        if from_token not in token_to_id or to_token not in token_to_id:
            raise RuntimeError(f"Transition token missing in vocab: {from_token} -> {to_token}")
        from_id = token_to_id[from_token]
        to_id = token_to_id[to_token]
        logits[from_id, :] = -1.0
        logits[from_id, to_id] = 8.0
    return logits


def build_transformers_reference_model(vocab: Sequence[str], target_logits: np.ndarray) -> GPT2LMHeadModel:
    vocab_size = len(vocab)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=128,
        n_embd=vocab_size,
        n_layer=1,
        n_head=1,
        n_inner=vocab_size * 2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        tie_word_embeddings=False,
        bos_token_id=vocab.index("<bos>") if "<bos>" in vocab else 0,
        eos_token_id=vocab.index("<eos>") if "<eos>" in vocab else 0,
    )
    model = GPT2LMHeadModel(config)
    model.eval()

    with torch.no_grad():
        model.transformer.wte.weight.zero_()
        model.transformer.wte.weight.copy_(torch.eye(vocab_size, dtype=torch.float32))
        model.transformer.wpe.weight.zero_()

        for block in model.transformer.h:
            block.ln_1.weight.fill_(1.0)
            block.ln_1.bias.zero_()
            block.ln_2.weight.fill_(1.0)
            block.ln_2.bias.zero_()

            block.attn.c_attn.weight.zero_()
            block.attn.c_attn.bias.zero_()
            block.attn.c_proj.weight.zero_()
            block.attn.c_proj.bias.zero_()

            block.mlp.c_fc.weight.zero_()
            block.mlp.c_fc.bias.zero_()
            block.mlp.c_proj.weight.zero_()
            block.mlp.c_proj.bias.zero_()

        model.transformer.ln_f.weight.fill_(1.0)
        model.transformer.ln_f.bias.zero_()

        token_ids = torch.arange(vocab_size, dtype=torch.long).unsqueeze(1)
        hidden = model.transformer(input_ids=token_ids).last_hidden_state[:, -1, :].double()
        target = torch.tensor(target_logits, dtype=torch.double)

        # Solve H * W^T = target for W, so model logits match target exactly for each token.
        weight_t = torch.linalg.solve(hidden, target)
        model.lm_head.weight.copy_(weight_t.t().float())

        predicted = model(input_ids=token_ids).logits[:, -1, :]
        max_abs_err = float(torch.max(torch.abs(predicted - torch.tensor(target_logits))).item())
        if max_abs_err > 1.0e-4:
            raise RuntimeError(f"Failed to fit lm_head to target logits. max_abs_err={max_abs_err:.6f}")

    return model


def write_vocab(path: Path, vocab: Sequence[str]) -> None:
    text = "\n".join(vocab) + "\n"
    path.write_text(text, encoding="utf-8")


def write_transitions(path: Path, transitions: Sequence[Tuple[str, str]]) -> None:
    lines = [f"{src} {dst}" for src, dst in transitions]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_config(path: Path, default_next_token: str, max_cache_length: int, weights_file: str) -> None:
    cfg = {
        "model_type": "tiny_decoder_block",
        "default_next_token": default_next_token,
        "max_cache_length": max_cache_length,
        "weights_file": weights_file,
    }
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def write_weights_file(
    path: Path, vocab: Sequence[str], transitions: Sequence[Tuple[str, str]], default_next_token: str
) -> None:
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    vocab_size = len(vocab)
    hidden_size = vocab_size
    mlp_size = hidden_size * 2

    lines: List[str] = []
    lines.append("# Exported from transformers GPT2 reference by scripts/export_transformers_tiny_gpt2.py")
    lines.append("hidden_size {}".format(hidden_size))
    lines.append("mlp_size {}".format(mlp_size))
    lines.append("")
    lines.append(f"tensor embedding 2 {vocab_size} {hidden_size}")
    lines.append("fill 0")
    lines.append("identity 1")
    lines.append("end")
    lines.append("")
    lines.append(f"tensor w_q 2 {hidden_size} {hidden_size}")
    lines.append("fill 0")
    lines.append("identity 1")
    lines.append("end")
    lines.append("")
    lines.append(f"tensor w_k 2 {hidden_size} {hidden_size}")
    lines.append("fill 0")
    lines.append("identity 1")
    lines.append("end")
    lines.append("")
    lines.append(f"tensor w_v 2 {hidden_size} {hidden_size}")
    lines.append("fill 0")
    lines.append("end")
    lines.append("")
    lines.append(f"tensor w1 2 {hidden_size} {mlp_size}")
    lines.append("fill 0")
    lines.append("end")
    lines.append("")
    lines.append(f"tensor b1 1 {mlp_size}")
    lines.append("fill 0")
    lines.append("end")
    lines.append("")
    lines.append(f"tensor w2 2 {mlp_size} {hidden_size}")
    lines.append("fill 0")
    lines.append("end")
    lines.append("")
    lines.append(f"tensor b2 1 {hidden_size}")
    lines.append("fill 0")
    lines.append("end")
    lines.append("")
    lines.append(f"tensor w_out 2 {hidden_size} {vocab_size}")
    lines.append("fill -1")
    lines.append(f"all_rows_transition {token_to_id[default_next_token]} 8 -1")
    for src, dst in transitions:
        lines.append(f"row_transition {token_to_id[src]} {token_to_id[dst]} 8 -1")
    lines.append("end")
    lines.append("")
    lines.append(f"tensor b_out 1 {vocab_size}")
    lines.append("fill 0")
    lines.append("end")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a tiny deterministic GPT2 transformers model and matching trtf model assets."
    )
    parser.add_argument(
        "--output-dir",
        default="models/tiny-cake-v1",
        help="Model directory to write trtf assets + transformers_reference.pt",
    )
    parser.add_argument("--max-cache-length", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab(out_dir / "vocab.txt")
    transitions = load_transitions(out_dir / "transitions.txt")

    config_path = out_dir / "config.json"
    default_next = "to"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(cfg.get("default_next_token"), str):
                default_next = cfg["default_next_token"]
        except Exception:
            pass

    target_logits = build_transition_logits(vocab, transitions, default_next)
    model = build_transformers_reference_model(vocab, target_logits)

    reference_path = out_dir / "transformers_reference.pt"
    torch.save(
        {
            "config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "vocab": list(vocab),
            "default_next_token": default_next,
            "transitions": list(transitions),
        },
        reference_path,
    )

    write_vocab(out_dir / "vocab.txt", vocab)
    write_transitions(out_dir / "transitions.txt", transitions)
    write_config(out_dir / "config.json", default_next, args.max_cache_length, "weights.txt")
    write_weights_file(out_dir / "weights.txt", vocab, transitions, default_next)

    print(f"Exported transformers reference: {reference_path}")
    print(f"Exported trtf model assets in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
