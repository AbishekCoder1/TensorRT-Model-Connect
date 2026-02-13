#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare trtf pipeline output against direct transformers output for a local HF model."
    )
    parser.add_argument("--model-dir", required=True, help="Path to local HF model directory")
    parser.add_argument("--binary", required=True, help="Path to trtf_text_generation executable")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    return parser.parse_args()


def parse_pipeline_generated_text(stdout: str) -> str:
    generated_line = None
    for line in stdout.splitlines():
        if "generated_text" in line:
            generated_line = line.strip()
            break
    if generated_line is None:
        raise RuntimeError(f"Unable to find generated_text in pipeline output:\n{stdout}")

    payload = ast.literal_eval(generated_line)
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"Unexpected pipeline output payload: {generated_line}")
    item = payload[0]
    if not isinstance(item, dict) or "generated_text" not in item:
        raise RuntimeError(f"Unexpected pipeline output item: {generated_line}")
    return str(item["generated_text"])


def main() -> int:
    args = parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"model-dir does not exist: {model_dir}")

    binary = Path(args.binary)
    if not binary.exists():
        raise SystemExit(f"binary does not exist: {binary}")

    proc = subprocess.run(
        [str(binary), str(model_dir), args.prompt],
        capture_output=True,
        text=True,
        check=True,
    )
    pipeline_text = parse_pipeline_generated_text(proc.stdout)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        use_safetensors=True,
    )
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    transformers_text = tokenizer.decode(out[0], skip_special_tokens=False)

    pipe_ids = tokenizer.encode(pipeline_text, add_special_tokens=False)
    ref_ids = tokenizer.encode(transformers_text, add_special_tokens=False)
    overlap = min(len(pipe_ids), len(ref_ids))
    matches = sum(1 for i in range(overlap) if pipe_ids[i] == ref_ids[i])
    token_accuracy = (matches / overlap) if overlap > 0 else 1.0
    exact_match = pipeline_text == transformers_text

    report = {
        "exact_match": exact_match,
        "token_accuracy": token_accuracy,
        "pipeline_token_count": len(pipe_ids),
        "transformers_token_count": len(ref_ids),
        "pipeline_text": pipeline_text,
        "transformers_text": transformers_text,
    }
    print(json.dumps(report, indent=2))
    return 0 if exact_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
