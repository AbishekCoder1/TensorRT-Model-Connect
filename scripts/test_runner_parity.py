#!/usr/bin/env python3
"""Cross-validation: verify Python TrtRunner matches C++ trtf binary.

Runs the same bundle+prompt through both paths and asserts identical
generated tokens. This is the consistency guarantee between the Python
debug runner and the C++ runtime — if either side changes its mask,
cache, or position logic, this test catches the divergence.

Usage (inside container):
    python3 scripts/test_runner_parity.py \
      --bundle /tmp/qwen3.trtfb \
      --binary ./build/trtf \
      --hf-python .venv/bin/python \
      --prompt "The capital of France is" \
      --max-new-tokens 20
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np


def run_cpp(binary: str, bundle: str, prompt: str, max_new_tokens: int,
            hf_python: str) -> str:
    """Run C++ trtf binary, return generated text."""
    cmd = [binary, "run", bundle, "--prompt", prompt,
           "--max-new-tokens", str(max_new_tokens)]
    if hf_python:
        cmd.extend(["--hf-python", hf_python])

    env = os.environ.copy()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env,
                            timeout=120)
    if result.returncode != 0:
        print(f"C++ stderr:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"C++ binary failed (rc={result.returncode})")

    # stdout contains the generated text (prompt + completion)
    return result.stdout.strip()


def run_python(bundle: str, prompt: str, max_new_tokens: int) -> tuple[str, list[int]]:
    """Run Python TrtRunner, return (generated_text, all_token_ids)."""
    from trtf_build.debug_runner import load_engine_from_bundle, TrtRunner

    engine_plan, header = load_engine_from_bundle(bundle)

    # We need a tokenizer — extract from the bundle or use HF
    import json
    import struct
    import tempfile
    from pathlib import Path

    # Read tokenizer files from bundle
    with open(bundle, "rb") as f:
        magic = f.read(8)
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_raw = json.loads(f.read(header_len).decode("utf-8"))
        sections = header_raw.get("sections", {})
        data_start = 16 + header_len

        tmpdir = tempfile.mkdtemp(prefix="trtf_parity_")
        for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
            if name in sections:
                meta = sections[name]
                f.seek(data_start + meta["offset"])
                data = f.read(meta["size"])
                Path(tmpdir, name).write_bytes(data)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tmpdir, trust_remote_code=True)

    runner = TrtRunner(
        engine_plan=engine_plan,
        max_cache_length=header_raw["max_cache_length"],
        num_layers=header_raw["num_layers"],
    )

    # Encode prompt
    input_ids = tokenizer.encode(prompt)

    # Prefill
    for tid in input_ids:
        result = runner.step(tid)

    # Generate
    gen_ids = list(input_ids)
    for _ in range(max_new_tokens):
        logits = result["logits"].flatten()
        next_token = int(np.argmax(logits))
        gen_ids.append(next_token)
        result = runner.step(next_token)

    # Decode
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    return text, gen_ids


def main():
    parser = argparse.ArgumentParser(
        description="Cross-validate Python TrtRunner vs C++ trtf binary")
    parser.add_argument("--bundle", required=True, help=".trtfb bundle path")
    parser.add_argument("--binary", default="./build/trtf",
                        help="Path to trtf C++ binary")
    parser.add_argument("--hf-python", default="",
                        help="Python path for HF tokenizer bridge")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    args = parser.parse_args()

    print(f"Bundle: {args.bundle}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print()

    # Run C++ binary
    print("Running C++ binary ...", file=sys.stderr)
    cpp_text = run_cpp(args.binary, args.bundle, args.prompt,
                       args.max_new_tokens, args.hf_python)
    print(f"C++:    {cpp_text!r}")

    # Run Python runner
    print("Running Python runner ...", file=sys.stderr)
    py_text, py_ids = run_python(args.bundle, args.prompt,
                                  args.max_new_tokens)
    print(f"Python: {py_text!r}")

    # Compare
    match = cpp_text == py_text
    print(f"\nExact match: {match}")

    if not match:
        # Find first divergence point
        cpp_words = cpp_text.split()
        py_words = py_text.split()
        for i, (cw, pw) in enumerate(zip(cpp_words, py_words)):
            if cw != pw:
                print(f"First word divergence at position {i}: "
                      f"C++={cw!r} Python={pw!r}")
                break
        print("FAIL")
        sys.exit(1)

    print("PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
