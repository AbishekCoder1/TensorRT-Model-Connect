#!/usr/bin/env python3
"""E2E logit comparison between trtf binary and HuggingFace transformers.

Usage:
    python3 scripts/diff_logits.py \
      --model-dir models/hf/Qwen__Qwen3-0.6B \
      --binary ./build-container-phase1/trtf_load_model \
      --backend-flag --force-trt --atol 1e-3 \
      --prompt "The capital of France is"
"""
import argparse
import json
import os
import re
import subprocess
import sys

STANDARD_PROMPTS = [
    ("factual", "The capital of France is"),
    ("reasoning", "Explain why water boils at 100 degrees Celsius at sea level in one sentence."),
    ("code", "Write a Python function that checks if a number is prime:"),
    ("multi-turn", "User: What is 2+2?\nAssistant:"),
]


def parse_trtf_debug_logits(stderr_text: str):
    """Parse TRTF_DEBUG_LOGITS lines from stderr."""
    steps = {}
    for line in stderr_text.splitlines():
        if not line.startswith("TRTF_DEBUG_LOGITS"):
            continue
        match = re.match(r"TRTF_DEBUG_LOGITS step=(\d+)(.*)", line)
        if not match:
            continue
        step = int(match.group(1))
        pairs = match.group(2).strip().split()
        logit_map = {}
        for pair in pairs:
            parts = pair.split(":")
            if len(parts) == 2:
                token_id = int(parts[0])
                logit_val = float(parts[1])
                logit_map[token_id] = logit_val
        steps[step] = logit_map
    return steps


def run_trtf(binary, model_dir, prompt, backend_flag, max_new_tokens):
    """Run the trtf binary and capture output + debug logits."""
    env = os.environ.copy()
    env["TRTF_DEBUG_LOGITS_TOPK"] = "10"
    env["TRTF_MAX_NEW_TOKENS"] = str(max_new_tokens)
    env["TRTF_MAX_CACHE_LENGTH"] = str(max_new_tokens + 128)

    cmd = [binary]
    if backend_flag:
        cmd.append(backend_flag)
    cmd.extend([model_dir, prompt])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    except subprocess.TimeoutExpired:
        print("ERROR: trtf binary timed out after 300s", file=sys.stderr)
        return None, None, None

    generated_text = result.stdout.strip()
    debug_logits = parse_trtf_debug_logits(result.stderr)
    return generated_text, debug_logits, result.returncode


def run_hf_transformers(model_dir, prompt, max_new_tokens):
    """Run HuggingFace transformers and capture output + logits."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("WARNING: transformers/torch not available, skipping HF comparison",
              file=sys.stderr)
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    generated_ids = outputs.sequences[0].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    hf_logits = {}
    if hasattr(outputs, "scores") and outputs.scores:
        for step_idx, score_tensor in enumerate(outputs.scores):
            logits = score_tensor[0]
            top_vals, top_ids = torch.topk(logits, k=min(10, logits.shape[0]))
            hf_logits[step_idx] = {
                int(tid): float(tval)
                for tid, tval in zip(top_ids.tolist(), top_vals.tolist())
            }

    return generated_text, hf_logits


def compare_logits(trtf_logits, hf_logits, atol):
    """Compare top-K logits from both sources."""
    max_abs_diff = 0.0
    compared = 0

    common_steps = set(trtf_logits.keys()) & set(hf_logits.keys())
    for step in sorted(common_steps):
        trtf_step = trtf_logits[step]
        hf_step = hf_logits[step]
        common_tokens = set(trtf_step.keys()) & set(hf_step.keys())
        for tid in common_tokens:
            diff = abs(trtf_step[tid] - hf_step[tid])
            max_abs_diff = max(max_abs_diff, diff)
            compared += 1

    return max_abs_diff, compared


def compute_token_match_rate(trtf_text, hf_text):
    """Compute simple word-level match rate."""
    trtf_words = trtf_text.split()
    hf_words = hf_text.split()
    if not trtf_words and not hf_words:
        return 1.0
    if not trtf_words or not hf_words:
        return 0.0
    matches = sum(1 for a, b in zip(trtf_words, hf_words) if a == b)
    return matches / max(len(trtf_words), len(hf_words))


def main():
    parser = argparse.ArgumentParser(description="E2E logit diff between trtf and HF transformers")
    parser.add_argument("--model-dir", required=True, help="Path to HF model directory")
    parser.add_argument("--binary", required=True, help="Path to trtf binary")
    parser.add_argument("--backend-flag", default="", help="Backend flag (e.g., --force-trt)")
    parser.add_argument("--prompt", default="", help="Single prompt to test (overrides battery)")
    parser.add_argument("--max-new-tokens", type=int, default=20, help="Max tokens to generate")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for logit comparison")
    parser.add_argument("--battery", action="store_true", help="Run standard prompt battery")
    args = parser.parse_args()

    prompts = []
    if args.prompt:
        prompts = [("custom", args.prompt)]
    elif args.battery:
        prompts = STANDARD_PROMPTS
    else:
        prompts = [("default", "The capital of France is")]

    all_passed = True
    for label, prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt [{label}]: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"{'='*60}")

        trtf_text, trtf_logits, rc = run_trtf(
            args.binary, args.model_dir, prompt, args.backend_flag or None, args.max_new_tokens
        )
        if rc is None or rc != 0:
            print(f"  TRTF: FAILED (rc={rc})")
            all_passed = False
            continue

        print(f"  TRTF output: {trtf_text[:120]}{'...' if len(trtf_text) > 120 else ''}")
        print(f"  TRTF debug logit steps: {len(trtf_logits)}")

        hf_text, hf_logits = run_hf_transformers(args.model_dir, prompt, args.max_new_tokens)
        if hf_text is None:
            print("  HF: skipped (not available)")
            continue

        print(f"  HF output: {hf_text[:120]}{'...' if len(hf_text) > 120 else ''}")

        text_exact = trtf_text.strip() == hf_text.strip()
        token_match = compute_token_match_rate(trtf_text, hf_text)
        max_diff, compared = compare_logits(trtf_logits, hf_logits, args.atol)

        print(f"  text_exact_match: {text_exact}")
        print(f"  token_match_rate: {token_match:.4f}")
        print(f"  max_abs_logit_diff: {max_diff:.6f} (compared {compared} values)")
        print(f"  within_atol ({args.atol}): {max_diff <= args.atol}")

        if max_diff > args.atol and compared > 0:
            print(f"  FAIL: logit diff {max_diff:.6f} exceeds atol {args.atol}")
            all_passed = False
        else:
            print(f"  PASS")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
