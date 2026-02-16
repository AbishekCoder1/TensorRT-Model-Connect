#!/usr/bin/env python3
"""Pure-Python E2E logit comparison between TRT engine and HF transformers.

No C++ binary needed. Builds a TRT engine via trtf_build, runs inference
in Python, and compares per-step logits against HF transformers.

Usage:
    python3 scripts/diff_logits.py \
      --model Qwen/Qwen3-0.6B \
      --prompt "The capital of France is" \
      --max-new-tokens 10 --atol 1e-3

    python3 scripts/diff_logits.py \
      --model models/hf/Qwen__Qwen3-0.6B --battery
"""
import argparse
import sys

import numpy as np

STANDARD_PROMPTS = [
    ("factual", "The capital of France is"),
    ("reasoning", "Explain why water boils at 100 degrees Celsius."),
    ("code", "Write a Python function that checks if a number is prime:"),
    ("multi-turn", "User: What is 2+2?\nAssistant:"),
]


def build_trt_engine(model_id_or_path, max_cache_length, verbose):
    """Build TRT engine and return (engine_plan_bytes, config, weights)."""
    from trtf_build.engine_builder import _resolve_model
    from trtf_build.config import ModelConfig
    from trtf_build.families import find_plugin

    model_dir = _resolve_model(model_id_or_path)
    config = ModelConfig.from_dir(model_dir)
    plugin = find_plugin(config.model_type)
    if plugin is None:
        raise ValueError(f"No plugin for model_type={config.model_type!r}")

    print(f"[diff] Loading weights ({config.model_type}) ...", file=sys.stderr)
    weights = plugin.load_weights(model_dir, config)
    print(f"[diff] Building TRT engine (cache={max_cache_length}) ...",
          file=sys.stderr)
    engine_plan = plugin.build_engine(
        config, weights, max_cache_length, verbose=verbose)
    print(f"[diff] Engine built ({len(engine_plan) / 1e6:.1f} MB)",
          file=sys.stderr)

    return engine_plan, config, model_dir


def run_trt(engine_plan, config, input_ids, max_new_tokens, max_cache_length):
    """Run TRT inference, return list of logit arrays (one per step)."""
    from trtf_build.debug_runner import TrtRunner

    runner = TrtRunner(
        engine_plan=engine_plan,
        max_cache_length=max_cache_length,
        num_layers=config.num_hidden_layers,
    )

    results = runner.generate(input_ids, max_new_tokens)
    return [r["logits"].flatten() for r in results]


def _load_hf_model(model_dir, trust_remote_code=False):
    """Load HF model. Uses native transformers support by default.

    If the model requires custom code (e.g. older repos without native
    transformers support), pass --trust-remote-code to enable it.
    This executes Python code from the model repository.
    """
    import torch
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=False, torch_dtype=torch.float32)
    except (ValueError, KeyError, ImportError) as e:
        if trust_remote_code:
            print(f"[diff] Native loading failed ({e}), "
                  f"retrying with trust_remote_code=True ...",
                  file=sys.stderr)
            return AutoModelForCausalLM.from_pretrained(
                model_dir, trust_remote_code=True, torch_dtype=torch.float32)
        raise ValueError(
            f"Failed to load model from {model_dir} without custom code. "
            f"If this model requires custom code, re-run with "
            f"--trust-remote-code. Original error: {e}"
        ) from e


def run_hf(model_dir, input_ids, max_new_tokens, trust_remote_code=False):
    """Run HF transformers, return list of logit arrays (one per step)."""
    import torch
    from transformers import AutoTokenizer

    model = _load_hf_model(model_dir, trust_remote_code=trust_remote_code)
    model.eval()

    # Run prefill to get logits at each position
    ids_tensor = torch.tensor([input_ids], dtype=torch.long)
    all_logits = []

    with torch.no_grad():
        # Prefill: get logits at each input position
        outputs = model(ids_tensor)
        prefill_logits = outputs.logits[0].numpy()  # (seq_len, vocab)
        for i in range(len(input_ids)):
            all_logits.append(prefill_logits[i])

        # Generate: autoregressive
        gen_ids = list(input_ids)
        for _ in range(max_new_tokens):
            next_token = int(np.argmax(all_logits[-1]))
            gen_ids.append(next_token)
            ids_tensor = torch.tensor([gen_ids], dtype=torch.long)
            outputs = model(ids_tensor)
            all_logits.append(outputs.logits[0, -1].numpy())

    return all_logits


def compare_logits(trt_logits, hf_logits, atol, top_k=10):
    """Compare logit arrays step by step. Returns (max_diff, report_lines)."""
    n = min(len(trt_logits), len(hf_logits))
    max_diff = 0.0
    lines = []

    for step in range(n):
        trt_l = trt_logits[step]
        hf_l = hf_logits[step]

        if trt_l.shape != hf_l.shape:
            lines.append(f"  step {step}: shape mismatch "
                         f"trt={trt_l.shape} hf={hf_l.shape}")
            continue

        # Full logit comparison
        diff = np.abs(trt_l - hf_l)
        step_max = float(diff.max())
        max_diff = max(max_diff, step_max)

        # Top-K token agreement
        trt_top = set(np.argsort(trt_l)[-top_k:])
        hf_top = set(np.argsort(hf_l)[-top_k:])
        overlap = len(trt_top & hf_top)

        trt_argmax = int(np.argmax(trt_l))
        hf_argmax = int(np.argmax(hf_l))
        argmax_match = "Y" if trt_argmax == hf_argmax else "N"

        lines.append(
            f"  step {step:3d}: max_diff={step_max:10.6f}  "
            f"argmax_match={argmax_match}  "
            f"top{top_k}_overlap={overlap}/{top_k}")

    return max_diff, lines


def main():
    parser = argparse.ArgumentParser(
        description="Pure-Python E2E logit comparison: TRT vs HF transformers")
    parser.add_argument("--model", required=True,
                        help="HF repo ID or local model directory")
    parser.add_argument("--prompt", default="",
                        help="Single prompt (overrides --battery)")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--max-cache-length", type=int, default=64)
    parser.add_argument("--atol", type=float, default=1e-3,
                        help="Absolute tolerance for logit comparison")
    parser.add_argument("--battery", action="store_true",
                        help="Run standard prompt battery")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Allow executing custom Python code from the "
                             "model repository (required for models without "
                             "native transformers support)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    prompts = []
    if args.prompt:
        prompts = [("custom", args.prompt)]
    elif args.battery:
        prompts = STANDARD_PROMPTS
    else:
        prompts = [("default", "The capital of France is")]

    # Build engine once
    engine_plan, config, model_dir = build_trt_engine(
        args.model, args.max_cache_length, args.verbose)

    # Load HF tokenizer for encoding prompts
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=args.trust_remote_code)

    all_passed = True
    for label, prompt in prompts:
        print(f"\n{'=' * 60}")
        print(f"Prompt [{label}]: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"{'=' * 60}")

        input_ids = tokenizer.encode(prompt)
        print(f"  Input tokens: {len(input_ids)}")

        # Run TRT
        print(f"  Running TRT ...", file=sys.stderr)
        trt_logits = run_trt(
            engine_plan, config, input_ids,
            args.max_new_tokens, args.max_cache_length)

        # Run HF
        print(f"  Running HF ...", file=sys.stderr)
        hf_logits = run_hf(model_dir, input_ids, args.max_new_tokens,
                           trust_remote_code=args.trust_remote_code)

        # Compare
        max_diff, report = compare_logits(trt_logits, hf_logits, args.atol)

        # Decode generated text
        trt_gen_ids = [int(np.argmax(l)) for l in trt_logits[len(input_ids) - 1:]]
        hf_gen_ids = [int(np.argmax(l)) for l in hf_logits[len(input_ids) - 1:]]
        trt_text = tokenizer.decode(trt_gen_ids, skip_special_tokens=True)
        hf_text = tokenizer.decode(hf_gen_ids, skip_special_tokens=True)

        print(f"  TRT text: {trt_text[:120]}")
        print(f"  HF  text: {hf_text[:120]}")
        print(f"  Text match: {trt_text.strip() == hf_text.strip()}")
        print()

        for line in report:
            print(line)

        passed = max_diff <= args.atol
        print(f"\n  max_abs_logit_diff: {max_diff:.6f}")
        print(f"  atol: {args.atol}")
        print(f"  {'PASS' if passed else 'FAIL'}")

        if not passed:
            all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
