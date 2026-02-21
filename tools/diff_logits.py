#!/usr/bin/env python3
"""Pure-Python E2E logit comparison between TRT engine and HF transformers.

No C++ binary needed. Builds a TRT engine via trtf_build, runs inference
in Python, and compares per-step logits against HF transformers.

Usage:
    python3 tools/diff_logits.py \
      --model Qwen/Qwen3-0.6B \
      --prompt "The capital of France is" \
      --max-new-tokens 10 --atol 1e-3

    python3 tools/diff_logits.py \
      --model models/hf/Qwen__Qwen3-0.6B --battery
"""
import argparse
import sys

import numpy as np

from tool_helpers import build_trt_engine, load_hf_model as _load_hf_model

STANDARD_PROMPTS = [
    ("factual", "The capital of France is"),
    ("reasoning", "Explain why water boils at 100 degrees Celsius."),
    ("code", "Write a Python function that checks if a number is prime:"),
    ("multi-turn", "User: What is 2+2?\nAssistant:"),
]


def run_trt(engine_plan, config, input_ids, max_new_tokens, max_cache_length):
    """Run TRT inference, return list of logit arrays (one per step)."""
    # Use MambaTrtRunner for SSM models, TrtRunner for standard decoders.
    if config.model_type.lower() == "mamba":
        from trtf_build.debug_runner import MambaTrtRunner
        runner = MambaTrtRunner(
            engine_plan=engine_plan,
            num_layers=config.num_hidden_layers,
        )
    else:
        from trtf_build.debug_runner import TrtRunner
        runner = TrtRunner(
            engine_plan=engine_plan,
            max_cache_length=max_cache_length,
            num_layers=config.num_hidden_layers,
        )

    results = runner.generate(input_ids, max_new_tokens)
    return [r["logits"].flatten() for r in results]


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


def run_as_diff_test(ctx):
    """Framework entry point. Returns DiffResult."""
    from diff_framework.protocol import DiffResult
    import time as _time

    t0 = _time.monotonic()
    try:
        engine_plan, config, model_dir = build_trt_engine(
            ctx.model, ctx.max_cache_length, ctx.verbose)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=ctx.trust_remote_code)

        prompts = STANDARD_PROMPTS

        worst_diff = 0.0
        all_passed = True
        details_lines = []

        for label, prompt in prompts:
            input_ids = tokenizer.encode(prompt)
            trt_logits = run_trt(
                engine_plan, config, input_ids,
                ctx.max_new_tokens, ctx.max_cache_length)
            hf_logits = run_hf(model_dir, input_ids, ctx.max_new_tokens,
                               trust_remote_code=ctx.trust_remote_code)
            max_diff, report = compare_logits(trt_logits, hf_logits, ctx.atol)
            worst_diff = max(worst_diff, max_diff)
            passed = max_diff <= ctx.atol
            if not passed:
                all_passed = False
            details_lines.append(
                f"[{label}] max_diff={max_diff:.6f} "
                f"{'PASS' if passed else 'FAIL'}")

        return DiffResult(
            test_name="logit_diff", model=ctx.model,
            runtime_strategy=ctx.runtime_strategy,
            passed=all_passed,
            status="PASS" if all_passed else "FAIL",
            message=f"max_abs_logit_diff={worst_diff:.6f} (atol={ctx.atol})",
            metrics={"max_abs_diff": worst_diff, "atol": ctx.atol,
                     "num_prompts": len(prompts)},
            duration_s=_time.monotonic() - t0,
            details="\n".join(details_lines))
    except Exception as e:
        return DiffResult.error(
            "logit_diff", ctx.model, ctx.runtime_strategy, str(e))


def run_as_diff_test(ctx):
    """Framework entry point. Returns DiffResult."""
    from diff_framework.protocol import DiffResult
    import time as _time

    t0 = _time.monotonic()
    try:
        engine_plan, config, model_dir = build_trt_engine(
            ctx.model, ctx.max_cache_length, ctx.verbose)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=ctx.trust_remote_code)

        prompts = STANDARD_PROMPTS
        worst_diff = 0.0
        all_passed = True
        details_lines = []

        for label, prompt in prompts:
            input_ids = tokenizer.encode(prompt)
            trt_logits = run_trt(
                engine_plan, config, input_ids,
                ctx.max_new_tokens, ctx.max_cache_length)
            hf_logits = run_hf(model_dir, input_ids, ctx.max_new_tokens,
                               trust_remote_code=ctx.trust_remote_code)
            max_diff, lines = compare_logits(trt_logits, hf_logits, ctx.atol)
            worst_diff = max(worst_diff, max_diff)
            passed = max_diff <= ctx.atol
            if not passed:
                all_passed = False
            details_lines.append(
                f"[{label}] max_diff={max_diff:.6f} "
                f"{'PASS' if passed else 'FAIL'}")

        return DiffResult(
            test_name="logit_diff", model=ctx.model,
            runtime_strategy=ctx.runtime_strategy,
            passed=all_passed,
            status="PASS" if all_passed else "FAIL",
            message=f"max_abs_logit_diff={worst_diff:.6f} (atol={ctx.atol})",
            metrics={"max_abs_diff": worst_diff, "atol": ctx.atol},
            duration_s=_time.monotonic() - t0,
            details="\n".join(details_lines))
    except Exception as e:
        return DiffResult.error(
            "logit_diff", ctx.model, ctx.runtime_strategy, str(e))


if __name__ == "__main__":
    main()
