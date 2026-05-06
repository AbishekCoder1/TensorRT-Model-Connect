#!/usr/bin/env python3
"""Pure-Python E2E logit comparison between TRT engine and HF transformers.

No C++ binary needed. Builds a TRT engine via tensorrt_model_connect, runs inference
in Python, and compares per-step logits against HF transformers.

Usage:
    python3 tools/diff_logits.py \
      --model Qwen/Qwen3-0.6B \
      --prompt "The capital of France is" \
      --max-new-tokens 10 --atol 1e-3

    python3 tools/diff_logits.py \
      --model models/hf/Qwen__Qwen3-0.6B --battery

    # Machine-readable JSON output for automated accuracy gating:
    python3 tools/diff_logits.py \
      --model Qwen/Qwen3-0.6B --battery --json results.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

STANDARD_PROMPTS = [
    ("factual", "The capital of France is"),
    ("reasoning", "Explain why water boils at 100 degrees Celsius."),
    ("code", "Write a Python function that checks if a number is prime:"),
    ("multi-turn", "User: What is 2+2?\nAssistant:"),
]


def build_trt_engine(model_id_or_path, max_cache_length, verbose):
    """Build TRT engine and return (engine_plan_bytes, config, model_dir).

    For Whisper, also returns encoder_plan via config._encoder_plan.
    """
    from tensorrt_model_connect.engine_builder import _resolve_model
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.families import find_plugin

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

    # For Whisper: also build encoder engine
    build_vision = getattr(plugin, 'build_vision_engine', None)
    if build_vision is not None and config.model_type.lower() == "whisper":
        print(f"[diff] Building Whisper encoder engine ...", file=sys.stderr)
        encoder_plan = build_vision(model_dir, config, weights, verbose=verbose)
        if encoder_plan is not None:
            print(f"[diff] Encoder built ({len(encoder_plan) / 1e6:.1f} MB)",
                  file=sys.stderr)
            config._encoder_plan = encoder_plan
            config._weights = weights  # save for mel info

    return engine_plan, config, model_dir


def run_trt(engine_plan, config, input_ids, max_new_tokens, max_cache_length):
    """Run TRT inference, return list of logit arrays (one per step)."""
    # Use model-specific runner based on model_type.
    if config.model_type.lower() == "mamba":
        from tensorrt_model_connect.debug_runner import MambaTrtRunner
        runner = MambaTrtRunner(
            engine_plan=engine_plan,
            num_layers=config.num_hidden_layers,
        )
    elif config.model_type.lower() == "rwkv":
        from tensorrt_model_connect.debug_runner import RwkvTrtRunner
        runner = RwkvTrtRunner(
            engine_plan=engine_plan,
            num_layers=config.num_hidden_layers,
        )
    elif config.model_type.lower() == "whisper":
        from tensorrt_model_connect.debug_runner import WhisperTrtRunner
        encoder_plan = getattr(config, '_encoder_plan', None)
        if encoder_plan is None:
            raise RuntimeError("Whisper encoder plan not built")
        raw = config.raw
        num_layers = raw.get("decoder_layers", config.num_hidden_layers)
        max_source = raw.get("max_source_positions", 1500)
        num_mel_bins = raw.get("num_mel_bins", 80)
        runner = WhisperTrtRunner(
            decoder_plan=engine_plan,
            encoder_plan=encoder_plan,
            num_layers=num_layers,
            max_cache_length=max_cache_length,
            max_source_positions=max_source,
            hidden_size=config.hidden_size,
        )
        # Run encoder on dummy mel features (zeros)
        mel_length = max_source * 2
        mel_features = np.zeros((num_mel_bins, mel_length), dtype=np.float32)
        print(f"  Running TRT encoder (mel={num_mel_bins}x{mel_length}) ...",
              file=sys.stderr)
        runner.run_encoder(mel_features)
    else:
        from tensorrt_model_connect.debug_runner import TrtRunner
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

    For vision-language models (e.g. Qwen2.5-VL), loads the full VL model
    but only uses the text decoder path for comparison.
    """
    import json
    import torch
    from transformers import AutoModelForCausalLM

    # Check if this is a VL model that requires a different AutoModel class.
    config_path = Path(model_dir) / "config.json"
    is_vl_model = False
    model_type = ""
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        model_type = cfg.get("model_type", "").lower()
        if "vl" in model_type or "vision" in model_type:
            is_vl_model = True

    if model_type == "whisper":
        from transformers import WhisperForConditionalGeneration
        print("[diff] Loading Whisper model via WhisperForConditionalGeneration ...",
              file=sys.stderr)
        return WhisperForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch.float32)

    if is_vl_model:
        from transformers import AutoModelForImageTextToText
        print("[diff] Loading VL model via AutoModelForImageTextToText ...",
              file=sys.stderr)
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code,
            torch_dtype=torch.float32)
        return model

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


def run_hf(model_dir, config, input_ids, max_new_tokens, trust_remote_code=False):
    """Run HF transformers, return list of logit arrays (one per step)."""
    import torch
    from transformers import AutoTokenizer

    model = _load_hf_model(model_dir, trust_remote_code=trust_remote_code)
    model.eval()

    # Whisper: encoder-decoder model with cross-attention
    if config.model_type.lower() == "whisper":
        return _run_hf_whisper(model, config, input_ids, max_new_tokens)

    # Standard causal LM
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


def _run_hf_whisper(model, config, input_ids, max_new_tokens):
    """Run HF Whisper encoder-decoder with dummy mel, return per-step decoder logits.

    Uses single-token-at-a-time decoding to match TRT's incremental KV cache
    behavior exactly. Each step feeds only the current token and relies on
    HF's past_key_values for context (equivalent to TRT's device KV cache).
    """
    import torch
    from transformers.modeling_outputs import BaseModelOutput

    raw = config.raw
    max_source = raw.get("max_source_positions", 1500)
    num_mel_bins = raw.get("num_mel_bins", 80)
    mel_length = max_source * 2

    # Same dummy mel features as TRT (zeros)
    mel_features = torch.zeros(1, num_mel_bins, mel_length, dtype=torch.float32)

    all_logits = []
    with torch.no_grad():
        # Run encoder
        encoder_outputs = model.model.encoder(mel_features)
        encoder_hidden = encoder_outputs.last_hidden_state  # [1, max_source, hidden]
        enc_out = BaseModelOutput(last_hidden_state=encoder_hidden)

        # Decoder: single-token steps with past_key_values (matches TRT KV cache)
        past_key_values = None
        gen_ids = list(input_ids)

        for step_idx in range(len(input_ids) + max_new_tokens):
            if step_idx < len(input_ids):
                token = input_ids[step_idx]
            else:
                token = int(np.argmax(all_logits[-1]))
                gen_ids.append(token)

            ids_tensor = torch.tensor([[token]], dtype=torch.long)

            outputs = model(
                decoder_input_ids=ids_tensor,
                encoder_outputs=enc_out,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[0, -1].numpy()  # [vocab]
            all_logits.append(logits)

    return all_logits


def _cosine_similarity(a, b):
    """Cosine similarity between two 1-D arrays. Returns 0.0 if either is zero."""
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def compare_logits(trt_logits, hf_logits, atol, top_k=10):
    """Compare logit arrays step by step.

    Returns (max_diff, report_lines, step_metrics) where step_metrics is a
    list of dicts with per-step structured data (cosine_sim, argmax_match,
    mean_abs_diff).  Steps with shape mismatches are omitted from
    step_metrics.
    """
    n = min(len(trt_logits), len(hf_logits))
    max_diff = 0.0
    lines = []
    step_metrics = []

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
        step_mean = float(diff.mean())
        max_diff = max(max_diff, step_max)

        # Cosine similarity
        cosine = _cosine_similarity(trt_l, hf_l)

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

        step_metrics.append({
            "step": step,
            "cosine_sim": cosine,
            "argmax_match": trt_argmax == hf_argmax,
            "mean_abs_diff": step_mean,
            "max_abs_diff": step_max,
        })

    return max_diff, lines, step_metrics


def _build_json_report(prompt_results, atol):
    """Build a machine-readable JSON report from accumulated prompt results.

    Args:
        prompt_results: list of dicts, each with keys:
            - label: prompt label string
            - passed: bool, whether this prompt passed the atol gate
            - max_diff: float, max absolute logit diff for this prompt
            - step_metrics: list of per-step metric dicts from compare_logits
            - trt_text: decoded TRT output text
            - hf_text: decoded HF output text
        atol: float, absolute tolerance used for the comparison

    Returns:
        dict with top-level summary fields and per-prompt details.
    """
    # Collect all step metrics across all prompts
    all_cosines = []
    all_argmax_matches = []
    all_mean_abs_diffs = []
    for pr in prompt_results:
        for sm in pr["step_metrics"]:
            all_cosines.append(sm["cosine_sim"])
            all_argmax_matches.append(sm["argmax_match"])
            all_mean_abs_diffs.append(sm["mean_abs_diff"])

    overall_pass = all(pr["passed"] for pr in prompt_results)

    # Compute aggregate metrics (safe defaults when no steps exist)
    if all_cosines:
        cosine_p5 = float(np.percentile(all_cosines, 5))
    else:
        cosine_p5 = 0.0

    if all_argmax_matches:
        top1_match_rate = float(
            sum(all_argmax_matches) / len(all_argmax_matches))
    else:
        top1_match_rate = 0.0

    if all_mean_abs_diffs:
        mean_abs_diff = float(np.mean(all_mean_abs_diffs))
    else:
        mean_abs_diff = 0.0

    # Token agreement: fraction of prompts where TRT and HF produce the
    # same decoded text (stripped).
    if prompt_results:
        text_matches = sum(
            1 for pr in prompt_results
            if pr["trt_text"].strip() == pr["hf_text"].strip()
        )
        token_agreement = float(text_matches / len(prompt_results))
    else:
        token_agreement = 0.0

    report = {
        "pass": overall_pass,
        "cosine_p5": cosine_p5,
        "top1_match_rate": top1_match_rate,
        "token_agreement": token_agreement,
        "mean_abs_diff": mean_abs_diff,
        "atol": atol,
        "num_prompts": len(prompt_results),
        "prompts": [],
    }

    for pr in prompt_results:
        prompt_entry = {
            "label": pr["label"],
            "passed": pr["passed"],
            "max_diff": pr["max_diff"],
            "trt_text": pr["trt_text"],
            "hf_text": pr["hf_text"],
            "num_steps": len(pr["step_metrics"]),
        }
        report["prompts"].append(prompt_entry)

    return report


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
    parser.add_argument("--json", metavar="PATH", default=None,
                        help="Write machine-readable JSON report to PATH")
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

    # For Whisper, use decoder start tokens as the "prompt"
    if config.model_type.lower() == "whisper":
        # Whisper decoder starts with: <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>
        # Use forced_decoder_ids or fallback to standard tokens
        start_ids = [50258, 50259, 50359, 50363]  # startoftranscript, en, transcribe, notimestamps
        prompts = [("whisper-decode", f"[decoder start tokens: {start_ids}]")]

    all_passed = True
    prompt_results = []
    for label, prompt in prompts:
        print(f"\n{'=' * 60}")
        print(f"Prompt [{label}]: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"{'=' * 60}")

        if config.model_type.lower() == "whisper":
            input_ids = start_ids
        else:
            input_ids = tokenizer.encode(prompt)
        print(f"  Input tokens: {len(input_ids)}")

        # Run TRT
        print(f"  Running TRT ...", file=sys.stderr)
        trt_logits = run_trt(
            engine_plan, config, input_ids,
            args.max_new_tokens, args.max_cache_length)

        # Run HF
        print(f"  Running HF ...", file=sys.stderr)
        hf_logits = run_hf(model_dir, config, input_ids, args.max_new_tokens,
                           trust_remote_code=args.trust_remote_code)

        # Compare
        max_diff, report, step_metrics = compare_logits(
            trt_logits, hf_logits, args.atol)

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

        prompt_results.append({
            "label": label,
            "passed": passed,
            "max_diff": max_diff,
            "step_metrics": step_metrics,
            "trt_text": trt_text,
            "hf_text": hf_text,
        })

    # Write JSON report if requested
    if args.json:
        report_dict = _build_json_report(prompt_results, args.atol)
        json_path = Path(args.json)
        json_path.write_text(json.dumps(report_dict, indent=2))
        print(f"\n[diff] JSON report written to {json_path}", file=sys.stderr)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
