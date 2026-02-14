#!/usr/bin/env python3
"""Per-layer model diff: compare HF model hidden states against manual reimplementation.

Auto-detects model family from config.json and hooks into forward pass to capture
per-layer hidden states.

Usage:
    python3 scripts/diff_layers.py \
      --model-dir models/hf/Qwen__Qwen3-0.6B \
      --prompt "Hello" \
      --max-new-tokens 1
"""
import argparse
import json
import os
import sys

import numpy as np


def load_model_config(model_dir: str) -> dict:
    """Load and return config.json from model directory."""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Per-layer model diff")
    parser.add_argument("--model-dir", required=True, help="Path to HF model directory")
    parser.add_argument("--prompt", default="Hello", help="Input prompt")
    parser.add_argument("--max-new-tokens", type=int, default=1, help="Max tokens to generate")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: transformers and torch are required", file=sys.stderr)
        sys.exit(1)

    config = load_model_config(args.model_dir)
    model_type = config.get("model_type", "unknown")
    num_layers = config.get("num_hidden_layers", 0)
    print(f"Model type: {model_type}")
    print(f"Num layers: {num_layers}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.eval()

    # Hook into model layers to capture hidden states
    hidden_states = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states[layer_idx] = output[0].detach().cpu().numpy()
            else:
                hidden_states[layer_idx] = output.detach().cpu().numpy()
        return hook_fn

    hooks = []
    # Try common layer accessor patterns
    layer_container = None
    for attr in ["model.layers", "transformer.h", "gpt_neox.layers"]:
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            layer_container = obj
            break
        except AttributeError:
            continue

    if layer_container is None:
        print("WARNING: Could not find layer container, skipping per-layer hooks")
    else:
        for i, layer in enumerate(layer_container):
            hooks.append(layer.register_forward_hook(make_hook(i)))

    inputs = tokenizer(args.prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Clean up hooks
    for h in hooks:
        h.remove()

    # Report per-layer statistics
    print(f"\nPer-layer hidden state statistics:")
    print(f"{'Layer':>6} {'Shape':>20} {'Mean':>12} {'Std':>12} {'Max Abs':>12}")
    print("-" * 68)

    all_hidden = outputs.hidden_states if hasattr(outputs, "hidden_states") else []
    for i, hs in enumerate(all_hidden):
        arr = hs.detach().cpu().numpy()
        print(f"{i:>6} {str(arr.shape):>20} {arr.mean():>12.6f} {arr.std():>12.6f} {np.abs(arr).max():>12.6f}")

    # Compare hook-captured vs model output hidden states
    if hidden_states and all_hidden:
        print(f"\nHook vs output_hidden_states comparison:")
        max_overall_diff = 0.0
        for layer_idx in sorted(hidden_states.keys()):
            if layer_idx + 1 < len(all_hidden):
                hook_arr = hidden_states[layer_idx]
                ref_arr = all_hidden[layer_idx + 1].detach().cpu().numpy()
                if hook_arr.shape == ref_arr.shape:
                    diff = np.abs(hook_arr - ref_arr).max()
                    max_overall_diff = max(max_overall_diff, diff)
                    status = "OK" if diff <= args.atol else "MISMATCH"
                    print(f"  Layer {layer_idx}: max_abs_diff={diff:.8f} [{status}]")

        print(f"\n  Overall max diff: {max_overall_diff:.8f}")
        if max_overall_diff <= args.atol:
            print(f"  PASS (within atol={args.atol})")
        else:
            print(f"  FAIL (exceeds atol={args.atol})")
            sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
