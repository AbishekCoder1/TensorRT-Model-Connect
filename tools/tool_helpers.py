"""Shared utilities for standard decoder/comparison tools.

Used by: diff_logits.py, diff_layers.py, perf_compare.py,
         debug_diffusion_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def build_trt_engine(model_id_or_path, max_cache_length, verbose, *, tag="diff"):
    """Build TRT engine and return (engine_plan_bytes, config, model_dir)."""
    from trtf_build.engine_builder import _resolve_model
    from trtf_build.config import ModelConfig
    from trtf_build.families import find_plugin

    model_dir = _resolve_model(model_id_or_path)
    config = ModelConfig.from_dir(model_dir)
    plugin = find_plugin(config.model_type)
    if plugin is None:
        raise ValueError(f"No plugin for model_type={config.model_type!r}")

    print(f"[{tag}] Loading weights ({config.model_type}) ...", file=sys.stderr)
    weights = plugin.load_weights(model_dir, config)
    print(f"[{tag}] Building TRT engine (cache={max_cache_length}) ...",
          file=sys.stderr)
    engine_plan = plugin.build_engine(
        config, weights, max_cache_length, verbose=verbose)
    print(f"[{tag}] Engine built ({len(engine_plan) / 1e6:.1f} MB)",
          file=sys.stderr)

    return engine_plan, config, model_dir


def load_hf_model(model_dir, *, trust_remote_code=False, torch_dtype=None,
                   tag="diff"):
    """Load HF model with VL detection and trust_remote_code fallback.

    Returns the model on CPU. Callers should move to device and call eval()
    as needed.
    """
    import json
    import torch
    from transformers import AutoModelForCausalLM

    if torch_dtype is None:
        torch_dtype = torch.float32

    # Check if this is a VL model that requires a different AutoModel class.
    config_path = Path(model_dir) / "config.json"
    is_vl_model = False
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        model_type = cfg.get("model_type", "").lower()
        if "vl" in model_type or "vision" in model_type:
            is_vl_model = True

    if is_vl_model:
        from transformers import AutoModelForImageTextToText
        print(f"[{tag}] Loading VL model via AutoModelForImageTextToText ...",
              file=sys.stderr)
        return AutoModelForImageTextToText.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype)

    try:
        return AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=False, torch_dtype=torch_dtype)
    except (ValueError, KeyError, ImportError) as e:
        if trust_remote_code:
            print(f"[{tag}] Native loading failed ({e}), "
                  f"retrying with trust_remote_code=True ...",
                  file=sys.stderr)
            return AutoModelForCausalLM.from_pretrained(
                model_dir, trust_remote_code=True, torch_dtype=torch_dtype)
        raise ValueError(
            f"Failed to load model from {model_dir} without custom code. "
            f"If this model requires custom code, re-run with "
            f"--trust-remote-code. Original error: {e}"
        ) from e


def cosine_sim(a, b):
    """Compute cosine similarity between two arrays."""
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def compare_arrays(name, ours, ref, atol):
    """Compare two arrays and print PASS/FAIL. Returns True if within tolerance."""
    diff = np.abs(ours.flatten() - ref.flatten())
    mx, mn = float(diff.max()), float(diff.mean())
    cs = cosine_sim(ours, ref)
    ok = mx <= atol
    tag = "PASS" if ok else "FAIL"
    print(f"   {tag}: max_diff={mx:.6f}, mean_diff={mn:.6f}, cosine_sim={cs:.6f}")
    return ok
