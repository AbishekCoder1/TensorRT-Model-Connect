"""Orchestrator: load model → build engine → write bundle."""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import ModelConfig
from .families import find_plugin, _ALL_PLUGINS
from .bundle_writer import BundleInfo, BundleSection, write_bundle

# Standard HF file patterns to download (matches what the builder needs).
_HF_ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "*.model",
]


def _resolve_model(model_id_or_path: str) -> str:
    """Resolve a HuggingFace repo ID or local path to a local directory.

    If model_id_or_path is an existing directory with config.json, returns it
    directly. Otherwise, downloads via huggingface_hub.snapshot_download().
    """
    local = Path(model_id_or_path)
    if local.is_dir() and (local / "config.json").exists():
        return str(local)

    # Treat as HuggingFace repo ID — download to HF cache.
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for auto-downloading models. "
            "Install it with: pip install huggingface_hub"
        )

    print(f"[trtf-build] Downloading {model_id_or_path} ...", file=sys.stderr)
    local_dir = snapshot_download(
        repo_id=model_id_or_path,
        allow_patterns=_HF_ALLOW_PATTERNS,
    )
    print(f"[trtf-build] Downloaded to {local_dir}", file=sys.stderr)
    return local_dir


def _get_trt_version() -> str:
    try:
        import tensorrt as trt
        return trt.__version__
    except Exception:
        return "unknown"


def _get_gpu_name() -> str:
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return ""


def build_bundle(
    model_dir: str,
    output_path: str,
    max_cache_length: int = 256,
    *,
    verbose: bool = False,
) -> None:
    """Full pipeline: load HF model → build TRT engine → write .trtfb bundle.

    Args:
        model_dir: Path to HF model directory with config.json + safetensors.
        output_path: Where to write the .trtfb bundle.
        max_cache_length: KV cache length for the engine.
        verbose: Print detailed logs.
    """
    model_dir_path = Path(model_dir)
    t0 = time.monotonic()

    # 1. Parse config
    config = ModelConfig.from_dir(model_dir_path)
    print(f"[trtf-build] Model: {config.model_type} "
          f"(layers={config.num_hidden_layers}, hidden={config.hidden_size}, "
          f"vocab={config.vocab_size})", file=sys.stderr)

    # 2. Find family plugin
    plugin = find_plugin(config.model_type)
    if plugin is None:
        supported = ", ".join(p.name for p in _ALL_PLUGINS)
        raise ValueError(
            f"No family plugin for model_type={config.model_type!r}. "
            f"Supported: {supported}")

    print(f"[trtf-build] Family: {plugin.name}", file=sys.stderr)

    # 3. Load weights
    t1 = time.monotonic()
    print(f"[trtf-build] Loading weights ...", file=sys.stderr)
    weights = plugin.load_weights(str(model_dir_path), config)
    t2 = time.monotonic()
    print(f"[trtf-build] Weights loaded [{t2 - t1:.1f}s]", file=sys.stderr)

    # 4. Build TRT engine
    print(f"[trtf-build] Building TRT engine (cache={max_cache_length}) ...",
          file=sys.stderr)
    engine_plan = plugin.build_engine(
        config, weights, max_cache_length, verbose=verbose)
    t3 = time.monotonic()
    print(f"[trtf-build] Engine built [{t3 - t2:.1f}s] "
          f"({len(engine_plan) / (1024 * 1024):.1f} MB)", file=sys.stderr)

    # 5. Write bundle
    info = BundleInfo(
        model_id=model_dir_path.name,
        model_type=config.model_type,
        family=plugin.name,
        trt_version=_get_trt_version(),
        gpu_name=_get_gpu_name(),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_cache_length=max_cache_length,
    )

    sections = [BundleSection("engine_plan", engine_plan)]

    # Embed tokenizer + config files
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        file_path = model_dir_path / filename
        if file_path.exists():
            sections.append(BundleSection(filename, file_path.read_bytes()))

    write_bundle(output_path, info, sections)
    t4 = time.monotonic()
    print(f"[trtf-build] Bundle saved: {output_path} [{t4 - t0:.1f}s total]",
          file=sys.stderr)


def build(
    model_id_or_path: str,
    output_path: str,
    max_cache_length: int = 256,
    *,
    verbose: bool = False,
) -> None:
    """Build a .trtfb bundle from a HuggingFace model ID or local path.

    Like HF transformers, accepts either:
    - A HuggingFace repo ID: ``"Qwen/Qwen3-0.6B"`` (auto-downloads)
    - A local directory: ``"models/hf/Qwen__Qwen3-0.6B"``

    Args:
        model_id_or_path: HF repo ID or local directory with config.json + safetensors.
        output_path: Where to write the .trtfb bundle.
        max_cache_length: KV cache length for the engine.
        verbose: Print detailed TRT builder logs.
    """
    model_dir = _resolve_model(model_id_or_path)
    build_bundle(model_dir, output_path, max_cache_length, verbose=verbose)
