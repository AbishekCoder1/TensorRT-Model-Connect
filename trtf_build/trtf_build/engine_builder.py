"""Orchestrator: load model → build engine → write bundle."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import ModelConfig
from .families import find_plugin, find_diffusion_plugin, _ALL_PLUGINS
from .bundle_writer import BundleInfo, BundleSection, write_bundle

# Standard HF file patterns to download (matches what the builder needs).
_HF_ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "*.model",
    # Diffusers format
    "model_index.json",
    "*/config.json",
    "*/model.safetensors",
    "*/model-*.safetensors",
    "*/model.safetensors.index.json",
    "*/diffusion_pytorch_model.safetensors",
    "*/diffusion_pytorch_model-*.safetensors",
    "*/diffusion_pytorch_model.safetensors.index.json",
    "scheduler/*",
    "tokenizer/*",
]


def _resolve_model(model_id_or_path: str) -> str:
    """Resolve a HuggingFace repo ID or local path to a local directory.

    If model_id_or_path is an existing directory with config.json, returns it
    directly. Otherwise, downloads via huggingface_hub.snapshot_download().
    """
    local = Path(model_id_or_path)
    if local.is_dir() and ((local / "config.json").exists()
                           or (local / "model_index.json").exists()):
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


def _ensure_tokenizer_json(model_dir: Path) -> None:
    """If the model directory lacks tokenizer.json, generate it from the
    slow tokenizer using HF transformers. This ensures the C++ runtime can
    always load the tokenizer via AutoTokenizer."""
    if (model_dir / "tokenizer.json").exists():
        return
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False)
        tok.save_pretrained(str(model_dir))
        if (model_dir / "tokenizer.json").exists():
            print("[trtf-build] Generated tokenizer.json from slow tokenizer",
                  file=sys.stderr)
    except Exception as e:
        print(f"[trtf-build] Warning: could not generate tokenizer.json: {e}",
              file=sys.stderr)


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
    model_id_or_path_orig = getattr(
        build_bundle, '_model_id_or_path_orig', model_dir)
    t0 = time.monotonic()

    # Detect diffusers format (model_index.json present)
    is_diffusers = (model_dir_path / "model_index.json").exists()

    if is_diffusers:
        _build_diffusion_bundle(
            model_dir_path, output_path, max_cache_length,
            verbose=verbose, t0=t0)
        return

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

    # 4b. Build vision engine (optional, VL models only)
    vision_plan = None
    build_vision = getattr(plugin, 'build_vision_engine', None)
    if build_vision is not None:
        print(f"[trtf-build] Building vision encoder engine ...",
              file=sys.stderr)
        vision_plan = build_vision(
            str(model_dir_path), config, weights, verbose=verbose)
        if vision_plan is not None:
            t3b = time.monotonic()
            print(f"[trtf-build] Vision engine built [{t3b - t3:.1f}s] "
                  f"({len(vision_plan) / (1024 * 1024):.1f} MB)",
                  file=sys.stderr)

    # 4c. Build extra engines (optional, multi-engine models like Bark)
    extra_engines = {}
    build_extra = getattr(plugin, 'build_extra_engines', None)
    if build_extra is not None:
        print(f"[trtf-build] Building extra engines ...", file=sys.stderr)
        extra_engines = build_extra(
            config, weights, max_cache_length, verbose=verbose) or {}
        t3c = time.monotonic()
        for ename, eplan in extra_engines.items():
            print(f"[trtf-build]   {ename}: {len(eplan) / (1024 * 1024):.1f} MB",
                  file=sys.stderr)

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
        runtime_strategy=getattr(plugin, "runtime_strategy", ""),
    )

    sections = [BundleSection("engine_plan", engine_plan)]

    # Add vision engine section if present
    if vision_plan is not None:
        sections.append(BundleSection("vision_engine_plan", vision_plan))

    # Add extra engine sections (coarse, fine, codec for Bark, etc.)
    for ename, eplan in extra_engines.items():
        sections.append(BundleSection(ename, eplan))

    # If model lacks tokenizer.json (fast format), generate it from the
    # slow tokenizer so the C++ runtime can always load via AutoTokenizer.
    # Skip for non-text models (segmentation, audio) that don't use tokenizers.
    runtime_strategy = getattr(plugin, "runtime_strategy", "")
    if runtime_strategy not in ("segmentation", "neural_operator", "object_detection", "prompted_segmentation"):
        _ensure_tokenizer_json(model_dir_path)

    # Inject encoder_only config overrides
    if runtime_strategy == "encoder_only":
        # Use max_cache_length as max_seq_length for encoder
        pass

    # Embed tokenizer + config files.
    # For config.json, inject runtime_strategy if the plugin provides one.
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json",
                     "vocab.json", "merges.txt", "special_tokens_map.json",
                     "tokenizer.model", "preprocessor_config.json"):
        file_path = model_dir_path / filename
        if file_path.exists():
            data = file_path.read_bytes()
            # Inject runtime_strategy and VL fields into config.json.
            if filename == "config.json":
                cfg_dict = json.loads(data)
                runtime_strategy = getattr(plugin, "runtime_strategy", None)
                if runtime_strategy:
                    cfg_dict["runtime_strategy"] = runtime_strategy
                embed_input = getattr(plugin, "embed_input", False)
                if embed_input:
                    cfg_dict["embed_input"] = True
                if vision_plan is not None:
                    cfg_dict["has_vision_engine"] = True
                # Inject VL config from plugin (image_token_id, prompt template, etc.)
                get_vl_config = getattr(plugin, 'get_vl_config', None)
                if get_vl_config is not None:
                    vl_cfg = get_vl_config(config)
                    if vl_cfg is not None:
                        cfg_dict.update(vl_cfg)
                # Inject segmentation config from plugin
                get_seg_config = getattr(plugin, 'get_segmentation_config', None)
                if get_seg_config is not None:
                    seg_cfg = get_seg_config(config)
                    if seg_cfg is not None:
                        cfg_dict.update(seg_cfg)
                # Inject detection config from plugin
                get_det_config = getattr(plugin, 'get_detection_config', None)
                if get_det_config is not None:
                    det_cfg = get_det_config(config)
                    if det_cfg is not None:
                        cfg_dict.update(det_cfg)
                # Inject audio config from plugin
                get_audio_config = getattr(plugin, 'get_audio_config', None)
                if get_audio_config is not None:
                    audio_cfg = get_audio_config(config)
                    if audio_cfg is not None:
                        cfg_dict.update(audio_cfg)
                # Inject generic config overrides from plugin.
                # Build the final dict so overrides appear FIRST in the
                # serialized JSON.  The C++ fast_path_config parser uses
                # flat text search (text.find) which picks up the first
                # occurrence of a key.  For models with nested configs
                # (e.g. Qwen3-Omni thinker_config.text_config) the nested
                # copy of "hidden_size" etc. would otherwise shadow the
                # top-level value.
                get_overrides = getattr(plugin, 'get_bundle_config_overrides', None)
                if get_overrides is not None:
                    overrides = get_overrides(config)
                    if overrides is not None:
                        # Put overrides first, then original dict.  Dict
                        # union preserves insertion order; overrides keys
                        # appear before any nested dicts.
                        merged = dict(overrides)
                        merged.update(cfg_dict)
                        # Ensure overrides win for top-level keys.
                        merged.update(overrides)
                        cfg_dict = merged
                data = json.dumps(cfg_dict, indent=2).encode("utf-8")
            sections.append(BundleSection(filename, data))

    write_bundle(output_path, info, sections)
    t4 = time.monotonic()
    print(f"[trtf-build] Bundle saved: {output_path} [{t4 - t0:.1f}s total]",
          file=sys.stderr)


def _build_diffusion_bundle(
    model_dir_path: Path,
    output_path: str,
    max_cache_length: int,
    *,
    verbose: bool = False,
    t0: float = 0.0,
) -> None:
    """Build a diffusion model bundle from a diffusers-format directory."""
    import json as json_module

    # Parse model_index.json to determine pipeline type
    model_index = json.loads(
        (model_dir_path / "model_index.json").read_text())
    pipeline_class = model_index.get("_class_name", "")

    print(f"[trtf-build] Diffusion pipeline: {pipeline_class}",
          file=sys.stderr)

    # Auto-discover plugin from pipeline_classes attribute
    plugin = find_diffusion_plugin(pipeline_class)
    if plugin is None:
        # Fallback: try model_type-based lookup with lowercased pipeline class
        plugin = find_plugin(pipeline_class.lower())
    if plugin is None:
        supported = ", ".join(p.name for p in _ALL_PLUGINS)
        raise ValueError(
            f"No family plugin for diffusion pipeline {pipeline_class!r}. "
            f"Supported: {supported}")

    model_type = getattr(plugin, 'name', pipeline_class.lower())
    config = ModelConfig(model_type=model_type, raw=model_index)

    print(f"[trtf-build] Family: {plugin.name}", file=sys.stderr)

    # Load weights (lightweight — just paths for diffusion)
    t1 = time.monotonic()
    weights = plugin.load_weights(str(model_dir_path), config)
    t2 = time.monotonic()

    # Propagate transformer config to ModelConfig so get_diffusion_config can access it
    if "_transformer_config" in weights:
        config.raw["_transformer_config"] = weights["_transformer_config"]

    # Build all component engines
    build_components = getattr(plugin, 'build_components', None)
    if build_components is None:
        raise ValueError(
            f"Plugin {plugin.name} does not support build_components()")

    components = build_components(
        str(model_dir_path), config, weights, verbose=verbose)
    if components is None:
        raise ValueError(
            f"Plugin {plugin.name}.build_components() returned None")

    t3 = time.monotonic()
    print(f"[trtf-build] All engines built [{t3 - t1:.1f}s]", file=sys.stderr)

    # Assemble bundle sections
    sections = []

    # Text encoder plans
    for i, (enc_name, enc_plan) in enumerate(components["text_encoders"]):
        sections.append(BundleSection(
            f"text_encoder_{i}_plan", enc_plan))
        print(f"  text_encoder_{i} ({enc_name}): "
              f"{len(enc_plan) / (1024 * 1024):.1f} MB", file=sys.stderr)

    # Denoiser plan
    denoiser_plan = components["denoiser"]
    sections.append(BundleSection("denoiser_plan", denoiser_plan))
    print(f"  denoiser: {len(denoiser_plan) / (1024 * 1024):.1f} MB",
          file=sys.stderr)

    # VAE decoder plan
    vae_plan = components["vae_decoder"]
    sections.append(BundleSection("vae_decoder_plan", vae_plan))
    print(f"  vae_decoder: {len(vae_plan) / (1024 * 1024):.1f} MB",
          file=sys.stderr)

    # Preprocessor weights (patch embedding, timestep MLP, text projection)
    if "preprocessor_weights" in components:
        pp_data = components["preprocessor_weights"]
        sections.append(BundleSection("preprocessor_weights", pp_data))
        print(f"  preprocessor_weights: {len(pp_data) / (1024):.1f} KB",
              file=sys.stderr)

    # Build config.json with diffusion config injected
    cfg_dict = {
        "model_type": model_type,
        "runtime_strategy": getattr(plugin, "runtime_strategy", "diffusion"),
        "num_text_encoders": len(components["text_encoders"]),
    }

    # Inject diffusion config from plugin
    get_diff_config = getattr(plugin, 'get_diffusion_config', None)
    if get_diff_config is not None:
        diff_cfg = get_diff_config(config)
        if diff_cfg is not None:
            cfg_dict.update(diff_cfg)

    cfg_data = json.dumps(cfg_dict, indent=2).encode("utf-8")
    sections.append(BundleSection("config.json", cfg_data))

    # Embed tokenizer files from tokenizer subdirectories.
    # Multi-encoder models (FLUX, SD3) have tokenizer/ (CLIP) and
    # tokenizer_2/ (T5).  Prefer tokenizer_2/ if it has tokenizer.json
    # (fast tokenizer format) since T5 provides the main text conditioning.
    # Fall back to tokenizer/ for single-tokenizer models (Wan, Z-Image).
    _tok_filenames = ("tokenizer.json", "tokenizer_config.json",
                      "special_tokens_map.json", "vocab.json",
                      "merges.txt", "spiece.model", "tokenizer.model")
    _tok_embedded = set()

    for tok_subdir in ("tokenizer_2", "tokenizer"):
        tokenizer_dir = model_dir_path / tok_subdir
        if not tokenizer_dir.is_dir():
            continue
        for filename in _tok_filenames:
            if filename in _tok_embedded:
                continue  # already embedded from higher-priority dir
            file_path = tokenizer_dir / filename
            if file_path.exists():
                sections.append(BundleSection(filename, file_path.read_bytes()))
                _tok_embedded.add(filename)

    # For dual-tokenizer models (FLUX): also embed CLIP tokenizer files
    # under prefixed names so the C++ runtime can create a separate CLIP
    # tokenizer.  CLIP lives in tokenizer/ (BPE with vocab.json + merges.txt).
    _clip_file_map = {
        "vocab.json": "clip_vocab.json",
        "merges.txt": "clip_merges.txt",
        "tokenizer_config.json": "clip_tokenizer_config.json",
        "special_tokens_map.json": "clip_special_tokens_map.json",
    }
    clip_tokenizer_dir = model_dir_path / "tokenizer"
    if clip_tokenizer_dir.is_dir() and (model_dir_path / "tokenizer_2").is_dir():
        for src_name, dst_name in _clip_file_map.items():
            file_path = clip_tokenizer_dir / src_name
            if file_path.exists():
                sections.append(BundleSection(dst_name, file_path.read_bytes()))

    # Write bundle
    info = BundleInfo(
        model_id=model_dir_path.name,
        model_type=model_type,
        family=plugin.name,
        trt_version=_get_trt_version(),
        gpu_name=_get_gpu_name(),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        runtime_strategy=getattr(plugin, "runtime_strategy", "diffusion"),
    )

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
    build_bundle._model_id_or_path_orig = model_id_or_path
    build_bundle(model_dir, output_path, max_cache_length, verbose=verbose)
