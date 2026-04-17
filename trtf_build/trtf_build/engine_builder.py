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
    "model.safetensors-*.safetensors",
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


def _raise_friendly_download_error(model_id: str, exc: Exception) -> None:
    """Re-raise HF download errors with clear, actionable messages."""
    exc_type = type(exc).__name__

    if "RepositoryNotFound" in exc_type:
        raise RuntimeError(
            f"Model '{model_id}' not found on HuggingFace Hub. "
            f"Check the repo ID for typos (format: 'org/model-name'). "
            f"If it's a private repo, run: huggingface-cli login"
        ) from exc

    if "GatedRepo" in exc_type:
        raise RuntimeError(
            f"Model '{model_id}' is gated. Accept the license at "
            f"https://huggingface.co/{model_id} then run: huggingface-cli login"
        ) from exc

    if "LocalEntryNotFound" in exc_type or "EntryNotFound" in exc_type:
        raise RuntimeError(
            f"Model '{model_id}' exists but required files are missing. "
            f"The model may use a non-standard layout."
        ) from exc

    if "HTTPError" in exc_type or "ConnectionError" in exc_type:
        raise RuntimeError(
            f"Network error downloading '{model_id}': {exc}. "
            f"Check your internet connection and try again."
        ) from exc

    if "OSError" in exc_type and "disk" in str(exc).lower():
        raise RuntimeError(
            f"Disk error downloading '{model_id}': {exc}. "
            f"Check available disk space."
        ) from exc

    # Fallback: re-raise with context
    raise RuntimeError(
        f"Failed to download '{model_id}' from HuggingFace Hub: {exc}"
    ) from exc


def _is_hf_model_dir(path: Path) -> bool:
    """Return True if path contains a standard HF model entrypoint config."""
    return (path / "config.json").exists() or (path / "model_index.json").exists()


def _resolve_model(model_id_or_path: str) -> str:
    """Resolve a HuggingFace repo ID or local path to a local directory.

    If model_id_or_path is an existing directory with config.json, returns it
    directly. Otherwise, downloads via huggingface_hub.snapshot_download().
    Handles .nemo archives by extracting config and creating a synthetic dir.
    """
    local = Path(model_id_or_path)
    if local.is_dir() and _is_hf_model_dir(local):
        return str(local)

    # Handle .nemo archives (NeMo models like MagpieTTS)
    if local.is_file() and local.suffix == ".nemo":
        return _resolve_nemo_archive(local)

    # Handle HF directories that contain .nemo files
    if local.is_dir():
        nemo_files = list(local.glob("*.nemo"))
        if nemo_files:
            return _resolve_nemo_archive(nemo_files[0])

    # Treat as HuggingFace repo ID — download to HF cache.
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for auto-downloading models. "
            "Install it with: pip install huggingface_hub"
        )

    print(f"[trtf-build] Downloading {model_id_or_path} ...", file=sys.stderr)
    try:
        local_dir = snapshot_download(
            repo_id=model_id_or_path,
            allow_patterns=_HF_ALLOW_PATTERNS + ["*.nemo"],
        )
    except Exception as exc:
        _raise_friendly_download_error(model_id_or_path, exc)

    # Prefer HF config when both HF files and .nemo are present.
    dl_path = Path(local_dir)
    if _is_hf_model_dir(dl_path):
        print(f"[trtf-build] Downloaded to {local_dir}", file=sys.stderr)
        return local_dir

    # Fallback for NeMo-only snapshots.
    nemo_files = sorted(dl_path.glob("*.nemo"))
    if nemo_files:
        return _resolve_nemo_archive(nemo_files[0])

    print(f"[trtf-build] Downloaded to {local_dir}", file=sys.stderr)
    return local_dir


def _resolve_nemo_archive(nemo_path: Path) -> str:
    """Extract a .nemo archive and create a synthetic HF-compatible directory.

    NeMo .nemo files are tar archives containing model_config.yaml and
    model_weights.ckpt. We extract the YAML config, generate a synthetic
    config.json with model_type for plugin dispatch, and store the .nemo
    path for the plugin's load_weights() to use.
    """
    import json
    import tempfile

    print(f"[trtf-build] Resolving NeMo archive: {nemo_path}", file=sys.stderr)

    # Extract model_config.yaml from the tar
    import tarfile
    cfg = {}
    with tarfile.open(str(nemo_path), "r") as tar:
        for member in tar.getmembers():
            if member.name.endswith("model_config.yaml"):
                import yaml
                f = tar.extractfile(member)
                if f is not None:
                    cfg = yaml.safe_load(f)
                break

    # Determine model_type from NeMo config
    target = cfg.get("target", "")
    model_type = "unknown"
    if "MagpieTTS" in target or "magpietts" in target.lower():
        model_type = "magpie_tts"
    elif "EncDecMultiTaskModel" in target or "canary" in target.lower():
        model_type = "canary"
    elif cfg.get("model_type", ""):
        model_type = cfg["model_type"]

    # Create a temp dir that looks like an HF model dir
    tmp_dir = tempfile.mkdtemp(prefix="trtf_nemo_")
    tmp_path = Path(tmp_dir)

    # Write synthetic config.json for ModelConfig.from_dir()
    enc_cfg = cfg.get("encoder", {})
    dec_cfg = cfg.get("decoder", cfg.get("transf_decoder", {}))
    hidden = enc_cfg.get("d_model", 768)
    # Decoder fields vary by NeMo model type
    dec_layers = dec_cfg.get("n_layers",
                             dec_cfg.get("num_layers", 12))
    dec_heads = dec_cfg.get("sa_n_heads",
                            dec_cfg.get("num_attention_heads", 12))
    dec_ffn = dec_cfg.get("d_ffn",
                          dec_cfg.get("inner_size", 3072))
    synthetic_config = {
        "model_type": model_type,
        "hidden_size": hidden,
        "num_hidden_layers": dec_layers,
        "num_attention_heads": dec_heads,
        "intermediate_size": dec_ffn,
        "vocab_size": 2380,  # Will be overridden from weights
        "rms_norm_eps": 1e-5,
        "_nemo_archive_path": str(nemo_path),
    }
    with open(tmp_path / "config.json", "w") as f:
        json.dump(synthetic_config, f, indent=2)

    # Symlink the .nemo file into the temp dir for easy access
    nemo_link = tmp_path / nemo_path.name
    if not nemo_link.exists():
        import os
        os.symlink(str(nemo_path.resolve()), str(nemo_link))

    print(f"[trtf-build] NeMo resolved: model_type={model_type}, "
          f"tmp_dir={tmp_dir}", file=sys.stderr)
    return tmp_dir


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


def _detect_tokenizer_add_special_tokens(model_dir: Path) -> bool:
    """Detect whether the HF tokenizer adds special tokens (BOS/EOS) by default.

    Reads tokenizer_config.json to check add_bos_token / add_special_tokens.
    Falls back to checking if the tokenizer's encode() adds a BOS token.
    """
    # Check tokenizer_config.json for explicit add_bos_token
    tok_config_path = model_dir / "tokenizer_config.json"
    if tok_config_path.exists():
        try:
            tok_cfg = json.load(open(tok_config_path))
            # Explicit add_bos_token field (used by Llama, Nemotron, etc.)
            if "add_bos_token" in tok_cfg:
                return bool(tok_cfg["add_bos_token"])
        except Exception:
            pass

    # Fallback: instantiate the tokenizer and check if encode adds BOS
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        # Encode with and without special tokens, compare
        ids_with = tok.encode("hello", add_special_tokens=True)
        ids_without = tok.encode("hello", add_special_tokens=False)
        return ids_with != ids_without
    except Exception:
        return False


def _ensure_tokenizer_json(model_dir: Path) -> None:
    """If the model directory lacks tokenizer.json, generate it from the
    slow tokenizer using HF transformers. This ensures the C++ runtime can
    always load the tokenizer natively (BPE / WordPiece / Unigram).

    Fallback chain:
      1. AutoTokenizer(use_fast=False).save_pretrained() — works for most models
      2. SentencePiece .spm → tokenizers.Unigram conversion — for Marian / NLLB
    """
    if (model_dir / "tokenizer.json").exists():
        return

    # --- Attempt 1: standard HF slow → fast conversion ---
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False)
        tok.save_pretrained(str(model_dir))
        if (model_dir / "tokenizer.json").exists():
            print("[trtf-build] Generated tokenizer.json from slow tokenizer",
                  file=sys.stderr)
            return
    except Exception:
        pass

    # --- Attempt 2: build from SentencePiece .spm + vocab.json ---
    # Marian/NLLB models have source.spm (encoder-side SentencePiece) and
    # vocab.json (combined source+target vocabulary with IDs).  We build a
    # Unigram tokenizer.json using the full combined vocab (so IDs match the
    # TRT engine) with scores from the SPM model for source tokens and a
    # default low score for target-only tokens.
    spm_candidates = list(model_dir.glob("*.spm"))
    source_spm = model_dir / "source.spm"
    spm_path = source_spm if source_spm.exists() else (spm_candidates[0] if spm_candidates else None)
    vocab_json_path = model_dir / "vocab.json"
    if spm_path is not None:
        try:
            import sentencepiece as spm_lib
            from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders
            from tokenizers.models import Unigram

            sp = spm_lib.SentencePieceProcessor()
            sp.Load(str(spm_path))
            # Build score lookup from SPM model
            spm_scores = {}
            for i in range(sp.GetPieceSize()):
                spm_scores[sp.IdToPiece(i)] = sp.GetScore(i)
            min_score = min(spm_scores.values()) if spm_scores else 0.0
            default_score = min_score - 10.0  # worse than any real token

            # Build combined vocab with correct IDs from vocab.json
            if vocab_json_path.exists():
                with open(vocab_json_path) as f:
                    combined_vocab = json.load(f)
                # combined_vocab is {token_str: id_int}, build id-ordered list
                max_id = max(combined_vocab.values())
                vocab = [("", default_score)] * (max_id + 1)
                for token, tid in combined_vocab.items():
                    score = spm_scores.get(token, default_score)
                    vocab[tid] = (token, score)
            else:
                # Fallback: use SPM vocab only
                vocab = [(sp.IdToPiece(i), sp.GetScore(i)) for i in range(sp.GetPieceSize())]

            unk_id = combined_vocab.get("<unk>", 0) if vocab_json_path.exists() else 0

            tokenizer = Tokenizer(Unigram(vocab, unk_id))
            tokenizer.normalizer = normalizers.Sequence([
                normalizers.Prepend(prepend="\u2581"),
                normalizers.Replace(" ", "\u2581"),
            ])
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence([])
            tokenizer.decoder = decoders.Metaspace()

            out_path = str(model_dir / "tokenizer.json")
            tokenizer.save(out_path)
            print(f"[trtf-build] Generated tokenizer.json from {spm_path.name} "
                  f"({len(vocab)} tokens)", file=sys.stderr)
            return
        except Exception as e:
            print(f"[trtf-build] Warning: SentencePiece conversion failed: {e}",
                  file=sys.stderr)

    print("[trtf-build] Warning: could not generate tokenizer.json "
          "(C++ runtime may fail to create tokenizer)", file=sys.stderr)


def build_bundle(
    model_dir: str,
    output_path: str,
    max_cache_length: int = 256,
    *,
    precision: str = "fp32",
    quantize: str | None = None,
    quant_scales: str | None = None,
    quant_calibration_samples: int = 512,
    verbose: bool = False,
    kernel_artifacts: list[tuple[str, str]] | None = None,
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

    # Detect diffusers format (model_index.json present)
    is_diffusers = (model_dir_path / "model_index.json").exists()

    if is_diffusers:
        fp8_scales = getattr(build_bundle, '_fp8_scales', None)
        save_fp8_scales = getattr(build_bundle, '_save_fp8_scales', None)
        _build_diffusion_bundle(
            model_dir_path, output_path, max_cache_length,
            precision=precision, verbose=verbose, t0=t0,
            fp8_scales=fp8_scales, save_fp8_scales=save_fp8_scales)
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
    print("[trtf-build] Loading weights ...", file=sys.stderr)
    weights = plugin.load_weights(str(model_dir_path), config)
    t2 = time.monotonic()
    print(f"[trtf-build] Weights loaded [{t2 - t1:.1f}s]", file=sys.stderr)

    # 3b. Build quantization context (if requested)
    quant_ctx = None
    if quantize:
        from .quantization import build_quant_context
        exclude_patterns = (plugin.quant_exclude_patterns(quantize)
                            if hasattr(plugin, 'quant_exclude_patterns') else None)
        quant_ctx = build_quant_context(
            format_name=quantize,
            model_dir=str(model_dir_path),
            config=config,
            exclude_patterns=exclude_patterns,
            scales_json=quant_scales,
            num_calibration_samples=quant_calibration_samples,
        )
        print(f"[trtf-build] Quantization: {quantize}", file=sys.stderr)

    # 4. Build TRT engine
    print(f"[trtf-build] Building TRT engine (cache={max_cache_length}) ...",
          file=sys.stderr)
    # Pass precision/quant_ctx only if the plugin accepts them (not all do).
    import inspect
    sig = inspect.signature(plugin.build_engine)
    extra_kwargs = {}
    if 'precision' in sig.parameters:
        extra_kwargs['precision'] = precision
    if 'quant_ctx' in sig.parameters:
        extra_kwargs['quant_ctx'] = quant_ctx
    engine_plan = plugin.build_engine(
        config, weights, max_cache_length, verbose=verbose, **extra_kwargs)
    t3 = time.monotonic()
    print(f"[trtf-build] Engine built [{t3 - t2:.1f}s] "
          f"({len(engine_plan) / (1024 * 1024):.1f} MB)", file=sys.stderr)

    # 4b. Build vision engine (optional, VL models only)
    vision_plan = None
    build_vision = getattr(plugin, 'build_vision_engine', None)
    if build_vision is not None:
        print("[trtf-build] Building vision encoder engine ...",
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
        print("[trtf-build] Building extra engines ...", file=sys.stderr)
        extra_engines = build_extra(
            config, weights, max_cache_length, verbose=verbose) or {}
        for ename, eplan in extra_engines.items():
            print(f"[trtf-build]   {ename}: {len(eplan) / (1024 * 1024):.1f} MB",
                  file=sys.stderr)

    # 5. Detect tokenizer special-tokens behavior from HF config
    tokenizer_add_special_tokens = _detect_tokenizer_add_special_tokens(
        model_dir_path)

    # 6. Write bundle
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
        precision=precision,
        quantization=quantize or "none",
        tokenizer_add_special_tokens=tokenizer_add_special_tokens,
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
                cfg_dict["precision"] = precision
                if quantize:
                    cfg_dict["quantization"] = {"format": quantize}
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

    # Package FFI kernel .so files into the bundle
    if kernel_artifacts:
        import json as _json
        manifest_entries = []
        for global_name, so_path in kernel_artifacts:
            section_name = f"kernel_{global_name.replace('.', '_')}.so"
            so_data = Path(so_path).read_bytes()
            sections.append(BundleSection(section_name, so_data))
            manifest_entries.append({
                "global_name": global_name,
                "func_name": "run",
                "section": section_name,
            })
        manifest_json = _json.dumps({"kernels": manifest_entries}).encode("utf-8")
        sections.append(BundleSection("kernel_manifest.json", manifest_json))

    write_bundle(output_path, info, sections)
    t4 = time.monotonic()
    print(f"[trtf-build] Bundle saved: {output_path} [{t4 - t0:.1f}s total]",
          file=sys.stderr)


def _build_diffusion_bundle(
    model_dir_path: Path,
    output_path: str,
    max_cache_length: int,
    *,
    precision: str = "fp32",
    verbose: bool = False,
    t0: float = 0.0,
    fp8_scales: dict | None = None,
    save_fp8_scales: str | None = None,
) -> None:
    """Build a diffusion model bundle from a diffusers-format directory."""
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

    # Propagate transformer config to ModelConfig so get_diffusion_config can access it
    if "_transformer_config" in weights:
        config.raw["_transformer_config"] = weights["_transformer_config"]

    # Auto-calibrate FP8 if requested
    if fp8_scales == "auto":
        calibrate_fn = getattr(plugin, 'fp8_calibrate', None)
        if calibrate_fn is None:
            raise ValueError(
                f"Plugin {plugin.name} does not support FP8 auto-calibration. "
                f"Use --fp8-scales with a pre-computed scales JSON instead.")
        print(f"[trtf-build] Running FP8 auto-calibration for {plugin.name} ...",
              file=sys.stderr)
        fp8_scales = calibrate_fn(str(model_dir_path), config)
        print(f"[trtf-build] Calibrated {len(fp8_scales)} layers",
              file=sys.stderr)

    # Save FP8 scales to JSON if requested
    if save_fp8_scales and isinstance(fp8_scales, dict):
        with open(save_fp8_scales, "w") as _sf:
            json.dump(fp8_scales, _sf, indent=2)
        print(f"[trtf-build] Saved FP8 scales to {save_fp8_scales} "
              f"({len(fp8_scales)} layers)", file=sys.stderr)

    # Build all component engines
    build_components = getattr(plugin, 'build_components', None)
    if build_components is None:
        raise ValueError(
            f"Plugin {plugin.name} does not support build_components()")

    components = build_components(
        str(model_dir_path), config, weights, verbose=verbose,
        fp8_scales=fp8_scales)
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
    _effective_precision = "bf16" if fp8_scales else precision
    cfg_dict = {
        "model_type": model_type,
        "runtime_strategy": getattr(plugin, "runtime_strategy", "diffusion"),
        "precision": _effective_precision,
        "num_text_encoders": len(components["text_encoders"]),
    }
    if fp8_scales:
        cfg_dict["quantization"] = {"format": "fp8"}

    # Inject diffusion config from plugin
    get_diff_config = getattr(plugin, 'get_diffusion_config', None)
    if get_diff_config is not None:
        diff_cfg = get_diff_config(config)
        if diff_cfg is not None:
            cfg_dict.update(diff_cfg)

    cfg_data = json.dumps(cfg_dict, indent=2).encode("utf-8")
    sections.append(BundleSection("config.json", cfg_data))

    # Ensure tokenizer.json exists for diffusion tokenizer directories.
    # SentencePiece-only tokenizers (T5, PixArt) may lack tokenizer.json
    # which the native C++ tokenizer needs.
    for tok_subdir in ("tokenizer_2", "tokenizer"):
        tok_dir = model_dir_path / tok_subdir
        if tok_dir.is_dir() and not (tok_dir / "tokenizer.json").exists():
            _ensure_tokenizer_json(tok_dir)

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
        "tokenizer.json": "clip_tokenizer.json",
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
        precision=precision,
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
    precision: str = "fp32",
    quantize: str | None = None,
    quant_scales: str | None = None,
    quant_calibration_samples: int = 512,
    verbose: bool = False,
    fp8_scales: dict | str | None = None,
    save_fp8_scales: str | None = None,
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
        fp8_scales: Per-layer FP8 scales dict, or ``"auto"`` for auto-calibration.
        save_fp8_scales: Path to save calibrated FP8 scales JSON.
    """
    model_dir = _resolve_model(model_id_or_path)
    build_bundle._model_id_or_path_orig = model_id_or_path
    build_bundle._fp8_scales = fp8_scales
    build_bundle._save_fp8_scales = save_fp8_scales
    build_bundle(model_dir, output_path, max_cache_length,
                 precision=precision,
                 quantize=quantize,
                 quant_scales=quant_scales,
                 quant_calibration_samples=quant_calibration_samples,
                 verbose=verbose)
