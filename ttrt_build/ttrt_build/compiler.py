"""Core compilation: torch.export + torch_tensorrt raw TRT engine.

This module orchestrates the full build pipeline:
  1. Load HF model via family plugin
  2. Select build strategy based on plugin's runtime_strategy
  3. Wrap model with strategy-specific I/O adapter
  4. Export with torch.export (strict=False)
  5. Convert to raw TRT engine via torch_tensorrt
  6. Package into a .trtfb bundle (compatible with C++ runtime)

Strategies handle different model architectures:
  - "decoder": CausalLM with StatelessCacheWrapper (KV cache I/O)
  - "encoder_only": Encoder-only models (BERT) with EncoderOnlyWrapper

No LibTorch dependency at runtime — the engine is a pure TRT .plan file.
"""

from __future__ import annotations

import gc
import json
import logging
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Suppress known harmless third-party warnings emitted during torch_tensorrt import:
#
# 1. "TRTLLM_PLUGIN_PATH is not set" — torch_tensorrt._utils logs this when
#    USE_TRTLLM_PLUGINS env isn't set. We don't use TRT-LLM plugins.
logging.getLogger("torch_tensorrt").setLevel(logging.ERROR)
#
# 2. "transformers version X is not tested with nvidia-modelopt" — modelopt
#    hasn't updated its version check for transformers 5.x yet.
warnings.filterwarnings("ignore", message="transformers version.*nvidia-modelopt")
#
# 3. "The logger passed into createInferBuilder differs from one already
#    registered" — TRT allows only one global ILogger. torch_tensorrt's
#    _TRTLogger is registered on first Builder creation; the tensorrt.plugin
#    module (loaded during import) registers its own logger first, so all
#    subsequent Builder() calls emit this warning. Harmless — TRT correctly
#    uses the first logger. This is an upstream torch_tensorrt issue.
logging.getLogger("torch_tensorrt [TensorRT Conversion Context]").setLevel(
    logging.ERROR
)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from .config import ModelConfig  # noqa: E402

PRECISION_DTYPE_MAP: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def precision_to_dtype(precision: str) -> torch.dtype:
    """Convert a precision string to a torch dtype."""
    if precision not in PRECISION_DTYPE_MAP:
        valid = ", ".join(sorted(PRECISION_DTYPE_MAP))
        raise ValueError(f"Unknown precision {precision!r}. Valid: {valid}")
    return PRECISION_DTYPE_MAP[precision]
from .families import find_plugin, ALL_PLUGINS  # noqa: E402
from .bundle_writer import TtrtBundleInfo, BundleSection, write_bundle  # noqa: E402
from .strategies import get_strategy  # noqa: E402

# Backward-compat aliases — tests and external code may import these from compiler.
from .strategies.decoder import StatelessCacheWrapper, patch_static_cache_scatter  # noqa: F401, E402


# ---------------------------------------------------------------------------
# Engine compilation
# ---------------------------------------------------------------------------

def _get_torch_version() -> str:
    return torch.__version__


def _get_torchtrt_version() -> str:
    try:
        import torch_tensorrt
        return torch_tensorrt.__version__
    except ImportError:
        return "not installed"


def _get_trt_version() -> str:
    try:
        import tensorrt
        return tensorrt.__version__
    except ImportError:
        return ""


def _get_gpu_name() -> str:
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return ""


def _detect_tokenizer_add_special_tokens(model_dir: Path) -> bool:
    """Detect whether the HF tokenizer adds special tokens by default."""
    tok_config_path = model_dir / "tokenizer_config.json"
    if tok_config_path.exists():
        try:
            tok_cfg = json.load(open(tok_config_path))
            if "add_bos_token" in tok_cfg:
                return bool(tok_cfg["add_bos_token"])
        except Exception:
            pass
    return False


def compile_model(
    wrapper: nn.Module,
    example_args: tuple,
    *,
    trt_inputs: tuple | None = None,
    verbose: bool = False,
    workspace_size: int = 1 << 30,
) -> bytes:
    """Export a wrapped model via torch.export and convert to raw TRT engine.

    Args:
        wrapper: Wrapped model (StatelessCacheWrapper, EncoderOnlyWrapper, etc.)
                 in eval mode. Can be on CPU for CPU-side export.
        example_args: Tuple of example tensors matching forward() signature.
                      Used for torch.export tracing. Must be on same device as
                      the wrapper (CPU or CUDA).
        trt_inputs: Optional separate inputs for torch_tensorrt conversion.
                    When provided, these are used as the ``inputs`` argument to
                    convert_exported_program_to_serialized_trt_engine() instead
                    of example_args. This enables CPU-side export: trace on CPU
                    with example_args, then convert with CUDA trt_inputs so the
                    TRT engine targets GPU. If None, example_args is used for
                    both steps.
        verbose: Enable detailed logging.
        workspace_size: TRT workspace size in bytes (default 1GB).

    Returns:
        Raw TRT engine bytes (.plan format).
    """
    import torch_tensorrt

    # 1. Export to ExportedProgram
    if verbose:
        print("[ttrt-build] Running torch.export ...", file=sys.stderr)

    t0 = time.monotonic()
    with torch.no_grad():
        exported = torch.export.export(
            wrapper,
            args=example_args,
            strict=False,
        )
    t1 = time.monotonic()

    if verbose:
        inputs_nodes = [n for n in exported.graph.nodes if n.op == 'placeholder']
        user_inputs = [n for n in inputs_nodes if not n.name.startswith('p_')]
        weight_params = [n for n in inputs_nodes if n.name.startswith('p_')]
        print(f"[ttrt-build] torch.export complete [{t1-t0:.1f}s] "
              f"({len(user_inputs)} user inputs, {len(weight_params)} weights)",
              file=sys.stderr)

    # 2. Convert to raw TRT engine
    if verbose:
        print("[ttrt-build] Converting to raw TRT engine ...", file=sys.stderr)

    conversion_inputs = list(trt_inputs) if trt_inputs is not None else list(example_args)

    t2 = time.monotonic()
    engine_bytes = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
        exported,
        inputs=conversion_inputs,
        use_explicit_typing=True,
        workspace_size=workspace_size,
        min_block_size=1,
        truncate_double=True,
    )
    t3 = time.monotonic()

    # Free the ExportedProgram to release GPU memory
    del exported

    if verbose:
        print(f"[ttrt-build] Raw TRT engine: {len(engine_bytes)/(1024*1024):.1f} MB "
              f"[{t3-t2:.1f}s]", file=sys.stderr)

    return engine_bytes


def _inspect_engine(engine_bytes: bytes) -> dict:
    """Inspect a TRT engine and return I/O tensor name mapping."""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    rt = trt.Runtime(logger)
    engine = rt.deserialize_cuda_engine(engine_bytes)

    io_map = {"inputs": {}, "outputs": {}}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = list(engine.get_tensor_shape(name))
        dtype = str(engine.get_tensor_dtype(name))
        mode = engine.get_tensor_mode(name)
        entry = {"shape": shape, "dtype": dtype}
        if mode == trt.TensorIOMode.INPUT:
            io_map["inputs"][name] = entry
        else:
            io_map["outputs"][name] = entry

    return io_map


def _parse_model_config(model_dir_path: Path) -> ModelConfig:
    """Parse model config, falling back to model_index.json for diffusers models."""
    config_path = model_dir_path / "config.json"
    if config_path.exists():
        return ModelConfig.from_dir(model_dir_path)

    # Diffusers-format models use model_index.json at the top level
    index_path = model_dir_path / "model_index.json"
    if index_path.exists():
        raw = json.loads(index_path.read_text())
        # Use pipeline class name as model_type (e.g. "PixArtSigmaPipeline")
        model_type = raw.get("_class_name", "")
        return ModelConfig(model_type=model_type, raw=raw)

    raise FileNotFoundError(
        f"No config.json or model_index.json found in {model_dir_path}")


def _build_multi_engine_bundle(
    model_dir_path: Path,
    plugin,
    config: ModelConfig,
    strategy,
    output_path: str,
    *,
    precision: str = "fp16",
    verbose: bool = False,
    t0: float = 0.0,
) -> None:
    """Build a multi-engine bundle (diffusion models).

    The family plugin's build_components() loads each component, wraps it,
    and calls compile_model() for each. Results are packaged into a single
    .trtfb bundle with multiple engine plan sections.
    """
    print(f"[ttrt-build] Multi-engine build (precision={precision}) ...",
          file=sys.stderr)

    result = plugin.build_components(
        str(model_dir_path), config, compile_model,
        precision=precision, verbose=verbose,
    )

    component_sections = result["sections"]
    bundle_runtime_strategy = result["runtime_strategy"]
    diffusion_config = result.get("diffusion_config", {})

    # Build config.json for the bundle
    bundle_config = {
        "runtime_strategy": bundle_runtime_strategy,
        **diffusion_config,
    }
    config_data = json.dumps(bundle_config, indent=2).encode("utf-8")

    # Assemble all sections: engine plans + config
    sections = list(component_sections)
    sections.append(BundleSection("config.json", config_data))

    # Embed tokenizer files (T5 tokenizer for diffusion models)
    tokenizer_dir = model_dir_path / "tokenizer"
    tokenizer_search_dirs = [tokenizer_dir, model_dir_path]
    for search_dir in tokenizer_search_dirs:
        if not search_dir.exists():
            continue
        for filename in ("tokenizer.json", "tokenizer_config.json",
                         "spiece.model", "special_tokens_map.json"):
            file_path = search_dir / filename
            if file_path.exists():
                sections.append(BundleSection(filename, file_path.read_bytes()))

    info = TtrtBundleInfo(
        model_id=model_dir_path.name,
        model_type=config.model_type,
        family=plugin.name,
        torch_version=_get_torch_version(),
        torchtrt_version=_get_torchtrt_version(),
        gpu_name=_get_gpu_name(),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        precision=precision,
        runtime_strategy=bundle_runtime_strategy,
    )

    write_bundle(output_path, info, sections)
    t_end = time.monotonic()
    total_engine_mb = sum(len(s.data) for s in component_sections) / (1024 * 1024)
    print(f"[ttrt-build] Bundle saved: {output_path} "
          f"({len(component_sections)} engines, {total_engine_mb:.1f} MB total) "
          f"[{t_end - t0:.1f}s]", file=sys.stderr)


def build_bundle(
    model_dir: str,
    output_path: str,
    max_cache_length: int = 256,
    *,
    precision: str = "fp16",
    verbose: bool = False,
) -> None:
    """Full pipeline: load HF model -> compile to raw TRT engine -> write .trtfb bundle.

    The build strategy is selected based on the family plugin's runtime_strategy
    attribute (defaults to "decoder" if absent). Each strategy handles model
    wrapping, export arg construction, and pre-export setup.

    Supports two build paths:
      - Single-engine (decoder, encoder-only): standard wrap -> export -> compile
      - Multi-engine (diffusion): plugin.build_components() compiles each
        component separately and returns multiple engine sections

    Args:
        model_dir: Path to HF model directory with config.json + safetensors,
                   or diffusers-format directory with model_index.json.
        output_path: Where to write the .trtfb bundle.
        max_cache_length: KV cache / max sequence length.
        precision: Compute precision (model loaded in this precision).
        verbose: Print detailed logs.
    """
    model_dir_path = Path(model_dir)
    compute_dtype = precision_to_dtype(precision)
    t0 = time.monotonic()

    # 1. Parse config (supports both config.json and model_index.json)
    config = _parse_model_config(model_dir_path)
    print(f"[ttrt-build] Model: {config.model_type} "
          f"(layers={config.num_hidden_layers}, hidden={config.hidden_size}, "
          f"vocab={config.vocab_size})", file=sys.stderr)

    # 2. Find family plugin
    plugin = find_plugin(config.model_type)
    if plugin is None:
        supported = ", ".join(p.name for p in ALL_PLUGINS)
        raise ValueError(
            f"No Torch-TRT family plugin for model_type={config.model_type!r}. "
            f"Supported: {supported}")

    print(f"[ttrt-build] Family: {plugin.name}", file=sys.stderr)

    # 3. Select build strategy from plugin (defaults to "decoder")
    strategy_name = getattr(plugin, 'runtime_strategy', 'decoder')
    strategy = get_strategy(strategy_name)
    print(f"[ttrt-build] Strategy: {strategy.name} "
          f"(runtime_strategy={strategy.runtime_strategy})", file=sys.stderr)

    # Multi-engine path: diffusion and other multi-component models.
    # The family plugin's build_components() handles loading, wrapping,
    # and compiling each component, calling compile_model() for each.
    if hasattr(plugin, 'build_components'):
        _build_multi_engine_bundle(
            model_dir_path, plugin, config, strategy, output_path,
            precision=precision, verbose=verbose, t0=t0)
        return

    # Single-engine path: standard decoder, encoder-only, etc.
    model = None
    wrapper = None
    try:
        # 4. Load HF model in the requested precision
        t1 = time.monotonic()
        print(f"[ttrt-build] Loading model (dtype={compute_dtype}) ...",
              file=sys.stderr)
        model = plugin.load_model(
            str(model_dir_path), config, max_cache_length,
            dtype=compute_dtype)
        t2 = time.monotonic()
        print(f"[ttrt-build] Model loaded [{t2 - t1:.1f}s]", file=sys.stderr)

        # 5. Pre-export setup (e.g. patch StaticCache for decoder strategy)
        strategy.pre_export_setup()

        # 6. Wrap model with strategy-specific I/O adapter
        hf_config = model.config  # HF PretrainedConfig
        wrapper = strategy.wrap_model(
            model, hf_config, max_cache_length,
            compute_dtype=compute_dtype)
        wrapper.eval()

        # 7. Build example inputs for torch.export
        export_args = strategy.make_export_args(
            hf_config, max_cache_length, precision=precision)

        # 8. Compile to raw TRT engine
        print(f"[ttrt-build] Compiling (precision={precision}, "
              f"cache={max_cache_length}) ...", file=sys.stderr)
        engine_bytes = compile_model(
            wrapper, export_args,
            verbose=verbose,
        )
        t3 = time.monotonic()
        print(f"[ttrt-build] Compiled [{t3 - t2:.1f}s] "
              f"({len(engine_bytes) / (1024 * 1024):.1f} MB)", file=sys.stderr)

        # 9. Inspect engine I/O for bundle metadata
        io_map = _inspect_engine(engine_bytes)
        if verbose:
            print(f"[ttrt-build] Engine I/O: {len(io_map['inputs'])} inputs, "
                  f"{len(io_map['outputs'])} outputs", file=sys.stderr)

        # 10. Detect tokenizer behavior
        tokenizer_add_special_tokens = _detect_tokenizer_add_special_tokens(
            model_dir_path)

        # 11. Write .trtfb bundle
        info = TtrtBundleInfo(
            model_id=model_dir_path.name,
            model_type=config.model_type,
            family=plugin.name,
            torch_version=_get_torch_version(),
            torchtrt_version=_get_torchtrt_version(),
            gpu_name=_get_gpu_name(),
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_cache_length=max_cache_length,
            precision=precision,
            runtime_strategy=strategy.runtime_strategy,
            tokenizer_add_special_tokens=tokenizer_add_special_tokens,
        )

        # Use engine_plan as section name (C++ bundle reader looks for this)
        sections = [BundleSection("engine_plan", engine_bytes)]

        # Embed config + tokenizer files
        for filename in ("config.json", "tokenizer.json", "tokenizer_config.json",
                         "vocab.json", "merges.txt", "special_tokens_map.json",
                         "tokenizer.model"):
            file_path = model_dir_path / filename
            if file_path.exists():
                data = file_path.read_bytes()
                if filename == "config.json":
                    cfg_dict = json.loads(data)
                    cfg_dict["runtime_strategy"] = strategy.runtime_strategy
                    cfg_dict["torchtrt_io_map"] = io_map
                    data = json.dumps(cfg_dict, indent=2).encode("utf-8")
                sections.append(BundleSection(filename, data))

        write_bundle(output_path, info, sections)
        t4 = time.monotonic()
        print(f"[ttrt-build] Bundle saved: {output_path} [{t4 - t0:.1f}s total]",
              file=sys.stderr)

    finally:
        # Explicit GPU memory cleanup — prevents OOM when building multiple
        # bundles in the same process (e.g. multi-agent or batch builds).
        del wrapper
        del model
        gc.collect()
        torch.cuda.empty_cache()
