# Torch-TRT Cleanup: Backend Architecture

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize torch-trt as an optional build backend — eliminate C++ runtime awareness of torch-trt, absorb `tensorrt_model_connect/` into `tensorrt_model_connect/backends/torch_trt/`, and unify strategy registries.

**Architecture:** The C++ runtime auto-detects IO naming from engine tensor names (existing `has_input()`/`has_output()` pattern). The Python builder gains a `--backend` flag; `tensorrt_model_connect/` becomes `tensorrt_model_connect/backends/torch_trt/`. Strategy registries are deduplicated — `test_impact.py` imports from `contracts.py` instead of maintaining its own copy.

**Tech Stack:** C++17, Python 3.12, TensorRT, torch_tensorrt (optional), pytest, CMake/Ninja

---

## File Structure

### Files to Create
- `tensorrt_model_connect/tensorrt_model_connect/backends/__init__.py` — Backend registry with auto-discovery
- `tensorrt_model_connect/tensorrt_model_connect/backends/base.py` — `BuildBackend` protocol
- `tensorrt_model_connect/tensorrt_model_connect/backends/torch_trt/__init__.py` — Entry point (moved from `tensorrt_model_connect/__init__.py`)
- `tensorrt_model_connect/tensorrt_model_connect/backends/torch_trt/compiler.py` — (moved from `tensorrt_model_connect/compiler.py`)
- `tensorrt_model_connect/tensorrt_model_connect/backends/torch_trt/cache_config.py` — (moved)
- `tensorrt_model_connect/tensorrt_model_connect/backends/torch_trt/config.py` — (moved)
- `tensorrt_model_connect/tensorrt_model_connect/backends/torch_trt/bundle_writer.py` — (moved)
- `tensorrt_model_connect/tensorrt_model_connect/backends/torch_trt/bundle_reader.py` — (moved)
- `tensorrt_model_connect/tensorrt_model_connect/backends/torch_trt/families/` — (moved from `tensorrt_model_connect/families/`)
- `tensorrt_model_connect/tensorrt_model_connect/backends/torch_trt/strategies/` — (moved from `tensorrt_model_connect/strategies/`)
- `tests/backends/__init__.py`
- `tests/backends/torch_trt/` — (moved from `tests/torchtrt_builder/`)

### Files to Modify
- `include/trtmc/runtime/kv_cache.h` — Remove `NamingScheme` enum, simplify constructor
- `src/runtime/core/kv_cache.cpp` — Auto-detect naming in `bind_to()` via `has_input()`
- `src/runtime/plugins/decoder_plugin.cpp` — Remove `is_torchtrt` branch + `"torchtrt_decoder"` registration
- `src/runtime/plugins/force_link_plugins.cpp` — Rename anchor: `TorchTrtDiffusion` → `PixArt`
- `src/runtime/pipelines/text_generation_pipeline.h` — Auto-detect `logits_output_name` in constructor
- `CMakeLists.txt` — Rename source files
- `tensorrt_model_connect/tensorrt_model_connect/cli.py` — `--torch-trt` → `--backend`
- `tests/e2e_harness/contracts.py:812-842` — Remove `torchtrt_decoder` and `torchtrt_diffusion`
- `tests/e2e_harness/manifest_loader.py:425-453` — Remove from `_KNOWN_RUNTIME_STRATEGIES`
- `tools/test_impact.py:28-103` — Import from `contracts.py` instead of maintaining copies
- `tests/e2e/models/qwen2.5-0.5b-torchtrt.json` — Change `runtime_strategy` to `decoder_kv_cache`
- `tests/e2e/models/pixart-sigma-1024-torchtrt.json` — Change `runtime_strategy` to `diffusion_pixart`

### Files to Rename
- `src/runtime/plugins/torchtrt_diffusion_plugin.cpp` → `src/runtime/plugins/pixart_plugin.cpp`
- `src/runtime/pipelines/torchtrt_diffusion_pipeline.h` → `src/runtime/pipelines/pixart_pipeline.h`
- `src/runtime/pipelines/torchtrt_diffusion_pipeline.cpp` → `src/runtime/pipelines/pixart_pipeline.cpp`

### Files to Delete (after move)
- `tensorrt_model_connect/` (entire directory — absorbed into `tensorrt_model_connect/backends/torch_trt/`)
- `tests/torchtrt_builder/` (moved to `tests/backends/torch_trt/`)

---

## Task 1: C++ Runtime — Auto-detect KV Cache IO Naming

Remove torch-trt awareness from KV cache. The runtime probes the engine's tensor names instead of reading a `NamingScheme` enum.

**Files:**
- Modify: `include/trtmc/runtime/kv_cache.h`
- Modify: `src/runtime/core/kv_cache.cpp:73-95`
- Test: `tests/cpp/test_torchtrt_decoder.cpp` (update to validate auto-detect)

- [ ] **Step 1: Modify `kv_cache.h` — remove NamingScheme enum, simplify constructor**

```cpp
// include/trtmc/runtime/kv_cache.h
// Remove the NamingScheme enum entirely (lines 27-30).
// Remove the NamingScheme parameter from the constructor (line 32).
// Remove the naming_ member (line 74).

// Before:
    enum class NamingScheme {
        kStandard, // cache_k_N / cache_v_N / present_k_N / present_v_N
        kTorchTrt, // cache_kv_{2N} / cache_kv_{2N+1} / output{2N+1} / output{2N+2}
    };

    KvCache(int32_t num_layers, int32_t max_length, int32_t kv_dim, cudaStream_t stream,
            DType cache_dtype = DType::kFloat32, NamingScheme naming = NamingScheme::kStandard);

// After:
    KvCache(int32_t num_layers, int32_t max_length, int32_t kv_dim, cudaStream_t stream,
            DType cache_dtype = DType::kFloat32);
```

- [ ] **Step 2: Modify `kv_cache.cpp` — auto-detect naming in `bind_to()`**

```cpp
// src/runtime/core/kv_cache.cpp

// In the constructor, remove the naming parameter:
// Before:
KvCache::KvCache(int32_t num_layers, int32_t max_length, int32_t kv_dim, cudaStream_t stream,
                 DType cache_dtype, NamingScheme naming)
    : ... naming_(naming) {

// After:
KvCache::KvCache(int32_t num_layers, int32_t max_length, int32_t kv_dim, cudaStream_t stream,
                 DType cache_dtype)
    : ... {

// In bind_to(), replace the NamingScheme branch with auto-detection:
void KvCache::bind_to(TrtModule& module) {
    has_position_input_ = module.has_input("position_id");

    // Auto-detect IO naming from engine tensor names.
    // Torch-TRT engines use cache_kv_{2i} naming; standard TRT API uses cache_k_N.
    bool torchtrt_naming = module.has_input("cache_kv_0");

    for (int32_t i = 0; i < num_layers_; ++i) {
        auto li = static_cast<std::size_t>(i);
        if (torchtrt_naming) {
            std::string k_idx = std::to_string(2 * i);
            std::string v_idx = std::to_string(2 * i + 1);
            module.bind_external("cache_kv_" + k_idx, cache_k_[li].data());
            module.bind_external("cache_kv_" + v_idx, cache_v_[li].data());
            module.bind_external("output" + std::to_string(2 * i + 1), present_k_[li].data());
            module.bind_external("output" + std::to_string(2 * i + 2), present_v_[li].data());
        } else {
            std::string suffix = "_" + std::to_string(i);
            module.bind_external("cache_k" + suffix, cache_k_[li].data());
            module.bind_external("cache_v" + suffix, cache_v_[li].data());
            module.bind_external("present_k" + suffix, present_k_[li].data());
            module.bind_external("present_v" + suffix, present_v_[li].data());
        }
    }
}
```

- [ ] **Step 3: Fix all callers that pass NamingScheme**

Search for `NamingScheme` across the codebase and update every constructor call:
```bash
grep -rn "NamingScheme\|kTorchTrt\|kStandard" src/ include/ tests/cpp/
```
Each call site that passes `NamingScheme::kStandard` or `NamingScheme::kTorchTrt` should simply omit the parameter.

- [ ] **Step 4: Build and run C++ tests**

```bash
docker exec trtmc-dev-gb300-agent-4 bash -c "cd /workspace/tensorrt-model-connect && cmake --build build -j && ctest --test-dir build --output-on-failure"
```

- [ ] **Step 5: Commit**

```bash
git add include/trtmc/runtime/kv_cache.h src/runtime/core/kv_cache.cpp
git commit -m "refactor(kv_cache): auto-detect IO naming from engine tensors

Remove NamingScheme enum. bind_to() probes has_input(\"cache_kv_0\")
to detect torch-trt naming vs standard naming. No behavioral change."
```

---

## Task 2: C++ Runtime — Auto-detect Logits Output Name

Remove the `logits_output_name` config field from `TextGenConfig`. Auto-detect whether the engine uses `"logits"` or `"output0"`.

**Files:**
- Modify: `src/runtime/pipelines/text_generation_pipeline.h:31`
- Modify: `src/runtime/pipelines/text_generation_pipeline.cpp:153`
- Modify: `src/runtime/plugins/decoder_plugin.cpp:39-40`

- [ ] **Step 1: Auto-detect logits output name in `TextGenerationPipeline` constructor**

```cpp
// src/runtime/pipelines/text_generation_pipeline.cpp
// In the constructor, after storing decoder_, detect the logits output name:

TextGenerationPipeline::TextGenerationPipeline(...)
    : decoder_(std::move(decoder)), ... {
    ...
    // Auto-detect logits output name from engine.
    if (decoder_->has_output("logits")) {
        logits_output_name_ = "logits";
    } else if (decoder_->has_output("output0")) {
        logits_output_name_ = "output0";
    } else {
        throw std::runtime_error("TextGenerationPipeline: engine has no 'logits' or 'output0' output");
    }
}
```

Add `std::string logits_output_name_;` as a private member on the class.

- [ ] **Step 2: Remove `logits_output_name` from `TextGenConfig`**

```cpp
// src/runtime/pipelines/text_generation_pipeline.h
// Remove this line from TextGenConfig:
//     std::string logits_output_name{"logits"};

// Update run_step to use the member:
// In text_generation_pipeline.cpp line 153:
    auto it = outputs.find(logits_output_name_);
```

- [ ] **Step 3: Remove `is_torchtrt` logic from `decoder_plugin.cpp`**

```cpp
// src/runtime/plugins/decoder_plugin.cpp
// Remove lines 21-23 (is_torchtrt detection + naming):
//     bool is_torchtrt = (ctx.config.runtime_strategy == "torchtrt_decoder");
//     auto naming = is_torchtrt ? ... : ...;
// Just create KvCache without naming parameter:
    std::unique_ptr<IInferenceState> state = std::make_unique<KvCache>(
        ctx.config.num_layers, ctx.config.max_cache_length, kv_dim, stream, cache_dtype);

// Remove lines 39-40 (logits output override):
//     if (is_torchtrt)
//         tgc.logits_output_name = "output0";

// Remove the "torchtrt_decoder" registration at line 63:
//     static trtmc::PluginRegistrar g_DecoderPlugin_reg3("torchtrt_decoder", ...);
// Keep "decoder_kv_cache" and "decoder_moe" registrations.
```

- [ ] **Step 4: Build and run tests**

```bash
docker exec trtmc-dev-gb300-agent-4 bash -c "cd /workspace/tensorrt-model-connect && cmake --build build -j && ctest --test-dir build --output-on-failure"
```

- [ ] **Step 5: Commit**

```bash
git add src/runtime/pipelines/text_generation_pipeline.h \
        src/runtime/pipelines/text_generation_pipeline.cpp \
        src/runtime/plugins/decoder_plugin.cpp
git commit -m "refactor(decoder): auto-detect logits output name, remove torchtrt_decoder strategy

TextGenerationPipeline probes has_output(\"logits\") vs has_output(\"output0\").
decoder_plugin.cpp no longer registers torchtrt_decoder — torch-trt decoders
use decoder_kv_cache strategy with auto-detected IO."
```

---

## Task 3: C++ Runtime — Rename torchtrt_diffusion → pixart

This pipeline is a PixArt pipeline. The fact that it was built via torch-trt is irrelevant to the runtime.

**Files:**
- Rename: `src/runtime/plugins/torchtrt_diffusion_plugin.cpp` → `src/runtime/plugins/pixart_plugin.cpp`
- Rename: `src/runtime/pipelines/torchtrt_diffusion_pipeline.h` → `src/runtime/pipelines/pixart_pipeline.h`
- Rename: `src/runtime/pipelines/torchtrt_diffusion_pipeline.cpp` → `src/runtime/pipelines/pixart_pipeline.cpp`
- Modify: `src/runtime/plugins/force_link_plugins.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Rename files**

```bash
git mv src/runtime/plugins/torchtrt_diffusion_plugin.cpp src/runtime/plugins/pixart_plugin.cpp
git mv src/runtime/pipelines/torchtrt_diffusion_pipeline.h src/runtime/pipelines/pixart_pipeline.h
git mv src/runtime/pipelines/torchtrt_diffusion_pipeline.cpp src/runtime/pipelines/pixart_pipeline.cpp
```

- [ ] **Step 2: Update internal references in the renamed files**

In `pixart_plugin.cpp`:
```cpp
// Change:
#include "runtime/pipelines/torchtrt_diffusion_pipeline.h"
// To:
#include "runtime/pipelines/pixart_pipeline.h"

// Change class name:
// class TorchTrtDiffusionPlugin → class PixArtPlugin
// TorchTrtDiffusionPipeline → PixArtDiffusionPipeline

// Change force-link anchor:
// volatile int kForceLink_TorchTrtDiffusionPlugin = 0;
// → volatile int kForceLink_PixArtPlugin = 0;

// Change registration — register under "diffusion_pixart" (already a known strategy):
// static trtmc::PluginRegistrar g_... ("torchtrt_diffusion", ...);
// → static trtmc::PluginRegistrar g_... ("diffusion_pixart", ...);
```

In `pixart_pipeline.h` and `pixart_pipeline.cpp`:
```cpp
// Rename class: TorchTrtDiffusionPipeline → PixArtDiffusionPipeline
// Update pipeline_type(): return "PixArtDiffusionPipeline";
// Update header guard if using pragma once (no change needed)
// Update all include paths
```

- [ ] **Step 3: Update `force_link_plugins.cpp`**

```cpp
// Change:
extern volatile int kForceLink_TorchTrtDiffusionPlugin;
// To:
extern volatile int kForceLink_PixArtPlugin;

// Change in array:
&kForceLink_TorchTrtDiffusionPlugin → &kForceLink_PixArtPlugin
```

- [ ] **Step 4: Update `CMakeLists.txt`**

Find and replace the source file paths:
```
torchtrt_diffusion_plugin.cpp → pixart_plugin.cpp
torchtrt_diffusion_pipeline.cpp → pixart_pipeline.cpp
```

- [ ] **Step 5: Build and test**

```bash
docker exec trtmc-dev-gb300-agent-4 bash -c "cd /workspace/tensorrt-model-connect && cmake --build build -j && ctest --test-dir build --output-on-failure"
```

- [ ] **Step 6: Commit**

```bash
git add -A src/runtime/plugins/ src/runtime/pipelines/ CMakeLists.txt
git commit -m "refactor: rename torchtrt_diffusion → pixart (build backend is not a runtime concern)

The PixArt pipeline is a PixArt pipeline regardless of whether the engine
was built via TRT API or torch-trt. Register under diffusion_pixart."
```

---

## Task 4: Python — Create Backend System Scaffold

Create the `backends/` package with the protocol and registry.

**Files:**
- Create: `tensorrt_model_connect/tensorrt_model_connect/backends/__init__.py`
- Create: `tensorrt_model_connect/tensorrt_model_connect/backends/base.py`

- [ ] **Step 1: Create `backends/base.py` — BuildBackend protocol**

```python
# tensorrt_model_connect/tensorrt_model_connect/backends/base.py
"""Build backend protocol — all backends implement this interface."""

from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class BuildBackend(Protocol):
    """Protocol for engine build backends.

    Each backend takes a resolved model directory and produces a .trtfb bundle.
    The backend is a build-time concern — the resulting bundle uses standard
    runtime_strategy values and is indistinguishable at runtime.
    """

    name: str

    def is_available(self) -> bool:
        """Check whether this backend's dependencies are installed."""
        ...

    def build(
        self,
        model_dir: str,
        output_path: str,
        max_cache_length: int = 256,
        *,
        precision: str = "fp16",
        verbose: bool = False,
    ) -> None:
        """Build a .trtfb bundle from a model directory."""
        ...
```

- [ ] **Step 2: Create `backends/__init__.py` — registry with lazy discovery**

```python
# tensorrt_model_connect/tensorrt_model_connect/backends/__init__.py
"""Backend registry — discovers and dispatches to build backends.

The default backend is 'trt' (TRT Network API). Optional backends like
'torch_trt' are discovered if their dependencies are installed.
"""

from __future__ import annotations

import importlib
import logging
from typing import Dict, Optional

from .base import BuildBackend

logger = logging.getLogger(__name__)

_backends: Dict[str, BuildBackend] = {}
_discovered = False

# Known backend modules — maps name to importable module path.
# Each module must expose a module-level `backend` attribute.
_BACKEND_MODULES = {
    "torch_trt": ".backends.torch_trt",
}


def _discover() -> None:
    global _discovered
    if _discovered:
        return
    _discovered = True

    for name, mod_path in _BACKEND_MODULES.items():
        try:
            mod = importlib.import_module(mod_path, package="tensorrt_model_connect")
            backend = getattr(mod, "backend", None)
            if backend is not None and isinstance(backend, BuildBackend):
                _backends[name] = backend
                logger.debug("Registered backend: %s", name)
        except ImportError:
            logger.debug("Backend %s not available (missing dependencies)", name)
        except Exception:
            logger.warning("Failed to load backend %s", name, exc_info=True)


def get_backend(name: str) -> Optional[BuildBackend]:
    """Look up a backend by name. Returns None if not available."""
    _discover()
    return _backends.get(name)


def list_backends() -> Dict[str, BuildBackend]:
    """Return all available backends."""
    _discover()
    return dict(_backends)
```

- [ ] **Step 3: Commit**

```bash
git add tensorrt_model_connect/tensorrt_model_connect/backends/
git commit -m "feat(backends): add BuildBackend protocol and registry scaffold"
```

---

## Task 5: Python — Move tensorrt_model_connect into backends/torch_trt

Move the entire `tensorrt_model_connect/tensorrt_model_connect/` tree into `tensorrt_model_connect/tensorrt_model_connect/backends/torch_trt/` and fix all internal imports.

**Files:**
- Move: `tensorrt_model_connect/tensorrt_model_connect/*` → `tensorrt_model_connect/tensorrt_model_connect/backends/torch_trt/`
- Delete: `tensorrt_model_connect/` (after move)

- [ ] **Step 1: Move files**

```bash
# Move all source files
cp -r tensorrt_model_connect/tensorrt_model_connect/* tensorrt_model_connect/tensorrt_model_connect/backends/torch_trt/

# The __init__.py needs rewriting (Step 2), but families/, strategies/,
# compiler.py, cache_config.py, config.py, bundle_writer.py, bundle_reader.py
# can be moved as-is with import fixes.
```

- [ ] **Step 2: Rewrite `backends/torch_trt/__init__.py` to implement BuildBackend**

```python
# tensorrt_model_connect/tensorrt_model_connect/backends/torch_trt/__init__.py
"""Torch-TRT build backend — optional performance optimization.

Compiles HuggingFace models into .trtfb bundles via torch.export +
torch_tensorrt. Produces standard bundles with standard runtime_strategy
values (decoder_kv_cache, diffusion_pixart, etc.).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


class TorchTrtBackend:
    """BuildBackend implementation for torch-trt."""

    name = "torch_trt"

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import torch_tensorrt  # noqa: F401
            return True
        except ImportError:
            return False

    def build(
        self,
        model_dir: str,
        output_path: str,
        max_cache_length: int = 256,
        *,
        precision: str = "fp16",
        verbose: bool = False,
    ) -> None:
        from .compiler import build_bundle

        t0 = time.monotonic()
        build_bundle(
            model_dir, output_path, max_cache_length,
            precision=precision, verbose=verbose,
        )
        elapsed = time.monotonic() - t0
        print(f"[torch-trt] Done [{elapsed:.1f}s total]", file=sys.stderr)


# Module-level attribute for auto-discovery by backends registry
backend = TorchTrtBackend()
```

- [ ] **Step 3: Fix all internal imports in moved files**

In every `.py` file under `tensorrt_model_connect/tensorrt_model_connect/backends/torch_trt/`:

Replace:
```python
from tensorrt_model_connect.xxx import yyy
from .xxx import yyy  # these are fine — relative imports still work
```

Specifically in `compiler.py`:
```python
# Change:
from .config import ModelConfig
# This still works (relative import within backends/torch_trt/)

# The soft import of tensorrt_model_connect.config.ModelConfig is now a sibling:
# Change:
#   from tensorrt_model_connect.config import ModelConfig
# To:
#   from tensorrt_model_connect.config import ModelConfig  # still works — same package
```

In `config.py`, the fallback import of `tensorrt_model_connect.config.ModelConfig` now works directly since we're inside `tensorrt_model_connect`.

- [ ] **Step 4: Update `compiler.py` — bundles must use standard runtime_strategy**

In `compiler.py`, where `runtime_strategy` is set on the bundle info:

```python
# The strategy written to the bundle should be the STANDARD runtime strategy,
# not a torch-trt-specific one. Add a mapping:
_TORCHTRT_TO_STANDARD_STRATEGY = {
    "torchtrt_decoder": "decoder_kv_cache",
    "torchtrt_encoder": "encoder_only",
    "torchtrt_diffusion": "diffusion_pixart",
    "decoder": "decoder_kv_cache",
    "encoder_only": "encoder_only",
    "diffusion": "diffusion_pixart",
}

# Where the bundle info is populated, normalize:
info.runtime_strategy = _TORCHTRT_TO_STANDARD_STRATEGY.get(
    raw_strategy, raw_strategy)

# Also add a build_backend metadata field:
# In the header dict construction in bundle_writer.py, add:
"build_backend": "torch_trt",
```

- [ ] **Step 5: Verify the moved package imports correctly**

```bash
cd tensorrt_model_connect && python -c "from tensorrt_model_connect.backends import get_backend; b = get_backend('torch_trt'); print(b.name if b else 'not available')"
```

- [ ] **Step 6: Delete the old `tensorrt_model_connect/` directory**

```bash
git rm -r tensorrt_model_connect/
```

- [ ] **Step 7: Commit**

```bash
git add tensorrt_model_connect/tensorrt_model_connect/backends/torch_trt/ tensorrt_model_connect/
git commit -m "refactor: absorb tensorrt_model_connect into tensorrt_model_connect/backends/torch_trt

Single package, optional backend. Torch-TRT bundles now emit standard
runtime_strategy values (decoder_kv_cache, diffusion_pixart).
build_backend: torch_trt recorded in bundle metadata."
```

---

## Task 6: Python — Update CLI with --backend Flag

Replace `--torch-trt` boolean flag with `--backend` choice.

**Files:**
- Modify: `tensorrt_model_connect/tensorrt_model_connect/cli.py:28-62`

- [ ] **Step 1: Replace --torch-trt with --backend in argument parser**

Find where `--torch-trt` is added to the argparse subparser and replace:

```python
# Before (somewhere in the arg setup):
#     build_parser.add_argument('--torch-trt', action='store_true', ...)

# After:
build_parser.add_argument(
    '--backend', type=str, default='trt',
    choices=['trt', 'torchtrt'],
    help='Build backend: trt (default, TRT Network API) or torchtrt (torch-trt)')
```

- [ ] **Step 2: Update `_cmd_build()` dispatch logic**

```python
# tensorrt_model_connect/tensorrt_model_connect/cli.py

def _cmd_build(args: argparse.Namespace) -> int:
    if not args.model:
        print("Error: model (HF repo ID or local directory) required", file=sys.stderr)
        return 1
    if not args.output:
        print("Error: -o / --output required", file=sys.stderr)
        return 1

    backend_name = getattr(args, 'backend', 'trt')

    if backend_name != 'trt':
        # Dispatch to the requested backend
        from .backends import get_backend
        backend = get_backend(backend_name)
        if backend is None:
            print(f"Error: backend '{backend_name}' is not available. "
                  f"Install its dependencies (e.g., pip install torch_tensorrt).",
                  file=sys.stderr)
            return 1
        try:
            backend.build(
                args.model, args.output,
                max_cache_length=args.max_cache_length,
                precision=args.precision,
                verbose=args.verbose,
            )
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    # Default: TRT Network API path
    from .engine_builder import build
    # ... rest of existing TRT API logic unchanged ...
```

- [ ] **Step 3: Run CLI tests**

```bash
docker exec trtmc-dev-gb300-agent-4 bash -c "cd /workspace/tensorrt-model-connect && python -m pytest tests/builder/test_cli.py -v"
```

- [ ] **Step 4: Commit**

```bash
git add tensorrt_model_connect/tensorrt_model_connect/cli.py
git commit -m "refactor(cli): replace --torch-trt flag with --backend choice

trtmc-build build <model> -o out.trtfb                    # TRT API (default)
trtmc-build build <model> -o out.trtfb --backend torchtrt  # Torch-TRT"
```

---

## Task 7: Strategy Registry Cleanup

Remove `torchtrt_decoder` and `torchtrt_diffusion` entries from all strategy registries. Make `test_impact.py` import from `contracts.py` instead of maintaining its own copy.

**Files:**
- Modify: `tests/e2e_harness/contracts.py:836-841`
- Modify: `tests/e2e_harness/manifest_loader.py:425-453`
- Modify: `tools/test_impact.py:28-103`

- [ ] **Step 1: Remove torch-trt strategies from `contracts.py`**

```python
# tests/e2e_harness/contracts.py
# Remove these two lines from RUNTIME_TO_TASK_STRATEGY:
#     "torchtrt_diffusion": "torchtrt_diffusion",
#     "torchtrt_decoder": "text_generation_causal",
```

- [ ] **Step 2: Remove from `manifest_loader.py`**

```python
# tests/e2e_harness/manifest_loader.py
# Remove "torchtrt_diffusion" from _KNOWN_RUNTIME_STRATEGIES frozenset.
# Remove "torchtrt_decoder" from _KNOWN_RUNTIME_STRATEGIES frozenset.
# Remove "torchtrt_diffusion" from _DEFAULT_REFERENCE_BACKEND dict.
# Remove "torchtrt_diffusion" from _DEFAULT_STAGES dict.
```

- [ ] **Step 3: Deduplicate `test_impact.py` — import from contracts**

```python
# tools/test_impact.py
# Replace the hardcoded RUNTIME_TO_TASK_STRATEGY dict (lines 28-58) with:
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
from e2e_harness.contracts import RUNTIME_TO_TASK_STRATEGY

# Remove the torchtrt entries from CPP_PLUGIN_STRATEGIES and CPP_PIPELINE_STRATEGIES:
# In CPP_PLUGIN_STRATEGIES, remove "torchtrt_diffusion" from wan_plugin entry.
# In CPP_PIPELINE_STRATEGIES, remove "torchtrt_diffusion" from wan_pipeline and diffusion_pipeline entries.
```

- [ ] **Step 4: Run E2E harness tests**

```bash
docker exec trtmc-dev-gb300-agent-4 bash -c "cd /workspace/tensorrt-model-connect && python -m pytest tests/tools/ -v"
```

- [ ] **Step 5: Commit**

```bash
git add tests/e2e_harness/contracts.py tests/e2e_harness/manifest_loader.py tools/test_impact.py
git commit -m "refactor: remove torchtrt_decoder/torchtrt_diffusion from strategy registries

Torch-TRT bundles now use standard strategies (decoder_kv_cache, diffusion_pixart).
test_impact.py imports RUNTIME_TO_TASK_STRATEGY from contracts.py (single source)."
```

---

## Task 8: E2E Manifests — Normalize torch-trt Models

Update the two torch-trt E2E manifests to use standard runtime strategies and add `build_backend` field.

**Files:**
- Modify: `tests/e2e/models/qwen2.5-0.5b-torchtrt.json`
- Modify: `tests/e2e/models/pixart-sigma-1024-torchtrt.json`

- [ ] **Step 1: Update qwen2.5-0.5b-torchtrt.json**

```json
{
  "name": "qwen2.5-0.5b-torchtrt",
  "hf_id": "Qwen/Qwen2.5-0.5B",
  "bundle": "qwen2.5-0.5b-torchtrt.trtfb",
  "family": "qwen",
  "runtime_strategy": "decoder_kv_cache",
  "build_backend": "torch_trt",
  "max_cache_length": 256,
  "precision": "fp16",
  "prompt": "The largest ocean on Earth is",
  "max_new_tokens": 20,
  "logit_atol": 1e-2,
  "build_args": {
    "backend": "torchtrt",
    "max_cache_length": 256
  }
}
```

- [ ] **Step 2: Update pixart-sigma-1024-torchtrt.json**

```json
{
  "name": "pixart-sigma-1024-torchtrt",
  "runtime_strategy": "diffusion_pixart",
  "build_backend": "torch_trt",
  "build_args": {
    "backend": "torchtrt",
    "max_cache_length": 256
  }
}
```
(Keep all other fields unchanged.)

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/models/qwen2.5-0.5b-torchtrt.json tests/e2e/models/pixart-sigma-1024-torchtrt.json
git commit -m "refactor(e2e): torch-trt manifests use standard runtime_strategy

qwen2.5-0.5b-torchtrt: torchtrt_decoder → decoder_kv_cache
pixart-sigma-1024-torchtrt: torchtrt_diffusion → diffusion_pixart
build_backend field records the build method."
```

---

## Task 9: Move Tests and Cleanup

Move torch-trt builder tests to `tests/backends/torch_trt/` and clean up stale references.

**Files:**
- Move: `tests/torchtrt_builder/` → `tests/backends/torch_trt/`
- Create: `tests/backends/__init__.py`
- Update: `tests/backends/torch_trt/conftest.py` (fix imports)

- [ ] **Step 1: Move test directory**

```bash
mkdir -p tests/backends
touch tests/backends/__init__.py
git mv tests/torchtrt_builder tests/backends/torch_trt
```

- [ ] **Step 2: Fix imports in moved tests**

In all `tests/backends/torch_trt/test_*.py` files, update imports:
```python
# Change:
from tensorrt_model_connect.xxx import yyy
# To:
from tensorrt_model_connect.backends.torch_trt.xxx import yyy
```

- [ ] **Step 3: Run the moved tests**

```bash
docker exec trtmc-dev-gb300-agent-4 bash -c "cd /workspace/tensorrt-model-connect && python -m pytest tests/backends/ -v --co"
```

- [ ] **Step 4: Update any CI config referencing old paths**

Search for `torchtrt_builder` in `.gitlab-ci.yml` and other CI configs:
```bash
grep -rn "torchtrt_builder" .gitlab-ci.yml Makefile scripts/
```

- [ ] **Step 5: Commit**

```bash
git add tests/backends/ tests/torchtrt_builder/
git commit -m "refactor(tests): move torchtrt_builder tests to tests/backends/torch_trt"
```

---

## Task 10: Documentation Update

Update CLAUDE.md, torch-trt docs, and any references to the old structure.

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/torch-trt/TORCHTRT_TRANSFORM_GUIDE.md`
- Modify: `docs/torch-trt/TORCHTRT_AGENT_GUIDE.md`

- [ ] **Step 1: Update CLAUDE.md**

In the source layout section:
- Remove `tensorrt_model_connect/` as a top-level entry
- Add `tensorrt_model_connect/tensorrt_model_connect/backends/` section describing torch_trt backend
- Update `torchtrt_diffusion_plugin.cpp` → `pixart_plugin.cpp`
- Update `torchtrt_diffusion_pipeline.h/.cpp` → `pixart_pipeline.h/.cpp`
- Update CLI examples: `--torch-trt` → `--backend torchtrt`
- Remove `torchtrt_decoder` and `torchtrt_diffusion` from strategy tables

- [ ] **Step 2: Update torch-trt docs**

In `TORCHTRT_TRANSFORM_GUIDE.md` and `TORCHTRT_AGENT_GUIDE.md`:
- Update package name: `tensorrt_model_connect` → `tensorrt_model_connect.backends.torch_trt`
- Update CLI: `trtmc-build build --torch-trt` → `trtmc-build build --backend torchtrt`
- Update import paths in code examples
- Note that bundles now use standard runtime_strategy values

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md docs/torch-trt/
git commit -m "docs: update for torch-trt backend restructure"
```

---

## Task Dependencies

```
Task 1 (KV auto-detect) ──┐
Task 2 (logits auto-detect)├── Can run in parallel (C++ only)
Task 3 (rename pixart) ───┘

Task 4 (backend scaffold) → Task 5 (move tensorrt_model_connect) → Task 6 (CLI update)
                                                          ↓
Task 7 (strategy cleanup) ─────────────────────────── Task 8 (manifests)
                                                          ↓
                                                     Task 9 (move tests)
                                                          ↓
                                                     Task 10 (docs)
```

**Parallel workstreams:**
- **Workstream A** (Tasks 1-3): C++ runtime cleanup — independent of Python changes
- **Workstream B** (Tasks 4-6): Python backend system — independent of C++ changes
- **Workstream C** (Tasks 7-9): Registry + tests — depends on A and B for strategy names
- **Workstream D** (Task 10): Docs — runs last
