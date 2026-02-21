"""Unified runner — detect strategy, list tests, run tests."""

from __future__ import annotations

import time

from .protocol import DiffResult, TestContext
from .registry import get_all_tests, get_tests_for_strategy, get_test_by_name


def detect_runtime_strategy(model: str) -> str:
    """Auto-detect runtime_strategy from HF config via family plugin.

    Falls back to "decoder_kv_cache" if detection fails.
    """
    try:
        from trtf_build.engine_builder import _resolve_model
        from trtf_build.config import ModelConfig
        from trtf_build.families import find_plugin

        model_dir = _resolve_model(model)
        config = ModelConfig.from_dir(model_dir)
        plugin = find_plugin(config.model_type)
        if plugin is not None:
            return getattr(plugin, "runtime_strategy", "decoder_kv_cache")
    except Exception:
        pass
    return "decoder_kv_cache"


def detect_runtime_strategy_from_bundle(bundle_path: str) -> str:
    """Read runtime_strategy from a bundle's config.json."""
    import json
    import struct

    with open(bundle_path, "rb") as f:
        _magic = f.read(8)
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        sections = header.get("sections", {})
        data_start = 16 + header_len

        if "config.json" in sections:
            meta = sections["config.json"]
            f.seek(data_start + meta["offset"])
            cfg = json.loads(f.read(meta["size"]).decode("utf-8"))
            return cfg.get("runtime_strategy", "decoder_kv_cache")

    return "decoder_kv_cache"


def list_tests(runtime_strategy: str | None = None) -> list[dict]:
    """List available tests, optionally filtered by strategy."""
    if runtime_strategy:
        classes = get_tests_for_strategy(runtime_strategy)
    else:
        classes = get_all_tests()

    return [
        {
            "name": cls.name,
            "description": cls.description,
            "runtime_strategies": cls.runtime_strategies,
            "requires_bundle": cls.requires_bundle,
            "requires_gpu": cls.requires_gpu,
        }
        for cls in classes
    ]


def run_tests(
    ctx: TestContext,
    test_names: list[str] | None = None,
) -> list[DiffResult]:
    """Run applicable tests. Auto-discovers if test_names is None.

    Returns list of DiffResult objects.
    """
    if test_names is not None:
        classes = []
        for name in test_names:
            cls = get_test_by_name(name)
            if cls is None:
                raise ValueError(f"Unknown test: {name!r}. Available: "
                                 f"{[c.name for c in get_all_tests()]}")
            classes.append(cls)
    else:
        classes = get_tests_for_strategy(ctx.runtime_strategy)

    results = []
    for cls in classes:
        # Skip tests that require a bundle if none provided
        if cls.requires_bundle and not ctx.bundle_path:
            results.append(DiffResult.skip(
                cls.name, ctx.model, ctx.runtime_strategy,
                "No bundle provided (--bundle required)"))
            continue

        instance = cls()
        t0 = time.monotonic()
        try:
            result = instance.run(ctx)
        except Exception as e:
            result = DiffResult.error(
                cls.name, ctx.model, ctx.runtime_strategy,
                str(e), details=f"{type(e).__name__}: {e}")
        result.duration_s = time.monotonic() - t0
        results.append(result)

    return results
