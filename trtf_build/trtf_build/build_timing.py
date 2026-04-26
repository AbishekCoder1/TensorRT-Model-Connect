"""Best-effort structured timing helpers for bundle builds."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
from typing import Iterator


BUILD_TIMING_ENV = "TRTF_BUILD_TIMING_JSON"


def new_build_timing() -> dict:
    return {
        "schema_version": 1,
        "phases": {},
    }


def add_build_timing(timing: dict | None, key: str, seconds: float) -> None:
    if timing is None:
        return
    phases = timing.setdefault("phases", {})
    phases[key] = float(phases.get(key, 0.0)) + float(seconds)


def write_build_timing(timing: dict | None) -> None:
    if timing is None:
        return
    path = os.environ.get(BUILD_TIMING_ENV, "").strip()
    if not path:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(timing, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        # Timing should never make a build fail.
        pass


@contextmanager
def timed_build_phase(timing: dict | None, key: str) -> Iterator[None]:
    t0 = time.monotonic()
    try:
        yield
    finally:
        add_build_timing(timing, key, time.monotonic() - t0)
        write_build_timing(timing)


@contextmanager
def timed_weight_loading(timing: dict | None, component: str) -> Iterator[None]:
    key = f"weights_loading_{component}_s"
    t0 = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - t0
        add_build_timing(timing, "weights_loading_s", elapsed)
        add_build_timing(timing, key, elapsed)
        write_build_timing(timing)


@contextmanager
def timed_trt_compile(timing: dict | None, component: str) -> Iterator[None]:
    t0 = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - t0
        add_trt_compile_timing(timing, component, elapsed)


def add_trt_compile_timing(
    timing: dict | None,
    component: str,
    seconds: float,
) -> None:
    key = f"trt_compile_{component}_s"
    add_build_timing(timing, "trt_compile_s", seconds)
    add_build_timing(timing, key, seconds)
    write_build_timing(timing)
