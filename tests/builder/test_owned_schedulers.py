"""Unit tests for owned scheduler modules.

These tests target deterministic math and branch behavior only.
"""

from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path

import numpy as np
import pytest


# Ensure imports resolve to this workspace's Python package.
_PKG_ROOT = Path(__file__).resolve().parents[2] / "trtf_build"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))


@pytest.mark.unit
def test_flow_match_set_timesteps_builds_expected_schedule() -> None:
    """Intent: verify deterministic timestep construction for shift=1.

    Preconditions: Scheduler is created with default train timesteps and shift.
    Postconditions: Timesteps are decreasing float32 values and sigma appends 0.
    """
    from trtf_build.schedulers.flow_match_euler import FlowMatchEulerScheduler

    scheduler = FlowMatchEulerScheduler(num_train_timesteps=1000, shift=1.0)
    scheduler.set_timesteps(num_inference_steps=4)

    np.testing.assert_allclose(
        scheduler.timesteps,
        np.array([1000.0, 667.0, 334.0, 1.0], dtype=np.float32),
    )
    assert scheduler.timesteps.dtype == np.float32
    np.testing.assert_allclose(scheduler._sigmas[-1], 0.0)
    assert np.all(np.diff(scheduler.timesteps) < 0.0)


@pytest.mark.unit
def test_flow_match_set_timesteps_with_shift_changes_tail() -> None:
    """Intent: validate shifted schedule branch is used when shift != 1.

    Preconditions: Scheduler is created with non-default shift value.
    Postconditions: Final timestep is larger than unshifted schedule tail.
    """
    from trtf_build.schedulers.flow_match_euler import FlowMatchEulerScheduler

    scheduler = FlowMatchEulerScheduler(num_train_timesteps=1000, shift=2.0)
    scheduler.set_timesteps(num_inference_steps=4)

    assert scheduler.timesteps[0] == pytest.approx(1000.0, abs=1e-6)
    assert scheduler.timesteps[-1] > 1.0
    assert np.all(np.diff(scheduler.timesteps) < 0.0)


@pytest.mark.unit
def test_flow_match_step_uses_sigma_delta_and_returns_float32() -> None:
    """Intent: verify Euler update uses sigma_next - sigma.

    Preconditions: Internal sigma schedule is initialized with two values.
    Postconditions: Step output matches expected update and is float32.
    """
    from trtf_build.schedulers.flow_match_euler import FlowMatchEulerScheduler

    scheduler = FlowMatchEulerScheduler()
    scheduler._sigmas = np.array([1.0, 0.5], dtype=np.float64)

    sample = np.array([1.0, 1.0], dtype=np.float32)
    model_output = np.array([2.0, -2.0], dtype=np.float32)

    updated = scheduler.step(
        model_output=model_output,
        timestep=999.0,
        sample=sample,
        step_index=0,
    )

    np.testing.assert_allclose(updated, np.array([0.0, 2.0], dtype=np.float32))
    assert updated.dtype == np.float32


@pytest.mark.unit
def test_flow_match_add_noise_is_linear_interpolation() -> None:
    """Intent: verify add_noise follows z_t = (1-sigma)x + sigma*noise.

    Preconditions: Original sample, noise sample, and timestep are provided.
    Postconditions: Output equals the expected convex combination.
    """
    from trtf_build.schedulers.flow_match_euler import FlowMatchEulerScheduler

    scheduler = FlowMatchEulerScheduler(num_train_timesteps=1000)
    original = np.array([4.0, -4.0], dtype=np.float32)
    noise = np.array([0.0, 8.0], dtype=np.float32)

    mixed = scheduler.add_noise(original=original, noise=noise, timestep=250.0)

    np.testing.assert_allclose(mixed, np.array([3.0, -1.0], dtype=np.float32))
    assert mixed.dtype == np.float32


@pytest.mark.unit
def test_get_scheduler_factory_and_unknown_error() -> None:
    """Intent: validate scheduler factory dispatch and error branch.

    Preconditions: Scheduler name is valid once and invalid once.
    Postconditions: Valid name returns instance; invalid name raises ValueError.
    """
    from trtf_build.schedulers import FlowMatchEulerScheduler, get_scheduler

    scheduler = get_scheduler("flow_match_euler", shift=1.5)
    assert isinstance(scheduler, FlowMatchEulerScheduler)
    assert scheduler.shift == pytest.approx(1.5)

    with pytest.raises(ValueError, match="Unknown scheduler"):
        get_scheduler("does_not_exist")


@pytest.mark.unit
def test_scheduler_protocol_declares_required_methods() -> None:
    """Intent: smoke-test protocol surface for scheduler interface.

    Preconditions: Scheduler protocol is importable.
    Postconditions: Protocol exposes all required API method names.
    """
    from trtf_build.schedulers.base import Scheduler

    assert hasattr(Scheduler, "timesteps")
    assert hasattr(Scheduler, "set_timesteps")
    assert hasattr(Scheduler, "step")
    assert hasattr(Scheduler, "add_noise")


@pytest.mark.unit
def test_package_main_module_invokes_cli_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intent: validate `python -m trtf_build` delegates to cli.main.

    Preconditions: A fake `trtf_build.cli` module with callable `main` is injected.
    Postconditions: Importing/executing `trtf_build.__main__` calls fake `main` once.
    """
    calls: list[str] = []

    fake_cli = types.ModuleType("trtf_build.cli")

    def _fake_main() -> None:
        calls.append("called")

    fake_cli.main = _fake_main  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "trtf_build.cli", fake_cli)
    sys.modules.pop("trtf_build.__main__", None)

    runpy.run_module("trtf_build.__main__", run_name="__main__")

    assert calls == ["called"]


@pytest.mark.unit
def test_package_init_exports_expected_symbols() -> None:
    """Intent: verify top-level package re-exports expected public API.

    Preconditions: Local `trtf_build` package is importable.
    Postconditions: Public symbols from `trtf_build.__init__` exist and are usable.
    """
    import importlib
    import trtf_build

    # Repo layout may expose an outer namespace package at `trtf_build`.
    # Validate exports from the concrete runtime package in either layout.
    pkg = trtf_build
    if getattr(trtf_build, "__file__", None) is None:
        pkg = importlib.import_module("trtf_build.trtf_build")

    if hasattr(pkg, "__version__"):
        assert pkg.__version__ == "0.1.0"
    assert callable(pkg.build)
    assert callable(pkg.build_bundle)
    assert callable(pkg.write_bundle)
    assert hasattr(pkg, "ModelConfig")
    assert hasattr(pkg, "Pipeline")
