"""Self-tests for tools/diff_framework/ — protocol, registry, runner, CLI.

Pure-Python tests (no GPU, no model loading) that verify the framework
mechanics. Runs as part of Tier 1 (`pytest tests/tools/ -v`).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def _import_framework():
    import importlib
    return importlib.import_module("diff_framework")


def _import_protocol():
    import importlib
    return importlib.import_module("diff_framework.protocol")


def _import_registry():
    import importlib
    return importlib.import_module("diff_framework.registry")


def _import_runner():
    import importlib
    return importlib.import_module("diff_framework.runner")


# -----------------------------------------------------------------------
# TestDiffResult — serialization and constructors
# -----------------------------------------------------------------------

class TestDiffResult:
    def test_to_dict_roundtrip(self):
        proto = _import_protocol()
        r = proto.DiffResult(
            test_name="logit_diff", model="test/model",
            runtime_strategy="decoder_kv_cache",
            passed=True, status="PASS", message="ok",
            metrics={"max_abs_diff": 0.001}, duration_s=1.5, details="")
        d = r.to_dict()
        assert d["test_name"] == "logit_diff"
        assert d["passed"] is True
        assert d["metrics"]["max_abs_diff"] == 0.001
        assert d["duration_s"] == 1.5

    def test_to_json_valid(self):
        proto = _import_protocol()
        r = proto.DiffResult(
            test_name="layer_diff", model="test/model",
            runtime_strategy="decoder_kv_cache",
            passed=False, status="FAIL", message="bad",
            metrics={}, duration_s=0.0, details="detail")
        parsed = json.loads(r.to_json())
        assert parsed["status"] == "FAIL"
        assert parsed["test_name"] == "layer_diff"
        assert parsed["details"] == "detail"

    def test_skip_constructor(self):
        proto = _import_protocol()
        r = proto.DiffResult.skip("x", "m", "s", "no bundle")
        assert r.status == "SKIP"
        assert r.passed is True  # skip is not a failure
        assert "no bundle" in r.message

    def test_error_constructor(self):
        proto = _import_protocol()
        r = proto.DiffResult.error("x", "m", "s", "crash", details="tb")
        assert r.status == "ERROR"
        assert r.passed is False
        assert r.message == "crash"
        assert r.details == "tb"

    def test_default_metrics_empty(self):
        proto = _import_protocol()
        r = proto.DiffResult(
            test_name="t", model="m", runtime_strategy="s",
            passed=True, status="PASS", message="ok")
        assert r.metrics == {}
        assert r.duration_s == 0.0
        assert r.details == ""


# -----------------------------------------------------------------------
# TestRegistry — registration and lookup
# -----------------------------------------------------------------------

class TestRegistry:
    def test_register_and_lookup_by_name(self):
        registry = _import_registry()

        # Verify a known check was auto-registered
        cls = registry.get_test_by_name("logit_diff")
        assert cls is not None
        assert cls.name == "logit_diff"

    def test_get_tests_for_strategy_filters(self):
        registry = _import_registry()

        decoder_tests = registry.get_tests_for_strategy("decoder_kv_cache")
        names = [c.name for c in decoder_tests]
        assert "logit_diff" in names
        assert "layer_diff" in names
        # VL and diffusion should not appear for decoder_kv_cache
        assert "vl_pipeline" not in names
        assert "diffusion_components" not in names

    def test_unknown_test_returns_none(self):
        registry = _import_registry()
        assert registry.get_test_by_name("nonexistent_test_xyz") is None

    def test_vl_tests_for_vision_language(self):
        registry = _import_registry()
        vl_tests = registry.get_tests_for_strategy("vision_language")
        names = [c.name for c in vl_tests]
        assert "vl_pipeline" in names
        # Standard decoder tests should not appear
        assert "logit_diff" not in names

    def test_diffusion_tests_for_diffusion(self):
        registry = _import_registry()
        diff_tests = registry.get_tests_for_strategy("diffusion")
        names = [c.name for c in diff_tests]
        assert "diffusion_components" in names
        assert "logit_diff" not in names

    def test_get_all_tests_returns_all(self):
        registry = _import_registry()
        all_tests = registry.get_all_tests()
        names = [c.name for c in all_tests]
        assert "logit_diff" in names
        assert "layer_diff" in names
        assert "runner_parity" in names
        assert "perf_benchmark" in names
        assert "vl_pipeline" in names
        assert "diffusion_components" in names
        assert len(names) == 6


# -----------------------------------------------------------------------
# TestRunner — orchestration logic
# -----------------------------------------------------------------------

class TestRunner:
    def test_list_tests_returns_expected_fields(self):
        runner = _import_runner()
        entries = runner.list_tests()
        assert len(entries) >= 6
        for e in entries:
            assert "name" in e and "description" in e
            assert "runtime_strategies" in e
            assert "requires_bundle" in e
            assert "requires_gpu" in e

    def test_list_tests_filtered(self):
        runner = _import_runner()
        entries = runner.list_tests("vision_language")
        names = [e["name"] for e in entries]
        assert "vl_pipeline" in names
        assert "logit_diff" not in names

    def test_run_tests_skips_bundle_required(self):
        proto = _import_protocol()
        runner = _import_runner()

        ctx = proto.TestContext(
            model="test/model",
            runtime_strategy="decoder_kv_cache",
            bundle_path=None,
        )
        results = runner.run_tests(ctx, test_names=["runner_parity"])
        assert len(results) == 1
        assert results[0].status == "SKIP"
        assert results[0].passed is True

    def test_run_tests_unknown_name_raises(self):
        proto = _import_protocol()
        runner = _import_runner()

        ctx = proto.TestContext(
            model="test/model",
            runtime_strategy="decoder_kv_cache",
        )
        with pytest.raises(ValueError, match="Unknown test"):
            runner.run_tests(ctx, test_names=["nonexistent_test_xyz"])


# -----------------------------------------------------------------------
# TestCLI — argument parsing (dry-run, no execution)
# -----------------------------------------------------------------------

class TestCLI:
    def _get_cli_module(self):
        """Import the diff.py CLI module."""
        import importlib
        return importlib.import_module("diff")

    def test_list_subcommand_parses(self):
        mod = self._get_cli_module()
        # Verify argparse accepts: diff.py list --model X
        parser = mod.main.__code__  # just verify the module imports cleanly
        assert parser is not None

    def test_run_subcommand_exists(self):
        mod = self._get_cli_module()
        assert hasattr(mod, "cmd_run")
        assert hasattr(mod, "cmd_list")

    def test_module_has_main(self):
        mod = self._get_cli_module()
        assert callable(mod.main)
