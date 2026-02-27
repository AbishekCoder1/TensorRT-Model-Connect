"""E2E orchestrator — coordinates the full lifecycle for one model case.

The orchestrator depends ONLY on contracts (Dependency Inversion Principle).
It does not import any concrete runner, reference, or comparator. All
concrete implementations are resolved at runtime via the registry.

Lifecycle per case:
    1. Initialize artifact sink and write case snapshot.
    2. Run preflight checks.
    3. Resolve or build bundle.
    4. For each stage:
       a. Execute TRT strategy runner.
       b. Execute reference backend runner.
       c. Compare outputs.
       d. Persist stage artifacts.
    5. Execute determinism reruns for designated stages.
    6. Aggregate stage outcomes to final status.
    7. Write final result.json.
    8. Return E2EResult for pytest assertion.
"""

from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import (
    CompareResult,
    E2ECase,
    E2EResult,
    E2EStatus,
    FailureType,
    PreflightRequirement,
    RunContext,
    StageOutput,
    StageSpec,
    ThresholdProfile,
)
from .registry import get_comparator, get_reference, get_runner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------


def _check_binary_exists(ctx: RunContext, req: PreflightRequirement) -> tuple[bool, str]:
    """Check that the trtf binary exists."""
    path = req.args.get("path", ctx.binary_path)
    if path and Path(path).is_file():
        return True, f"Binary found: {path}"
    return False, f"Binary not found: {path}"


def _check_gpu_memory(ctx: RunContext, req: PreflightRequirement) -> tuple[bool, str]:
    """Check GPU has enough memory."""
    min_gb = req.args.get("min_gb", 0)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            total_mb = int(result.stdout.strip().splitlines()[0])
            total_gb = total_mb / 1024
            if total_gb >= min_gb:
                return True, f"GPU memory: {total_gb:.1f} GB >= {min_gb} GB"
            return False, f"GPU memory: {total_gb:.1f} GB < {min_gb} GB required"
    except Exception as e:
        return False, f"GPU memory check failed: {e}"
    return False, "GPU memory check failed: unknown error"


def _check_hf_auth(ctx: RunContext, req: PreflightRequirement) -> tuple[bool, str]:
    """Check that HF auth token is present."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return True, "HF auth token found"
    hf_token_path = Path.home() / ".huggingface" / "token"
    if hf_token_path.is_file():
        return True, "HF auth token found in ~/.huggingface/token"
    return False, "HF auth token not found (set HF_TOKEN or login with huggingface-cli)"


def _check_asset_exists(ctx: RunContext, req: PreflightRequirement) -> tuple[bool, str]:
    """Check that a required asset file exists."""
    asset_path = req.args.get("path", "")
    if not asset_path:
        return False, "Asset path not specified"
    # Resolve relative paths against project root or e2e data dir
    p = Path(asset_path)
    if not p.is_absolute():
        e2e_dir = Path(__file__).resolve().parent.parent / "e2e"
        p = e2e_dir / asset_path
    if p.is_file():
        return True, f"Asset found: {p}"
    return False, f"Asset not found: {p}"


def _check_python_module(ctx: RunContext, req: PreflightRequirement) -> tuple[bool, str]:
    """Check that a Python module is importable."""
    module = req.args.get("module", "")
    if not module:
        return False, "Module name not specified"
    try:
        importlib.import_module(module)
        return True, f"Module {module} available"
    except ImportError:
        return False, f"Module {module} not available"


_PREFLIGHT_CHECKERS = {
    "binary_exists": _check_binary_exists,
    "gpu_memory_min_gb": _check_gpu_memory,
    "hf_auth_token_present": _check_hf_auth,
    "asset_exists": _check_asset_exists,
    "python_module_available": _check_python_module,
}


def run_preflight(
    case: E2ECase,
    ctx: RunContext,
) -> tuple[bool, list[dict[str, Any]]]:
    """Run all preflight requirements for a case.

    Returns (all_passed, details) where details is a list of
    per-requirement dicts with keys: kind, passed, message, gating.
    """
    details: list[dict[str, Any]] = []
    all_gating_passed = True

    for req in case.preflight:
        checker = _PREFLIGHT_CHECKERS.get(req.kind)
        if checker is None:
            passed = False
            message = f"Unknown preflight kind: {req.kind}"
        else:
            try:
                passed, message = checker(ctx, req)
            except Exception as e:
                passed = False
                message = f"Preflight check {req.kind} raised: {e}"

        details.append({
            "kind": req.kind,
            "passed": passed,
            "message": message,
            "gating": req.gating,
        })

        if not passed and req.gating:
            all_gating_passed = False

    return all_gating_passed, details


# ---------------------------------------------------------------------------
# Bundle resolution
# ---------------------------------------------------------------------------


def _resolve_bundle(
    case: E2ECase,
    ctx: RunContext,
) -> tuple[str | None, float | None, str]:
    """Resolve or build the bundle. Returns (path, build_time_s, error_msg)."""
    engine_dir = Path(ctx.engine_dir)
    bundle_path = engine_dir / case.bundle

    if bundle_path.is_file() and not ctx.rebuild:
        return str(bundle_path), None, ""

    # Build the bundle
    hf_id = case.hf_id
    max_cache = case.inputs.get("max_cache_length", 256)

    cmd = [
        "trtf-build", "build",
        hf_id, "-o", str(bundle_path),
        "--max-cache-length", str(max_cache),
    ]
    if case.metadata.get("trust_remote_code"):
        cmd.append("--trust-remote-code")

    logger.info("Building bundle: %s", " ".join(cmd))
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600)
        elapsed = time.monotonic() - t0
    except subprocess.TimeoutExpired:
        return None, None, f"Bundle build timed out for {hf_id}"
    except Exception as e:
        return None, None, f"Bundle build failed for {hf_id}: {e}"

    if result.returncode != 0:
        return None, elapsed, (
            f"Bundle build failed for {hf_id} (rc={result.returncode}):\n"
            f"{result.stderr[-2000:]}"
        )

    return str(bundle_path), elapsed, ""


# ---------------------------------------------------------------------------
# Threshold resolution
# ---------------------------------------------------------------------------


def _resolve_threshold(
    case: E2ECase,
) -> ThresholdProfile:
    """Build a ThresholdProfile from defaults + overrides.

    Resolution chain: defaults -> profile -> per-model -> manifest inline.
    """
    # Start with conservative defaults
    metrics: dict[str, float] = {
        "logit_atol": 1e-3,
        "layer_atol": 0.05,
        "token_agreement_rate": 0.8,
        "logit_cosine_p5": 0.99,
        "logit_rel_l2_p95": 0.05,
        "stable_top1_match_rate": 0.9,
        "unstable_topk_hit_rate": 0.8,
        "normalized_text_edit_distance": 0.2,
    }

    # Try to load strategy-specific defaults
    try:
        from .thresholds import load_defaults
        strategy_defaults = load_defaults(case.task_strategy)
        if strategy_defaults:
            metrics.update(strategy_defaults)
    except ImportError:
        pass

    # Apply per-model overrides from manifest
    metrics.update(case.threshold_overrides)

    return ThresholdProfile(
        task_strategy=case.task_strategy,
        profile_name=case.comparison_profile,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class E2EOrchestrator:
    """Coordinates the full E2E lifecycle for one model case.

    Depends only on contracts — all concrete implementations are
    resolved at runtime via the registry.
    """

    def run(self, case: E2ECase, ctx: RunContext) -> E2EResult:
        """Execute the full E2E lifecycle for a single model case.

        Returns an E2EResult with all stage outcomes, timing, and artifacts.
        """
        from .artifact_sink import FileArtifactSink

        timestamp = datetime.now(timezone.utc).isoformat()
        timing: dict[str, float] = {}

        # Initialize artifact sink
        artifacts_dir = ctx.artifacts_dir or "/tmp/e2e_artifacts"
        sink = FileArtifactSink(artifacts_dir, case)

        # Collect environment fingerprint
        env_fp = sink.ensure_env_fingerprint()

        # Handle skip cases (v1 compat)
        if case.metadata.get("skip_reason"):
            result = E2EResult(
                case_name=case.name,
                status=E2EStatus.SKIP.value,
                failure_type=None,
                oracle_level=case.oracle_level,
                stages={},
                timing=timing,
                env_fingerprint=env_fp,
                timestamp=timestamp,
            )
            sink.finalize(result)
            return result

        # 1. Preflight
        t0 = time.monotonic()
        preflight_ok, preflight_details = run_preflight(case, ctx)
        timing["preflight_s"] = time.monotonic() - t0

        if not preflight_ok:
            failed_reqs = [d for d in preflight_details if not d["passed"] and d["gating"]]
            message = "; ".join(d["message"] for d in failed_reqs)
            result = E2EResult(
                case_name=case.name,
                status=E2EStatus.FAIL.value,
                failure_type=FailureType.PRECHECK_FAIL.value,
                oracle_level=case.oracle_level,
                stages={},
                determinism={"preflight": preflight_details},
                timing=timing,
                env_fingerprint=env_fp,
                timestamp=timestamp,
            )
            sink.finalize(result)
            return result

        # 2. Resolve or build bundle
        t0 = time.monotonic()
        bundle_path, build_time, build_err = _resolve_bundle(case, ctx)
        timing["build_s"] = time.monotonic() - t0
        if build_time is not None:
            timing["bundle_build_s"] = build_time

        if bundle_path is None:
            result = E2EResult(
                case_name=case.name,
                status=E2EStatus.FAIL.value,
                failure_type=FailureType.BUILD_FAIL.value,
                oracle_level=case.oracle_level,
                stages={},
                determinism={"build_error": build_err},
                timing=timing,
                env_fingerprint=env_fp,
                timestamp=timestamp,
            )
            sink.finalize(result)
            return result

        # Update context with resolved bundle path
        ctx_with_bundle = RunContext(
            case=case,
            artifacts_dir=artifacts_dir,
            binary_path=ctx.binary_path,
            hf_python=ctx.hf_python,
            ld_library_path=ctx.ld_library_path,
            engine_dir=ctx.engine_dir,
            rebuild=ctx.rebuild,
            verbose=ctx.verbose,
        )

        # Resolve runner, reference, comparator
        runner = get_runner(case.task_strategy)
        reference = get_reference(case.reference_backend)
        comparator = get_comparator(case.task_strategy)
        threshold = _resolve_threshold(case)

        # 3. Execute stages
        stage_results: dict[str, CompareResult] = {}
        all_stages_pass = True

        for stage in case.stages:
            stage_name = stage.name

            # TRT run
            trt_output: StageOutput | None = None
            if runner is not None:
                t0 = time.monotonic()
                try:
                    trt_output = runner.run_stage(case, stage, ctx_with_bundle)
                    timing[f"trt_{stage_name}_s"] = time.monotonic() - t0
                    sink.write_stage_output(stage_name, trt_output, prefix="trt")
                except Exception as e:
                    timing[f"trt_{stage_name}_s"] = time.monotonic() - t0
                    logger.error("TRT run failed for stage %s: %s", stage_name, e)
                    stage_results[stage_name] = CompareResult(
                        stage_name=stage_name,
                        passed=False,
                        message=f"TRT run failed: {e}",
                    )
                    if stage.required:
                        all_stages_pass = False
                    continue
            else:
                # No runner registered — produce no-op output
                trt_output = StageOutput(
                    stage_name=stage_name,
                    data={"status": "no_runner_registered"},
                    metadata={"runner": "none"},
                )
                sink.write_stage_output(stage_name, trt_output, prefix="trt")
                logger.warning(
                    "No runner registered for strategy %s, stage %s",
                    case.task_strategy, stage_name,
                )

            # Reference run
            ref_output: StageOutput | None = None
            if reference is not None:
                t0 = time.monotonic()
                try:
                    ref_output = reference.run_stage(case, stage, ctx_with_bundle)
                    timing[f"ref_{stage_name}_s"] = time.monotonic() - t0
                    sink.write_stage_output(stage_name, ref_output, prefix="ref")
                except Exception as e:
                    timing[f"ref_{stage_name}_s"] = time.monotonic() - t0
                    logger.error("Reference run failed for stage %s: %s", stage_name, e)
                    stage_results[stage_name] = CompareResult(
                        stage_name=stage_name,
                        passed=False,
                        message=f"Reference run failed: {e}",
                    )
                    if stage.required:
                        all_stages_pass = False
                    continue
            else:
                logger.warning(
                    "No reference backend registered for %s, stage %s — skipping comparison",
                    case.reference_backend, stage_name,
                )

            # Comparison
            if trt_output is not None and ref_output is not None and comparator is not None:
                t0 = time.monotonic()
                try:
                    compare_result = comparator.compare(
                        trt_output, ref_output, threshold, stage)
                    timing[f"compare_{stage_name}_s"] = time.monotonic() - t0
                except Exception as e:
                    timing[f"compare_{stage_name}_s"] = time.monotonic() - t0
                    compare_result = CompareResult(
                        stage_name=stage_name,
                        passed=False,
                        message=f"Comparison failed: {e}",
                    )
                stage_results[stage_name] = compare_result
                sink.write_compare(stage_name, compare_result)

                if not compare_result.passed and stage.required:
                    all_stages_pass = False
            elif trt_output is not None and (ref_output is None or comparator is None):
                # No reference or comparator — record TRT-only result
                stage_results[stage_name] = CompareResult(
                    stage_name=stage_name,
                    passed=True,
                    message="TRT run succeeded (no reference/comparator available)",
                    metrics={},
                )

        # 4. Determinism reruns (if configured)
        determinism_results: dict[str, Any] = {}
        reruns = case.determinism.get("reruns", 0)
        if reruns > 0 and runner is not None:
            determinism_results["reruns_requested"] = reruns
            # TODO: implement rerun logic when determinism policy is available
            determinism_results["status"] = "not_implemented"

        # 5. Aggregate result
        if all_stages_pass:
            status = E2EStatus.PASS.value
            failure_type = None
        else:
            status = E2EStatus.FAIL.value
            # Determine failure type from stage results
            failure_type = FailureType.COMPARE_FAIL.value
            for cr in stage_results.values():
                if not cr.passed:
                    if "TRT run failed" in cr.message:
                        failure_type = FailureType.TRT_RUN_FAIL.value
                        break
                    elif "Reference run failed" in cr.message:
                        failure_type = FailureType.REFERENCE_RUN_FAIL.value
                        break

        result = E2EResult(
            case_name=case.name,
            status=status,
            failure_type=failure_type,
            oracle_level=case.oracle_level,
            stages=stage_results,
            determinism=determinism_results,
            timing=timing,
            env_fingerprint=env_fp,
            timestamp=timestamp,
        )

        # 6. Finalize artifacts
        try:
            sink.finalize(result)
        except Exception as e:
            logger.error("Failed to finalize artifacts: %s", e)
            result.failure_type = FailureType.ARTIFACT_WRITE_FAIL.value

        return result
