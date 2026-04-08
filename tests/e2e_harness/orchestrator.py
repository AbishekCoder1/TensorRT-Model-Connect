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
import shlex
import subprocess
import time
import traceback
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
    StageStatus,
    ThresholdProfile,
)
from . import save_full_stderr
from .registry import get_comparator, get_reference, get_runner

logger = logging.getLogger(__name__)

_MIGRATED_RUNTIME_STRATEGIES = frozenset({
    "decoder_kv_cache",
    "decoder_moe",
    "ssm_recurrent",
    "rwkv_recurrent",
    "hybrid_mamba_attention",
    "encoder_only",
    "embedding",
    "reranking",
    "vision_language",
    "segmentation",
    "prompted_segmentation",
    "object_detection",
    "neural_operator",
    "text_to_audio",
    "speech_to_text",
    "speech_to_speech",
    "omni_multimodal",
    "diffusion",
})
_NEW_RUNTIME_MARKER = "backend=trt_new_runtime"
_LEGACY_RUNTIME_MARKER = "Runtime path: compatibility factory mode"


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
    # Resolve relative paths against project root, then e2e data dir
    p = Path(asset_path)
    if not p.is_absolute():
        project_root = Path(__file__).resolve().parent.parent.parent
        candidate = project_root / asset_path
        if candidate.is_file():
            p = candidate
        else:
            # Fallback: try e2e data dir with just the filename
            e2e_data = project_root / "tests" / "e2e" / "data"
            p = e2e_data / Path(asset_path).name
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
) -> tuple[str | None, float | None, str, dict[str, Any]]:
    """Resolve or build the bundle.

    Returns (path, build_time_s, error_msg, build_info) where build_info
    contains the subprocess command, stdout, stderr, and returncode when a
    build was executed.  build_info is empty when the bundle already exists.
    """
    engine_dir = Path(ctx.engine_dir)
    bundle_path = engine_dir / case.bundle

    if bundle_path.is_file() and not ctx.rebuild:
        return str(bundle_path), None, "", {}

    # Build the bundle
    hf_id = case.hf_id
    max_cache = case.inputs.get("max_cache_length", 256)

    build_args = case.metadata.get("build_args", {})
    use_torch_trt = build_args.get("torch_trt", False)

    cmd = [
        "trtf-build", "build",
        hf_id, "-o", str(bundle_path),
        "--max-cache-length", str(max_cache),
    ]
    if use_torch_trt:
        cmd.append("--torch-trt")
    precision = case.metadata.get("precision", "fp32")
    if precision != "fp32":
        cmd.extend(["--precision", precision])
    if case.metadata.get("trust_remote_code"):
        cmd.append("--trust-remote-code")

    logger.info("Building bundle: %s", " ".join(cmd))
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600)
        elapsed = time.monotonic() - t0
    except subprocess.TimeoutExpired:
        return None, None, f"Bundle build timed out for {hf_id}", {
            "command": cmd,
            "returncode": -1,
            "stdout": "",
            "stderr": "timeout",
        }
    except Exception as e:
        return None, None, f"Bundle build failed for {hf_id}: {e}", {
            "command": cmd,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
        }

    build_info: dict[str, Any] = {
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

    if result.returncode != 0:
        truncated, log_path = save_full_stderr(
            result.stderr, ctx.artifacts_dir or "", "bundle_build", case.name)
        msg = f"Bundle build failed for {hf_id} (rc={result.returncode}):\n{truncated}"
        if log_path:
            msg += f" (full stderr: {log_path})"
        return None, elapsed, msg, build_info

    return str(bundle_path), elapsed, "", build_info


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


def _log_stage_subprocess(
    sink: Any,
    stage_name: str,
    output: StageOutput,
    prefix: str,
) -> None:
    """Extract subprocess info from StageOutput.metadata and log to the sink.

    Runners store subprocess details in metadata under various conventions:
    - Text generation: metadata = {"cpp": {...}, "debug_runner": {...}}
    - Vision language: metadata has "command", "returncode", "stdout", "stderr"
    - Other runners: flat metadata with "command", "returncode", etc.

    This function handles all conventions and writes log files for each
    subprocess found in the metadata.
    """
    meta = output.metadata
    if not meta:
        return

    # Check for nested sub-metadata dicts (e.g., text_generation has "cpp" and "debug_runner")
    nested_found = False
    for key, value in meta.items():
        if isinstance(value, dict) and "returncode" in value:
            nested_found = True
            label = f"{stage_name}_{prefix}_{key}"
            cmd = value.get("command", [])
            rc = value.get("returncode", -1)
            stdout = value.get("stdout", "")
            stderr = value.get("stderr", "")
            sink.log_command(
                command=cmd if isinstance(cmd, list) else [str(cmd)],
                rc=rc,
                stdout=str(stdout),
                stderr=str(stderr),
                label=label,
            )

    # If no nested dicts, check for flat metadata with command/returncode
    if not nested_found and "returncode" in meta:
        label = f"{stage_name}_{prefix}"
        cmd = meta.get("command", [])
        rc = meta.get("returncode", -1)
        stdout = meta.get("stdout", "")
        stderr = meta.get("stderr", "")
        sink.log_command(
            command=cmd if isinstance(cmd, list) else [str(cmd)],
            rc=rc,
            stdout=str(stdout),
            stderr=str(stderr),
            label=label,
        )


def _auto_register_artifacts(sink: Any, output: StageOutput, prefix: str) -> None:
    """Scan StageOutput.data for known artifact keys and register on sink."""
    _ARTIFACT_KEYS = {
        "wav_path": "wav",
        "frames_dir": "frames",
        "logits_path": "logits",
        "features_path": "features",
        "output_path": "output",
        "segmentation_map_path": "segmentation_map",
    }
    for data_key, artifact_key in _ARTIFACT_KEYS.items():
        value = output.data.get(data_key)
        if value and isinstance(value, str):
            # Store relative to artifacts dir if possible
            base = str(sink.base_dir)
            rel = value
            if value.startswith(base):
                rel = value[len(base):].lstrip("/")
            sink.register_artifact(f"{prefix}_{artifact_key}", rel)


def _collect_runtime_guard_payloads(value: Any) -> list[tuple[list[str], list[str], int | None]]:
    """Collect (command argv, stderr payloads, returncode) tuples from nested output structures."""
    payloads: list[tuple[list[str], list[str], int | None]] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            command = node.get("command")
            command_argv: list[str] = []
            if isinstance(command, list):
                command_argv = [str(part) for part in command]
            elif isinstance(command, str) and command:
                try:
                    command_argv = shlex.split(command)
                except ValueError:
                    command_argv = [command]

            stderr_texts: list[str] = []

            for key in ("stderr", "stderr_truncated"):
                text = node.get(key)
                if isinstance(text, str) and text:
                    stderr_texts.append(text)

            stderr_log = node.get("stderr_log")
            if isinstance(stderr_log, str) and stderr_log:
                try:
                    stderr_texts.append(Path(stderr_log).read_text(encoding="utf-8"))
                except OSError:
                    pass

            returncode = node.get("returncode")
            if not isinstance(returncode, int):
                returncode = None

            if command_argv:
                payloads.append((command_argv, stderr_texts, returncode))

            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return payloads


def _validate_trt_runtime_path(
    case: E2ECase,
    ctx: RunContext,
    output: StageOutput,
) -> str | None:
    """Ensure TRT E2E subprocesses for migrated strategies use the new runtime path."""
    if case.runtime_strategy not in _MIGRATED_RUNTIME_STRATEGIES:
        return None

    payloads = _collect_runtime_guard_payloads({
        "metadata": output.metadata,
        "data": output.data,
    })
    if not payloads:
        return None

    binary_name = Path(ctx.binary_path).name if ctx.binary_path else ""
    relevant_stderr: list[str] = []
    for command_argv, stderr_texts, returncode in payloads:
        executable = command_argv[0] if command_argv else ""
        executable_name = Path(executable).name if executable else ""
        if ctx.binary_path and executable == ctx.binary_path:
            combined_payload_stderr = "\n".join(stderr_texts)
            if returncode not in (None, 0) and _NEW_RUNTIME_MARKER not in combined_payload_stderr \
                    and _LEGACY_RUNTIME_MARKER not in combined_payload_stderr:
                continue
            relevant_stderr.extend(stderr_texts)
            continue
        if binary_name and executable_name == binary_name:
            combined_payload_stderr = "\n".join(stderr_texts)
            if returncode not in (None, 0) and _NEW_RUNTIME_MARKER not in combined_payload_stderr \
                    and _LEGACY_RUNTIME_MARKER not in combined_payload_stderr:
                continue
            relevant_stderr.extend(stderr_texts)
    if not relevant_stderr:
        return None

    combined_stderr = "\n".join(relevant_stderr)
    if _LEGACY_RUNTIME_MARKER in combined_stderr:
        return (
            f"E2E runtime guard failed for strategy {case.runtime_strategy}: "
            "TRT subprocess used legacy compatibility factory mode"
        )
    if _NEW_RUNTIME_MARKER not in combined_stderr:
        return (
            f"E2E runtime guard failed for strategy {case.runtime_strategy}: "
            "TRT subprocess did not confirm the new runtime path"
        )
    return None


def _build_repro_commands(
    case: E2ECase,
    ctx: RunContext,
    bundle_path: str | None,
    build_info: dict[str, Any],
) -> dict[str, str]:
    """Construct shell commands that reproduce each phase of the E2E test.

    Returns a dict with keys like "build_bundle", "trt_inference", "rerun_test",
    each mapped to a copy-pasteable shell command string.
    """
    repro: dict[str, str] = {}

    # Build command
    max_cache = case.inputs.get("max_cache_length", 256)
    bundle_target = bundle_path or str(Path(ctx.engine_dir) / case.bundle)
    build_parts = [
        "trtf-build", "build", case.hf_id,
        "-o", bundle_target,
        "--max-cache-length", str(max_cache),
    ]
    if case.metadata.get("trust_remote_code"):
        build_parts.append("--trust-remote-code")
    repro["build_bundle"] = " ".join(build_parts)

    # TRT inference command (task-specific C++ binary entrypoint)
    if bundle_path and ctx.binary_path:
        image = (case.inputs.get("image") or case.inputs.get("test_image")
                 or case.inputs.get("image_path"))
        task_strategy = case.task_strategy or ""
        if task_strategy == "diffusion_media_generation":
            infer_parts = [
                ctx.binary_path, "generate-video", bundle_path,
                "--prompt", _shell_quote(case.inputs.get("prompt", case.inputs.get("test_prompt", ""))),
                "--output", "/tmp/trtf_frames",
                "--num-steps", str(case.inputs.get("num_inference_steps", 30)),
            ]
        elif task_strategy == "prompted_segmentation":
            infer_parts = [
                ctx.binary_path, "segment-sam", bundle_path,
                "--image", str(image or ""),
                "--output", "/tmp/trtf_masks",
                "--point-x", str(case.inputs.get("point_x", 0.5)),
                "--point-y", str(case.inputs.get("point_y", 0.5)),
            ]
            if not case.inputs.get("is_foreground", True):
                infer_parts.append("--background")
        elif task_strategy == "segmentation":
            infer_parts = [
                ctx.binary_path, "segment", bundle_path,
                "--image", str(image or ""),
                "--output", "/tmp/trtf_segmentation.png",
            ]
        else:
            infer_parts = [
                ctx.binary_path, "run", bundle_path,
                "--prompt", _shell_quote(case.inputs.get("prompt", "")),
                "--max-new-tokens", str(case.inputs.get("max_new_tokens", 20)),
            ]
            if image:
                infer_parts.extend(["--image", str(image)])
        if ctx.hf_python:
            infer_parts.extend(["--hf-python", ctx.hf_python])
        repro["trt_inference"] = " ".join(infer_parts)

    # Rerun this exact test case
    rerun_parts = [
        "pytest", f"tests/test_e2e.py::test_e2e[{case.name}]", "-v",
        "--engine-dir", ctx.engine_dir,
        "--trtf-binary", ctx.binary_path,
        "--hf-python", ctx.hf_python or "python",
    ]
    repro["rerun_test"] = " ".join(rerun_parts)

    # Rerun with forced rebuild
    rebuild_parts = list(rerun_parts) + ["--rebuild-engines"]
    repro["rerun_test_rebuild"] = " ".join(rebuild_parts)

    return repro


def _shell_quote(s: str) -> str:
    """Simple shell quoting for inclusion in repro commands."""
    if not s:
        return '""'
    # If string contains special chars, wrap in single quotes
    if any(c in s for c in " \t\n\"'\\$!&|;(){}[]<>?*~`#"):
        # Escape single quotes inside
        escaped = s.replace("'", "'\\''")
        return f"'{escaped}'"
    return s


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

        # skip_reason: TRT inference still runs; only reference/comparison is skipped.
        skip_reason = case.metadata.get("skip_reason")

        # 1. Preflight
        t0 = time.monotonic()
        preflight_ok, preflight_details = run_preflight(case, ctx)
        timing["preflight_s"] = time.monotonic() - t0

        if not preflight_ok:
            result = E2EResult(
                case_name=case.name,
                status=E2EStatus.SKIP.value,
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
        bundle_path, build_time, build_err, build_info = _resolve_bundle(case, ctx)
        timing["build_s"] = time.monotonic() - t0
        if build_time is not None:
            timing["bundle_build_s"] = build_time

        # Log build subprocess output
        if build_info:
            sink.log_command(
                command=build_info.get("command", []),
                rc=build_info.get("returncode", -1),
                stdout=build_info.get("stdout", ""),
                stderr=build_info.get("stderr", ""),
                label="build",
            )

        if bundle_path is None:
            repro = _build_repro_commands(case, ctx, None, build_info)
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
                repro_commands=repro,
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
                    _log_stage_subprocess(sink, stage_name, trt_output, "trt")
                    _auto_register_artifacts(sink, trt_output, "trt")
                    runtime_path_error = _validate_trt_runtime_path(
                        case, ctx_with_bundle, trt_output)
                    if runtime_path_error is not None:
                        stage_results[stage_name] = CompareResult(
                            stage_name=stage_name,
                            status=StageStatus.ERROR.value,
                            message=f"TRT run failed: {runtime_path_error}",
                        )
                        if stage.required:
                            all_stages_pass = False
                        continue
                except Exception as e:
                    timing[f"trt_{stage_name}_s"] = time.monotonic() - t0
                    tb = traceback.format_exc()
                    logger.error("TRT run failed for stage %s: %s\n%s", stage_name, e, tb)
                    stage_results[stage_name] = CompareResult(
                        stage_name=stage_name,
                        status=StageStatus.ERROR.value,
                        message=f"TRT run failed: {e}\n{tb}",
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

            # Reference run (skipped when skip_reason is set)
            ref_output: StageOutput | None = None
            if skip_reason:
                logger.info(
                    "Skipping reference for %s stage %s: %s",
                    case.name, stage_name, skip_reason,
                )
            elif reference is not None:
                t0 = time.monotonic()
                try:
                    ref_output = reference.run_stage(case, stage, ctx_with_bundle)
                    timing[f"ref_{stage_name}_s"] = time.monotonic() - t0
                    sink.write_stage_output(stage_name, ref_output, prefix="ref")
                    _log_stage_subprocess(sink, stage_name, ref_output, "ref")
                    _auto_register_artifacts(sink, ref_output, "ref")
                except Exception as e:
                    timing[f"ref_{stage_name}_s"] = time.monotonic() - t0
                    tb = traceback.format_exc()
                    logger.error("Reference run failed for stage %s: %s\n%s", stage_name, e, tb)
                    stage_results[stage_name] = CompareResult(
                        stage_name=stage_name,
                        status=StageStatus.ERROR.value,
                        message=f"Reference run failed: {e}\n{tb}",
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
                    tb = traceback.format_exc()
                    compare_result = CompareResult(
                        stage_name=stage_name,
                        status=StageStatus.ERROR.value,
                        message=f"Comparison failed: {e}\n{tb}",
                    )
                stage_results[stage_name] = compare_result
                sink.write_compare(stage_name, compare_result)

                if not compare_result.passed and stage.required:
                    all_stages_pass = False
            elif trt_output is not None and (ref_output is None or comparator is None):
                # No reference or comparator — record as skipped
                stage_results[stage_name] = CompareResult(
                    stage_name=stage_name,
                    status=StageStatus.SKIPPED.value,
                    message="TRT run succeeded (no reference/comparator available)",
                )

        # 4. Determinism reruns (if configured)
        determinism_results: dict[str, Any] = {}
        reruns = case.determinism.get("reruns", 0)
        if reruns > 0 and runner is not None:
            determinism_results["reruns_requested"] = reruns
            # TODO: implement rerun logic when determinism policy is available
            determinism_results["status"] = "not_implemented"

        # 5. Build reproducibility commands
        repro = _build_repro_commands(case, ctx, bundle_path, build_info)

        # 6. Aggregate result
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
            repro_commands=repro,
        )

        # 7. Finalize artifacts
        try:
            sink.finalize(result)
        except Exception as e:
            logger.error("Failed to finalize artifacts: %s", e)
            result.failure_type = FailureType.ARTIFACT_WRITE_FAIL.value

        return result
