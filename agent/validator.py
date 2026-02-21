from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schemas import TaskRecord
from .utils import normalize_hf_model_id, run_cmd, slugify


@dataclass
class CheckResult:
    name: str
    passed: bool
    command: str
    duration_s: float
    stdout: str = ""
    stderr: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    model_id: str
    runtime_strategy: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "runtime_strategy": self.runtime_strategy,
            "passed": self.passed,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "command": c.command,
                    "duration_s": c.duration_s,
                    "metrics": c.metrics,
                    "stdout_tail": c.stdout[-1200:],
                    "stderr_tail": c.stderr[-1200:],
                }
                for c in self.checks
            ],
        }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(name: str, cmd: list[str], cwd: Path, timeout: int = 1800, env: dict | None = None) -> CheckResult:
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout, env=env)
    dt = time.monotonic() - t0
    return CheckResult(
        name=name,
        passed=(proc.returncode == 0),
        command=" ".join(shlex.quote(x) for x in cmd),
        duration_s=dt,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def _resolve_model_dir(model_id: str, cwd: Path) -> str:
    code = "from trtf_build.engine_builder import _resolve_model; import sys; print(_resolve_model(sys.argv[1]))"
    rc, out, err = run_cmd(["python3", "-c", code, model_id], cwd=cwd)
    if rc != 0:
        raise RuntimeError(f"failed to resolve model directory: {err.strip()}")
    return out.strip()


def _extract_generated_text(raw: str) -> str:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return ""

    for line in reversed(lines):
        if "generated_text" not in line:
            continue
        try:
            payload = ast.literal_eval(line)
            if isinstance(payload, list) and payload and isinstance(payload[0], dict):
                value = payload[0].get("generated_text")
                if isinstance(value, str):
                    return value.strip()
            if isinstance(payload, dict):
                value = payload.get("generated_text")
                if isinstance(value, str):
                    return value.strip()
        except Exception:
            continue

    return lines[-1].strip()


def _compare_text_exact(cpp_text: str, hf_text: str) -> tuple[bool, dict[str, Any]]:
    cpp_norm = _extract_generated_text(cpp_text)
    hf_norm = _extract_generated_text(hf_text)
    return cpp_norm == hf_norm, {
        "cpp_sha256": hashlib.sha256(cpp_norm.encode("utf-8")).hexdigest(),
        "hf_sha256": hashlib.sha256(hf_norm.encode("utf-8")).hexdigest(),
        "cpp_preview": cpp_norm[:200],
        "hf_preview": hf_norm[:200],
    }


def _build_bundle(model_id: str, runtime_strategy: str, cwd: Path, cache_len: int = 256) -> tuple[CheckResult, Path]:
    slug = slugify(normalize_hf_model_id(model_id))
    bundle = Path("/tmp") / f"agent-{slug}.trtfb"
    if runtime_strategy in {"encoder_only", "encoder_decoder", "vision_encoder", "audio_encoder"}:
        cache_len = min(cache_len, 128)
    cmd = ["trtf-build", "build", model_id, "-o", str(bundle), "--max-cache-length", str(cache_len)]
    return _run("build_bundle", cmd, cwd=cwd, timeout=3600), bundle


def _ld_env(base: dict[str, str], cwd: Path) -> dict[str, str]:
    env = dict(base)
    try:
        code = "import importlib.util; s=importlib.util.find_spec('tensorrt_libs'); print(s.submodule_search_locations[0] if s else '')"
        rc, out, _ = run_cmd(["python3", "-c", code], cwd=cwd)
        if rc == 0 and out.strip():
            env["LD_LIBRARY_PATH"] = f"{out.strip()}:/usr/local/cuda/lib64:{env.get('LD_LIBRARY_PATH', '')}"
    except Exception:
        pass
    return env


def _run_template_hook(
    *,
    result: ValidationResult,
    name: str,
    env_var: str,
    cwd: Path,
    template_values: dict[str, Any],
    timeout: int = 3600,
    missing_hint: str,
) -> None:
    tpl = os.environ.get(env_var, "").strip()
    if not tpl:
        result.checks.append(
            CheckResult(
                name=name,
                passed=False,
                command=env_var,
                duration_s=0.0,
                stderr=missing_hint,
            )
        )
        return

    try:
        cmd = tpl.format(**template_values)
    except Exception as exc:
        result.checks.append(
            CheckResult(
                name=name,
                passed=False,
                command=env_var,
                duration_s=0.0,
                stderr=f"failed to render template: {exc}",
            )
        )
        return

    result.checks.append(_run(name, ["bash", "-lc", cmd], cwd=cwd, timeout=timeout))


def _decoder_validation(
    model_id: str,
    runtime_strategy: str,
    cwd: Path,
    prompt: str,
    max_new_tokens: int,
    atol: float,
    cache_len: int,
) -> ValidationResult:
    res = ValidationResult(model_id=model_id, runtime_strategy=runtime_strategy)

    res.checks.append(
        _run(
            "diff_logits_battery",
            [
                "python3",
                "tools/diff_logits.py",
                "--model",
                model_id,
                "--battery",
                "--max-new-tokens",
                str(max_new_tokens),
                "--atol",
                str(atol),
            ],
            cwd=cwd,
            timeout=3600,
        )
    )

    bundle_check, bundle_path = _build_bundle(model_id, runtime_strategy, cwd, cache_len=cache_len)
    res.checks.append(bundle_check)
    if not bundle_check.passed:
        return res

    cpp = _run(
        "cpp_run",
        [
            "./build/trtf",
            "run",
            str(bundle_path),
            "--prompt",
            prompt,
            "--max-new-tokens",
            str(max_new_tokens),
            "--hf-python",
            str((cwd / ".venv" / "bin" / "python")),
        ],
        cwd=cwd,
        timeout=1200,
        env=_ld_env(os.environ.copy(), cwd),
    )
    res.checks.append(cpp)
    if not cpp.passed:
        return res

    try:
        model_dir = _resolve_model_dir(model_id, cwd)
    except Exception as exc:
        res.checks.append(
            CheckResult(
                name="resolve_model_dir",
                passed=False,
                command="python3 -c _resolve_model",
                duration_s=0.0,
                stderr=str(exc),
            )
        )
        return res

    hf = _run(
        "hf_generate",
        [
            "python3",
            "scripts/hf_generate.py",
            "--model-dir",
            model_dir,
            "--prompt",
            prompt,
            "--max-new-tokens",
            str(max_new_tokens),
            "--do-sample",
            "0",
            "--temperature",
            "1.0",
        ],
        cwd=cwd,
        timeout=1200,
    )
    res.checks.append(hf)
    if not hf.passed:
        return res

    ok, metrics = _compare_text_exact(cpp.stdout, hf.stdout)
    res.checks.append(
        CheckResult(
            name="exact_text_parity_cpp_vs_hf",
            passed=ok,
            command="compare cpp output to hf output",
            duration_s=0.0,
            metrics=metrics,
        )
    )

    return res


def _encoder_validation(
    model_id: str,
    runtime_strategy: str,
    cwd: Path,
    atol: float,
    prompt: str,
    max_new_tokens: int,
    cache_len: int,
) -> ValidationResult:
    res = ValidationResult(model_id=model_id, runtime_strategy=runtime_strategy)

    res.checks.append(
        _run(
            "diff_encoder_battery",
            [
                "python3",
                "tools/diff_encoder.py",
                "--model",
                model_id,
                "--battery",
                "--atol",
                str(atol),
            ],
            cwd=cwd,
            timeout=3600,
        )
    )

    bundle_check, bundle_path = _build_bundle(model_id, runtime_strategy, cwd, cache_len=cache_len)
    res.checks.append(bundle_check)
    if not bundle_check.passed:
        return res

    _run_template_hook(
        result=res,
        name="encoder_cpp_parity",
        env_var="TRTF_ENCODER_CPP_PARITY_CMD_TEMPLATE",
        cwd=cwd,
        template_values={
            "model": model_id,
            "runtime": runtime_strategy,
            "bundle": str(bundle_path),
            "prompt": prompt,
            "max_new_tokens": str(max_new_tokens),
            "binary": str(cwd / "build" / "trtf"),
            "hf_python": str(cwd / ".venv" / "bin" / "python"),
        },
        missing_hint=(
            "Missing strict C++ parity hook. Set TRTF_ENCODER_CPP_PARITY_CMD_TEMPLATE "
            "(supports {model}, {bundle}, {binary}, {hf_python}) to enforce exact encoder parity."
        ),
    )
    return res


def _encoder_decoder_validation(
    model_id: str,
    runtime_strategy: str,
    cwd: Path,
    prompt: str,
    max_new_tokens: int,
    cache_len: int,
) -> ValidationResult:
    res = ValidationResult(model_id=model_id, runtime_strategy=runtime_strategy)
    bundle_check, bundle_path = _build_bundle(model_id, runtime_strategy, cwd, cache_len=cache_len)
    res.checks.append(bundle_check)
    if not bundle_check.passed:
        return res

    _run_template_hook(
        result=res,
        name="encoder_decoder_cpp_parity",
        env_var="TRTF_ENCODER_DECODER_CPP_PARITY_CMD_TEMPLATE",
        cwd=cwd,
        template_values={
            "model": model_id,
            "runtime": runtime_strategy,
            "bundle": str(bundle_path),
            "prompt": prompt,
            "max_new_tokens": str(max_new_tokens),
            "binary": str(cwd / "build" / "trtf"),
            "hf_python": str(cwd / ".venv" / "bin" / "python"),
        },
        missing_hint=(
            "Missing encoder/decoder parity hook. Set TRTF_ENCODER_DECODER_CPP_PARITY_CMD_TEMPLATE "
            "to run strict C++ vs HF output comparison."
        ),
    )
    return res


def _single_encoder_hook_validation(
    model_id: str,
    runtime_strategy: str,
    cwd: Path,
    cache_len: int,
    *,
    check_name: str,
    env_var: str,
    hint: str,
) -> ValidationResult:
    res = ValidationResult(model_id=model_id, runtime_strategy=runtime_strategy)
    bundle_check, bundle_path = _build_bundle(model_id, runtime_strategy, cwd, cache_len=cache_len)
    res.checks.append(bundle_check)
    if not bundle_check.passed:
        return res

    _run_template_hook(
        result=res,
        name=check_name,
        env_var=env_var,
        cwd=cwd,
        template_values={
            "model": model_id,
            "runtime": runtime_strategy,
            "bundle": str(bundle_path),
            "binary": str(cwd / "build" / "trtf"),
            "hf_python": str(cwd / ".venv" / "bin" / "python"),
        },
        missing_hint=hint,
    )
    return res


def _template_only_validation(
    model_id: str,
    runtime_strategy: str,
    cwd: Path,
    cache_len: int,
    *,
    check_name: str,
    env_var: str,
    hint: str,
    prompt: str,
    max_new_tokens: int,
    extra_values: dict[str, Any] | None = None,
) -> ValidationResult:
    res = ValidationResult(model_id=model_id, runtime_strategy=runtime_strategy)
    bundle_check, bundle_path = _build_bundle(model_id, runtime_strategy, cwd, cache_len=cache_len)
    res.checks.append(bundle_check)
    if not bundle_check.passed:
        return res

    values = {
        "model": model_id,
        "runtime": runtime_strategy,
        "bundle": str(bundle_path),
        "binary": str(cwd / "build" / "trtf"),
        "hf_python": str(cwd / ".venv" / "bin" / "python"),
        "prompt": prompt,
        "max_new_tokens": str(max_new_tokens),
    }
    if extra_values:
        values.update(extra_values)

    _run_template_hook(
        result=res,
        name=check_name,
        env_var=env_var,
        cwd=cwd,
        template_values=values,
        missing_hint=hint,
        timeout=7200,
    )
    return res


def _vl_validation(
    model_id: str,
    runtime_strategy: str,
    cwd: Path,
    prompt: str,
    max_new_tokens: int,
    atol: float,
    cache_len: int,
) -> ValidationResult:
    res = ValidationResult(model_id=model_id, runtime_strategy=runtime_strategy)
    image_path = os.environ.get("TRTF_VL_IMAGE", "").strip()
    if not image_path:
        res.checks.append(
            CheckResult(
                name="vl_image_provided",
                passed=False,
                command="TRTF_VL_IMAGE=<path>",
                duration_s=0.0,
                stderr="VL validation requires TRTF_VL_IMAGE pointing to an image file",
            )
        )
        return res

    bundle_check, bundle_path = _build_bundle(model_id, runtime_strategy, cwd, cache_len=cache_len)
    res.checks.append(bundle_check)
    if not bundle_check.passed:
        return res

    res.checks.append(
        _run(
            "diff_vl",
            [
                "python3",
                "tools/diff_vl.py",
                "--bundle",
                str(bundle_path),
                "--image",
                image_path,
                "--model",
                model_id,
                "--atol",
                str(atol),
                "--binary",
                "./build/trtf",
                "--hf-python",
                str((cwd / ".venv" / "bin" / "python")),
                "--max-new-tokens",
                str(max_new_tokens),
            ],
            cwd=cwd,
            timeout=3600,
        )
    )
    return res


def _video_diffusion_validation(
    model_id: str,
    runtime_strategy: str,
    cwd: Path,
    prompt: str,
    max_new_tokens: int,
    cache_len: int,
) -> ValidationResult:
    out_dir = cwd / "agent" / "artifacts" / f"video-diffusion-{slugify(model_id)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return _template_only_validation(
        model_id=model_id,
        runtime_strategy=runtime_strategy,
        cwd=cwd,
        cache_len=cache_len,
        check_name="video_diffusion_frame_parity",
        env_var="TRTF_VIDEO_DIFFUSION_FRAME_PARITY_CMD_TEMPLATE",
        hint=(
            "Missing video diffusion parity hook. Set "
            "TRTF_VIDEO_DIFFUSION_FRAME_PARITY_CMD_TEMPLATE to compare generated video frames "
            "against a Hugging Face reference."
        ),
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        extra_values={"output_dir": str(out_dir)},
    )


def _time_series_validation(
    model_id: str,
    runtime_strategy: str,
    cwd: Path,
    prompt: str,
    max_new_tokens: int,
    cache_len: int,
) -> ValidationResult:
    return _template_only_validation(
        model_id=model_id,
        runtime_strategy=runtime_strategy,
        cwd=cwd,
        cache_len=cache_len,
        check_name="time_series_cpp_parity",
        env_var="TRTF_TIME_SERIES_CPP_PARITY_CMD_TEMPLATE",
        hint=(
            "Missing time-series parity hook. Set "
            "TRTF_TIME_SERIES_CPP_PARITY_CMD_TEMPLATE for strict C++ vs HF tensor parity."
        ),
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )


def _rl_validation(
    model_id: str,
    runtime_strategy: str,
    cwd: Path,
    prompt: str,
    max_new_tokens: int,
    cache_len: int,
) -> ValidationResult:
    if runtime_strategy == "rl_value_inference":
        env_var = "TRTF_RL_VALUE_CPP_PARITY_CMD_TEMPLATE"
        check_name = "rl_value_cpp_parity"
        hint = (
            "Missing RL value parity hook. Set "
            "TRTF_RL_VALUE_CPP_PARITY_CMD_TEMPLATE for strict C++ vs HF value-head parity."
        )
    else:
        env_var = "TRTF_RL_POLICY_CPP_PARITY_CMD_TEMPLATE"
        check_name = "rl_policy_cpp_parity"
        hint = (
            "Missing RL policy parity hook. Set "
            "TRTF_RL_POLICY_CPP_PARITY_CMD_TEMPLATE for strict C++ vs HF policy-logit parity."
        )
    return _template_only_validation(
        model_id=model_id,
        runtime_strategy=runtime_strategy,
        cwd=cwd,
        cache_len=cache_len,
        check_name=check_name,
        env_var=env_var,
        hint=hint,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )


def _collect_frame_stats(frame_dir: Path) -> dict[str, Any]:
    from PIL import Image
    import numpy as np

    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        return {"count": 0, "mean": 0.0, "std": 0.0}
    pixels = []
    for fp in frames:
        arr = np.array(Image.open(fp).convert("RGB"), dtype=np.float32) / 255.0
        pixels.append(arr.flatten())
    merged = np.concatenate(pixels)
    return {
        "count": len(frames),
        "mean": float(merged.mean()),
        "std": float(merged.std()),
        "min": float(merged.min()),
        "max": float(merged.max()),
    }


def _build_contact_sheet(frame_dir: Path, out_path: Path) -> None:
    from PIL import Image

    frames = sorted(frame_dir.glob("frame_*.png"))[:9]
    if not frames:
        return
    imgs = [Image.open(fp).convert("RGB") for fp in frames]
    w, h = imgs[0].size
    grid = Image.new("RGB", (w * 3, h * 3))
    for idx, img in enumerate(imgs):
        x = (idx % 3) * w
        y = (idx // 3) * h
        grid.paste(img, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def _diffusion_validation(
    model_id: str,
    runtime_strategy: str,
    cwd: Path,
    prompt: str,
    cache_len: int,
) -> ValidationResult:
    res = ValidationResult(model_id=model_id, runtime_strategy=runtime_strategy)

    res.checks.append(
        _run(
            "debug_diffusion_pipeline",
            ["python3", "tools/debug_diffusion_pipeline.py", "--model", model_id],
            cwd=cwd,
            timeout=7200,
        )
    )

    bundle_check, bundle_path = _build_bundle(model_id, runtime_strategy, cwd, cache_len=cache_len)
    res.checks.append(bundle_check)
    if not bundle_check.passed:
        return res

    out_dir = cwd / "agent" / "artifacts" / f"diffusion-{slugify(model_id)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    video = _run(
        "cpp_generate_video",
        [
            "./build/trtf",
            "generate-video",
            str(bundle_path),
            "--prompt",
            prompt,
            "--output",
            str(out_dir),
            "--num-steps",
            "30",
            "--guidance-scale",
            "5.0",
            "--hf-python",
            str((cwd / ".venv" / "bin" / "python")),
        ],
        cwd=cwd,
        timeout=7200,
    )
    res.checks.append(video)
    if not video.passed:
        return res

    stats = _collect_frame_stats(out_dir)
    passed = stats.get("count", 0) > 0 and stats.get("std", 0.0) > 0.01
    res.checks.append(
        CheckResult(
            name="diffusion_frame_stats",
            passed=passed,
            command="compute frame mean/std",
            duration_s=0.0,
            metrics=stats,
            stderr="Low-variance output indicates likely collapse" if not passed else "",
        )
    )
    _build_contact_sheet(out_dir, out_dir / "contact_sheet.png")

    _run_template_hook(
        result=res,
        name="diffusion_frame_parity",
        env_var="TRTF_DIFFUSION_FRAME_PARITY_CMD_TEMPLATE",
        cwd=cwd,
        template_values={
            "model": model_id,
            "runtime": runtime_strategy,
            "bundle": str(bundle_path),
            "output_dir": str(out_dir),
            "binary": str(cwd / "build" / "trtf"),
            "hf_python": str(cwd / ".venv" / "bin" / "python"),
            "prompt": prompt,
        },
        missing_hint=(
            "Missing diffusion frame parity hook. Set TRTF_DIFFUSION_FRAME_PARITY_CMD_TEMPLATE "
            "to compare generated frames/images against a Hugging Face reference."
        ),
        timeout=7200,
    )

    return res


def _load_validation_profiles(cwd: Path) -> dict[str, Any]:
    path = cwd / "agent" / "config" / "validation_profiles.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _task_validation_settings(task: TaskRecord, cwd: Path) -> dict[str, Any]:
    cfg = _load_validation_profiles(cwd)
    out: dict[str, Any] = {
        "atol": 1e-3,
        "prompt": "The capital of France is",
        "max_new_tokens": 20,
        "cache_len": 256,
    }

    def merge(src: dict[str, Any] | None) -> None:
        if not isinstance(src, dict):
            return
        for key in ("atol", "prompt", "max_new_tokens", "cache_len"):
            if key in src:
                out[key] = src[key]

    merge(cfg.get("defaults"))
    merge(cfg.get("by_modality", {}).get(task.modality))
    merge(cfg.get("by_runtime_strategy", {}).get(task.runtime_strategy))
    merge(task.metadata.get("validation"))

    # Backward-compatible task-level overrides.
    if "logit_atol" in task.metadata:
        out["atol"] = task.metadata["logit_atol"]
    for key in ("prompt", "max_new_tokens", "cache_len"):
        if key in task.metadata:
            out[key] = task.metadata[key]

    out["atol"] = float(out["atol"])
    out["prompt"] = str(out["prompt"])
    out["max_new_tokens"] = int(out["max_new_tokens"])
    out["cache_len"] = int(out["cache_len"])
    return out


def validate_model(
    model_id: str,
    runtime_strategy: str,
    *,
    atol: float = 1e-3,
    prompt: str = "The capital of France is",
    max_new_tokens: int = 20,
    cache_len: int = 256,
) -> ValidationResult:
    cwd = _project_root()
    if runtime_strategy in {"decoder_kv_cache", "decoder_moe", "ssm_recurrent"}:
        return _decoder_validation(model_id, runtime_strategy, cwd, prompt, max_new_tokens, atol, cache_len)
    if runtime_strategy == "encoder_only":
        return _encoder_validation(model_id, runtime_strategy, cwd, atol, prompt, max_new_tokens, cache_len)
    if runtime_strategy == "encoder_decoder":
        return _encoder_decoder_validation(model_id, runtime_strategy, cwd, prompt, max_new_tokens, cache_len)
    if runtime_strategy == "vision_encoder":
        return _single_encoder_hook_validation(
            model_id,
            runtime_strategy,
            cwd,
            cache_len,
            check_name="vision_encoder_cpp_parity",
            env_var="TRTF_VISION_ENCODER_CPP_PARITY_CMD_TEMPLATE",
            hint=(
                "Missing vision encoder parity hook. Set "
                "TRTF_VISION_ENCODER_CPP_PARITY_CMD_TEMPLATE for strict C++ vs HF comparison."
            ),
        )
    if runtime_strategy == "audio_encoder":
        return _single_encoder_hook_validation(
            model_id,
            runtime_strategy,
            cwd,
            cache_len,
            check_name="audio_encoder_cpp_parity",
            env_var="TRTF_AUDIO_ENCODER_CPP_PARITY_CMD_TEMPLATE",
            hint=(
                "Missing audio encoder parity hook. Set "
                "TRTF_AUDIO_ENCODER_CPP_PARITY_CMD_TEMPLATE for strict C++ vs HF comparison."
            ),
        )
    if runtime_strategy == "vision_language":
        return _vl_validation(model_id, runtime_strategy, cwd, prompt, max_new_tokens, atol, cache_len)
    if runtime_strategy == "diffusion":
        return _diffusion_validation(model_id, runtime_strategy, cwd, prompt, cache_len)
    if runtime_strategy == "video_diffusion":
        return _video_diffusion_validation(model_id, runtime_strategy, cwd, prompt, max_new_tokens, cache_len)
    if runtime_strategy == "video_encoder":
        return _template_only_validation(
            model_id=model_id,
            runtime_strategy=runtime_strategy,
            cwd=cwd,
            cache_len=cache_len,
            check_name="video_encoder_cpp_parity",
            env_var="TRTF_VIDEO_ENCODER_CPP_PARITY_CMD_TEMPLATE",
            hint=(
                "Missing video encoder parity hook. Set "
                "TRTF_VIDEO_ENCODER_CPP_PARITY_CMD_TEMPLATE for strict C++ vs HF comparison."
            ),
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
    if runtime_strategy in {"time_series_encoder", "time_series_forecaster"}:
        return _time_series_validation(model_id, runtime_strategy, cwd, prompt, max_new_tokens, cache_len)
    if runtime_strategy in {"rl_policy_inference", "rl_value_inference"}:
        return _rl_validation(model_id, runtime_strategy, cwd, prompt, max_new_tokens, cache_len)

    out = ValidationResult(model_id=model_id, runtime_strategy=runtime_strategy)
    out.checks.append(
        CheckResult(
            name="runtime_support",
            passed=False,
            command="runtime support check",
            duration_s=0.0,
            stderr=f"Runtime strategy {runtime_strategy!r} not yet supported by strict validator",
        )
    )
    return out


def validate_task(task: TaskRecord) -> ValidationResult:
    settings = _task_validation_settings(task, _project_root())
    return validate_model(
        model_id=task.model_id,
        runtime_strategy=task.runtime_strategy,
        atol=float(settings["atol"]),
        prompt=str(settings["prompt"]),
        max_new_tokens=int(settings["max_new_tokens"]),
        cache_len=int(settings["cache_len"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Standardized model parity validator")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--runtime-strategy", required=True)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--cache-len", type=int, default=256)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    result = validate_model(
        model_id=args.model_id,
        runtime_strategy=args.runtime_strategy,
        atol=args.atol,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        cache_len=args.cache_len,
    )
    payload = result.to_dict()
    print(json.dumps(payload, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
