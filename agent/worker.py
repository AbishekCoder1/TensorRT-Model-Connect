from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .schemas import TaskRecord, TaskType
from .store import TaskStore
from .validator import validate_task
from .utils import slugify


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _template_for(task: TaskRecord) -> Path:
    base = Path(__file__).resolve().parent / "prompts"
    if task.task_type in {
        TaskType.IMPLEMENT_RUNTIME_DISPATCH,
        TaskType.IMPLEMENT_BUILDER,
        TaskType.FOUNDATION_BUILDER,
        TaskType.FOUNDATION_RUNTIME,
        TaskType.FOUNDATION_VALIDATOR,
        TaskType.FOUNDATION_CANARY,
    }:
        return base / "infra_builder.md"
    if task.task_type == TaskType.PARITY_VALIDATION:
        return base / "validation_runner.md"
    return base / "model_integrator.md"


def _render_prompt(task: TaskRecord) -> Path:
    template = _template_for(task)
    if not template.exists():
        raise FileNotFoundError(f"missing template: {template}")

    rendered = template.read_text(encoding="utf-8")
    replacements = {
        "{{TASK_ID}}": task.id,
        "{{TASK_TYPE}}": task.task_type.value,
        "{{MODEL_ID}}": task.model_id,
        "{{RUNTIME_STRATEGY}}": task.runtime_strategy,
        "{{MODALITY}}": task.modality,
        "{{TASK_JSON}}": json.dumps(task.to_dict(), indent=2),
    }
    for k, v in replacements.items():
        rendered = rendered.replace(k, v)

    out_dir = Path(__file__).resolve().parent / "prompts" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slugify(task.id)}.md"
    out_path.write_text(rendered, encoding="utf-8")
    return out_path


def _ensure_task_worktree(task: TaskRecord) -> Path:
    root = _project_root()
    wt_root = root / "agent" / "worktrees"
    wt_root.mkdir(parents=True, exist_ok=True)

    branch = task.branch or f"agent/{slugify(task.id)}"
    path = wt_root / slugify(branch)

    if path.exists() and (path / ".git").exists():
        return path
    if path.exists():
        raise RuntimeError(f"worktree path exists but is not a git worktree: {path}")

    create = subprocess.run(
        ["git", "worktree", "add", "-b", branch, str(path), "master"],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    if create.returncode == 0:
        return path

    # Branch may already exist from earlier retries; attach worktree to that branch.
    attach = subprocess.run(
        ["git", "worktree", "add", str(path), branch],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    if attach.returncode != 0:
        raise RuntimeError(
            "failed to create worktree: "
            f"create_err={create.stderr.strip()} attach_err={attach.stderr.strip()}"
        )
    return path


def _run_subagent(task: TaskRecord, worker_id: str) -> dict:
    root = _project_root()
    worktree = _ensure_task_worktree(task)
    prompt = _render_prompt(task)

    cmd = [
        "bash",
        str(root / "agent" / "run_subagent.sh"),
        "--task-json",
        json.dumps(task.to_dict()),
        "--prompt-file",
        str(prompt),
        "--worktree",
        str(worktree),
        "--worker-id",
        worker_id,
    ]
    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "prompt_file": str(prompt),
        "worktree": str(worktree),
    }


def _run_merge_ready_check(task: TaskRecord) -> tuple[bool, dict]:
    root = _project_root()
    branch = task.branch or f"agent/{slugify(task.id)}"

    checks = []
    for cmd in (
        ["git", "rev-parse", "--verify", branch],
        ["git", "show-ref", "--verify", f"refs/heads/{branch}"],
    ):
        proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
        checks.append({"cmd": cmd, "rc": proc.returncode, "stderr": proc.stderr})
        if proc.returncode != 0:
            return False, {"checks": checks}

    return True, {"checks": checks}


def handle_task(task: TaskRecord, worker_id: str) -> tuple[bool, dict]:
    if task.task_type == TaskType.PARITY_VALIDATION:
        result = validate_task(task)
        return result.passed, result.to_dict()

    if task.task_type == TaskType.MERGE_READY_CHECK:
        return _run_merge_ready_check(task)

    if task.task_type == TaskType.INTAKE_MODEL:
        return True, {"message": "intake already completed by orchestrator"}

    if task.task_type in {
        TaskType.FOUNDATION_BUILDER,
        TaskType.FOUNDATION_RUNTIME,
        TaskType.FOUNDATION_VALIDATOR,
        TaskType.FOUNDATION_CANARY,
        TaskType.IMPLEMENT_PLUGIN,
        TaskType.IMPLEMENT_BUILDER,
        TaskType.IMPLEMENT_RUNTIME_DISPATCH,
        TaskType.ADD_MANIFEST,
    }:
        details = _run_subagent(task, worker_id)
        return details["returncode"] == 0, details

    return False, {"error": f"unhandled task type {task.task_type.value}"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute a claimed task")
    parser.add_argument("--store", default="agent")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--gpu-id", default="")
    args = parser.parse_args()

    store = TaskStore(args.store)
    task = store.get_task(args.task_id)
    if task is None:
        raise SystemExit(f"task not found: {args.task_id}")

    ok = False
    details: dict = {}
    try:
        ok, details = handle_task(task, args.worker_id)
    except Exception as exc:
        details = {"exception": str(exc)}
        ok = False

    artifact_dir = Path(store.root) / "artifacts" / slugify(task.id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "result.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")

    if ok:
        if task.task_type == TaskType.MERGE_READY_CHECK:
            store.mark_merge_pending(task.id, artifacts={"artifact_dir": str(artifact_dir)})
        else:
            store.mark_completed(task.id, artifacts={"artifact_dir": str(artifact_dir)})
        return 0

    err = details.get("error") or details.get("exception") or details.get("stderr") or "task failed"
    store.mark_failed(task.id, str(err))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
