from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .schemas import TaskRecord
from .store import TaskStore
from .utils import run_cmd, slugify


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@contextmanager
def _merge_lock(root: Path) -> Iterator[None]:
    lock_path = root / "agent" / "state" / "merge.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _run(cmd: list[str], cwd: Path, timeout: int = 1800) -> tuple[bool, str, str]:
    rc, out, err = run_cmd(cmd, cwd=cwd, timeout=timeout)
    return rc == 0, out, err


def _load_canary_config(root: Path) -> dict:
    path = root / "agent" / "config" / "canaries.json"
    if not path.exists():
        return {
            "tier_a": {"global": [], "by_modality": {}, "by_runtime_strategy": {}},
            "tier_b": {"global": [], "by_modality": {}, "by_runtime_strategy": {}},
        }

    raw = json.loads(path.read_text(encoding="utf-8"))
    if "tier_a" in raw or "tier_b" in raw:
        cfg = {
            "tier_a": raw.get("tier_a", {}),
            "tier_b": raw.get("tier_b", {}),
        }
    else:
        # Backward-compatible legacy format.
        cfg = {
            "tier_a": {
                "global": raw.get("global", []),
                "by_modality": raw.get("by_modality", {}),
                "by_runtime_strategy": raw.get("by_runtime_strategy", {}),
            },
            "tier_b": {"global": [], "by_modality": {}, "by_runtime_strategy": {}},
        }
    for tier in ("tier_a", "tier_b"):
        tcfg = cfg.setdefault(tier, {})
        tcfg.setdefault("global", [])
        tcfg.setdefault("by_modality", {})
        tcfg.setdefault("by_runtime_strategy", {})
    return cfg


def _commands_for_tier(cfg: dict, tier: str, task: TaskRecord) -> list:
    tcfg = cfg.get(tier, {})
    cmds = []
    cmds.extend(tcfg.get("global", []))
    cmds.extend(tcfg.get("by_modality", {}).get(task.modality, []))
    cmds.extend(tcfg.get("by_runtime_strategy", {}).get(task.runtime_strategy, []))
    return cmds


def _run_gate(task: TaskRecord, root: Path) -> tuple[bool, list[dict]]:
    cfg = _load_canary_config(root)
    run_tier_b = bool(task.metadata.get("run_tier_b", False))
    tiers = ["tier_a"] + (["tier_b"] if run_tier_b else [])
    results = []
    for tier in tiers:
        for cmd in _commands_for_tier(cfg, tier, task):
            timeout = 2400
            rendered = cmd
            if isinstance(cmd, dict):
                rendered = str(cmd.get("cmd", "")).strip()
                timeout = int(cmd.get("timeout_s", timeout))
            if not rendered:
                continue

            ok, out, err = _run(["bash", "-lc", rendered], cwd=root, timeout=timeout)
            results.append(
                {
                    "tier": tier,
                    "cmd": rendered,
                    "ok": ok,
                    "stdout": out[-2000:],
                    "stderr": err[-2000:],
                    "timeout_s": timeout,
                }
            )
            if not ok:
                return False, results

    return True, results


def _ensure_clean_repo(root: Path) -> tuple[bool, str]:
    ok, out, _ = _run(["git", "status", "--porcelain"], cwd=root)
    if not ok:
        return False, "failed to read git status"
    if out.strip():
        return False, "repository has local changes; merge manager requires a clean checkout"
    return True, ""


def _integrate_task(task: TaskRecord, root: Path) -> tuple[bool, dict]:
    branch = task.branch
    if not branch:
        return False, {"error": "task branch missing"}

    clean_ok, clean_error = _ensure_clean_repo(root)
    if not clean_ok:
        return False, {"error": clean_error}

    temp_branch = f"agent/integration-{slugify(task.id)}"
    steps: list[dict] = []

    def step(cmd: list[str], timeout: int = 1800) -> bool:
        ok, out, err = _run(cmd, cwd=root, timeout=timeout)
        steps.append(
            {
                "cmd": cmd,
                "ok": ok,
                "stdout": out[-1200:],
                "stderr": err[-1200:],
            }
        )
        return ok

    if not step(["git", "show-ref", "--verify", f"refs/heads/{branch}"]):
        return False, {"steps": steps, "error": f"branch not found: {branch}"}

    for cmd in (
        ["git", "checkout", "master"],
        ["git", "branch", "-D", temp_branch],
    ):
        ok = step(cmd)
        # Temp branch deletion may fail if it does not exist.
        if cmd[2] == "-D" and not ok:
            steps[-1]["ignored"] = True
            continue
        if not ok:
            return False, {"steps": steps, "error": "failed preparing integration branch"}

    if not step(["git", "checkout", "-b", temp_branch, "master"]):
        return False, {"steps": steps, "error": "failed creating integration branch"}

    if not step(["git", "merge", "--no-ff", branch, "-m", f"merge({task.id})"]):
        step(["git", "merge", "--abort"])
        step(["git", "checkout", "master"])
        step(["git", "branch", "-D", temp_branch])
        return False, {"steps": steps, "error": "merge conflict"}

    gate_ok, gate_steps = _run_gate(task, root)
    steps.append({"gate": gate_steps, "ok": gate_ok})
    if not gate_ok:
        step(["git", "checkout", "master"])
        step(["git", "branch", "-D", temp_branch])
        return False, {"steps": steps, "error": "merge gate failed"}

    if not step(["git", "checkout", "master"]):
        return False, {"steps": steps, "error": "failed to return to master"}

    if not step(["git", "merge", "--ff-only", temp_branch]):
        return False, {"steps": steps, "error": "failed to fast-forward master"}

    step(["git", "branch", "-D", temp_branch])
    return True, {"steps": steps}


def main() -> int:
    parser = argparse.ArgumentParser(description="Serial merge manager")
    parser.add_argument("--store", default="agent")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    root = _project_root()
    store = TaskStore(args.store)
    store.ensure_layout()

    with _merge_lock(root):
        pending = store.merge_pending_tasks()
        if not pending:
            print("no merge-pending tasks")
            return 0

        # Serial merge by priority/created time order already kept in backlog.
        for task in pending:
            ok, details = _integrate_task(task, root)
            artifact_dir = Path(store.root) / "artifacts" / f"merge-{task.id.replace(':', '-') }"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "merge.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")

            if ok:
                store.mark_merged(task.id, artifacts={"merge_artifact_dir": str(artifact_dir)})
                print(f"merged={task.id}")
            else:
                store.mark_failed(task.id, details.get("error", "merge failed"))
                print(f"merge_failed={task.id}")
                if args.once:
                    return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
