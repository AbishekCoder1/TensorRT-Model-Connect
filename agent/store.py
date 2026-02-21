from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .schemas import (
    TERMINAL_STATUSES,
    ResourceProfile,
    TaskRecord,
    TaskStatus,
    utc_now,
)


class TaskStore:
    def __init__(self, root: Path | str = "agent") -> None:
        self.root = Path(root)
        self.backlog_path = self.root / "backlog.json"
        self.lock_path = self.root / "state" / "backlog.lock"

    def ensure_layout(self) -> None:
        for sub in (
            "claimed",
            "completed",
            "failed",
            "logs",
            "state",
            "inbox",
            "config",
            "prompts",
            "worktrees",
            "artifacts",
        ):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        if not self.backlog_path.exists():
            self.backlog_path.write_text(
                json.dumps({"version": 1, "tasks": []}, indent=2) + "\n",
                encoding="utf-8",
            )
        (self.root / "inbox" / "links.jsonl").touch(exist_ok=True)

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.ensure_layout()
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o644)
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

    def _load_backlog_unlocked(self) -> dict:
        return json.loads(self.backlog_path.read_text(encoding="utf-8"))

    def _save_backlog_unlocked(self, data: dict) -> None:
        tmp = self.backlog_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.backlog_path)

    def list_tasks(self) -> list[TaskRecord]:
        with self._lock():
            data = self._load_backlog_unlocked()
            return [TaskRecord.from_dict(t) for t in data.get("tasks", [])]

    def get_task(self, task_id: str) -> TaskRecord | None:
        for task in self.list_tasks():
            if task.id == task_id:
                return task
        return None

    def upsert_tasks(self, tasks: list[TaskRecord]) -> None:
        with self._lock():
            data = self._load_backlog_unlocked()
            by_id = {t["id"]: t for t in data.get("tasks", [])}
            for task in tasks:
                task.touch()
                by_id[task.id] = task.to_dict()
            data["tasks"] = sorted(
                by_id.values(),
                key=lambda t: (int(t.get("priority", 50)), t.get("created_at", ""), t["id"]),
            )
            self._save_backlog_unlocked(data)

    def set_task(self, task: TaskRecord) -> None:
        self.upsert_tasks([task])

    def _dependency_satisfied(self, task: TaskRecord, task_map: dict[str, TaskRecord]) -> bool:
        for dep in task.depends_on:
            dep_task = task_map.get(dep)
            if dep_task is None:
                return False
            if dep_task.status not in {TaskStatus.COMPLETED, TaskStatus.MERGED}:
                return False
        return True

    def ready_tasks(self, allowed_profiles: set[ResourceProfile] | None = None) -> list[TaskRecord]:
        tasks = self.list_tasks()
        task_map = {t.id: t for t in tasks}
        out: list[TaskRecord] = []
        for task in tasks:
            if task.is_terminal:
                continue
            if task.status in {TaskStatus.DISPATCHED, TaskStatus.RUNNING, TaskStatus.VALIDATING, TaskStatus.MERGE_PENDING}:
                continue
            if allowed_profiles and task.resource_profile not in allowed_profiles:
                continue
            if self._dependency_satisfied(task, task_map):
                if task.status == TaskStatus.NEW:
                    task.status = TaskStatus.READY
                    task.touch()
                    self.set_task(task)
                out.append(task)
        return sorted(out, key=lambda t: (t.priority, t.created_at, t.id))

    def claim_task(self, task_id: str, worker_id: str) -> bool:
        with self._lock():
            data = self._load_backlog_unlocked()
            for idx, raw in enumerate(data.get("tasks", [])):
                if raw.get("id") != task_id:
                    continue
                task = TaskRecord.from_dict(raw)
                if task.status not in {TaskStatus.READY, TaskStatus.NEW}:
                    return False
                claim = self.root / "claimed" / f"{task_id}.lock"
                if claim.exists():
                    return False
                task.status = TaskStatus.DISPATCHED
                task.owner = worker_id
                task.updated_at = utc_now()
                data["tasks"][idx] = task.to_dict()
                claim.write_text(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "worker_id": worker_id,
                            "claimed_at": utc_now(),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                self._save_backlog_unlocked(data)
                return True
        return False

    def mark_running(self, task_id: str, pid: int) -> None:
        task = self.get_task(task_id)
        if task is None:
            return
        task.status = TaskStatus.RUNNING
        task.metadata["pid"] = pid
        task.touch()
        self.set_task(task)

    def mark_completed(self, task_id: str, artifacts: dict | None = None) -> None:
        task = self.get_task(task_id)
        if task is None:
            return
        task.status = TaskStatus.COMPLETED
        if artifacts:
            task.artifacts.update(artifacts)
        task.last_error = ""
        task.touch()
        self.set_task(task)
        snap = self.root / "completed" / f"{task_id}.json"
        snap.write_text(json.dumps(task.to_dict(), indent=2) + "\n", encoding="utf-8")
        claim = self.root / "claimed" / f"{task_id}.lock"
        if claim.exists():
            claim.unlink()

    def mark_merge_pending(self, task_id: str, artifacts: dict | None = None) -> None:
        task = self.get_task(task_id)
        if task is None:
            return
        task.status = TaskStatus.MERGE_PENDING
        if artifacts:
            task.artifacts.update(artifacts)
        task.touch()
        self.set_task(task)
        claim = self.root / "claimed" / f"{task_id}.lock"
        if claim.exists():
            claim.unlink()

    def mark_merged(self, task_id: str, artifacts: dict | None = None) -> None:
        task = self.get_task(task_id)
        if task is None:
            return
        task.status = TaskStatus.MERGED
        if artifacts:
            task.artifacts.update(artifacts)
        task.touch()
        self.set_task(task)
        snap = self.root / "completed" / f"{task_id}.json"
        snap.write_text(json.dumps(task.to_dict(), indent=2) + "\n", encoding="utf-8")
        claim = self.root / "claimed" / f"{task_id}.lock"
        if claim.exists():
            claim.unlink()

    def mark_failed(self, task_id: str, error: str, keep_claim: bool = False) -> None:
        task = self.get_task(task_id)
        if task is None:
            return
        task.retry_count += 1
        task.last_error = error
        task.status = TaskStatus.FAILED
        task.touch()
        self.set_task(task)
        snap = self.root / "failed" / f"{task_id}.json"
        snap.write_text(json.dumps(task.to_dict(), indent=2) + "\n", encoding="utf-8")
        if not keep_claim:
            claim = self.root / "claimed" / f"{task_id}.lock"
            if claim.exists():
                claim.unlink()

    def mark_blocked(self, task_id: str, reason: str) -> None:
        task = self.get_task(task_id)
        if task is None:
            return
        task.status = TaskStatus.BLOCKED
        task.last_error = reason
        task.touch()
        self.set_task(task)

    def mark_deferred(self, task_id: str, reason: str) -> None:
        task = self.get_task(task_id)
        if task is None:
            return
        task.status = TaskStatus.DEFERRED
        task.last_error = reason
        task.touch()
        self.set_task(task)

    def merge_pending_tasks(self) -> list[TaskRecord]:
        return [t for t in self.list_tasks() if t.status == TaskStatus.MERGE_PENDING]

    def enqueue_link(self, link: str) -> None:
        inbox = self.root / "inbox" / "links.jsonl"
        payload = {"link": link.strip(), "submitted_at": utc_now()}
        with inbox.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def read_inbox(self) -> list[dict]:
        inbox = self.root / "inbox" / "links.jsonl"
        out: list[dict] = []
        if not inbox.exists():
            return out
        for line in inbox.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def drain_inbox(self) -> list[dict]:
        with self._lock():
            inbox = self.root / "inbox" / "links.jsonl"
            processed = self.root / "inbox" / "processed.jsonl"
            out: list[dict] = []
            if inbox.exists():
                for line in inbox.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            if out:
                with processed.open("a", encoding="utf-8") as f:
                    for entry in out:
                        f.write(json.dumps(entry) + "\n")
            inbox.write_text("", encoding="utf-8")
            return out
