from __future__ import annotations

from pathlib import Path

from agent.schemas import ResourceProfile, TaskRecord, TaskStatus, TaskType
from agent.scheduler import DEFAULT_RESOURCE_POLICY, ResourceSnapshot, Scheduler
from agent.store import TaskStore


class _DummyProc:
    def poll(self):
        return None


def _task(task_id: str, profile: ResourceProfile) -> TaskRecord:
    return TaskRecord(
        id=task_id,
        model_id="Qwen/Qwen3-0.6B",
        hf_link="https://huggingface.co/Qwen/Qwen3-0.6B",
        task_type=TaskType.IMPLEMENT_PLUGIN,
        modality="decoder",
        runtime_strategy="decoder_kv_cache",
        resource_profile=profile,
    )


def test_dispatch_does_not_claim_when_no_capacity(monkeypatch, tmp_path: Path):
    store = TaskStore(tmp_path / "agent")
    store.ensure_layout()
    store.upsert_tasks([_task("cpu-task", ResourceProfile.CPU_LIGHT)])

    scheduler = Scheduler(store, poll_interval=1)
    monkeypatch.setattr(scheduler, "_load_policy", lambda: DEFAULT_RESOURCE_POLICY)
    monkeypatch.setattr(
        scheduler,
        "snapshot",
        lambda backlog_exists, policy: ResourceSnapshot(cpu_slots=0, gpu_slots={}, gpu_stats={}),
    )

    launched = scheduler.dispatch_once()
    assert launched == 0

    task = store.get_task("cpu-task")
    assert task is not None
    assert task.status == TaskStatus.READY
    assert not (store.root / "claimed" / "cpu-task.lock").exists()


def test_dispatch_claims_and_launches_when_capacity_available(monkeypatch, tmp_path: Path):
    store = TaskStore(tmp_path / "agent")
    store.ensure_layout()
    store.upsert_tasks(
        [
            _task("cpu-task", ResourceProfile.CPU_LIGHT),
            _task("gpu-task", ResourceProfile.GPU_MEDIUM),
        ]
    )

    scheduler = Scheduler(store, poll_interval=1)
    monkeypatch.setattr(scheduler, "_load_policy", lambda: DEFAULT_RESOURCE_POLICY)
    monkeypatch.setattr(
        scheduler,
        "snapshot",
        lambda backlog_exists, policy: ResourceSnapshot(cpu_slots=2, gpu_slots={0: 1}, gpu_stats={}),
    )

    launched: list[tuple[str, int | None]] = []

    def fake_spawn(task: TaskRecord, gpu_id: int | None) -> None:
        launched.append((task.id, gpu_id))
        pid = 1000 + len(launched)
        store.mark_running(task.id, pid)
        scheduler.running[pid] = {
            "proc": _DummyProc(),
            "task_id": task.id,
            "profile": task.resource_profile.value,
            "gpu_id": gpu_id,
        }

    monkeypatch.setattr(scheduler, "_spawn_worker", fake_spawn)

    num = scheduler.dispatch_once()
    assert num == 2
    assert ("cpu-task", None) in launched
    assert ("gpu-task", 0) in launched


def test_dispatch_honors_max_workers_per_strategy(monkeypatch, tmp_path: Path):
    store = TaskStore(tmp_path / "agent")
    store.ensure_layout()
    t1 = _task("gpu-task-1", ResourceProfile.GPU_MEDIUM)
    t2 = _task("gpu-task-2", ResourceProfile.GPU_MEDIUM)
    t2.runtime_strategy = "decoder_kv_cache"
    store.upsert_tasks([t1, t2])

    scheduler = Scheduler(store, poll_interval=1)
    policy = {
        **DEFAULT_RESOURCE_POLICY,
        "max_workers_per_strategy": {"decoder_kv_cache": 1},
    }
    monkeypatch.setattr(scheduler, "_load_policy", lambda: policy)
    monkeypatch.setattr(
        scheduler,
        "snapshot",
        lambda backlog_exists, policy: ResourceSnapshot(cpu_slots=2, gpu_slots={0: 2}, gpu_stats={}),
    )

    launched: list[tuple[str, int | None]] = []

    def fake_spawn(task: TaskRecord, gpu_id: int | None) -> None:
        launched.append((task.id, gpu_id))
        pid = 2000 + len(launched)
        store.mark_running(task.id, pid)
        scheduler.running[pid] = {
            "proc": _DummyProc(),
            "task_id": task.id,
            "profile": task.resource_profile.value,
            "gpu_id": gpu_id,
            "runtime_strategy": task.runtime_strategy,
        }

    monkeypatch.setattr(scheduler, "_spawn_worker", fake_spawn)

    num = scheduler.dispatch_once()
    assert num == 1
    assert len(launched) == 1
