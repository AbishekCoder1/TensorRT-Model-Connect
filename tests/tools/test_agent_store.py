from __future__ import annotations

from pathlib import Path

from agent.schemas import ResourceProfile, TaskRecord, TaskStatus, TaskType
from agent.store import TaskStore


def _task(task_id: str, *, depends_on: list[str] | None = None) -> TaskRecord:
    return TaskRecord(
        id=task_id,
        model_id="Qwen/Qwen3-0.6B",
        hf_link="https://huggingface.co/Qwen/Qwen3-0.6B",
        task_type=TaskType.IMPLEMENT_PLUGIN,
        modality="decoder",
        runtime_strategy="decoder_kv_cache",
        depends_on=depends_on or [],
        resource_profile=ResourceProfile.CPU_LIGHT,
    )


def test_ready_tasks_follow_dependencies(tmp_path: Path):
    store = TaskStore(tmp_path / "agent")
    store.ensure_layout()

    first = _task("t1")
    second = _task("t2", depends_on=["t1"])
    store.upsert_tasks([first, second])

    ready_ids = [t.id for t in store.ready_tasks()]
    assert ready_ids == ["t1"]

    store.mark_completed("t1")
    ready_ids = [t.id for t in store.ready_tasks()]
    assert ready_ids == ["t2"]


def test_claim_and_complete_lifecycle(tmp_path: Path):
    store = TaskStore(tmp_path / "agent")
    store.ensure_layout()

    task = _task("claimable")
    store.upsert_tasks([task])
    store.ready_tasks()

    assert store.claim_task("claimable", "worker-1") is True
    claimed = store.get_task("claimable")
    assert claimed is not None
    assert claimed.status == TaskStatus.DISPATCHED
    assert (store.root / "claimed" / "claimable.lock").exists()

    store.mark_completed("claimable", artifacts={"ok": True})
    done = store.get_task("claimable")
    assert done is not None
    assert done.status == TaskStatus.COMPLETED
    assert done.artifacts["ok"] is True
    assert not (store.root / "claimed" / "claimable.lock").exists()


def test_drain_inbox_moves_entries_to_processed(tmp_path: Path):
    store = TaskStore(tmp_path / "agent")
    store.ensure_layout()

    store.enqueue_link("https://huggingface.co/Qwen/Qwen3-0.6B")
    store.enqueue_link("bert-base-uncased")

    drained = store.drain_inbox()
    assert len(drained) == 2
    assert (store.root / "inbox" / "links.jsonl").read_text(encoding="utf-8") == ""

    processed = (store.root / "inbox" / "processed.jsonl").read_text(encoding="utf-8")
    assert "Qwen/Qwen3-0.6B" in processed
    assert "bert-base-uncased" in processed
