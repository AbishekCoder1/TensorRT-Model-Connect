from __future__ import annotations

import json
from pathlib import Path

from agent import merge_manager
from agent.schemas import TaskRecord, TaskType


def _merge_task(run_tier_b: bool) -> TaskRecord:
    return TaskRecord(
        id="merge:qwen:gate",
        model_id="Qwen/Qwen3-0.6B",
        hf_link="https://huggingface.co/Qwen/Qwen3-0.6B",
        task_type=TaskType.MERGE_READY_CHECK,
        modality="decoder",
        runtime_strategy="decoder_kv_cache",
        metadata={"run_tier_b": run_tier_b},
    )


def test_load_canary_config_legacy_shape(tmp_path: Path):
    root = tmp_path / "repo"
    cfg_dir = root / "agent" / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "canaries.json").write_text(
        json.dumps({"global": ["echo legacy"], "by_modality": {"decoder": ["echo modal"]}}),
        encoding="utf-8",
    )

    cfg = merge_manager._load_canary_config(root)
    assert "tier_a" in cfg
    assert "tier_b" in cfg
    assert cfg["tier_a"]["global"] == ["echo legacy"]
    assert cfg["tier_a"]["by_modality"]["decoder"] == ["echo modal"]


def test_run_gate_respects_tier_b_flag(monkeypatch, tmp_path: Path):
    root = tmp_path / "repo"
    cfg_dir = root / "agent" / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "canaries.json").write_text(
        json.dumps(
            {
                "tier_a": {"global": ["echo tier-a"], "by_modality": {}, "by_runtime_strategy": {}},
                "tier_b": {"global": ["echo tier-b"], "by_modality": {}, "by_runtime_strategy": {}},
            }
        ),
        encoding="utf-8",
    )

    seen: list[str] = []

    def fake_run(cmd, cwd, timeout=1800):
        seen.append(" ".join(cmd))
        return True, "", ""

    monkeypatch.setattr(merge_manager, "_run", fake_run)

    ok_a, _ = merge_manager._run_gate(_merge_task(run_tier_b=False), root)
    assert ok_a is True
    assert len(seen) == 1

    seen.clear()
    ok_b, _ = merge_manager._run_gate(_merge_task(run_tier_b=True), root)
    assert ok_b is True
    assert len(seen) == 2
