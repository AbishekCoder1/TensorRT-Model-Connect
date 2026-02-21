from __future__ import annotations

from agent import orchestrator
from agent.schemas import TaskType


def test_build_task_plan_decoder_plugin_missing(monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "_fetch_metadata",
        lambda model_id: {"model_id": model_id, "model_type": "qwen", "pipeline_class": None},
    )
    monkeypatch.setattr(
        orchestrator,
        "_detect_support",
        lambda model_type, pipeline_class: {
            "plugin_exists": False,
            "diffusion_plugin_exists": False,
        },
    )
    monkeypatch.setattr(orchestrator, "_load_capability_matrix", lambda root=None: orchestrator.DEFAULT_CAPABILITY_MATRIX)

    tasks = orchestrator.build_task_plan("https://huggingface.co/Qwen/Qwen3-0.6B")
    types = [t.task_type for t in tasks]

    assert types[0] == TaskType.INTAKE_MODEL
    assert TaskType.IMPLEMENT_PLUGIN in types
    assert TaskType.ADD_MANIFEST in types
    assert TaskType.PARITY_VALIDATION in types
    assert TaskType.MERGE_READY_CHECK in types
    assert TaskType.IMPLEMENT_RUNTIME_DISPATCH not in types
    assert TaskType.IMPLEMENT_BUILDER not in types
    assert TaskType.FOUNDATION_BUILDER not in types


def test_build_task_plan_encoder_decoder_requires_builder_only(monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "_fetch_metadata",
        lambda model_id: {"model_id": model_id, "model_type": "t5", "pipeline_class": None},
    )
    monkeypatch.setattr(
        orchestrator,
        "_detect_support",
        lambda model_type, pipeline_class: {
            "plugin_exists": True,
            "diffusion_plugin_exists": False,
        },
    )
    monkeypatch.setattr(orchestrator, "_load_capability_matrix", lambda root=None: orchestrator.DEFAULT_CAPABILITY_MATRIX)

    tasks = orchestrator.build_task_plan("google-t5/t5-small")
    by_type = {t.task_type: t for t in tasks}

    assert TaskType.IMPLEMENT_PLUGIN not in by_type
    assert TaskType.IMPLEMENT_BUILDER in by_type
    assert TaskType.IMPLEMENT_RUNTIME_DISPATCH not in by_type

    builder = by_type[TaskType.IMPLEMENT_BUILDER]
    manifest = by_type[TaskType.ADD_MANIFEST]
    assert builder.depends_on == [tasks[0].id]
    assert manifest.depends_on == [builder.id]


def test_build_task_plan_creates_foundation_tasks_for_unready_strategy(monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "_fetch_metadata",
        lambda model_id: {
            "model_id": model_id,
            "model_type": "timesfm",
            "pipeline_class": None,
            "pipeline_tag": "time-series-forecasting",
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "_detect_support",
        lambda model_type, pipeline_class: {
            "plugin_exists": False,
            "diffusion_plugin_exists": False,
        },
    )
    matrix = orchestrator.DEFAULT_CAPABILITY_MATRIX | {
        "time_series_forecaster": {
            "builder_ready": False,
            "runtime_ready": False,
            "validator_ready": False,
            "canary_ready": False,
        }
    }
    monkeypatch.setattr(orchestrator, "_load_capability_matrix", lambda root=None: matrix)

    tasks = orchestrator.build_task_plan("amazon/chronos-2")
    by_id = {t.id: t for t in tasks}
    merge_id = orchestrator._foundation_task_id("time_series_forecaster", "merge")

    assert orchestrator._foundation_task_id("time_series_forecaster", "builder") in by_id
    assert orchestrator._foundation_task_id("time_series_forecaster", "runtime") in by_id
    assert orchestrator._foundation_task_id("time_series_forecaster", "validator") in by_id
    assert orchestrator._foundation_task_id("time_series_forecaster", "canary") in by_id
    assert merge_id in by_id

    plugin = next(t for t in tasks if t.task_type == TaskType.IMPLEMENT_PLUGIN)
    assert merge_id in plugin.depends_on
