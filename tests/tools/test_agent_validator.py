from __future__ import annotations

import json

from agent.schemas import TaskRecord, TaskType
from agent.validator import ValidationResult, _task_validation_settings, validate_task


def _task() -> TaskRecord:
    return TaskRecord(
        id="validate:qwen:parity",
        model_id="Qwen/Qwen3-0.6B",
        hf_link="https://huggingface.co/Qwen/Qwen3-0.6B",
        task_type=TaskType.PARITY_VALIDATION,
        modality="decoder",
        runtime_strategy="decoder_kv_cache",
    )


def test_task_validation_settings_merge(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    cfg_dir = root / "agent" / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "validation_profiles.json").write_text(
        json.dumps(
            {
                "defaults": {"atol": 0.01, "prompt": "default", "max_new_tokens": 10, "cache_len": 64},
                "by_modality": {"decoder": {"max_new_tokens": 22}},
                "by_runtime_strategy": {"decoder_kv_cache": {"atol": 0.001}},
            }
        ),
        encoding="utf-8",
    )

    import agent.validator as validator

    monkeypatch.setattr(validator, "_project_root", lambda: root)
    task = _task()
    task.metadata["validation"] = {"prompt": "task prompt", "cache_len": 128}

    settings = _task_validation_settings(task, root)
    assert settings["atol"] == 0.001
    assert settings["max_new_tokens"] == 22
    assert settings["prompt"] == "task prompt"
    assert settings["cache_len"] == 128


def test_validate_task_passes_settings_to_validate_model(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    cfg_dir = root / "agent" / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "validation_profiles.json").write_text("{}", encoding="utf-8")

    import agent.validator as validator

    captured: dict[str, object] = {}

    def fake_validate_model(model_id, runtime_strategy, **kwargs):
        captured["model_id"] = model_id
        captured["runtime_strategy"] = runtime_strategy
        captured.update(kwargs)
        return ValidationResult(model_id=model_id, runtime_strategy=runtime_strategy, checks=[])

    monkeypatch.setattr(validator, "_project_root", lambda: root)
    monkeypatch.setattr(validator, "validate_model", fake_validate_model)

    task = _task()
    task.metadata.update({"prompt": "abc", "max_new_tokens": 9, "logit_atol": 0.123, "cache_len": 77})
    result = validate_task(task)

    assert result.passed is True
    assert captured["model_id"] == task.model_id
    assert captured["runtime_strategy"] == task.runtime_strategy
    assert captured["prompt"] == "abc"
    assert captured["max_new_tokens"] == 9
    assert captured["atol"] == 0.123
    assert captured["cache_len"] == 77


def test_validate_model_routes_time_series_runtime(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    (root / "agent" / "config").mkdir(parents=True, exist_ok=True)

    import agent.validator as validator

    called: dict[str, object] = {}

    def fake_time_series(model_id, runtime_strategy, cwd, prompt, max_new_tokens, cache_len):
        called["model_id"] = model_id
        called["runtime_strategy"] = runtime_strategy
        called["cwd"] = cwd
        return ValidationResult(model_id=model_id, runtime_strategy=runtime_strategy, checks=[])

    monkeypatch.setattr(validator, "_project_root", lambda: root)
    monkeypatch.setattr(validator, "_time_series_validation", fake_time_series)

    result = validator.validate_model(
        model_id="amazon/chronos-2",
        runtime_strategy="time_series_forecaster",
        prompt="Forecast next 24 timesteps",
        max_new_tokens=0,
        cache_len=128,
    )

    assert result.passed is True
    assert called["model_id"] == "amazon/chronos-2"
    assert called["runtime_strategy"] == "time_series_forecaster"
