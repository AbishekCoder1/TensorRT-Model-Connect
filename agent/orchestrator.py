from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .schemas import ResourceProfile, TaskRecord, TaskType
from .store import TaskStore
from .utils import normalize_hf_model_id, slugify

CAPABILITY_FLAGS = (
    "builder_ready",
    "runtime_ready",
    "validator_ready",
    "canary_ready",
)

DEFAULT_CAPABILITY_MATRIX: dict[str, dict[str, bool]] = {
    "decoder_kv_cache": {
        "builder_ready": True,
        "runtime_ready": True,
        "validator_ready": True,
        "canary_ready": True,
    },
    "decoder_moe": {
        "builder_ready": True,
        "runtime_ready": True,
        "validator_ready": True,
        "canary_ready": True,
    },
    "ssm_recurrent": {
        "builder_ready": True,
        "runtime_ready": True,
        "validator_ready": True,
        "canary_ready": True,
    },
    "encoder_only": {
        "builder_ready": True,
        "runtime_ready": True,
        "validator_ready": True,
        "canary_ready": True,
    },
    "encoder_decoder": {
        "builder_ready": True,
        "runtime_ready": True,
        "validator_ready": True,
        "canary_ready": True,
    },
    "vision_encoder": {
        "builder_ready": True,
        "runtime_ready": True,
        "validator_ready": True,
        "canary_ready": True,
    },
    "audio_encoder": {
        "builder_ready": True,
        "runtime_ready": True,
        "validator_ready": True,
        "canary_ready": True,
    },
    "vision_language": {
        "builder_ready": True,
        "runtime_ready": True,
        "validator_ready": True,
        "canary_ready": True,
    },
    "diffusion": {
        "builder_ready": True,
        "runtime_ready": True,
        "validator_ready": True,
        "canary_ready": True,
    },
    "video_diffusion": {
        "builder_ready": False,
        "runtime_ready": False,
        "validator_ready": False,
        "canary_ready": False,
    },
    "video_encoder": {
        "builder_ready": False,
        "runtime_ready": False,
        "validator_ready": False,
        "canary_ready": False,
    },
    "time_series_encoder": {
        "builder_ready": False,
        "runtime_ready": False,
        "validator_ready": False,
        "canary_ready": False,
    },
    "time_series_forecaster": {
        "builder_ready": False,
        "runtime_ready": False,
        "validator_ready": False,
        "canary_ready": False,
    },
    "rl_policy_inference": {
        "builder_ready": False,
        "runtime_ready": False,
        "validator_ready": False,
        "canary_ready": False,
    },
    "rl_value_inference": {
        "builder_ready": False,
        "runtime_ready": False,
        "validator_ready": False,
        "canary_ready": False,
    },
}

SUPPORTED_RUNTIME_STRATEGIES = set(DEFAULT_CAPABILITY_MATRIX.keys())

MODEL_BUILDER_RUNTIMES = {
    "encoder_decoder",
    "vision_encoder",
    "audio_encoder",
    "video_encoder",
    "video_diffusion",
    "time_series_encoder",
    "time_series_forecaster",
    "rl_policy_inference",
    "rl_value_inference",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_capability_matrix(root: Path | None = None) -> dict[str, dict[str, bool]]:
    merged = {
        runtime: {
            flag: bool(values.get(flag, False))
            for flag in CAPABILITY_FLAGS
        }
        for runtime, values in DEFAULT_CAPABILITY_MATRIX.items()
    }

    cfg_root = root or _project_root()
    path = cfg_root / "agent" / "config" / "capability_matrix.json"
    if not path.exists():
        return merged

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return merged

    runtime_map = raw.get("runtime_strategies", raw)
    if not isinstance(runtime_map, dict):
        return merged

    for runtime, values in runtime_map.items():
        if not isinstance(values, dict):
            continue
        target = merged.setdefault(runtime, {flag: False for flag in CAPABILITY_FLAGS})
        for flag in CAPABILITY_FLAGS:
            if flag in values:
                target[flag] = bool(values[flag])
    return merged


def _capability_entry(runtime_strategy: str, matrix: dict[str, dict[str, bool]]) -> dict[str, bool]:
    defaults = {flag: False for flag in CAPABILITY_FLAGS}
    values = matrix.get(runtime_strategy, {})
    for flag in CAPABILITY_FLAGS:
        defaults[flag] = bool(values.get(flag, False))
    return defaults


def _guess_runtime_strategy(
    model_type: str,
    pipeline_class: str | None,
    pipeline_tag: str | None,
) -> str:
    mt = (model_type or "").lower().replace("-", "_")
    pt = (pipeline_tag or "").lower()
    pc = (pipeline_class or "").lower()

    if pc:
        if "video" in pc:
            return "video_diffusion"
        return "diffusion"

    if pt in {"text-to-video", "image-to-video", "video-to-video"}:
        return "video_diffusion"
    if pt in {"video-classification", "video-feature-extraction"}:
        return "video_encoder"
    if pt in {"time-series-forecasting"}:
        return "time_series_forecaster"
    if pt in {"reinforcement-learning"}:
        return "rl_policy_inference"
    if pt in {"feature-extraction"} and mt in {"timesfm", "chronos", "patchtst"}:
        return "time_series_encoder"

    if mt in {"bert", "roberta", "distilbert", "mpnet", "xlm_roberta"}:
        return "encoder_only"
    if mt in {"t5", "mt5", "bart", "mbart", "marian", "m2m_100"}:
        return "encoder_decoder"
    if mt in {"wav2vec2", "wav2vec2_conformer", "clap", "whisper"}:
        return "audio_encoder"
    if "vl" in mt or mt in {"llava", "internvl", "idefics2", "idefics3"}:
        return "vision_language"
    if mt in {"vit", "dinov2", "siglip", "clip", "segformer", "superpoint"}:
        return "vision_encoder"
    if mt in {"videomae", "timesformer", "xclip", "vivit"}:
        return "video_encoder"
    if mt in {"time_series_transformer", "informer", "autoformer", "patchtst", "timesfm", "chronos"}:
        return "time_series_forecaster"
    if mt in {"decision_transformer", "trajectory_transformer"}:
        return "rl_policy_inference"
    if mt in {"mamba"}:
        return "ssm_recurrent"
    return "decoder_kv_cache"


def _guess_modality(runtime_strategy: str) -> str:
    mapping = {
        "decoder_kv_cache": "decoder",
        "decoder_moe": "decoder",
        "ssm_recurrent": "decoder",
        "encoder_only": "encoder",
        "encoder_decoder": "encoder_decoder",
        "vision_language": "vision_language",
        "vision_encoder": "vision_encoder",
        "audio_encoder": "audio_encoder",
        "diffusion": "diffusion",
        "video_diffusion": "video_diffusion",
        "video_encoder": "video_encoder",
        "time_series_encoder": "time_series",
        "time_series_forecaster": "time_series",
        "rl_policy_inference": "rl_inference",
        "rl_value_inference": "rl_inference",
    }
    return mapping.get(runtime_strategy, "unknown")


def _resource_profile(task_type: TaskType, runtime_strategy: str) -> ResourceProfile:
    if task_type == TaskType.PARITY_VALIDATION:
        if runtime_strategy == "diffusion":
            return ResourceProfile.GPU_DIFFUSION
        if runtime_strategy == "video_diffusion":
            return ResourceProfile.GPU_VIDEO_DIFFUSION
        if runtime_strategy == "video_encoder":
            return ResourceProfile.GPU_VIDEO_ENCODER
        if runtime_strategy in {"time_series_encoder", "time_series_forecaster"}:
            return ResourceProfile.CPU_TIME_SERIES
        if runtime_strategy in {"rl_policy_inference", "rl_value_inference"}:
            return ResourceProfile.CPU_RL_INFERENCE
        if runtime_strategy in {"vision_language", "vision_encoder", "audio_encoder", "encoder_decoder"}:
            return ResourceProfile.GPU_HEAVY
        return ResourceProfile.GPU_MEDIUM
    if task_type == TaskType.IMPLEMENT_RUNTIME_DISPATCH:
        return ResourceProfile.CPU_HEAVY
    if task_type in {
        TaskType.IMPLEMENT_BUILDER,
        TaskType.FOUNDATION_BUILDER,
        TaskType.FOUNDATION_RUNTIME,
        TaskType.FOUNDATION_VALIDATOR,
        TaskType.FOUNDATION_CANARY,
    }:
        return ResourceProfile.CPU_HEAVY
    return ResourceProfile.CPU_LIGHT


def _fetch_metadata(model_id: str) -> dict[str, Any]:
    out: dict[str, Any] = {"model_id": model_id, "model_type": "", "pipeline_class": None}

    local_path = Path(model_id)
    if local_path.exists() and local_path.is_dir():
        cfg = local_path / "config.json"
        if cfg.exists():
            d = json.loads(cfg.read_text(encoding="utf-8"))
            out["model_type"] = d.get("model_type", "")
        idx = local_path / "model_index.json"
        if idx.exists():
            d = json.loads(idx.read_text(encoding="utf-8"))
            out["pipeline_class"] = d.get("_class_name")
        return out

    try:
        from huggingface_hub import hf_hub_download, model_info

        try:
            cfg_path = hf_hub_download(model_id, "config.json")
            d = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
            out["model_type"] = d.get("model_type", "")
        except Exception:
            pass

        try:
            idx_path = hf_hub_download(model_id, "model_index.json")
            d = json.loads(Path(idx_path).read_text(encoding="utf-8"))
            out["pipeline_class"] = d.get("_class_name")
        except Exception:
            pass

        try:
            info = model_info(model_id)
            out["pipeline_tag"] = getattr(info, "pipeline_tag", "")
            out["private"] = bool(getattr(info, "private", False))
            out["gated"] = bool(getattr(info, "gated", False))
        except Exception:
            pass
    except Exception:
        pass

    return out


def _detect_support(model_type: str, pipeline_class: str | None) -> dict[str, bool]:
    plugin_exists = False
    diffusion_plugin_exists = False

    try:
        from trtf_build.families import find_plugin, find_diffusion_plugin

        if model_type:
            plugin_exists = find_plugin(model_type) is not None
        if pipeline_class:
            diffusion_plugin_exists = find_diffusion_plugin(pipeline_class) is not None
    except Exception:
        # conservative default if import stack not ready
        plugin_exists = False
        diffusion_plugin_exists = False

    return {
        "plugin_exists": plugin_exists,
        "diffusion_plugin_exists": diffusion_plugin_exists,
    }


def _task_id(prefix: str, model_slug: str, suffix: str) -> str:
    return f"{prefix}:{model_slug}:{suffix}"


def _foundation_task_id(runtime_strategy: str, stage: str) -> str:
    return f"foundation:{slugify(runtime_strategy)}:{stage}"


def _build_foundation_tasks(
    *,
    runtime_strategy: str,
    modality: str,
    trigger_link: str,
    matrix: dict[str, dict[str, bool]],
) -> tuple[list[TaskRecord], str | None]:
    cap = _capability_entry(runtime_strategy, matrix)
    missing_flags = [flag for flag in CAPABILITY_FLAGS if not cap[flag]]
    if not missing_flags:
        return [], None

    branch = f"agent/foundation/{slugify(runtime_strategy)}"
    foundation_model_id = f"foundation/{runtime_strategy}"
    tasks: list[TaskRecord] = []
    prev: str | None = None

    stage_map: dict[str, tuple[TaskType, str, int, str]] = {
        "builder_ready": (
            TaskType.FOUNDATION_BUILDER,
            "builder",
            12,
            "foundation builder path added for runtime strategy",
        ),
        "runtime_ready": (
            TaskType.FOUNDATION_RUNTIME,
            "runtime",
            13,
            "foundation runtime dispatch path added for strategy",
        ),
        "validator_ready": (
            TaskType.FOUNDATION_VALIDATOR,
            "validator",
            14,
            "strict validator path wired for strategy",
        ),
        "canary_ready": (
            TaskType.FOUNDATION_CANARY,
            "canary",
            15,
            "canary profile added for strategy",
        ),
    }

    for flag in CAPABILITY_FLAGS:
        if flag not in missing_flags:
            continue
        task_type, suffix, priority, must_text = stage_map[flag]
        task = TaskRecord(
            id=_foundation_task_id(runtime_strategy, suffix),
            model_id=foundation_model_id,
            hf_link=f"foundation://{runtime_strategy}",
            task_type=task_type,
            modality=modality,
            runtime_strategy=runtime_strategy,
            priority=priority,
            depends_on=[prev] if prev else [],
            branch=branch,
            resource_profile=_resource_profile(task_type, runtime_strategy),
            metadata={
                "foundation": True,
                "trigger_link": trigger_link,
                "capability_flag": flag,
                "runtime_strategy": runtime_strategy,
                "capability_requirements": cap,
                "validator_profile": runtime_strategy,
                "fixture_set": modality,
            },
            acceptance={"must": [must_text]},
        )
        tasks.append(task)
        prev = task.id

    merge = TaskRecord(
        id=_foundation_task_id(runtime_strategy, "merge"),
        model_id=foundation_model_id,
        hf_link=f"foundation://{runtime_strategy}",
        task_type=TaskType.MERGE_READY_CHECK,
        modality=modality,
        runtime_strategy=runtime_strategy,
        priority=16,
        depends_on=[prev] if prev else [],
        branch=branch,
        resource_profile=_resource_profile(TaskType.MERGE_READY_CHECK, runtime_strategy),
        metadata={
            "foundation": True,
            "trigger_link": trigger_link,
            "gate_scope": "strategy_foundation",
            "run_tier_b": True,
            "runtime_strategy": runtime_strategy,
            "capability_requirements": cap,
            "validator_profile": runtime_strategy,
            "fixture_set": modality,
        },
        acceptance={
            "must": [
                "foundation branch merged to master",
                "tier-a and tier-b canaries pass for strategy",
            ]
        },
    )
    tasks.append(merge)
    return tasks, merge.id


def build_task_plan(model_link: str) -> list[TaskRecord]:
    model_id = normalize_hf_model_id(model_link)
    model_slug = slugify(model_id)
    model_branch = f"agent/{model_slug}"
    meta = _fetch_metadata(model_id)
    capability_matrix = _load_capability_matrix(_project_root())
    runtime = _guess_runtime_strategy(
        meta.get("model_type", ""),
        meta.get("pipeline_class"),
        meta.get("pipeline_tag", ""),
    )
    modality = _guess_modality(runtime)
    support = _detect_support(meta.get("model_type", ""), meta.get("pipeline_class"))
    capability = _capability_entry(runtime, capability_matrix)
    foundation_tasks, foundation_merge_dep = _build_foundation_tasks(
        runtime_strategy=runtime,
        modality=modality,
        trigger_link=model_link,
        matrix=capability_matrix,
    )

    tasks: list[TaskRecord] = []
    intake_id = _task_id("intake", model_slug, "model")

    intake = TaskRecord(
        id=intake_id,
        model_id=model_id,
        hf_link=model_link,
        task_type=TaskType.INTAKE_MODEL,
        modality=modality,
        runtime_strategy=runtime,
        priority=10,
        branch=model_branch,
        resource_profile=_resource_profile(TaskType.INTAKE_MODEL, runtime),
        metadata={
            "model_type": meta.get("model_type", ""),
            "capability_requirements": capability,
            "validator_profile": runtime,
            "fixture_set": modality,
            **meta,
            **support,
        },
        acceptance={"goal": "Model metadata resolved and task DAG created"},
    )
    tasks.append(intake)
    tasks.extend(foundation_tasks)

    tail_deps = [intake.id]
    if foundation_merge_dep:
        tail_deps.append(foundation_merge_dep)

    needs_plugin = not (support["plugin_exists"] or support["diffusion_plugin_exists"])
    if needs_plugin:
        task = TaskRecord(
            id=_task_id("impl", model_slug, "plugin"),
            model_id=model_id,
            hf_link=model_link,
            task_type=TaskType.IMPLEMENT_PLUGIN,
            modality=modality,
            runtime_strategy=runtime,
            priority=20,
            depends_on=tail_deps,
            branch=model_branch,
            resource_profile=_resource_profile(TaskType.IMPLEMENT_PLUGIN, runtime),
            metadata={
                "model_type": meta.get("model_type", ""),
                "capability_requirements": capability,
                "validator_profile": runtime,
                "fixture_set": modality,
                **meta,
            },
            acceptance={
                "must": [
                    "family plugin added",
                    "plugin auto-discovered",
                    "family unit tests updated",
                ]
            },
        )
        tasks.append(task)
        tail_deps = [task.id]

    if runtime not in SUPPORTED_RUNTIME_STRATEGIES:
        rt_task = TaskRecord(
            id=_task_id("infra", model_slug, "runtime"),
            model_id=model_id,
            hf_link=model_link,
            task_type=TaskType.IMPLEMENT_RUNTIME_DISPATCH,
            modality=modality,
            runtime_strategy=runtime,
            priority=25,
            depends_on=tail_deps,
            branch=model_branch,
            resource_profile=_resource_profile(TaskType.IMPLEMENT_RUNTIME_DISPATCH, runtime),
            metadata={
                "runtime_strategy": runtime,
                "unknown_runtime_strategy": True,
                "capability_requirements": capability,
                "validator_profile": runtime,
                "fixture_set": modality,
                **meta,
            },
            acceptance={
                "must": [
                    "runtime_strategy parsed",
                    "c++ dispatch branch added",
                    "strategy integration tests pass",
                ]
            },
        )
        tasks.append(rt_task)
        tail_deps = [rt_task.id]

    if runtime in MODEL_BUILDER_RUNTIMES:
        builder_task = TaskRecord(
            id=_task_id("impl", model_slug, "builder"),
            model_id=model_id,
            hf_link=model_link,
            task_type=TaskType.IMPLEMENT_BUILDER,
            modality=modality,
            runtime_strategy=runtime,
            priority=30,
            depends_on=tail_deps,
            branch=model_branch,
            resource_profile=_resource_profile(TaskType.IMPLEMENT_BUILDER, runtime),
            metadata={
                "runtime_strategy": runtime,
                "capability_requirements": capability,
                "validator_profile": runtime,
                "fixture_set": modality,
                **meta,
            },
            acceptance={
                "must": [
                    "builder implemented",
                    "runtime strategy wired",
                    "unit tests for builder/runtime pass",
                ]
            },
        )
        tasks.append(builder_task)
        tail_deps = [builder_task.id]

    manifest_task = TaskRecord(
        id=_task_id("impl", model_slug, "manifest"),
        model_id=model_id,
        hf_link=model_link,
        task_type=TaskType.ADD_MANIFEST,
        modality=modality,
        runtime_strategy=runtime,
        priority=40,
        depends_on=tail_deps,
        branch=model_branch,
        resource_profile=_resource_profile(TaskType.ADD_MANIFEST, runtime),
        metadata={
            "runtime_strategy": runtime,
            "capability_requirements": capability,
            "validator_profile": runtime,
            "fixture_set": modality,
            **meta,
        },
        acceptance={
            "must": [
                "tests/e2e/models entry exists",
                "model-specific tolerances declared",
            ]
        },
    )
    tasks.append(manifest_task)

    validation_task = TaskRecord(
        id=_task_id("validate", model_slug, "parity"),
        model_id=model_id,
        hf_link=model_link,
        task_type=TaskType.PARITY_VALIDATION,
        modality=modality,
        runtime_strategy=runtime,
        priority=50,
        depends_on=[manifest_task.id],
        branch=model_branch,
        resource_profile=_resource_profile(TaskType.PARITY_VALIDATION, runtime),
        metadata={
            "runtime_strategy": runtime,
            "capability_requirements": capability,
            "validator_profile": runtime,
            "fixture_set": modality,
            **meta,
        },
        acceptance={
            "strictness": "exact_tokens_plus_tight_numeric",
            "must": [
                "python_build_and_cpp_run parity with HF",
                "modality-specific diff checks pass",
            ],
        },
    )
    tasks.append(validation_task)

    merge_task = TaskRecord(
        id=_task_id("merge", model_slug, "gate"),
        model_id=model_id,
        hf_link=model_link,
        task_type=TaskType.MERGE_READY_CHECK,
        modality=modality,
        runtime_strategy=runtime,
        priority=60,
        depends_on=[validation_task.id],
        branch=model_branch,
        resource_profile=_resource_profile(TaskType.MERGE_READY_CHECK, runtime),
        metadata={
            "runtime_strategy": runtime,
            "gate_scope": "model",
            "run_tier_b": False,
            "capability_requirements": capability,
            "validator_profile": runtime,
            "fixture_set": modality,
            **meta,
        },
        acceptance={
            "must": [
                "tier-A targeted gate passes",
                "tier-B canary gate passes when run_tier_b is enabled",
                "master stays green",
            ],
            "gate_budget_minutes": 20,
        },
    )
    tasks.append(merge_task)

    return tasks


def process_links(store: TaskStore, links: list[str]) -> int:
    created = 0
    existing_ids = {t.id for t in store.list_tasks()}
    for raw in links:
        raw = raw.strip()
        if not raw:
            continue
        plan = build_task_plan(raw)
        new = [t for t in plan if t.id not in existing_ids]
        if not new:
            continue
        store.upsert_tasks(new)
        # Mark intake as completed immediately (decomposition already happened).
        if plan[0].id in {t.id for t in new}:
            store.mark_completed(plan[0].id, artifacts={"decomposed": True})
        existing_ids.update(t.id for t in new)
        created += len(new)
    return created


def process_inbox_once(store: TaskStore) -> int:
    links = [entry.get("link", "") for entry in store.drain_inbox()]
    # Deduplicate while preserving order.
    seen = set()
    ordered: list[str] = []
    for link in links:
        normalized = normalize_hf_model_id(link)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(link)
    return process_links(store, ordered)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build task DAGs from HF links")
    parser.add_argument("--store", default="agent", help="Task store directory")
    parser.add_argument("--link", action="append", default=[], help="HF link or model ID")
    parser.add_argument(
        "--links-file",
        default="",
        help="Optional file with one HF link/model ID per line",
    )
    parser.add_argument("--inbox", action="store_true", help="Process inbox links.jsonl")
    args = parser.parse_args()

    store = TaskStore(args.store)
    store.ensure_layout()

    created = 0
    if args.link:
        created += process_links(store, args.link)
    if args.links_file:
        links = [
            line.strip()
            for line in Path(args.links_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        created += process_links(store, links)
    if args.inbox:
        created += process_inbox_once(store)

    print(f"created_tasks={created}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
