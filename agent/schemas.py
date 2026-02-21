from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class TaskType(str, Enum):
    INTAKE_MODEL = "intake_model"
    FOUNDATION_BUILDER = "foundation_builder"
    FOUNDATION_RUNTIME = "foundation_runtime"
    FOUNDATION_VALIDATOR = "foundation_validator"
    FOUNDATION_CANARY = "foundation_canary"
    IMPLEMENT_PLUGIN = "implement_plugin"
    IMPLEMENT_BUILDER = "implement_builder"
    IMPLEMENT_RUNTIME_DISPATCH = "implement_runtime_dispatch"
    ADD_MANIFEST = "add_manifest"
    PARITY_VALIDATION = "parity_validation"
    MERGE_READY_CHECK = "merge_ready_check"


class TaskStatus(str, Enum):
    NEW = "new"
    READY = "ready"
    DISPATCHED = "dispatched"
    RUNNING = "running"
    VALIDATING = "validating"
    COMPLETED = "completed"
    MERGE_PENDING = "merge_pending"
    MERGED = "merged"
    FAILED = "failed"
    BLOCKED = "blocked"
    DEFERRED = "deferred"


class ResourceProfile(str, Enum):
    CPU_LIGHT = "cpu_light"
    CPU_HEAVY = "cpu_heavy"
    CPU_TIME_SERIES = "cpu_time_series"
    CPU_RL_INFERENCE = "cpu_rl_inference"
    GPU_LIGHT = "gpu_light"
    GPU_MEDIUM = "gpu_medium"
    GPU_HEAVY = "gpu_heavy"
    GPU_DIFFUSION = "gpu_diffusion"
    GPU_VIDEO_ENCODER = "gpu_video_encoder"
    GPU_VIDEO_DIFFUSION = "gpu_video_diffusion"


TERMINAL_STATUSES = {
    TaskStatus.COMPLETED,
    TaskStatus.MERGED,
    TaskStatus.FAILED,
    TaskStatus.BLOCKED,
    TaskStatus.DEFERRED,
}


@dataclass
class TaskRecord:
    id: str
    model_id: str
    hf_link: str
    task_type: TaskType
    modality: str
    runtime_strategy: str
    priority: int = 50
    status: TaskStatus = TaskStatus.NEW
    depends_on: list[str] = field(default_factory=list)
    branch: str = ""
    owner: str = ""
    retry_count: int = 0
    max_retries: int = 3
    resource_profile: ResourceProfile = ResourceProfile.CPU_LIGHT
    metadata: dict[str, Any] = field(default_factory=dict)
    acceptance: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    last_error: str = ""

    def touch(self) -> None:
        self.updated_at = utc_now()

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model_id": self.model_id,
            "hf_link": self.hf_link,
            "task_type": self.task_type.value,
            "modality": self.modality,
            "runtime_strategy": self.runtime_strategy,
            "priority": self.priority,
            "status": self.status.value,
            "depends_on": self.depends_on,
            "branch": self.branch,
            "owner": self.owner,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "resource_profile": self.resource_profile.value,
            "metadata": self.metadata,
            "acceptance": self.acceptance,
            "artifacts": self.artifacts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_error": self.last_error,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> TaskRecord:
        return TaskRecord(
            id=data["id"],
            model_id=data.get("model_id", ""),
            hf_link=data.get("hf_link", ""),
            task_type=TaskType(data["task_type"]),
            modality=data.get("modality", "unknown"),
            runtime_strategy=data.get("runtime_strategy", ""),
            priority=int(data.get("priority", 50)),
            status=TaskStatus(data.get("status", TaskStatus.NEW.value)),
            depends_on=list(data.get("depends_on", [])),
            branch=data.get("branch", ""),
            owner=data.get("owner", ""),
            retry_count=int(data.get("retry_count", 0)),
            max_retries=int(data.get("max_retries", 3)),
            resource_profile=ResourceProfile(
                data.get("resource_profile", ResourceProfile.CPU_LIGHT.value)
            ),
            metadata=dict(data.get("metadata", {})),
            acceptance=dict(data.get("acceptance", {})),
            artifacts=dict(data.get("artifacts", {})),
            created_at=data.get("created_at", utc_now()),
            updated_at=data.get("updated_at", utc_now()),
            last_error=data.get("last_error", ""),
        )
