from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import ResourceProfile, TaskRecord
from .store import TaskStore


CPU_PROFILES = {
    ResourceProfile.CPU_LIGHT,
    ResourceProfile.CPU_HEAVY,
    ResourceProfile.CPU_TIME_SERIES,
    ResourceProfile.CPU_RL_INFERENCE,
}

DEFAULT_RESOURCE_POLICY: dict[str, Any] = {
    "cpu": {
        "core_fraction": 0.5,
        "ram_reserve_gb": 256.0,
        "ram_per_slot_gb": 16.0,
        "min_slots": 1,
        "max_slots": 48,
        "backlog_min_slots": 8,
    },
    "gpu": {
        "slot_thresholds": [
            {"min_free_gb": 220.0, "slots": 3},
            {"min_free_gb": 160.0, "slots": 2},
            {"min_free_gb": 100.0, "slots": 1},
        ],
        "per_profile_max_per_gpu": {
            "gpu_diffusion": 1,
            "gpu_video_diffusion": 1,
        },
    },
    "strategy_mutexes": [
        ["diffusion", "video_diffusion"],
    ],
    "max_workers_per_strategy": {
        "diffusion": 4,
        "video_diffusion": 2,
    },
}


@dataclass
class GpuStats:
    index: int
    total_gb: float
    free_gb: float
    util: float


@dataclass
class ResourceSnapshot:
    cpu_slots: int
    gpu_slots: dict[int, int]
    gpu_stats: dict[int, GpuStats]


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _parse_thresholds(raw: str) -> list[dict[str, float | int]]:
    out: list[dict[str, float | int]] = []
    for part in raw.split(","):
        item = part.strip()
        if not item or ":" not in item:
            continue
        min_free, slots = item.split(":", 1)
        try:
            out.append({"min_free_gb": float(min_free.strip()), "slots": int(slots.strip())})
        except Exception:
            continue
    return sorted(out, key=lambda x: float(x["min_free_gb"]), reverse=True)


def _parse_strategy_caps(raw: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in raw.split(","):
        item = part.strip()
        if not item or "=" not in item:
            continue
        key, val = item.split("=", 1)
        try:
            out[key.strip()] = int(val.strip())
        except Exception:
            continue
    return out


def _parse_strategy_mutexes(raw: str) -> list[list[str]]:
    groups: list[list[str]] = []
    for block in raw.split(";"):
        vals = [v.strip() for v in block.split("|") if v.strip()]
        if len(vals) >= 2:
            groups.append(vals)
    return groups


class Scheduler:
    def __init__(self, store: TaskStore, poll_interval: int = 5) -> None:
        self.store = store
        self.poll_interval = max(1, poll_interval)
        self.host = socket.gethostname()
        self.worker_id = f"scheduler:{self.host}:{os.getpid()}"
        self.running: dict[int, dict] = {}

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def _load_policy(self) -> dict[str, Any]:
        policy = json.loads(json.dumps(DEFAULT_RESOURCE_POLICY))
        cfg_path = Path(self.store.root) / "config" / "resource_policy.json"
        if cfg_path.exists():
            try:
                raw = json.loads(cfg_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    policy = _deep_update(policy, raw)
            except Exception:
                pass

        if os.environ.get("AGENT_CPU_SLOT_CAP", "").strip():
            try:
                policy["cpu"]["max_slots"] = int(os.environ["AGENT_CPU_SLOT_CAP"])
            except Exception:
                pass
        if os.environ.get("AGENT_RAM_RESERVE_GB", "").strip():
            try:
                policy["cpu"]["ram_reserve_gb"] = float(os.environ["AGENT_RAM_RESERVE_GB"])
            except Exception:
                pass
        if os.environ.get("AGENT_GPU_SLOT_THRESHOLDS", "").strip():
            parsed = _parse_thresholds(os.environ["AGENT_GPU_SLOT_THRESHOLDS"])
            if parsed:
                policy["gpu"]["slot_thresholds"] = parsed
        if os.environ.get("AGENT_MAX_WORKERS_PER_STRATEGY", "").strip():
            parsed = _parse_strategy_caps(os.environ["AGENT_MAX_WORKERS_PER_STRATEGY"])
            if parsed:
                policy["max_workers_per_strategy"] = _deep_update(
                    policy.get("max_workers_per_strategy", {}),
                    parsed,
                )
        if os.environ.get("AGENT_STRATEGY_MUTEXES", "").strip():
            parsed = _parse_strategy_mutexes(os.environ["AGENT_STRATEGY_MUTEXES"])
            if parsed:
                policy["strategy_mutexes"] = parsed
        return policy

    def _available_ram_gb(self) -> float:
        try:
            text = Path("/proc/meminfo").read_text(encoding="utf-8")
            kv = {}
            for line in text.splitlines():
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                parts = v.strip().split()
                if not parts:
                    continue
                kv[k] = float(parts[0]) / (1024 * 1024)
            return kv.get("MemAvailable", kv.get("MemFree", 0.0))
        except Exception:
            return 0.0

    def _read_gpu_stats(self) -> dict[int, GpuStats]:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if proc.returncode != 0:
                return {}
            out: dict[int, GpuStats] = {}
            for line in proc.stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 4:
                    continue
                idx = int(parts[0])
                total_gb = float(parts[1]) / 1024.0
                free_gb = float(parts[2]) / 1024.0
                util = float(parts[3])
                out[idx] = GpuStats(idx, total_gb, free_gb, util)
            return out
        except Exception:
            return {}

    def _gpu_slots(self, stats: dict[int, GpuStats], policy: dict[str, Any]) -> dict[int, int]:
        thresholds = policy.get("gpu", {}).get("slot_thresholds", [])
        out: dict[int, int] = {}
        for idx, gpu in stats.items():
            slots = 0
            for row in thresholds:
                try:
                    if gpu.free_gb >= float(row.get("min_free_gb", 0.0)):
                        slots = int(row.get("slots", 0))
                        break
                except Exception:
                    continue
            out[idx] = max(0, slots)
        return out

    def snapshot(self, backlog_exists: bool, policy: dict[str, Any]) -> ResourceSnapshot:
        cores = os.cpu_count() or 8
        avail_ram = self._available_ram_gb()
        cpu_cfg = policy.get("cpu", {})

        core_fraction = float(cpu_cfg.get("core_fraction", 0.5))
        ram_reserve_gb = float(cpu_cfg.get("ram_reserve_gb", 256.0))
        ram_per_slot_gb = float(cpu_cfg.get("ram_per_slot_gb", 16.0))
        min_slots = int(cpu_cfg.get("min_slots", 1))
        max_slots = int(cpu_cfg.get("max_slots", 48))
        backlog_min_slots = int(cpu_cfg.get("backlog_min_slots", 8))

        cpu_from_cores = max(min_slots, int(cores * core_fraction))
        cpu_from_ram = max(min_slots, int(max(avail_ram - ram_reserve_gb, 0.0) / max(ram_per_slot_gb, 1.0)))
        cpu_target = min(max_slots, cpu_from_cores, cpu_from_ram)
        cpu_target = max(min_slots, cpu_target)
        if backlog_exists:
            cpu_target = max(cpu_target, backlog_min_slots)

        gpu_stats = self._read_gpu_stats()
        gpu_slots = self._gpu_slots(gpu_stats, policy)
        return ResourceSnapshot(cpu_slots=cpu_target, gpu_slots=gpu_slots, gpu_stats=gpu_stats)

    def _reap(self) -> None:
        done: list[int] = []
        for pid, meta in self.running.items():
            rc = meta["proc"].poll()
            if rc is None:
                continue
            task_id = meta["task_id"]
            if rc != 0:
                self.store.mark_failed(task_id, f"worker exit code={rc}")
            done.append(pid)
        for pid in done:
            del self.running[pid]

    def _used_slots(
        self,
    ) -> tuple[int, dict[int, int], dict[int, dict[str, int]], dict[int, set[str]], dict[str, int]]:
        cpu_used = 0
        gpu_used: dict[int, int] = {}
        gpu_profile_used: dict[int, dict[str, int]] = {}
        gpu_runtime_used: dict[int, set[str]] = {}
        strategy_counts: dict[str, int] = {}

        for meta in self.running.values():
            profile = ResourceProfile(meta["profile"])
            runtime = str(meta.get("runtime_strategy", ""))
            gpu = meta.get("gpu_id")
            if runtime:
                strategy_counts[runtime] = strategy_counts.get(runtime, 0) + 1

            if profile in CPU_PROFILES:
                cpu_used += 1
                continue
            if gpu is None:
                continue
            gpu_used[gpu] = gpu_used.get(gpu, 0) + 1
            prof_map = gpu_profile_used.setdefault(gpu, {})
            prof_map[profile.value] = prof_map.get(profile.value, 0) + 1
            runtime_set = gpu_runtime_used.setdefault(gpu, set())
            if runtime:
                runtime_set.add(runtime)
        return cpu_used, gpu_used, gpu_profile_used, gpu_runtime_used, strategy_counts

    def _mutex_blocks_gpu(
        self,
        task_runtime: str,
        gpu_runtime_set: set[str],
        mutex_groups: list[list[str]],
    ) -> bool:
        if not task_runtime:
            return False
        for group in mutex_groups:
            g = set(group)
            if task_runtime not in g:
                continue
            if g.intersection(gpu_runtime_set):
                return True
        return False

    def _pick_gpu(
        self,
        task: TaskRecord,
        snapshot: ResourceSnapshot,
        gpu_used: dict[int, int],
        gpu_profile_used: dict[int, dict[str, int]],
        gpu_runtime_used: dict[int, set[str]],
        policy: dict[str, Any],
    ) -> int | None:
        profile_key = task.resource_profile.value
        per_profile_caps = policy.get("gpu", {}).get("per_profile_max_per_gpu", {})
        mutex_groups = policy.get("strategy_mutexes", [])

        candidates = sorted(
            snapshot.gpu_slots.keys(),
            key=lambda gpu_id: snapshot.gpu_stats.get(gpu_id, GpuStats(gpu_id, 0.0, 0.0, 0.0)).free_gb,
            reverse=True,
        )
        for gpu_id in candidates:
            cap = snapshot.gpu_slots.get(gpu_id, 0)
            used = gpu_used.get(gpu_id, 0)
            if used >= cap:
                continue

            per_profile_cap = int(per_profile_caps.get(profile_key, cap))
            prof_used = gpu_profile_used.get(gpu_id, {}).get(profile_key, 0)
            if prof_used >= per_profile_cap:
                continue

            runtime_set = gpu_runtime_used.get(gpu_id, set())
            if self._mutex_blocks_gpu(task.runtime_strategy, runtime_set, mutex_groups):
                continue
            return gpu_id
        return None

    def _spawn_worker(self, task: TaskRecord, gpu_id: int | None) -> None:
        cmd = [
            sys.executable,
            "-m",
            "agent.worker",
            "--store",
            str(self.store.root),
            "--task-id",
            task.id,
            "--worker-id",
            self.worker_id,
        ]
        env = os.environ.copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            cmd.extend(["--gpu-id", str(gpu_id)])

        proc = subprocess.Popen(cmd, env=env, cwd=str(self._project_root()))
        self.store.mark_running(task.id, proc.pid)
        self.running[proc.pid] = {
            "proc": proc,
            "task_id": task.id,
            "profile": task.resource_profile.value,
            "gpu_id": gpu_id,
            "runtime_strategy": task.runtime_strategy,
        }

    def dispatch_once(self) -> int:
        self._reap()

        policy = self._load_policy()
        ready = self.store.ready_tasks()
        snapshot = self.snapshot(backlog_exists=bool(ready), policy=policy)
        cpu_used, gpu_used, gpu_profile_used, gpu_runtime_used, strategy_counts = self._used_slots()
        strategy_caps = policy.get("max_workers_per_strategy", {})

        launched = 0
        for task in ready:
            cap = strategy_caps.get(task.runtime_strategy)
            if cap is not None and strategy_counts.get(task.runtime_strategy, 0) >= int(cap):
                continue

            profile = task.resource_profile
            if profile in CPU_PROFILES:
                if cpu_used >= snapshot.cpu_slots:
                    continue
                if not self.store.claim_task(task.id, self.worker_id):
                    continue
                try:
                    self._spawn_worker(task, gpu_id=None)
                except Exception as exc:
                    self.store.mark_failed(task.id, f"spawn failure: {exc}")
                    continue
                cpu_used += 1
                strategy_counts[task.runtime_strategy] = strategy_counts.get(task.runtime_strategy, 0) + 1
                launched += 1
                continue

            gpu_id = self._pick_gpu(task, snapshot, gpu_used, gpu_profile_used, gpu_runtime_used, policy)
            if gpu_id is None:
                continue
            if not self.store.claim_task(task.id, self.worker_id):
                continue
            try:
                self._spawn_worker(task, gpu_id=gpu_id)
            except Exception as exc:
                self.store.mark_failed(task.id, f"spawn failure: {exc}")
                continue
            gpu_used[gpu_id] = gpu_used.get(gpu_id, 0) + 1
            gpu_profile = gpu_profile_used.setdefault(gpu_id, {})
            profile_key = profile.value
            gpu_profile[profile_key] = gpu_profile.get(profile_key, 0) + 1
            runtime_set = gpu_runtime_used.setdefault(gpu_id, set())
            runtime_set.add(task.runtime_strategy)
            strategy_counts[task.runtime_strategy] = strategy_counts.get(task.runtime_strategy, 0) + 1
            launched += 1

        state_file = self.store.root / "state" / "scheduler_state.json"
        state_file.write_text(
            json.dumps(
                {
                    "host": self.host,
                    "worker_id": self.worker_id,
                    "cpu_slots": snapshot.cpu_slots,
                    "gpu_slots": snapshot.gpu_slots,
                    "policy": policy,
                    "running": {
                        str(pid): {
                            "task_id": meta["task_id"],
                            "profile": meta["profile"],
                            "gpu_id": meta["gpu_id"],
                            "runtime_strategy": meta.get("runtime_strategy", ""),
                        }
                        for pid, meta in self.running.items()
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        return launched

    def run_forever(self) -> None:
        while True:
            launched = self.dispatch_once()
            if launched:
                print(f"[scheduler] launched={launched} running={len(self.running)}")
            time.sleep(self.poll_interval)


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous task scheduler")
    parser.add_argument("--store", default="agent")
    parser.add_argument("--poll-interval", type=int, default=5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    store = TaskStore(args.store)
    store.ensure_layout()
    scheduler = Scheduler(store, poll_interval=args.poll_interval)

    if args.once:
        launched = scheduler.dispatch_once()
        print(f"launched={launched}")
        return 0

    scheduler.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
