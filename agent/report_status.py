from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .store import TaskStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Print orchestrator status summary")
    parser.add_argument("--store", default="agent")
    args = parser.parse_args()

    store = TaskStore(args.store)
    tasks = store.list_tasks()
    counts = Counter(t.status.value for t in tasks)
    runtime_counts = Counter(t.runtime_strategy for t in tasks)

    print(f"tasks_total={len(tasks)}")
    for key in sorted(counts):
        print(f"{key}={counts[key]}")
    for runtime in sorted(runtime_counts):
        print(f"runtime_{runtime}={runtime_counts[runtime]}")

    scheduler_state = Path(args.store) / "state" / "scheduler_state.json"
    if scheduler_state.exists():
        data = json.loads(scheduler_state.read_text(encoding="utf-8"))
        print("scheduler_host=" + str(data.get("host", "")))
        print("cpu_slots=" + str(data.get("cpu_slots", 0)))
        print("running_workers=" + str(len(data.get("running", {}))))
        policy = data.get("policy", {})
        if policy:
            print("policy_cpu_max_slots=" + str(policy.get("cpu", {}).get("max_slots", "")))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
