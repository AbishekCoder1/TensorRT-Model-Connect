from __future__ import annotations

import argparse
import time

from .orchestrator import process_inbox_once
from .store import TaskStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuously ingest HF links")
    parser.add_argument("--store", default="agent")
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    store = TaskStore(args.store)
    store.ensure_layout()

    if args.once:
        created = process_inbox_once(store)
        print(f"ingested_tasks={created}")
        return 0

    while True:
        created = process_inbox_once(store)
        if created:
            print(f"[intake] created_tasks={created}")
        time.sleep(max(args.interval, 1))


if __name__ == "__main__":
    raise SystemExit(main())
