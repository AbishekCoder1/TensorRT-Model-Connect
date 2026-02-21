from __future__ import annotations

import argparse
import time

from .scheduler import Scheduler
from .store import TaskStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Worker supervisor")
    parser.add_argument("--store", default="agent")
    parser.add_argument("--poll-interval", type=int, default=5)
    args = parser.parse_args()

    store = TaskStore(args.store)
    store.ensure_layout()
    scheduler = Scheduler(store, poll_interval=args.poll_interval)

    while True:
        scheduler.dispatch_once()
        time.sleep(max(1, args.poll_interval))


if __name__ == "__main__":
    raise SystemExit(main())
