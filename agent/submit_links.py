from __future__ import annotations

import argparse
from pathlib import Path

from .store import TaskStore
from .utils import normalize_hf_model_id


def _collect_links(args: argparse.Namespace) -> list[str]:
    links = [x.strip() for x in args.link if x.strip()]
    if args.links_file:
        links.extend(
            line.strip()
            for line in Path(args.links_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    out: list[str] = []
    seen: set[str] = set()
    for raw in links:
        normalized = normalize_hf_model_id(raw)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(raw)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Queue Hugging Face links for agent intake")
    parser.add_argument("--store", default="agent", help="Task store directory")
    parser.add_argument("--link", action="append", default=[], help="HF link or model id (repeatable)")
    parser.add_argument(
        "--links-file",
        default="",
        help="Optional file with one HF link/model id per line",
    )
    args = parser.parse_args()

    links = _collect_links(args)
    if not links:
        print("queued=0")
        return 0

    store = TaskStore(args.store)
    store.ensure_layout()
    for link in links:
        store.enqueue_link(link)

    print(f"queued={len(links)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
