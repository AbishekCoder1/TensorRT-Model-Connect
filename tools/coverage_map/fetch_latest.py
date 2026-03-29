#!/usr/bin/env python3
"""Fetch the latest coverage_map.json artifact for CI use.

Tries GitLab Jobs API first, then falls back to a local path.
Exits with code 1 if no map is available (signals full-tier fallback).

Usage:
    python tools/coverage_map/fetch_latest.py --output coverage_map.json
    python tools/coverage_map/fetch_latest.py --output coverage_map.json --local-fallback /shared/coverage_map.json
"""

import argparse
import shutil
import sys
import urllib.request
import urllib.error
from pathlib import Path


def _try_gitlab_download(api_url: str, output_path: Path) -> bool:
    """Try to download coverage_map.json from GitLab artifacts API."""
    import os
    token = os.environ.get("CI_JOB_TOKEN") or os.environ.get("PRIVATE_TOKEN")
    if not token or not api_url:
        return False

    headers = {"PRIVATE-TOKEN": token} if os.environ.get("PRIVATE_TOKEN") else {"JOB-TOKEN": token}

    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(resp.read())
            return True
    except (urllib.error.URLError, OSError) as e:
        print(f"WARNING: GitLab API fetch failed: {e}", file=sys.stderr)
        return False


def resolve_coverage_map(
    output_path: Path,
    local_fallback: str = "",
    gitlab_api_url: str = None,
) -> bool:
    """Try to resolve a coverage map, writing it to output_path.

    Tries in order:
    1. GitLab artifacts API (if gitlab_api_url is set)
    2. Local fallback path (if provided and exists)

    Returns True if a map was written to output_path, False otherwise.
    """
    if gitlab_api_url:
        if _try_gitlab_download(gitlab_api_url, output_path):
            print("[fetch] Downloaded coverage map from GitLab API", file=sys.stderr)
            return True

    if local_fallback:
        local_path = Path(local_fallback)
        if local_path.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, output_path)
            print(f"[fetch] Using local fallback: {local_path}", file=sys.stderr)
            return True

    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch latest coverage_map.json for CI.",
    )
    parser.add_argument("--output", "-o", required=True, help="Output path")
    parser.add_argument("--local-fallback", default="",
                        help="Local path to fall back to if API fails")
    parser.add_argument("--gitlab-api-url", default=None,
                        help="GitLab artifacts API URL")
    args = parser.parse_args()

    found = resolve_coverage_map(
        output_path=Path(args.output),
        local_fallback=args.local_fallback,
        gitlab_api_url=args.gitlab_api_url,
    )

    if not found:
        print("WARNING: No coverage map available. Run all tests.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
