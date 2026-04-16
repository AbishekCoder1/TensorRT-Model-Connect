#!/usr/bin/env python3
"""Warm the HF Hub file/metadata cache before a parallel E2E rebuild phase.

Two operating modes, selected by --models-file:

Nightly mode (no --models-file):
    Calls snapshot_download() for every non-skipped E2E model.  This refreshes
    the cached commit SHA (model_info() call) and downloads any stale/missing
    files, so the parallel rebuild phase can set HF_HUB_OFFLINE=1 safely.
    Sequential with a 0.3 s inter-request delay to stay below the HF API rate
    limit (10 k requests / 5 min).

MR CI selective mode (--models-file FILE):
    For each model in FILE:
      - Already in cache → skip entirely (zero network calls).
      - Not in cache     → call snapshot_download() to download it.
    This handles newly added model families whose weights are not yet in the
    persistent cache without making unnecessary API calls for the majority of
    models that are already cached.

Usage:
    # Nightly — warm all non-skipped E2E models:
    python scripts/warm_hf_cache.py

    # MR CI — only download models missing from cache:
    python scripts/warm_hf_cache.py --models-file e2e_models.txt

Exit code 0 even on partial failures — missing cache entries produce a warning
but do not block CI.
"""

import argparse
import json
import pathlib
import sys
import time

try:
    from huggingface_hub import constants as hf_constants
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("ERROR: huggingface_hub not available", file=sys.stderr)
    sys.exit(1)

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "--models-file",
    metavar="FILE",
    help="Path to a file with one model name per line (manifest stems). "
         "When given, only those models are considered and already-cached "
         "models are skipped (no network call). Intended for MR CI selective "
         "warm.",
)
args = parser.parse_args()

ROOT = pathlib.Path(__file__).resolve().parent.parent
manifests = sorted((ROOT / "tests" / "e2e" / "models").glob("*.json"))

# Optional filter: only consider models listed in --models-file
filter_names: set[str] | None = None
if args.models_file:
    p = pathlib.Path(args.models_file)
    if not p.is_file():
        print(f"ERROR: --models-file {p} not found", file=sys.stderr)
        sys.exit(1)
    filter_names = {line.strip() for line in p.read_text().splitlines() if line.strip()}

entries: list[tuple[str, str]] = []
for m in manifests:
    d = json.loads(m.read_text())
    name = d.get("name", m.stem)
    if d.get("skip"):
        continue
    if not d.get("hf_id"):
        continue
    if filter_names is not None and name not in filter_names:
        continue
    entries.append((name, d["hf_id"]))


def _is_cached(hf_id: str) -> bool:
    """Return True if the model has at least one local snapshot.

    Checks the HF cache directory structure without making any network call.
    Respects HF_HOME / HUGGINGFACE_HUB_CACHE environment variables via the
    huggingface_hub constants module.
    """
    cache_dir = pathlib.Path(hf_constants.HF_HUB_CACHE)
    # HF cache layout: models--{org}--{model}/snapshots/{sha}/
    repo_dir = cache_dir / ("models--" + hf_id.replace("/", "--"))
    snapshots_dir = repo_dir / "snapshots"
    return snapshots_dir.is_dir() and any(snapshots_dir.iterdir())


selective = filter_names is not None
scope = f"selective ({len(entries)} models)" if selective else f"all {len(entries)} models"
print(f"Warming HF cache — {scope}...")

warned: list[str] = []
skipped: list[str] = []

for i, (name, hf_id) in enumerate(entries, 1):
    if selective and _is_cached(hf_id):
        print(f"  [{i:3d}/{len(entries)}] {name}  CACHED (skip)")
        skipped.append(name)
        continue

    try:
        snapshot_download(hf_id)
        print(f"  [{i:3d}/{len(entries)}] {name}  OK")
    except HfHubHTTPError as e:
        print(f"  [{i:3d}/{len(entries)}] {name}  WARN (HTTP {e.response.status_code}): {e}")
        warned.append(name)
    except Exception as e:  # noqa: BLE001
        print(f"  [{i:3d}/{len(entries)}] {name}  WARN: {e}")
        warned.append(name)
    # Small inter-request delay to stay well below the API rate limit.
    time.sleep(0.3)

print()
if selective and skipped:
    print(f"Skipped {len(skipped)} already-cached models (no network calls).")
if warned:
    print(
        f"Warning: {len(warned)}/{len(entries)} models could not be warmed: {warned}",
        file=sys.stderr,
    )
    print("Parallel E2E phase may re-issue model_info() for these models.")
else:
    downloaded = len(entries) - len(skipped)
    if downloaded == 0:
        print("All models already cached — zero network calls.")
    else:
        print(f"Downloaded {downloaded} model(s) successfully.")
