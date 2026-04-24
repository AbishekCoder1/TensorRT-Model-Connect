#!/usr/bin/env python3
"""Schedule E2E tests across GPUs and workers for balanced load.

Reads pytest test IDs (one per line on stdin) and model manifests,
classifies models as large/small, then distributes them evenly across
GPU×worker slots so each GPU runs a mix of heavy and light models.

Usage:
    pytest tests/test_e2e.py --co -q | grep test_e2e | \\
        python scripts/schedule_e2e.py --num-gpus 4 --workers-per-gpu 4

Output (JSON to stdout):
    {
      "0": [["tests/...test_e2e[model-a]", ...], ["tests/...test_e2e[model-b]", ...], ...],
      "1": [["tests/...test_e2e[model-c]", ...], ...],
      ...
    }
    Key = GPU index, value = list of worker queues (each queue is a list of test IDs).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Size classification
# ---------------------------------------------------------------------------

# Strategies that are inherently GPU-heavy regardless of param count.
_HEAVY_STRATEGIES = frozenset({
    "vision_language",
    "speech_to_speech",
})

_EXCLUSIVE_GPU_RESOURCE = "exclusive_gpu"


def _param_billions(hf_id: str) -> float | None:
    """Extract approximate parameter count in billions from HF ID string."""
    # Match patterns like "7B", "0.6B", "30B-A3B", "1.3B", "350M"
    for m in re.finditer(r"(\d+\.?\d*)\s*([BbMm])", hf_id):
        val = float(m.group(1))
        unit = m.group(2).upper()
        if unit == "M":
            return val / 1000.0
        return val
    return None


def classify_size(manifest: dict) -> str:
    """Classify a model as 'large' or 'small' based on manifest metadata.

    Large = likely to use significant GPU memory (>= 3B params, or heavy
    strategy like diffusion/VL/speech-to-speech).
    """
    strategy = str(manifest.get("runtime_strategy", "") or "")
    hf_id = str(manifest.get("hf_id", "") or "")

    # Heavy strategies are always large. Diffusion strategies are named by
    # backend family, e.g. diffusion_flux and diffusion_ltx.
    if strategy == "diffusion" or strategy.startswith("diffusion_") or strategy in _HEAVY_STRATEGIES:
        return "large"

    # MoE with real weights (not the 15M toy)
    if strategy == "decoder_moe":
        params = _param_billions(hf_id)
        if params is not None and params >= 1.0:
            return "large"

    # Hybrid models (Nemotron-H)
    if strategy == "hybrid_mamba_attention":
        return "large"

    # Check param count from HF ID
    params = _param_billions(hf_id)
    if params is not None and params >= 3.0:
        return "large"

    # bark-large has "bark" in the name but no size suffix — treat as large
    name = str(manifest.get("name", "") or "")
    if "bark-large" in name:
        return "large"

    return "small"


def classify_parallel_resource(manifest: dict) -> str:
    """Return the requested parallel scheduling resource tier."""
    if manifest.get("e2e_parallel_resource") == _EXCLUSIVE_GPU_RESOURCE:
        return _EXCLUSIVE_GPU_RESOURCE
    return "shared"


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


def schedule(
    test_ids: list[str],
    manifest_dir: Path,
    num_gpus: int,
    workers_per_gpu: int,
) -> dict[str, list[list[str]]]:
    """Produce balanced GPU×worker assignments.

    Algorithm:
    1. Classify each test as large or small using its manifest.
    2. Round-robin large models across GPUs, then small models across GPUs.
       This ensures each GPU gets ~equal counts of each size tier.
    3. For each GPU, interleave its large and small models (L, S, L, S, ...)
       then round-robin across workers.  This ensures at any moment,
       ~half the running models are large and ~half are small.
    """
    # Load manifests keyed by model name
    manifests: dict[str, dict] = {}
    for f in manifest_dir.glob("*.json"):
        try:
            m = json.loads(f.read_text())
            manifests[m["name"]] = m
        except (json.JSONDecodeError, KeyError):
            continue

    # Classify
    exclusive_tests: list[str] = []
    large_tests: list[str] = []
    small_tests: list[str] = []
    for tid in test_ids:
        # Extract model name from "tests/test_e2e.py::test_e2e[model-name]"
        match = re.search(r"\[(.+?)\]", tid)
        name = match.group(1) if match else ""
        m = manifests.get(name, {})
        if classify_parallel_resource(m) == _EXCLUSIVE_GPU_RESOURCE:
            exclusive_tests.append(tid)
        elif classify_size(m) == "large":
            large_tests.append(tid)
        else:
            small_tests.append(tid)

    # Reserve whole GPUs for exclusive tests. This keeps high-memory build
    # cases from colliding with unrelated workers while preserving parallelism
    # on the remaining GPUs.
    result: dict[str, list[list[str]]] = {}
    reserved_gpu_count = min(len(exclusive_tests), num_gpus)
    for i, tid in enumerate(exclusive_tests):
        gpu_id = i % max(reserved_gpu_count, 1)
        result.setdefault(str(gpu_id), [[]])
        result[str(gpu_id)][0].append(tid)

    shared_gpu_ids = list(range(reserved_gpu_count, num_gpus))
    if not shared_gpu_ids:
        shared_gpu_ids = list(range(num_gpus))
        for gpu_id in shared_gpu_ids:
            result.setdefault(str(gpu_id), [[]])

    # Round-robin large across GPUs, then small across GPUs
    gpu_large: dict[int, list[str]] = {gpu_id: [] for gpu_id in shared_gpu_ids}
    gpu_small: dict[int, list[str]] = {gpu_id: [] for gpu_id in shared_gpu_ids}

    for i, tid in enumerate(large_tests):
        gpu_large[shared_gpu_ids[i % len(shared_gpu_ids)]].append(tid)
    for i, tid in enumerate(small_tests):
        gpu_small[shared_gpu_ids[i % len(shared_gpu_ids)]].append(tid)

    # For each GPU: interleave large/small, then distribute across workers
    for gpu_id in shared_gpu_ids:
        L = gpu_large[gpu_id]
        S = gpu_small[gpu_id]

        # Interleave: L, S, L, S, ...
        interleaved: list[str] = []
        li = si = 0
        while li < len(L) or si < len(S):
            if li < len(L):
                interleaved.append(L[li])
                li += 1
            if si < len(S):
                interleaved.append(S[si])
                si += 1

        # Round-robin across workers
        workers: list[list[str]] = [[] for _ in range(workers_per_gpu)]
        for i, tid in enumerate(interleaved):
            workers[i % workers_per_gpu].append(tid)

        # Drop empty workers
        shared_workers = [w for w in workers if w]
        if str(gpu_id) in result:
            result[str(gpu_id)][0].extend([tid for worker in shared_workers for tid in worker])
        else:
            result[str(gpu_id)] = shared_workers

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tests" / "e2e" / "models",
    )
    args = parser.parse_args()

    # Read test IDs from stdin
    test_ids = [line.strip() for line in sys.stdin if line.strip()]
    if not test_ids:
        print("ERROR: No test IDs on stdin.", file=sys.stderr)
        sys.exit(1)

    assignments = schedule(
        test_ids,
        args.manifest_dir,
        args.num_gpus,
        args.workers_per_gpu,
    )

    # Print summary to stderr
    total_large = 0
    total_small = 0
    manifests: dict[str, dict] = {}
    for f in args.manifest_dir.glob("*.json"):
        try:
            m = json.loads(f.read_text())
            manifests[m["name"]] = m
        except (json.JSONDecodeError, KeyError):
            continue

    print(f"Schedule: {len(test_ids)} tests across {args.num_gpus} GPUs "
          f"x {args.workers_per_gpu} workers/GPU", file=sys.stderr)
    for gpu_id, workers in sorted(assignments.items(), key=lambda x: int(x[0])):
        n_tests = sum(len(w) for w in workers)
        n_large = 0
        n_exclusive = 0
        for w in workers:
            for tid in w:
                match = re.search(r"\[(.+?)\]", tid)
                name = match.group(1) if match else ""
                manifest = manifests.get(name, {})
                if classify_parallel_resource(manifest) == _EXCLUSIVE_GPU_RESOURCE:
                    n_exclusive += 1
                if classify_size(manifest) == "large":
                    n_large += 1
        n_small = n_tests - n_large
        total_large += n_large
        total_small += n_small
        print(f"  GPU {gpu_id}: {n_tests} tests ({n_large}L + {n_small}S) "
              f"across {len(workers)} workers"
              f"{f' [{n_exclusive} exclusive]' if n_exclusive else ''}",
              file=sys.stderr)
    print(f"  Total: {total_large} large + {total_small} small", file=sys.stderr)

    # Output JSON to stdout
    json.dump(assignments, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
