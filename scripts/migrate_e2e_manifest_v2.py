#!/usr/bin/env python3
"""Automated manifest migrator: v1 -> v2 E2E model manifests.

Reads all v1 JSON manifests from tests/e2e/models/, adds v2 fields
(task_strategy, reference_backend, oracle_level, stages, comparison_profile,
determinism, preflight_requirements), converts skip -> preflight_requirements
+ known_limitations, preserves ALL existing v1 fields, and writes updated
manifests back (in-place or to output dir).

Usage:
    python scripts/migrate_e2e_manifest_v2.py
    python scripts/migrate_e2e_manifest_v2.py --output-dir /tmp/v2_manifests
    python scripts/migrate_e2e_manifest_v2.py --dry-run
    python scripts/migrate_e2e_manifest_v2.py --validate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

# Allow importing from the project
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from tests.e2e_harness.contracts import RUNTIME_TO_TASK_STRATEGY, OracleLevel

logger = logging.getLogger(__name__)

# Default models directory
_DEFAULT_MODELS_DIR = _PROJECT_ROOT / "tests" / "e2e" / "models"

# ---------------------------------------------------------------------------
# Inference rules (same as manifest_loader.py but standalone for migration)
# ---------------------------------------------------------------------------

_DEFAULT_REFERENCE_BACKEND: dict[str, str] = {
    "text_generation_causal": "hf_transformers",
    "vision_language_generation": "hf_transformers",
    "speech_to_text": "hf_transformers",
    "text_to_audio": "hf_transformers",
    "speech_to_speech": "torch_reference",
    "segmentation": "hf_transformers",
    "prompted_segmentation": "hf_transformers",
    "object_detection": "hf_transformers",
    "diffusion_media_generation": "hf_diffusers",
    "embedding": "hf_transformers",
    "reranking": "hf_transformers",
    "encoder_only_nlp": "hf_transformers",
    "neural_operator": "torch_reference",
    "omni_multimodal": "torch_reference",
    "composite_pipeline": "hf_diffusers",
}

_DEFAULT_ORACLE_LEVEL: dict[str, str] = {
    "hf_transformers": OracleLevel.L1_EXTERNAL_REFERENCE.value,
    "hf_diffusers": OracleLevel.L1_EXTERNAL_REFERENCE.value,
    "torch_reference": OracleLevel.L2_INTERNAL_REFERENCE.value,
    "custom_python": OracleLevel.L2_INTERNAL_REFERENCE.value,
    "golden_snapshot": OracleLevel.L3_SNAPSHOT_REGRESSION.value,
    "invariant_only": OracleLevel.L4_INVARIANTS.value,
}

_DEFAULT_STAGES: dict[str, list[dict[str, Any]]] = {
    "text_generation_causal": [
        {"name": "full_generation", "required": True},
    ],
    "vision_language_generation": [
        {"name": "vision_encode", "required": True},
        {"name": "full_generation", "required": True},
    ],
    "speech_to_text": [
        {"name": "full_generation", "required": True},
    ],
    "text_to_audio": [
        {"name": "full_generation", "required": True},
    ],
    "speech_to_speech": [
        {"name": "full_generation", "required": True},
    ],
    "segmentation": [
        {"name": "full_inference", "required": True},
    ],
    "prompted_segmentation": [
        {"name": "full_inference", "required": True},
    ],
    "object_detection": [
        {"name": "full_inference", "required": True},
    ],
    "diffusion_media_generation": [
        {"name": "t5_encode", "required": True},
        {"name": "dit_step", "required": True},
        {"name": "vae_decode", "required": True},
        {"name": "end_to_end", "required": True},
    ],
    "embedding": [
        {"name": "full_inference", "required": True},
    ],
    "reranking": [
        {"name": "full_inference", "required": True},
    ],
    "encoder_only_nlp": [
        {"name": "full_inference", "required": True},
    ],
    "neural_operator": [
        {"name": "full_inference", "required": True},
    ],
    "omni_multimodal": [
        {"name": "thinker_decode", "required": True},
        {"name": "vision_encode", "required": False},
        {"name": "audio_encode", "required": False},
        {"name": "talker_decode", "required": True},
        {"name": "end_to_end", "required": True},
    ],
    "composite_pipeline": [
        {"name": "end_to_end", "required": True},
    ],
}

# Fields that map to threshold_overrides in v2
_THRESHOLD_FIELDS = {
    "logit_atol", "layer_atol", "min_pixel_agreement", "min_pixel_mean",
    "max_pixel_mean", "min_pixel_std", "speech_min_token_match",
    "speech_min_frame_exact", "speech_min_rms", "num_expected_masks",
}

# Standard v1 fields that are preserved as-is (not moved to metadata)
_V1_STANDARD_FIELDS = {
    "name", "hf_id", "model_id", "bundle", "family", "runtime_strategy",
    "max_cache_length", "prompt", "test_prompt", "max_new_tokens",
    "trust_remote_code", "test_image", "test_input_audio",
    "speech_reference_tokens", "speech_test_max_frames",
    "point_x", "point_y", "video_num_frames", "video_height", "video_width",
    "num_inference_steps", "build_args",
}


# ---------------------------------------------------------------------------
# Migration logic
# ---------------------------------------------------------------------------


def _infer_task_strategy(manifest: dict) -> str:
    """Infer task_strategy from runtime_strategy."""
    rs = manifest.get("runtime_strategy", "decoder_kv_cache")
    return RUNTIME_TO_TASK_STRATEGY.get(rs, "text_generation_causal")


def _infer_reference_backend(task_strategy: str) -> str:
    return _DEFAULT_REFERENCE_BACKEND.get(task_strategy, "hf_transformers")


def _infer_oracle_level(reference_backend: str) -> str:
    return _DEFAULT_ORACLE_LEVEL.get(
        reference_backend, OracleLevel.L2_INTERNAL_REFERENCE.value
    )


def _build_preflight(manifest: dict, task_strategy: str) -> list[dict]:
    """Build preflight_requirements from v1 manifest fields."""
    reqs: list[dict] = []

    reqs.append({"kind": "binary_exists", "args": {}, "gating": True})

    if manifest.get("test_image"):
        reqs.append({
            "kind": "asset_exists",
            "args": {"path": manifest["test_image"]},
            "gating": True,
        })

    if manifest.get("test_input_audio"):
        reqs.append({
            "kind": "asset_exists",
            "args": {"path": manifest["test_input_audio"]},
            "gating": True,
        })

    if manifest.get("speech_reference_tokens"):
        reqs.append({
            "kind": "asset_exists",
            "args": {"path": manifest["speech_reference_tokens"]},
            "gating": True,
        })

    if task_strategy == "diffusion_media_generation":
        reqs.append({
            "kind": "python_module_available",
            "args": {"module": "diffusers"},
            "gating": True,
        })

    if manifest.get("trust_remote_code"):
        reqs.append({
            "kind": "hf_auth_token_present",
            "args": {},
            "gating": False,
        })

    return reqs


def _build_threshold_overrides(manifest: dict) -> dict:
    """Extract threshold fields from v1 manifest into overrides dict."""
    overrides: dict[str, Any] = {}
    for field in _THRESHOLD_FIELDS:
        if field in manifest:
            overrides[field] = manifest[field]
    return overrides


def _build_known_limitations(manifest: dict) -> list[dict] | None:
    """Convert v1 skip field to known_limitations."""
    skip_reason = manifest.get("skip")
    if skip_reason:
        return [{
            "reason": skip_reason,
            "source": "v1_skip_migration",
        }]
    return None


def migrate_manifest(manifest: dict) -> dict:
    """Migrate a single v1 manifest to v2 format.

    Preserves ALL existing v1 fields and adds new v2 fields.
    """
    v2 = deepcopy(manifest)

    # Core v2 fields
    task_strategy = _infer_task_strategy(manifest)
    reference_backend = _infer_reference_backend(task_strategy)
    oracle_level = _infer_oracle_level(reference_backend)

    # Only add v2 fields if not already present
    if "task_strategy" not in v2:
        v2["task_strategy"] = task_strategy

    if "reference_backend" not in v2:
        v2["reference_backend"] = reference_backend

    if "oracle_level" not in v2:
        v2["oracle_level"] = oracle_level

    if "stages" not in v2:
        ts = v2.get("task_strategy", task_strategy)
        v2["stages"] = _DEFAULT_STAGES.get(
            ts, [{"name": "full_generation", "required": True}]
        )

    if "comparison_profile" not in v2:
        v2["comparison_profile"] = "default"

    if "determinism" not in v2:
        v2["determinism"] = {
            "seed": manifest.get("seed", 42),
            "reruns": 0,
        }

    if "preflight_requirements" not in v2:
        ts = v2.get("task_strategy", task_strategy)
        v2["preflight_requirements"] = _build_preflight(manifest, ts)

    # Build threshold_overrides from scattered v1 fields
    if "threshold_overrides" not in v2:
        overrides = _build_threshold_overrides(manifest)
        if overrides:
            v2["threshold_overrides"] = overrides

    # Convert skip -> known_limitations
    known_limitations = _build_known_limitations(manifest)
    if known_limitations:
        if "known_limitations" not in v2:
            v2["known_limitations"] = known_limitations

    # Add manifest_version marker
    v2["manifest_version"] = 2

    # Reorder keys: put v2 fields right after the core identifiers
    return _reorder_keys(v2)


def _reorder_keys(manifest: dict) -> dict:
    """Reorder manifest keys for readability."""
    # Define preferred key order
    key_order = [
        "name", "hf_id", "model_id", "bundle", "family", "runtime_strategy",
        "manifest_version",
        # v2 core
        "task_strategy", "reference_backend", "oracle_level",
        "stages", "comparison_profile", "determinism",
        "preflight_requirements",
        # inputs
        "prompt", "test_prompt", "max_cache_length", "max_new_tokens",
        "test_image", "test_input_audio", "speech_reference_tokens",
        "speech_test_max_frames", "point_x", "point_y",
        "video_num_frames", "video_height", "video_width",
        "num_inference_steps",
        # thresholds
        "threshold_overrides",
        "logit_atol", "layer_atol",
        "min_pixel_agreement", "min_pixel_mean", "max_pixel_mean",
        "min_pixel_std", "speech_min_token_match", "speech_min_frame_exact",
        "speech_min_rms", "num_expected_masks",
        # flags
        "trust_remote_code", "build_args",
        # deprecation
        "known_limitations", "skip",
        # metadata
        "notes", "description",
    ]

    ordered: dict[str, Any] = {}
    for key in key_order:
        if key in manifest:
            ordered[key] = manifest[key]

    # Append any remaining keys not in the order list
    for key in manifest:
        if key not in ordered:
            ordered[key] = manifest[key]

    return ordered


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_manifest(manifest: dict, path: str) -> list[str]:
    """Validate a v2 manifest and return list of warnings."""
    warnings: list[str] = []

    required = ["name", "hf_id", "family", "runtime_strategy"]
    for field in required:
        if field not in manifest and (field != "hf_id" or "model_id" not in manifest):
            warnings.append(f"{path}: missing required field '{field}'")

    if "task_strategy" not in manifest:
        warnings.append(f"{path}: missing v2 field 'task_strategy'")

    if "reference_backend" not in manifest:
        warnings.append(f"{path}: missing v2 field 'reference_backend'")

    if "stages" not in manifest:
        warnings.append(f"{path}: missing v2 field 'stages'")

    ts = manifest.get("task_strategy", "")
    if ts and ts not in _DEFAULT_STAGES:
        warnings.append(f"{path}: unknown task_strategy '{ts}'")

    rb = manifest.get("reference_backend", "")
    if rb and rb not in _DEFAULT_ORACLE_LEVEL:
        warnings.append(f"{path}: unknown reference_backend '{rb}'")

    # Validate stages structure
    stages = manifest.get("stages", [])
    if stages and not isinstance(stages, list):
        warnings.append(f"{path}: 'stages' should be a list")

    # Try loading through manifest_loader for full validation
    try:
        from tests.e2e_harness.manifest_loader import load_manifest
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp:
            json.dump(manifest, tmp)
            tmp_path = tmp.name
        try:
            load_manifest(tmp_path)
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        warnings.append(f"{path}: manifest_loader validation failed: {e}")

    return warnings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate v1 E2E model manifests to v2 format."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=_DEFAULT_MODELS_DIR,
        help="Directory containing v1 JSON manifests (default: tests/e2e/models/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for v2 manifests. If not set, writes in-place.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print changes without writing files.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate v2 manifests after migration.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed migration info.",
    )
    parser.add_argument(
        "manifests",
        nargs="*",
        help="Specific manifest files to migrate (default: all in models-dir).",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Collect manifest files
    if args.manifests:
        manifest_files = [Path(m) for m in args.manifests]
    else:
        models_dir = args.models_dir
        if not models_dir.is_dir():
            logger.error("Models directory not found: %s", models_dir)
            return 1
        manifest_files = sorted(models_dir.glob("*.json"))

    if not manifest_files:
        logger.error("No manifest files found")
        return 1

    # Create output directory if needed
    output_dir = args.output_dir
    if output_dir and not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    migrated = 0
    skipped = 0
    errors = 0
    all_warnings: list[str] = []

    for manifest_path in manifest_files:
        try:
            with open(manifest_path, encoding="utf-8") as f:
                v1 = json.load(f)
        except Exception as e:
            logger.error("Failed to read %s: %s", manifest_path, e)
            errors += 1
            continue

        # Skip already-v2 manifests
        if v1.get("manifest_version") == 2:
            logger.info("Already v2, skipping: %s", manifest_path.name)
            skipped += 1
            continue

        # Migrate
        try:
            v2 = migrate_manifest(v1)
        except Exception as e:
            logger.error("Failed to migrate %s: %s", manifest_path, e)
            errors += 1
            continue

        # Validate
        if args.validate:
            warnings = validate_manifest(v2, str(manifest_path))
            all_warnings.extend(warnings)
            if warnings:
                for w in warnings:
                    logger.warning(w)

        # Compute added fields for logging
        added_fields = set(v2.keys()) - set(v1.keys())

        if args.dry_run:
            print(f"\n{'='*60}")
            print(f"File: {manifest_path.name}")
            print(f"Added fields: {sorted(added_fields)}")
            if args.verbose:
                print(json.dumps(v2, indent=4))
            else:
                # Show only the new fields
                for field in sorted(added_fields):
                    val = v2[field]
                    val_str = json.dumps(val, indent=2)
                    if len(val_str) > 80:
                        val_str = val_str[:77] + "..."
                    print(f"  + {field}: {val_str}")
        else:
            # Write output
            if output_dir:
                out_path = output_dir / manifest_path.name
            else:
                out_path = manifest_path

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(v2, f, indent=4)
                f.write("\n")

            logger.info(
                "Migrated %s -> %s (added: %s)",
                manifest_path.name,
                out_path.name if output_dir else "in-place",
                ", ".join(sorted(added_fields)) or "none",
            )

        migrated += 1

    # Summary
    print(f"\nMigration complete: {migrated} migrated, {skipped} already v2, {errors} errors")
    if all_warnings:
        print(f"Validation warnings: {len(all_warnings)}")
        for w in all_warnings:
            print(f"  ! {w}")

    return 1 if errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
