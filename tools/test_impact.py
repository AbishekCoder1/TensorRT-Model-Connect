#!/usr/bin/env python3
"""Test impact analysis -- selective CI execution based on changed files.

Determines which E2E models and unit test tiers need to run based on
git diff between base and head. Safety invariant: ZERO false negatives.
Any file that doesn't match a known rule triggers ALL model tests.

Usage:
    python3 tools/test_impact.py [--base REF] [--head REF] [--json] [--verbose]
    python3 tools/test_impact.py --files path/to/file1.py,path/to/file2.cpp
    python3 tools/test_impact.py --validate
    python3 tools/test_impact.py --files trtf_build/trtf_build/families/qwen.py --cap 15
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants -- strategy mappings (mirrored from e2e_harness/contracts.py)
# ---------------------------------------------------------------------------

RUNTIME_TO_TASK_STRATEGY: Dict[str, str] = {
    "decoder_kv_cache": "text_generation_causal",
    "decoder_moe": "text_generation_causal",
    "ssm_recurrent": "text_generation_causal",
    "rwkv_recurrent": "text_generation_causal",
    "hybrid_mamba_attention": "text_generation_causal",
    "vision_language": "vision_language_generation",
    "speech_to_text": "speech_to_text",
    "text_to_audio": "text_to_audio",
    "text_to_audio_bark": "text_to_audio",
    "text_to_audio_magpie": "text_to_audio",
    "speech_to_speech": "speech_to_speech",
    "segmentation": "segmentation",
    "prompted_segmentation": "prompted_segmentation",
    "object_detection": "object_detection",
    "embedding": "embedding",
    "reranking": "reranking",
    "encoder_only": "encoder_only_nlp",
    "neural_operator": "neural_operator",
    "diffusion": "diffusion_media_generation",
    "diffusion_flux": "diffusion_media_generation",
    "diffusion_wan": "diffusion_media_generation",
    "diffusion_zimage": "diffusion_media_generation",
    "diffusion_pixart": "diffusion_media_generation",
    "omni_multimodal": "omni_multimodal",
}

# C++ plugin filename (stem) -> registered runtime_strategies
CPP_PLUGIN_STRATEGIES: Dict[str, List[str]] = {
    "decoder_plugin": ["decoder_kv_cache", "decoder_moe"],
    "ssm_plugin": ["ssm_recurrent"],
    "rwkv_plugin": ["rwkv_recurrent"],
    "hybrid_plugin": ["hybrid_mamba_attention"],
    "vl_plugin": ["vision_language"],
    "whisper_plugin": ["speech_to_text"],
    "bark_plugin": ["text_to_audio_bark"],
    "magpie_plugin": ["text_to_audio_magpie"],
    "speech_plugin": ["speech_to_speech"],
    "encoder_plugin": ["encoder_only", "embedding", "reranking", "neural_operator"],
    "segmentation_plugin": ["segmentation", "prompted_segmentation"],
    "object_detection_plugin": ["object_detection"],
    "omni_plugin": ["omni_multimodal"],
    "flux_plugin": ["diffusion_flux"],
    "wan_plugin": ["diffusion_wan", "diffusion_pixart"],
    "zimage_plugin": ["diffusion_zimage"],
}

# C++ pipeline filename (stem) -> runtime_strategies it serves
CPP_PIPELINE_STRATEGIES: Dict[str, List[str]] = {
    "text_generation_pipeline": ["decoder_kv_cache", "decoder_moe"],
    "recurrent_pipeline": ["ssm_recurrent", "rwkv_recurrent", "hybrid_mamba_attention"],
    "vl_pipeline": ["vision_language"],
    "audio_pipeline": [
        "speech_to_text", "text_to_audio_bark", "text_to_audio_magpie",
        "speech_to_speech", "omni_multimodal",
    ],
    "encoder_pipeline": [
        "encoder_only", "embedding", "reranking", "neural_operator",
        "segmentation", "prompted_segmentation", "object_detection",
    ],
    "flux_pipeline": ["diffusion_flux"],
    "wan_pipeline": ["diffusion_wan", "diffusion_pixart"],
    "z_image_pipeline": ["diffusion_zimage"],
    "diffusion_pipeline": [
        "diffusion_flux", "diffusion_wan", "diffusion_pixart", "diffusion_zimage",
    ],
}

# E2E runner filename (stem) -> task_strategies
RUNNER_TASK_STRATEGIES: Dict[str, List[str]] = {
    "text_generation": ["text_generation_causal"],
    "vision_language": ["vision_language_generation"],
    "audio_speech": ["speech_to_text", "text_to_audio", "speech_to_speech"],
    "diffusion": ["diffusion_media_generation"],
    "segmentation": ["segmentation", "prompted_segmentation", "object_detection"],
    "embedding": ["embedding"],
    "reranking": ["reranking"],
    "encoder_only": ["encoder_only_nlp"],
    "omni": ["omni_multimodal"],
    "neural_operator": ["neural_operator"],
}

# E2E comparator filename (stem) -> task_strategies
COMPARATOR_TASK_STRATEGIES: Dict[str, List[str]] = {
    "text": ["text_generation_causal"],
    "vision_language": ["vision_language_generation"],
    "speech_to_text": ["speech_to_text"],
    "text_to_audio": ["text_to_audio"],
    "speech_to_speech": ["speech_to_speech"],
    "encoder_only": ["encoder_only_nlp"],
    "embedding": ["embedding"],
    "reranking": ["reranking"],
    "segmentation": ["segmentation", "prompted_segmentation", "object_detection"],
    "diffusion": ["diffusion_media_generation"],
    "omni": ["omni_multimodal"],
    "neural_operator": ["neural_operator"],
}

# E2E reference filename (stem) -> task_strategies
REFERENCE_TASK_STRATEGIES: Dict[str, List[str]] = {
    "hf_transformers": [
        "text_generation_causal", "vision_language_generation", "text_to_audio",
        "speech_to_text", "encoder_only_nlp", "embedding", "reranking",
        "segmentation", "prompted_segmentation", "object_detection",
    ],
    "hf_diffusers": ["diffusion_media_generation"],
    "torch_reference": ["speech_to_speech", "omni_multimodal"],
}

# Shared C++ helper -> affected task_strategies
SHARED_CPP_HELPER_STRATEGIES: Dict[str, List[str]] = {
    "diffusion_helpers": [
        "diffusion_flux", "diffusion_wan", "diffusion_pixart", "diffusion_zimage",
    ],
    "audio_helpers": [
        "speech_to_text", "text_to_audio_bark", "text_to_audio_magpie",
        "speech_to_speech", "omni_multimodal",
    ],
}

# Orchestrator modules in trtf_build/ -- not treated as specialized builders
_ORCHESTRATOR_MODULES = {
    "engine_builder", "cli", "__init__", "__main__", "pipeline",
    "debug_runner", "diffusion_runner",
}

# Patterns for files that never affect E2E or unit tests
_NO_IMPACT_PATTERNS = [
    r"^docs/",
    r"^\.gitignore$",
    r"^\.clang-format$",
    r"^\.editorconfig$",
    r"^\.github/",
    r"^\.claude/",
    r"^LICENSE",
    r"^CLAUDE\.md$",
    r"^recovery-",
]

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RuleMatch:
    rule: str
    models: List[str]
    unit_tiers: List[str]
    rebuild_cpp: bool


@dataclass
class ImpactMap:
    family_to_models: Dict[str, List[str]]
    strategy_to_models: Dict[str, List[str]]       # runtime_strategy -> models
    task_strategy_to_models: Dict[str, List[str]]   # task_strategy -> models
    all_model_names: List[str]
    all_model_names_set: Set[str]
    core_models: List[str]
    builder_to_families: Dict[str, List[str]]       # parent module -> families


@dataclass
class ImpactResult:
    e2e_models: List[str]
    unit_tiers: List[str]
    rebuild_cpp: bool
    cap_applied: bool
    matched_rules: List[Dict]

# ---------------------------------------------------------------------------
# Impact map construction
# ---------------------------------------------------------------------------


def _scan_family_imports(families_dir: Path) -> Dict[str, List[str]]:
    """Build reverse index: parent_module -> [family_names that import it].

    Only returns entries for *_builder modules (excluding orchestrators).
    """
    reverse: Dict[str, Set[str]] = {}
    for py_file in sorted(families_dir.glob("*.py")):
        name = py_file.stem
        if name in ("__init__", "base"):
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
        except OSError:
            continue
        # from ..module_name import ...
        for m in re.finditer(r"from\s+\.\.(\w+)\s+import", content):
            module = m.group(1)
            reverse.setdefault(module, set()).add(name)
        # from .. import module_name
        for m in re.finditer(r"from\s+\.\.\s+import\s+([\w,\s]+)", content):
            for mod in m.group(1).split(","):
                mod = mod.strip()
                if mod:
                    reverse.setdefault(mod, set()).add(name)
    # Filter to *_builder modules only (excluding orchestrators)
    filtered: Dict[str, List[str]] = {}
    for module, families in reverse.items():
        if module.endswith("_builder") and module not in _ORCHESTRATOR_MODULES:
            filtered[module] = sorted(families)
    return filtered


def build_impact_map(repo_root: Path) -> ImpactMap:
    """Build the impact map by scanning manifests and family plugins."""
    models_dir = repo_root / "tests" / "e2e" / "models"
    families_dir = repo_root / "trtf_build" / "trtf_build" / "families"

    family_to_models: Dict[str, List[str]] = {}
    strategy_to_models: Dict[str, List[str]] = {}
    task_strategy_to_models: Dict[str, List[str]] = {}
    all_model_names: List[str] = []
    core_models: List[str] = []

    for manifest_path in sorted(models_dir.glob("*.json")):
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        name = data.get("name", manifest_path.stem)
        family = data.get("family", "")
        runtime_strategy = data.get("runtime_strategy", "")
        is_core = data.get("core", False)

        all_model_names.append(name)

        if family:
            family_to_models.setdefault(family, []).append(name)
        if runtime_strategy:
            strategy_to_models.setdefault(runtime_strategy, []).append(name)
            task_strategy = RUNTIME_TO_TASK_STRATEGY.get(runtime_strategy, "")
            if task_strategy:
                task_strategy_to_models.setdefault(task_strategy, []).append(name)
        if is_core:
            core_models.append(name)

    builder_to_families = _scan_family_imports(families_dir) if families_dir.is_dir() else {}

    return ImpactMap(
        family_to_models=family_to_models,
        strategy_to_models=strategy_to_models,
        task_strategy_to_models=task_strategy_to_models,
        all_model_names=sorted(all_model_names),
        all_model_names_set=set(all_model_names),
        core_models=sorted(core_models),
        builder_to_families=builder_to_families,
    )

# ---------------------------------------------------------------------------
# Helper: resolve models from runtime/task strategies
# ---------------------------------------------------------------------------


def _models_for_runtime_strategies(
    strategies: List[str], imap: ImpactMap,
) -> List[str]:
    models: Set[str] = set()
    for s in strategies:
        models.update(imap.strategy_to_models.get(s, []))
    return sorted(models)


def _models_for_task_strategies(
    task_strategies: List[str], imap: ImpactMap,
) -> List[str]:
    models: Set[str] = set()
    for ts in task_strategies:
        models.update(imap.task_strategy_to_models.get(ts, []))
    return sorted(models)


def _infer_unit_tiers(path: str) -> List[str]:
    """Infer which unit test tiers a file change implies."""
    tiers: List[str] = []
    if path.startswith("trtf_build/"):
        tiers.append("builder")
    if (path.startswith("src/") or path.startswith("include/")
            or path == "CMakeLists.txt" or path.startswith("cmake/")):
        tiers.append("cpp")
    if path.startswith("tests/builder/"):
        tiers.append("builder")
    if path.startswith("tests/cpp/"):
        tiers.append("cpp")
    if path.startswith("tests/tools/"):
        tiers.append("tools")
    return sorted(set(tiers))


def _infer_rebuild_cpp(path: str) -> bool:
    """Does this file change require a C++ rebuild?"""
    return (path.startswith("src/") or path.startswith("include/")
            or path == "CMakeLists.txt" or path.startswith("cmake/")
            or path.startswith("tests/cpp/"))

# ---------------------------------------------------------------------------
# File classification (ordered rules)
# ---------------------------------------------------------------------------


def classify_file(path: str, imap: ImpactMap) -> RuleMatch:
    """Classify a single changed file. First matching rule wins."""
    # Normalize path separators
    path = path.replace("\\", "/").strip("/")
    unit_tiers = _infer_unit_tiers(path)
    rebuild = _infer_rebuild_cpp(path)

    # Rule 0: E2E model manifest
    m = re.match(r"tests/e2e/models/(.+)\.json$", path)
    if m:
        name = m.group(1)
        models = [name] if name in imap.all_model_names_set else []
        return RuleMatch("manifest", models, unit_tiers, rebuild)

    # Rule 1: Family plugin (not __init__ or base)
    m = re.match(r"trtf_build/trtf_build/families/(\w+)\.py$", path)
    if m and m.group(1) not in ("__init__", "base"):
        family = m.group(1)
        models = imap.family_to_models.get(family, [])
        return RuleMatch("family_plugin", sorted(models), unit_tiers, rebuild)

    # Rule 1b: Family __init__.py or base.py -> ALL models
    m = re.match(r"trtf_build/trtf_build/families/((__init__|base)\.py)$", path)
    if m:
        return RuleMatch("family_base", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 2: Specialized builder (auto-detected via import scan)
    m = re.match(r"trtf_build/trtf_build/(\w+)\.py$", path)
    if m:
        module_name = m.group(1)
        if (module_name.endswith("_builder")
                and module_name not in _ORCHESTRATOR_MODULES
                and module_name in imap.builder_to_families):
            families = imap.builder_to_families[module_name]
            models: Set[str] = set()
            for fam in families:
                models.update(imap.family_to_models.get(fam, []))
            if models:
                return RuleMatch("specialized_builder", sorted(models), unit_tiers, rebuild)
        # Fall through to Rule 3 for non-builder or unmatched builder

    # Rule 3: Any other file under trtf_build/
    if path.startswith("trtf_build/"):
        return RuleMatch("shared_builder_module", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 4: C++ plugin
    m = re.match(r"src/runtime/plugins/(\w+)\.cpp$", path)
    if m:
        plugin_stem = m.group(1)
        if plugin_stem == "force_link_plugins":
            return RuleMatch("cpp_force_link", list(imap.all_model_names), unit_tiers, rebuild)
        strategies = CPP_PLUGIN_STRATEGIES.get(plugin_stem, [])
        if strategies:
            return RuleMatch(
                "cpp_plugin", _models_for_runtime_strategies(strategies, imap),
                unit_tiers, rebuild,
            )
        # Unknown plugin -> all models (safety)
        return RuleMatch("cpp_plugin_unknown", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 5: C++ pipeline
    m = re.match(r"src/runtime/pipelines/(\w+)\.(h|cpp)$", path)
    if m:
        pipeline_stem = m.group(1)
        strategies = CPP_PIPELINE_STRATEGIES.get(pipeline_stem, [])
        if strategies:
            return RuleMatch(
                "cpp_pipeline", _models_for_runtime_strategies(strategies, imap),
                unit_tiers, rebuild,
            )
        return RuleMatch("cpp_pipeline_unknown", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 6: Shared C++ helpers
    m = re.match(r"src/runtime/plugins/shared/(\w+)\.(h|cpp)$", path)
    if m:
        helper_stem = m.group(1)
        if helper_stem == "plugin_helpers":
            return RuleMatch("cpp_shared_plugin_helpers", list(imap.all_model_names), unit_tiers, rebuild)
        runtime_strategies = SHARED_CPP_HELPER_STRATEGIES.get(helper_stem, [])
        if runtime_strategies:
            return RuleMatch(
                "cpp_shared_helper",
                _models_for_runtime_strategies(runtime_strategies, imap),
                unit_tiers, rebuild,
            )
        return RuleMatch("cpp_shared_helper_unknown", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 7: Any other C++ source/header
    if path.startswith("src/") or path.startswith("include/"):
        return RuleMatch("cpp_source", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 8a: E2E runner
    m = re.match(r"tests/e2e_harness/runners/(\w+)\.py$", path)
    if m:
        runner_stem = m.group(1)
        if runner_stem == "__init__":
            return RuleMatch("harness_runner_init", list(imap.all_model_names), unit_tiers, rebuild)
        task_strategies = RUNNER_TASK_STRATEGIES.get(runner_stem, [])
        if task_strategies:
            return RuleMatch(
                "harness_runner",
                _models_for_task_strategies(task_strategies, imap),
                unit_tiers, rebuild,
            )
        return RuleMatch("harness_runner_unknown", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 8b: E2E comparator
    m = re.match(r"tests/e2e_harness/comparators/(\w+)\.py$", path)
    if m:
        comp_stem = m.group(1)
        if comp_stem == "__init__":
            return RuleMatch("harness_comparator_init", list(imap.all_model_names), unit_tiers, rebuild)
        task_strategies = COMPARATOR_TASK_STRATEGIES.get(comp_stem, [])
        if task_strategies:
            return RuleMatch(
                "harness_comparator",
                _models_for_task_strategies(task_strategies, imap),
                unit_tiers, rebuild,
            )
        return RuleMatch("harness_comparator_unknown", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 8c: E2E reference
    m = re.match(r"tests/e2e_harness/references/(\w+)\.py$", path)
    if m:
        ref_stem = m.group(1)
        if ref_stem == "__init__":
            return RuleMatch("harness_reference_init", list(imap.all_model_names), unit_tiers, rebuild)
        task_strategies = REFERENCE_TASK_STRATEGIES.get(ref_stem, [])
        if task_strategies:
            return RuleMatch(
                "harness_reference",
                _models_for_task_strategies(task_strategies, imap),
                unit_tiers, rebuild,
            )
        return RuleMatch("harness_reference_unknown", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 8d: Any other E2E harness file
    if path.startswith("tests/e2e_harness/"):
        return RuleMatch("harness_shared", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 9: test_e2e.py or conftest.py
    if path in ("tests/test_e2e.py", "tests/conftest.py"):
        return RuleMatch("e2e_entrypoint", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 10: Unit test directories (no E2E)
    if path.startswith("tests/builder/"):
        return RuleMatch("unit_builder", [], unit_tiers, rebuild)
    if path.startswith("tests/cpp/"):
        return RuleMatch("unit_cpp", [], unit_tiers, rebuild)
    if path.startswith("tests/tools/"):
        return RuleMatch("unit_tools", [], unit_tiers, rebuild)

    # Rule 11: CMake / build system
    if path == "CMakeLists.txt" or path.startswith("cmake/"):
        return RuleMatch("cmake", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 12: Non-code files (no impact)
    if path.startswith("tools/") or path.startswith("scripts/"):
        return RuleMatch("no_impact", [], [], False)
    for pattern in _NO_IMPACT_PATTERNS:
        if re.match(pattern, path):
            return RuleMatch("no_impact", [], [], False)
    # *.md files anywhere
    if path.endswith(".md"):
        return RuleMatch("no_impact", [], [], False)

    # CATCH-ALL: unknown file -> ALL models (safety net)
    return RuleMatch("catch_all", list(imap.all_model_names), unit_tiers, True)

# ---------------------------------------------------------------------------
# Impact analysis (aggregate across all changed files)
# ---------------------------------------------------------------------------


def analyze_impact(
    changed_files: List[str],
    imap: ImpactMap,
    cap: Optional[int] = None,
) -> ImpactResult:
    """Analyze impact of all changed files and return aggregated result."""
    all_models: Set[str] = set()
    all_tiers: Set[str] = set()
    rebuild_cpp = False
    matched_rules: List[Dict] = []

    for fpath in changed_files:
        match = classify_file(fpath, imap)
        all_models.update(match.models)
        all_tiers.update(match.unit_tiers)
        rebuild_cpp = rebuild_cpp or match.rebuild_cpp
        matched_rules.append({
            "file": fpath,
            "rule": match.rule,
            "models": match.models,
        })

    e2e_models = sorted(all_models)
    cap_applied = False
    if cap is not None and len(e2e_models) > cap:
        e2e_models = sorted(imap.core_models)
        cap_applied = True

    return ImpactResult(
        e2e_models=e2e_models,
        unit_tiers=sorted(all_tiers),
        rebuild_cpp=rebuild_cpp,
        cap_applied=cap_applied,
        matched_rules=matched_rules,
    )

# ---------------------------------------------------------------------------
# Git diff
# ---------------------------------------------------------------------------


def get_changed_files(base: str, head: str, repo_root: Path) -> Optional[List[str]]:
    """Get list of changed files between base and head.

    Returns None if git diff fails (e.g. shallow clone without base ref),
    signaling the caller to treat ALL files as changed (safety net).
    """
    for cmd in [
        ["git", "diff", "--name-only", "--diff-filter=ACMRT", f"{base}...{head}"],
        ["git", "diff", "--name-only", "--diff-filter=ACMRT", base, head],
    ]:
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, cwd=repo_root,
            )
            files = [f.strip() for f in result.stdout.strip().splitlines() if f.strip()]
            return sorted(files)
        except subprocess.CalledProcessError:
            continue
    # Both diffs failed (shallow clone, missing ref, etc.)
    print(f"WARNING: git diff failed for {base}..{head} -- "
          "treating as all files changed (safety net)", file=sys.stderr)
    return None

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_map(imap: ImpactMap, repo_root: Path) -> List[str]:
    """Validate impact map consistency. Returns list of error strings."""
    errors: List[str] = []
    warnings: List[str] = []
    families_dir = repo_root / "trtf_build" / "trtf_build" / "families"

    # 1. Every family in a manifest has a corresponding .py plugin file
    for family in imap.family_to_models:
        plugin_file = families_dir / f"{family}.py"
        if not plugin_file.exists():
            errors.append(f"Family '{family}' in manifests has no plugin file: {plugin_file}")

    # 2. Every family plugin .py has at least one manifest (warn only)
    if families_dir.is_dir():
        for py_file in sorted(families_dir.glob("*.py")):
            name = py_file.stem
            if name in ("__init__", "base"):
                continue
            if name not in imap.family_to_models:
                warnings.append(f"Family plugin '{name}.py' has no manifests using it")

    # 3. Core model set covers all distinct task_strategies
    core_task_strategies: Set[str] = set()
    for model in imap.core_models:
        for ts, models in imap.task_strategy_to_models.items():
            if model in models:
                core_task_strategies.add(ts)
    all_task_strategies = set(imap.task_strategy_to_models.keys())
    missing = all_task_strategies - core_task_strategies
    if missing:
        warnings.append(
            f"Core models don't cover task_strategies: {sorted(missing)}"
        )

    # 4. Every runtime_strategy in manifests is in RUNTIME_TO_TASK_STRATEGY
    for strategy in imap.strategy_to_models:
        if strategy not in RUNTIME_TO_TASK_STRATEGY:
            errors.append(
                f"Unknown runtime_strategy '{strategy}' in manifests "
                f"(not in RUNTIME_TO_TASK_STRATEGY)"
            )

    # 5. Every rule pattern matches at least one real file (spot checks)
    spot_checks = {
        "families_dir": families_dir.is_dir(),
        "models_dir": (repo_root / "tests" / "e2e" / "models").is_dir(),
        "src_dir": (repo_root / "src").is_dir(),
        "tests_e2e_harness": (repo_root / "tests" / "e2e_harness").is_dir(),
    }
    for name, exists in spot_checks.items():
        if not exists:
            errors.append(f"Expected directory missing for rule validation: {name}")

    # Print warnings to stderr
    for w in warnings:
        print(f"  WARN: {w}", file=sys.stderr)

    return errors

# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_human(result: ImpactResult) -> str:
    lines: List[str] = []
    if result.e2e_models:
        lines.append(f"# E2E tests to run ({len(result.e2e_models)} models):")
        for model in result.e2e_models:
            lines.append(f"tests/test_e2e.py::test_e2e[{model}]")
    else:
        lines.append("# No E2E models affected.")
    if result.unit_tiers:
        lines.append(f"# Unit test tiers: {', '.join(result.unit_tiers)}")
    lines.append(f"# C++ rebuild needed: {'yes' if result.rebuild_cpp else 'no'}")
    if result.cap_applied:
        lines.append("# WARNING: Cap applied -- running core models only.")
    return "\n".join(lines)


def format_json(result: ImpactResult) -> str:
    return json.dumps({
        "e2e_models": result.e2e_models,
        "unit_tiers": result.unit_tiers,
        "rebuild_cpp": result.rebuild_cpp,
        "cap_applied": result.cap_applied,
        "matched_rules": result.matched_rules,
    }, indent=2)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test impact analysis for selective CI execution.",
    )
    parser.add_argument("--base", default="origin/master",
                        help="Git ref for diff base (default: origin/master)")
    parser.add_argument("--head", default="HEAD",
                        help="Git ref for diff head (default: HEAD)")
    parser.add_argument("--files",
                        help="Explicit comma-separated file list (overrides git diff)")
    parser.add_argument("--cap", type=int, default=None,
                        help="If affected models > N, limit to core set + warn")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output structured JSON for CI consumption")
    parser.add_argument("--validate", action="store_true",
                        help="Check map consistency (no diff needed)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-file rule matches")
    parser.add_argument("--repo-root", default=None,
                        help="Repository root (default: auto-detect)")
    args = parser.parse_args()

    # Resolve repo root
    if args.repo_root:
        repo_root = Path(args.repo_root)
    else:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, check=True,
            )
            repo_root = Path(result.stdout.strip())
        except subprocess.CalledProcessError:
            repo_root = Path.cwd()

    imap = build_impact_map(repo_root)

    if args.validate:
        errors = validate_map(imap, repo_root)
        if errors:
            print("Validation FAILED:", file=sys.stderr)
            for e in errors:
                print(f"  ERROR: {e}", file=sys.stderr)
            return 1
        print(f"Validation passed. {len(imap.all_model_names)} models, "
              f"{len(imap.core_models)} core, "
              f"{len(imap.family_to_models)} families.",
              file=sys.stderr)
        return 0

    # Get changed files
    if args.files:
        changed: Optional[List[str]] = [f.strip() for f in args.files.split(",") if f.strip()]
    else:
        changed = get_changed_files(args.base, args.head, repo_root)

    if changed is None:
        # Git diff failed -- safety net: run everything
        print("Running all tests (git diff unavailable).", file=sys.stderr)
        result_obj = ImpactResult(
            e2e_models=list(imap.all_model_names),
            unit_tiers=["builder", "cpp", "tools"],
            rebuild_cpp=True,
            cap_applied=False,
            matched_rules=[{
                "file": "<all>", "rule": "git_diff_failed",
                "models": list(imap.all_model_names),
            }],
        )
    elif not changed:
        print("No changed files detected.", file=sys.stderr)
        result_obj = ImpactResult(
            e2e_models=[], unit_tiers=[], rebuild_cpp=False,
            cap_applied=False, matched_rules=[],
        )
    else:
        result_obj = analyze_impact(changed, imap, cap=args.cap)

    if args.verbose:
        for rule in result_obj.matched_rules:
            n = len(rule["models"])
            print(f"  {rule['file']} -> {rule['rule']} ({n} models)",
                  file=sys.stderr)

    if args.json_output:
        print(format_json(result_obj))
    else:
        print(format_human(result_obj))

    return 0


if __name__ == "__main__":
    sys.exit(main())
