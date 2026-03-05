#!/usr/bin/env python3
"""Validate runtime strategy governance matrix against source-of-truth files."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_PATH = PROJECT_ROOT / "tests" / "runtime_strategy_matrix.yaml"
DEFAULT_CPP_PATH = PROJECT_ROOT / "src" / "cabi" / "api" / "trtf_c.cpp"
DEFAULT_CONTRACTS_PATH = PROJECT_ROOT / "tests" / "e2e_harness" / "contracts.py"
DEFAULT_DIFF_CHECKS_DIR = PROJECT_ROOT / "tools" / "diff_framework" / "checks"
DEFAULT_RUNNERS_DIR = PROJECT_ROOT / "tests" / "e2e_harness" / "runners"
DEFAULT_COMPARATORS_DIR = PROJECT_ROOT / "tests" / "e2e_harness" / "comparators"
DEFAULT_CABI_DIR = PROJECT_ROOT / "src" / "cabi"

_REGISTER_FACTORY_RE = re.compile(r'register_backend_factory\(\s*"([^"]+)"')
_REGISTER_FACTORY_BINDING_RE = re.compile(
    r'register_backend_factory\(\s*"([^"]+)"\s*,\s*&([A-Za-z_]\w*)\s*\)'
)
_STRATEGY_EQ_RE = re.compile(
    r'\b(?:strategy|fp_cfg_early\.runtime_strategy)\s*==\s*"([^"]+)"'
)
_STRATEGY_NE_RE = re.compile(r'\bstrategy\s*!=\s*"([^"]+)"')
_DIRECT_RUNTIME_STRATEGY_RE = re.compile(
    r'\bfp_cfg_early\.runtime_strategy\s*==\s*"([^"]+)"'
)
_WRAPPER_DEFINITION_RE = re.compile(
    r"\b(?:static\s+)?trtf::IPipeline\*\s+([A-Za-z_]\w*_via_registry)\s*"
    r"\(\s*void\s*\*\s*(?:[A-Za-z_]\w*)?\s*\)\s*\{",
    re.MULTILINE,
)


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _class_basename(class_ref: str) -> str:
    return class_ref.rsplit(".", 1)[-1].strip()


def load_yaml_like(path: Path) -> Any:
    """Load YAML if available, otherwise require JSON-compatible YAML."""
    text = path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        yaml = None

    if yaml is not None:
        return yaml.safe_load(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} is not JSON-compatible YAML and PyYAML is unavailable."
        ) from exc


def load_runtime_strategy_matrix(path: Path) -> dict[str, dict[str, Any]]:
    """Load and validate matrix schema."""
    data = load_yaml_like(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected mapping at top level.")

    raw = data.get("runtime_strategies")
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected 'runtime_strategies' mapping.")

    matrix: dict[str, dict[str, Any]] = {}
    for runtime_strategy, entry in raw.items():
        if not _is_nonempty_str(runtime_strategy):
            raise ValueError(f"{path}: runtime strategy keys must be non-empty strings.")
        if not isinstance(entry, dict):
            raise ValueError(
                f"{path}: runtime strategy '{runtime_strategy}' must map to an object."
            )
        matrix[runtime_strategy] = dict(entry)
    return matrix


def extract_runtime_strategies_from_cpp(path: Path) -> set[str]:
    """Extract runtime_strategy keys from C++ registry + dispatch comparisons."""
    text = path.read_text(encoding="utf-8")
    strategies: set[str] = set()
    strategies.update(_REGISTER_FACTORY_RE.findall(text))
    strategies.update(_STRATEGY_EQ_RE.findall(text))
    strategies.update(_STRATEGY_NE_RE.findall(text))
    return strategies


def extract_runtime_strategies_from_cpp_files(cpp_paths: Iterable[Path]) -> set[str]:
    """Extract runtime_strategy keys from multiple C++ files."""
    strategies: set[str] = set()
    for file_path in cpp_paths:
        strategies.update(extract_runtime_strategies_from_cpp(file_path))
    return strategies


def extract_registry_strategy_wrapper_bindings_from_cpp(path: Path) -> dict[str, str]:
    """Extract runtime_strategy -> wrapper symbol bindings from registry registration."""
    text = path.read_text(encoding="utf-8")
    bindings: dict[str, str] = {}
    for strategy, wrapper_symbol in _REGISTER_FACTORY_BINDING_RE.findall(text):
        existing = bindings.get(strategy)
        if existing is not None and existing != wrapper_symbol:
            raise ValueError(
                f"{path}: strategy '{strategy}' registered with multiple wrappers "
                f"('{existing}' vs '{wrapper_symbol}')."
            )
        bindings[strategy] = wrapper_symbol
    return bindings


def extract_registry_strategy_wrapper_bindings_from_cpp_files(
    cpp_paths: Iterable[Path],
) -> dict[str, str]:
    """Extract runtime_strategy -> wrapper symbol bindings from multiple C++ files."""
    bindings: dict[str, str] = {}
    binding_sources: dict[str, Path] = {}
    for file_path in cpp_paths:
        file_bindings = extract_registry_strategy_wrapper_bindings_from_cpp(file_path)
        for strategy, wrapper_symbol in file_bindings.items():
            existing = bindings.get(strategy)
            if existing is not None and existing != wrapper_symbol:
                first_source = binding_sources[strategy]
                raise ValueError(
                    f"{file_path}: strategy '{strategy}' registered with wrapper "
                    f"'{wrapper_symbol}' but already mapped to '{existing}' in "
                    f"{first_source}."
                )
            bindings[strategy] = wrapper_symbol
            binding_sources[strategy] = file_path
    return bindings


def extract_direct_runtime_strategies_from_cpp(path: Path) -> set[str]:
    """Extract runtime strategies that bypass registry dispatch directly."""
    text = path.read_text(encoding="utf-8")
    return set(_DIRECT_RUNTIME_STRATEGY_RE.findall(text))


def extract_direct_runtime_strategies_from_cpp_files(cpp_paths: Iterable[Path]) -> set[str]:
    """Extract direct runtime_strategy dispatch keys from multiple C++ files."""
    direct_strategies: set[str] = set()
    for file_path in cpp_paths:
        direct_strategies.update(extract_direct_runtime_strategies_from_cpp(file_path))
    return direct_strategies


def extract_registry_wrapper_definitions_from_cpp(path: Path) -> set[str]:
    """Extract registry wrapper symbol definitions from a C++ source file."""
    text = path.read_text(encoding="utf-8")
    return set(_WRAPPER_DEFINITION_RE.findall(text))


def extract_registry_wrapper_definitions_by_cpp_file(
    cabi_dir: Path,
) -> dict[Path, set[str]]:
    """Collect registry wrapper definitions from src/cabi/**/*.cpp files."""
    return extract_registry_wrapper_definitions_by_cpp_files(
        cabi_dir.rglob("*.cpp")
    )


def extract_registry_wrapper_definitions_by_cpp_files(
    cpp_paths: Iterable[Path],
) -> dict[Path, set[str]]:
    """Collect registry wrapper definitions keyed by C++ file path."""
    definitions_by_file: dict[Path, set[str]] = {}
    for file_path in sorted({Path(path).resolve() for path in cpp_paths}):
        symbols = extract_registry_wrapper_definitions_from_cpp(file_path)
        if symbols:
            definitions_by_file[file_path] = symbols
    return definitions_by_file


def discover_cabi_cpp_files(*, cpp_path: Path, cabi_dir: Path) -> list[Path]:
    """Discover C++ files participating in runtime-strategy dispatch checks."""
    discovered: list[Path] = []
    if cabi_dir.exists():
        discovered.extend(path.resolve() for path in sorted(cabi_dir.rglob("*.cpp")))

    if cpp_path.exists():
        resolved_cpp_path = cpp_path.resolve()
        if resolved_cpp_path not in discovered:
            discovered.append(resolved_cpp_path)

    return discovered


def infer_cabi_dir_from_cpp_path(*, cpp_path: Path, default_cabi_dir: Path) -> Path:
    """Infer src/cabi root from cpp path (handles nested paths like api/trtf_c.cpp)."""
    resolved_cpp = cpp_path.resolve()
    for parent in resolved_cpp.parents:
        if parent.name == "cabi":
            return parent
    return default_cabi_dir.resolve()


def extract_runtime_to_task_strategy(path: Path) -> dict[str, str]:
    """Extract RUNTIME_TO_TASK_STRATEGY literal mapping from contracts.py."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for node in tree.body:
        value: Any | None = None
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "RUNTIME_TO_TASK_STRATEGY":
                    value = ast.literal_eval(node.value)
                    break
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "RUNTIME_TO_TASK_STRATEGY":
                value = ast.literal_eval(node.value)

        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"{path}: RUNTIME_TO_TASK_STRATEGY must be a dict literal.")

        mapping: dict[str, str] = {}
        for key, mapped in value.items():
            if not isinstance(key, str) or not isinstance(mapped, str):
                raise ValueError(
                    f"{path}: RUNTIME_TO_TASK_STRATEGY keys/values must be strings."
                )
            mapping[key] = mapped
        return mapping

    raise ValueError(f"{path}: RUNTIME_TO_TASK_STRATEGY not found.")


def _extract_constant_return(class_node: ast.ClassDef, method_name: str) -> str | None:
    for node in class_node.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != method_name:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Constant):
                if isinstance(sub.value.value, str):
                    return sub.value.value
    return None


def _extract_class_map_by_method(directory: Path, method_name: str) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for file_path in sorted(directory.glob("*.py")):
        if file_path.name.startswith("_"):
            continue
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            key = _extract_constant_return(node, method_name)
            if key is None:
                continue
            mapping.setdefault(key, set()).add(node.name)
    return mapping


def extract_runner_classes_by_task_strategy(runners_dir: Path) -> dict[str, set[str]]:
    """Map task_strategy -> runner classes that advertise it."""
    return _extract_class_map_by_method(runners_dir, "strategy_name")


def extract_comparator_classes_by_task_strategy(
    comparators_dir: Path,
) -> dict[str, set[str]]:
    """Map task_strategy -> comparator classes that advertise it."""
    return _extract_class_map_by_method(comparators_dir, "task_strategy")


def extract_diff_framework_checks(checks_dir: Path) -> dict[str, set[str]]:
    """Map runtime_strategy -> diff-framework check class names."""
    mapping: dict[str, set[str]] = {}
    for file_path in sorted(checks_dir.glob("*.py")):
        if file_path.name.startswith("_"):
            continue
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            runtime_strategies: list[str] | None = None
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == "runtime_strategies":
                        parsed = ast.literal_eval(stmt.value)
                        if not isinstance(parsed, list):
                            raise ValueError(
                                f"{file_path}: class {node.name} runtime_strategies must be a list literal."
                            )
                        if not all(isinstance(item, str) for item in parsed):
                            raise ValueError(
                                f"{file_path}: class {node.name} runtime_strategies must be strings."
                            )
                        runtime_strategies = parsed
                        break
            if runtime_strategies is None:
                continue
            for strategy in runtime_strategies:
                mapping.setdefault(strategy, set()).add(node.name)
    return mapping


def _append_set_mismatch(
    errors: list[str],
    *,
    left_name: str,
    left_values: set[str],
    right_name: str,
    right_values: set[str],
) -> None:
    missing = sorted(left_values - right_values)
    extra = sorted(right_values - left_values)
    if missing:
        errors.append(
            f"{right_name} missing {len(missing)} runtime strategies from {left_name}: {missing}"
        )
    if extra:
        errors.append(
            f"{right_name} has {len(extra)} extra runtime strategies vs {left_name}: {extra}"
        )


def validate_matrix_data(
    *,
    matrix: dict[str, dict[str, Any]],
    cpp_runtime_strategies: set[str],
    runtime_to_task_strategy: dict[str, str],
    diff_checks_by_strategy: dict[str, set[str]],
    runner_classes_by_task: dict[str, set[str]],
    comparator_classes_by_task: dict[str, set[str]],
) -> list[str]:
    """Validate matrix consistency and coverage requirements."""
    errors: list[str] = []

    matrix_strategies = set(matrix.keys())
    contracts_strategies = set(runtime_to_task_strategy.keys())

    _append_set_mismatch(
        errors,
        left_name="contracts.py RUNTIME_TO_TASK_STRATEGY",
        left_values=contracts_strategies,
        right_name="src/cabi/**/*.cpp strategy keys",
        right_values=cpp_runtime_strategies,
    )
    _append_set_mismatch(
        errors,
        left_name="src/cabi/**/*.cpp strategy keys",
        left_values=cpp_runtime_strategies,
        right_name="tests/runtime_strategy_matrix.yaml",
        right_values=matrix_strategies,
    )
    _append_set_mismatch(
        errors,
        left_name="contracts.py RUNTIME_TO_TASK_STRATEGY",
        left_values=contracts_strategies,
        right_name="tests/runtime_strategy_matrix.yaml",
        right_values=matrix_strategies,
    )

    wildcard_diff_checks = diff_checks_by_strategy.get("*", set())
    for runtime_strategy in sorted(matrix_strategies):
        entry = matrix[runtime_strategy]
        expected_task = runtime_to_task_strategy.get(runtime_strategy)

        if expected_task is None:
            errors.append(
                f"Matrix entry '{runtime_strategy}' is not present in RUNTIME_TO_TASK_STRATEGY."
            )
            continue

        task_strategy = entry.get("task_strategy")
        if task_strategy != expected_task:
            errors.append(
                f"{runtime_strategy}: task_strategy='{task_strategy}' "
                f"does not match contracts mapping '{expected_task}'."
            )

        cli_commands = entry.get("cli_commands")
        if (
            not isinstance(cli_commands, list)
            or not cli_commands
            or not all(_is_nonempty_str(item) for item in cli_commands)
        ):
            errors.append(
                f"{runtime_strategy}: 'cli_commands' must be a non-empty list of strings."
            )

        runner_class_ref = entry.get("runner_class")
        if not _is_nonempty_str(runner_class_ref):
            errors.append(f"{runtime_strategy}: 'runner_class' must be a non-empty string.")
        else:
            expected_runner_classes = runner_classes_by_task.get(expected_task, set())
            if not expected_runner_classes:
                errors.append(
                    f"{runtime_strategy}: no runner class found for task_strategy '{expected_task}'."
                )
            elif _class_basename(runner_class_ref) not in expected_runner_classes:
                errors.append(
                    f"{runtime_strategy}: runner_class '{runner_class_ref}' "
                    f"not in discovered runner classes {sorted(expected_runner_classes)}."
                )

        comparator_class_ref = entry.get("comparator_class")
        if comparator_class_ref is not None and not _is_nonempty_str(comparator_class_ref):
            errors.append(
                f"{runtime_strategy}: 'comparator_class' must be a non-empty string when provided."
            )
            comparator_class_ref = None

        if _is_nonempty_str(comparator_class_ref):
            expected_comparator_classes = comparator_classes_by_task.get(expected_task, set())
            if not expected_comparator_classes:
                errors.append(
                    f"{runtime_strategy}: no comparator class found for task_strategy '{expected_task}'."
                )
            elif _class_basename(comparator_class_ref) not in expected_comparator_classes:
                errors.append(
                    f"{runtime_strategy}: comparator_class '{comparator_class_ref}' "
                    f"not in discovered comparator classes {sorted(expected_comparator_classes)}."
                )

        matrix_diff_checks = entry.get("diff_framework_check_classes", [])
        if not isinstance(matrix_diff_checks, list):
            errors.append(
                f"{runtime_strategy}: 'diff_framework_check_classes' must be a list."
            )
            matrix_diff_checks = []
        elif not all(_is_nonempty_str(item) for item in matrix_diff_checks):
            errors.append(
                f"{runtime_strategy}: 'diff_framework_check_classes' entries must be non-empty strings."
            )
            matrix_diff_checks = []

        if len(set(matrix_diff_checks)) != len(matrix_diff_checks):
            errors.append(f"{runtime_strategy}: duplicate entries in diff_framework_check_classes.")

        actual_diff_checks = sorted(
            diff_checks_by_strategy.get(runtime_strategy, set()) | wildcard_diff_checks
        )
        matrix_diff_check_set = sorted(set(matrix_diff_checks))
        if matrix_diff_check_set != actual_diff_checks:
            errors.append(
                f"{runtime_strategy}: diff_framework_check_classes={matrix_diff_check_set} "
                f"does not match discovered checks={actual_diff_checks}."
            )

        diff_exemption = entry.get("diff_framework_exemption")
        has_exemption = _is_nonempty_str(diff_exemption)
        if matrix_diff_check_set and has_exemption:
            errors.append(
                f"{runtime_strategy}: diff_framework_exemption must be omitted when checks exist."
            )
        if not matrix_diff_check_set and not has_exemption:
            errors.append(
                f"{runtime_strategy}: requires 'diff_framework_exemption' when no diff-framework checks exist."
            )

        has_comparator = _is_nonempty_str(comparator_class_ref)
        if not has_comparator and not matrix_diff_check_set and not has_exemption:
            errors.append(
                f"{runtime_strategy}: requires comparator/check class coverage or explicit exemption."
            )

    return errors


def validate_registry_wrapper_coverage(
    *,
    matrix_strategies: set[str],
    registry_wrapper_bindings: dict[str, str],
    direct_runtime_strategies: set[str],
    wrapper_definitions: set[str],
) -> list[str]:
    """Validate that matrix registry strategies are covered by wrapper bindings."""
    errors: list[str] = []
    expected_registry_strategies = matrix_strategies - direct_runtime_strategies
    bound_strategies = set(registry_wrapper_bindings.keys())

    _append_set_mismatch(
        errors,
        left_name="tests/runtime_strategy_matrix.yaml strategies requiring registry dispatch",
        left_values=expected_registry_strategies,
        right_name="register_backend_factory strategy keys",
        right_values=bound_strategies,
    )

    bound_wrapper_symbols = set(registry_wrapper_bindings.values())
    missing_definitions = sorted(bound_wrapper_symbols - wrapper_definitions)
    if missing_definitions:
        errors.append(
            "register_backend_factory references wrapper symbols without definitions: "
            f"{missing_definitions}"
        )

    extra_definitions = sorted(wrapper_definitions - bound_wrapper_symbols)
    if extra_definitions:
        errors.append(
            "Found *_via_registry wrapper definitions not referenced by "
            f"register_backend_factory: {extra_definitions}"
        )

    return errors


def validate_trtf_c_wrapper_extraction_boundary(
    *,
    trtf_c_wrapper_definitions: set[str],
    extracted_wrapper_definitions: set[str],
) -> list[str]:
    """Ensure trtf_c.cpp no longer carries wrappers once extraction starts."""
    if extracted_wrapper_definitions and trtf_c_wrapper_definitions:
        lingering = sorted(trtf_c_wrapper_definitions)
        return [
            "src/cabi/trtf_c.cpp still defines *_via_registry wrappers after extraction: "
            f"{lingering}"
        ]
    return []


def validate_matrix_paths(
    *,
    matrix_path: Path = DEFAULT_MATRIX_PATH,
    cpp_path: Path = DEFAULT_CPP_PATH,
    contracts_path: Path = DEFAULT_CONTRACTS_PATH,
    diff_checks_dir: Path = DEFAULT_DIFF_CHECKS_DIR,
    runners_dir: Path = DEFAULT_RUNNERS_DIR,
    comparators_dir: Path = DEFAULT_COMPARATORS_DIR,
) -> list[str]:
    """Load all sources and validate the runtime strategy matrix."""
    cpp_path = cpp_path.resolve()
    cabi_dir = infer_cabi_dir_from_cpp_path(
        cpp_path=cpp_path,
        default_cabi_dir=DEFAULT_CABI_DIR,
    )
    cabi_cpp_files = discover_cabi_cpp_files(cpp_path=cpp_path, cabi_dir=cabi_dir)

    matrix = load_runtime_strategy_matrix(matrix_path)
    cpp_runtime_strategies = extract_runtime_strategies_from_cpp_files(cabi_cpp_files)
    runtime_to_task_strategy = extract_runtime_to_task_strategy(contracts_path)
    diff_checks_by_strategy = extract_diff_framework_checks(diff_checks_dir)
    runner_classes_by_task = extract_runner_classes_by_task_strategy(runners_dir)
    comparator_classes_by_task = extract_comparator_classes_by_task_strategy(
        comparators_dir
    )
    errors = validate_matrix_data(
        matrix=matrix,
        cpp_runtime_strategies=cpp_runtime_strategies,
        runtime_to_task_strategy=runtime_to_task_strategy,
        diff_checks_by_strategy=diff_checks_by_strategy,
        runner_classes_by_task=runner_classes_by_task,
        comparator_classes_by_task=comparator_classes_by_task,
    )

    registry_wrapper_bindings = extract_registry_strategy_wrapper_bindings_from_cpp_files(
        cabi_cpp_files
    )
    direct_runtime_strategies = extract_direct_runtime_strategies_from_cpp_files(
        cabi_cpp_files
    )
    wrapper_definitions_by_file = extract_registry_wrapper_definitions_by_cpp_files(
        cabi_cpp_files
    )

    all_wrapper_definitions: set[str] = set()
    for symbols in wrapper_definitions_by_file.values():
        all_wrapper_definitions.update(symbols)

    trtf_c_wrapper_definitions = wrapper_definitions_by_file.get(cpp_path, set())
    extracted_wrapper_definitions: set[str] = set()
    for file_path, symbols in wrapper_definitions_by_file.items():
        if file_path == cpp_path:
            continue
        extracted_wrapper_definitions.update(symbols)

    errors.extend(
        validate_registry_wrapper_coverage(
            matrix_strategies=set(matrix.keys()),
            registry_wrapper_bindings=registry_wrapper_bindings,
            direct_runtime_strategies=direct_runtime_strategies,
            wrapper_definitions=all_wrapper_definitions,
        )
    )
    errors.extend(
        validate_trtf_c_wrapper_extraction_boundary(
            trtf_c_wrapper_definitions=trtf_c_wrapper_definitions,
            extracted_wrapper_definitions=extracted_wrapper_definitions,
        )
    )
    return errors


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate tests/runtime_strategy_matrix.yaml consistency."
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=DEFAULT_MATRIX_PATH,
        help="Path to runtime strategy matrix YAML file.",
    )
    parser.add_argument(
        "--cpp",
        type=Path,
        default=DEFAULT_CPP_PATH,
        help="Path to src/cabi/api/trtf_c.cpp.",
    )
    parser.add_argument(
        "--contracts",
        type=Path,
        default=DEFAULT_CONTRACTS_PATH,
        help="Path to tests/e2e_harness/contracts.py.",
    )
    parser.add_argument(
        "--diff-checks-dir",
        type=Path,
        default=DEFAULT_DIFF_CHECKS_DIR,
        help="Path to tools/diff_framework/checks directory.",
    )
    parser.add_argument(
        "--runners-dir",
        type=Path,
        default=DEFAULT_RUNNERS_DIR,
        help="Path to tests/e2e_harness/runners directory.",
    )
    parser.add_argument(
        "--comparators-dir",
        type=Path,
        default=DEFAULT_COMPARATORS_DIR,
        help="Path to tests/e2e_harness/comparators directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        errors = validate_matrix_paths(
            matrix_path=args.matrix,
            cpp_path=args.cpp,
            contracts_path=args.contracts,
            diff_checks_dir=args.diff_checks_dir,
            runners_dir=args.runners_dir,
            comparators_dir=args.comparators_dir,
        )
    except Exception as exc:
        print(f"[runtime-strategy-matrix] ERROR: {exc}", file=sys.stderr)
        return 2

    if errors:
        print("[runtime-strategy-matrix] FAIL")
        for issue in errors:
            print(f" - {issue}")
        return 1

    print(
        "[runtime-strategy-matrix] PASS: matrix is consistent with C++ dispatch, "
        "contracts mapping, and diff-framework checks."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
