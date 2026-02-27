"""Plugin registry for strategy runners, reference backends, and comparators.

Supports both explicit registration and auto-discovery from subdirectories.
Auto-discovery scans ``runners/``, ``references/``, and ``comparators/``
sibling packages for modules that expose a module-level ``plugin`` attribute
implementing the corresponding protocol.

Usage:
    from tests.e2e_harness.registry import get_runner, get_comparator

    runner = get_runner("text_generation_causal")
    output = runner.run_stage(case, stage, ctx)

Auto-discovery is lazy: it runs on first access if the registry is empty.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Dict, Optional

from .contracts import Comparator, ReferenceBackendRunner, TaskStrategyRunner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal registries
# ---------------------------------------------------------------------------

_strategy_runners: Dict[str, TaskStrategyRunner] = {}
_reference_backends: Dict[str, ReferenceBackendRunner] = {}
_comparators: Dict[str, Comparator] = {}
_discovered = False


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_runner(runner: TaskStrategyRunner) -> None:
    """Register a TaskStrategyRunner by its strategy_name."""
    name = runner.strategy_name
    if name in _strategy_runners:
        logger.warning("Overwriting strategy runner for %s", name)
    _strategy_runners[name] = runner


def register_reference(ref: ReferenceBackendRunner) -> None:
    """Register a ReferenceBackendRunner by its backend_name."""
    name = ref.backend_name
    if name in _reference_backends:
        logger.warning("Overwriting reference backend for %s", name)
    _reference_backends[name] = ref


def register_comparator(comp: Comparator) -> None:
    """Register a Comparator by its task_strategy."""
    name = comp.task_strategy
    if name in _comparators:
        logger.warning("Overwriting comparator for %s", name)
    _comparators[name] = comp


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _scan_package(package_dir: Path, package_prefix: str) -> list:
    """Scan a directory for modules with a ``plugin`` attribute.

    Returns list of (module_name, plugin_object) pairs.
    """
    plugins = []
    if not package_dir.is_dir():
        return plugins

    for importer, mod_name, is_pkg in pkgutil.iter_modules([str(package_dir)]):
        if mod_name.startswith("_"):
            continue
        full_name = f"{package_prefix}.{mod_name}"
        try:
            mod = importlib.import_module(full_name)
        except Exception:
            logger.warning("Failed to import plugin module %s", full_name, exc_info=True)
            continue
        plugin = getattr(mod, "plugin", None)
        if plugin is not None:
            plugins.append((full_name, plugin))
    return plugins


def discover_plugins() -> None:
    """Scan runners/, references/, and comparators/ for plugin modules.

    Each scanned module should have a module-level ``plugin`` attribute that
    is an instance of TaskStrategyRunner, ReferenceBackendRunner, or
    Comparator respectively. Plugins are registered automatically.

    Safe to call multiple times; subsequent calls are no-ops unless
    ``_discovered`` is reset.
    """
    global _discovered
    if _discovered:
        return
    _discovered = True

    harness_dir = Path(__file__).resolve().parent

    # Discover strategy runners
    runners_dir = harness_dir / "runners"
    runners_prefix = __package__ + ".runners" if __package__ else "e2e_harness.runners"
    for mod_name, plugin in _scan_package(runners_dir, runners_prefix):
        if isinstance(plugin, TaskStrategyRunner):
            register_runner(plugin)
            logger.debug("Auto-registered runner from %s: %s", mod_name, plugin.strategy_name)
        else:
            logger.warning(
                "Module %s has plugin attribute but it is not a TaskStrategyRunner", mod_name
            )

    # Discover reference backends
    refs_dir = harness_dir / "references"
    refs_prefix = __package__ + ".references" if __package__ else "e2e_harness.references"
    for mod_name, plugin in _scan_package(refs_dir, refs_prefix):
        if isinstance(plugin, ReferenceBackendRunner):
            register_reference(plugin)
            logger.debug("Auto-registered reference from %s: %s", mod_name, plugin.backend_name)
        else:
            logger.warning(
                "Module %s has plugin attribute but it is not a ReferenceBackendRunner", mod_name
            )

    # Discover comparators
    comps_dir = harness_dir / "comparators"
    comps_prefix = __package__ + ".comparators" if __package__ else "e2e_harness.comparators"
    for mod_name, plugin in _scan_package(comps_dir, comps_prefix):
        if isinstance(plugin, Comparator):
            register_comparator(plugin)
            logger.debug("Auto-registered comparator from %s: %s", mod_name, plugin.task_strategy)
        else:
            logger.warning(
                "Module %s has plugin attribute but it is not a Comparator", mod_name
            )


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def _ensure_discovered() -> None:
    """Trigger auto-discovery on first access if not already done."""
    if not _discovered:
        discover_plugins()


def get_runner(strategy_name: str) -> Optional[TaskStrategyRunner]:
    """Look up a registered TaskStrategyRunner by strategy name.

    Returns None if no runner is registered for the given strategy.
    Triggers auto-discovery on first call.
    """
    _ensure_discovered()
    return _strategy_runners.get(strategy_name)


def get_reference(backend_name: str) -> Optional[ReferenceBackendRunner]:
    """Look up a registered ReferenceBackendRunner by backend name.

    Returns None if no backend is registered for the given name.
    Triggers auto-discovery on first call.
    """
    _ensure_discovered()
    return _reference_backends.get(backend_name)


def get_comparator(task_strategy: str) -> Optional[Comparator]:
    """Look up a registered Comparator by task strategy.

    Returns None if no comparator is registered for the given strategy.
    Triggers auto-discovery on first call.
    """
    _ensure_discovered()
    return _comparators.get(task_strategy)


def list_runners() -> Dict[str, TaskStrategyRunner]:
    """Return a copy of all registered strategy runners."""
    _ensure_discovered()
    return dict(_strategy_runners)


def list_references() -> Dict[str, ReferenceBackendRunner]:
    """Return a copy of all registered reference backends."""
    _ensure_discovered()
    return dict(_reference_backends)


def list_comparators() -> Dict[str, Comparator]:
    """Return a copy of all registered comparators."""
    _ensure_discovered()
    return dict(_comparators)


def reset() -> None:
    """Clear all registries and reset discovery state. For testing only."""
    global _discovered
    _strategy_runners.clear()
    _reference_backends.clear()
    _comparators.clear()
    _discovered = False
