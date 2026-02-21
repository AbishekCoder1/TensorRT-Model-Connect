"""Check registry — register and look up diff test checks."""

from __future__ import annotations

from .protocol import DiffTest

_REGISTRY: list[type] = []


def register(cls):
    """Class decorator — registers a DiffTest implementation."""
    _REGISTRY.append(cls)
    return cls


def get_all_tests() -> list[type]:
    """Return all registered test classes."""
    return list(_REGISTRY)


def get_tests_for_strategy(strategy: str) -> list[type]:
    """Return test classes that apply to the given runtime_strategy."""
    result = []
    for cls in _REGISTRY:
        strategies = cls.runtime_strategies
        if "*" in strategies or strategy in strategies:
            result.append(cls)
    return result


def get_test_by_name(name: str) -> type | None:
    """Look up a test class by name. Returns None if not found."""
    for cls in _REGISTRY:
        if cls.name == name:
            return cls
    return None
