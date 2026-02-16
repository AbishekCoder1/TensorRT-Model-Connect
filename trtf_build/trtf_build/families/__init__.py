"""Auto-discover family plugins from this package.

Any .py file in this directory (excluding _-prefixed and base.py) that exposes
a module-level ``plugin`` attribute is automatically registered. Adding a new
family = drop a .py file, zero edits to shared files.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from .base import FamilyPlugin

_ALL_PLUGINS: list[FamilyPlugin] = []

# Scan every .py module in this directory.
_pkg_dir = str(Path(__file__).parent)
for _finder, _name, _ispkg in pkgutil.iter_modules([_pkg_dir]):
    # Skip private modules and the base protocol definition.
    if _name.startswith("_") or _name == "base":
        continue
    _mod = importlib.import_module(f"{__name__}.{_name}")
    _plugin = getattr(_mod, "plugin", None)
    if _plugin is not None:
        _ALL_PLUGINS.append(_plugin)


def find_plugin(model_type: str) -> FamilyPlugin | None:
    """Find the first plugin that matches the given model_type."""
    for p in _ALL_PLUGINS:
        if p.matches(model_type):
            return p
    return None
