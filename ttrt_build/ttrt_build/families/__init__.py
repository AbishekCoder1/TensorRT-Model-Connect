"""Auto-discover Torch-TRT family plugins from this package.

Same pattern as trtf_build/families: any .py file in this directory
(excluding _-prefixed and base.py) that exposes a module-level ``plugin``
attribute is automatically registered.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from .base import TorchTrtFamilyPlugin

ALL_PLUGINS: list[TorchTrtFamilyPlugin] = []

_pkg_dir = str(Path(__file__).parent)
for _finder, _name, _ispkg in pkgutil.iter_modules([_pkg_dir]):
    if _name.startswith("_") or _name == "base":
        continue
    try:
        _mod = importlib.import_module(f"{__name__}.{_name}")
    except ImportError:
        continue
    _plugin = getattr(_mod, "plugin", None)
    if _plugin is not None:
        ALL_PLUGINS.append(_plugin)


def find_plugin(model_type: str) -> TorchTrtFamilyPlugin | None:
    """Find the first plugin that matches the given model_type."""
    for p in ALL_PLUGINS:
        if p.matches(model_type):
            return p
    return None
