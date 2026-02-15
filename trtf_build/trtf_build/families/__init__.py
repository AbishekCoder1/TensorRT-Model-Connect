"""Auto-discover family plugins from this package."""

from __future__ import annotations

from .base import FamilyPlugin
from . import qwen, llama, mistral, gemma

_ALL_PLUGINS: list[FamilyPlugin] = [
    qwen.plugin,
    llama.plugin,
    mistral.plugin,
    gemma.plugin,
]


def find_plugin(model_type: str) -> FamilyPlugin | None:
    """Find the first plugin that matches the given model_type."""
    for p in _ALL_PLUGINS:
        if p.matches(model_type):
            return p
    return None
