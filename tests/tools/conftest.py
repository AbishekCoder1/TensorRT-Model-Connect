"""Shared fixtures for diff framework self-tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _add_tools_to_path():
    """Ensure tools/ is importable."""
    tools_dir = str(Path(__file__).resolve().parents[2] / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
