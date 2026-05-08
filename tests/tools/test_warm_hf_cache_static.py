"""Static contract checks for scripts/warm_hf_cache.py."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WARM_HF_CACHE = REPO_ROOT / "scripts" / "warm_hf_cache.py"


def test_magpie_reference_dependencies_are_warmed() -> None:
    text = WARM_HF_CACHE.read_text()
    assert "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps" in text
    assert "google/byt5-small" in text
    assert "microsoft/wavlm-base-plus" in text


def test_nemo_archives_count_as_complete_snapshots() -> None:
    text = WARM_HF_CACHE.read_text()
    assert (
        'if any(fnmatch.fnmatch(name, "*.nemo") for name in files):\n'
        "        return True"
    ) in text
