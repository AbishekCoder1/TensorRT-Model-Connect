"""Tests for scripts/publish_ci_report.py."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


def _import_publisher():
    scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return importlib.import_module("publish_ci_report")


def _write_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<!doctype html><html><body>report</body></html>", encoding="utf-8")


def _load_index(site_root: Path) -> dict:
    return json.loads((site_root / "index.json").read_text(encoding="utf-8"))


def test_publishes_nightly_report_with_fourteen_day_expiry(tmp_path: Path) -> None:
    mod = _import_publisher()
    report = tmp_path / "source" / "report.html"
    site = tmp_path / "site"
    _write_report(report)

    rc = mod.main(
        [
            "--site-root",
            str(site),
            "--report-html",
            str(report),
            "--kind",
            "nightly",
            "--run-id",
            "12345",
            "--workflow-name",
            "TensorRT-Model-Connect Nightly CI",
            "--conclusion",
            "failure",
            "--created-at",
            "2026-05-07T00:00:00Z",
            "--now",
            "2026-05-07T00:00:00Z",
            "--base-url",
            "https://reports.example/trtmc",
        ]
    )

    assert rc == 0
    assert (site / "nightly" / "12345" / "index.html").is_file()
    index = _load_index(site)
    assert index["reports"][0]["kind"] == "nightly"
    assert index["reports"][0]["expires_at"] == "2026-05-21T00:00:00Z"


def test_publishes_premerge_report_with_one_day_expiry(tmp_path: Path) -> None:
    mod = _import_publisher()
    report = tmp_path / "source" / "report.html"
    site = tmp_path / "site"
    _write_report(report)

    rc = mod.main(
        [
            "--site-root",
            str(site),
            "--report-html",
            str(report),
            "--kind",
            "premerge",
            "--run-id",
            "98765",
            "--pr-number",
            "42",
            "--created-at",
            "2026-05-07T00:00:00Z",
            "--now",
            "2026-05-07T00:00:00Z",
        ]
    )

    assert rc == 0
    assert (site / "premerge" / "98765" / "index.html").is_file()
    index = _load_index(site)
    assert index["reports"][0]["kind"] == "premerge"
    assert index["reports"][0]["pr_number"] == "42"
    assert index["reports"][0]["expires_at"] == "2026-05-08T00:00:00Z"


def test_prunes_expired_reports_and_keeps_active_reports(tmp_path: Path) -> None:
    mod = _import_publisher()
    site = tmp_path / "site"
    expired_dir = site / "premerge" / "old"
    active_dir = site / "nightly" / "new"
    expired_dir.mkdir(parents=True)
    active_dir.mkdir(parents=True)
    (expired_dir / "index.html").write_text("old", encoding="utf-8")
    (active_dir / "index.html").write_text("new", encoding="utf-8")
    (site / "index.json").write_text(
        json.dumps(
            {
                "version": 1,
                "reports": [
                    {
                        "kind": "premerge",
                        "run_id": "old",
                        "path": "premerge/old/index.html",
                        "created_at": "2026-05-06T00:00:00Z",
                        "expires_at": "2026-05-07T00:00:00Z",
                    },
                    {
                        "kind": "nightly",
                        "run_id": "new",
                        "path": "nightly/new/index.html",
                        "created_at": "2026-05-07T00:00:00Z",
                        "expires_at": "2026-05-21T00:00:00Z",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    rc = mod.main(
        [
            "--site-root",
            str(site),
            "--now",
            "2026-05-07T00:00:00Z",
            "--prune-only",
        ]
    )

    assert rc == 0
    assert not expired_dir.exists()
    assert active_dir.exists()
    index = _load_index(site)
    assert [entry["run_id"] for entry in index["reports"]] == ["new"]


def test_missing_report_refreshes_indexes_without_failure(tmp_path: Path) -> None:
    mod = _import_publisher()
    site = tmp_path / "site"

    rc = mod.main(
        [
            "--site-root",
            str(site),
            "--report-html",
            str(tmp_path / "missing.html"),
            "--kind",
            "premerge",
            "--run-id",
            "123",
            "--now",
            "2026-05-07T00:00:00Z",
        ]
    )

    assert rc == 0
    assert (site / "index.html").is_file()
    assert _load_index(site)["reports"] == []
