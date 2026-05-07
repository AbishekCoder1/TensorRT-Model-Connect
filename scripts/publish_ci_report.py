#!/usr/bin/env python3
"""Publish and prune static GitHub Actions CI report indexes.

This script manages the contents of a separate static report site. It copies a
single self-contained HTML report into a retention-scoped path, updates
``index.json``, renders human-readable index pages, and removes expired report
directories.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any


RETENTION_HOURS = {
    "nightly": 14 * 24,
    "premerge": 24,
}

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _parse_timestamp(value: str | None) -> datetime:
    if not value:
        return _utc_now()
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"invalid timestamp: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _safe_id(value: str, field: str) -> str:
    if not value or not _SAFE_ID_RE.match(value):
        raise ValueError(
            f"{field} must be non-empty and contain only letters, digits, '.', '_', or '-'"
        )
    return value


def _load_index(site_root: Path) -> dict[str, Any]:
    index_path = site_root / "index.json"
    if not index_path.is_file():
        return {"version": 1, "reports": []}
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid report index JSON: {index_path}") from exc
    reports = data.get("reports")
    if not isinstance(reports, list):
        raise ValueError(f"report index has no reports list: {index_path}")
    return {"version": int(data.get("version", 1)), "reports": reports}


def _write_index(site_root: Path, reports: list[dict[str, Any]], now: datetime) -> None:
    payload = {
        "version": 1,
        "generated_at": _format_timestamp(now),
        "reports": reports,
    }
    site_root.mkdir(parents=True, exist_ok=True)
    (site_root / "index.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _entry_dir(site_root: Path, entry: dict[str, Any]) -> Path | None:
    raw_path = str(entry.get("path", ""))
    path = PurePosixPath(raw_path)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 3:
        return None
    kind, run_id, filename = path.parts
    if kind not in RETENTION_HOURS or filename != "index.html":
        return None
    if not _SAFE_ID_RE.match(run_id):
        return None
    report_dir = site_root / kind / run_id
    try:
        report_dir.resolve().relative_to(site_root.resolve())
    except ValueError:
        return None
    return report_dir


def _prune_reports(
    site_root: Path, reports: list[dict[str, Any]], now: datetime
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for entry in reports:
        try:
            expires_at = _parse_timestamp(str(entry.get("expires_at", "")))
        except ValueError:
            expires_at = now
        if expires_at <= now:
            report_dir = _entry_dir(site_root, entry)
            if report_dir and report_dir.is_dir():
                shutil.rmtree(report_dir)
            continue
        kept.append(entry)
    return kept


def _status_class(conclusion: str) -> str:
    conclusion = conclusion.lower()
    if conclusion == "success":
        return "success"
    if conclusion in {"failure", "timed_out", "action_required"}:
        return "failure"
    if conclusion in {"cancelled", "skipped"}:
        return "muted"
    return "neutral"


def _esc(value: Any) -> str:
    return html.escape(str(value)) if value is not None else ""


def _short_sha(value: str) -> str:
    return value[:12] if value else ""


def _report_link(entry: dict[str, Any], index_kind: str | None) -> str:
    if index_kind:
        return f"{_esc(entry.get('run_id', ''))}/index.html"
    return _esc(entry.get("path", "#"))


def _render_index_page(
    *,
    title: str,
    reports: list[dict[str, Any]],
    base_url: str,
    index_kind: str | None = None,
) -> str:
    reports = sorted(reports, key=lambda item: str(item.get("created_at", "")), reverse=True)
    rows: list[str] = []
    for entry in reports:
        conclusion = str(entry.get("conclusion", "unknown"))
        run_url = str(entry.get("run_url", ""))
        run_id = _esc(entry.get("run_id", ""))
        pr_number = entry.get("pr_number")
        pr_text = f"#{_esc(pr_number)}" if pr_number else ""
        link = _report_link(entry, index_kind)
        run_link = (
            f'<a href="{_esc(run_url)}">workflow run</a>' if run_url else "workflow run"
        )
        rows.append(
            "<tr>"
            f'<td><span class="pill { _status_class(conclusion) }">'
            f"{_esc(conclusion or 'unknown')}</span></td>"
            f'<td><a href="{link}">{run_id}</a></td>'
            f"<td>{_esc(entry.get('kind', ''))}</td>"
            f"<td>{_esc(entry.get('workflow_name', ''))}</td>"
            f"<td>{_esc(entry.get('head_branch', ''))}</td>"
            f"<td>{_esc(_short_sha(str(entry.get('head_sha', ''))))}</td>"
            f"<td>{pr_text}</td>"
            f"<td>{_esc(entry.get('created_at', ''))}</td>"
            f"<td>{_esc(entry.get('expires_at', ''))}</td>"
            f"<td>{run_link}</td>"
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="10" class="empty">No retained reports.</td></tr>')

    base = base_url.rstrip("/")
    nav = (
        '<nav><a href="index.html">All</a> <a href="nightly/index.html">Nightly</a> '
        '<a href="premerge/index.html">Premerge</a></nav>'
        if index_kind is None
        else '<nav><a href="../index.html">All</a></nav>'
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f8fafc;
      color: #111827;
    }}
    body {{ margin: 0; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 48px; }}
    header {{ display: flex; align-items: flex-end; justify-content: space-between; gap: 24px; margin-bottom: 22px; }}
    h1 {{ font-size: 28px; line-height: 1.2; margin: 0 0 6px; }}
    p {{ margin: 0; color: #4b5563; }}
    nav {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    nav a {{ color: #1d4ed8; text-decoration: none; font-weight: 600; }}
    table {{ width: 100%; border-collapse: collapse; background: #ffffff; border: 1px solid #d1d5db; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 14px; vertical-align: top; }}
    th {{ background: #f3f4f6; color: #374151; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    tr:last-child td {{ border-bottom: 0; }}
    a {{ color: #1d4ed8; }}
    .pill {{ display: inline-block; min-width: 72px; padding: 3px 8px; border-radius: 999px; color: #fff; text-align: center; font-size: 12px; font-weight: 700; }}
    .success {{ background: #15803d; }}
    .failure {{ background: #b91c1c; }}
    .muted {{ background: #6b7280; }}
    .neutral {{ background: #4b5563; }}
    .empty {{ color: #6b7280; text-align: center; padding: 28px; }}
    @media (max-width: 900px) {{
      header {{ align-items: flex-start; flex-direction: column; }}
      table {{ display: block; overflow-x: auto; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>{_esc(title)}</h1>
        <p>Static CI reports served from { _esc(base) or "the configured report site" }.</p>
      </div>
      {nav}
    </header>
    <table>
      <thead>
        <tr>
          <th>Status</th>
          <th>Report</th>
          <th>Kind</th>
          <th>Workflow</th>
          <th>Branch</th>
          <th>Commit</th>
          <th>PR</th>
          <th>Created</th>
          <th>Expires</th>
          <th>GitHub</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </main>
</body>
</html>
"""


def _render_indexes(
    site_root: Path, reports: list[dict[str, Any]], base_url: str
) -> None:
    site_root.mkdir(parents=True, exist_ok=True)
    (site_root / "index.html").write_text(
        _render_index_page(
            title="TensorRT-Model-Connect CI Reports",
            reports=reports,
            base_url=base_url,
        ),
        encoding="utf-8",
    )
    for kind in ("nightly", "premerge"):
        kind_dir = site_root / kind
        kind_dir.mkdir(parents=True, exist_ok=True)
        kind_reports = [entry for entry in reports if entry.get("kind") == kind]
        (kind_dir / "index.html").write_text(
            _render_index_page(
                title=f"{kind.title()} CI Reports",
                reports=kind_reports,
                base_url=base_url,
                index_kind=kind,
            ),
            encoding="utf-8",
        )


def _build_entry(args: argparse.Namespace, now: datetime) -> dict[str, Any]:
    kind = args.kind
    run_id = _safe_id(str(args.run_id), "run-id")
    created_at = _parse_timestamp(args.created_at) if args.created_at else now
    expires_at = created_at + timedelta(hours=RETENTION_HOURS[kind])
    return {
        "kind": kind,
        "run_id": run_id,
        "run_number": str(args.run_number or ""),
        "workflow_name": str(args.workflow_name or ""),
        "event_name": str(args.event_name or ""),
        "conclusion": str(args.conclusion or "unknown"),
        "head_sha": str(args.head_sha or ""),
        "head_branch": str(args.head_branch or ""),
        "pr_number": str(args.pr_number or ""),
        "created_at": _format_timestamp(created_at),
        "expires_at": _format_timestamp(expires_at),
        "path": f"{kind}/{run_id}/index.html",
        "run_url": str(args.run_url or ""),
    }


def _write_github_output(path: Path | None, values: dict[str, str]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            handle.write(f"{key}={value}\n")


def publish(args: argparse.Namespace) -> dict[str, str]:
    now = _parse_timestamp(args.now) if args.now else _utc_now()
    site_root = args.site_root
    index_data = _load_index(site_root)
    reports = _prune_reports(site_root, list(index_data["reports"]), now)

    published = False
    report_url = args.base_url.rstrip("/") + "/" if args.base_url else ""
    if not args.prune_only:
        if args.report_html and args.report_html.is_file():
            entry = _build_entry(args, now)
            report_dir = site_root / entry["kind"] / entry["run_id"]
            report_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.report_html, report_dir / "index.html")
            reports = [
                item
                for item in reports
                if not (
                    item.get("kind") == entry["kind"]
                    and str(item.get("run_id")) == entry["run_id"]
                )
            ]
            reports.append(entry)
            published = True
            report_url = args.base_url.rstrip("/") + "/" + entry["path"]
        else:
            print("No report HTML found; pruning and refreshing indexes only.", file=sys.stderr)

    reports = sorted(reports, key=lambda item: str(item.get("created_at", "")), reverse=True)
    _write_index(site_root, reports, now)
    _render_indexes(site_root, reports, args.base_url)
    output = {"published": str(published).lower(), "report_url": report_url}
    _write_github_output(args.github_output, output)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish a CI report to a separate static report site."
    )
    parser.add_argument("--site-root", type=Path, required=True)
    parser.add_argument("--report-html", type=Path)
    parser.add_argument("--kind", choices=sorted(RETENTION_HOURS))
    parser.add_argument("--run-id")
    parser.add_argument("--run-number")
    parser.add_argument("--run-url")
    parser.add_argument("--workflow-name")
    parser.add_argument("--event-name")
    parser.add_argument("--conclusion")
    parser.add_argument("--head-sha")
    parser.add_argument("--head-branch")
    parser.add_argument("--pr-number")
    parser.add_argument("--created-at")
    parser.add_argument("--now")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--prune-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.prune_only and (not args.kind or not args.run_id):
        parser.error("--kind and --run-id are required unless --prune-only is used")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        publish(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
