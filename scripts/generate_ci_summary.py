#!/usr/bin/env python3
"""Generate a GitHub Actions Markdown summary for CI artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


_STATUS_ORDER = ("fail", "error", "skip", "pass")
_PASS_STATUSES = {"pass", "passed", "success", "succeeded"}
_METRIC_PRIORITY = (
    "logit_cosine_p5",
    "token_agreement_rate",
    "miou",
    "psnr",
    "ssim",
    "mel_distance",
    "pixel_accuracy",
    "normalized_text_edit_distance",
)


def _md(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\n", " ").replace("|", r"\|")


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _load_results(artifacts_dir: Path) -> list[dict[str, Any]]:
    if not artifacts_dir.is_dir():
        return []
    results: list[dict[str, Any]] = []
    for result_path in sorted(artifacts_dir.rglob("result.json")):
        try:
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: skipping {result_path}: {exc}", file=sys.stderr)
    return results


def _status(result: dict[str, Any]) -> str:
    return str(result.get("status") or "error").lower()


def _key_metric(result: dict[str, Any]) -> str:
    stages = result.get("stages", {}) or {}
    for stage_data in stages.values():
        metrics = stage_data.get("metrics", {}) if isinstance(stage_data, dict) else {}
        for key in _METRIC_PRIORITY:
            if key in metrics:
                metric = metrics[key]
                value = metric.get("value", metric) if isinstance(metric, dict) else metric
                return f"{key}={_format_value(value)}"
    return ""


def _total_time_seconds(result: dict[str, Any]) -> float | None:
    timing = result.get("timing", {}) or {}
    if not isinstance(timing, dict) or not timing:
        return None
    total = 0.0
    saw_value = False
    for value in timing.values():
        try:
            total += float(value)
        except (TypeError, ValueError):
            continue
        saw_value = True
    return total if saw_value else None


def _total_time(result: dict[str, Any]) -> str:
    total = _total_time_seconds(result)
    return "" if total is None else f"{total:.1f}s"


def _failure_note(result: dict[str, Any]) -> str:
    failure_type = result.get("failure_type")
    if failure_type:
        return str(failure_type)
    for stage_name, stage_data in (result.get("stages", {}) or {}).items():
        if not isinstance(stage_data, dict):
            continue
        status = str(stage_data.get("status", "")).lower()
        if status and status not in _PASS_STATUSES:
            message = stage_data.get("message")
            if message:
                return f"{stage_name}: {message}"
            return str(stage_name)
    return ""


def _case_row(result: dict[str, Any], include_failure: bool = False) -> str:
    config = result.get("case_config", {}) or {}
    cols = [
        _md(result.get("case_name", "unknown")),
        _md(config.get("family", "")),
        _md(config.get("task_strategy", "")),
        _md(_status(result)),
        _md(_key_metric(result)),
        _md(_total_time(result)),
    ]
    if include_failure:
        cols.insert(4, _md(_failure_note(result)))
    return "| " + " | ".join(cols) + " |"


def _render_table(headers: list[str], rows: list[str]) -> list[str]:
    if not rows:
        return []
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *rows,
    ]


def _sort_key(result: dict[str, Any]) -> tuple[int, float, str]:
    status = _status(result)
    status_rank = _STATUS_ORDER.index(status) if status in _STATUS_ORDER else 1
    total = _total_time_seconds(result)
    return (status_rank, -(total or 0.0), str(result.get("case_name", "")))


def render_summary(
    *,
    results: list[dict[str, Any]],
    mode: str,
    report_path: Path,
    html_artifact_name: str,
    full_artifact_name: str,
    run_url: str,
    max_rows: int,
) -> str:
    lines: list[str] = [f"## TensorRT-Model-Connect CI Summary ({mode})", ""]

    if run_url:
        lines.append(f"Artifacts: [open this workflow run]({run_url})")
    else:
        lines.append("Artifacts: open this workflow run")
    lines.append(f"HTML report artifact: `{html_artifact_name}`")
    lines.append(f"Full debug artifact: `{full_artifact_name}`")
    if report_path.is_file():
        lines.append("HTML report contents: `e2e_report.html`")
    else:
        lines.append("HTML report contents: not generated for this run")
    lines.append("")

    if not results:
        lines.append("No E2E `result.json` files were found for this run.")
        lines.append("")
        return "\n".join(lines)

    counts: dict[str, int] = {}
    for result in results:
        counts[_status(result)] = counts.get(_status(result), 0) + 1

    lines.append("### E2E Result Counts")
    count_rows = [
        f"| {_md(status)} | {counts[status]} |"
        for status in sorted(counts, key=lambda s: (_STATUS_ORDER.index(s) if s in _STATUS_ORDER else 1, s))
    ]
    lines.extend(_render_table(["Status", "Count"], count_rows))
    lines.append("")

    failures = [r for r in results if _status(r) not in _PASS_STATUSES]
    if failures:
        lines.append("### Failures")
        rows = [_case_row(r, include_failure=True) for r in sorted(failures, key=_sort_key)[:max_rows]]
        lines.extend(
            _render_table(
                ["Model", "Family", "Task", "Status", "Failure", "Key Metric", "Time"],
                rows,
            )
        )
        if len(failures) > max_rows:
            lines.append(f"\nShowing {max_rows} of {len(failures)} non-passing cases.")
        lines.append("")

    timed = [r for r in results if _total_time_seconds(r) is not None]
    if timed:
        lines.append("### Slowest E2E Cases")
        rows = [
            _case_row(r)
            for r in sorted(timed, key=lambda item: _total_time_seconds(item) or 0.0, reverse=True)[:10]
        ]
        lines.extend(_render_table(["Model", "Family", "Task", "Status", "Key Metric", "Time"], rows))
        lines.append("")

    lines.append("### All E2E Model Status")
    rows = [_case_row(r) for r in sorted(results, key=_sort_key)]
    lines.extend(_render_table(["Model", "Family", "Task", "Status", "Key Metric", "Time"], rows))
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a GitHub Actions CI summary.")
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--html-artifact-name", required=True)
    parser.add_argument("--full-artifact-name", required=True)
    parser.add_argument("--run-url", default="")
    parser.add_argument("--max-rows", type=int, default=40)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = _load_results(args.artifacts_dir)
    print(
        render_summary(
            results=results,
            mode=args.mode,
            report_path=args.report_path,
            html_artifact_name=args.html_artifact_name,
            full_artifact_name=args.full_artifact_name,
            run_url=args.run_url,
            max_rows=args.max_rows,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
