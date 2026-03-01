#!/usr/bin/env python3
"""Generate a self-contained HTML report from E2E test artifacts.

Reads result.json files produced by the unified E2E harness and assembles
a single HTML page with summary dashboard, per-model details, embedded
media (images, audio, video frames), and reproduction commands.

Usage:
    python scripts/generate_e2e_report.py \\
      --artifacts-dir /tmp/e2e_artifacts/artifacts \\
      -o /tmp/e2e_artifacts/e2e_report.html \\
      [--project-dir .] \\
      [--title "E2E Report"]
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Maximum file size to embed inline (10 MB).
_MAX_EMBED_BYTES = 10 * 1024 * 1024

# Number of evenly-spaced frames to embed for diffusion models.
_MAX_DIFFUSION_FRAMES = 6

# ---------------------------------------------------------------------------
# Modality classification
# ---------------------------------------------------------------------------

_TASK_STRATEGY_TO_MODALITY = {
    "text_generation_causal": "text",
    "vision_language_generation": "vl",
    "diffusion_media_generation": "diffusion",
    "text_to_audio": "audio",
    "speech_to_text": "audio",
    "speech_to_speech": "audio",
    "segmentation": "segmentation",
    "prompted_segmentation": "segmentation",
    "encoder_only_nlp": "generic",
    "embedding": "generic",
    "reranking": "generic",
}


def classify_modality(result: Dict[str, Any]) -> str:
    """Return a modality string for the given result dict."""
    ts = (result.get("case_config") or {}).get("task_strategy", "")
    return _TASK_STRATEGY_TO_MODALITY.get(ts, "generic")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_all_results(artifacts_dir: Path) -> List[Dict[str, Any]]:
    """Walk *artifacts_dir* and parse every ``result.json`` found."""
    results: List[Dict[str, Any]] = []
    if not artifacts_dir.is_dir():
        return results
    for p in sorted(artifacts_dir.iterdir()):
        rj = p / "result.json"
        if rj.is_file():
            try:
                data = json.loads(rj.read_text(encoding="utf-8"))
                # Stash the directory so renderers can find artifacts.
                data["_artifact_dir"] = str(p)
                results.append(data)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"WARNING: skipping {rj}: {exc}", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# File embedding helpers
# ---------------------------------------------------------------------------


def encode_file_base64(path: Path, mime: str) -> Optional[str]:
    """Return a ``data:`` URI for *path*, or ``None`` if too large / missing."""
    if not path.is_file():
        return None
    if path.stat().st_size > _MAX_EMBED_BYTES:
        return None
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _mime_for_ext(ext: str) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".wav": "audio/wav",
        ".gif": "image/gif",
    }.get(ext.lower(), "application/octet-stream")


# ---------------------------------------------------------------------------
# HTML primitives
# ---------------------------------------------------------------------------

_STATUS_COLORS = {
    "pass": "#22c55e",
    "fail": "#ef4444",
    "skip": "#eab308",
    "error": "#f97316",
    "passed": "#22c55e",
    "failed": "#ef4444",
    "skipped": "#eab308",
}


def _badge(status: str) -> str:
    color = _STATUS_COLORS.get(status, "#6b7280")
    return (
        f'<span class="badge" style="background:{color}">'
        f"{html.escape(status.upper())}</span>"
    )


def _esc(text: Any) -> str:
    return html.escape(str(text)) if text is not None else ""


def _code_block(text: str, block_id: str) -> str:
    """Render a dark code block with a copy button."""
    return (
        f'<div class="code-wrap">'
        f'<button class="copy-btn" onclick="copyCmd(\'{block_id}\')">Copy</button>'
        f'<pre id="{block_id}"><code>{_esc(text)}</code></pre>'
        f"</div>"
    )


def _select_frames(frame_paths: List[Path], max_frames: int) -> List[Path]:
    """Pick *max_frames* evenly-spaced frames from *frame_paths*."""
    n = len(frame_paths)
    if n <= max_frames:
        return frame_paths
    indices = [int(i * (n - 1) / (max_frames - 1)) for i in range(max_frames)]
    return [frame_paths[i] for i in indices]


def _list_frames_in_dir(dir_path: Path) -> List[Path]:
    """Return sorted frame/image files from *dir_path*."""
    frame_paths = sorted(dir_path.glob("frame_*.png"))
    if frame_paths:
        return frame_paths
    # Backward compatibility for runs that save a single non-frame_*.png image.
    ext_globs = ("*.png", "*.jpg", "*.jpeg")
    files: List[Path] = []
    for pattern in ext_globs:
        files.extend(sorted(dir_path.glob(pattern)))
    return files


def _resolve_frame_paths(
    frame_refs: Any,
    art_dir: Path,
    fallback_dir_name: str,
) -> List[Path]:
    """Resolve frame refs (file(s) or directory path(s)) into concrete files."""
    refs: List[str] = []
    if isinstance(frame_refs, str):
        refs = [frame_refs]
    elif isinstance(frame_refs, list):
        refs = [str(ref) for ref in frame_refs if ref]

    frame_paths: List[Path] = []
    for ref in refs:
        p = Path(ref)
        if not p.is_absolute():
            p = art_dir / ref
        if p.is_dir():
            frame_paths.extend(_list_frames_in_dir(p))
        elif p.is_file():
            frame_paths.append(p)

    if not frame_paths:
        fallback_dir = art_dir / fallback_dir_name
        if fallback_dir.is_dir():
            frame_paths = _list_frames_in_dir(fallback_dir)

    # De-duplicate while preserving order.
    deduped: List[Path] = []
    seen: set[str] = set()
    for fp in frame_paths:
        key = str(fp)
        if key not in seen:
            seen.add(key)
            deduped.append(fp)
    return deduped


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------


def _render_metrics_table(stages: Dict[str, Any]) -> str:
    """Render a table of all metrics across all stages."""
    rows: List[str] = []
    for stage_name, stage_data in stages.items():
        metrics = stage_data.get("metrics", {})
        for metric_name, m in metrics.items():
            if isinstance(m, dict):
                value = m.get("value", "")
                threshold = m.get("threshold")
                operator = m.get("operator", "")
                passed = m.get("passed", True)
                note = m.get("note", "")
            else:
                value = m
                threshold = None
                operator = ""
                passed = True
                note = ""
            icon = "&#10003;" if passed else "&#10007;"
            icon_cls = "pass-icon" if passed else "fail-icon"
            row_cls = "metric-pass" if passed else "metric-fail"
            thr_str = f"{threshold}" if threshold is not None else "&mdash;"
            rows.append(
                f"<tr class='{row_cls}'>"
                f"<td>{_esc(stage_name)}</td>"
                f"<td>{_esc(metric_name)}</td>"
                f"<td>{_format_value(value)}</td>"
                f"<td>{thr_str}</td>"
                f"<td>{_esc(operator)}</td>"
                f"<td class='{icon_cls}'>{icon}</td>"
                f"<td>{_esc(note)}</td>"
                f"</tr>"
            )
    if not rows:
        return "<p><em>No metrics available.</em></p>"
    return (
        '<table class="metrics-table">'
        "<thead><tr>"
        "<th>Stage</th><th>Metric</th><th>Value</th>"
        "<th>Threshold</th><th>Op</th><th>Pass</th><th>Note</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )


def _format_value(v: Any) -> str:
    if isinstance(v, float):
        if abs(v) < 0.001 and v != 0:
            return f"{v:.2e}"
        return f"{v:.4f}"
    return _esc(v)


# ---------------------------------------------------------------------------
# Timing table
# ---------------------------------------------------------------------------


def _render_timing_table(timing: Dict[str, float]) -> str:
    if not timing:
        return ""
    rows = []
    total = 0.0
    for phase, secs in timing.items():
        s = float(secs) if secs is not None else 0.0
        total += s
        rows.append(f"<tr><td>{_esc(phase)}</td><td>{s:.2f}s</td></tr>")
    rows.append(f"<tr class='total-row'><td>Total</td><td>{total:.2f}s</td></tr>")
    return (
        '<table class="timing-table">'
        "<thead><tr><th>Phase</th><th>Time</th></tr></thead>"
        "<tbody>" + "\n".join(rows) + "</tbody></table>"
    )


# ---------------------------------------------------------------------------
# Repro commands
# ---------------------------------------------------------------------------

_CMD_COUNTER = 0


def _next_cmd_id() -> str:
    global _CMD_COUNTER  # noqa: PLW0603
    _CMD_COUNTER += 1
    return f"cmd_{_CMD_COUNTER}"


def _render_repro_commands(repro: Dict[str, str]) -> str:
    if not repro:
        return ""
    parts = ['<div class="repro-section"><h4>Reproduction Commands</h4>']
    for label, cmd in repro.items():
        cid = _next_cmd_id()
        parts.append(f"<p><strong>{_esc(label)}</strong></p>")
        parts.append(_code_block(cmd, cid))
    parts.append("</div>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Text comparison
# ---------------------------------------------------------------------------


def _render_text_comparison(
    trt_text: Optional[str], ref_text: Optional[str]
) -> str:
    if trt_text is None and ref_text is None:
        return ""
    return (
        '<div class="text-compare">'
        '<div class="text-col">'
        "<h4>TRT Output</h4>"
        f"<pre>{_esc(trt_text or '(none)')}</pre>"
        "</div>"
        '<div class="text-col">'
        "<h4>Reference Output</h4>"
        f"<pre>{_esc(ref_text or '(none)')}</pre>"
        "</div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Modality renderers
# ---------------------------------------------------------------------------


def _get_stage_text(stage_outputs: Dict[str, Any], prefix: str) -> Optional[str]:
    """Extract .text from the first stage output matching *prefix*."""
    for key, val in stage_outputs.items():
        if key.startswith(prefix) and isinstance(val, dict):
            t = val.get("text")
            if t is not None:
                return str(t)
    return None


def render_text_model(result: Dict[str, Any]) -> str:
    """Render detail section for a text-generation model."""
    cc = result.get("case_config", {})
    prompt = (cc.get("inputs") or {}).get("prompt", "")
    stage_outputs = result.get("stage_outputs", {})
    trt_text = _get_stage_text(stage_outputs, "trt_")
    ref_text = _get_stage_text(stage_outputs, "ref_")

    parts = []
    if prompt:
        parts.append(f"<p><strong>Prompt:</strong> {_esc(prompt)}</p>")
    parts.append(_render_text_comparison(trt_text, ref_text))
    parts.append(_render_metrics_table(result.get("stages", {})))
    parts.append(_render_repro_commands(result.get("repro_commands", {})))
    parts.append(_render_timing_table(result.get("timing", {})))
    return "\n".join(parts)


def render_vl_model(result: Dict[str, Any], project_dir: Optional[Path]) -> str:
    """Render detail section for a vision-language model."""
    cc = result.get("case_config", {})
    inputs = cc.get("inputs") or {}
    prompt = inputs.get("prompt", "")
    image_rel = inputs.get("image", "")

    stage_outputs = result.get("stage_outputs", {})
    trt_text = _get_stage_text(stage_outputs, "trt_")
    ref_text = _get_stage_text(stage_outputs, "ref_")

    parts = []

    # Embed input image
    if image_rel and project_dir:
        img_path = project_dir / image_rel
        uri = encode_file_base64(img_path, _mime_for_ext(img_path.suffix))
        if uri:
            parts.append(
                f'<p><strong>Input Image:</strong></p>'
                f'<img src="{uri}" class="preview-img" />'
            )
        else:
            parts.append(f"<p><em>Image not found: {_esc(image_rel)}</em></p>")

    if prompt:
        parts.append(f"<p><strong>Prompt:</strong> {_esc(prompt)}</p>")
    parts.append(_render_text_comparison(trt_text, ref_text))
    parts.append(_render_metrics_table(result.get("stages", {})))
    parts.append(_render_repro_commands(result.get("repro_commands", {})))
    parts.append(_render_timing_table(result.get("timing", {})))
    return "\n".join(parts)


def render_diffusion_model(result: Dict[str, Any]) -> str:
    """Render detail section for a diffusion model."""
    art_dir = Path(result.get("_artifact_dir", ""))
    artifacts = result.get("artifacts", {})

    parts = []

    # TRT frames gallery
    trt_frame_paths = _resolve_frame_paths(
        artifacts.get("trt_frames", []),
        art_dir=art_dir,
        fallback_dir_name="frames",
    )

    if trt_frame_paths:
        selected = _select_frames(trt_frame_paths, _MAX_DIFFUSION_FRAMES)
        parts.append("<h4>TRT Generated Frames</h4>")
        parts.append('<div class="frame-gallery">')
        for fp in selected:
            uri = encode_file_base64(fp, "image/png")
            if uri:
                parts.append(f'<img src="{uri}" class="frame-img" />')
            else:
                parts.append(f"<span class='missing'>Frame too large</span>")
        parts.append("</div>")

    # Reference frames (side-by-side if available)
    ref_frame_paths = _resolve_frame_paths(
        artifacts.get("ref_frames", []),
        art_dir=art_dir,
        fallback_dir_name="ref_frames",
    )

    if ref_frame_paths:
        selected_ref = _select_frames(ref_frame_paths, _MAX_DIFFUSION_FRAMES)
        parts.append("<h4>Reference Frames</h4>")
        parts.append('<div class="frame-gallery">')
        for fp in selected_ref:
            uri = encode_file_base64(fp, "image/png")
            if uri:
                parts.append(f'<img src="{uri}" class="frame-img" />')
        parts.append("</div>")

    parts.append(_render_metrics_table(result.get("stages", {})))
    parts.append(_render_repro_commands(result.get("repro_commands", {})))
    parts.append(_render_timing_table(result.get("timing", {})))
    return "\n".join(parts)


def render_audio_model(result: Dict[str, Any]) -> str:
    """Render detail section for an audio model (Whisper, Bark, etc.)."""
    art_dir = Path(result.get("_artifact_dir", ""))
    artifacts = result.get("artifacts", {})
    stage_outputs = result.get("stage_outputs", {})
    cc = result.get("case_config", {})
    task_strategy = cc.get("task_strategy", "")

    parts = []

    # For speech_to_text, show transcript comparison
    if task_strategy == "speech_to_text":
        trt_text = _get_stage_text(stage_outputs, "trt_")
        ref_text = _get_stage_text(stage_outputs, "ref_")
        if trt_text or ref_text:
            parts.append("<h4>Transcript Comparison</h4>")
            parts.append(_render_text_comparison(trt_text, ref_text))

    # Embed TRT audio
    trt_wav = artifacts.get("trt_wav", "")
    if trt_wav:
        wav_path = art_dir / trt_wav
        uri = encode_file_base64(wav_path, "audio/wav")
        if uri:
            parts.append("<h4>TRT Audio</h4>")
            parts.append(f'<audio controls src="{uri}"></audio>')
        else:
            parts.append("<p><em>TRT WAV not found or too large.</em></p>")

    # Embed reference audio
    ref_wav = artifacts.get("ref_wav", "")
    if ref_wav:
        wav_path = art_dir / ref_wav
        uri = encode_file_base64(wav_path, "audio/wav")
        if uri:
            parts.append("<h4>Reference Audio</h4>")
            parts.append(f'<audio controls src="{uri}"></audio>')

    parts.append(_render_metrics_table(result.get("stages", {})))
    parts.append(_render_repro_commands(result.get("repro_commands", {})))
    parts.append(_render_timing_table(result.get("timing", {})))
    return "\n".join(parts)


def render_segmentation_model(
    result: Dict[str, Any], project_dir: Optional[Path]
) -> str:
    """Render detail section for a segmentation model."""
    art_dir = Path(result.get("_artifact_dir", ""))
    artifacts = result.get("artifacts", {})
    cc = result.get("case_config", {})
    inputs = cc.get("inputs") or {}
    image_rel = inputs.get("image", "")

    parts = []

    # Input image
    if image_rel and project_dir:
        img_path = project_dir / image_rel
        uri = encode_file_base64(img_path, _mime_for_ext(img_path.suffix))
        if uri:
            parts.append(
                f'<p><strong>Input Image:</strong></p>'
                f'<img src="{uri}" class="preview-img" />'
            )

    # Segmentation map
    seg_map = artifacts.get("trt_segmentation_map", "")
    if seg_map:
        seg_path = art_dir / seg_map
        uri = encode_file_base64(seg_path, "image/png")
        if uri:
            parts.append("<h4>TRT Segmentation Map</h4>")
            parts.append(f'<img src="{uri}" class="preview-img" />')

    ref_seg_map = artifacts.get("ref_segmentation_map", "")
    if ref_seg_map:
        seg_path = art_dir / ref_seg_map
        uri = encode_file_base64(seg_path, "image/png")
        if uri:
            parts.append("<h4>Reference Segmentation Map</h4>")
            parts.append(f'<img src="{uri}" class="preview-img" />')

    parts.append(_render_metrics_table(result.get("stages", {})))
    parts.append(_render_repro_commands(result.get("repro_commands", {})))
    parts.append(_render_timing_table(result.get("timing", {})))
    return "\n".join(parts)


def render_generic_model(result: Dict[str, Any]) -> str:
    """Render detail section for generic models (BERT, embedding, etc.)."""
    cc = result.get("case_config", {})
    prompt = (cc.get("inputs") or {}).get("prompt", "")
    stage_outputs = result.get("stage_outputs", {})
    trt_text = _get_stage_text(stage_outputs, "trt_")
    ref_text = _get_stage_text(stage_outputs, "ref_")

    parts = []
    if prompt:
        parts.append(f"<p><strong>Prompt:</strong> {_esc(prompt)}</p>")
    if trt_text or ref_text:
        parts.append(_render_text_comparison(trt_text, ref_text))
    parts.append(_render_metrics_table(result.get("stages", {})))
    parts.append(_render_repro_commands(result.get("repro_commands", {})))
    parts.append(_render_timing_table(result.get("timing", {})))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Per-model collapsible section
# ---------------------------------------------------------------------------


def render_model_section(
    result: Dict[str, Any],
    project_dir: Optional[Path],
) -> str:
    """Render a single collapsible ``<details>`` for one model."""
    name = result.get("case_name", "unknown")
    status = result.get("status", "error")
    cc = result.get("case_config", {})
    family = cc.get("family", "")
    task_strategy = cc.get("task_strategy", "")
    hf_id = cc.get("hf_id", "")

    modality = classify_modality(result)
    badge = _badge(status)

    header = (
        f'<details id="model-{_esc(name)}">'
        f"<summary>{badge} <strong>{_esc(name)}</strong>"
        f" &mdash; {_esc(family)} / {_esc(task_strategy)}"
    )
    if hf_id:
        header += f" <small>({_esc(hf_id)})</small>"
    header += "</summary>"

    # Failure info
    body_parts = []
    failure_type = result.get("failure_type")
    if failure_type:
        body_parts.append(
            f'<p class="failure-info">Failure type: '
            f"<strong>{_esc(failure_type)}</strong></p>"
        )

    # Dispatch to modality renderer
    if modality == "text":
        body_parts.append(render_text_model(result))
    elif modality == "vl":
        body_parts.append(render_vl_model(result, project_dir))
    elif modality == "diffusion":
        body_parts.append(render_diffusion_model(result))
    elif modality == "audio":
        body_parts.append(render_audio_model(result))
    elif modality == "segmentation":
        body_parts.append(render_segmentation_model(result, project_dir))
    else:
        body_parts.append(render_generic_model(result))

    body = "\n".join(body_parts)
    return f'{header}\n<div class="model-body">{body}</div>\n</details>'


# ---------------------------------------------------------------------------
# Summary dashboard
# ---------------------------------------------------------------------------


def _key_metric(result: Dict[str, Any]) -> str:
    """Extract the single most representative metric for the summary row."""
    stages = result.get("stages", {})
    # Priority: logit_cosine_p5, token_agreement_rate, miou, psnr, mel_distance
    priority = [
        "logit_cosine_p5",
        "token_agreement_rate",
        "miou",
        "psnr",
        "ssim",
        "mel_distance",
        "pixel_accuracy",
        "normalized_text_edit_distance",
    ]
    for stage_data in stages.values():
        metrics = stage_data.get("metrics", {})
        for key in priority:
            if key in metrics:
                m = metrics[key]
                val = m.get("value", m) if isinstance(m, dict) else m
                return f"{key}={_format_value(val)}"
    return ""


def _total_time(result: Dict[str, Any]) -> str:
    timing = result.get("timing", {})
    if not timing:
        return ""
    total = sum(float(v) for v in timing.values() if v is not None)
    return f"{total:.1f}s"


def render_summary_dashboard(results: List[Dict[str, Any]]) -> str:
    """Render the top-of-page summary table with counters and filters."""
    counts: Dict[str, int] = {"pass": 0, "fail": 0, "skip": 0, "error": 0}
    for r in results:
        s = r.get("status", "error")
        counts[s] = counts.get(s, 0) + 1

    counters = (
        f'<div class="counters">'
        f'<span class="counter pass-counter">{counts["pass"]} Passed</span>'
        f'<span class="counter fail-counter">{counts["fail"]} Failed</span>'
        f'<span class="counter skip-counter">{counts["skip"]} Skipped</span>'
        f'<span class="counter error-counter">{counts["error"]} Error</span>'
        f'<span class="counter total-counter">{len(results)} Total</span>'
        f"</div>"
    )

    filters = (
        '<div class="filters">'
        '<input type="text" id="search-box" placeholder="Search models..." '
        'oninput="filterModels()" />'
        '<select id="status-filter" onchange="filterModels()">'
        '<option value="">All</option>'
        '<option value="pass">Pass</option>'
        '<option value="fail">Fail</option>'
        '<option value="skip">Skip</option>'
        '<option value="error">Error</option>'
        "</select>"
        "</div>"
    )

    rows: List[str] = []
    for r in results:
        name = r.get("case_name", "unknown")
        status = r.get("status", "error")
        cc = r.get("case_config", {})
        family = cc.get("family", "")
        task_strategy = cc.get("task_strategy", "")
        km = _key_metric(r)
        tt = _total_time(r)
        rows.append(
            f'<tr class="summary-row" data-status="{_esc(status)}" '
            f'data-name="{_esc(name.lower())}">'
            f'<td><a href="#model-{_esc(name)}">{_esc(name)}</a></td>'
            f"<td>{_esc(family)}</td>"
            f"<td>{_esc(task_strategy)}</td>"
            f"<td>{_badge(status)}</td>"
            f"<td>{km}</td>"
            f"<td>{tt}</td>"
            f"</tr>"
        )

    table = (
        '<table class="summary-table" id="summary-table">'
        "<thead><tr>"
        "<th>Model</th><th>Family</th><th>Task Strategy</th>"
        "<th>Status</th><th>Key Metric</th><th>Time</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )

    return f'<section class="dashboard">{counters}\n{filters}\n{table}</section>'


# ---------------------------------------------------------------------------
# Environment section
# ---------------------------------------------------------------------------


def render_env_section(results: List[Dict[str, Any]]) -> str:
    """Render environment info from the first result that has it."""
    for r in results:
        env = r.get("env_fingerprint", {})
        if env:
            items = []
            for k, v in env.items():
                if k == "timestamp":
                    continue
                items.append(f"<li><strong>{_esc(k)}:</strong> {_esc(v)}</li>")
            return (
                '<section class="env-section">'
                "<h2>Environment</h2>"
                f'<ul class="env-list">{"".join(items)}</ul>'
                "</section>"
            )
    return ""


# ---------------------------------------------------------------------------
# Full report assembly
# ---------------------------------------------------------------------------

_CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
  Helvetica, Arial, sans-serif; background: #f8f9fa; color: #1a1a2e;
  max-width: 1400px; margin: 0 auto; padding: 20px; }
h1 { margin-bottom: 8px; }
h2 { margin: 24px 0 12px; }
h4 { margin: 12px 0 6px; }
.subtitle { color: #6b7280; margin-bottom: 20px; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
  color: #fff; font-size: 0.8em; font-weight: 600; }
.counters { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }
.counter { padding: 6px 16px; border-radius: 8px; font-weight: 600;
  font-size: 0.95em; }
.pass-counter { background: #dcfce7; color: #166534; }
.fail-counter { background: #fee2e2; color: #991b1b; }
.skip-counter { background: #fef9c3; color: #854d0e; }
.error-counter { background: #ffedd5; color: #9a3412; }
.total-counter { background: #e0e7ff; color: #3730a3; }
.filters { display: flex; gap: 8px; margin-bottom: 12px; }
#search-box { padding: 6px 12px; border: 1px solid #d1d5db; border-radius: 6px;
  flex: 1; max-width: 300px; }
#status-filter { padding: 6px 12px; border: 1px solid #d1d5db; border-radius: 6px; }
.summary-table, .metrics-table, .timing-table { width: 100%;
  border-collapse: collapse; margin: 8px 0; font-size: 0.9em; }
.summary-table th, .metrics-table th, .timing-table th { background: #1e293b;
  color: #fff; padding: 8px 12px; text-align: left; }
.summary-table td, .metrics-table td, .timing-table td { padding: 6px 12px;
  border-bottom: 1px solid #e2e8f0; }
.summary-table tbody tr:hover { background: #f1f5f9; }
.metric-pass { background: #f0fdf4; }
.metric-fail { background: #fef2f2; }
.pass-icon { color: #16a34a; font-weight: bold; }
.fail-icon { color: #dc2626; font-weight: bold; }
.total-row td { font-weight: 700; border-top: 2px solid #1e293b; }
details { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
  margin: 8px 0; }
details[open] { border-color: #94a3b8; }
summary { padding: 12px 16px; cursor: pointer; font-size: 1em; }
summary:hover { background: #f8fafc; }
.model-body { padding: 12px 16px; }
.failure-info { color: #dc2626; margin-bottom: 8px; }
.text-compare { display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
  margin: 8px 0; }
.text-col pre { background: #f1f5f9; padding: 12px; border-radius: 6px;
  white-space: pre-wrap; word-break: break-word; font-size: 0.85em;
  max-height: 300px; overflow-y: auto; }
.code-wrap { position: relative; margin: 6px 0; }
.code-wrap pre { background: #1e293b; color: #e2e8f0; padding: 12px;
  border-radius: 6px; overflow-x: auto; font-size: 0.85em; }
.copy-btn { position: absolute; top: 6px; right: 6px; background: #475569;
  color: #fff; border: none; border-radius: 4px; padding: 2px 8px;
  cursor: pointer; font-size: 0.75em; }
.copy-btn:hover { background: #64748b; }
.frame-gallery { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; }
.frame-img { max-width: 220px; max-height: 180px; border-radius: 4px;
  border: 1px solid #e2e8f0; }
.preview-img { max-width: 400px; max-height: 300px; border-radius: 6px;
  margin: 6px 0; border: 1px solid #e2e8f0; }
audio { margin: 6px 0; }
.missing { color: #9ca3af; font-style: italic; }
.env-section ul { list-style: none; columns: 2; }
.env-section li { padding: 2px 0; font-size: 0.9em; }
.repro-section { margin-top: 12px; }
@media (max-width: 768px) {
  .text-compare { grid-template-columns: 1fr; }
  .env-section ul { columns: 1; }
}
"""

_JS = """\
function copyCmd(id) {
  var el = document.getElementById(id);
  if (!el) return;
  var text = el.textContent || el.innerText;
  navigator.clipboard.writeText(text).then(function() {
    var btn = el.parentElement.querySelector('.copy-btn');
    if (btn) { btn.textContent = 'Copied!'; setTimeout(function() {
      btn.textContent = 'Copy'; }, 1500); }
  });
}
function filterModels() {
  var q = (document.getElementById('search-box').value || '').toLowerCase();
  var s = document.getElementById('status-filter').value;
  var rows = document.querySelectorAll('.summary-row');
  for (var i = 0; i < rows.length; i++) {
    var name = rows[i].getAttribute('data-name') || '';
    var status = rows[i].getAttribute('data-status') || '';
    var show = (!q || name.indexOf(q) >= 0) && (!s || status === s);
    rows[i].style.display = show ? '' : 'none';
  }
}
"""


def render_report(
    results: List[Dict[str, Any]],
    title: str = "E2E Test Report",
    project_dir: Optional[Path] = None,
) -> str:
    """Assemble the full self-contained HTML report."""
    # Reset command counter for deterministic output.
    global _CMD_COUNTER  # noqa: PLW0603
    _CMD_COUNTER = 0

    parts: List[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en"><head>')
    parts.append('<meta charset="utf-8" />')
    parts.append(
        '<meta name="viewport" content="width=device-width, initial-scale=1" />'
    )
    parts.append(f"<title>{_esc(title)}</title>")
    parts.append(f"<style>{_CSS}</style>")
    parts.append("</head><body>")
    parts.append(f"<h1>{_esc(title)}</h1>")

    # Timestamp
    if results:
        ts = results[0].get("timestamp", "")
        if ts:
            parts.append(f'<p class="subtitle">Generated from run at {_esc(ts)}</p>')

    # Environment
    parts.append(render_env_section(results))

    # Summary dashboard
    parts.append("<h2>Summary</h2>")
    parts.append(render_summary_dashboard(results))

    # Per-model details
    parts.append("<h2>Model Details</h2>")
    for r in results:
        parts.append(render_model_section(r, project_dir))

    parts.append(f"<script>{_JS}</script>")
    parts.append("</body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML report from E2E artifacts."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Root directory containing per-model result directories.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output HTML file path.",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help="Project root for resolving relative image paths.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="E2E Test Report",
        help="Report title (shown in header and <title>).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    results = load_all_results(args.artifacts_dir)
    if not results:
        print(
            f"WARNING: No result.json files found in {args.artifacts_dir}",
            file=sys.stderr,
        )

    html_content = render_report(
        results,
        title=args.title,
        project_dir=args.project_dir,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_content, encoding="utf-8")
    size_kb = args.output.stat().st_size / 1024
    print(
        f"Report written to {args.output} ({size_kb:.0f} KB, "
        f"{len(results)} models)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
