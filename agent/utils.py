from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "item"


def normalize_hf_model_id(link_or_id: str) -> str:
    raw = link_or_id.strip()
    if raw.startswith("https://huggingface.co/"):
        tail = raw.split("huggingface.co/", 1)[1]
        tail = tail.split("?", 1)[0].split("#", 1)[0]
        parts = [p for p in tail.split("/") if p]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return tail.strip("/")
    return raw


def run_cmd(cmd: list[str], cwd: Path, timeout: int = 1800, env: dict | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        timeout=timeout,
        capture_output=True,
        text=True,
        env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
