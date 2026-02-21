#!/usr/bin/env bash
set -euo pipefail

# Single-agent smoke example:
# - Uses echo-based templates from env.smoke.sh
# - Validates ingestion + scheduling + worker task completion flow

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STORE_DIR="${1:-$ROOT_DIR/.tmp-agent-one-example}"
MODEL_LINK="${2:-https://huggingface.co/Qwen/Qwen3-0.6B}"

cd "$ROOT_DIR"
source agent/config/env.smoke.sh

rm -rf "$STORE_DIR"

python3 -m agent.submit_links --store "$STORE_DIR" --link "$MODEL_LINK"
python3 -m agent.intake_daemon --store "$STORE_DIR" --once

# First pass usually runs implement_plugin.
python3 -m agent.scheduler --store "$STORE_DIR" --once
sleep 1

# Second pass usually runs add_manifest after plugin completion.
python3 -m agent.scheduler --store "$STORE_DIR" --once
sleep 1

python3 -m agent.report_status --store "$STORE_DIR"

python3 - <<PY
import json
from pathlib import Path

p = Path(r"$STORE_DIR") / "backlog.json"
d = json.loads(p.read_text())
print("\\nTask statuses:")
for t in d["tasks"]:
    print(f"  {t['id']}: {t['status']}")
PY

echo
echo "Smoke run complete."
echo "Next step for real runs: source agent/config/env.production.sh and set strict parity hook env vars."
