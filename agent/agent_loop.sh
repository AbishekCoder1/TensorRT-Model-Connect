#!/usr/bin/env bash
set -euo pipefail

STORE_DIR="${1:-agent}"
INTERVAL="${AGENT_LOOP_INTERVAL_SECONDS:-5}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT_DIR"

while true; do
  python3 -m agent.intake_daemon --store "$STORE_DIR" --once
  python3 -m agent.scheduler --store "$STORE_DIR" --once
  python3 -m agent.merge_manager --store "$STORE_DIR" --once || true
  sleep "$INTERVAL"
done
