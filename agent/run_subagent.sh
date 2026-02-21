#!/usr/bin/env bash
set -euo pipefail

TASK_JSON=""
PROMPT_FILE=""
WORKTREE=""
WORKER_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-json) TASK_JSON="$2"; shift 2 ;;
    --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
    --worktree) WORKTREE="$2"; shift 2 ;;
    --worker-id) WORKER_ID="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$TASK_JSON" || -z "$PROMPT_FILE" || -z "$WORKTREE" ]]; then
  echo "Missing required args: --task-json --prompt-file --worktree" >&2
  exit 2
fi

TEMPLATE="${AGENT_SUBAGENT_CMD_TEMPLATE:-}"
if [[ -z "$TEMPLATE" ]]; then
  cat >&2 <<'MSG'
AGENT_SUBAGENT_CMD_TEMPLATE is not set.
Set a command template with placeholders:
  {prompt_file} {worktree} {task_json} {task_json_file} {worker_id}

Example:
  export AGENT_SUBAGENT_CMD_TEMPLATE='codex exec -C {worktree} --prompt-file {prompt_file}'
MSG
  exit 3
fi

TASK_JSON_FILE=$(mktemp)
printf '%s\n' "$TASK_JSON" > "$TASK_JSON_FILE"

CMD=$(python3 - "$TEMPLATE" "$PROMPT_FILE" "$WORKTREE" "$WORKER_ID" "$TASK_JSON" "$TASK_JSON_FILE" <<'PY'
import shlex
import sys

template = sys.argv[1]
values = {
    "prompt_file": sys.argv[2],
    "worktree": sys.argv[3],
    "worker_id": sys.argv[4],
    "task_json": sys.argv[5],
    "task_json_file": sys.argv[6],
}

rendered = template
for key, value in values.items():
    rendered = rendered.replace("{" + key + "}", shlex.quote(value))
print(rendered)
PY
)

set +e
(
  cd "$WORKTREE"
  bash -lc "$CMD"
)
rc=$?
set -e

rm -f "$TASK_JSON_FILE"
exit "$rc"
