#!/usr/bin/env bash
# Run all E2E tests in parallel across available GPUs.
#
# Each GPU runs multiple concurrent pytest workers (default 4) to maximize
# throughput.  Models are distributed using a size-aware scheduler that
# interleaves large and small models so each GPU runs a balanced mix.
#
# Usage (inside container):
#   ./scripts/run_e2e_parallel.sh --rebuild-engines
#   ./scripts/run_e2e_parallel.sh --engine-dir /path/to/engines --hf-python /opt/venv/bin/python
#   ./scripts/run_e2e_parallel.sh --task-strategy text_generation_causal --rebuild-engines
#
# Usage (from host):
#   docker exec trtf-dev-gb300 bash -c \
#     "cd /workspace/trt-transformers-cpp && ./scripts/run_e2e_parallel.sh --rebuild-engines"
#
# CLI options (override defaults):
#   --engine-dir PATH        Engine/bundle storage  (default: /workspace/users/yifeif/trt-transformers/engines)
#   --result-dir PATH        Test output directory  (default: /workspace/users/yifeif/trt-transformers/test-result)
#   --trtf-binary PATH       Path to trtf binary    (default: ./build/trtf)
#   --hf-python PATH         Python with HF deps    (default: /opt/venv/bin/python)
#   --num-gpus N             Number of GPUs to use  (default: auto-detect)
#   --workers-per-gpu N      Concurrent workers per GPU (default: 4)
#   --task-strategy STR      Filter by task strategy
#   --progress-interval N    Progress print interval in seconds (default: 30)
#   All other args are passed through to pytest (e.g., --rebuild-engines)
#
# Environment variables (lower priority than CLI):
#   ENGINE_DIR, RESULT_DIR, NUM_GPUS, WORKERS_PER_GPU, TRTF_BINARY, HF_PYTHON

set -euo pipefail

# --- Configuration -----------------------------------------------------------

ENGINE_DIR="${ENGINE_DIR:-/workspace/users/yifeif/trt-transformers/engines}"
RESULT_DIR="${RESULT_DIR:-/workspace/users/yifeif/trt-transformers/test-result}"
TRTF_BINARY="${TRTF_BINARY:-./build/trtf}"
HF_PYTHON="${HF_PYTHON:-/opt/venv/bin/python}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-30}"

# Auto-detect GPUs if not specified
if [ -z "${NUM_GPUS:-}" ]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$NUM_GPUS" -eq 0 ]; then
        echo "ERROR: No GPUs detected. Set NUM_GPUS=1 to run on CPU (will likely fail)." >&2
        exit 1
    fi
fi

# Passthrough args (e.g., --rebuild-engines, --task-strategy ...)
EXTRA_ARGS=()
FILTER_ARGS=()
MODELS_FILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --engine-dir)         ENGINE_DIR="$2"; shift 2 ;;
        --result-dir)         RESULT_DIR="$2"; shift 2 ;;
        --trtf-binary)        TRTF_BINARY="$2"; shift 2 ;;
        --hf-python)          HF_PYTHON="$2"; shift 2 ;;
        --num-gpus)           NUM_GPUS="$2"; shift 2 ;;
        --workers-per-gpu)    WORKERS_PER_GPU="$2"; shift 2 ;;
        --progress-interval)  PROGRESS_INTERVAL="$2"; shift 2 ;;
        --task-strategy)
            FILTER_ARGS+=(--e2e-task-strategy "$2")
            shift 2
            ;;
        --models-file)
            MODELS_FILE="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# --- Setup --------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

mkdir -p "$RESULT_DIR" "$ENGINE_DIR"

echo "=== E2E Parallel Test Runner ==="
echo "  GPUs:            $NUM_GPUS"
echo "  Workers/GPU:     $WORKERS_PER_GPU"
echo "  Engines:         $ENGINE_DIR"
echo "  Results:         $RESULT_DIR"
echo "  Binary:          $TRTF_BINARY"
echo "  HF Python:       $HF_PYTHON"
echo "  Progress every:  ${PROGRESS_INTERVAL}s"
echo "  Extra args:      ${EXTRA_ARGS[*]:-none}"
echo "  Filter:          ${FILTER_ARGS[*]:-all models}"
echo "  Models file:     ${MODELS_FILE:-none (collect all)}"
echo ""

# --- Collect test IDs ---------------------------------------------------------

if [ -n "$MODELS_FILE" ] && [ -f "$MODELS_FILE" ]; then
    # Selective mode: read model names from file (one per line), convert to test IDs
    TESTS=$(sed '/^$/d' "$MODELS_FILE" | while read -r model || [ -n "$model" ]; do
        echo "tests/test_e2e.py::test_e2e[${model}]"
    done | sort)
    echo "  Models file:     $MODELS_FILE ($(echo "$TESTS" | wc -l) models)"
else
    # Full mode: collect all tests via pytest
    TESTS=$("$HF_PYTHON" -m pytest tests/test_e2e.py --co -q "${FILTER_ARGS[@]}" 2>/dev/null \
        | grep "test_e2e\[" | sort)
fi
TOTAL=$(echo "$TESTS" | wc -l)

if [ "$TOTAL" -eq 0 ]; then
    echo "ERROR: No tests collected. Check --task-strategy filter." >&2
    exit 1
fi

echo "Collected $TOTAL tests"

# --- Schedule tests across GPUs × workers ------------------------------------

SCHEDULE_JSON="$RESULT_DIR/schedule.json"
echo "$TESTS" | python "$SCRIPT_DIR/schedule_e2e.py" \
    --num-gpus "$NUM_GPUS" \
    --workers-per-gpu "$WORKERS_PER_GPU" \
    > "$SCHEDULE_JSON"

echo ""

# --- Launch workers -----------------------------------------------------------

PIDS=()
WORKER_LABELS=()
START_TIME=$(date +%s)

# Parse schedule JSON and launch one pytest per worker slot.
# jq-free: use Python to emit "gpu_id worker_idx test1 test2 ..." lines.
while IFS= read -r line; do
    GPU_ID=$(echo "$line" | cut -d' ' -f1)
    WORKER_IDX=$(echo "$line" | cut -d' ' -f2)
    WORKER_TESTS=$(echo "$line" | cut -d' ' -f3-)
    WORKER_COUNT=$(echo "$WORKER_TESTS" | wc -w)

    [ "$WORKER_COUNT" -eq 0 ] && continue

    LABEL="gpu${GPU_ID}-w${WORKER_IDX}"
    echo "  $LABEL: $WORKER_COUNT tests"

    (
        export CUDA_VISIBLE_DEVICES=$GPU_ID
        # shellcheck disable=SC2086
        "$HF_PYTHON" -m pytest $WORKER_TESTS -v \
            --engine-dir "$ENGINE_DIR" \
            --trtf-binary "$TRTF_BINARY" \
            --hf-python "$HF_PYTHON" \
            --e2e-artifacts-dir "$RESULT_DIR/artifacts" \
            --junitxml="$RESULT_DIR/junit-${LABEL}.xml" \
            "${EXTRA_ARGS[@]}" \
            > "$RESULT_DIR/console-${LABEL}.log" 2>&1
    ) &
    PIDS+=($!)
    WORKER_LABELS+=("$LABEL")

done < <(python -c "
import json, sys
schedule = json.load(open('$SCHEDULE_JSON'))
for gpu_id in sorted(schedule, key=int):
    for w_idx, tests in enumerate(schedule[gpu_id]):
        print(f'{gpu_id} {w_idx} {\" \".join(tests)}')
")

TOTAL_WORKERS=${#PIDS[@]}

echo ""
echo "Workers launched: $TOTAL_WORKERS (PIDs: ${PIDS[*]})"
echo "Logs: $RESULT_DIR/console-gpu*-w*.log"
echo "  (live output suppressed to avoid interleaving — tail -f a log to watch)"
echo "Waiting for all workers..."
echo ""

# --- Helpers ------------------------------------------------------------------

format_duration() {
    local total="$1"
    local h=$(( total / 3600 ))
    local m=$(( (total % 3600) / 60 ))
    local s=$(( total % 60 ))
    if [ "$h" -gt 0 ]; then
        printf "%dh %02dm %02ds" "$h" "$m" "$s"
    else
        printf "%dm %02ds" "$m" "$s"
    fi
}

collect_test_progress() {
    local done=0 pass=0 fail=0 skip=0 xfail=0
    local files=()
    local f status

    for f in "$RESULT_DIR"/console-gpu*-w*.log; do
        [ -f "$f" ] && files+=("$f")
    done

    if [ "${#files[@]}" -eq 0 ]; then
        echo "0 0 0 0 0"
        return
    fi

    while IFS= read -r status; do
        [ -z "$status" ] && continue
        done=$((done + 1))
        case "$status" in
            PASSED) pass=$((pass + 1)) ;;
            SKIPPED) skip=$((skip + 1)) ;;
            XFAIL) xfail=$((xfail + 1)) ;;
            FAILED|ERROR|XPASS) fail=$((fail + 1)) ;;
        esac
    done < <(
        awk '
            /test_e2e\[/ {
                for (i = 1; i <= NF; i++) {
                    if ($i ~ /^(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)$/) {
                        print $i
                        break
                    }
                }
            }
        ' "${files[@]}" 2>/dev/null || true
    )

    echo "$done $pass $fail $skip $xfail"
}

print_progress() {
    local workers_done="$1"
    local workers_running="$2"
    local elapsed now done pass fail skip xfail pct eta
    local eta_str=""

    now=$(date +%s)
    elapsed=$(( now - START_TIME ))
    read -r done pass fail skip xfail < <(collect_test_progress)
    pct=$(awk -v d="$done" -v t="$TOTAL" 'BEGIN { if (t == 0) printf "0.0"; else printf "%.1f", (100.0 * d / t) }')

    if [ "$done" -gt 0 ] && [ "$done" -lt "$TOTAL" ]; then
        eta=$(( elapsed * (TOTAL - done) / done ))
        eta_str=" | ETA $(format_duration "$eta")"
    fi

    echo "[progress $(date +%H:%M:%S)] tests ${done}/${TOTAL} (${pct}%) pass=${pass} fail=${fail} skip=${skip} xfail=${xfail} | workers ${workers_done}/${TOTAL_WORKERS} done, ${workers_running} running | elapsed $(format_duration "$elapsed")${eta_str}"
}

# --- Wait and collect exit codes ----------------------------------------------

FAILURES=0
WORKERS_DONE=0
LAST_PROGRESS_TS=0
declare -a WORKER_FINISHED
for i in "${!PIDS[@]}"; do
    WORKER_FINISHED[$i]=0
done

while [ "$WORKERS_DONE" -lt "$TOTAL_WORKERS" ]; do
    RUNNING_NOW=0
    for i in "${!PIDS[@]}"; do
        if [ "${WORKER_FINISHED[$i]}" -eq 1 ]; then
            continue
        fi

        pid="${PIDS[$i]}"
        if kill -0 "$pid" 2>/dev/null; then
            RUNNING_NOW=$((RUNNING_NOW + 1))
            continue
        fi

        if wait "$pid"; then
            rc=0
        else
            rc=$?
        fi

        WORKER_FINISHED[$i]=1
        WORKERS_DONE=$((WORKERS_DONE + 1))
        if [ "$rc" -ne 0 ]; then
            FAILURES=$((FAILURES + 1))
            echo "  ${WORKER_LABELS[$i]}: FAILED (exit code $rc)"
        else
            echo "  ${WORKER_LABELS[$i]}: OK"
        fi

        LOG="$RESULT_DIR/console-${WORKER_LABELS[$i]}.log"
        SUMMARY=$(grep -E "^=+ .* in .* =+$" "$LOG" | tail -1 || true)
        [ -n "$SUMMARY" ] && echo "    $SUMMARY"
    done

    NOW_TS=$(date +%s)
    if [ "$LAST_PROGRESS_TS" -eq 0 ] || [ $(( NOW_TS - LAST_PROGRESS_TS )) -ge "$PROGRESS_INTERVAL" ] || [ "$RUNNING_NOW" -eq 0 ]; then
        print_progress "$WORKERS_DONE" "$RUNNING_NOW"
        LAST_PROGRESS_TS="$NOW_TS"
    fi

    [ "$WORKERS_DONE" -lt "$TOTAL_WORKERS" ] && sleep 5
done

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINUTES=$(( ELAPSED / 60 ))
SECONDS_REM=$(( ELAPSED % 60 ))

echo ""
echo "=== All $TOTAL_WORKERS workers finished in ${MINUTES}m ${SECONDS_REM}s ==="

# --- Merge JUnit XMLs --------------------------------------------------------

python -c "
from junitparser import JUnitXml
import glob, sys
files = sorted(glob.glob('$RESULT_DIR/junit-gpu*-w*.xml'))
if not files:
    print('No JUnit XML files found to merge.')
    sys.exit(0)
merged = JUnitXml()
for f in files:
    try:
        merged += JUnitXml.from_file(f)
    except Exception as e:
        print(f'Warning: could not parse {f}: {e}')
merged.write('$RESULT_DIR/junit.xml')
t = sum(1 for _ in merged)
print(f'Merged {len(files)} files -> $RESULT_DIR/junit.xml')
print(f'Total: {t} tests')
" 2>/dev/null || echo "(install junitparser to auto-merge: pip install junitparser)"

# --- Summary ------------------------------------------------------------------

echo ""
echo "Output files:"
echo "  Schedule:      $RESULT_DIR/schedule.json"
echo "  Console logs:  $RESULT_DIR/console-gpu*-w*.log"
echo "  JUnit XML:     $RESULT_DIR/junit-gpu*-w*.xml (merged: $RESULT_DIR/junit.xml)"
echo "  Artifacts:     $RESULT_DIR/artifacts/"
echo ""

# Per-test results from each worker
echo "--- Per-test results ---"
for LABEL in "${WORKER_LABELS[@]}"; do
    LOG="$RESULT_DIR/console-${LABEL}.log"
    [ -f "$LOG" ] || continue
    echo ""
    echo "  [$LABEL]"
    grep -E "PASSED|FAILED|SKIPPED|ERROR" "$LOG" \
        | grep -E "^tests/" \
        | sed 's/^/    /' || echo "    (no test results found)"
    SUMMARY=$(grep -E "^=" "$LOG" | tail -1)
    [ -n "$SUMMARY" ] && echo "    $SUMMARY"
done

# Print failure details so CI console shows root causes
if [ "$FAILURES" -gt 0 ]; then
    echo ""
    echo "--- Failure details ---"
    for LABEL in "${WORKER_LABELS[@]}"; do
        LOG="$RESULT_DIR/console-${LABEL}.log"
        [ -f "$LOG" ] || continue
        python -c "
import re, sys

log = open('$LOG').read()
failed = re.findall(r'tests/test_e2e\.py::test_e2e\[(.+?)\] FAILED', log)
if not failed:
    sys.exit(0)

# Extract the FAILURES section
failures_match = re.search(r'=+ FAILURES =+\n(.+?)(?=\n=+ )', log, re.DOTALL)
if not failures_match:
    for name in failed:
        print(f'  [{name}]')
        for line in log.splitlines():
            if 'E2E failed for' in line and name in line:
                print(f'    {line.strip()}')
                break
            if 'Failed:' in line and name in line:
                print(f'    {line.strip()}')
                break
        else:
            print(f'    (no detail found — check console log)')
    sys.exit(0)

failures_text = failures_match.group(1)
blocks = re.split(r'_+ test_e2e\[(.+?)\] _+\n', failures_text)
for i in range(1, len(blocks), 2):
    name = blocks[i]
    body = blocks[i + 1] if i + 1 < len(blocks) else ''
    print(f'  [{name}]')
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith('E '):
            print(f'    {stripped}')
" 2>/dev/null || true
    done
fi

echo ""
exit "$FAILURES"
