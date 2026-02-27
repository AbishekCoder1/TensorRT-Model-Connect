#!/usr/bin/env bash
# Run all E2E tests in parallel across available GPUs.
#
# Usage (inside container):
#   ./scripts/run_e2e_parallel.sh
#   ./scripts/run_e2e_parallel.sh --rebuild-engines
#   ./scripts/run_e2e_parallel.sh --task-strategy text_generation_causal
#
# Usage (from host):
#   docker exec trtf-dev-gb300 bash -c \
#     "cd /workspace/trt-transformers-cpp && ./scripts/run_e2e_parallel.sh"
#
# Environment variables:
#   ENGINE_DIR      Engine/bundle storage  (default: /workspace/users/yifeif/trt-transformers/engines)
#   RESULT_DIR      Test output directory  (default: /workspace/users/yifeif/trt-transformers/test-result)
#   NUM_GPUS        Number of GPUs to use  (default: auto-detect)
#   TRTF_BINARY     Path to trtf binary    (default: ./build/trtf)
#   HF_PYTHON       Python with HF deps    (default: .venv/bin/python)

set -euo pipefail

# --- Configuration -----------------------------------------------------------

ENGINE_DIR="${ENGINE_DIR:-/workspace/users/yifeif/trt-transformers/engines}"
RESULT_DIR="${RESULT_DIR:-/workspace/users/yifeif/trt-transformers/test-result}"
TRTF_BINARY="${TRTF_BINARY:-./build/trtf}"
HF_PYTHON="${HF_PYTHON:-.venv/bin/python}"

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
while [ $# -gt 0 ]; do
    case "$1" in
        --task-strategy)
            FILTER_ARGS+=(--e2e-task-strategy "$2")
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# --- Setup --------------------------------------------------------------------

cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

mkdir -p "$RESULT_DIR" "$ENGINE_DIR"

echo "=== E2E Parallel Test Runner ==="
echo "  GPUs:       $NUM_GPUS"
echo "  Engines:    $ENGINE_DIR"
echo "  Results:    $RESULT_DIR"
echo "  Binary:     $TRTF_BINARY"
echo "  HF Python:  $HF_PYTHON"
echo "  Extra args: ${EXTRA_ARGS[*]:-none}"
echo "  Filter:     ${FILTER_ARGS[*]:-all models}"
echo ""

# --- Collect test IDs ---------------------------------------------------------

TESTS=$(python -m pytest tests/test_e2e.py --co -q "${FILTER_ARGS[@]}" 2>/dev/null \
    | grep "test_e2e\[" | sort)
TOTAL=$(echo "$TESTS" | wc -l)

if [ "$TOTAL" -eq 0 ]; then
    echo "ERROR: No tests collected. Check --task-strategy filter." >&2
    exit 1
fi

CHUNK=$(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS ))
echo "Collected $TOTAL tests, $CHUNK per GPU ($NUM_GPUS workers)"
echo ""

# --- Launch workers -----------------------------------------------------------

PIDS=()
START_TIME=$(date +%s)

for GPU in $(seq 0 $((NUM_GPUS - 1))); do
    SUBSET=$(echo "$TESTS" | sed -n "$(( GPU * CHUNK + 1 )),$(( (GPU + 1) * CHUNK ))p")
    [ -z "$SUBSET" ] && continue

    COUNT=$(echo "$SUBSET" | wc -l)
    FIRST=$(echo "$SUBSET" | head -1 | sed 's/.*\[//;s/\].*//')
    LAST=$(echo "$SUBSET" | tail -1 | sed 's/.*\[//;s/\].*//')
    echo "GPU $GPU: $COUNT tests ($FIRST ... $LAST)"

    (
        export CUDA_VISIBLE_DEVICES=$GPU
        echo "$SUBSET" | tr "\n" " " | xargs \
            python -m pytest -v \
            --engine-dir "$ENGINE_DIR" \
            --trtf-binary "$TRTF_BINARY" \
            --hf-python "$HF_PYTHON" \
            --e2e-artifacts-dir "$RESULT_DIR/artifacts" \
            --junitxml="$RESULT_DIR/junit-gpu${GPU}.xml" \
            "${EXTRA_ARGS[@]}" \
            2>&1 | tee "$RESULT_DIR/console-gpu${GPU}.log"
    ) &
    PIDS+=($!)
done

echo ""
echo "Workers launched: ${PIDS[*]}"
echo "Logs: $RESULT_DIR/console-gpu{0..$((NUM_GPUS-1))}.log"
echo "Waiting for all workers..."
echo ""

# --- Wait and collect exit codes ----------------------------------------------

FAILURES=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        FAILURES=$((FAILURES + 1))
        echo "  GPU $i: FAILED (exit code $?)"
    else
        echo "  GPU $i: OK"
    fi
done

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINUTES=$(( ELAPSED / 60 ))
SECONDS=$(( ELAPSED % 60 ))

echo ""
echo "=== All workers finished in ${MINUTES}m ${SECONDS}s ==="

# --- Merge JUnit XMLs --------------------------------------------------------

python -c "
from junitparser import JUnitXml
import glob, sys
files = sorted(glob.glob('$RESULT_DIR/junit-gpu*.xml'))
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
f = sum(1 for tc in merged if any(True for _ in tc.iterchildren()) and not tc.is_skipped)
s = sum(1 for tc in merged if tc.is_skipped)
print(f'Merged {len(files)} files -> $RESULT_DIR/junit.xml')
print(f'Total: {t} tests')
" 2>/dev/null || echo "(install junitparser to auto-merge: pip install junitparser)"

# --- Summary ------------------------------------------------------------------

echo ""
echo "Output files:"
echo "  Console logs:  $RESULT_DIR/console-gpu*.log"
echo "  JUnit XML:     $RESULT_DIR/junit-gpu*.xml (merged: $RESULT_DIR/junit.xml)"
echo "  Artifacts:     $RESULT_DIR/artifacts/"
echo ""

# Quick pass/fail summary from each log
for GPU in $(seq 0 $((NUM_GPUS - 1))); do
    LOG="$RESULT_DIR/console-gpu${GPU}.log"
    [ -f "$LOG" ] || continue
    SUMMARY=$(grep -E "^(PASSED|FAILED|ERROR|=)" "$LOG" | tail -1)
    echo "  GPU $GPU: $SUMMARY"
done

exit "$FAILURES"
