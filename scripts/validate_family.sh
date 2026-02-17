#!/usr/bin/env bash
# Validate a model family end-to-end: build bundle, diff logits, diff layers, runner parity.
#
# Usage:
#   ./scripts/validate_family.sh Qwen/Qwen3-0.6B                       # HF repo ID
#   ./scripts/validate_family.sh models/hf/Qwen__Qwen3-0.6B            # local dir
#   ./scripts/validate_family.sh Qwen/Qwen3-0.6B --max-cache-length 512
#   ./scripts/validate_family.sh Qwen/Qwen3-0.6B --binary ./build/trtf
#
# Requirements: torch, trtf_build installed, C++ binary built.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Defaults
MAX_CACHE_LENGTH=256
BINARY="${PROJECT_DIR}/build/trtf"
BUNDLE_DIR="/tmp"
TRUST_REMOTE_CODE=""

# Parse args
MODEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-cache-length) MAX_CACHE_LENGTH="$2"; shift 2 ;;
        --binary) BINARY="$2"; shift 2 ;;
        --bundle-dir) BUNDLE_DIR="$2"; shift 2 ;;
        --trust-remote-code) TRUST_REMOTE_CODE="--trust-remote-code"; shift ;;
        -h|--help)
            echo "Usage: $0 <model-id-or-path> [--max-cache-length N] [--binary PATH] [--bundle-dir DIR] [--trust-remote-code]"
            exit 0
            ;;
        *)
            if [[ -z "$MODEL" ]]; then
                MODEL="$1"
            else
                echo "ERROR: unexpected argument: $1" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "ERROR: model ID or path required." >&2
    echo "Usage: $0 <model-id-or-path> [--max-cache-length N] [--binary PATH]" >&2
    exit 1
fi

# Derive a safe bundle filename from the model ID.
SAFE_NAME="$(echo "$MODEL" | tr '/' '_' | tr ' ' '_')"
BUNDLE_PATH="${BUNDLE_DIR}/${SAFE_NAME}.trtfb"

# Detect venv python for --hf-python
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    HF_PYTHON="${VIRTUAL_ENV}/bin/python"
else
    HF_PYTHON="${PROJECT_DIR}/.venv/bin/python"
fi

# Set up LD_LIBRARY_PATH for TRT
TRT_LIB_DIR=$(python3 -c "import importlib.util; s=importlib.util.find_spec('tensorrt_libs'); print(s.submodule_search_locations[0])" 2>/dev/null || true)
if [[ -n "$TRT_LIB_DIR" ]]; then
    export LD_LIBRARY_PATH="${TRT_LIB_DIR}:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

PASS=0
FAIL=0
STEPS=()

run_step() {
    local name="$1"
    shift
    echo ""
    echo "==== $name ===="
    if "$@"; then
        STEPS+=("PASS  $name")
        PASS=$((PASS + 1))
    else
        STEPS+=("FAIL  $name")
        FAIL=$((FAIL + 1))
    fi
}

# Step 1: Build bundle
run_step "Build bundle" \
    trtf-build build "$MODEL" -o "$BUNDLE_PATH" --max-cache-length "$MAX_CACHE_LENGTH"

# Step 2: diff_logits (E2E logit comparison, 4 prompts)
run_step "diff_logits (battery)" \
    python3 "${PROJECT_DIR}/tools/diff_logits.py" \
        --model "$MODEL" --atol 1e-3 --battery \
        --max-cache-length "$MAX_CACHE_LENGTH" $TRUST_REMOTE_CODE

# Step 3: diff_layers (per-layer hidden state comparison)
run_step "diff_layers" \
    python3 "${PROJECT_DIR}/tools/diff_layers.py" \
        --model "$MODEL" --atol 0.05 \
        --max-cache-length "$MAX_CACHE_LENGTH" $TRUST_REMOTE_CODE

# Step 4: Runner parity (Python vs C++)
if [[ -x "$BINARY" ]]; then
    run_step "test_runner_parity" \
        python3 "${PROJECT_DIR}/tools/test_runner_parity.py" \
            --bundle "$BUNDLE_PATH" --binary "$BINARY" \
            --hf-python "$HF_PYTHON" --max-new-tokens 20
else
    echo ""
    echo "==== test_runner_parity ===="
    echo "SKIP: C++ binary not found at $BINARY"
    STEPS+=("SKIP  test_runner_parity (no binary)")
fi

# Summary
echo ""
echo "========================================"
echo "  Validation Summary: $MODEL"
echo "========================================"
for s in "${STEPS[@]}"; do
    echo "  $s"
done
echo "----------------------------------------"
echo "  $PASS passed, $FAIL failed"
echo "========================================"

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
