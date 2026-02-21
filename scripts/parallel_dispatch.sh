#!/usr/bin/env bash
# Orchestrator: reads a JSON manifest, pre-downloads HF models, and sets up
# all parallel slots (worktree + container + venv + build).
#
# Usage: ./scripts/parallel_dispatch.sh <families_manifest.json>
#
# Phases:
#   1. Pre-download HF models (sequential, avoids rate limits)
#   2. Setup all slots in parallel (worktree + container + venv + build)
#   3. Print slot status
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <families_manifest.json>" >&2
    exit 1
fi

MANIFEST="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v jq &>/dev/null; then
    echo "ERROR: jq is required. Install with: apt-get install jq" >&2
    exit 1
fi

# Validate manifest
if ! jq empty "$MANIFEST" 2>/dev/null; then
    echo "ERROR: Invalid JSON in $MANIFEST" >&2
    exit 1
fi

NUM_SLOTS=$(jq length "$MANIFEST")
echo "=== Parallel Family Dispatch: $NUM_SLOTS slots ==="
echo ""

# Phase 1: Pre-download HF models into shared cache (sequential to avoid rate limits)
echo "=== Phase 1: Pre-download HF models ==="
# Use existing container or host python for downloads
DOWNLOAD_CONTAINER=""
if docker ps --format '{{.Names}}' | grep -q trtf-dev-gb300; then
    DOWNLOAD_CONTAINER="trtf-dev-gb300"
fi

for entry in $(jq -c '.[]' "$MANIFEST"); do
    HF_REPO=$(echo "$entry" | jq -r '.hf_repo')
    TEST_MODEL=$(echo "$entry" | jq -r '.test_model')

    for model in "$HF_REPO" "$TEST_MODEL"; do
        echo "  Downloading $model ..."
        if [[ -n "$DOWNLOAD_CONTAINER" ]]; then
            docker exec "$DOWNLOAD_CONTAINER" bash -c \
                "source .venv/bin/activate && python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('$model')\"" \
                || echo "    WARNING: failed to download $model (continuing)"
        else
            python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$model')" \
                || echo "    WARNING: failed to download $model (continuing)"
        fi
    done
done

echo ""
echo "=== Phase 2: Setup slots in parallel ==="
PIDS=()
SLOT_INFO=()
for entry in $(jq -c '.[]' "$MANIFEST"); do
    SLOT=$(echo "$entry" | jq -r '.slot')
    GPU=$(echo "$entry" | jq -r '.gpu')
    FAMILY=$(echo "$entry" | jq -r '.family')

    echo "  Starting slot $SLOT: family=$FAMILY gpu=$GPU"
    "$SCRIPT_DIR/parallel_family_setup.sh" "$SLOT" "$FAMILY" "$GPU" &
    PIDS+=($!)
    SLOT_INFO+=("$SLOT:$FAMILY:$GPU")
done

echo "  Waiting for all $NUM_SLOTS slots..."
FAILED=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "  FAILED: slot ${SLOT_INFO[$i]}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=== Phase 3: Status ==="
for entry in $(jq -c '.[]' "$MANIFEST"); do
    SLOT=$(echo "$entry" | jq -r '.slot')
    FAMILY=$(echo "$entry" | jq -r '.family')
    CONTAINER="trtf-slot-${SLOT}"
    STATUS=$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo "not found")
    echo "  Slot $SLOT ($FAMILY): container=$STATUS"
done

echo ""
if [[ "$FAILED" -gt 0 ]]; then
    echo "WARNING: $FAILED of $NUM_SLOTS slot(s) failed setup."
    exit 1
fi
echo "All $NUM_SLOTS slots ready. Launch subagents now."
