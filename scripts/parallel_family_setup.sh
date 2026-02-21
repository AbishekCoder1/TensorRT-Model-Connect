#!/usr/bin/env bash
# Per-slot setup: create git worktree, Docker container, venv, and C++ build.
#
# Usage: ./scripts/parallel_family_setup.sh <slot_id> <family_name> <gpu_id>
#
# Creates:
#   .claude/worktrees/family-<family_name>/   (git worktree)
#   Container: trtf-slot-<slot_id>            (pinned to GPU <gpu_id>)
#   Separate .venv inside the worktree
#   Separate build/ inside the worktree
#   Separate engine dir: engines-slot-<slot_id>/
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <slot_id> <family_name> <gpu_id>" >&2
    exit 1
fi

SLOT_ID=$1; FAMILY=$2; GPU_ID=$3
REPO_ROOT=$(git rev-parse --show-toplevel)
WORKTREE_DIR="$REPO_ROOT/.claude/worktrees/family-${FAMILY}"
STORAGE_ROOT="/workspace/users/yifeif/trt-transformers"
ENGINE_DIR="${STORAGE_ROOT}/engines-slot-${SLOT_ID}"
CONTAINER="trtf-slot-${SLOT_ID}"
HF_CACHE="/mnt/storage/trt-transformers/model-weights"
IMAGE="trtf-dev-gb300"

# 1. Create git worktree on a new branch
if [[ -d "$WORKTREE_DIR" ]]; then
    echo "Worktree already exists at $WORKTREE_DIR — reusing."
else
    git worktree add -b "family/${FAMILY}" "$WORKTREE_DIR" HEAD
fi

# 2. Create per-slot engine directory and shared HF cache
mkdir -p "$ENGINE_DIR" "$HF_CACHE" 2>/dev/null || true

# 3. Remove stale container if exists
docker rm -f "$CONTAINER" 2>/dev/null || true

# 4. Launch container pinned to one GPU, mounting the worktree
docker run -d \
  --gpus "\"device=${GPU_ID}\"" \
  -v "$WORKTREE_DIR":/workspace/trt-transformers-cpp \
  -v "${STORAGE_ROOT}:${STORAGE_ROOT}" \
  -v "$ENGINE_DIR":/mnt/storage/trt-transformers/engines \
  -v "${HF_CACHE}":/root/.cache/huggingface/hub \
  -w /workspace/trt-transformers-cpp \
  --name "$CONTAINER" \
  "$IMAGE" sleep infinity

# 5. Run setup inside container (creates .venv, builds C++)
docker exec "$CONTAINER" bash ./scripts/setup_gb300.sh

echo "Slot $SLOT_ID ready: family=$FAMILY gpu=$GPU_ID container=$CONTAINER worktree=$WORKTREE_DIR"
