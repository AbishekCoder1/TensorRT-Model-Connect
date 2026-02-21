#!/usr/bin/env bash
# Per-slot cleanup: stop container, optionally remove worktree.
#
# Usage: ./scripts/parallel_family_teardown.sh <slot_id> <family_name> [--remove-worktree]
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <slot_id> <family_name> [--remove-worktree]" >&2
    exit 1
fi

SLOT_ID=$1; FAMILY=$2
REMOVE_WORKTREE=false
if [[ "${3:-}" == "--remove-worktree" ]]; then
    REMOVE_WORKTREE=true
fi

CONTAINER="trtf-slot-${SLOT_ID}"
WORKTREE_DIR="$(git rev-parse --show-toplevel)/.claude/worktrees/family-${FAMILY}"

# Stop and remove container
docker stop "$CONTAINER" 2>/dev/null && docker rm "$CONTAINER" 2>/dev/null || true
echo "Container $CONTAINER stopped."

# Optionally remove worktree
if $REMOVE_WORKTREE; then
    git worktree remove "$WORKTREE_DIR" --force 2>/dev/null || true
    git branch -D "family/${FAMILY}" 2>/dev/null || true
    echo "Worktree and branch family/${FAMILY} removed."
else
    echo "Worktree preserved at $WORKTREE_DIR (branch: family/${FAMILY})."
    echo "To remove: git worktree remove $WORKTREE_DIR && git branch -D family/${FAMILY}"
fi
