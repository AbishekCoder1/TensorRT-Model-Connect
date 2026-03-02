#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Dockerfile.gb300 does not COPY from the repository.
# Use an empty context to avoid uploading large local artifacts.
EMPTY_CONTEXT="${TMPDIR:-/tmp}/trtf-empty-docker-context"
mkdir -p "$EMPTY_CONTEXT"

docker build -t trtf-dev-gb300 -f "$REPO_ROOT/Dockerfile.gb300" "$EMPTY_CONTEXT"
