#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Persistent storage paths (host)
STORAGE_ROOT="/mnt/storage/trt-transformers"
HF_CACHE="${STORAGE_ROOT}/model-weights"
ENGINE_DIR="${STORAGE_ROOT}/engines"

# Create dirs if needed (may fail if storage not mounted — non-fatal)
mkdir -p "$HF_CACHE" "$ENGINE_DIR" 2>/dev/null || true

docker run --rm -it \
  --gpus all \
  -v "$PWD":/workspace/trt-transformers-cpp \
  -v "${STORAGE_ROOT}:${STORAGE_ROOT}" \
  -v "${HF_CACHE}":/root/.cache/huggingface/hub \
  -w /workspace/trt-transformers-cpp \
  --name trtf-dev-gb300 \
  trtf-dev-gb300 bash
