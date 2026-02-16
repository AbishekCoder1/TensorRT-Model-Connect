#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Persistent storage paths (host)
STORAGE_ROOT="/mnt/storage/trt-transformers"
HF_CACHE="${STORAGE_ROOT}/model-weights"
ENGINE_DIR="${STORAGE_ROOT}/engines"

# Create dirs if needed
mkdir -p "$HF_CACHE" "$ENGINE_DIR"

docker run --rm -it \
  --gpus all \
  -v "$PWD":/workspace/trt-transformers-cpp \
  -v "${STORAGE_ROOT}:${STORAGE_ROOT}" \
  -v "${HF_CACHE}":/root/.cache/huggingface/hub \
  -w /workspace/trt-transformers-cpp \
  --name trtf-dev \
  trtf-dev bash
