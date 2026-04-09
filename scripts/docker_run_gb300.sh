#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Persistent storage paths (host) — adjust these to your local environment
STORAGE_ROOT="${TRTF_STORAGE_ROOT:-$HOME/trtf-storage}"
HF_CACHE="${TRTF_HF_CACHE:-$HOME/.cache/huggingface/hub}"
ENGINE_DIR="${STORAGE_ROOT}/engines"

# Create dirs if needed
mkdir -p "$HF_CACHE" "$ENGINE_DIR" 2>/dev/null || true

docker run --rm -it \
  --gpus all \
  -v "$PWD":/workspace/trt-transformers-cpp \
  -v "${STORAGE_ROOT}:${STORAGE_ROOT}" \
  -v "${HF_CACHE}":/root/.cache/huggingface/hub \
  -e ENGINE_DIR="${ENGINE_DIR}" \
  -w /workspace/trt-transformers-cpp \
  --name trtf-dev-gb300 \
  trtf-dev-gb300 bash
