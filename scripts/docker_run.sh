#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

docker run --rm -it \
  --gpus all \
  -v "$PWD":/workspace/trt-transformers-cpp \
  -w /workspace/trt-transformers-cpp \
  --name trtf-dev \
  trtf-dev bash
