#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

docker run --rm -it \
  --gpus all \
  -e TRT_ROOT=/opt/trt \
  -e LD_LIBRARY_PATH=/opt/trt/Debug/lib:/opt/trt/Release/lib:/opt/trt/lib:/opt/trt/myelin-ext/myelin/lib/Debug:/usr/local/cuda/lib64 \
  -v "$PWD":/workspace/trt-transformers-cpp \
  -v /home/yifeif/repos/trt/build/tensorrt-base-dev/rel-10.15-native-x86_64-ubuntu20.04-cuda11.8-auto/cmake:/opt/trt:ro \
  -w /workspace/trt-transformers-cpp \
  trtf-dev bash
