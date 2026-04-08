#!/usr/bin/env bash
# One-shot repo setup inside the dev container.
# All runtime dependencies are already baked into Dockerfile/Dockerfile.gb300.
set -euo pipefail

echo "=== Verifying container runtime deps ==="
python3 -c "import tensorrt, torch, transformers; print('ok: tensorrt/torch/transformers')"
lizard --version >/dev/null
echo "ok: lizard"
python3 -c "import torch_tensorrt; print('ok: torch_tensorrt', torch_tensorrt.__version__)"

echo "=== Installing local build packages (editable) ==="
# Use strict editable mode: the project dirs (trtf_build/, ttrt_build/) have the
# same name as their Python packages, so the default finder-based editable install
# gets shadowed by CWD when running from the repo root. Strict mode uses symlinks
# in site-packages which avoids the namespace-package collision.
pip install --no-deps --config-settings editable_mode=strict -e trtf_build/
pip install --no-deps --config-settings editable_mode=strict -e ttrt_build/

echo "=== Configuring C++ build ==="
if [ -z "${TRT_INC_DIR:-}" ]; then
  TRT_INC_DIR=$(find /usr/include -maxdepth 2 -name NvInferRuntime.h -printf '%h' -quit 2>/dev/null || echo "")
  if [ -z "$TRT_INC_DIR" ]; then
    echo "WARNING: NvInferRuntime.h not found, TRT backend will be disabled"
    TRT_INC_DIR="/usr/include"
  fi
  echo "  Auto-detected TRT_INC_DIR=$TRT_INC_DIR"
fi
TRT_LIB_DIR="${TRT_LIB_DIR:-/opt/venv/lib/python3.12/site-packages/tensorrt_libs}"

cmake -S . -B build -G Ninja \
  -DTRTF_TRT_INCLUDE_DIR="$TRT_INC_DIR" \
  -DTRTF_TRT_LIBRARY="$TRT_LIB_DIR/libnvinfer.so" \
  -DTRTF_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
  -DTRTF_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so

echo "=== Building C++ runtime ==="
cmake --build build -j

echo "=== Running C++ unit tests ==="
ctest --test-dir build --output-on-failure

echo ""
echo "=== Setup complete ==="
echo "Container environment is ready. Use /opt/venv/bin/python for test/inference commands."
echo "  Raw TRT:   trtf-build build <model> -o out.trtfb"
echo "  Torch-TRT: trtf-build build --torch-trt <model> -o out.trtfb"
