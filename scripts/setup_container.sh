#!/usr/bin/env bash
# One-shot repo setup inside the dev container.
# All runtime dependencies are already baked into Dockerfile/Dockerfile.gb300.
set -euo pipefail

echo "=== Verifying container runtime deps ==="
python3 -c "import tensorrt, torch, transformers; print('ok: tensorrt/torch/transformers')"

echo "=== Installing local trtf_build package (editable) ==="
pip install --no-deps -e trtf_build/

echo "=== Configuring C++ build ==="
TRT_INC_DIR="${TRT_INC_DIR:-/usr/include/aarch64-linux-gnu}"
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
