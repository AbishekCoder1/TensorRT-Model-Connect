#!/usr/bin/env bash
# One-shot setup inside the trtf-dev-gb300 container (aarch64 / CUDA 13.x).
# Run this once after `docker run` to install all deps and build.
set -euo pipefail

echo "=== Creating Python venv ==="
python3 -m venv .venv
source .venv/bin/activate

echo "=== Installing Python packages ==="
pip install -U pip

# Build tools (not available via apt in this base image)
pip install cmake ninja

# TensorRT (auto-selects cu13 for CUDA 13.x)
pip install tensorrt && pip install tensorrt --no-deps

# CUDA Python bindings (needed by debug_runner.py / diff tools)
pip install cuda-python

# Python deps
pip install "transformers>=4.57.0" tokenizers safetensors sentencepiece huggingface_hub ml_dtypes datasets
pip install pytest torch accelerate diffusers

# Install trtf_build
pip install --no-deps -e trtf_build/

echo "=== Configuring C++ build ==="
TRT_LIB_DIR=$(python3 -c "import importlib.util; s=importlib.util.find_spec('tensorrt_libs'); print(s.submodule_search_locations[0])")

# Create libnvinfer.so symlink if needed (pip installs libnvinfer.so.10)
[ ! -f "$TRT_LIB_DIR/libnvinfer.so" ] && ln -sf libnvinfer.so.10 "$TRT_LIB_DIR/libnvinfer.so"

# Find TRT headers (libnvinfer-headers-dev on aarch64)
TRT_INC=$(find /usr/include -name NvInferRuntime.h -printf '%h' -quit 2>/dev/null || echo "")
if [ -z "$TRT_INC" ]; then
  echo "WARNING: NvInferRuntime.h not found under /usr/include; TRT backend will be disabled"
  TRT_INC="/usr/include"
fi

cmake -S . -B build -G Ninja \
  -DTRTF_TRT_INCLUDE_DIR="$TRT_INC" \
  -DTRTF_TRT_LIBRARY="$TRT_LIB_DIR/libnvinfer.so" \
  -DTRTF_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
  -DTRTF_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so

echo "=== Building C++ runtime ==="
cmake --build build -j

echo "=== Running tests ==="
export LD_LIBRARY_PATH="$TRT_LIB_DIR:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
ctest --test-dir build --output-on-failure

echo ""
echo "=== Setup complete ==="
echo "Activate the venv:  source .venv/bin/activate"
echo "Set LD_LIBRARY_PATH: export LD_LIBRARY_PATH=\"$TRT_LIB_DIR:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}\""
