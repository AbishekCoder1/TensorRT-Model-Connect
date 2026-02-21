#!/usr/bin/env bash
# One-shot setup inside the trtf-dev container.
# Run this once after `docker run` to install all deps and build.
set -euo pipefail

echo "=== Creating Python venv ==="
python3 -m venv .venv
source .venv/bin/activate

echo "=== Installing Python packages ==="
pip install -U pip
pip install tensorrt_cu12 && pip install tensorrt --no-deps
pip install "transformers>=4.57.0" tokenizers safetensors sentencepiece huggingface_hub ml_dtypes datasets
pip install pytest torch --index-url https://download.pytorch.org/whl/cu124
pip install accelerate diffusers
pip install --no-deps -e trtf_build/

echo "=== Configuring C++ build ==="
TRT_LIB_DIR=$(python3 -c "import importlib.util; s=importlib.util.find_spec('tensorrt_libs'); print(s.submodule_search_locations[0])")

# Create libnvinfer.so symlink if needed (pip installs libnvinfer.so.10)
[ ! -f "$TRT_LIB_DIR/libnvinfer.so" ] && ln -sf libnvinfer.so.10 "$TRT_LIB_DIR/libnvinfer.so"

cmake -S . -B build -G Ninja \
  -DTRTF_TRT_INCLUDE_DIR=/usr/include/x86_64-linux-gnu \
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
