#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${TRTF_BUILD_DIR:-build-container-phase1}"
PROMPT="${1:-Hello}"
LOG_FILE="${TRTF_E2E_LOG_FILE:-/tmp/trtf_qwen3_trt_e2e.log}"
CACHE_DIR="${TRTF_TRT_ENGINE_CACHE_DIR:-/tmp/trtf_engine_cache_qwen3}"
HF_PY="${TRTF_HF_PYTHON:-$PWD/.venv-hf/bin/python}"

if [[ ! -x "$HF_PY" ]]; then
  echo "[e2e] missing HF python bridge: $HF_PY" >&2
  exit 1
fi

echo "[e2e] build dir: $BUILD_DIR"
echo "[e2e] prompt: $PROMPT"
echo "[e2e] cache dir: $CACHE_DIR"
echo "[e2e] log file: $LOG_FILE"
echo "[e2e] hf python: $HF_PY"

touch "$LOG_FILE"

cmake -S . -B "$BUILD_DIR" -G Ninja \
  -DTRTF_TRT_INCLUDE_DIR=${TRTF_TRT_INCLUDE_DIR:-/opt/trt/include/zapped_headers} \
  -DTRTF_TRT_LIBRARY=${TRTF_TRT_LIBRARY:-/opt/trt/Debug/lib/libnvinfer.so} \
  -DTRTF_CUDA_INCLUDE_DIR=${TRTF_CUDA_INCLUDE_DIR:-/usr/local/cuda/include} \
  -DTRTF_CUDART_LIBRARY=${TRTF_CUDART_LIBRARY:-/usr/local/cuda/lib64/libcudart.so}
cmake --build "$BUILD_DIR" -j
ctest --test-dir "$BUILD_DIR" --output-on-failure

for run in 1 2; do
  echo "[e2e] ===== run ${run}/2 =====" | tee -a "$LOG_FILE"
  {
    time env \
      TRTF_TRT_LOG_STDERR=1 \
      TRTF_TRT_LOG_MIN_SEVERITY="${TRTF_TRT_LOG_MIN_SEVERITY:-INFO}" \
      TRTF_TRT_ENGINE_CACHE_DIR="$CACHE_DIR" \
      TRTF_HF_PYTHON="$HF_PY" \
      TRTF_MAX_CACHE_LENGTH="${TRTF_MAX_CACHE_LENGTH:-1}" \
      TRTF_MAX_NEW_TOKENS="${TRTF_MAX_NEW_TOKENS:-1}" \
      "./${BUILD_DIR}/trtf_load_model" --force-trt QWEN3 "$PROMPT"
  } 2>&1 | tee -a "$LOG_FILE"

done

echo "[e2e] completed. logs written to $LOG_FILE"
