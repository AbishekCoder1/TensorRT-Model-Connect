# M0 E2E Result (2026-02-09)

## Native host commands executed
```bash
cmake -S . -B build -G Ninja -UTRTF_*
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/trtf_text_generation
```

## Native host observed result
- Build: success
- Tests: `2/2` passed
- TensorRT/CUDA detection: resolved
- Example output:

```text
backend=cpu-reference
[{'generated_text': 'the secret to baking a really good cake is to use fresh butter and measure carefully and follow recipe exactly.'}]
```

## Docker commands executed
```bash
./scripts/docker_build.sh
docker run --rm --gpus all trtf-dev bash -lc 'nvidia-smi -L'
docker run --rm --gpus all \
  -e TRT_ROOT=/opt/trt \
  -v /home/yifeif/repos/trt-transformers-cpp:/workspace/trt-transformers-cpp \
  -v /home/yifeif/repos/trt/build/tensorrt-base-dev/rel-10.15-native-x86_64-ubuntu20.04-cuda11.8-auto/cmake:/opt/trt:ro \
  -w /workspace/trt-transformers-cpp \
  trtf-dev bash -lc '
    cmake -S . -B build-docker -G Ninja \
      -DTRTF_TRT_INCLUDE_DIR=/opt/trt/include/zapped_headers \
      -DTRTF_TRT_LIBRARY=/opt/trt/lib/libnvinfer.so \
      -DTRTF_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
      -DTRTF_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so &&
    cmake --build build-docker -j &&
    ctest --test-dir build-docker --output-on-failure &&
    ./build-docker/trtf_text_generation
  '
```

## Docker observed result
- Image build: success (`trtf-dev`)
- GPU visibility in container: success (RTX 3090 Ti listed)
- In-container build/test/example: success (`2/2` tests passed)

## Notes
- TensorRT and CUDA development dependencies are detected at configure time.
- M0 keeps `trt` backend as a scaffold and intentionally uses `cpu-reference` for deterministic E2E execution.
- M1 will move execution path to real TRT network build/run while keeping CPU fallback for diagnostics.
