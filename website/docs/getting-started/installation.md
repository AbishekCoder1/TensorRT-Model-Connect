---
title: Installation
---

TensorRT-Model-Connect has two install surfaces:

- The Python builder package, `trtmc-build`, installed from `tensorrt_model_connect/`.
- The C++ runtime executable and library, built with CMake from the repository root.

## Host requirements

- Linux with an NVIDIA GPU.
- Docker and NVIDIA Container Toolkit for the standard project workflow.
- CUDA development files. The C++ core requires CUDA headers and `cudart`.
- TensorRT SDK when building the standard TensorRT backend DSO.
- Optional TensorRT-RTX SDK when building the RTX backend DSO.

Use [Environment and First Repro](environment-and-repro.md) for the full setup check before building a model.

## Container workflow

```bash
./scripts/docker_build_gb300.sh
./scripts/docker_run_gb300.sh
```

Then, inside the container:

```bash
pip install -e tensorrt_model_connect/
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DTRTMC_TRT_INCLUDE_DIR="${TRT_INC_DIR:-/usr/include/aarch64-linux-gnu}" \
  -DTRTMC_TRT_LIBRARY="${TRT_LIB_DIR:-/opt/venv/lib/python3.12/site-packages/tensorrt_libs}/libnvinfer.so" \
  -DTRTMC_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
  -DTRTMC_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so
cmake --build build -j
```

The configure step should print `TRT backend DSO: enabled`. If it says the TensorRT backend is skipped, fix the `TRTMC_TRT_INCLUDE_DIR` and `TRTMC_TRT_LIBRARY` paths before building bundles.

Use `pip install --no-deps -e tensorrt_model_connect/` only when the container already has the builder dependencies installed and you intentionally want to avoid dependency resolution.

## Python package

The package metadata is in `tensorrt_model_connect/pyproject.toml`. The installed console script is:

```bash
trtmc-build
```

The builder dependencies include `safetensors`, `numpy`, `ml_dtypes`, `onnx`, `onnxscript`, and `transformers`. TensorRT is intentionally not pinned in the package because it depends on the CUDA and TensorRT distribution installed in the target environment.

## C++ runtime build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DTRTMC_TRT_INCLUDE_DIR="${TRT_INC_DIR:-/usr/include/aarch64-linux-gnu}" \
  -DTRTMC_TRT_LIBRARY="${TRT_LIB_DIR:-/opt/venv/lib/python3.12/site-packages/tensorrt_libs}/libnvinfer.so" \
  -DTRTMC_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
  -DTRTMC_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so
cmake --build build -j
```

The main executable is:

```bash
./build/trtmc version
```

Run it inside the dev container unless you have exported equivalent runtime library paths. A host-side failure such as `libtorch.so: cannot open shared object file` usually means the executable was built against libraries available in the container environment but not on the host loader path.

The core runtime does not directly link `libnvinfer`. TensorRT execution lives behind backend DSOs loaded at runtime. The standard DSO is `libtrtmc_backend_trt.so`, with an ABI-suffixed alias when TensorRT headers expose a major/minor version.

## Optional build switches

| Switch | Default | Purpose |
| --- | --- | --- |
| `TRTMC_ENABLE_TRT` | `ON` | Enable TensorRT backend DSO integration. |
| `TRTMC_BUILD_BACKEND_TRT` | `ON` | Build the standard TensorRT backend DSO when headers and libraries exist. |
| `TRTMC_BUILD_BACKEND_RTX` | `OFF` | Build the TensorRT-RTX backend DSO. |
| `TRTMC_ENABLE_TVM_FFI` | `ON` | Enable the optional TVM-FFI module loader and plugin when available. |
