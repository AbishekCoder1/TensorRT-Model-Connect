FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    ca-certificates \
    pkg-config \
    python3 \
    python3-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/trt-transformers-cpp

# TensorRT is expected via mounted host path, e.g. /opt/trt
ENV TRT_ROOT=/opt/trt

CMD ["bash"]
