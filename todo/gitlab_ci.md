# GitLab CI Setup for trt-transformers

## Summary

Set up GitLab CI/CD on `gitlab-master.nvidia.com/yifeif/trt-transformers` with a self-hosted GPU runner on a GB300 machine. Uses a custom `.gitlab-ci.yml` (NOT Auto DevOps — it has no GPU support and targets web apps).

The Docker image (`Dockerfile.gb300`) is "fat" — all Python dependencies are baked in so the container is ready to go after `docker build`. No setup scripts needed inside the container.

## Why Not Auto DevOps

- No C++ buildpack support (Heroku buildpacks only)
- Zero GPU/CUDA awareness in any stage
- Auto Test uses Herokuish — cannot run `ctest` or custom `pytest`
- Auto Deploy assumes Kubernetes web service on port 5000
- Project needs: cmake + TRT + multi-tier GPU testing — none of which Auto DevOps handles

## Fat Docker Image (Dockerfile.gb300)

The image bakes in all dependencies so no `setup_gb300.sh` is needed:

- **System**: `build-essential`, `cmake`, `ninja-build`, `git`, `libnvinfer-headers-dev`
- **Python venv** at `/opt/venv` (auto-active via `ENV PATH`):
  - `tensorrt_cu13`, `tensorrt` (TRT 10.15)
  - `cuda-python` (debug_runner / diff tools)
  - `transformers`, `tokenizers`, `safetensors`, `sentencepiece`, `huggingface_hub`, `ml_dtypes`, `datasets`
  - `torch`, `torchaudio`, `torchvision` (cu130 wheels from `https://download.pytorch.org/whl/cu130`)
  - `pytest`, `pytest-xdist`, `accelerate`, `diffusers`, `protobuf`, `scipy`
- **ENV vars** pre-set: `TRT_LIB_DIR`, `TRT_INC_DIR`, `LD_LIBRARY_PATH` — cmake and runtime find TRT automatically
- **libnvinfer.so** symlink + `ldconfig` done at build time

After `docker build`, only 3 commands needed inside the container:
```bash
pip install --no-deps -e trtf_build/
cmake -S . -B build -G Ninja \
  -DTRTF_TRT_INCLUDE_DIR=$TRT_INC_DIR \
  -DTRTF_TRT_LIBRARY=$TRT_LIB_DIR/libnvinfer.so \
  -DTRTF_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
  -DTRTF_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so
cmake --build build -j
```

## Setup Steps

### Step 1: Install GitLab Runner on GB300 (aarch64)

**Note:** GB300 machines use `/workspace/users/` as the home base (not `/home/`).
The `/home` partition may not exist or be read-only. All working directories must
be under `/workspace/users/`.

```bash
# On the GB300 host (e.g. gb300-nvl-019-compute01.nvidia.com):
sudo curl -L --output /usr/local/bin/gitlab-runner \
  "https://s3.dualstack.us-east-1.amazonaws.com/gitlab-runner-downloads/latest/binaries/gitlab-runner-linux-arm64"
sudo chmod +x /usr/local/bin/gitlab-runner

# Create home dir manually first (/workspace is NFS/autofs — useradd --create-home hangs)
sudo mkdir -p /workspace/users/gitlab-runner
sudo useradd --comment 'GitLab Runner' --no-create-home \
  --home-dir /workspace/users/gitlab-runner --shell /bin/bash gitlab-runner
sudo chown gitlab-runner:gitlab-runner /workspace/users/gitlab-runner

sudo gitlab-runner install --user=gitlab-runner --working-directory=/workspace/users/gitlab-runner
sudo gitlab-runner start
```

### Step 2: Register Runner with GitLab

1. Go to **gitlab-master.nvidia.com/yifeif/trt-transformers → Settings → CI/CD → Runners**
2. Expand the **Runners** section, click **New project runner**
3. Fill in:
   - **Tags**: `gpu-gb300,aarch64,cuda13`
   - **Description**: `gb300-gpu-runner`
   - Check **Run untagged jobs** if you want it to pick up jobs without tags
4. Click **Create runner**
5. The next page shows a **runner authentication token** starting with `glrt-...` — copy it (shown only once)

```bash
# Replace glrt-XXXXX with the REAL token from the GitLab UI (Step 2.5 above)
# NOTE: --tag-list is NOT allowed with runner auth tokens (v18+).
# Tags are set in the GitLab UI when creating the runner (Step 2 above).
sudo gitlab-runner register \
  --non-interactive \
  --url "https://gitlab-master.nvidia.com/" \
  --token "glrt-XXXXX" \
  --executor "docker" \
  --docker-image "trtf-dev-gb300:latest" \
  --description "gb300-gpu-runner"
```

This writes `/etc/gitlab-runner/config.toml` with the real token. Verify it worked:
```bash
sudo gitlab-runner verify
# Should show: Verifying runner... is valid
```

### Step 3: Configure GPU Passthrough

The `register` command wrote a basic config. Now edit it to add GPU passthrough
and volume mounts — **do NOT replace the token** (it's already correct from register):

```bash
sudo nano /etc/gitlab-runner/config.toml
```

Add/modify the `[runners.docker]` section (keep the existing `token` line as-is):

```toml
concurrent = 1
check_interval = 3

[[runners]]
  name = "gb300-gpu-runner"
  url = "https://gitlab-master.nvidia.com/"
  token = "glrt-XXXXX"              # ← KEEP the real token from register, do NOT change
  executor = "docker"
  limit = 1

  [runners.docker]
    image = "trtf-dev-gb300:latest"
    gpus = "all"                    # KEY: passes GPUs to containers
    privileged = false
    pull_policy = "if-not-present"
    volumes = [
      "/cache",
      "/workspace/users/yifeif/trt-transformers:/workspace/users/yifeif/trt-transformers",
      "/mnt/storage/trt-transformers/model-weights:/root/.cache/huggingface/hub"
    ]
```

Then restart and verify:
```bash
sudo gitlab-runner restart && sudo gitlab-runner verify
# Should show: Verifying runner... is valid
```

### Step 4: Build the Fat Docker Image

```bash
cd /workspace/users/yifeif/trt-transformers-cpp
docker build -t trtf-dev-gb300:latest -f Dockerfile.gb300 .
```

Build takes ~5 min on GB300. The image is ~8 GB (mostly CUDA/TRT/torch wheels).

### Step 5: Add `.gitlab-ci.yml` to Repo Root

```yaml
# .gitlab-ci.yml
stages:
  - build
  - tier1-unit
  - tier2-graph-ops
  - tier3-smoke
  - tier4-full-e2e

# Optional: include GitLab security scanning templates
include:
  - template: Jobs/SAST.gitlab-ci.yml
  - template: Jobs/Secret-Detection.gitlab-ci.yml

variables:
  ENGINE_DIR: /workspace/users/yifeif/trt-transformers/engines
  GIT_STRATEGY: fetch

default:
  image: trtf-dev-gb300:latest
  tags: [gpu-gb300]
  # No before_script needed — fat image has /opt/venv auto-active via ENV PATH

# ── Build ──────────────────────────────────────

build-all:
  stage: build
  script:
    - pip install --no-deps -e trtf_build/
    - cmake -S . -B build -G Ninja
      -DTRTF_TRT_INCLUDE_DIR=$TRT_INC_DIR
      -DTRTF_TRT_LIBRARY=$TRT_LIB_DIR/libnvinfer.so
      -DTRTF_CUDA_INCLUDE_DIR=/usr/local/cuda/include
      -DTRTF_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so
    - cmake --build build -j
  artifacts:
    paths:
      - build/
    expire_in: 1 day
  timeout: 15m

# ── Tier 1: Unit Tests ~60s ──────────────────

test-python-builder:
  stage: tier1-unit
  needs: [build-all]
  script:
    - python -m pytest tests/builder/ -v --ignore=tests/builder/test_cli.py
  timeout: 15m

test-tools:
  stage: tier1-unit
  needs: [build-all]
  script:
    - python -m pytest tests/tools/ -v
  timeout: 5m

test-cpp-unit:
  stage: tier1-unit
  needs: [build-all]
  script:
    - ctest --test-dir build --output-on-failure
  timeout: 5m

# ── Tier 2: Graph-Op GPU Tests ~3min ─────────

test-graph-ops:
  stage: tier2-graph-ops
  needs: [test-python-builder, test-cpp-unit]
  script:
    - nvidia-smi
    - python -m pytest tests/builder/test_graph_ops.py
      tests/builder/test_graph_ops_extended.py
      tests/builder/test_graph_blocks.py -v
  timeout: 10m

# ── Tier 3: E2E Smoke Test ~3min ─────────────

test-e2e-smoke:
  stage: tier3-smoke
  needs: [test-graph-ops]
  script:
    - python -m pytest tests/test_e2e.py::test_e2e[qwen3-0.6b] -v
      --engine-dir $ENGINE_DIR
      --trtf-binary ./build/trtf
      --hf-python /opt/venv/bin/python
      --rebuild-engines
  timeout: 30m

# ── Tier 4: Full E2E ~2h (master or manual) ──

test-e2e-full:
  stage: tier4-full-e2e
  needs: [test-e2e-smoke]
  script:
    - python -m pytest tests/test_e2e.py -v
      --engine-dir $ENGINE_DIR
      --trtf-binary ./build/trtf
      --hf-python /opt/venv/bin/python
      --rebuild-engines
      --e2e-artifacts-dir /tmp/e2e_artifacts
  artifacts:
    paths:
      - /tmp/e2e_artifacts/
    when: always
    expire_in: 7 days
  timeout: 3h
  rules:
    - if: $CI_COMMIT_BRANCH == "master"           # Auto on master
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
      when: manual                                 # Manual trigger on MRs
      allow_failure: true
```

## Pipeline Flow

```
build-all → ┬─ test-python-builder ─┬─ test-graph-ops → test-e2e-smoke → test-e2e-full
             ├─ test-tools           │                                     (master only
             └─ test-cpp-unit  ──────┘                                      or manual)
```

- **Tiers 1-3** run on every push/MR (~20 min total)
- **Tier 4** runs automatically on `master` merges or manually on MRs (~2h)

## Validated Test Results (2026-02-27, gb300-nvl-019-compute02)

Full test run validated on `gb300-nvl-019-compute02` (4x NVIDIA GB300, aarch64, CUDA 13.0).

### How to reproduce

```bash
# 1. Build the fat image
docker build -t trtf-dev-gb300:latest -f Dockerfile.gb300 .

# 2. Start a container (detached, with source + storage mounts)
docker run -d --gpus all \
  -v "$PWD":/workspace/trt-transformers-cpp \
  -v /workspace/users/yifeif/trt-transformers:/workspace/users/yifeif/trt-transformers \
  -v /mnt/storage/trt-transformers/model-weights:/root/.cache/huggingface/hub \
  -w /workspace/trt-transformers-cpp \
  --name trtf-ci-gb300 \
  trtf-dev-gb300:latest sleep infinity

# 3. Install editable package + build C++ runtime
docker exec trtf-ci-gb300 pip install --no-deps -e trtf_build/
docker exec trtf-ci-gb300 bash -c \
  'cmake -S . -B build -G Ninja \
    -DTRTF_TRT_INCLUDE_DIR=$TRT_INC_DIR \
    -DTRTF_TRT_LIBRARY=$TRT_LIB_DIR/libnvinfer.so \
    -DTRTF_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
    -DTRTF_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so'
docker exec trtf-ci-gb300 bash -c 'cmake --build build -j'

# 4. Run tests tier by tier
# Tier 1: C++ unit tests
docker exec trtf-ci-gb300 ctest --test-dir build --output-on-failure

# Tier 1: Python builder unit tests
docker exec trtf-ci-gb300 python -m pytest tests/builder/ -v --ignore=tests/builder/test_cli.py

# Tier 1: Tools self-tests
docker exec trtf-ci-gb300 python -m pytest tests/tools/ -v

# Tier 2: Graph-op GPU tests
docker exec trtf-ci-gb300 python -m pytest \
  tests/builder/test_graph_ops.py \
  tests/builder/test_graph_ops_extended.py \
  tests/builder/test_graph_blocks.py -v

# Tier 3: E2E smoke test (single model)
docker exec trtf-ci-gb300 python -m pytest tests/test_e2e.py::test_e2e[qwen3-0.6b] -v \
  --engine-dir /workspace/users/yifeif/trt-transformers/engines \
  --trtf-binary ./build/trtf \
  --hf-python /opt/venv/bin/python \
  --rebuild-engines

# Tier 4: Full E2E (50 models across 4 GPUs)
docker exec trtf-ci-gb300 bash -c \
  'ENGINE_DIR=/workspace/users/yifeif/trt-transformers/engines \
   RESULT_DIR=/workspace/users/yifeif/trt-transformers/test-result \
   TRTF_BINARY=./build/trtf \
   HF_PYTHON=/opt/venv/bin/python \
   ./scripts/run_e2e_parallel.sh --rebuild-engines'

# 5. Cleanup
docker rm -f trtf-ci-gb300
```

### Results

| Tier | Test Suite | Result | Time |
|------|-----------|--------|------|
| 1 | C++ unit tests (`ctest`) | 20/20 passed | 2s |
| 1 | Python builder tests | 939 passed, 15 skipped | 10m |
| 1 | Tools self-tests | 162 passed, 1 skipped | 6s |
| 2 | Graph-op GPU tests | 143 passed | 2m37s |
| 3 | E2E smoke (qwen3-0.6b) | 1 passed | 2m22s |
| 4 | Full E2E (50 models, 4 GPUs) | 42 passed, 9 skipped, 1 xfailed | 104m |

**Skipped models** (expected):
- `gemma-2-2b` — gated model, needs HF access approval
- `eagle-embed-vl-1b-v2`, `eagle-rerank-vl-1b-v2` — HF repos 404
- `phi4-multimodal` — model-specific skip
- `internlm2-1.8b` — DynamicCache compatibility
- `nemotron-h-nano-9b` — requires `mamba-ssm` package
- 3 others — various expected skip reasons

**xfailed**: `pythia-70m` — ALiBi attention numerical tolerance (known)

### Key gotchas discovered during validation

1. **`-m trt` marker does not work** for graph-op tests — `requires_trt` is a `skipif` marker, not a selection marker. Run without `-m trt`.

2. **Missing `protobuf`** — `transformers` tokenizer for Qwen models needs it. Added to Dockerfile.

3. **Missing `scipy`** — Whisper's HF reference uses it. Added to Dockerfile.

4. **HF rate limiting** — Without `HF_TOKEN`, anonymous requests get 429'd. Copy the token into the container:
   ```bash
   # Copy from a container that has it, or write directly
   docker exec trtf-ci-gb300 bash -c 'echo "hf_YOUR_TOKEN" > /root/.cache/huggingface/token'
   ```

5. **`/opt/venv` vs `.venv`** — The fat image uses `/opt/venv` (auto-active via `ENV PATH`). The `run_e2e_parallel.sh` script does `source .venv/bin/activate` and falls back to any existing `.venv` from volume mounts. For CI, `--hf-python /opt/venv/bin/python` must be passed explicitly so subprocess references use the image's venv.

6. **`cuda-python` version conflict** — `torch` pins `cuda-bindings==13.0.3` but `cuda-python 13.1.1` wants `cuda-bindings~=13.1.1`. Non-breaking; both work fine at runtime.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Fat Docker image (all deps baked in) | Container is ready to go after build — no setup scripts needed |
| `/opt/venv` with `ENV PATH` | Venv auto-active in every `docker exec` and CI job — no `source activate` needed |
| `ENV TRT_LIB_DIR`, `TRT_INC_DIR` | Pre-computed paths so cmake works without manual discovery |
| `concurrent = 1` on runner | GPU jobs are memory-heavy, no parallelism |
| `gpus = "all"` | NVIDIA Container Toolkit passes GPUs through to Docker |
| Persistent volume mounts | Avoids re-downloading engine bundles and HF model cache |
| `--rebuild-engines` in E2E | Avoids testing stale cached bundles |
| `pull_policy = "if-not-present"` | Avoids pulling large CUDA image every time |
| Tier 4 manual on MRs | 2h jobs shouldn't block every MR; auto on master |
| `run_e2e_parallel.sh` for Tier 4 | Distributes 50 models across 4 GPUs (~104 min vs ~6h serial) |

## Prerequisites Checklist

- [ ] Docker + NVIDIA Container Toolkit installed on GB300
- [ ] GitLab Runner installed and registered
- [ ] `config.toml` edited with `gpus = "all"` and volume mounts
- [ ] `trtf-dev-gb300` Docker image built from fat `Dockerfile.gb300`
- [ ] `.gitlab-ci.yml` added to repo root and pushed
- [ ] HF token available in container (`/root/.cache/huggingface/token`)
- [ ] Volume mount paths adjusted for GB300 storage layout

## References

- [GitLab Runner installation (Linux arm64)](https://docs.gitlab.com/runner/install/linux-manually/)
- [GitLab Runner GPU configuration](https://docs.gitlab.com/runner/configuration/gpus/)
- [GitLab CI/CD YAML syntax](https://docs.gitlab.com/ci/yaml/)
- [config.toml reference](https://docs.gitlab.com/runner/configuration/advanced-configuration/)
