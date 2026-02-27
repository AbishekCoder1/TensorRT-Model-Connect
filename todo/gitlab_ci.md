# GitLab CI Setup for trt-transformers

## Summary

Set up GitLab CI/CD on `gitlab-master.nvidia.com/yifeif/trt-transformers` with a self-hosted GPU runner on `gb300-nvl-019-compute01.nvidia.com`. Uses a custom `.gitlab-ci.yml` (NOT Auto DevOps — it has no GPU support and targets web apps).

## Why Not Auto DevOps

- No C++ buildpack support (Heroku buildpacks only)
- Zero GPU/CUDA awareness in any stage
- Auto Test uses Herokuish — cannot run `ctest` or custom `pytest`
- Auto Deploy assumes Kubernetes web service on port 5000
- Project needs: cmake + TRT + multi-tier GPU testing — none of which Auto DevOps handles

## Setup Steps

### Step 1: Install GitLab Runner on GB300 (aarch64)

```bash
# On gb300-nvl-019-compute01.nvidia.com:
sudo curl -L --output /usr/local/bin/gitlab-runner \
  "https://s3.dualstack.us-east-1.amazonaws.com/gitlab-runner-downloads/latest/binaries/gitlab-runner-linux-arm64"
sudo chmod +x /usr/local/bin/gitlab-runner

sudo useradd --comment 'GitLab Runner' --create-home gitlab-runner --shell /bin/bash
sudo gitlab-runner install --user=gitlab-runner --working-directory=/home/gitlab-runner
sudo gitlab-runner start
```

### Step 2: Register Runner with GitLab

1. Go to **gitlab-master.nvidia.com/yifeif/trt-transformers → Settings → CI/CD → Runners → New project runner**
2. Set tags: `gpu-gb300`
3. Copy the authentication token (`glrt-...`)

```bash
sudo gitlab-runner register \
  --non-interactive \
  --url "https://gitlab-master.nvidia.com/" \
  --token "glrt-YOUR_TOKEN_HERE" \
  --executor "docker" \
  --docker-image "trtf-dev-gb300:latest" \
  --description "gb300-gpu-runner" \
  --tag-list "gpu-gb300,aarch64,cuda13"
```

### Step 3: Configure GPU Passthrough

Edit `/etc/gitlab-runner/config.toml` on the GB300:

```toml
concurrent = 1
check_interval = 3

[[runners]]
  name = "gb300-gpu-runner"
  url = "https://gitlab-master.nvidia.com/"
  token = "glrt-YOUR_TOKEN_HERE"
  executor = "docker"
  limit = 1

  [runners.docker]
    image = "trtf-dev-gb300:latest"
    gpus = "all"                    # KEY: passes GPUs to containers
    privileged = false
    pull_policy = "if-not-present"
    volumes = [
      "/cache",
      "/workspace/users/yifeif/trt-transformers/storage:/mnt/storage/trt-transformers",
      "/root/.cache/huggingface/hub:/root/.cache/huggingface/hub"
    ]
```

Then restart:
```bash
sudo gitlab-runner restart && sudo gitlab-runner verify
```

### Step 4: Pre-build the CI Docker Image

```bash
cd /workspace/users/yifeif/trt-transformers
docker build -t trtf-dev-gb300 -f Dockerfile.gb300 .
# Then inside container: run setup_gb300.sh to bake in the venv
```

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
  ENGINE_DIR: /mnt/storage/trt-transformers/engines
  GIT_STRATEGY: fetch

default:
  image: trtf-dev-gb300:latest
  tags: [gpu-gb300]
  before_script:
    - source .venv/bin/activate
    - |
      TRT_LIB_DIR=$(python3 -c "import importlib.util; s=importlib.util.find_spec('tensorrt_libs'); print(s.submodule_search_locations[0])")
      export LD_LIBRARY_PATH="$TRT_LIB_DIR:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# ── Build ──────────────────────────────────────

build-all:
  stage: build
  script:
    - pip install --no-deps -e trtf_build/
    - cmake -S . -B build -G Ninja
      -DTRTF_TRT_INCLUDE_DIR=/usr/include/aarch64-linux-gnu
      -DTRTF_TRT_LIBRARY="$TRT_LIB_DIR/libnvinfer.so"
      -DTRTF_CUDA_INCLUDE_DIR=/usr/local/cuda/include
      -DTRTF_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so
    - cmake --build build -j
  artifacts:
    paths:
      - build/
    expire_in: 1 day
  timeout: 15m

# ── Tier 1: Unit Tests (no GPU needed) ~60s ───

test-python-builder:
  stage: tier1-unit
  needs: [build-all]
  script:
    - python -m pytest tests/builder/ -v --ignore=tests/builder/test_cli.py
  timeout: 5m

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

# ── Tier 2: Graph-Op GPU Tests ~2min ──────────

test-graph-ops:
  stage: tier2-graph-ops
  needs: [test-python-builder, test-cpp-unit]
  script:
    - nvidia-smi
    - python -m pytest tests/builder/test_graph_ops.py -v -m trt
  timeout: 10m

# ── Tier 3: E2E Smoke Test ~5min ──────────────

test-e2e-smoke:
  stage: tier3-smoke
  needs: [test-graph-ops]
  script:
    - python -m pytest tests/e2e/test_full_pipeline.py -v -k qwen3-0.6b
      --engine-dir $ENGINE_DIR
      --trtf-binary ./build/trtf
      --hf-python $PWD/.venv/bin/python
      --rebuild-engines
  timeout: 30m

# ── Tier 4: Full E2E ~90min (master or manual) ─

test-e2e-full:
  stage: tier4-full-e2e
  needs: [test-e2e-smoke]
  script:
    - python -m pytest tests/e2e/ -v
      --engine-dir $ENGINE_DIR
      --trtf-binary ./build/trtf
      --hf-python $PWD/.venv/bin/python
      --rebuild-engines
  timeout: 2h
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

- **Tiers 1-3** run on every push/MR (~10 min total)
- **Tier 4** runs automatically on `master` merges or manually on MRs (~90 min)

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `concurrent = 1` on runner | GPU jobs are memory-heavy, no parallelism |
| `gpus = "all"` | NVIDIA Container Toolkit passes GPUs through to Docker |
| Persistent volume mounts | Avoids re-downloading engine bundles and HF model cache |
| `--rebuild-engines` in E2E | Avoids testing stale cached bundles |
| `pull_policy = "if-not-present"` | Avoids pulling large CUDA image every time |
| Tier 4 manual on MRs | 90-min jobs shouldn't block every MR; auto on master |

## Prerequisites Checklist

- [ ] Docker + NVIDIA Container Toolkit installed on GB300
- [ ] GitLab Runner installed and registered
- [ ] `config.toml` edited with `gpus = "all"` and volume mounts
- [ ] `trtf-dev-gb300` Docker image pre-built on GB300
- [ ] `.gitlab-ci.yml` added to repo root and pushed
- [ ] Volume mount paths adjusted for GB300 storage layout

## References

- [GitLab Runner installation (Linux arm64)](https://docs.gitlab.com/runner/install/linux-manually/)
- [GitLab Runner GPU configuration](https://docs.gitlab.com/runner/configuration/gpus/)
- [GitLab CI/CD YAML syntax](https://docs.gitlab.com/ci/yaml/)
- [config.toml reference](https://docs.gitlab.com/runner/configuration/advanced-configuration/)
