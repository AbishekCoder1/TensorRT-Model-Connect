# TensorRT-Model-Connect

TensorRT-Model-Connect turns HuggingFace-style model checkpoints into deployable TensorRT bundles and runs those bundles from a native C++ runtime.

[Documentation site](https://legendary-doodle-r37686v.pages.github.io/) | [GitHub Actions](https://github.com/NVIDIA-dev/TensorRT-Model-Connct/actions) | [Docs source](website/docs/intro.md)

The GitHub Pages site is private while this repository is private. Use the local documentation links below when reading from a checkout without Pages access.

## What It Does

- Builds `.trtfb` bundles from HuggingFace repo IDs or local model directories.
- Runs those bundles through `./build/trtmc` or the public C++ API in `include/trtmc/pipeline.h`.
- Keeps Python-heavy model conversion on the build side and C++/TensorRT execution on the runtime side.
- Supports raw TensorRT engines and Torch-TensorRT compiled engines behind the same bundle and runtime contract.
- Covers multiple task families, including text generation, vision-language generation, transcription, speech generation, diffusion image/video, segmentation, embeddings, reranking, neural operators, and time-series models.

## Quick Links

| Need | GitHub Pages | Source in this repo |
| --- | --- | --- |
| Start from zero | [Learning Path](https://legendary-doodle-r37686v.pages.github.io/learning-path) | [website/docs/learning-path.md](website/docs/learning-path.md) |
| Set up the environment | [Environment and First Repro](https://legendary-doodle-r37686v.pages.github.io/getting-started/environment-and-repro) | [website/docs/getting-started/environment-and-repro.md](website/docs/getting-started/environment-and-repro.md) |
| Build and run one model | [Quick Start](https://legendary-doodle-r37686v.pages.github.io/getting-started/quick-start) | [website/docs/getting-started/quick-start.md](website/docs/getting-started/quick-start.md) |
| Check supported workloads | [Model Support](https://legendary-doodle-r37686v.pages.github.io/getting-started/model-support) | [website/docs/getting-started/model-support.md](website/docs/getting-started/model-support.md) |
| Use the CLI | [CLI Reference](https://legendary-doodle-r37686v.pages.github.io/api/cli-reference) | [website/docs/api/cli-reference.md](website/docs/api/cli-reference.md) |
| Use Python or C++ APIs | [API Overview](https://legendary-doodle-r37686v.pages.github.io/api/overview) | [website/docs/api/overview.md](website/docs/api/overview.md) |
| Understand internals | [Architecture](https://legendary-doodle-r37686v.pages.github.io/architecture/overview) | [website/docs/architecture/overview.md](website/docs/architecture/overview.md) |
| Run validation | [Testing](https://legendary-doodle-r37686v.pages.github.io/reference/testing) | [website/docs/reference/testing.md](website/docs/reference/testing.md) |

## Quick Start

The standard workflow runs inside the project dev container.

```bash
./scripts/docker_build_gb300.sh
./scripts/docker_run_gb300.sh
```

Inside the container, install the Python builder and build the C++ runtime:

```bash
pip install -e tensorrt_model_connect/
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Build a small text-generation bundle and run it from C++:

```bash
trtmc-build build Qwen/Qwen3-0.6B \
  -o /tmp/qwen3.trtfb \
  --max-cache-length 256

./build/trtmc run /tmp/qwen3.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --hf-python /opt/venv/bin/python
```

See [Installation](website/docs/getting-started/installation.md) and [Build and Run](website/docs/getting-started/build-and-run.md) for the full setup and workload-specific examples.

## How It Fits Together

| Phase | Main entry point | Responsibility |
| --- | --- | --- |
| Build | `trtmc-build` and `tensorrt_model_connect.build()` | Resolve model assets, select a family plugin, build TensorRT engine plans, and package a `.trtfb` bundle. |
| Bundle | `.trtfb` artifact | Carry engine plans, tokenizer assets, model metadata, backend selection, and runtime strategy. |
| Run | `./build/trtmc` and `trtmc::load()` | Load the bundle, choose the runtime plugin, deserialize engines, and expose task APIs. |

Two build paths produce the same runtime artifact:

| Pipeline | Use it when | Runtime dependency |
| --- | --- | --- |
| Raw TensorRT | You need maximum control over the TensorRT network and handwritten graph ops. | TensorRT backend DSO loaded by the C++ runtime. |
| Torch-TensorRT | You want to compile an exported PyTorch graph without writing the TensorRT graph by hand. | Raw TensorRT plan inside the same `.trtfb` bundle. No LibTorch dependency at request time. |

## API Examples

Python builder:

```python
import tensorrt_model_connect

tensorrt_model_connect.build(
    "Qwen/Qwen3-0.6B",
    "/tmp/qwen3.trtfb",
    max_cache_length=256,
    precision="fp16",
)
```

C++ runtime:

```cpp
#include <trtmc/pipeline.h>

#include <iostream>

int main() {
    auto pipe = trtmc::load("/tmp/qwen3.trtfb", "/opt/venv/bin/python");

    trtmc::GenerateConfig cfg;
    cfg.max_new_tokens = 20;

    auto out = pipe->generate("The capital of France is", cfg);
    std::cout << out.text << "\n";
}
```

More detail:

- [Python Builder API](website/docs/api/python-builder.md)
- [C++ and C ABI](website/docs/api/cpp-api.md)
- [CLI Reference](website/docs/api/cli-reference.md)

## Repository Layout

| Path | Purpose |
| --- | --- |
| `tensorrt_model_connect/` | Python package and `trtmc-build` console script. |
| `include/trtmc/` | Public C++ headers. |
| `src/` | C++ runtime, backend loading, runtime plugins, and task pipelines. |
| `examples/` | Small runnable C++ examples and the `trtmc` CLI entry point. |
| `tests/` | Builder, C++, tool, E2E, and report-generation tests. |
| `website/` | Docusaurus documentation published to GitHub Pages. |
| `.github/workflows/trtmc-ci.yml` | GitHub CI gates, including premerge-style checks and full nightly E2E mode. |
| `.github/workflows/pages.yml` | GitHub Pages build and deployment workflow. |

## Validation

Run the fast local checks inside the dev container:

```bash
pytest tests/builder -q
ctest --test-dir build --output-on-failure
```

Run an E2E model path when engines and model assets are available:

```bash
pytest tests/test_e2e.py \
  --engine-dir /path/to/engines \
  --trtmc-binary ./build/trtmc \
  --hf-python /opt/venv/bin/python
```

The GitHub workflow separates normal branch or pull-request checks from scheduled and manually requested full E2E runs. See [Testing](website/docs/reference/testing.md) for the full matrix and artifact expectations.

## Project Status

This repository is under active development. Treat runtime strategies, model-family coverage, and bundle compatibility as evolving unless a release note states otherwise.

This checkout currently does not include `LICENSE`, `CONTRIBUTING`, `SECURITY`, or `CODE_OF_CONDUCT` files. Add those policy files before treating the repository as externally releasable.

## Help

Start with the [Documentation site](https://legendary-doodle-r37686v.pages.github.io/) for user docs, [Environment and First Repro](website/docs/getting-started/environment-and-repro.md) for setup failures, and [GitHub Actions](https://github.com/NVIDIA-dev/TensorRT-Model-Connct/actions) for the latest CI state on the GitHub mirror.
