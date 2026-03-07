# TRT-Transformers-CPP Wiki

A split-language system for TensorRT inference: **Python builds** optimized TRT engines from HuggingFace model checkpoints, **C++ runs** them at maximum speed. The Python `trtf_build/` package reads safetensors, constructs TRT networks via the TensorRT Python API, and produces self-contained `.trtfb` bundles. The C++ runtime loads those bundles and executes one strategy-composed runtime graph per bundle.

## Quick Navigation

| Page | Description |
|------|-------------|
| **[Architecture Overview](Architecture-Overview.md)** | End-to-end architecture for the live system: Python builder, C++ runtime, strategy dispatch, and governance |
| **[Runtime Architecture Standard](Runtime-Target-Architecture.md)** | Authoritative runtime design rules for modularity, scalability, and unit-testability |
| **[Static Design](Static-Design.md)** | Current class/module structure for builders, services, router, ports, and executors |
| **[Dynamic Design](Dynamic-Design.md)** | Current runtime and build-time sequence diagrams |
| **[Pipeline Deep Dive](Pipeline-Deep-Dive.md)** | Detailed walkthrough of bundle loading, builder composition, and request routing |
| **[TRT Internals](TRT-Internals.md)** | TensorRT graph-building and engine lifecycle details |
| **[HF vs TRT Comparison](HF-vs-TRT-Comparison.md)** | HuggingFace reference flow versus TRT runtime flow |
| **[Adding a Model Family](Adding-a-Model-Family.md)** | Adding a new Python family plugin |
| **[Testing and Validation](Testing-and-Validation.md)** | Test tiers, smoke/E2E policy, and CCN gate |
| **[Coverage in GitLab](../coverage/gitlab_coverage_report.md)** | Coverage tooling and CI report format |
| **[Traceability Matrix](Traceability-Matrix.md)** | Architecture/design/test traceability requirements |
| **[Agent Orchestration](../AGENT_ORCHESTRATION.md)** | Multi-agent workflows used in this repo |
| **[Extensibility Assessment](Architecture-Extensibility-Assessment.md)** | Coverage of model families and runtime strategies |
| **[Source Layout](Source-Layout.md)** | File and directory guide for the current codebase |

## Core Design Principles

1. **Python builds, C++ runs**: checkpoint loading, graph construction, and engine compilation stay in Python; low-latency inference stays in C++.
2. **One runtime creation path**: `trtf_c.cpp -> strategy builder -> PipelineServices -> PipelineRouter` is the only runtime assembly flow.
3. **Strategy resolved once**: `runtime_strategy` is decoded at bundle-load time, not redispatched on every request.
4. **Services are the core runtime API**: `PipelineServices` is the internal capability model; `IPipeline` is a stable compatibility shell.
5. **IO lives at the edge**: file paths, media decode, and artifact persistence belong to adapters and the router, not the core services.
6. **Builders compose, services orchestrate, executors execute**: each layer owns one responsibility.
7. **Executor-heavy code must be split**: planner, tensor mapper, executor shell, and postprocessor are separate units whenever backend logic grows.
8. **Every module must be unit-testable**: pure seams get direct unit tests; side-effecting modules depend on fakeable ports.
9. **Complexity budget is enforced**: C++ CCN stays under the repository gate.
10. **Traceability is mandatory**: architecture decisions, unit design, and tests must stay linked.

## Architecture At A Glance

The system has two phases.

1. **Build phase (Python)**
   - `trtf-build build <hf-model-or-dir> -o model.trtfb`
   - Family plugin matches `config.json`
   - Checkpoint mapper normalizes weights
   - Graph builder emits TRT networks
   - Bundle writer packages engine plans and metadata into `.trtfb`

2. **Run phase (C++)**
   - `trtf` or the C ABI loads the `.trtfb`
   - `src/cabi/api/trtf_c.cpp` validates input and reads the bundle
   - Bundle and TRT adapters populate a canonical `BuildContext`
   - A strategy-family builder validates the bundle and composes `PipelineServices`
   - `PipelineRouter` exposes the stable `IPipeline` API
   - Service ports and TRT executors perform inference
   - IO adapters convert between user-facing files and canonical in-memory DTOs

```text
trtf_create_pipeline_ex()
  -> ReadBundleFile()
  -> BuildContext
  -> resolve strategy family
  -> StrategyBuilder::build()
  -> PipelineServices
  -> PipelineRouter
  -> service ports / adapters / TRT executors
```

## Runtime Design Rules

1. **Core services consume canonical DTOs**: text, image, audio, detection, segmentation, speech, and video requests are represented as structured in-memory data inside the runtime core.
2. **Router owns compatibility and file/media edges**: `PipelineRouter` translates `IPipeline` path-based calls to decoded media and writes artifacts back out.
3. **Builders own dependency wiring**: tokenizer access, bundle section lookup, TRT runtime handles, and service adapters are assembled in strategy builders.
4. **Ports isolate side effects**: filesystem, subprocesses, media decode, artifact writing, and TRT/CUDA calls sit behind interfaces that can be faked in tests.
5. **TRT backends stay thin**: low-level backends should mostly execute prepared plans rather than own high-level policy.
6. **Pure seams are first-class units**: scheduling, stopping, tensor-shape planning, conditioning logic, and postprocessing should live in helper modules with direct unit tests.

## Container Setup And Usage

All development and test workflows run inside the dev container.

```bash
./scripts/docker_build_gb300.sh
./scripts/docker_run_gb300.sh
```

Inside the container:

```bash
./scripts/setup_container.sh

trtf-build build Qwen/Qwen3-0.6B \
  -o /workspace/users/yifeif/trt-transformers/engines/qwen3.trtfb

./build/trtf run /workspace/users/yifeif/trt-transformers/engines/qwen3.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --hf-python /opt/venv/bin/python
```

## Minimum Viable Example

```bash
trtf-build build path/to/Qwen3-0.6B -o qwen3.trtfb

trtf run qwen3.trtfb --prompt "What is the capital of France?" --max-new-tokens 30
trtf encode bert.trtfb --prompt "hello"
trtf segment segformer.trtfb --image input.png --output mask.png
trtf transcribe whisper.trtfb --audio sample.wav
trtf generate-video wan_t2v.trtfb --prompt "A cat riding a bike" --output frames/
```

## Built-In Model Support

Family plugins are auto-discovered from `trtf_build/trtf_build/families/`; there is no static registration list to update. Any HF model whose `model_type` maps to a plugin is buildable.

Canonical source of truth in your checkout:

```bash
ls trtf_build/trtf_build/families/*.py \
  | sed 's|.*/||; s|\.py$||' \
  | rg -v '^(__init__|base)$' \
  | sort
```

As of **February 26, 2026**, the repository contains **44** family modules:

`bark`, `bert`, `bloom`, `codegen`, `deeponet`, `deepseek_ocr`, `deepseek_v2`, `eagle_vlm`, `falcon`, `flux`, `fno`, `gemma`, `gpt2`, `gpt_neo`, `gpt_neox`, `granite`, `internlm`, `internvl`, `llama`, `mamba`, `mistral`, `mixtral`, `nemotron`, `nemotron_h`, `olmo`, `opt`, `personaplex`, `phi`, `phi4_multimodal`, `phi_moe`, `qwen`, `qwen3_omni`, `qwen_moe`, `qwen_vl`, `rwkv`, `sam`, `segformer`, `stablelm`, `starcoder2`, `wan_t2v`, `whisper`, `xglm`, `yolox`, `z_image`.

These families map to runtime strategies that are composed by the C++ strategy builders. New families normally require Python-side plugin work; new runtime strategies require builder, service, and test coverage work in the C++ runtime.
