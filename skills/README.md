# Skills

Project-specific Claude Code skills. Each skill is a `SKILL.md` file in its own subdirectory.

| Skill | Purpose |
|-------|---------|
| [profile-model](profile-model/SKILL.md) | E2E performance profiling — TRT vs HF vs torch.compile, per-layer timing, CPU phase breakdown, bottleneck classification |
| [debug-trt-mismatch](debug-trt-mismatch/SKILL.md) | 5-level numerical divergence investigation — logits, layers, VL, C++ parity, graph op isolation |
| [doc-sync](doc-sync/SKILL.md) | Daily documentation maintenance — ADR upkeep, wiki drift repair, traceability audit |
| [submit-gitlab-mr](submit-gitlab-mr/SKILL.md) | Push branch and create GitLab MR via glab, with automatic ADR creation |
| [mr-babysitter](mr-babysitter/SKILL.md) | Monitor GitLab MR CI pipelines, diagnose and fix failures |
| [fp16-trt-network](fp16-trt-network/SKILL.md) | Guide for building FP16 TensorRT networks in strongly-typed mode |
| [optimize-model-precision](optimize-model-precision/SKILL.md) | Autonomous precision optimization — find best low-precision config |
