# Pipeline Orchestration

`pipeline_impl.*` contains the concrete `PipelineImpl` implementation.

Focus areas:
- Task-level request routing (`generate`, `encode`, `transcribe`, etc.)
- Runtime strategy handoff to strategy-specific backend instances
- Shared execution-time checks and response adaptation
