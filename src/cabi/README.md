# C ABI Runtime Assembly

This directory contains pipeline assembly and strategy dispatch code behind the public C API.

Subdirectories:
- `api/`: C API entrypoint glue (`trtf_c.cpp`).
- `pipeline/`: `PipelineImpl` orchestration and end-to-end API behavior.
- `config/`: `FastPathModelConfig` parsing from bundle `config.json`.
- `bundle/`: shared bundle extraction and TensorRT engine/tokenizer helpers.
- `factories/`: strategy-specific backend construction modules.
- `registry/`: backend factory registry, dispatch, and built-in plugin registration.

Suggested reading order:
1. `api/trtf_c.cpp` (public API entry and lifecycle)
2. `pipeline/pipeline_impl.*` (runtime pipeline behavior)
3. `registry/` + `factories/` (strategy selection and backend construction)
4. `bundle/` + `config/` (bundle parsing and config wiring)
