# C ABI Runtime Edge

This directory now contains only the thin C ABI edge for the service-composed runtime.

Subdirectories:
- `api/`: exported C API entrypoint glue (`trtmc_c.cpp`)
- `config/`: `FastPathModelConfig` parsing from bundle `config.json`
- `bundle/`: shared bundle extraction and tokenizer/bundle helpers reused by the runtime builders

Suggested reading order:
1. `api/trtmc_c.cpp` (public API validation, bundle loading, builder composition entry)
2. `bundle/` (bundle metadata and tokenizer extraction helpers)
3. `config/` (config parsing consumed by builders and TRT executors)
