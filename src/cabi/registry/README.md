# Backend Registry

Registry modules decouple runtime strategy names from concrete factory implementations.

Files:
- `backend_registry.*`: thread-safe registration/lookup API
- `backend_registry_dispatch.*`: one-time built-in registration entrypoint
- `backend_registry_strategy_wrappers.*`: typed wrappers that adapt dispatch context to factory calls
- `backend_registry_strategy_plugins.*`: plugin registration declarations
- `backend_registry_strategy_plugins_*.cpp`: per-domain built-in registrations (text/vision/encoder/audio/misc)

When introducing a new runtime strategy, add the wrapper here and register it in the appropriate plugin file.
