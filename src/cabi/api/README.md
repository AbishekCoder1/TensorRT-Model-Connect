# C API Entrypoint

`trtf_c.cpp` implements exported C API functions and wires API calls to the runtime pipeline.

Focus areas:
- Handle validation and error mapping
- Bundle open/close lifecycle
- Pipeline creation and command dispatch
- Registry bootstrap (`register_builtin_backend_factories_once`)
