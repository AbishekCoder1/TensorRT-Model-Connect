# Recurrent Backends

Recurrent/stateful decode implementations.

Key files:
- `mamba_*`: Mamba state types, single-step runtime, and generate loop.
- `rwkv_*`: RWKV state types, single-step runtime, and generate loop.
- `hybrid_backend.*`: hybrid Mamba+attention runtime path.

How to understand:
1. Start with `*_backend.cpp` (`generate`).
2. Follow into `*_decode_runtime.*` for one-step TensorRT execution.
3. Inspect `*_step_state.*` for persistent state layout/updates.
