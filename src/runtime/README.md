# Runtime Layer

This directory contains the C++ inference runtime implementation.

- `trt/`: TensorRT-backed runtime modules, grouped by domain (core, recurrent, audio, multimodal, encoder, perception, diffusion).

Start from `src/cabi/` for pipeline assembly and dispatch, then follow includes into `src/runtime/trt/` backends.
