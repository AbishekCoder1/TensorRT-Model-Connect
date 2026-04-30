---
name: native-trt-builder-guidelines
description: >-
  Use when modifying TensorRT builders or shared graph construction code. Enforce
  strongly typed networks and shared helpers for basic TensorRT primitives, while
  allowing model-specific builder variants and general utility code to remain
  local when appropriate.
---

# Native TRT Builder Guidelines

## Principles

- Builders must create strongly typed networks:
  `builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))`.
- Basic TensorRT primitives should go through shared helpers in
  `graph_ops.py` or `graph_blocks.py` when the tensor contract and semantics
  match.
- Do not force reuse for general utility functions or model-specific building
  parts. A builder may keep local logic for custom dataflow, ordering, cache
  handling, bias terms, multimodal/task-specific behavior, or other architecture
  variants.
- If a native TRT primitive is insufficient for a basic operation, keep the
  fallback narrow and document the missing capability near the implementation.

## Workflow

1. Check network creation with `rg -n "create_network\\("`; every builder path
   should pass `NetworkDefinitionCreationFlag.STRONGLY_TYPED`.
2. Prefer shared helpers for exact duplicates of basic TRT primitive logic.
3. Extend a shared helper only when the change represents a reusable primitive
   contract. Do not introduce broad abstractions solely to collapse legitimate
   model-specific variants.
4. Validate with `git diff --check` and targeted compile/tests when available.
