"""Pure-Python TRT inference runner for debugging and diff testing.

No C++ binary needed. Deserializes a TRT engine plan and runs
single-step autoregressive decoding with KV cache management.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import tensorrt as trt

# cuda-python >= 13 uses cuda.bindings.runtime; older versions use cuda.cudart.
try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart  # type: ignore[no-redef]


def _check_cuda(status):
    """Raise on CUDA error."""
    if hasattr(cudart, "cudaError_t"):
        success = cudart.cudaError_t.cudaSuccess
    else:
        success = 0
    if status != success:
        raise RuntimeError(f"CUDA error: {status}")


class TrtRunner:
    """Run inference on a TRT engine in Python.

    Supports autoregressive decoding with KV cache, matching the
    C++ runtime's behavior exactly.
    """

    def __init__(
        self,
        engine_plan: bytes,
        max_cache_length: int,
        num_layers: int,
        attention_size: int | None = None,
    ):
        self.max_cache_length = max_cache_length
        self.num_layers = num_layers

        # Deserialize engine
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_plan)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TRT engine")
        self.context = self.engine.create_execution_context()

        # Auto-detect attention_size from the cache_k_0 tensor shape
        if attention_size is None:
            cache_shape = tuple(self.engine.get_tensor_shape("cache_k_0"))
            attention_size = cache_shape[1]  # (max_cache_length, attention_size)
        self.attention_size = attention_size

        # Create CUDA stream
        err, self.stream = cudart.cudaStreamCreate()
        _check_cuda(err)

        # Track cache state (matches C++ KvCacheStepState)
        self.cache_length = 0  # number of valid entries in cache

        # Initialize KV cache (zeros)
        self.cache_k = [
            np.zeros((max_cache_length, attention_size), dtype=np.float32)
            for _ in range(num_layers)
        ]
        self.cache_v = [
            np.zeros((max_cache_length, attention_size), dtype=np.float32)
            for _ in range(num_layers)
        ]

        # Discover all output tensor names and shapes
        self._output_names = []
        self._output_shapes = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.engine.get_tensor_shape(name))
                self._output_names.append(name)
                self._output_shapes[name] = shape

        # Allocate device buffers for all tensors
        self._device_buffers: dict[str, int] = {}
        self._host_buffers: dict[str, np.ndarray] = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype_trt = self.engine.get_tensor_dtype(name)
            dtype_np = trt.nptype(dtype_trt)
            nbytes = int(np.prod(shape)) * np.dtype(dtype_np).itemsize

            err, d_ptr = cudart.cudaMalloc(nbytes)
            _check_cuda(err)
            self._device_buffers[name] = d_ptr
            self._host_buffers[name] = np.zeros(shape, dtype=dtype_np)

    def step(self, token_id: int) -> dict[str, np.ndarray]:
        """Run one decode step (manages position and cache internally).

        Args:
            token_id: Input token ID.

        Returns:
            Dict with 'logits' (1D float32 array of vocab size) and any
            debug outputs (e.g. 'debug_hidden_0', etc.).
        """
        attention_window = self.max_cache_length + 1

        # Position ID: min(cache_length, max_cache_length)
        position_id = min(self.cache_length, self.max_cache_length)

        # Build attention mask (matches C++ build_attention_mask exactly):
        #   mask[0..cache_length-1] = 0.0  (valid cached positions)
        #   mask[cache_length..max_cache-1] = -1e9  (unused cache slots)
        #   mask[max_cache_length] = 0.0  (current token slot)
        mask = np.full((1, attention_window), -1e9, dtype=np.float32)
        valid = min(self.cache_length, self.max_cache_length)
        mask[0, :valid] = 0.0
        mask[0, -1] = 0.0  # current token (last position in concat)

        # Prepare input host buffers
        self._host_buffers["token_id"][:] = np.array([token_id], dtype=np.int32)
        self._host_buffers["position_id"][:] = np.array([position_id], dtype=np.int32)
        self._host_buffers["attention_mask"][:] = mask

        for i in range(self.num_layers):
            ck_name = f"cache_k_{i}"
            cv_name = f"cache_v_{i}"
            self._host_buffers[ck_name][:] = self.cache_k[i]
            self._host_buffers[cv_name][:] = self.cache_v[i]

        # Copy inputs to device
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            self.context.set_tensor_address(name, self._device_buffers[name])

            if mode == trt.TensorIOMode.INPUT:
                h_buf = self._host_buffers[name]
                cudart.cudaMemcpyAsync(
                    self._device_buffers[name],
                    h_buf.ctypes.data,
                    h_buf.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self.stream,
                )

        # Execute
        self.context.execute_async_v3(self.stream)

        # Copy outputs to host
        results: dict[str, np.ndarray] = {}
        for name in self._output_names:
            h_buf = self._host_buffers[name]
            cudart.cudaMemcpyAsync(
                h_buf.ctypes.data,
                self._device_buffers[name],
                h_buf.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            )

        cudart.cudaStreamSynchronize(self.stream)

        # Collect results
        for name in self._output_names:
            results[name] = self._host_buffers[name].copy()

        # Update KV cache (matches C++ append_cache_state exactly)
        for i in range(self.num_layers):
            pk = results[f"present_k_{i}"].flatten()
            pv = results[f"present_v_{i}"].flatten()
            if self.cache_length < self.max_cache_length:
                self.cache_k[i][self.cache_length] = pk
                self.cache_v[i][self.cache_length] = pv
            else:
                # Shift left and append at the end
                self.cache_k[i][:-1] = self.cache_k[i][1:]
                self.cache_k[i][-1] = pk
                self.cache_v[i][:-1] = self.cache_v[i][1:]
                self.cache_v[i][-1] = pv

        self.cache_length = min(self.cache_length + 1, self.max_cache_length)
        return results

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int,
    ) -> list[dict[str, np.ndarray]]:
        """Run autoregressive generation.

        Args:
            input_ids: Prompt token IDs.
            max_new_tokens: Number of tokens to generate after the prompt.

        Returns:
            List of per-step result dicts. Each contains 'logits' and debug
            outputs. The list includes both prefill steps (one per input token)
            and generation steps.
        """
        all_results = []

        # Prefill: process input tokens one by one
        for tid in input_ids:
            result = self.step(tid)
            all_results.append(result)

        # Generate: autoregressive decoding
        for _ in range(max_new_tokens):
            last_logits = all_results[-1]["logits"].flatten()
            next_token = int(np.argmax(last_logits))
            result = self.step(next_token)
            all_results.append(result)

        return all_results

    def __del__(self):
        for d_ptr in self._device_buffers.values():
            cudart.cudaFree(d_ptr)
        if hasattr(self, "stream"):
            cudart.cudaStreamDestroy(self.stream)


def load_engine_from_bundle(bundle_path: str) -> tuple[bytes, dict]:
    """Load engine plan bytes and metadata from a .trtfb bundle.

    Returns:
        (engine_plan_bytes, header_dict)
    """
    import json
    import struct

    with open(bundle_path, "rb") as f:
        magic = f.read(8)
        if magic != b"TRTFB\x00\x01\x00":
            raise ValueError(f"Not a valid .trtfb bundle: {bundle_path}")
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        sections = header.get("sections", {})
        engine_meta = sections.get("engine_plan", {})
        f.seek(16 + header_len + engine_meta["offset"])
        engine_plan = f.read(engine_meta["size"])

    return engine_plan, header


def runner_from_bundle(bundle_path: str) -> TrtRunner:
    """Create a TrtRunner from a .trtfb bundle file."""
    engine_plan, header = load_engine_from_bundle(bundle_path)
    return TrtRunner(
        engine_plan=engine_plan,
        max_cache_length=header["max_cache_length"],
        num_layers=header["num_layers"],
    )
