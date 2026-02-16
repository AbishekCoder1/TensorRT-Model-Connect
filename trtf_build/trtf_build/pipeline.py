"""Thin Python wrapper around the C++ trtf CLI.

C++ is the single source of truth for inference. This wrapper calls the
trtf binary as a subprocess, providing a Pythonic API.

Usage:
    import trtf_build
    pipe = trtf_build.Pipeline("model.trtfb")
    print(pipe("Hello world", max_new_tokens=10))
    print(pipe("Describe this image", image="photo.jpg", max_new_tokens=50))
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class Pipeline:
    """Python wrapper that calls the C++ trtf CLI as a subprocess.

    Args:
        bundle_path: Path to a .trtfb bundle file.
        binary: Path to the trtf binary. Auto-detected if not specified.
        hf_python: Path to Python interpreter for tokenizer. Auto-detected.
    """

    def __init__(
        self,
        bundle_path: str,
        binary: str | None = None,
        hf_python: str | None = None,
    ):
        self.bundle_path = str(bundle_path)

        if binary is not None:
            self.binary = binary
        else:
            self.binary = self._find_binary()

        self.hf_python = hf_python

    def __call__(
        self,
        prompt: str,
        *,
        image: str | None = None,
        max_new_tokens: int = 20,
        timeout: float = 120.0,
    ) -> str:
        """Generate text (and optionally process an image).

        Args:
            prompt: Input text prompt.
            image: Optional path to an image file (for VL models).
            max_new_tokens: Maximum tokens to generate.
            timeout: Subprocess timeout in seconds.

        Returns:
            Generated text string.
        """
        cmd = [
            self.binary, "run", self.bundle_path,
            "--prompt", prompt,
            "--max-new-tokens", str(max_new_tokens),
        ]

        if image is not None:
            cmd.extend(["--image", str(image)])

        if self.hf_python is not None:
            cmd.extend(["--hf-python", self.hf_python])

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(
                f"trtf run failed (exit={result.returncode}): {stderr}")

        return result.stdout.strip()

    @staticmethod
    def _find_binary() -> str:
        """Auto-detect the trtf binary location."""
        # Check common locations
        candidates = [
            "./build/trtf",
            "trtf",
        ]
        for candidate in candidates:
            path = Path(candidate)
            if path.exists() and path.is_file():
                return str(path)

        # Check PATH
        found = shutil.which("trtf")
        if found is not None:
            return found

        raise FileNotFoundError(
            "trtf binary not found. Build it with: "
            "cmake -S . -B build -G Ninja && cmake --build build")

    def inspect(self) -> str:
        """Run trtf inspect on the bundle."""
        cmd = [self.binary, "inspect", self.bundle_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"trtf inspect failed: {result.stderr.strip()}")
        return result.stdout.strip()

    def __repr__(self) -> str:
        return f"Pipeline({self.bundle_path!r})"
