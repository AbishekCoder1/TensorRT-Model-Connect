"""Tests for pipeline.py — Python wrapper around the C++ trtf CLI.

Pure Python tests with mocked subprocess calls. No GPU or TRT needed.

Trace: ARCH-FAC-001, UD-FAC-PIPELINE
Intent: Validate Pipeline subprocess wrapper init, binary detection, and CLI argument construction
Preconditions: subprocess calls are mocked; no real C++ binary or GPU required
Postconditions: Pipeline correctly stores paths, auto-detects binary, and constructs valid subprocess commands
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

try:
    from trtf_build.pipeline import Pipeline
except (ImportError, ModuleNotFoundError):
    pytest.skip("trtf_build requires tensorrt", allow_module_level=True)


# ---------------------------------------------------------------------------
# Pipeline.__init__
# ---------------------------------------------------------------------------


class TestPipelineInit:
    def test_explicit_binary(self):
        """Explicit binary path is stored directly, no auto-detection."""
        pipe = Pipeline("/tmp/model.trtfb", binary="/usr/bin/trtf")
        assert pipe.binary == "/usr/bin/trtf"
        assert pipe.bundle_path == "/tmp/model.trtfb"
        assert pipe.hf_python is None

    def test_explicit_binary_and_hf_python(self):
        """Both binary and hf_python are stored when provided."""
        pipe = Pipeline(
            "/tmp/model.trtfb",
            binary="/usr/bin/trtf",
            hf_python="/opt/venv/bin/python",
        )
        assert pipe.binary == "/usr/bin/trtf"
        assert pipe.hf_python == "/opt/venv/bin/python"

    def test_auto_detect_calls_find_binary(self):
        """When binary is None, _find_binary is called."""
        with patch.object(Pipeline, "_find_binary", return_value="/auto/trtf"):
            pipe = Pipeline("/tmp/model.trtfb")
            assert pipe.binary == "/auto/trtf"

    def test_binary_not_found_raises(self):
        """_find_binary raises FileNotFoundError when nothing is found."""
        with patch.object(
            Pipeline, "_find_binary",
            side_effect=FileNotFoundError("trtf binary not found"),
        ):
            with pytest.raises(FileNotFoundError, match="trtf binary not found"):
                Pipeline("/tmp/model.trtfb")

    def test_bundle_path_converted_to_str(self):
        """Path objects are converted to str."""
        pipe = Pipeline(Path("/tmp/model.trtfb"), binary="/usr/bin/trtf")
        assert isinstance(pipe.bundle_path, str)
        assert pipe.bundle_path == "/tmp/model.trtfb"

    def test_repr(self):
        """__repr__ includes the bundle path."""
        pipe = Pipeline("/tmp/model.trtfb", binary="/usr/bin/trtf")
        assert repr(pipe) == "Pipeline('/tmp/model.trtfb')"


# ---------------------------------------------------------------------------
# Pipeline.__call__
# ---------------------------------------------------------------------------


class TestPipelineCall:
    def _make_pipeline(self, hf_python=None):
        return Pipeline(
            "/tmp/model.trtfb", binary="/usr/bin/trtf", hf_python=hf_python)

    def test_basic_prompt(self):
        """Basic text prompt constructs correct command."""
        pipe = self._make_pipeline()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Hello world!\n"

        with patch("trtf_build.pipeline.subprocess.run",
                    return_value=mock_result) as mock_run:
            output = pipe("Say hello", max_new_tokens=5)

            mock_run.assert_called_once_with(
                [
                    "/usr/bin/trtf", "run", "/tmp/model.trtfb",
                    "--prompt", "Say hello",
                    "--max-new-tokens", "5",
                ],
                capture_output=True,
                text=True,
                timeout=120.0,
            )
            assert output == "Hello world!"

    def test_prompt_with_image(self):
        """Image argument is appended to the command."""
        pipe = self._make_pipeline()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "A cat sitting on a couch\n"

        with patch("trtf_build.pipeline.subprocess.run",
                    return_value=mock_result) as mock_run:
            output = pipe(
                "Describe this image",
                image="/tmp/photo.jpg",
                max_new_tokens=30,
            )

            cmd = mock_run.call_args[0][0]
            assert "--image" in cmd
            idx = cmd.index("--image")
            assert cmd[idx + 1] == "/tmp/photo.jpg"
            assert output == "A cat sitting on a couch"

    def test_prompt_without_image(self):
        """When image is None, --image is not in the command."""
        pipe = self._make_pipeline()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "output"

        with patch("trtf_build.pipeline.subprocess.run",
                    return_value=mock_result) as mock_run:
            pipe("Hello")
            cmd = mock_run.call_args[0][0]
            assert "--image" not in cmd

    def test_hf_python_appended(self):
        """When hf_python is set, --hf-python is in the command."""
        pipe = self._make_pipeline(hf_python="/opt/venv/bin/python")
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "output"

        with patch("trtf_build.pipeline.subprocess.run",
                    return_value=mock_result) as mock_run:
            pipe("Hello")
            cmd = mock_run.call_args[0][0]
            assert "--hf-python" in cmd
            idx = cmd.index("--hf-python")
            assert cmd[idx + 1] == "/opt/venv/bin/python"

    def test_hf_python_not_appended_when_none(self):
        """When hf_python is None, --hf-python is not in the command."""
        pipe = self._make_pipeline(hf_python=None)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "output"

        with patch("trtf_build.pipeline.subprocess.run",
                    return_value=mock_result) as mock_run:
            pipe("Hello")
            cmd = mock_run.call_args[0][0]
            assert "--hf-python" not in cmd

    def test_nonzero_returncode_raises(self):
        """Non-zero exit code raises RuntimeError with stderr."""
        pipe = self._make_pipeline()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "CUDA out of memory"

        with patch("trtf_build.pipeline.subprocess.run",
                    return_value=mock_result):
            with pytest.raises(RuntimeError, match="trtf run failed"):
                pipe("Hello")

    def test_nonzero_returncode_includes_exit_code(self):
        """RuntimeError message includes the exit code."""
        pipe = self._make_pipeline()
        mock_result = MagicMock()
        mock_result.returncode = 137
        mock_result.stderr = "Killed"

        with patch("trtf_build.pipeline.subprocess.run",
                    return_value=mock_result):
            with pytest.raises(RuntimeError, match="exit=137"):
                pipe("Hello")

    def test_subprocess_timeout(self):
        """subprocess.TimeoutExpired is propagated."""
        pipe = self._make_pipeline()

        with patch(
            "trtf_build.pipeline.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="trtf", timeout=5.0),
        ):
            with pytest.raises(subprocess.TimeoutExpired):
                pipe("Hello", timeout=5.0)

    def test_custom_timeout(self):
        """Custom timeout is forwarded to subprocess.run."""
        pipe = self._make_pipeline()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "out"

        with patch("trtf_build.pipeline.subprocess.run",
                    return_value=mock_result) as mock_run:
            pipe("Hello", timeout=60.0)
            assert mock_run.call_args[1]["timeout"] == 60.0

    def test_default_max_new_tokens(self):
        """Default max_new_tokens is 20."""
        pipe = self._make_pipeline()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "out"

        with patch("trtf_build.pipeline.subprocess.run",
                    return_value=mock_result) as mock_run:
            pipe("Hello")
            cmd = mock_run.call_args[0][0]
            idx = cmd.index("--max-new-tokens")
            assert cmd[idx + 1] == "20"

    def test_stdout_stripped(self):
        """Leading/trailing whitespace in stdout is stripped."""
        pipe = self._make_pipeline()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "\n  Hello world  \n\n"

        with patch("trtf_build.pipeline.subprocess.run",
                    return_value=mock_result):
            assert pipe("x") == "Hello world"


# ---------------------------------------------------------------------------
# Pipeline.inspect
# ---------------------------------------------------------------------------


class TestPipelineInspect:
    def test_inspect_success(self):
        """inspect() returns stripped stdout."""
        pipe = Pipeline("/tmp/model.trtfb", binary="/usr/bin/trtf")
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Model ID:  qwen3\nLayers: 28\n"

        with patch("trtf_build.pipeline.subprocess.run",
                    return_value=mock_result) as mock_run:
            out = pipe.inspect()
            mock_run.assert_called_once_with(
                ["/usr/bin/trtf", "inspect", "/tmp/model.trtfb"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert out == "Model ID:  qwen3\nLayers: 28"

    def test_inspect_failure_raises(self):
        """Non-zero exit code in inspect raises RuntimeError."""
        pipe = Pipeline("/tmp/model.trtfb", binary="/usr/bin/trtf")
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "not a valid .trtfb bundle"

        with patch("trtf_build.pipeline.subprocess.run",
                    return_value=mock_result):
            with pytest.raises(RuntimeError, match="trtf inspect failed"):
                pipe.inspect()


# ---------------------------------------------------------------------------
# Pipeline._find_binary
# ---------------------------------------------------------------------------


class TestFindBinary:
    def test_build_trtf_exists(self, tmp_path):
        """Finds ./build/trtf when it exists."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        trtf = build_dir / "trtf"
        trtf.touch()

        # Patch Path to make ./build/trtf resolve to our temp file
        with patch("trtf_build.pipeline.Path") as MockPath:
            def path_side_effect(p):
                if p == "./build/trtf":
                    return trtf
                mock_p = MagicMock()
                mock_p.exists.return_value = False
                mock_p.is_file.return_value = False
                return mock_p

            MockPath.side_effect = path_side_effect
            result = Pipeline._find_binary()
            assert result == str(trtf)

    def test_found_on_path(self):
        """Falls back to shutil.which when local candidates missing."""
        with patch("trtf_build.pipeline.Path") as MockPath:
            mock_p = MagicMock()
            mock_p.exists.return_value = False
            mock_p.is_file.return_value = False
            MockPath.return_value = mock_p

            with patch("trtf_build.pipeline.shutil.which",
                        return_value="/usr/local/bin/trtf"):
                result = Pipeline._find_binary()
                assert result == "/usr/local/bin/trtf"

    def test_not_found_anywhere(self):
        """Raises FileNotFoundError when trtf is not found anywhere."""
        with patch("trtf_build.pipeline.Path") as MockPath:
            mock_p = MagicMock()
            mock_p.exists.return_value = False
            mock_p.is_file.return_value = False
            MockPath.return_value = mock_p

            with patch("trtf_build.pipeline.shutil.which",
                        return_value=None):
                with pytest.raises(FileNotFoundError, match="trtf binary not found"):
                    Pipeline._find_binary()
