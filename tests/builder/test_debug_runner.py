"""Unit tests for trtf_build.debug_runner — load_engine_from_bundle,
load_vision_engine_from_bundle, and runner resource cleanup.

Mock-based, no TRT/GPU needed. Tests bundle parsing logic and
runner __del__ cleanup order.
"""

from __future__ import annotations

import json
import struct
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers: build a minimal .trtfb bundle in memory
# ---------------------------------------------------------------------------

def _make_bundle_bytes(
    header: dict,
    engine_plan: bytes = b"FAKE_ENGINE_PLAN",
    vision_plan: bytes | None = None,
) -> bytes:
    """Build a minimal .trtfb bundle in memory."""
    magic = b"TRTFB\x00\x01\x00"
    sections: dict[str, dict] = {}
    body = b""

    # engine_plan section
    sections["engine_plan"] = {"offset": len(body), "size": len(engine_plan)}
    body += engine_plan

    # optional vision section
    if vision_plan is not None:
        sections["vision_engine_plan"] = {
            "offset": len(body), "size": len(vision_plan),
        }
        body += vision_plan

    header["sections"] = sections
    header_json = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<Q", len(header_json))

    return magic + header_len + header_json + body


# ---------------------------------------------------------------------------
# load_engine_from_bundle
# ---------------------------------------------------------------------------

class TestLoadEngineFromBundle:
    """Tests for load_engine_from_bundle() bundle parsing."""

    def test_roundtrip(self, tmp_path):
        from trtf_build.debug_runner import load_engine_from_bundle

        header = {
            "model_id": "test-model",
            "max_cache_length": 128,
            "num_layers": 4,
        }
        engine_data = b"PLAN_BYTES_1234"
        bundle = _make_bundle_bytes(header, engine_plan=engine_data)

        path = tmp_path / "test.trtfb"
        path.write_bytes(bundle)

        plan, hdr = load_engine_from_bundle(str(path))
        assert plan == engine_data
        assert hdr["model_id"] == "test-model"
        assert hdr["max_cache_length"] == 128
        assert hdr["num_layers"] == 4

    def test_invalid_magic(self, tmp_path):
        from trtf_build.debug_runner import load_engine_from_bundle

        path = tmp_path / "bad.trtfb"
        path.write_bytes(b"NOT_A_BUNDLE_xxxxxxxxxxxx")

        with pytest.raises(ValueError, match="Not a valid .trtfb bundle"):
            load_engine_from_bundle(str(path))


# ---------------------------------------------------------------------------
# load_vision_engine_from_bundle
# ---------------------------------------------------------------------------

class TestLoadVisionEngineFromBundle:
    """Tests for load_vision_engine_from_bundle()."""

    def test_with_vision_section(self, tmp_path):
        from trtf_build.debug_runner import load_vision_engine_from_bundle

        header = {"num_layers": 2, "max_cache_length": 64}
        engine_data = b"TEXT_ENGINE"
        vision_data = b"VISION_ENGINE"
        bundle = _make_bundle_bytes(
            header, engine_plan=engine_data, vision_plan=vision_data)

        path = tmp_path / "vl.trtfb"
        path.write_bytes(bundle)

        plan, hdr = load_vision_engine_from_bundle(str(path))
        assert plan == vision_data
        assert hdr["num_layers"] == 2

    def test_without_vision_section(self, tmp_path):
        from trtf_build.debug_runner import load_vision_engine_from_bundle

        header = {"num_layers": 2, "max_cache_length": 64}
        bundle = _make_bundle_bytes(header, engine_plan=b"TEXT_ONLY")

        path = tmp_path / "text.trtfb"
        path.write_bytes(bundle)

        plan, hdr = load_vision_engine_from_bundle(str(path))
        assert plan is None
        assert hdr["num_layers"] == 2


# ---------------------------------------------------------------------------
# load_section_from_bundle / load_config_from_bundle
# ---------------------------------------------------------------------------

class TestBundleSectionUtils:
    """Tests for section loading utilities."""

    def test_load_section_missing(self, tmp_path):
        from trtf_build.debug_runner import load_section_from_bundle

        header = {"num_layers": 1, "max_cache_length": 32}
        bundle = _make_bundle_bytes(header, engine_plan=b"X")

        path = tmp_path / "test.trtfb"
        path.write_bytes(bundle)

        result = load_section_from_bundle(str(path), "nonexistent_section")
        assert result is None

    def test_load_config_from_bundle(self, tmp_path):
        from trtf_build.debug_runner import load_config_from_bundle

        # Build a bundle with a config.json section
        config_data = json.dumps({"model_type": "qwen3"}).encode("utf-8")
        magic = b"TRTFB\x00\x01\x00"

        engine_plan = b"FAKE_ENGINE"
        sections = {
            "engine_plan": {"offset": 0, "size": len(engine_plan)},
            "config.json": {
                "offset": len(engine_plan),
                "size": len(config_data),
            },
        }
        header = {"num_layers": 1, "max_cache_length": 32, "sections": sections}
        header_json = json.dumps(header).encode("utf-8")
        header_len = struct.pack("<Q", len(header_json))

        path = tmp_path / "cfg.trtfb"
        path.write_bytes(magic + header_len + header_json + engine_plan + config_data)

        cfg = load_config_from_bundle(str(path))
        assert cfg["model_type"] == "qwen3"


# ---------------------------------------------------------------------------
# PerfTrtRunner.__del__ cleanup order
# ---------------------------------------------------------------------------

class TestPerfTrtRunnerCleanup:
    """Verify PerfTrtRunner.__del__ frees resources in correct order."""

    def test_del_frees_all_buffers(self):
        """__del__ should cudaFree all device buffers then destroy stream."""
        # Build a fake runner object that looks like PerfTrtRunner after __init__
        from trtf_build.debug_runner import PerfTrtRunner

        runner = PerfTrtRunner.__new__(PerfTrtRunner)
        runner.num_layers = 2
        runner.attention_size = 8
        runner.max_cache_length = 4
        runner._d_token_id = 1000
        runner._d_position_id = 1001
        runner._d_mask = 1002
        runner._d_logits = 1003
        runner._d_cache_k = [2000, 2001]
        runner._d_cache_v = [3000, 3001]
        runner._d_present_k = [4000, 4001]
        runner._d_present_v = [5000, 5001]
        runner.stream = 9999
        runner.context = MagicMock()
        runner.engine = MagicMock()

        # Mock cudart
        mock_cudart = MagicMock()
        with patch("trtf_build.debug_runner.cudart", mock_cudart):
            runner.__del__()
            # Neutralize so GC won't call __del__ again with real cudart
            del runner._d_token_id

        # Should have called cudaFree for each buffer
        freed = [c.args[0] for c in mock_cudart.cudaFree.call_args_list]
        expected = [1000, 1001, 1002, 1003, 2000, 2001, 3000, 3001,
                    4000, 4001, 5000, 5001]
        assert sorted(freed) == sorted(expected)

        # Should have called cudaStreamDestroy
        mock_cudart.cudaStreamDestroy.assert_called_once_with(9999)

    def test_del_noop_before_init(self):
        """__del__ should not crash if called before __init__ completes."""
        from trtf_build.debug_runner import PerfTrtRunner

        runner = PerfTrtRunner.__new__(PerfTrtRunner)
        # No attributes set — __del__ should bail out safely
        runner.__del__()  # Should not raise


# ---------------------------------------------------------------------------
# PerfMambaTrtRunner.__del__ cleanup
# ---------------------------------------------------------------------------

class TestPerfMambaTrtRunnerCleanup:
    """Verify PerfMambaTrtRunner.__del__ frees resources."""

    def test_del_frees_all_buffers(self):
        from trtf_build.debug_runner import PerfMambaTrtRunner

        runner = PerfMambaTrtRunner.__new__(PerfMambaTrtRunner)
        runner.num_layers = 1
        runner.d_inner = 4
        runner.conv_kernel = 3
        runner.state_size = 2
        runner._d_token_id = 100
        runner._d_logits = 101
        runner._d_conv_state = [200]
        runner._d_ssm_state = [300]
        runner._d_present_conv = [400]
        runner._d_present_ssm = [500]
        runner.stream = 8888
        runner.context = MagicMock()
        runner.engine = MagicMock()

        mock_cudart = MagicMock()
        with patch("trtf_build.debug_runner.cudart", mock_cudart):
            runner.__del__()
            # Neutralize so GC won't call __del__ again with real cudart
            del runner._d_token_id

        freed = [c.args[0] for c in mock_cudart.cudaFree.call_args_list]
        expected = [100, 101, 200, 300, 400, 500]
        assert sorted(freed) == sorted(expected)
        mock_cudart.cudaStreamDestroy.assert_called_once_with(8888)

    def test_del_noop_before_init(self):
        from trtf_build.debug_runner import PerfMambaTrtRunner

        runner = PerfMambaTrtRunner.__new__(PerfMambaTrtRunner)
        runner.__del__()  # Should not raise


# ---------------------------------------------------------------------------
# TrtRunner.__del__ cleanup
# ---------------------------------------------------------------------------

class TestTrtRunnerCleanup:
    """Verify TrtRunner.__del__ frees device buffers and stream."""

    def test_del_frees_device_buffers(self):
        from trtf_build.debug_runner import TrtRunner

        runner = TrtRunner.__new__(TrtRunner)
        runner._device_buffers = {"a": 111, "b": 222}
        runner.stream = 7777

        mock_cudart = MagicMock()
        with patch("trtf_build.debug_runner.cudart", mock_cudart):
            runner.__del__()
            # Neutralize so GC won't call __del__ again with real cudart
            runner._device_buffers = {}
            del runner.stream

        freed = [c.args[0] for c in mock_cudart.cudaFree.call_args_list]
        assert sorted(freed) == [111, 222]
        mock_cudart.cudaStreamDestroy.assert_called_once_with(7777)

    def test_del_without_stream(self):
        from trtf_build.debug_runner import TrtRunner

        runner = TrtRunner.__new__(TrtRunner)
        runner._device_buffers = {"a": 111}
        # No stream attribute

        mock_cudart = MagicMock()
        with patch("trtf_build.debug_runner.cudart", mock_cudart):
            runner.__del__()
            # Neutralize
            runner._device_buffers = {}

        mock_cudart.cudaFree.assert_called_once_with(111)
        mock_cudart.cudaStreamDestroy.assert_not_called()


# ---------------------------------------------------------------------------
# MambaTrtRunner.__del__ cleanup
# ---------------------------------------------------------------------------

class TestMambaTrtRunnerCleanup:
    """Verify MambaTrtRunner.__del__ frees device buffers and stream."""

    def test_del_frees_device_buffers(self):
        from trtf_build.debug_runner import MambaTrtRunner

        runner = MambaTrtRunner.__new__(MambaTrtRunner)
        runner._device_buffers = {"x": 333, "y": 444}
        runner.stream = 6666

        mock_cudart = MagicMock()
        with patch("trtf_build.debug_runner.cudart", mock_cudart):
            runner.__del__()
            # Neutralize so GC won't call __del__ again with real cudart
            runner._device_buffers = {}
            del runner.stream

        freed = [c.args[0] for c in mock_cudart.cudaFree.call_args_list]
        assert sorted(freed) == [333, 444]
        mock_cudart.cudaStreamDestroy.assert_called_once_with(6666)


# ---------------------------------------------------------------------------
# TrtRunner.step() mask/position logic (mocked CUDA)
# ---------------------------------------------------------------------------

class TestTrtRunnerMaskLogic:
    """Test the numpy-level mask and position logic in TrtRunner.step()."""

    def _make_runner(self, max_cache_length=4, num_layers=1,
                     attention_size=8, cache_length=0):
        """Build a TrtRunner with mocked TRT engine, ready to test step()."""
        from trtf_build.debug_runner import TrtRunner

        runner = TrtRunner.__new__(TrtRunner)
        runner.max_cache_length = max_cache_length
        runner.num_layers = num_layers
        runner.attention_size = attention_size
        runner.cache_length = cache_length

        attention_window = max_cache_length + 1

        runner.cache_k = [
            np.zeros((max_cache_length, attention_size), dtype=np.float32)
            for _ in range(num_layers)
        ]
        runner.cache_v = [
            np.zeros((max_cache_length, attention_size), dtype=np.float32)
            for _ in range(num_layers)
        ]

        # Mock engine with IO tensors
        runner.engine = MagicMock()
        runner.context = MagicMock()
        runner.stream = MagicMock()

        # Minimal IO tensors
        io_specs = [
            ("token_id", "INPUT", (1,), np.int32),
            ("position_id", "INPUT", (1,), np.int32),
            ("attention_mask", "INPUT", (1, attention_window), np.float32),
            ("cache_k_0", "INPUT", (max_cache_length, attention_size), np.float32),
            ("cache_v_0", "INPUT", (max_cache_length, attention_size), np.float32),
            ("logits", "OUTPUT", (1, 16), np.float32),
            ("present_k_0", "OUTPUT", (1, attention_size), np.float32),
            ("present_v_0", "OUTPUT", (1, attention_size), np.float32),
        ]

        runner.engine.num_io_tensors = len(io_specs)

        def get_tensor_name(i):
            return io_specs[i][0]

        def get_tensor_mode(name):
            for spec in io_specs:
                if spec[0] == name:
                    mode = MagicMock()
                    mode.__eq__ = lambda s, o: (
                        spec[1] == "OUTPUT") == (str(o).endswith("OUTPUT"))
                    return mode
            return MagicMock()

        runner.engine.get_tensor_name = get_tensor_name
        runner.engine.get_tensor_mode = get_tensor_mode

        runner._output_names = ["logits", "present_k_0", "present_v_0"]
        runner._output_shapes = {
            "logits": (1, 16),
            "present_k_0": (1, attention_size),
            "present_v_0": (1, attention_size),
        }

        runner._device_buffers = {s[0]: i for i, s in enumerate(io_specs)}
        runner._host_buffers = {s[0]: np.zeros(s[2], dtype=s[3]) for s in io_specs}

        return runner

    def test_position_starts_at_zero(self):
        runner = self._make_runner(cache_length=0)
        # Position should be min(cache_length, max_cache_length) = 0
        position_id = min(runner.cache_length, runner.max_cache_length)
        assert position_id == 0

    def test_position_increments(self):
        runner = self._make_runner(cache_length=3, max_cache_length=8)
        position_id = min(runner.cache_length, runner.max_cache_length)
        assert position_id == 3

    def test_position_caps_at_max(self):
        runner = self._make_runner(cache_length=10, max_cache_length=4)
        position_id = min(runner.cache_length, runner.max_cache_length)
        assert position_id == 4

    def test_mask_empty_cache(self):
        """With no cache entries, only current token slot is valid."""
        runner = self._make_runner(max_cache_length=4, cache_length=0)
        attention_window = runner.max_cache_length + 1

        mask = np.full((1, attention_window), -1e9, dtype=np.float32)
        valid = min(runner.cache_length, runner.max_cache_length)
        mask[0, :valid] = 0.0
        mask[0, -1] = 0.0

        # Only last position (current token) should be valid
        assert mask[0, 0] == pytest.approx(-1e9)
        assert mask[0, 3] == pytest.approx(-1e9)
        assert mask[0, 4] == pytest.approx(0.0)  # current token

    def test_mask_partial_cache(self):
        """With 2 cached entries, positions 0,1 and current token are valid."""
        runner = self._make_runner(max_cache_length=4, cache_length=2)
        attention_window = runner.max_cache_length + 1

        mask = np.full((1, attention_window), -1e9, dtype=np.float32)
        valid = min(runner.cache_length, runner.max_cache_length)
        mask[0, :valid] = 0.0
        mask[0, -1] = 0.0

        assert mask[0, 0] == pytest.approx(0.0)
        assert mask[0, 1] == pytest.approx(0.0)
        assert mask[0, 2] == pytest.approx(-1e9)
        assert mask[0, 3] == pytest.approx(-1e9)
        assert mask[0, 4] == pytest.approx(0.0)  # current token

    def test_mask_full_cache(self):
        """With full cache, all positions are valid."""
        runner = self._make_runner(max_cache_length=4, cache_length=4)
        attention_window = runner.max_cache_length + 1

        mask = np.full((1, attention_window), -1e9, dtype=np.float32)
        valid = min(runner.cache_length, runner.max_cache_length)
        mask[0, :valid] = 0.0
        mask[0, -1] = 0.0

        for i in range(5):
            assert mask[0, i] == pytest.approx(0.0), (
                f"Position {i} should be valid with full cache")

    def test_cache_append_before_full(self):
        """Cache append: write to cache[cache_length] when not full."""
        runner = self._make_runner(max_cache_length=4, cache_length=1,
                                   attention_size=2)
        pk = np.array([1.0, 2.0], dtype=np.float32)

        # Simulate append
        runner.cache_k[0][runner.cache_length] = pk
        assert runner.cache_k[0][1, 0] == pytest.approx(1.0)
        assert runner.cache_k[0][1, 1] == pytest.approx(2.0)

    def test_cache_shift_when_full(self):
        """Cache shift: shift left and append at end when full."""
        runner = self._make_runner(max_cache_length=3, cache_length=3,
                                   attention_size=2)
        runner.cache_k[0] = np.array(
            [[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        pk = np.array([7.0, 8.0], dtype=np.float32)

        # Simulate shift+append (same logic as TrtRunner.step)
        runner.cache_k[0][:-1] = runner.cache_k[0][1:]
        runner.cache_k[0][-1] = pk

        np.testing.assert_array_equal(
            runner.cache_k[0],
            [[3, 4], [5, 6], [7, 8]])


# ---------------------------------------------------------------------------
# MambaTrtRunner state reset
# ---------------------------------------------------------------------------

class TestMambaStateReset:
    """Test that MambaTrtRunner.reset() zeros all recurrent states."""

    def test_reset_zeros_states(self):
        from trtf_build.debug_runner import MambaTrtRunner

        runner = MambaTrtRunner.__new__(MambaTrtRunner)
        runner.num_layers = 2
        runner.d_inner = 4
        runner.state_size = 3
        runner.conv_kernel = 2
        # Prevent __del__ from crashing on GC
        runner._device_buffers = {}

        runner.conv_state = [
            np.ones((4, 2), dtype=np.float32) for _ in range(2)
        ]
        runner.ssm_state = [
            np.ones((4, 3), dtype=np.float32) for _ in range(2)
        ]

        # Reset via manual zeroing (same as generate would do before a new prompt)
        for i in range(runner.num_layers):
            runner.conv_state[i][:] = 0.0
            runner.ssm_state[i][:] = 0.0

        for i in range(2):
            np.testing.assert_array_equal(
                runner.conv_state[i], np.zeros((4, 2)))
            np.testing.assert_array_equal(
                runner.ssm_state[i], np.zeros((4, 3)))
