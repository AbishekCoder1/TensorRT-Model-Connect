"""Tests for bundle_writer.py — .trtfb binary format round-trip.

Pure Python, no TRT needed.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

pytest.importorskip("trtf_build", reason="trtf_build requires tensorrt")
from trtf_build.bundle_writer import (
    BUNDLE_MAGIC,
    BundleInfo,
    BundleSection,
    write_bundle,
)


class TestBundleMagic:
    def test_magic_bytes(self):
        assert BUNDLE_MAGIC == b"TRTFB\x00\x01\x00"
        assert len(BUNDLE_MAGIC) == 8


class TestWriteBundle:
    def _read_bundle(self, path: str) -> tuple[dict, dict[str, bytes]]:
        """Read a .trtfb bundle and return (header_dict, sections_data)."""
        with open(path, "rb") as f:
            magic = f.read(8)
            assert magic == BUNDLE_MAGIC
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len).decode("utf-8"))

            sections_data = {}
            data_start = 16 + header_len
            for name, meta in header.get("sections", {}).items():
                f.seek(data_start + meta["offset"])
                sections_data[name] = f.read(meta["size"])

        return header, sections_data

    def test_single_section(self, tmp_path):
        info = BundleInfo(
            model_id="test-model",
            model_type="qwen3",
            family="qwen",
            vocab_size=32000,
            hidden_size=1024,
            num_layers=12,
            max_cache_length=256,
        )
        data = b"fake engine plan data here"
        sections = [BundleSection("engine_plan", data)]

        out_path = str(tmp_path / "test.trtfb")
        write_bundle(out_path, info, sections)

        header, sdata = self._read_bundle(out_path)
        assert header["model_id"] == "test-model"
        assert header["model_type"] == "qwen3"
        assert header["family"] == "qwen"
        assert header["vocab_size"] == 32000
        assert header["hidden_size"] == 1024
        assert header["num_layers"] == 12
        assert header["max_cache_length"] == 256
        assert sdata["engine_plan"] == data

    def test_multi_section(self, tmp_path):
        info = BundleInfo(model_id="multi")
        section1 = BundleSection("engine_plan", b"ENGINE" * 100)
        section2 = BundleSection("config.json", b'{"test": true}')
        section3 = BundleSection("tokenizer.json", b'{"vocab": []}')

        out_path = str(tmp_path / "multi.trtfb")
        write_bundle(out_path, info, [section1, section2, section3])

        header, sdata = self._read_bundle(out_path)
        assert len(header["sections"]) == 3
        assert sdata["engine_plan"] == b"ENGINE" * 100
        assert sdata["config.json"] == b'{"test": true}'
        assert sdata["tokenizer.json"] == b'{"vocab": []}'

    def test_section_offsets(self, tmp_path):
        info = BundleInfo(model_id="offsets")
        s1_data = b"A" * 100
        s2_data = b"B" * 200
        s3_data = b"C" * 50
        sections = [
            BundleSection("s1", s1_data),
            BundleSection("s2", s2_data),
            BundleSection("s3", s3_data),
        ]

        out_path = str(tmp_path / "offsets.trtfb")
        write_bundle(out_path, info, sections)

        header, _ = self._read_bundle(out_path)
        secs = header["sections"]
        assert secs["s1"]["offset"] == 0
        assert secs["s1"]["size"] == 100
        assert secs["s2"]["offset"] == 100
        assert secs["s2"]["size"] == 200
        assert secs["s3"]["offset"] == 300
        assert secs["s3"]["size"] == 50

    def test_empty_section(self, tmp_path):
        info = BundleInfo(model_id="empty")
        sections = [BundleSection("empty_sec", b"")]
        out_path = str(tmp_path / "empty.trtfb")
        write_bundle(out_path, info, sections)

        header, sdata = self._read_bundle(out_path)
        assert sdata["empty_sec"] == b""
        assert header["sections"]["empty_sec"]["size"] == 0

    def test_no_sections(self, tmp_path):
        info = BundleInfo(model_id="nosec")
        out_path = str(tmp_path / "nosec.trtfb")
        write_bundle(out_path, info, [])

        header, sdata = self._read_bundle(out_path)
        assert len(header["sections"]) == 0
        assert len(sdata) == 0

    def test_binary_data_integrity(self, tmp_path):
        """Verify binary data survives the round-trip exactly."""
        rng = np.random.RandomState(42)
        binary_data = rng.bytes(4096)

        info = BundleInfo(model_id="binary")
        sections = [BundleSection("engine_plan", binary_data)]
        out_path = str(tmp_path / "binary.trtfb")
        write_bundle(out_path, info, sections)

        _, sdata = self._read_bundle(out_path)
        assert sdata["engine_plan"] == binary_data

    def test_all_info_fields(self, tmp_path):
        info = BundleInfo(
            model_id="full-test",
            model_type="llama",
            family="llama",
            trt_version="10.1.0",
            gpu_name="NVIDIA RTX 4090",
            created_at="2026-02-16T12:00:00Z",
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            max_cache_length=512,
        )
        out_path = str(tmp_path / "full.trtfb")
        write_bundle(out_path, info, [BundleSection("engine_plan", b"x")])

        header, _ = self._read_bundle(out_path)
        assert header["trt_version"] == "10.1.0"
        assert header["gpu_name"] == "NVIDIA RTX 4090"
        assert header["created_at"] == "2026-02-16T12:00:00Z"
        assert header["num_attention_heads"] == 32
        assert header["num_key_value_heads"] == 8
