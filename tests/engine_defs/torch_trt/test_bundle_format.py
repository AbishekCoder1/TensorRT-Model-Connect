"""Tests for ttrt_build bundle writer + reader — round-trip, magic, sections.

After LibTorch removal, bundles use TRTFB magic (same as raw TRT pipeline).
The reader still accepts legacy TTRTB magic for backward compatibility.
"""

from __future__ import annotations

import json
import pytest

try:
    from trtf_build.engine_defs.torch_trt.bundle_writer import (
        BUNDLE_MAGIC, TtrtBundleInfo, BundleSection, write_bundle,
    )
    from trtf_build.engine_defs.torch_trt.bundle_reader import (
        has_ttrtb_magic, read_bundle_header, read_bundle_section,
    )
except ImportError:
    pytest.skip("ttrt_build not importable", allow_module_level=True)


class TestBundleMagic:
    def test_magic_is_trtfb(self):
        """After LibTorch removal, bundles use TRTFB magic (raw TRT format)."""
        assert BUNDLE_MAGIC == b"TRTFB\x00\x01\x00"
        assert len(BUNDLE_MAGIC) == 8

    def test_magic_not_ttrtb(self):
        """TTRTB magic is the legacy format — we no longer produce it."""
        assert BUNDLE_MAGIC != b"TTRTB\x00\x01\x00"


class TestWriteBundle:
    def test_round_trip_header(self, tmp_path):
        bundle_path = tmp_path / "test.trtfb"
        info = TtrtBundleInfo(
            model_id="test-model",
            model_type="qwen3",
            family="qwen",
            torch_version="2.6.0",
            torchtrt_version="2.6.0",
            vocab_size=32000,
            hidden_size=1024,
            num_layers=28,
            precision="fp16",
            runtime_strategy="torchtrt_decoder",
        )
        sections = [
            BundleSection("engine_plan", b"\x00" * 100),
            BundleSection("config.json", b'{"model_type":"qwen3"}'),
        ]
        write_bundle(str(bundle_path), info, sections)

        assert has_ttrtb_magic(str(bundle_path))

        header = read_bundle_header(str(bundle_path))
        assert header["model_id"] == "test-model"
        assert header["model_type"] == "qwen3"
        assert header["family"] == "qwen"
        assert header["vocab_size"] == 32000
        assert header["precision"] == "fp16"
        assert header["runtime_strategy"] == "torchtrt_decoder"

    def test_round_trip_sections(self, tmp_path):
        bundle_path = tmp_path / "test.trtfb"
        engine_data = b"fake-trt-engine-plan-bytes-" * 10
        config_data = b'{"model_type":"qwen3","hidden_size":1024}'

        info = TtrtBundleInfo(model_type="qwen3")
        sections = [
            BundleSection("engine_plan", engine_data),
            BundleSection("config.json", config_data),
        ]
        write_bundle(str(bundle_path), info, sections)

        recovered_engine = read_bundle_section(str(bundle_path), "engine_plan")
        assert recovered_engine == engine_data

        recovered_config = read_bundle_section(str(bundle_path), "config.json")
        assert recovered_config == config_data

    def test_section_not_found(self, tmp_path):
        bundle_path = tmp_path / "test.trtfb"
        info = TtrtBundleInfo()
        sections = [BundleSection("engine_plan", b"\x00")]
        write_bundle(str(bundle_path), info, sections)

        with pytest.raises(KeyError, match="nonexistent"):
            read_bundle_section(str(bundle_path), "nonexistent")

    def test_empty_sections(self, tmp_path):
        bundle_path = tmp_path / "test.trtfb"
        info = TtrtBundleInfo(model_type="test")
        write_bundle(str(bundle_path), info, [])

        header = read_bundle_header(str(bundle_path))
        assert header["sections"] == {}

    def test_tokenizer_sections(self, tmp_path):
        """Verify tokenizer files survive round-trip (needed for C++ runtime)."""
        bundle_path = tmp_path / "test.trtfb"
        tok_data = b'{"tokenizer_class":"QWenTokenizer"}'
        vocab_data = b'{"hello":1,"world":2}'

        info = TtrtBundleInfo(model_type="qwen3")
        sections = [
            BundleSection("engine_plan", b"\x00" * 10),
            BundleSection("tokenizer.json", tok_data),
            BundleSection("tokenizer_config.json", tok_data),
            BundleSection("vocab.json", vocab_data),
        ]
        write_bundle(str(bundle_path), info, sections)

        assert read_bundle_section(str(bundle_path), "tokenizer.json") == tok_data
        assert read_bundle_section(str(bundle_path), "vocab.json") == vocab_data

    def test_runtime_strategy_in_header(self, tmp_path):
        """Verify runtime_strategy is persisted in the JSON header."""
        bundle_path = tmp_path / "test.trtfb"
        info = TtrtBundleInfo(
            model_type="qwen3",
            runtime_strategy="torchtrt_decoder",
        )
        write_bundle(str(bundle_path), info, [])

        header = read_bundle_header(str(bundle_path))
        assert header["runtime_strategy"] == "torchtrt_decoder"

    def test_runtime_strategy_omitted_when_empty(self, tmp_path):
        """If runtime_strategy is empty, it should not appear in the header."""
        bundle_path = tmp_path / "test.trtfb"
        info = TtrtBundleInfo(model_type="test", runtime_strategy="")
        write_bundle(str(bundle_path), info, [])

        header = read_bundle_header(str(bundle_path))
        assert "runtime_strategy" not in header

    def test_tokenizer_add_special_tokens(self, tmp_path):
        """Verify tokenizer_add_special_tokens is stored as int in header."""
        bundle_path = tmp_path / "test.trtfb"
        info = TtrtBundleInfo(
            model_type="qwen3",
            tokenizer_add_special_tokens=True,
        )
        write_bundle(str(bundle_path), info, [])

        header = read_bundle_header(str(bundle_path))
        assert header["tokenizer_add_special_tokens"] == 1

    def test_all_model_fields(self, tmp_path):
        """Verify all architecture fields survive round-trip."""
        bundle_path = tmp_path / "test.trtfb"
        info = TtrtBundleInfo(
            model_id="Qwen/Qwen3-0.6B",
            model_type="qwen3",
            family="qwen",
            torch_version="2.6.0",
            torchtrt_version="2.6.0",
            gpu_name="NVIDIA GH200",
            created_at="2026-03-09T12:00:00Z",
            vocab_size=151936,
            hidden_size=1024,
            num_layers=28,
            num_attention_heads=16,
            num_key_value_heads=2,
            max_cache_length=256,
            precision="fp16",
            runtime_strategy="torchtrt_decoder",
            tokenizer_add_special_tokens=False,
        )
        write_bundle(str(bundle_path), info, [])

        header = read_bundle_header(str(bundle_path))
        assert header["model_id"] == "Qwen/Qwen3-0.6B"
        assert header["num_attention_heads"] == 16
        assert header["num_key_value_heads"] == 2
        assert header["max_cache_length"] == 256
        assert header["gpu_name"] == "NVIDIA GH200"


class TestHasMagic:
    def test_valid_bundle(self, tmp_path):
        bundle_path = tmp_path / "test.trtfb"
        info = TtrtBundleInfo()
        write_bundle(str(bundle_path), info, [])
        assert has_ttrtb_magic(str(bundle_path))

    def test_invalid_file(self, tmp_path):
        bad_path = tmp_path / "not_a_bundle.txt"
        bad_path.write_bytes(b"hello world")
        assert not has_ttrtb_magic(str(bad_path))

    def test_legacy_ttrtb_accepted(self, tmp_path):
        """Reader should accept legacy TTRTB magic for backward compatibility."""
        legacy_path = tmp_path / "legacy.ttrtb"
        # Write raw TTRTB magic + minimal valid header
        header = json.dumps({"model_type": "test", "sections": {}}).encode()
        import struct
        with open(legacy_path, "wb") as f:
            f.write(b"TTRTB\x00\x01\x00")
            f.write(struct.pack("<Q", len(header)))
            f.write(header)
        assert has_ttrtb_magic(str(legacy_path))

    def test_nonexistent_file(self, tmp_path):
        assert not has_ttrtb_magic(str(tmp_path / "nonexistent"))

    def test_too_short_file(self, tmp_path):
        short = tmp_path / "short"
        short.write_bytes(b"TRTF")
        assert not has_ttrtb_magic(str(short))
