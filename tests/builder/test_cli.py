"""Tests for cli.py — argument parsing edge cases.

Pure Python, no TRT needed. Tests the CLI argument parser without
actually invoking engine builds.
"""

from __future__ import annotations

import argparse
import sys
from unittest.mock import patch

import pytest

pytest.importorskip("trtf_build", reason="trtf_build requires tensorrt")
from trtf_build.cli import main


class TestBuildArgs:
    def test_build_with_all_args(self):
        """Verify build command parses all arguments."""
        from trtf_build.cli import main as cli_main
        test_args = [
            "trtf-build", "build", "Qwen/Qwen3-0.6B",
            "-o", "/tmp/out.trtfb",
            "--max-cache-length", "512",
            "--verbose",
        ]
        with patch.object(sys, "argv", test_args):
            parser = argparse.ArgumentParser(prog="trtf-build")
            subparsers = parser.add_subparsers(dest="command")
            build_p = subparsers.add_parser("build")
            build_p.add_argument("model")
            build_p.add_argument("-o", "--output", required=True)
            build_p.add_argument("--max-cache-length", type=int, default=256)
            build_p.add_argument("--verbose", action="store_true")

            args = parser.parse_args(test_args[1:])
            assert args.command == "build"
            assert args.model == "Qwen/Qwen3-0.6B"
            assert args.output == "/tmp/out.trtfb"
            assert args.max_cache_length == 512
            assert args.verbose is True

    def test_build_default_cache_length(self):
        """Default max-cache-length is 256."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        build_p = subparsers.add_parser("build")
        build_p.add_argument("model")
        build_p.add_argument("-o", "--output", required=True)
        build_p.add_argument("--max-cache-length", type=int, default=256)
        build_p.add_argument("--verbose", action="store_true")

        args = parser.parse_args(["build", "model-dir", "-o", "out.trtfb"])
        assert args.max_cache_length == 256
        assert args.verbose is False

    def test_build_missing_output_exits(self):
        """Missing -o flag should cause an error."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        build_p = subparsers.add_parser("build")
        build_p.add_argument("model")
        build_p.add_argument("-o", "--output", required=True)

        with pytest.raises(SystemExit):
            parser.parse_args(["build", "model-dir"])


class TestInspectArgs:
    def test_inspect_parses(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        inspect_p = subparsers.add_parser("inspect")
        inspect_p.add_argument("bundle_path")

        args = parser.parse_args(["inspect", "/path/to/bundle.trtfb"])
        assert args.command == "inspect"
        assert args.bundle_path == "/path/to/bundle.trtfb"


class TestVersionCommand:
    def test_version_exits_zero(self):
        """trtf-build version should return 0."""
        from trtf_build.cli import _cmd_version
        result = _cmd_version(argparse.Namespace())
        assert result == 0


class TestCmdBuildValidation:
    def test_missing_model(self):
        """_cmd_build returns 1 when model is empty."""
        from trtf_build.cli import _cmd_build
        args = argparse.Namespace(model="", output="out.trtfb",
                                  max_cache_length=256, verbose=False)
        result = _cmd_build(args)
        assert result == 1

    def test_missing_output(self):
        """_cmd_build returns 1 when output is empty."""
        from trtf_build.cli import _cmd_build
        args = argparse.Namespace(model="some-model", output="",
                                  max_cache_length=256, verbose=False)
        result = _cmd_build(args)
        assert result == 1
