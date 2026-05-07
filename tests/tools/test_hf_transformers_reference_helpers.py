"""Tests for Hugging Face reference helper logic."""

from __future__ import annotations

from tests.e2e_harness.references.hf_transformers import _vl_fallback_prompt


def test_qwen_vl_fallback_prompt_includes_image_pad() -> None:
    assert _vl_fallback_prompt("Qwen/Qwen3-VL-2B-Instruct", "Describe it") == (
        "<|vision_start|><|image_pad|><|vision_end|>Describe it"
    )


def test_internvl_fallback_prompt_includes_image_placeholder() -> None:
    assert _vl_fallback_prompt("OpenGVLab/InternVL3-8B-hf", "Describe it") == (
        "<image>\nDescribe it"
    )


def test_non_vl_fallback_prompt_is_unchanged() -> None:
    assert _vl_fallback_prompt("Qwen/Qwen3-0.6B", "Hello") == "Hello"
