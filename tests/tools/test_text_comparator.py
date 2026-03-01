"""Unit tests for text_generation comparator normalization behavior."""

from __future__ import annotations

import numpy as np

from tests.e2e_harness.comparators.text import TextComparator
from tests.e2e_harness.contracts import StageOutput, StageSpec, StageStatus, ThresholdProfile


def _default_thresholds() -> ThresholdProfile:
    return ThresholdProfile(
        task_strategy="text_generation_causal",
        profile_name="test",
        metrics={
            "logit_cosine_p5": 0.99,
            "logit_rel_l2_p95": 0.05,
            "stable_top1_match_rate": 0.9,
            "unstable_topk_hit_rate": 0.8,
            "token_agreement_rate": 0.8,
            "normalized_text_edit_distance": 0.2,
        },
    )


def test_warning_preamble_before_prompt_does_not_fail_ned() -> None:
    prompt = "Once upon a time there was a little"
    continuation = (
        "girl named Lucy. She was three years old and she was very excited "
        "to go on an adventure."
    )
    warning = (
        "You are using the default legacy behaviour of the "
        "<class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>."
    )
    trt_text = f"{warning}\n{prompt} {continuation}"
    ref_text = continuation

    # Identical logits: all token/logit metrics should pass.
    logits = np.array(
        [
            [0.0, 1.0, -1.0],
            [0.0, 2.0, -2.0],
            [0.0, 3.0, -3.0],
        ],
        dtype=np.float32,
    )

    trt = StageOutput(
        stage_name="full_generation",
        data={"prompt": prompt},
        text=trt_text,
        logits=logits,
    )
    ref = StageOutput(
        stage_name="full_generation",
        data={},
        text=ref_text,
        logits=logits.copy(),
    )

    result = TextComparator().compare(
        trt=trt,
        ref=ref,
        threshold=_default_thresholds(),
        stage=StageSpec(name="full_generation"),
    )

    assert result.status == StageStatus.PASSED.value
    assert "normalized_text_edit_distance" in result.metrics
    assert result.metrics["normalized_text_edit_distance"].passed


def test_prefix_only_truncation_with_matching_tokens_does_not_hard_fail_ned() -> None:
    """If one side is an early-stopped prefix, NED should not hard-fail."""
    prompt = "The capital of France is"
    trt_text = "The capital of France is the capital of the country."
    ref_text = (
        "the capital of the country. "
        "The capital of the country is Paris. "
        "The capital of the country"
    )

    # Identical logits -> token-level agreement is perfect.
    logits = np.array(
        [
            [0.0, 1.0, -1.0],
            [0.0, 2.0, -2.0],
            [0.0, 3.0, -3.0],
            [0.0, 4.0, -4.0],
        ],
        dtype=np.float32,
    )

    trt = StageOutput(
        stage_name="full_generation",
        data={"prompt": prompt},
        text=trt_text,
        logits=logits,
    )
    ref = StageOutput(
        stage_name="full_generation",
        data={},
        text=ref_text,
        logits=logits.copy(),
    )

    result = TextComparator().compare(
        trt=trt,
        ref=ref,
        threshold=_default_thresholds(),
        stage=StageSpec(name="full_generation"),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["token_agreement_rate"].passed
    assert result.metrics["normalized_text_edit_distance"].passed
