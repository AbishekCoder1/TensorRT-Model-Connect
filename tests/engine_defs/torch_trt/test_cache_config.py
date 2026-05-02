"""Tests for ttrt_build.cache_config — raw TRT format cache tensors and export args.

Validates:
  - build_attention_mask(): correct mask shape, values, step progression
  - make_cache_tensors(): correct shapes/dtypes for compact GQA/MQA cache
  - make_export_args(): complete tuple matching StatelessCacheWrapper.forward()
"""

from __future__ import annotations

import pytest

try:
    from trtf_build.engine_defs.torch_trt.config import ModelConfig
    from trtf_build.engine_defs.torch_trt.cache_config import (
        build_attention_mask, make_cache_tensors, make_export_args,
    )
except ImportError:
    pytest.skip("ttrt_build not importable", allow_module_level=True)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not available")


@requires_torch
class TestBuildAttentionMask:
    """Tests for build_attention_mask() — 1D mask in raw TRT format."""

    def test_shape(self):
        mask = build_attention_mask(0, max_cache_length=32, device="cpu")
        assert mask.shape == (1, 33)  # [1, max_cache_length + 1]
        assert mask.dtype == torch.float32

    def test_step_zero(self):
        mask = build_attention_mask(0, max_cache_length=8, device="cpu")
        # Step 0: position 0 is valid (0.0), rest masked (-1e4)
        assert mask[0, 0].item() == 0.0
        assert mask[0, 1].item() == pytest.approx(-1e4)
        assert mask[0, 8].item() == pytest.approx(-1e4)

    def test_step_three(self):
        mask = build_attention_mask(3, max_cache_length=8, device="cpu")
        # Step 3: positions 0-3 valid, 4+ masked
        for i in range(4):
            assert mask[0, i].item() == 0.0, f"position {i} should be valid"
        for i in range(4, 9):
            assert mask[0, i].item() == pytest.approx(-1e4), f"position {i} should be masked"

    def test_full_cache(self):
        max_cache = 4
        mask = build_attention_mask(max_cache - 1, max_cache_length=max_cache, device="cpu")
        # All cache positions valid, +1 slot still masked
        for i in range(max_cache):
            assert mask[0, i].item() == 0.0
        assert mask[0, max_cache].item() == pytest.approx(-1e4)

    def test_different_cache_lengths(self):
        for cache_len in [16, 64, 256]:
            mask = build_attention_mask(0, max_cache_length=cache_len, device="cpu")
            assert mask.shape == (1, cache_len + 1)


@requires_torch
class TestMakeCacheTensors:
    """Tests for make_cache_tensors() — flat list of compact KV cache tensors."""

    def test_count_and_shape(self, sample_config):
        # sample_config: num_hidden_layers=2, num_key_value_heads=2, head_dim=16
        cache = make_cache_tensors(sample_config, max_cache_length=32, device="cpu")
        # 2 layers * 2 (k + v) = 4 tensors
        assert len(cache) == 4
        # Each: [max_cache_length, kv_dim] where kv_dim = num_kv_heads * head_dim
        kv_dim = 2 * 16
        for i, t in enumerate(cache):
            assert t.shape == (32, kv_dim), f"cache[{i}] shape mismatch"
            assert t.dtype == torch.float32, f"cache[{i}] dtype mismatch"

    def test_zeros(self, sample_config):
        cache = make_cache_tensors(sample_config, max_cache_length=16, device="cpu")
        for t in cache:
            assert (t == 0).all()

    def test_gqa_compact_cache(self):
        """Cache uses num_key_value_heads, not expanded query heads."""
        config = ModelConfig(
            model_type="qwen3",
            raw={
                "hidden_size": 1024,
                "num_hidden_layers": 1,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "head_dim": 64,
            },
        )
        cache = make_cache_tensors(config, max_cache_length=64, device="cpu")
        assert len(cache) == 2  # 1 layer * 2 (k + v)
        assert cache[0].shape == (64, 128)
        assert cache[0].shape != (64, 1024)

    def test_different_cache_lengths(self, sample_config):
        for cache_len in [8, 32, 256]:
            cache = make_cache_tensors(sample_config, max_cache_length=cache_len, device="cpu")
            assert cache[0].shape[0] == cache_len

    def test_mha_no_expansion(self):
        """MHA (kv_heads == heads): attention_size = hidden_size."""
        config = ModelConfig(
            model_type="gpt2",
            raw={
                "hidden_size": 768,
                "num_hidden_layers": 2,
                "num_attention_heads": 12,
                "num_key_value_heads": 12,
                "head_dim": 64,
            },
        )
        cache = make_cache_tensors(config, max_cache_length=32, device="cpu")
        assert cache[0].shape == (32, 768)


@requires_torch
class TestMakeExportArgs:
    """Tests for make_export_args() — full input tuple for torch.export."""

    def test_returns_tuple(self, sample_config):
        args = make_export_args(sample_config, max_cache_length=32, device="cpu")
        assert isinstance(args, tuple)

    def test_tuple_length(self, sample_config):
        # sample_config: 2 layers → 4 cache tensors
        # Total: token_id + position_id + attention_mask + 4 cache = 7
        args = make_export_args(sample_config, max_cache_length=32, device="cpu")
        assert len(args) == 7

    def test_token_id(self, sample_config):
        args = make_export_args(sample_config, max_cache_length=32, device="cpu")
        token_id = args[0]
        assert token_id.shape == (1,)
        assert token_id.dtype == torch.int32

    def test_position_id(self, sample_config):
        args = make_export_args(sample_config, max_cache_length=32, device="cpu")
        position_id = args[1]
        assert position_id.shape == (1,)
        assert position_id.dtype == torch.int32

    def test_attention_mask(self, sample_config):
        args = make_export_args(sample_config, max_cache_length=32, device="cpu")
        mask = args[2]
        assert mask.shape == (1, 33)  # [1, max_cache_length + 1]
        assert mask.dtype == torch.float32

    def test_cache_tensors(self, sample_config):
        args = make_export_args(sample_config, max_cache_length=32, device="cpu")
        # cache tensors start at index 3
        kv_dim = 2 * 16
        for i in range(3, len(args)):
            t = args[i]
            assert t.shape == (32, kv_dim), f"cache[{i-3}] shape"
            assert t.dtype == torch.float32, f"cache[{i-3}] dtype"

    def test_precision_ignored(self, sample_config):
        """precision parameter is ignored — always float32 for raw TRT."""
        args_fp16 = make_export_args(
            sample_config, max_cache_length=32, precision="fp16", device="cpu")
        args_fp32 = make_export_args(
            sample_config, max_cache_length=32, precision="fp32", device="cpu")
        # Both should produce float32 cache tensors
        assert args_fp16[3].dtype == torch.float32
        assert args_fp32[3].dtype == torch.float32

    def test_many_layers(self):
        """Verify tuple length scales with num_layers."""
        config = ModelConfig(
            model_type="llama",
            raw={
                "hidden_size": 2048,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "head_dim": 128,
            },
        )
        args = make_export_args(config, max_cache_length=64, device="cpu")
        # 3 core inputs + 28*2 cache tensors = 59
        assert len(args) == 59
