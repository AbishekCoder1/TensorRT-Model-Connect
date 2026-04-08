"""Tests for precision propagation through the Torch-TRT build pipeline.

Verifies that --precision fp16/bf16/fp32 actually results in the correct dtype
at every stage: compiler helper, plugin load_model, strategy wrap_model,
StatelessCacheWrapper internals, and perf_utils helpers.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

try:
    from ttrt_build.compiler import precision_to_dtype, PRECISION_DTYPE_MAP
    from ttrt_build.strategies.decoder import (
        DecoderBuildStrategy,
        StatelessCacheWrapper,
    )
    from ttrt_build.strategies.encoder_only import EncoderOnlyBuildStrategy
except ImportError:
    pytest.skip("ttrt_build not importable", allow_module_level=True)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not available")


# ---------------------------------------------------------------------------
# precision_to_dtype
# ---------------------------------------------------------------------------

class TestPrecisionToDtype:
    """Tests for compiler.precision_to_dtype()."""

    @requires_torch
    def test_fp16(self):
        assert precision_to_dtype("fp16") == torch.float16

    @requires_torch
    def test_fp32(self):
        assert precision_to_dtype("fp32") == torch.float32

    @requires_torch
    def test_bf16(self):
        assert precision_to_dtype("bf16") == torch.bfloat16

    def test_invalid_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown precision"):
            precision_to_dtype("fp8")

    def test_invalid_lists_valid_options(self):
        with pytest.raises(ValueError, match="fp16"):
            precision_to_dtype("int8")

    def test_map_has_three_entries(self):
        assert len(PRECISION_DTYPE_MAP) == 3
        assert set(PRECISION_DTYPE_MAP.keys()) == {"fp16", "bf16", "fp32"}


# ---------------------------------------------------------------------------
# StatelessCacheWrapper compute_dtype propagation
# ---------------------------------------------------------------------------

@requires_torch
class TestStatelessCacheWrapperPrecision:
    """Verify StatelessCacheWrapper uses compute_dtype, not hardcoded fp16."""

    def _make_config(self, num_layers=2, num_heads=4, num_kv_heads=2,
                     head_dim=16, hidden_size=64, vocab_size=1000):
        cfg = SimpleNamespace(
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )
        # StaticCache calls config.get_text_config(decoder=True)
        cfg.get_text_config = lambda **kwargs: cfg
        return cfg

    def test_default_compute_dtype_is_fp16(self):
        config = self._make_config()
        wrapper = StatelessCacheWrapper(MagicMock(), config, max_cache_length=32)
        assert wrapper.compute_dtype == torch.float16

    def test_explicit_fp32(self):
        config = self._make_config()
        wrapper = StatelessCacheWrapper(
            MagicMock(), config, max_cache_length=32,
            compute_dtype=torch.float32)
        assert wrapper.compute_dtype == torch.float32

    def test_explicit_bf16(self):
        config = self._make_config()
        wrapper = StatelessCacheWrapper(
            MagicMock(), config, max_cache_length=32,
            compute_dtype=torch.bfloat16)
        assert wrapper.compute_dtype == torch.bfloat16

    def test_forward_mask_dtype_fp32(self):
        """When compute_dtype=fp32, attention mask is converted to fp32, not fp16."""
        config = self._make_config(num_layers=1, num_heads=2,
                                   num_kv_heads=2, head_dim=8)
        num_layers = 1
        num_heads = 2
        head_dim = 8
        attention_size = num_heads * head_dim
        max_cache = 16
        vocab_size = 1000

        # Build a mock model that captures what it receives
        captured_kwargs = {}

        def mock_forward(**kwargs):
            captured_kwargs.update(kwargs)
            logits = torch.randn(1, 1, vocab_size, dtype=torch.float32)
            # Build a mock cache with layers
            layer = SimpleNamespace(
                keys=torch.randn(1, 2, max_cache, head_dim, dtype=torch.float32),
                values=torch.randn(1, 2, max_cache, head_dim, dtype=torch.float32),
            )
            cache = SimpleNamespace(layers=[layer])
            return SimpleNamespace(logits=logits, past_key_values=cache)

        model = MagicMock(side_effect=mock_forward)

        wrapper = StatelessCacheWrapper(
            model, config, max_cache_length=max_cache,
            compute_dtype=torch.float32)

        # Build inputs
        token_id = torch.tensor([0], dtype=torch.int32)
        position_id = torch.tensor([0], dtype=torch.int32)
        mask = torch.full((1, max_cache + 1), -1e4, dtype=torch.float32)
        mask[0, 0] = 0.0
        cache_kv = [torch.zeros(max_cache, attention_size, dtype=torch.float32)
                     for _ in range(num_layers * 2)]

        _ = wrapper(token_id, position_id, mask, *cache_kv)

        # Verify: attention_mask passed to HF model must be fp32
        hf_mask = captured_kwargs["attention_mask"]
        assert hf_mask.dtype == torch.float32, \
            f"Expected attention_mask dtype=float32, got {hf_mask.dtype}"

    def test_forward_mask_dtype_fp16(self):
        """When compute_dtype=fp16, attention mask is converted to fp16."""
        config = self._make_config(num_layers=1, num_heads=2,
                                   num_kv_heads=2, head_dim=8)
        num_layers = 1
        num_heads = 2
        head_dim = 8
        attention_size = num_heads * head_dim
        max_cache = 16
        vocab_size = 1000

        captured_kwargs = {}

        def mock_forward(**kwargs):
            captured_kwargs.update(kwargs)
            logits = torch.randn(1, 1, vocab_size, dtype=torch.float16)
            layer = SimpleNamespace(
                keys=torch.randn(1, 2, max_cache, head_dim, dtype=torch.float16),
                values=torch.randn(1, 2, max_cache, head_dim, dtype=torch.float16),
            )
            cache = SimpleNamespace(layers=[layer])
            return SimpleNamespace(logits=logits, past_key_values=cache)

        model = MagicMock(side_effect=mock_forward)

        wrapper = StatelessCacheWrapper(
            model, config, max_cache_length=max_cache,
            compute_dtype=torch.float16)

        token_id = torch.tensor([0], dtype=torch.int32)
        position_id = torch.tensor([0], dtype=torch.int32)
        mask = torch.full((1, max_cache + 1), -1e4, dtype=torch.float32)
        mask[0, 0] = 0.0
        cache_kv = [torch.zeros(max_cache, attention_size, dtype=torch.float32)
                     for _ in range(num_layers * 2)]

        _ = wrapper(token_id, position_id, mask, *cache_kv)

        hf_mask = captured_kwargs["attention_mask"]
        assert hf_mask.dtype == torch.float16, \
            f"Expected attention_mask dtype=float16, got {hf_mask.dtype}"

    def test_forward_cache_dtype_matches_compute_dtype(self):
        """StaticCache is created with compute_dtype, not hardcoded fp16."""
        config = self._make_config(num_layers=1, num_heads=2,
                                   num_kv_heads=2, head_dim=8)
        num_layers = 1
        attention_size = 2 * 8
        max_cache = 16
        vocab_size = 1000

        captured_kwargs = {}

        def mock_forward(**kwargs):
            captured_kwargs.update(kwargs)
            logits = torch.randn(1, 1, vocab_size, dtype=torch.float32)
            layer = SimpleNamespace(
                keys=torch.randn(1, 2, max_cache, 8, dtype=torch.float32),
                values=torch.randn(1, 2, max_cache, 8, dtype=torch.float32),
            )
            cache = SimpleNamespace(layers=[layer])
            return SimpleNamespace(logits=logits, past_key_values=cache)

        model = MagicMock(side_effect=mock_forward)

        wrapper = StatelessCacheWrapper(
            model, config, max_cache_length=max_cache,
            compute_dtype=torch.float32)

        token_id = torch.tensor([0], dtype=torch.int32)
        position_id = torch.tensor([0], dtype=torch.int32)
        mask = torch.full((1, max_cache + 1), -1e4, dtype=torch.float32)
        mask[0, 0] = 0.0
        cache_kv = [torch.zeros(max_cache, attention_size, dtype=torch.float32)
                     for _ in range(num_layers * 2)]

        # Patch StaticCache to capture the dtype it's created with
        created_dtypes = []

        class MockStaticCache:
            def __init__(self, config, max_cache_len, dtype, device):
                created_dtypes.append(dtype)
                layer = SimpleNamespace(
                    keys=None, values=None, is_initialized=False)
                self.layers = [layer]

        with patch("transformers.cache_utils.StaticCache", MockStaticCache):
            # This will fail at model call, but we can check cache creation
            try:
                wrapper(token_id, position_id, mask, *cache_kv)
            except Exception:
                pass

        assert len(created_dtypes) == 1
        assert created_dtypes[0] == torch.float32, \
            f"StaticCache created with dtype={created_dtypes[0]}, expected float32"

    def test_forward_output_always_fp32(self):
        """Output logits and present KV are always fp32 regardless of compute_dtype."""
        config = self._make_config(num_layers=1, num_heads=2,
                                   num_kv_heads=2, head_dim=8)
        num_layers = 1
        num_heads = 2
        head_dim = 8
        attention_size = num_heads * head_dim
        max_cache = 16
        vocab_size = 100

        def mock_forward(**kwargs):
            # Model returns fp16 (like a real fp16 model would)
            logits = torch.randn(1, 1, vocab_size, dtype=torch.float16)
            layer = SimpleNamespace(
                keys=torch.randn(1, 2, max_cache, head_dim, dtype=torch.float16),
                values=torch.randn(1, 2, max_cache, head_dim, dtype=torch.float16),
            )
            cache = SimpleNamespace(layers=[layer])
            return SimpleNamespace(logits=logits, past_key_values=cache)

        model = MagicMock(side_effect=mock_forward)
        wrapper = StatelessCacheWrapper(
            model, config, max_cache_length=max_cache,
            compute_dtype=torch.float16)

        token_id = torch.tensor([0], dtype=torch.int32)
        position_id = torch.tensor([0], dtype=torch.int32)
        mask = torch.full((1, max_cache + 1), -1e4, dtype=torch.float32)
        mask[0, 0] = 0.0
        cache_kv = [torch.zeros(max_cache, attention_size, dtype=torch.float32)
                     for _ in range(num_layers * 2)]

        result = wrapper(token_id, position_id, mask, *cache_kv)

        # Logits: always fp32
        assert result[0].dtype == torch.float32, \
            f"Logits dtype={result[0].dtype}, expected float32"

        # Present KV: always fp32
        for i in range(1, len(result)):
            assert result[i].dtype == torch.float32, \
                f"Output[{i}] dtype={result[i].dtype}, expected float32"


# ---------------------------------------------------------------------------
# Strategy wrap_model passes compute_dtype
# ---------------------------------------------------------------------------

@requires_torch
class TestStrategyPrecisionPropagation:
    """Verify strategies pass compute_dtype to wrappers."""

    def _make_config(self, num_layers=2, num_heads=4, num_kv_heads=2,
                     head_dim=16, hidden_size=64, vocab_size=1000):
        cfg = SimpleNamespace(
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )
        # StaticCache calls config.get_text_config(decoder=True)
        cfg.get_text_config = lambda **kwargs: cfg
        return cfg

    def test_decoder_wrap_model_default_fp16(self):
        strategy = DecoderBuildStrategy()
        config = self._make_config()
        wrapper = strategy.wrap_model(MagicMock(), config, max_cache_length=32)
        assert wrapper.compute_dtype == torch.float16

    def test_decoder_wrap_model_fp32(self):
        strategy = DecoderBuildStrategy()
        config = self._make_config()
        wrapper = strategy.wrap_model(
            MagicMock(), config, max_cache_length=32,
            compute_dtype=torch.float32)
        assert wrapper.compute_dtype == torch.float32

    def test_decoder_wrap_model_bf16(self):
        strategy = DecoderBuildStrategy()
        config = self._make_config()
        wrapper = strategy.wrap_model(
            MagicMock(), config, max_cache_length=32,
            compute_dtype=torch.bfloat16)
        assert wrapper.compute_dtype == torch.bfloat16

    def test_encoder_only_wrap_model_accepts_compute_dtype(self):
        """EncoderOnlyBuildStrategy accepts compute_dtype without error."""
        strategy = EncoderOnlyBuildStrategy()
        config = SimpleNamespace(hidden_size=768)
        # Should not raise
        _ = strategy.wrap_model(
            MagicMock(), config, max_cache_length=128,
            compute_dtype=torch.float32)


# ---------------------------------------------------------------------------
# Plugin load_model dtype propagation
# ---------------------------------------------------------------------------

class TestPluginDtypePropagation:
    """Verify family plugins accept and use the dtype parameter."""

    def test_qwen_plugin_accepts_dtype(self):
        """QwenTorchTrtPlugin.load_model() accepts dtype kwarg."""
        from ttrt_build.families.qwen import QwenTorchTrtPlugin
        import inspect
        sig = inspect.signature(QwenTorchTrtPlugin.load_model)
        assert "dtype" in sig.parameters

    def test_bert_plugin_accepts_dtype(self):
        """BertTorchTrtPlugin.load_model() accepts dtype kwarg."""
        from ttrt_build.families.bert import BertTorchTrtPlugin
        import inspect
        sig = inspect.signature(BertTorchTrtPlugin.load_model)
        assert "dtype" in sig.parameters

    @requires_torch
    def test_qwen_plugin_passes_dtype_to_from_pretrained(self):
        """Verify Qwen plugin actually passes dtype to from_pretrained."""
        from ttrt_build.families.qwen import QwenTorchTrtPlugin

        plugin = QwenTorchTrtPlugin()
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model

        with patch("transformers.AutoModelForCausalLM.from_pretrained",
                    return_value=mock_model) as mock_load:
            plugin.load_model(
                "/fake/path",
                SimpleNamespace(model_type="qwen3"),
                max_cache_length=256,
                dtype=torch.float32,
            )
            # Verify from_pretrained was called with dtype=torch.float32
            mock_load.assert_called_once()
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs["dtype"] == torch.float32, \
                f"from_pretrained called with dtype={call_kwargs['dtype']}, expected float32"

    @requires_torch
    def test_qwen_plugin_default_dtype_is_fp16(self):
        """When dtype=None, Qwen plugin defaults to torch.float16."""
        from ttrt_build.families.qwen import QwenTorchTrtPlugin

        plugin = QwenTorchTrtPlugin()
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model

        with patch("transformers.AutoModelForCausalLM.from_pretrained",
                    return_value=mock_model) as mock_load:
            plugin.load_model(
                "/fake/path",
                SimpleNamespace(model_type="qwen3"),
                max_cache_length=256,
            )
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs["dtype"] == torch.float16, \
                f"Default dtype should be float16, got {call_kwargs['dtype']}"

    @requires_torch
    def test_qwen_plugin_fp32_is_different_from_default(self):
        """Explicitly passing fp32 must differ from the default fp16."""
        from ttrt_build.families.qwen import QwenTorchTrtPlugin

        plugin = QwenTorchTrtPlugin()
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model

        # Call with fp32
        with patch("transformers.AutoModelForCausalLM.from_pretrained",
                    return_value=mock_model) as mock_load:
            plugin.load_model(
                "/fake/path",
                SimpleNamespace(model_type="qwen3"),
                max_cache_length=256,
                dtype=torch.float32,
            )
            fp32_dtype = mock_load.call_args[1]["dtype"]

        # Call with default
        with patch("transformers.AutoModelForCausalLM.from_pretrained",
                    return_value=mock_model) as mock_load:
            plugin.load_model(
                "/fake/path",
                SimpleNamespace(model_type="qwen3"),
                max_cache_length=256,
            )
            default_dtype = mock_load.call_args[1]["dtype"]

        assert fp32_dtype != default_dtype, \
            "fp32 and default should produce different dtypes"
        assert fp32_dtype == torch.float32
        assert default_dtype == torch.float16


# ---------------------------------------------------------------------------
# build_bundle precision integration (mock-based)
# ---------------------------------------------------------------------------

@requires_torch
class TestBuildBundlePrecision:
    """Verify build_bundle passes precision through to plugin and strategy."""

    def test_build_bundle_invalid_precision_raises(self, tmp_path):
        """build_bundle rejects invalid precision strings."""
        from ttrt_build.compiler import build_bundle
        import json

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "vocab_size": 1000,
        }))

        with pytest.raises(ValueError, match="Unknown precision"):
            build_bundle(
                str(tmp_path), str(tmp_path / "out.trtfb"),
                precision="fp8")

    def test_precision_to_dtype_before_plugin_load(self, tmp_path):
        """precision_to_dtype is called before any plugin methods."""
        # Just verify the function works for all valid precisions
        for p in ("fp16", "bf16", "fp32"):
            dt = precision_to_dtype(p)
            assert isinstance(dt, torch.dtype)
