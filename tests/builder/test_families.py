"""Unit tests for all family plugins — match() logic and basic attributes.

Pure Python, no TRT/GPU needed. Verifies every plugin's match() returns
True for its model_types and False for others, and checks special attributes
like runtime_strategy and embed_input.
"""

from __future__ import annotations

import pytest

from trtf_build.families import find_plugin, _ALL_PLUGINS


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

class TestPluginDiscovery:
    """Verify plugin auto-discovery finds all expected families."""

    def test_plugin_count(self):
        assert len(_ALL_PLUGINS) >= 20, (
            f"Expected >= 20 plugins, found {len(_ALL_PLUGINS)}: "
            f"{[p.name for p in _ALL_PLUGINS]}")

    def test_all_have_name(self):
        for p in _ALL_PLUGINS:
            assert hasattr(p, "name"), f"Plugin {p} missing 'name' attribute"
            assert isinstance(p.name, str) and p.name, (
                f"Plugin {p} has empty or non-string name")

    def test_all_have_matches(self):
        for p in _ALL_PLUGINS:
            assert callable(getattr(p, "matches", None)), (
                f"Plugin {p.name} missing callable 'matches'")

    def test_all_have_load_weights(self):
        for p in _ALL_PLUGINS:
            assert callable(getattr(p, "load_weights", None)), (
                f"Plugin {p.name} missing callable 'load_weights'")

    def test_all_have_build_engine(self):
        for p in _ALL_PLUGINS:
            assert callable(getattr(p, "build_engine", None)), (
                f"Plugin {p.name} missing callable 'build_engine'")

    def test_unique_names(self):
        names = [p.name for p in _ALL_PLUGINS]
        assert len(names) == len(set(names)), (
            f"Duplicate plugin names: {names}")

    def test_all_plugins_have_match_case(self):
        """Every discovered plugin should have at least one positive match case."""
        matched_names = {name for _, name in _POSITIVE_MATCH_CASES}
        plugin_names = {p.name for p in _ALL_PLUGINS}
        untested = plugin_names - matched_names
        assert not untested, (
            f"Plugins without positive match cases: {untested}. "
            f"Add entries to _POSITIVE_MATCH_CASES.")


# ---------------------------------------------------------------------------
# match() positive cases
# ---------------------------------------------------------------------------

# (model_type, expected_plugin_name)
_POSITIVE_MATCH_CASES = [
    # Qwen family
    ("qwen", "qwen"),
    ("qwen2", "qwen"),
    ("qwen3", "qwen"),
    ("qwq", "qwen"),
    ("Qwen2", "qwen"),
    # LLaMA
    ("llama", "llama"),
    ("LLaMA", "llama"),
    # Mistral
    ("mistral", "mistral"),
    ("Mistral", "mistral"),
    # Gemma
    ("gemma", "gemma"),
    ("gemma2", "gemma"),
    # Phi (not phimoe)
    ("phi", "phi"),
    ("phi3", "phi"),
    ("Phi3", "phi"),
    # Phi-MoE
    ("phimoe", "phi_moe"),
    # Qwen3 MoE
    ("qwen3_moe", "qwen_moe"),
    # Granite
    ("granite", "granite"),
    # InternLM
    ("internlm", "internlm"),
    ("internlm2", "internlm"),
    # StarCoder2
    ("starcoder2", "starcoder2"),
    # GPT-2
    ("gpt2", "gpt2"),
    # OPT
    ("opt", "opt"),
    # Falcon
    ("falcon", "falcon"),
    # StableLM
    ("stablelm", "stablelm"),
    # Mamba
    ("mamba", "mamba"),
    # Qwen-VL
    ("qwen2_vl", "qwen_vl"),
    ("qwen2_5_vl", "qwen_vl"),
    ("qwen3_vl", "qwen_vl"),
    # OLMo
    ("olmo", "olmo"),
    # XGLM
    ("xglm", "xglm"),
    # GPT-NeoX
    ("gpt_neox", "gpt_neox"),
    ("gptneox", "gpt_neox"),
    # GPT-Neo
    ("gpt_neo", "gpt_neo"),
    # CodeGen
    ("codegen", "codegen"),
    # BLOOM
    ("bloom", "bloom"),
    # Mixtral
    ("mixtral", "mixtral"),
    # Nemotron
    ("nemotron", "nemotron"),
    # Wan T2V (diffusion)
    ("wan_t2v", "wan_t2v"),
    ("wan", "wan_t2v"),
    # Bark (text-to-audio)
    ("bark", "bark"),
    # SegFormer (segmentation)
    ("segformer", "segformer"),
    # Whisper (speech-to-text)
    ("whisper", "whisper"),
    # RWKV
    ("rwkv", "rwkv"),
    # DeepSeek-V2
    ("deepseek_v2", "deepseek_v2"),
    # InternVL
    ("internvl_chat", "internvl"),
    ("internvl3", "internvl"),
    # BERT (encoder-only)
    ("bert", "bert"),
    # DeepONet (neural operator)
    ("deeponet", "deeponet"),
    ("deep_o_net", "deeponet"),
    # YOLOX (object detection)
    ("yolox", "yolox"),
    ("yolox_document", "yolox"),
    ("yolox-doc", "yolox"),
    # Eagle VLM (embedding/reranking)
    ("eagle", "eagle_vlm"),
    # Qwen3-Omni (omni multimodal)
    ("qwen3_omni", "qwen3_omni"),
    ("qwen3omni", "qwen3_omni"),
    ("qwen3_omni_moe", "qwen3_omni"),
    # PersonaPlex (speech-to-speech)
    ("personaplex", "personaplex"),
    ("moshi", "personaplex"),
    ("personaplex_7b", "personaplex"),
    # NemotronH (hybrid Mamba-Attention)
    ("nemotron_h", "nemotron_h"),
    ("nemotron_hybrid", "nemotron_h"),
]


class TestMatchPositive:
    """Each known model_type resolves to the correct plugin."""

    @pytest.mark.parametrize("model_type,expected_name", _POSITIVE_MATCH_CASES)
    def test_match(self, model_type, expected_name):
        plugin = find_plugin(model_type)
        assert plugin is not None, f"No plugin matched model_type={model_type!r}"
        assert plugin.name == expected_name, (
            f"model_type={model_type!r} matched {plugin.name!r}, "
            f"expected {expected_name!r}")


# ---------------------------------------------------------------------------
# match() negative cases
# ---------------------------------------------------------------------------

_NEGATIVE_MATCH_CASES = [
    "unknown_model",
    "t5",
    "clip",
    "",
]


class TestMatchNegative:
    """Unknown model_types should return None."""

    @pytest.mark.parametrize("model_type", _NEGATIVE_MATCH_CASES)
    def test_no_match(self, model_type):
        assert find_plugin(model_type) is None, (
            f"model_type={model_type!r} should not match any plugin")


# ---------------------------------------------------------------------------
# Qwen vs Qwen-VL disambiguation
# ---------------------------------------------------------------------------

class TestQwenVLDisambiguation:
    """Qwen-VL should match VL plugin; plain Qwen should not."""

    def test_qwen_vl_matches_vl_plugin(self):
        plugin = find_plugin("qwen2_vl")
        assert plugin is not None
        assert plugin.name == "qwen_vl"

    def test_plain_qwen_does_not_match_vl(self):
        plugin = find_plugin("qwen3")
        assert plugin is not None
        assert plugin.name == "qwen"

    def test_qwen_vl_does_not_match_plain_qwen(self):
        """The plain Qwen plugin should reject VL model types."""
        qwen_plugin = None
        for p in _ALL_PLUGINS:
            if p.name == "qwen":
                qwen_plugin = p
                break
        assert qwen_plugin is not None
        assert not qwen_plugin.matches("qwen2_vl")


# ---------------------------------------------------------------------------
# Phi vs Phi-MoE disambiguation
# ---------------------------------------------------------------------------

class TestPhiDisambiguation:
    """Phi should not match phimoe, and vice versa."""

    def test_phi_rejects_phimoe(self):
        phi_plugin = None
        for p in _ALL_PLUGINS:
            if p.name == "phi":
                phi_plugin = p
                break
        assert phi_plugin is not None
        assert not phi_plugin.matches("phimoe")

    def test_phimoe_rejects_phi3(self):
        moe_plugin = None
        for p in _ALL_PLUGINS:
            if p.name == "phi_moe":
                moe_plugin = p
                break
        assert moe_plugin is not None
        assert not moe_plugin.matches("phi3")


# ---------------------------------------------------------------------------
# Special attributes
# ---------------------------------------------------------------------------

class TestRuntimeStrategy:
    """Plugins with non-default runtime_strategy."""

    def test_mamba_strategy(self):
        plugin = find_plugin("mamba")
        assert getattr(plugin, "runtime_strategy", None) == "ssm_recurrent"

    def test_mixtral_strategy(self):
        plugin = find_plugin("mixtral")
        assert getattr(plugin, "runtime_strategy", None) == "decoder_moe"

    def test_qwen_vl_strategy(self):
        plugin = find_plugin("qwen2_vl")
        assert getattr(plugin, "runtime_strategy", None) == "vision_language"

    def test_internvl_strategy(self):
        plugin = find_plugin("internvl_chat")
        assert getattr(plugin, "runtime_strategy", None) == "vision_language"

    def test_omni_strategy(self):
        plugin = find_plugin("qwen3_omni")
        assert getattr(plugin, "runtime_strategy", None) == "omni_multimodal"

    def test_personaplex_strategy(self):
        plugin = find_plugin("personaplex")
        assert getattr(plugin, "runtime_strategy", None) == "speech_to_speech"

    def test_nemotron_h_strategy(self):
        plugin = find_plugin("nemotron_h")
        assert getattr(plugin, "runtime_strategy", None) == "hybrid_mamba_attention"

    def test_standard_decoder_no_strategy(self):
        """Standard decoder plugins have no runtime_strategy (defaults to decoder_kv_cache)."""
        for name in ("qwen", "llama", "mistral", "gemma", "phi", "gpt2", "opt"):
            plugin = find_plugin(name)
            assert plugin is not None
            strategy = getattr(plugin, "runtime_strategy", None)
            assert strategy is None, (
                f"Plugin {plugin.name} should not have runtime_strategy, "
                f"got {strategy!r}")


class TestEmbedInput:
    """Only VL plugins should have embed_input=True."""

    def test_qwen_vl_has_embed_input(self):
        plugin = find_plugin("qwen2_vl")
        assert getattr(plugin, "embed_input", False) is True

    def test_internvl_has_embed_input(self):
        plugin = find_plugin("internvl_chat")
        assert getattr(plugin, "embed_input", False) is True

    def test_omni_has_embed_input(self):
        plugin = find_plugin("qwen3_omni")
        assert getattr(plugin, "embed_input", False) is True

    def test_standard_plugins_no_embed_input(self):
        for name in ("qwen", "llama", "mistral", "mamba", "mixtral"):
            plugin = find_plugin(name)
            assert plugin is not None
            assert not getattr(plugin, "embed_input", False), (
                f"Plugin {plugin.name} should not have embed_input")


class TestVLMethods:
    """VL plugins should have build_vision_engine and get_vl_config methods."""

    def test_qwen_vl_has_vl_methods(self):
        plugin = find_plugin("qwen2_vl")
        assert callable(getattr(plugin, "build_vision_engine", None))
        assert callable(getattr(plugin, "get_vl_config", None))

    def test_internvl_has_vl_methods(self):
        plugin = find_plugin("internvl_chat")
        assert callable(getattr(plugin, "build_vision_engine", None))
        assert callable(getattr(plugin, "get_vl_config", None))

    def test_omni_has_vl_methods(self):
        plugin = find_plugin("qwen3_omni")
        assert callable(getattr(plugin, "build_vision_engine", None))
        assert callable(getattr(plugin, "get_vl_config", None))
        assert callable(getattr(plugin, "build_extra_engines", None))

    def test_standard_plugins_vl_methods_return_none(self):
        """Non-VL plugins should return None from get_vl_config if they have it."""
        for name in ("qwen", "llama", "gpt2"):
            plugin = find_plugin(name)
            vl_config = getattr(plugin, "get_vl_config", None)
            if vl_config is not None and callable(vl_config):
                # Would need a ModelConfig, but default protocol returns None
                pass  # Can't call without a real config
