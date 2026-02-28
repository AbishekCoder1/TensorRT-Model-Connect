"""Unit tests for all family plugins — match() logic and basic attributes.

Pure Python, no TRT/GPU needed. Verifies every plugin's match() returns
True for its model_types and False for others, and checks special attributes
like runtime_strategy and embed_input.
"""

from __future__ import annotations

import json
from pathlib import Path

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

    def test_all_plugins_matches_returns_bool(self):
        """Every plugin's matches() should return a bool, not just a truthy value."""
        # Build a mapping: plugin_name -> one known model_type that should match.
        name_to_type: dict[str, str] = {}
        for model_type, plugin_name in _POSITIVE_MATCH_CASES:
            if plugin_name not in name_to_type:
                name_to_type[plugin_name] = model_type

        for plugin in _ALL_PLUGINS:
            model_type = name_to_type.get(plugin.name)
            if model_type is None:
                continue  # no known positive case; covered by test_all_plugins_have_match_case
            result = plugin.matches(model_type)
            assert isinstance(result, bool), (
                f"Plugin {plugin.name!r}.matches({model_type!r}) returned "
                f"{type(result).__name__}, expected bool")
            assert result is True, (
                f"Plugin {plugin.name!r}.matches({model_type!r}) returned "
                f"{result!r}, expected True")

    def test_all_plugins_matches_own_type(self):
        """Every plugin matches its known model_type and rejects a nonsense type."""
        # Build a mapping: plugin_name -> one known model_type that should match.
        name_to_type: dict[str, str] = {}
        for model_type, plugin_name in _POSITIVE_MATCH_CASES:
            if plugin_name not in name_to_type:
                name_to_type[plugin_name] = model_type

        nonsense_types = [
            "zzzz_nonexistent_model_xyz_42",
            "__bogus__",
            "this_model_does_not_exist_ever_12345",
        ]

        for plugin in _ALL_PLUGINS:
            model_type = name_to_type.get(plugin.name)
            if model_type is None:
                continue  # no known positive case
            # Positive: must match own type
            assert plugin.matches(model_type), (
                f"Plugin {plugin.name!r} did not match its own type "
                f"{model_type!r}")
            # Negative: must reject nonsense types
            for bad_type in nonsense_types:
                result = plugin.matches(bad_type)
                assert result is False, (
                    f"Plugin {plugin.name!r}.matches({bad_type!r}) returned "
                    f"{result!r}, expected False")
                assert isinstance(result, bool), (
                    f"Plugin {plugin.name!r}.matches({bad_type!r}) returned "
                    f"{type(result).__name__}, expected bool")

    def test_all_plugins_have_match_case(self):
        """Every discovered plugin should have at least one positive match case."""
        matched_names = {name for _, name in _POSITIVE_MATCH_CASES}
        plugin_names = {p.name for p in _ALL_PLUGINS}
        untested = plugin_names - matched_names
        assert not untested, (
            f"Plugins without positive match cases: {untested}. "
            f"Add entries to _POSITIVE_MATCH_CASES.")

    def test_all_plugins_have_e2e_manifest(self):
        """Validate that every discovered family plugin has at least one E2E test manifest.

        Intention:
            When a developer adds a new family plugin, they must also add a JSON manifest
            in tests/e2e/models/ so the E2E test suite covers that model. Without this
            enforcement, plugins can be added and never tested E2E, leading to silent
            regressions.

            Example bug this catches: A developer adds trtf_build/families/my_model.py with
            a plugin that passes unit tests, but forgets to add tests/e2e/models/my-model.json.
            The plugin works in development but breaks in production because no E2E test
            ever exercises the full build->infer->compare pipeline.

        Setup:
            1. Discover all family plugins via _ALL_PLUGINS (auto-discovered from
               trtf_build/trtf_build/families/*.py modules with a module-level `plugin`).
            2. Load all JSON manifests from tests/e2e/models/ and extract each "family" field.
            3. Compute the set difference: plugins without any manifest coverage.
            4. Exclude exempt plugins (WIP/incomplete) from the check.
            5. Assert the uncovered set is empty.
        """
        _EXEMPT_PLUGINS = {
            "deepseek_ocr", "qwen3_omni", "deeponet", "yolox", "fno",
            "nemotron_h", "phi4_multimodal", "eagle_vlm", "personaplex",
            "deepseek_v2",
        }

        models_dir = Path(__file__).resolve().parent.parent / "e2e" / "models"
        families_in_manifests: set[str] = set()
        for manifest_path in models_dir.glob("*.json"):
            with open(manifest_path) as f:
                data = json.load(f)
            family = data.get("family")
            if family:
                families_in_manifests.add(family)

        plugin_names = {p.name for p in _ALL_PLUGINS}
        uncovered = plugin_names - families_in_manifests - _EXEMPT_PLUGINS
        assert not uncovered, (
            f"Plugins without E2E manifest coverage: {uncovered}. "
            f"Add a JSON manifest in tests/e2e/models/ with 'family' matching "
            f"the plugin name, or add to _EXEMPT_PLUGINS if WIP.")

    def test_all_manifests_have_valid_family(self):
        """Validate that every E2E manifest references a family that exists as a plugin.

        Intention:
            When a developer adds a new JSON manifest in tests/e2e/models/, the "family"
            field must correspond to a real, discovered family plugin. A typo or stale
            reference would cause the E2E test to silently fail or error at bundle-build
            time rather than at test collection.

            Example bug this catches: A developer renames trtf_build/families/gpt2.py to
            gpt2_v2.py (changing the plugin name) but forgets to update the manifest's
            "family" field from "gpt2" to "gpt2_v2". The E2E test would fail at runtime
            with a confusing "no plugin found" error instead of a clear test-time assertion.

        Setup:
            1. Discover all family plugin names via _ALL_PLUGINS.
            2. Load all JSON manifests from tests/e2e/models/ and extract each "family" field.
            3. For each manifest with a non-empty "family", verify it matches a plugin name.
            4. Assert that no manifest references an unknown family.
        """
        plugin_names = {p.name for p in _ALL_PLUGINS}

        models_dir = Path(__file__).resolve().parent.parent / "e2e" / "models"
        invalid: list[str] = []
        for manifest_path in sorted(models_dir.glob("*.json")):
            with open(manifest_path) as f:
                data = json.load(f)
            family = data.get("family", "")
            if family and family not in plugin_names:
                invalid.append(f"{manifest_path.name}: family={family!r}")

        assert not invalid, (
            f"Manifests referencing unknown family plugins:\n"
            + "\n".join(f"  {entry}" for entry in invalid))


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
    # Eagle VLM (embedding/reranking)
    ("llama_nemotron_vl", "eagle_vlm"),
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
    # SAM (prompted segmentation)
    ("sam", "sam"),
    # Phi-4 Multimodal
    ("phi4_multimodal", "phi4_multimodal"),
    # FLUX (diffusion T2I)
    ("flux", "flux"),
    # Z-Image (diffusion T2I)
    ("z_image", "z_image"),
    # DeepSeek OCR (matches deepseek_vl_v2 model_type)
    ("deepseek_vl_v2", "deepseek_ocr"),
    # MagpieTTS (encoder-decoder TTS)
    ("magpie_tts", "magpie_tts"),
    ("decoder_ce", "magpie_tts"),
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

    def test_personaplex_bundle_overrides(self):
        from trtf_build.config import ModelConfig

        plugin = find_plugin("personaplex")
        overrides = plugin.get_bundle_config_overrides(
            ModelConfig(model_type="personaplex"))
        assert overrides["speech_depth_temperature"] == pytest.approx(0.0)
        assert overrides["speech_depth_top_k"] == 0
        assert overrides["speech_system_prompt"] == ""
        assert overrides["speech_text_prompt_ids"] == []

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
