"""Unit tests for tools/test_impact.py -- zero-false-negative guarantee.

Tests use synthetic manifests and family plugins in tmp directories to
verify rule classification in isolation. The validate test uses the real
repo state.
"""

import json
import sys
from pathlib import Path

import pytest

# Add tools/ to path so we can import test_impact
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import test_impact  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_family(families_dir: Path, name: str, imports: str) -> None:
    (families_dir / f"{name}.py").write_text(imports, encoding="utf-8")


@pytest.fixture
def mock_repo(tmp_path):
    """Create a minimal mock repo with manifests and family plugins."""
    models_dir = tmp_path / "tests" / "e2e" / "models"
    models_dir.mkdir(parents=True)
    families_dir = tmp_path / "trtf_build" / "trtf_build" / "families"
    families_dir.mkdir(parents=True)
    (tmp_path / "src" / "runtime" / "plugins" / "shared").mkdir(parents=True)
    (tmp_path / "src" / "runtime" / "pipelines").mkdir(parents=True)
    (tmp_path / "include" / "trtf").mkdir(parents=True)
    (tmp_path / "tests" / "e2e_harness" / "runners").mkdir(parents=True)
    (tmp_path / "tests" / "e2e_harness" / "comparators").mkdir(parents=True)
    (tmp_path / "tests" / "e2e_harness" / "references").mkdir(parents=True)
    (tmp_path / "tests" / "builder").mkdir(parents=True)
    (tmp_path / "tests" / "cpp").mkdir(parents=True)
    (tmp_path / "tests" / "tools").mkdir(parents=True)
    (tmp_path / "tools").mkdir(parents=True)
    (tmp_path / "docs").mkdir(parents=True)

    # Manifests
    manifests = [
        {"name": "qwen3-0.6b", "family": "qwen", "runtime_strategy": "decoder_kv_cache",
         "hf_id": "Q/Q3", "core": True},
        {"name": "qwen3-4b", "family": "qwen", "runtime_strategy": "decoder_kv_cache",
         "hf_id": "Q/Q3-4b"},
        {"name": "llama-7b", "family": "llama", "runtime_strategy": "decoder_kv_cache",
         "hf_id": "meta/llama-7b"},
        {"name": "bert-base", "family": "bert", "runtime_strategy": "encoder_only",
         "hf_id": "bert-base", "core": True},
        {"name": "whisper-tiny", "family": "whisper", "runtime_strategy": "speech_to_text",
         "hf_id": "openai/whisper-tiny", "core": True},
        {"name": "flux-schnell", "family": "flux", "runtime_strategy": "diffusion_flux",
         "hf_id": "bf/FLUX", "core": True},
        {"name": "mamba-130m", "family": "mamba", "runtime_strategy": "ssm_recurrent",
         "hf_id": "ss/mamba", "core": True},
        {"name": "qwen25vl-3b", "family": "qwen_vl", "runtime_strategy": "vision_language",
         "hf_id": "Q/Q25VL", "core": True},
        {"name": "bark-small", "family": "bark", "runtime_strategy": "text_to_audio_bark",
         "hf_id": "suno/bark", "core": True},
        {"name": "sam-vit", "family": "sam", "runtime_strategy": "prompted_segmentation",
         "hf_id": "fb/sam", "core": True},
        {"name": "segformer-b0", "family": "segformer", "runtime_strategy": "segmentation",
         "hf_id": "nv/segformer", "core": True},
        {"name": "mixtral-15m", "family": "mixtral", "runtime_strategy": "decoder_moe",
         "hf_id": "mist/mixtral", "core": True},
    ]
    for m in manifests:
        _write_json(models_dir / f"{m['name']}.json", m)

    # Family plugins
    (families_dir / "__init__.py").write_text("")
    (families_dir / "base.py").write_text("")
    _write_family(families_dir, "qwen",
                  "from ..standard_decoder_builder import build\nfrom ..config import C\n")
    _write_family(families_dir, "llama",
                  "from ..standard_decoder_builder import build\nfrom ..config import C\n")
    _write_family(families_dir, "bert",
                  "from ..encoder_builder import build\nfrom ..config import C\n")
    _write_family(families_dir, "whisper",
                  "from ..config import C\nfrom ..graph_ops import rope\n")
    _write_family(families_dir, "flux",
                  "from ..config import C\n")
    _write_family(families_dir, "mamba",
                  "from ..config import C\nfrom ..graph_ops import ssm\n")
    _write_family(families_dir, "qwen_vl",
                  "from ..standard_decoder_builder import build\nfrom ..config import C\n")
    _write_family(families_dir, "bark",
                  "from ..standard_decoder_builder import build\nfrom ..config import C\n")
    _write_family(families_dir, "sam",
                  "from ..config import C\nfrom ..graph_ops import rope\n")
    _write_family(families_dir, "segformer",
                  "from ..config import C\nfrom ..graph_ops import conv\n")
    _write_family(families_dir, "mixtral",
                  "from ..standard_decoder_builder import build\nfrom ..config import C\n")

    # Placeholder source files
    (tmp_path / "trtf_build" / "trtf_build" / "standard_decoder_builder.py").write_text("")
    (tmp_path / "trtf_build" / "trtf_build" / "encoder_builder.py").write_text("")
    (tmp_path / "trtf_build" / "trtf_build" / "config.py").write_text("")
    (tmp_path / "trtf_build" / "trtf_build" / "checkpoint_mapper.py").write_text("")
    (tmp_path / "trtf_build" / "trtf_build" / "graph_ops.py").write_text("")

    return tmp_path


@pytest.fixture
def imap(mock_repo):
    return test_impact.build_impact_map(mock_repo)


# ---------------------------------------------------------------------------
# Family isolation tests
# ---------------------------------------------------------------------------


class TestFamilyPlugin:
    def test_family_only_change(self, imap):
        """families/qwen.py -> exactly qwen models."""
        match = test_impact.classify_file(
            "trtf_build/trtf_build/families/qwen.py", imap)
        assert match.rule == "family_plugin"
        assert sorted(match.models) == ["qwen3-0.6b", "qwen3-4b"]

    def test_family_isolation(self, imap):
        """families/qwen.py does NOT affect llama models."""
        match = test_impact.classify_file(
            "trtf_build/trtf_build/families/qwen.py", imap)
        assert "llama-7b" not in match.models

    def test_family_with_no_manifest(self, imap):
        """A family .py with no manifest -> empty models, no crash."""
        match = test_impact.classify_file(
            "trtf_build/trtf_build/families/nonexistent_family.py", imap)
        assert match.rule == "family_plugin"
        assert match.models == []

    def test_family_base_all_models(self, imap):
        """families/base.py -> ALL models."""
        match = test_impact.classify_file(
            "trtf_build/trtf_build/families/base.py", imap)
        assert match.rule == "family_base"
        assert sorted(match.models) == sorted(imap.all_model_names)

    def test_family_init_all_models(self, imap):
        """families/__init__.py -> ALL models."""
        match = test_impact.classify_file(
            "trtf_build/trtf_build/families/__init__.py", imap)
        assert match.rule == "family_base"
        assert len(match.models) == len(imap.all_model_names)


# ---------------------------------------------------------------------------
# Shared module tests (broad impact)
# ---------------------------------------------------------------------------


class TestSharedModules:
    def test_shared_module_all_models(self, imap):
        """checkpoint_mapper.py -> all models (no escalation)."""
        match = test_impact.classify_file(
            "trtf_build/trtf_build/checkpoint_mapper.py", imap)
        assert match.rule == "shared_builder_module"
        assert sorted(match.models) == sorted(imap.all_model_names)

    def test_shared_module_with_cap(self, imap):
        """checkpoint_mapper.py + cap -> core models only."""
        result = test_impact.analyze_impact(
            ["trtf_build/trtf_build/checkpoint_mapper.py"], imap, cap=5)
        assert result.cap_applied
        assert sorted(result.e2e_models) == sorted(imap.core_models)

    def test_graph_ops_all_models(self, imap):
        """graph_ops.py -> all models (shared utility, not a builder)."""
        match = test_impact.classify_file(
            "trtf_build/trtf_build/graph_ops.py", imap)
        assert match.rule == "shared_builder_module"
        assert len(match.models) == len(imap.all_model_names)

    def test_config_all_models(self, imap):
        """config.py -> all models."""
        match = test_impact.classify_file(
            "trtf_build/trtf_build/config.py", imap)
        assert match.rule == "shared_builder_module"
        assert len(match.models) == len(imap.all_model_names)


# ---------------------------------------------------------------------------
# Specialized builder tests
# ---------------------------------------------------------------------------


class TestSpecializedBuilder:
    def test_standard_decoder_builder(self, imap):
        """standard_decoder_builder.py -> only families that import it."""
        match = test_impact.classify_file(
            "trtf_build/trtf_build/standard_decoder_builder.py", imap)
        assert match.rule == "specialized_builder"
        # qwen, llama, qwen_vl, bark, mixtral import standard_decoder_builder
        expected_families = {"qwen", "llama", "qwen_vl", "bark", "mixtral"}
        expected_models = set()
        for fam in expected_families:
            expected_models.update(imap.family_to_models.get(fam, []))
        assert set(match.models) == expected_models

    def test_encoder_builder(self, imap):
        """encoder_builder.py -> only bert family."""
        match = test_impact.classify_file(
            "trtf_build/trtf_build/encoder_builder.py", imap)
        assert match.rule == "specialized_builder"
        assert set(match.models) == {"bert-base"}


# ---------------------------------------------------------------------------
# C++ scope tests
# ---------------------------------------------------------------------------


class TestCppScope:
    def test_cpp_plugin_scope(self, imap):
        """decoder_plugin.cpp -> only decoder_kv_cache/decoder_moe models."""
        match = test_impact.classify_file(
            "src/runtime/plugins/decoder_plugin.cpp", imap)
        assert match.rule == "cpp_plugin"
        assert match.rebuild_cpp is True
        # Should include qwen, llama (kv_cache) and mixtral (moe)
        assert "qwen3-0.6b" in match.models
        assert "mixtral-15m" in match.models
        # Should NOT include non-decoder models
        assert "bert-base" not in match.models
        assert "flux-schnell" not in match.models

    def test_cpp_plugin_vl(self, imap):
        """vl_plugin.cpp -> only vision_language models."""
        match = test_impact.classify_file(
            "src/runtime/plugins/vl_plugin.cpp", imap)
        assert match.rule == "cpp_plugin"
        assert set(match.models) == {"qwen25vl-3b"}

    def test_cpp_shared_audio(self, imap):
        """audio_helpers.h -> only audio pipeline models."""
        match = test_impact.classify_file(
            "src/runtime/plugins/shared/audio_helpers.h", imap)
        assert match.rule == "cpp_shared_helper"
        assert "whisper-tiny" in match.models
        assert "bark-small" in match.models
        assert "bert-base" not in match.models

    def test_cpp_shared_diffusion(self, imap):
        """diffusion_helpers.cpp -> only diffusion models."""
        match = test_impact.classify_file(
            "src/runtime/plugins/shared/diffusion_helpers.cpp", imap)
        assert match.rule == "cpp_shared_helper"
        assert "flux-schnell" in match.models
        assert "qwen3-0.6b" not in match.models

    def test_cpp_shared_plugin_helpers(self, imap):
        """plugin_helpers.h -> ALL models."""
        match = test_impact.classify_file(
            "src/runtime/plugins/shared/plugin_helpers.h", imap)
        assert match.rule == "cpp_shared_plugin_helpers"
        assert len(match.models) == len(imap.all_model_names)

    def test_cpp_wildcard_all(self, imap):
        """trt_common.cpp -> all models (generic C++ source)."""
        match = test_impact.classify_file(
            "src/runtime/trt/trt_common.cpp", imap)
        assert match.rule == "cpp_source"
        assert len(match.models) == len(imap.all_model_names)

    def test_cpp_pipeline_scope(self, imap):
        """text_generation_pipeline.cpp -> only decoder models."""
        match = test_impact.classify_file(
            "src/runtime/pipelines/text_generation_pipeline.cpp", imap)
        assert match.rule == "cpp_pipeline"
        assert "qwen3-0.6b" in match.models
        assert "bert-base" not in match.models

    def test_cpp_force_link(self, imap):
        """force_link_plugins.cpp -> ALL models."""
        match = test_impact.classify_file(
            "src/runtime/plugins/force_link_plugins.cpp", imap)
        assert match.rule == "cpp_force_link"
        assert len(match.models) == len(imap.all_model_names)


# ---------------------------------------------------------------------------
# Safety net tests
# ---------------------------------------------------------------------------


class TestSafetyNet:
    def test_unknown_file_triggers_all(self, imap):
        """Unknown file -> ALL models (catch-all)."""
        match = test_impact.classify_file("some/new/directory/file.py", imap)
        assert match.rule == "catch_all"
        assert sorted(match.models) == sorted(imap.all_model_names)
        assert match.rebuild_cpp is True

    def test_manifest_self(self, imap):
        """Changing a manifest JSON -> only that one model."""
        match = test_impact.classify_file(
            "tests/e2e/models/qwen3-0.6b.json", imap)
        assert match.rule == "manifest"
        assert match.models == ["qwen3-0.6b"]

    def test_cmake_triggers_all(self, imap):
        """CMakeLists.txt -> all models + rebuild flag."""
        match = test_impact.classify_file("CMakeLists.txt", imap)
        assert match.rule == "cmake"
        assert len(match.models) == len(imap.all_model_names)
        assert match.rebuild_cpp is True

    def test_include_header(self, imap):
        """include/ header -> all models."""
        match = test_impact.classify_file(
            "include/trtf/runtime/pipeline_factory.h", imap)
        assert match.rule == "cpp_source"
        assert len(match.models) == len(imap.all_model_names)


# ---------------------------------------------------------------------------
# No-impact tests
# ---------------------------------------------------------------------------


class TestNoImpact:
    def test_docs_no_impact(self, imap):
        """docs/ -> no E2E tests."""
        match = test_impact.classify_file("docs/wiki/README.md", imap)
        assert match.rule == "no_impact"
        assert match.models == []

    def test_tools_no_impact(self, imap):
        """tools/diff_logits.py -> no E2E tests."""
        match = test_impact.classify_file("tools/diff_logits.py", imap)
        assert match.rule == "no_impact"
        assert match.models == []

    def test_scripts_no_impact(self, imap):
        """scripts/ -> no E2E tests."""
        match = test_impact.classify_file("scripts/validate_family.sh", imap)
        assert match.rule == "no_impact"
        assert match.models == []

    def test_markdown_no_impact(self, imap):
        """*.md files -> no E2E tests."""
        match = test_impact.classify_file("CLAUDE.md", imap)
        assert match.rule == "no_impact"
        assert match.models == []

    def test_gitignore_no_impact(self, imap):
        """.gitignore -> no E2E tests."""
        match = test_impact.classify_file(".gitignore", imap)
        assert match.rule == "no_impact"
        assert match.models == []


# ---------------------------------------------------------------------------
# Unit tier tests
# ---------------------------------------------------------------------------


class TestUnitTiers:
    def test_unit_tier_builder(self, imap):
        """tests/builder/ -> unit tier 'builder', no E2E."""
        match = test_impact.classify_file(
            "tests/builder/test_config.py", imap)
        assert match.rule == "unit_builder"
        assert match.models == []
        assert "builder" in match.unit_tiers

    def test_unit_tier_cpp(self, imap):
        """tests/cpp/ -> unit tier 'cpp', no E2E."""
        match = test_impact.classify_file(
            "tests/cpp/test_bundle_format.cpp", imap)
        assert match.rule == "unit_cpp"
        assert match.models == []
        assert "cpp" in match.unit_tiers

    def test_unit_tier_tools(self, imap):
        """tests/tools/ -> unit tier 'tools', no E2E."""
        match = test_impact.classify_file(
            "tests/tools/test_diff_logits.py", imap)
        assert match.rule == "unit_tools"
        assert match.models == []
        assert "tools" in match.unit_tiers

    def test_source_implies_unit_tier(self, imap):
        """C++ source change implies 'cpp' unit tier alongside E2E."""
        match = test_impact.classify_file(
            "src/runtime/trt/trt_common.cpp", imap)
        assert "cpp" in match.unit_tiers
        assert len(match.models) > 0

    def test_builder_source_implies_unit_tier(self, imap):
        """Python builder source change implies 'builder' unit tier."""
        match = test_impact.classify_file(
            "trtf_build/trtf_build/families/qwen.py", imap)
        assert "builder" in match.unit_tiers


# ---------------------------------------------------------------------------
# E2E harness tests
# ---------------------------------------------------------------------------


class TestHarness:
    def test_harness_runner(self, imap):
        """runners/text_generation.py -> text_generation_causal models."""
        match = test_impact.classify_file(
            "tests/e2e_harness/runners/text_generation.py", imap)
        assert match.rule == "harness_runner"
        assert "qwen3-0.6b" in match.models
        assert "bert-base" not in match.models

    def test_harness_comparator(self, imap):
        """comparators/diffusion.py -> diffusion models."""
        match = test_impact.classify_file(
            "tests/e2e_harness/comparators/diffusion.py", imap)
        assert match.rule == "harness_comparator"
        assert "flux-schnell" in match.models
        assert "qwen3-0.6b" not in match.models

    def test_harness_shared(self, imap):
        """e2e_harness/orchestrator.py -> ALL models."""
        match = test_impact.classify_file(
            "tests/e2e_harness/orchestrator.py", imap)
        assert match.rule == "harness_shared"
        assert len(match.models) == len(imap.all_model_names)

    def test_test_e2e_entrypoint(self, imap):
        """tests/test_e2e.py -> ALL models."""
        match = test_impact.classify_file("tests/test_e2e.py", imap)
        assert match.rule == "e2e_entrypoint"
        assert len(match.models) == len(imap.all_model_names)

    def test_conftest_entrypoint(self, imap):
        """tests/conftest.py -> ALL models."""
        match = test_impact.classify_file("tests/conftest.py", imap)
        assert match.rule == "e2e_entrypoint"
        assert len(match.models) == len(imap.all_model_names)


# ---------------------------------------------------------------------------
# Aggregation / cap tests
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_multiple_families(self, imap):
        """Multiple family changes -> union of models."""
        result = test_impact.analyze_impact([
            "trtf_build/trtf_build/families/qwen.py",
            "trtf_build/trtf_build/families/llama.py",
        ], imap)
        assert "qwen3-0.6b" in result.e2e_models
        assert "llama-7b" in result.e2e_models
        assert not result.cap_applied

    def test_cap_not_applied_when_under(self, imap):
        """Cap not applied when affected models <= cap."""
        result = test_impact.analyze_impact(
            ["trtf_build/trtf_build/families/qwen.py"], imap, cap=5)
        assert not result.cap_applied
        assert sorted(result.e2e_models) == ["qwen3-0.6b", "qwen3-4b"]

    def test_cap_applied_when_over(self, imap):
        """Cap applied when affected models > cap."""
        result = test_impact.analyze_impact(
            ["trtf_build/trtf_build/checkpoint_mapper.py"], imap, cap=5)
        assert result.cap_applied
        assert sorted(result.e2e_models) == sorted(imap.core_models)

    def test_no_changed_files(self, imap):
        """No files -> no impact."""
        result = test_impact.analyze_impact([], imap)
        assert result.e2e_models == []
        assert result.unit_tiers == []
        assert not result.rebuild_cpp

    def test_mixed_impact(self, imap):
        """Family plugin + unit test -> models + unit tier."""
        result = test_impact.analyze_impact([
            "trtf_build/trtf_build/families/qwen.py",
            "tests/builder/test_config.py",
        ], imap)
        assert "qwen3-0.6b" in result.e2e_models
        assert "builder" in result.unit_tiers


# ---------------------------------------------------------------------------
# Validation test (uses real repo)
# ---------------------------------------------------------------------------


class TestValidation:
    def test_validate_consistency(self):
        """Runs --validate on the real repo and checks it passes."""
        real_root = REPO_ROOT
        if not (real_root / "tests" / "e2e" / "models").is_dir():
            pytest.skip("Not in the project repo")
        imap = test_impact.build_impact_map(real_root)
        errors = test_impact.validate_map(imap, real_root)
        assert errors == [], f"Validation errors: {errors}"

    def test_real_repo_has_core_models(self):
        """Real repo has at least 5 core models."""
        real_root = REPO_ROOT
        if not (real_root / "tests" / "e2e" / "models").is_dir():
            pytest.skip("Not in the project repo")
        imap = test_impact.build_impact_map(real_root)
        assert len(imap.core_models) >= 5, (
            f"Expected at least 5 core models, got {len(imap.core_models)}"
        )


# ---------------------------------------------------------------------------
# Output format tests
# ---------------------------------------------------------------------------


class TestOutput:
    def test_human_format(self, imap):
        result = test_impact.ImpactResult(
            e2e_models=["qwen3-0.6b"],
            unit_tiers=["builder"],
            rebuild_cpp=False,
            cap_applied=False,
            matched_rules=[],
        )
        output = test_impact.format_human(result)
        assert "qwen3-0.6b" in output
        assert "builder" in output
        assert "rebuild needed: no" in output

    def test_json_format(self, imap):
        result = test_impact.ImpactResult(
            e2e_models=["qwen3-0.6b", "qwen3-4b"],
            unit_tiers=["builder"],
            rebuild_cpp=False,
            cap_applied=False,
            matched_rules=[{"file": "f.py", "rule": "family_plugin", "models": ["qwen3-0.6b"]}],
        )
        output = test_impact.format_json(result)
        data = json.loads(output)
        assert data["e2e_models"] == ["qwen3-0.6b", "qwen3-4b"]
        assert data["rebuild_cpp"] is False

    def test_json_cap_applied(self, imap):
        result = test_impact.ImpactResult(
            e2e_models=sorted(imap.core_models),
            unit_tiers=[],
            rebuild_cpp=True,
            cap_applied=True,
            matched_rules=[],
        )
        output = test_impact.format_json(result)
        data = json.loads(output)
        assert data["cap_applied"] is True
