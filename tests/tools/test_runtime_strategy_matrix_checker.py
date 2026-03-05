"""Unit tests for tools/check_runtime_strategy_matrix.py."""

from __future__ import annotations

import importlib
from pathlib import Path


def _import_checker():
    return importlib.import_module("check_runtime_strategy_matrix")


def test_extract_runtime_strategies_from_cpp_reads_registry_dispatch_and_decoder_constraints(
    tmp_path: Path,
):
    mod = _import_checker()

    cpp = tmp_path / "trtf_c.cpp"
    cpp.write_text(
        """
        (void) trtf::cabi::register_backend_factory("decoder_kv_cache", nullptr);
        (void) trtf::cabi::register_backend_factory("segmentation", nullptr);
        if (fp_cfg_early.runtime_strategy == "diffusion") {}
        if (strategy == "vision_language") {}
        if (strategy != "decoder_kv_cache" && strategy != "decoder_moe") {}
        """,
        encoding="utf-8",
    )

    strategies = mod.extract_runtime_strategies_from_cpp(cpp)
    assert strategies == {
        "decoder_kv_cache",
        "decoder_moe",
        "diffusion",
        "segmentation",
        "vision_language",
    }


def test_extract_registry_strategy_wrapper_bindings_from_cpp_reads_wrapper_symbols(
    tmp_path: Path,
):
    mod = _import_checker()

    cpp = tmp_path / "trtf_c.cpp"
    cpp.write_text(
        """
        (void) trtf::cabi::register_backend_factory(
            "decoder_kv_cache", &create_decoder_pipeline_via_registry);
        (void) trtf::cabi::register_backend_factory(
            "segmentation",
            &create_segmentation_pipeline_via_registry);
        """,
        encoding="utf-8",
    )

    bindings = mod.extract_registry_strategy_wrapper_bindings_from_cpp(cpp)
    assert bindings == {
        "decoder_kv_cache": "create_decoder_pipeline_via_registry",
        "segmentation": "create_segmentation_pipeline_via_registry",
    }


def test_extract_registry_wrapper_definitions_from_cpp_ignores_declarations(
    tmp_path: Path,
):
    mod = _import_checker()

    cpp = tmp_path / "backend_registry_wrappers.cpp"
    cpp.write_text(
        """
        trtf::IPipeline* create_decoder_pipeline_via_registry(void* opaque_context);
        trtf::IPipeline* create_segmentation_pipeline_via_registry(void* opaque_context)
        {
            return nullptr;
        }
        """,
        encoding="utf-8",
    )

    symbols = mod.extract_registry_wrapper_definitions_from_cpp(cpp)
    assert symbols == {"create_segmentation_pipeline_via_registry"}


def test_validate_matrix_data_requires_exemption_when_no_diff_check():
    mod = _import_checker()
    errors = mod.validate_matrix_data(
        matrix={
            "rwkv_recurrent": {
                "task_strategy": "text_generation_causal",
                "cli_commands": ["run"],
                "runner_class": "TextGenerationCausalRunner",
                "comparator_class": "TextComparator",
                "diff_framework_check_classes": [],
            }
        },
        cpp_runtime_strategies={"rwkv_recurrent"},
        runtime_to_task_strategy={"rwkv_recurrent": "text_generation_causal"},
        diff_checks_by_strategy={},
        runner_classes_by_task={"text_generation_causal": {"TextGenerationCausalRunner"}},
        comparator_classes_by_task={"text_generation_causal": {"TextComparator"}},
    )

    assert any("diff_framework_exemption" in message for message in errors)


def test_validate_matrix_data_detects_diff_check_mismatch():
    mod = _import_checker()
    errors = mod.validate_matrix_data(
        matrix={
            "vision_language": {
                "task_strategy": "vision_language_generation",
                "cli_commands": ["run"],
                "runner_class": "VisionLanguageRunner",
                "comparator_class": "VisionLanguageComparator",
                "diff_framework_check_classes": ["WrongCheckName"],
            }
        },
        cpp_runtime_strategies={"vision_language"},
        runtime_to_task_strategy={"vision_language": "vision_language_generation"},
        diff_checks_by_strategy={"vision_language": {"VLPipelineTest"}},
        runner_classes_by_task={"vision_language_generation": {"VisionLanguageRunner"}},
        comparator_classes_by_task={"vision_language_generation": {"VisionLanguageComparator"}},
    )

    assert any("does not match discovered checks" in message for message in errors)


def test_validate_registry_wrapper_coverage_requires_bindings_for_matrix_strategies():
    mod = _import_checker()
    errors = mod.validate_registry_wrapper_coverage(
        matrix_strategies={"decoder_kv_cache", "diffusion", "vision_language"},
        registry_wrapper_bindings={
            "decoder_kv_cache": "create_decoder_pipeline_via_registry"
        },
        direct_runtime_strategies={"diffusion"},
        wrapper_definitions={"create_decoder_pipeline_via_registry"},
    )

    assert any(
        "register_backend_factory strategy keys missing" in message
        and "vision_language" in message
        for message in errors
    )


def test_validate_registry_wrapper_coverage_requires_wrapper_symbol_definitions():
    mod = _import_checker()
    errors = mod.validate_registry_wrapper_coverage(
        matrix_strategies={"decoder_kv_cache"},
        registry_wrapper_bindings={
            "decoder_kv_cache": "create_decoder_pipeline_via_registry"
        },
        direct_runtime_strategies=set(),
        wrapper_definitions=set(),
    )

    assert any("without definitions" in message for message in errors)


def test_validate_trtf_c_wrapper_extraction_boundary_is_noop_before_extraction():
    mod = _import_checker()
    errors = mod.validate_trtf_c_wrapper_extraction_boundary(
        trtf_c_wrapper_definitions={"create_decoder_pipeline_via_registry"},
        extracted_wrapper_definitions=set(),
    )
    assert errors == []


def test_validate_trtf_c_wrapper_extraction_boundary_detects_lingering_definitions():
    mod = _import_checker()
    errors = mod.validate_trtf_c_wrapper_extraction_boundary(
        trtf_c_wrapper_definitions={"create_decoder_pipeline_via_registry"},
        extracted_wrapper_definitions={"create_decoder_pipeline_via_registry"},
    )
    assert any(
        "trtf_c.cpp still defines *_via_registry wrappers after extraction" in message
        for message in errors
    )


def test_extract_runtime_strategies_from_cpp_files_aggregates_across_files(
    tmp_path: Path,
):
    mod = _import_checker()

    trtf_c = tmp_path / "trtf_c.cpp"
    trtf_c.write_text(
        """
        if (strategy == "vision_language") {}
        """,
        encoding="utf-8",
    )

    registry_dispatch = tmp_path / "backend_registry_dispatch.cpp"
    registry_dispatch.write_text(
        """
        (void) trtf::cabi::register_backend_factory("decoder_kv_cache", nullptr);
        if (strategy != "decoder_moe") {}
        """,
        encoding="utf-8",
    )

    strategies = mod.extract_runtime_strategies_from_cpp_files(
        [trtf_c, registry_dispatch]
    )
    assert strategies == {"decoder_kv_cache", "decoder_moe", "vision_language"}


def test_validate_matrix_paths_supports_multi_file_registry_extraction(tmp_path: Path):
    mod = _import_checker()

    matrix_path = tmp_path / "tests" / "runtime_strategy_matrix.yaml"
    matrix_path.parent.mkdir(parents=True)
    matrix_path.write_text(
        """
        {
          "runtime_strategies": {
            "decoder_kv_cache": {
              "task_strategy": "text_generation_causal",
              "cli_commands": ["run"],
              "runner_class": "TextGenerationCausalRunner",
              "comparator_class": "TextComparator",
              "diff_framework_check_classes": ["DecoderCheck"]
            }
          }
        }
        """,
        encoding="utf-8",
    )

    contracts_path = tmp_path / "tests" / "e2e_harness" / "contracts.py"
    contracts_path.parent.mkdir(parents=True)
    contracts_path.write_text(
        'RUNTIME_TO_TASK_STRATEGY = {"decoder_kv_cache": "text_generation_causal"}\n',
        encoding="utf-8",
    )

    diff_checks_dir = tmp_path / "tools" / "diff_framework" / "checks"
    diff_checks_dir.mkdir(parents=True)
    (diff_checks_dir / "decoder_check.py").write_text(
        'class DecoderCheck:\n    runtime_strategies = ["decoder_kv_cache"]\n',
        encoding="utf-8",
    )

    runners_dir = tmp_path / "tests" / "e2e_harness" / "runners"
    runners_dir.mkdir(parents=True)
    (runners_dir / "decoder_runner.py").write_text(
        (
            "class TextGenerationCausalRunner:\n"
            "    def strategy_name(self):\n"
            '        return "text_generation_causal"\n'
        ),
        encoding="utf-8",
    )

    comparators_dir = tmp_path / "tests" / "e2e_harness" / "comparators"
    comparators_dir.mkdir(parents=True)
    (comparators_dir / "decoder_comparator.py").write_text(
        (
            "class TextComparator:\n"
            "    def task_strategy(self):\n"
            '        return "text_generation_causal"\n'
        ),
        encoding="utf-8",
    )

    cabi_dir = tmp_path / "src" / "cabi"
    cabi_dir.mkdir(parents=True)
    (cabi_dir / "trtf_c.cpp").write_text(
        "// wrappers extracted elsewhere\n",
        encoding="utf-8",
    )
    (cabi_dir / "backend_registry_dispatch.cpp").write_text(
        """
        (void) trtf::cabi::register_backend_factory(
            "decoder_kv_cache", &create_decoder_pipeline_via_registry);
        """,
        encoding="utf-8",
    )
    (cabi_dir / "backend_registry_wrappers.cpp").write_text(
        """
        trtf::IPipeline* create_decoder_pipeline_via_registry(void* opaque_context) {
            return nullptr;
        }
        """,
        encoding="utf-8",
    )

    errors = mod.validate_matrix_paths(
        matrix_path=matrix_path,
        cpp_path=cabi_dir / "trtf_c.cpp",
        contracts_path=contracts_path,
        diff_checks_dir=diff_checks_dir,
        runners_dir=runners_dir,
        comparators_dir=comparators_dir,
    )
    assert errors == []


def test_infer_cabi_dir_from_nested_cpp_path(tmp_path: Path):
    mod = _import_checker()
    nested_cpp = tmp_path / "src" / "cabi" / "api" / "trtf_c.cpp"
    nested_cpp.parent.mkdir(parents=True)
    nested_cpp.write_text("// test", encoding="utf-8")

    inferred = mod.infer_cabi_dir_from_cpp_path(
        cpp_path=nested_cpp,
        default_cabi_dir=tmp_path / "fallback" / "cabi",
    )
    assert inferred == (tmp_path / "src" / "cabi").resolve()


def test_repository_matrix_is_consistent():
    mod = _import_checker()
    errors = mod.validate_matrix_paths()
    assert errors == []
