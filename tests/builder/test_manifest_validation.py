"""Tests for E2E manifest schema validation.

Trace: ARCH-E2E-001, UD-E2E-MANIFEST
Intent: Validate E2E manifest schema enforcement for required fields, type checks, and skip semantics
Preconditions: e2e_harness manifest_loader is importable
Postconditions: Invalid manifests raise appropriate ValueError/TypeError; valid manifests pass validation
"""
import json
import os
import pytest
import warnings

# Try to import manifest_loader
try:
    from tests.e2e_harness.manifest_loader import load_manifest, _validate_manifest
except ImportError:
    pytest.skip("e2e_harness not available", allow_module_level=True)


class TestManifestValidation:
    """Test manifest schema validation."""

    def _write_manifest(self, tmp_path, data):
        path = os.path.join(str(tmp_path), "test.json")
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def test_missing_name_raises(self, tmp_path):
        """Manifest without 'name' should raise ValueError."""
        path = self._write_manifest(tmp_path, {"hf_id": "org/model", "family": "qwen"})
        with pytest.raises(ValueError, match="name"):
            _validate_manifest(json.load(open(path)), path)

    def test_missing_hf_id_raises_when_not_skipped(self, tmp_path):
        """Manifest without 'hf_id' (and no skip) should raise ValueError."""
        path = self._write_manifest(tmp_path, {"name": "test-model", "family": "qwen"})
        with pytest.raises(ValueError, match="hf_id"):
            _validate_manifest(json.load(open(path)), path)

    def test_missing_family_raises_when_not_skipped(self, tmp_path):
        """Manifest without 'family' (and no skip) should raise ValueError."""
        path = self._write_manifest(tmp_path, {"name": "test-model", "hf_id": "org/model"})
        with pytest.raises(ValueError, match="family"):
            _validate_manifest(json.load(open(path)), path)

    def test_skipped_manifest_allows_missing_hf_id(self, tmp_path):
        """Skipped manifests don't need hf_id or family."""
        data = {"name": "test-model", "skip": "not available"}
        _validate_manifest(data, "test.json")  # Should not raise

    def test_wrong_type_max_new_tokens_string(self, tmp_path):
        """max_new_tokens must be int, not string."""
        data = {"name": "test", "hf_id": "org/m", "family": "qwen", "max_new_tokens": "20"}
        with pytest.raises(TypeError, match="max_new_tokens"):
            _validate_manifest(data, "test.json")

    def test_wrong_type_max_new_tokens_float(self, tmp_path):
        """max_new_tokens must be int, not float."""
        data = {"name": "test", "hf_id": "org/m", "family": "qwen", "max_new_tokens": 20.5}
        with pytest.raises(TypeError, match="max_new_tokens"):
            _validate_manifest(data, "test.json")

    def test_wrong_type_max_cache_length(self, tmp_path):
        """max_cache_length must be int, not float."""
        data = {"name": "test", "hf_id": "org/m", "family": "qwen", "max_cache_length": 256.5}
        with pytest.raises(TypeError, match="max_cache_length"):
            _validate_manifest(data, "test.json")

    def test_unknown_runtime_strategy_warns(self, tmp_path):
        """Unknown runtime_strategy should emit a warning."""
        data = {
            "name": "test",
            "hf_id": "org/m",
            "family": "qwen",
            "runtime_strategy": "bogus_strategy",
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_manifest(data, "test.json")
            assert any("bogus_strategy" in str(warning.message) for warning in w)

    def test_known_runtime_strategy_no_warning(self, tmp_path):
        """Known runtime_strategy should not emit a warning."""
        data = {
            "name": "test",
            "hf_id": "org/m",
            "family": "qwen",
            "runtime_strategy": "decoder_kv_cache",
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_manifest(data, "test.json")
            strategy_warnings = [
                x for x in w if "runtime_strategy" in str(x.message)
            ]
            assert len(strategy_warnings) == 0

    def test_valid_manifest_passes(self, tmp_path):
        """A fully valid manifest should pass without errors."""
        data = {
            "name": "qwen3-test",
            "hf_id": "Qwen/Qwen3-0.6B",
            "family": "qwen",
            "bundle": "qwen3-test.trtfb",
            "runtime_strategy": "decoder_kv_cache",
            "max_cache_length": 256,
            "max_new_tokens": 20,
            "prompt": "Hello",
        }
        _validate_manifest(data, "test.json")  # Should not raise

    def test_model_id_accepted_as_hf_id_alternative(self, tmp_path):
        """model_id is accepted as an alternative to hf_id."""
        data = {
            "name": "test-model",
            "model_id": "org/model",
            "family": "qwen",
        }
        _validate_manifest(data, "test.json")  # Should not raise

    def test_load_manifest_calls_validation(self, tmp_path):
        """load_manifest should call _validate_manifest and raise on bad input."""
        path = self._write_manifest(tmp_path, {"hf_id": "org/model", "family": "qwen"})
        with pytest.raises(ValueError, match="name"):
            load_manifest(path)

    def test_bool_not_accepted_as_int(self, tmp_path):
        """Boolean values should not pass the int type check."""
        data = {
            "name": "test",
            "hf_id": "org/m",
            "family": "qwen",
            "max_new_tokens": True,
        }
        with pytest.raises(TypeError, match="max_new_tokens"):
            _validate_manifest(data, "test.json")

    def test_execution_profiles_must_be_object(self, tmp_path):
        data = {
            "name": "test",
            "hf_id": "org/m",
            "family": "qwen",
            "execution_profiles": "chronos",
        }
        with pytest.raises(TypeError, match="execution_profiles"):
            _validate_manifest(data, "test.json")

    def test_execution_profiles_reject_unknown_phase(self, tmp_path):
        data = {
            "name": "test",
            "hf_id": "org/m",
            "family": "qwen",
            "execution_profiles": {"build": "chronos", "verify": "chronos"},
        }
        with pytest.raises(ValueError, match="unsupported phase"):
            _validate_manifest(data, "test.json")

    def test_load_manifest_applies_family_default_execution_profiles(self, tmp_path):
        path = self._write_manifest(
            tmp_path,
            {
                "name": "chronos-case",
                "hf_id": "amazon/chronos-bolt-tiny",
                "family": "chronos_bolt",
                "runtime_strategy": "chronos_bolt_torchtrt",
                "reference_backend": "torch_reference",
            },
        )
        case = load_manifest(path)
        assert case.execution_profiles["build"] == "chronos"
        assert case.execution_profiles["runtime"] == "base"
        assert case.execution_profiles["reference"] == "chronos"

    def test_load_manifest_preserves_execution_profile_overrides(self, tmp_path):
        path = self._write_manifest(
            tmp_path,
            {
                "name": "chronos-case",
                "hf_id": "amazon/chronos-bolt-tiny",
                "family": "chronos_bolt",
                "runtime_strategy": "chronos_bolt_torchtrt",
                "reference_backend": "torch_reference",
                "execution_profiles": {"runtime": "custom-runtime"},
            },
        )
        case = load_manifest(path)
        assert case.execution_profiles["build"] == "chronos"
        assert case.execution_profiles["runtime"] == "custom-runtime"
        assert case.execution_profiles["reference"] == "chronos"
