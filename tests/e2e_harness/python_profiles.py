"""Thin wrapper over the shared trtf-build Python profile registry."""

from trtf_build.python_profiles import (
    DEFAULT_PROFILE,
    PROFILE_PHASES,
    profile_env_var,
    profile_env_var_candidates,
    normalize_execution_profiles,
    resolve_case_profile_names,
    resolve_case_python_profiles,
    resolve_profile_python,
)

__all__ = [
    "DEFAULT_PROFILE",
    "PROFILE_PHASES",
    "profile_env_var",
    "profile_env_var_candidates",
    "normalize_execution_profiles",
    "resolve_case_profile_names",
    "resolve_case_python_profiles",
    "resolve_profile_python",
]
