"""Declarative, namespaced, self-registering runtime configuration.

Mirrors the C++ registry in ``include/trtf/config/`` and
``src/runtime/config/``. The design contract is documented in
``docs/context/2026-04-20-config-registry-status.md``:

    - Static init / module import: registers *schemas* only (metadata +
      defaults). No values.
    - Session start: the CLI/profile loader assembles a
      ``ConfigBundle`` by merging layered contributions against the
      registered schemas. Highest priority wins per field; schema
      defaults fill gaps.
    - Features query their own namespace via
      ``bundle.get("triattention", "kv_budget")``.

The bundle provides defaults, not ground truth. The runtime never
"overrides" the bundle — that word is forbidden in any identifier
inside this package.
"""

from trtf_build.runtime_config.schema_registry import (
    ConfigField,
    Layer,
    Schema,
    SchemaRegistry,
    clear_for_testing,
    lookup,
    register_schema,
    registered_namespaces,
)
from trtf_build.runtime_config.config_bundle import (
    ConfigBundle,
    LayerContribution,
    ResolvedValue,
    write_effective_config,
)
from trtf_build.runtime_config.cli_support import (
    build_cli_contribution,
    coerce_scalar,
    load_layered_file,
    parse_set_token,
    parse_set_tokens,
    resolve_cli_config,
    write_effective_config_next_to,
)

__all__ = [
    "ConfigBundle",
    "ConfigField",
    "Layer",
    "LayerContribution",
    "ResolvedValue",
    "Schema",
    "SchemaRegistry",
    "build_cli_contribution",
    "clear_for_testing",
    "coerce_scalar",
    "load_layered_file",
    "lookup",
    "parse_set_token",
    "parse_set_tokens",
    "register_schema",
    "registered_namespaces",
    "resolve_cli_config",
    "write_effective_config",
    "write_effective_config_next_to",
]
