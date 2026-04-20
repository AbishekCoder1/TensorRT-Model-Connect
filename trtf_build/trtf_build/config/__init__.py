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

from trtf_build.config.schema_registry import (
    ConfigField,
    Layer,
    Schema,
    SchemaRegistry,
    clear_for_testing,
    lookup,
    register_schema,
    registered_namespaces,
)
from trtf_build.config.config_bundle import (
    ConfigBundle,
    LayerContribution,
    ResolvedValue,
    write_effective_config,
)

__all__ = [
    "ConfigBundle",
    "ConfigField",
    "Layer",
    "LayerContribution",
    "ResolvedValue",
    "Schema",
    "SchemaRegistry",
    "clear_for_testing",
    "lookup",
    "register_schema",
    "registered_namespaces",
    "write_effective_config",
]
