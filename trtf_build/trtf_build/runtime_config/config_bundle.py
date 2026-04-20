"""Python mirror of ``trtf::config::ConfigBundle``.

Builds an immutable resolved configuration by merging layered
contributions against the currently-registered schemas. Priority order:

    SESSION_REQUEST > PLATFORM_PROFILE > BUNDLE_DEFAULT > BUILD_TIME > SCHEMA_DEFAULT

Also writes ``effective_config.json`` — a human-inspectable record of
what value each field received and which layer contributed it. One file
per session answers "what did I actually run with?".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from trtf_build.runtime_config.schema_registry import (
    Layer,
    Schema,
    SchemaRegistry,
    _singleton_for_testing,
    layer_name,
    registered_namespaces,
    lookup,
)


@dataclass(frozen=True)
class LayerContribution:
    """One layer's input: values keyed by ``(namespace, field)``."""

    layer: Layer
    values: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class ResolvedValue:
    """A resolved field value tagged with its source layer."""

    value: Any
    source: Layer


class ConfigBundle:
    """Immutable resolved configuration produced by :meth:`build`."""

    def __init__(self, resolved: Dict[str, Dict[str, ResolvedValue]]):
        self._resolved = resolved

    # ---- construction -------------------------------------------------

    @classmethod
    def build(
        cls,
        contributions: Iterable[LayerContribution],
        registry: Optional[SchemaRegistry] = None,
    ) -> "ConfigBundle":
        """Merge ``contributions`` against the registered schemas.

        Raises ``ValueError`` for allowlist violations, unknown
        namespace or field contributions, or validator rejections.
        """
        reg = registry if registry is not None else _singleton_for_testing()
        contribs = list(contributions)
        _validate_contributions(contribs, reg)

        resolved: Dict[str, Dict[str, ResolvedValue]] = {}
        for ns in reg.registered_namespaces():
            schema = reg.lookup(ns)
            assert schema is not None
            ns_resolved: Dict[str, ResolvedValue] = {}
            for cfg_field in schema.fields:
                picked = _pick_highest_priority(contribs, ns, cfg_field.name)
                if picked is None:
                    ns_resolved[cfg_field.name] = ResolvedValue(
                        value=cfg_field.default, source=Layer.SCHEMA_DEFAULT,
                    )
                else:
                    value, source = picked
                    ns_resolved[cfg_field.name] = ResolvedValue(
                        value=value, source=source,
                    )
            resolved[ns] = ns_resolved
        return cls(resolved)

    # ---- access -------------------------------------------------------

    def get(self, namespace: str, field_name: str) -> Any:
        """Return the resolved value. Raises ``KeyError`` if unknown."""
        ns_map = self._resolved.get(namespace)
        if ns_map is None:
            raise KeyError(f"ConfigBundle: unknown namespace: {namespace}")
        rv = ns_map.get(field_name)
        if rv is None:
            raise KeyError(
                f"ConfigBundle: unknown field: {namespace}.{field_name}"
            )
        return rv.value

    def source_of(self, namespace: str, field_name: str) -> Layer:
        ns_map = self._resolved.get(namespace)
        if ns_map is None:
            raise KeyError(f"ConfigBundle: unknown namespace: {namespace}")
        rv = ns_map.get(field_name)
        if rv is None:
            raise KeyError(
                f"ConfigBundle: unknown field: {namespace}.{field_name}"
            )
        return rv.source

    def all(self) -> Mapping[str, Mapping[str, ResolvedValue]]:
        return self._resolved

    # ---- serialization ------------------------------------------------

    def to_effective_dict(self) -> Dict[str, Any]:
        """Produce a JSON-serializable record of resolved values and sources."""
        out: Dict[str, Any] = {}
        for ns in sorted(self._resolved.keys()):
            ns_out: Dict[str, Any] = {}
            for fname in sorted(self._resolved[ns].keys()):
                rv = self._resolved[ns][fname]
                ns_out[fname] = {
                    "value": _jsonable(rv.value),
                    "source": layer_name(rv.source),
                }
            out[ns] = ns_out
        return out


# ---- helpers ----------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of Python values to JSON-serializable form.

    Accepts primitives, lists/tuples, dicts; converts ``frozenset``/``set``
    to sorted lists. Falls back to ``str`` for unknown types so the
    effective-config dump never fails the session.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted([_jsonable(v) for v in value])
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _validate_contributions(
    contributions: List[LayerContribution], registry: SchemaRegistry
) -> None:
    for contrib in contributions:
        for ns, fields in contrib.values.items():
            schema = registry.lookup(ns)
            if schema is None:
                raise ValueError(
                    f"Layer {layer_name(contrib.layer)} contributed value for "
                    f"unregistered namespace: {ns}"
                )
            fields_by_name = {f.name: f for f in schema.fields}
            for field_name, value in fields.items():
                cfg_field = fields_by_name.get(field_name)
                if cfg_field is None:
                    raise ValueError(
                        f"Layer {layer_name(contrib.layer)} contributed value "
                        f"for unknown field: {ns}.{field_name}"
                    )
                if contrib.layer not in cfg_field.allowed_layers:
                    raise ValueError(
                        f"Layer {layer_name(contrib.layer)} is not permitted "
                        f"to set {ns}.{field_name}"
                    )
                if cfg_field.validator is not None and not cfg_field.validator(value):
                    raise ValueError(
                        f"Validator rejected value for {ns}.{field_name} "
                        f"from layer {layer_name(contrib.layer)}"
                    )


def _pick_highest_priority(
    contributions: List[LayerContribution], ns: str, field_name: str,
) -> Optional[Tuple[Any, Layer]]:
    best: Optional[Tuple[Any, Layer]] = None
    for contrib in contributions:
        ns_map = contrib.values.get(ns)
        if ns_map is None:
            continue
        if field_name not in ns_map:
            continue
        value = ns_map[field_name]
        if best is None or contrib.layer.value > best[1].value:
            best = (value, contrib.layer)
    return best


# ---- effective_config.json writer --------------------------------------


def write_effective_config(
    bundle: ConfigBundle, path: str | Path,
) -> Path:
    """Write the effective configuration to ``path`` as pretty JSON.

    Returns the resolved :class:`~pathlib.Path` for convenience.
    """
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as handle:
        json.dump(bundle.to_effective_dict(), handle, indent=2, sort_keys=False)
        handle.write("\n")
    return resolved_path
