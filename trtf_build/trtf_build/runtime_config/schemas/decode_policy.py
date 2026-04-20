"""Schema for the ``decode_policy`` namespace.

Covers the build-time decoder-attention knobs previously read via
environment variables. The namespace is deliberately small — most runtime
decode behavior lives in other namespaces.
"""

from __future__ import annotations

from trtf_build.runtime_config import (
    ConfigField,
    Layer,
    Schema,
    register_schema,
)


_BUILD_AND_BUNDLE = frozenset({Layer.BUILD_TIME, Layer.BUNDLE_DEFAULT})


SCHEMA = Schema(
    namespace="decode_policy",
    fields=(
        # Replaces TRTF_FORCE_MANUAL_DECODER_ATTENTION (build-time env var).
        # Build-time only: chosen when the bundle is built and baked into
        # the engine graph — cannot change at run time. Session/platform
        # layers are not in the allowlist.
        ConfigField(
            name="force_manual_attention",
            type_tag="bool",
            default=False,
            allowed_layers=_BUILD_AND_BUNDLE,
        ),
    ),
)


register_schema(SCHEMA)
