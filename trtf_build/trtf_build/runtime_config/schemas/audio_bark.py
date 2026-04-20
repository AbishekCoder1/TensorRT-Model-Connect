"""Schema for the ``audio_bark`` namespace.

Per-session sampling / debug knobs for the Bark TTS pipeline. Replaces
``TRTF_BARK_DUMP``, ``TRTF_BARK_GREEDY``, ``TRTF_BARK_SEED``.

Using ``audio_bark`` (underscore) instead of ``audio.bark`` because the
current registry keys the namespace on a single string without dotted
sub-paths — matching how ``triattention``, ``decode_policy``, etc. work.
"""

from __future__ import annotations

from trtf_build.runtime_config import (
    ConfigField,
    Layer,
    Schema,
    register_schema,
)


_SESSION = frozenset({Layer.SESSION_REQUEST, Layer.PLATFORM_PROFILE})


SCHEMA = Schema(
    namespace="audio_bark",
    fields=(
        ConfigField(
            name="dump_path",
            type_tag="string",
            default="",  # empty -> no dump
            allowed_layers=_SESSION,
        ),
        ConfigField(
            name="greedy",
            type_tag="bool",
            default=False,
            allowed_layers=_SESSION,
        ),
        ConfigField(
            name="seed",
            type_tag="int64",
            default=-1,  # -1 -> use default RNG state
            allowed_layers=_SESSION,
        ),
    ),
)


register_schema(SCHEMA)
