# Declarative runtime config schema manifest.
#
# Each entry is:
#   schema_source.cpp|registration_function
#
# CMake consumes this list for both compilation and generated registrar calls,
# matching the pipeline plugin manifest pattern.

include("${CMAKE_CURRENT_LIST_DIR}/trtf_registration_manifest.cmake")

set(TRTF_CONFIG_SCHEMAS
  "triattention.cpp|register_triattention_schema"
  "decode_policy.cpp|register_decode_policy_schema"
  "text_trace.cpp|register_text_trace_schema"
  "runtime.cpp|register_runtime_schema"
  "audio_bark.cpp|register_audio_bark_schema"
  "audio_magpie.cpp|register_audio_magpie_schema"
  "platform.cpp|register_platform_schema"
)

set(TRTF_CONFIG_SCHEMA_REGISTRATION_SOURCE
  "${PROJECT_BINARY_DIR}/generated/register_schemas.cpp")
trtf_configure_registration_manifest(
  TRTF_CONFIG_SCHEMAS
  "${PROJECT_SOURCE_DIR}/src/runtime/config/schemas"
  "${CMAKE_CURRENT_LIST_DIR}/register_schemas.cpp.in"
  "${TRTF_CONFIG_SCHEMA_REGISTRATION_SOURCE}"
  TRTF_CONFIG_SCHEMA_SOURCES
  TRTF_CONFIG_SCHEMA_REGISTRATION_DECLS
  TRTF_CONFIG_SCHEMA_REGISTRATION_CALLS
  "    "
  "::trtf::config::SchemaRegistry"
)
