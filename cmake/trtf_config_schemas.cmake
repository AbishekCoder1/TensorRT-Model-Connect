# Declarative runtime config schema manifest.
#
# Each entry is:
#   schema_source.cpp|force_link_symbol
#
# CMake consumes this list for both compilation and generated linker anchors,
# matching the pipeline plugin manifest pattern.

include("${CMAKE_CURRENT_LIST_DIR}/trtf_force_link_manifest.cmake")

set(TRTF_CONFIG_SCHEMAS
  "triattention.cpp|kForceLink_triattention"
  "decode_policy.cpp|kForceLink_decode_policy"
  "text_trace.cpp|kForceLink_text_trace"
  "runtime.cpp|kForceLink_runtime"
  "audio_bark.cpp|kForceLink_audio_bark"
  "audio_magpie.cpp|kForceLink_audio_magpie"
  "platform.cpp|kForceLink_platform"
)

set(TRTF_CONFIG_SCHEMA_FORCE_LINK_SOURCE
  "${PROJECT_BINARY_DIR}/generated/force_link_schemas.cpp")
trtf_configure_force_link_manifest(
  TRTF_CONFIG_SCHEMAS
  "${PROJECT_SOURCE_DIR}/src/runtime/config/schemas"
  "${CMAKE_CURRENT_LIST_DIR}/force_link_schemas.cpp.in"
  "${TRTF_CONFIG_SCHEMA_FORCE_LINK_SOURCE}"
  TRTF_CONFIG_SCHEMA_SOURCES
  TRTF_CONFIG_SCHEMA_FORCE_LINK_DECLS
  TRTF_CONFIG_SCHEMA_FORCE_LINK_ANCHORS
  "    "
)
