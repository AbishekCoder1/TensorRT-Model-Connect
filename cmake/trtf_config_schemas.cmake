# Declarative runtime config schema manifest.
#
# Each entry is:
#   schema_source.cpp|force_link_symbol
#
# CMake consumes this list for both compilation and generated linker anchors,
# matching the pipeline plugin manifest pattern.

set(TRTF_CONFIG_SCHEMAS
  "triattention.cpp|kForceLink_triattention"
  "decode_policy.cpp|kForceLink_decode_policy"
  "text_trace.cpp|kForceLink_text_trace"
  "runtime.cpp|kForceLink_runtime"
  "audio_bark.cpp|kForceLink_audio_bark"
  "audio_magpie.cpp|kForceLink_audio_magpie"
  "platform.cpp|kForceLink_platform"
)

set(TRTF_CONFIG_SCHEMA_SOURCES)
set(TRTF_CONFIG_SCHEMA_FORCE_LINK_DECLS)
set(TRTF_CONFIG_SCHEMA_FORCE_LINK_ANCHORS)

foreach(_trtf_schema IN LISTS TRTF_CONFIG_SCHEMAS)
  string(REPLACE "|" ";" _trtf_schema_fields "${_trtf_schema}")
  list(LENGTH _trtf_schema_fields _trtf_schema_field_count)
  if(NOT _trtf_schema_field_count EQUAL 2)
    message(FATAL_ERROR "Invalid TRTF config schema entry: ${_trtf_schema}")
  endif()

  list(GET _trtf_schema_fields 0 _trtf_schema_source)
  list(GET _trtf_schema_fields 1 _trtf_schema_symbol)

  set(_trtf_schema_path "${PROJECT_SOURCE_DIR}/src/runtime/config/schemas/${_trtf_schema_source}")
  if(NOT EXISTS "${_trtf_schema_path}")
    message(FATAL_ERROR "TRTF config schema source does not exist: ${_trtf_schema_path}")
  endif()

  list(APPEND TRTF_CONFIG_SCHEMA_SOURCES "${_trtf_schema_path}")
  string(APPEND TRTF_CONFIG_SCHEMA_FORCE_LINK_DECLS
    "extern volatile int ${_trtf_schema_symbol};\n")
  string(APPEND TRTF_CONFIG_SCHEMA_FORCE_LINK_ANCHORS
    "    &${_trtf_schema_symbol},\n")
endforeach()

set(TRTF_CONFIG_SCHEMA_FORCE_LINK_SOURCE
  "${PROJECT_BINARY_DIR}/generated/force_link_schemas.cpp")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/generated")
configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/force_link_schemas.cpp.in"
  "${TRTF_CONFIG_SCHEMA_FORCE_LINK_SOURCE}"
  @ONLY
)
set_source_files_properties(
  "${TRTF_CONFIG_SCHEMA_FORCE_LINK_SOURCE}"
  PROPERTIES GENERATED TRUE
)
