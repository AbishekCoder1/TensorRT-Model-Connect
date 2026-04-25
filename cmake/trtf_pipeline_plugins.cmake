# Declarative runtime pipeline plugin manifest.
#
# Each entry is:
#   plugin_source.cpp|force_link_symbol
#
# CMake consumes this list for both compilation and generated linker anchors,
# so a plugin is added in one place instead of editing parallel source lists.

set(TRTF_PIPELINE_PLUGINS
  "decoder_plugin.cpp|kForceLink_DecoderPlugin"
  "ssm_plugin.cpp|kForceLink_SsmPlugin"
  "rwkv_plugin.cpp|kForceLink_RwkvPlugin"
  "hybrid_plugin.cpp|kForceLink_HybridPlugin"
  "encoder_plugin.cpp|kForceLink_EncoderPlugin"
  "patchtst_plugin.cpp|kForceLink_PatchTSTPlugin"
  "patchtsmixer_plugin.cpp|kForceLink_PatchTSMixerPlugin"
  "timesfm_plugin.cpp|kForceLink_TimesFmPlugin"
  "chronos_bolt_plugin.cpp|kForceLink_ChronosBoltPlugin"
  "segmentation_plugin.cpp|kForceLink_SegmentationPlugin"
  "object_detection_plugin.cpp|kForceLink_ObjectDetectionPlugin"
  "vl_plugin.cpp|kForceLink_VLPlugin"
  "whisper_plugin.cpp|kForceLink_WhisperPlugin"
  "bark_plugin.cpp|kForceLink_BarkPlugin"
  "magpie_plugin.cpp|kForceLink_MagpiePlugin"
  "speech_plugin.cpp|kForceLink_SpeechPlugin"
  "omni_plugin.cpp|kForceLink_OmniPlugin"
  "flux_plugin.cpp|kForceLink_FluxPlugin"
  "wan_plugin.cpp|kForceLink_WanPlugin"
  "zimage_plugin.cpp|kForceLink_ZImagePlugin"
  "t5_plugin.cpp|kForceLink_T5Plugin"
  "marian_plugin.cpp|kForceLink_MarianPlugin"
  "seq2seq_plugin.cpp|kForceLink_Seq2SeqPlugin"
  "pixart_plugin.cpp|kForceLink_PixArtPlugin"
  "pixart_torchtrt_plugin.cpp|kForceLink_PixArtTorchTrtPlugin"
)

set(TRTF_PIPELINE_PLUGIN_SOURCES)
set(TRTF_PIPELINE_PLUGIN_FORCE_LINK_DECLS)
set(TRTF_PIPELINE_PLUGIN_FORCE_LINK_ANCHORS)

foreach(_trtf_plugin IN LISTS TRTF_PIPELINE_PLUGINS)
  string(REPLACE "|" ";" _trtf_plugin_fields "${_trtf_plugin}")
  list(LENGTH _trtf_plugin_fields _trtf_plugin_field_count)
  if(NOT _trtf_plugin_field_count EQUAL 2)
    message(FATAL_ERROR "Invalid TRTF pipeline plugin entry: ${_trtf_plugin}")
  endif()

  list(GET _trtf_plugin_fields 0 _trtf_plugin_source)
  list(GET _trtf_plugin_fields 1 _trtf_plugin_symbol)

  set(_trtf_plugin_path "${PROJECT_SOURCE_DIR}/src/runtime/plugins/${_trtf_plugin_source}")
  if(NOT EXISTS "${_trtf_plugin_path}")
    message(FATAL_ERROR "TRTF pipeline plugin source does not exist: ${_trtf_plugin_path}")
  endif()

  list(APPEND TRTF_PIPELINE_PLUGIN_SOURCES
    "${_trtf_plugin_path}")
  string(APPEND TRTF_PIPELINE_PLUGIN_FORCE_LINK_DECLS
    "extern volatile int ${_trtf_plugin_symbol};\n")
  string(APPEND TRTF_PIPELINE_PLUGIN_FORCE_LINK_ANCHORS
    "        &${_trtf_plugin_symbol},\n")
endforeach()

set(TRTF_PIPELINE_PLUGIN_FORCE_LINK_SOURCE
  "${PROJECT_BINARY_DIR}/generated/force_link_plugins.cpp")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/generated")
configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/force_link_plugins.cpp.in"
  "${TRTF_PIPELINE_PLUGIN_FORCE_LINK_SOURCE}"
  @ONLY
)
set_source_files_properties(
  "${TRTF_PIPELINE_PLUGIN_FORCE_LINK_SOURCE}"
  PROPERTIES GENERATED TRUE
)
