# Declarative runtime pipeline plugin manifest.
#
# Each entry is:
#   plugin_source.cpp|registration_function
#
# CMake consumes this list for both compilation and generated registrar calls,
# so a plugin is added in one place instead of editing parallel source lists.

include("${CMAKE_CURRENT_LIST_DIR}/trtf_registration_manifest.cmake")

set(TRTF_PIPELINE_PLUGINS
  "decoder_plugin.cpp|register_decoder_plugin"
  "ssm_plugin.cpp|register_ssm_plugin"
  "rwkv_plugin.cpp|register_rwkv_plugin"
  "hybrid_plugin.cpp|register_hybrid_plugin"
  "encoder_plugin.cpp|register_encoder_plugin"
  "patchtst_plugin.cpp|register_patchtst_plugin"
  "patchtsmixer_plugin.cpp|register_patchtsmixer_plugin"
  "timesfm_plugin.cpp|register_timesfm_plugin"
  "chronos_bolt_plugin.cpp|register_chronos_bolt_plugin"
  "segmentation_plugin.cpp|register_segmentation_plugin"
  "object_detection_plugin.cpp|register_object_detection_plugin"
  "vl_plugin.cpp|register_vl_plugin"
  "whisper_plugin.cpp|register_whisper_plugin"
  "bark_plugin.cpp|register_bark_plugin"
  "magpie_plugin.cpp|register_magpie_plugin"
  "speech_plugin.cpp|register_speech_plugin"
  "omni_plugin.cpp|register_omni_plugin"
  "flux_plugin.cpp|register_flux_plugin"
  "wan_plugin.cpp|register_wan_plugin"
  "zimage_plugin.cpp|register_zimage_plugin"
  "t5_plugin.cpp|register_t5_plugin"
  "marian_plugin.cpp|register_marian_plugin"
  "seq2seq_plugin.cpp|register_seq2seq_plugin"
  "pixart_plugin.cpp|register_pixart_plugin"
  "pixart_torchtrt_plugin.cpp|register_pixart_torchtrt_plugin"
)

set(TRTF_PIPELINE_PLUGIN_REGISTRATION_SOURCE
  "${PROJECT_BINARY_DIR}/generated/register_plugins.cpp")
trtf_configure_registration_manifest(
  TRTF_PIPELINE_PLUGINS
  "${PROJECT_SOURCE_DIR}/src/runtime/plugins"
  "${CMAKE_CURRENT_LIST_DIR}/register_plugins.cpp.in"
  "${TRTF_PIPELINE_PLUGIN_REGISTRATION_SOURCE}"
  TRTF_PIPELINE_PLUGIN_SOURCES
  TRTF_PIPELINE_PLUGIN_REGISTRATION_DECLS
  TRTF_PIPELINE_PLUGIN_REGISTRATION_CALLS
  "    "
  "::trtf::PipelineRegistry"
)
