# Shared helper for declarative source manifests that also need generated
# force-link anchor sources.

function(trtf_configure_force_link_manifest manifest_var source_dir template_path generated_source
         sources_var decls_var anchors_var anchor_indent)
  set(_trtf_sources)
  set(_trtf_decls)
  set(_trtf_anchors)

  foreach(_trtf_entry IN LISTS ${manifest_var})
    string(REPLACE "|" ";" _trtf_fields "${_trtf_entry}")
    list(LENGTH _trtf_fields _trtf_field_count)
    if(NOT _trtf_field_count EQUAL 2)
      message(FATAL_ERROR "Invalid TRTF force-link manifest entry: ${_trtf_entry}")
    endif()

    list(GET _trtf_fields 0 _trtf_source)
    list(GET _trtf_fields 1 _trtf_symbol)

    set(_trtf_source_path "${source_dir}/${_trtf_source}")
    if(NOT EXISTS "${_trtf_source_path}")
      message(FATAL_ERROR "TRTF force-link manifest source does not exist: ${_trtf_source_path}")
    endif()

    list(APPEND _trtf_sources "${_trtf_source_path}")
    string(APPEND _trtf_decls "extern volatile int ${_trtf_symbol};\n")
    string(APPEND _trtf_anchors "${anchor_indent}&${_trtf_symbol},\n")
  endforeach()

  set(${sources_var} ${_trtf_sources} PARENT_SCOPE)
  set(${decls_var} "${_trtf_decls}" PARENT_SCOPE)
  set(${anchors_var} "${_trtf_anchors}" PARENT_SCOPE)

  set(${decls_var} "${_trtf_decls}")
  set(${anchors_var} "${_trtf_anchors}")
  get_filename_component(_trtf_generated_dir "${generated_source}" DIRECTORY)
  file(MAKE_DIRECTORY "${_trtf_generated_dir}")
  configure_file("${template_path}" "${generated_source}" @ONLY)
  set_source_files_properties("${generated_source}" PROPERTIES GENERATED TRUE)
endfunction()
