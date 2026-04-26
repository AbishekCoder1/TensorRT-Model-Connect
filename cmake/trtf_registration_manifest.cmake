# Shared helper for declarative source manifests that also need generated
# registration sources.

function(trtf_configure_registration_manifest manifest_var source_dir template_path generated_source
         sources_var decls_var calls_var call_indent registry_type)
  set(_trtf_sources)
  set(_trtf_decls)
  set(_trtf_calls)

  foreach(_trtf_entry IN LISTS ${manifest_var})
    string(REPLACE "|" ";" _trtf_fields "${_trtf_entry}")
    list(LENGTH _trtf_fields _trtf_field_count)
    if(NOT _trtf_field_count EQUAL 2)
      message(FATAL_ERROR "Invalid TRTF registration manifest entry: ${_trtf_entry}")
    endif()

    list(GET _trtf_fields 0 _trtf_source)
    list(GET _trtf_fields 1 _trtf_symbol)

    set(_trtf_source_path "${source_dir}/${_trtf_source}")
    if(NOT EXISTS "${_trtf_source_path}")
      message(FATAL_ERROR "TRTF registration manifest source does not exist: ${_trtf_source_path}")
    endif()

    list(APPEND _trtf_sources "${_trtf_source_path}")
    string(APPEND _trtf_decls "void ${_trtf_symbol}(${registry_type}& registry);\n")
    string(APPEND _trtf_calls "${call_indent}${_trtf_symbol}(registry);\n")
  endforeach()

  set(${sources_var} ${_trtf_sources} PARENT_SCOPE)
  set(${decls_var} "${_trtf_decls}" PARENT_SCOPE)
  set(${calls_var} "${_trtf_calls}" PARENT_SCOPE)

  set(${decls_var} "${_trtf_decls}")
  set(${calls_var} "${_trtf_calls}")
  get_filename_component(_trtf_generated_dir "${generated_source}" DIRECTORY)
  file(MAKE_DIRECTORY "${_trtf_generated_dir}")
  configure_file("${template_path}" "${generated_source}" @ONLY)
  set_source_files_properties("${generated_source}" PROPERTIES GENERATED TRUE)
endfunction()
