cmake_minimum_required(VERSION 3.20)
project(trt_transformers_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)

option(TRTF_ENABLE_TRT "Enable TensorRT backend integration" ON)

set(_trtf_default_search_paths
  /usr/include/x86_64-linux-gnu
  /usr/include
  /usr/lib/x86_64-linux-gnu
  /lib/x86_64-linux-gnu
  /usr/local/cuda
  /usr/local/cuda/include
  /usr/local/cuda/lib64
)

if(DEFINED ENV{TRT_ROOT})
  list(APPEND _trtf_default_search_paths "$ENV{TRT_ROOT}")
endif()

if(DEFINED ENV{CUDA_HOME})
  list(APPEND _trtf_default_search_paths "$ENV{CUDA_HOME}")
  list(APPEND _trtf_default_search_paths "$ENV{CUDA_HOME}/include")
  list(APPEND _trtf_default_search_paths "$ENV{CUDA_HOME}/lib64")
endif()

if(TRTF_ENABLE_TRT)
  find_path(TRTF_TRT_INCLUDE_DIR NvInferRuntime.h
    PATHS ${_trtf_default_search_paths}
    PATH_SUFFIXES "" include include/zapped_headers
  )

  find_library(TRTF_TRT_LIBRARY
    NAMES nvinfer libnvinfer.so.10 nvinfer.so.10
    PATHS ${_trtf_default_search_paths}
    PATH_SUFFIXES "" lib lib64
  )

  find_path(TRTF_CUDA_INCLUDE_DIR cuda_runtime_api.h
    PATHS ${_trtf_default_search_paths}
    PATH_SUFFIXES "" include
  )

  find_library(TRTF_CUDART_LIBRARY
    NAMES cudart libcudart.so
    PATHS ${_trtf_default_search_paths}
    PATH_SUFFIXES "" lib lib64
  )
endif()

add_library(trtf_core
  src/pipeline/pipeline.cpp
  src/model/model_loader.cpp
  src/model/safetensors_loader.cpp
  src/model/hf_family_registry.cpp
  src/model/model_resolver.cpp
  src/model/trt_model_definition.cpp
  src/model/trt_model_definition_populator.cpp
  src/model/standard_trt_model_definition_populator.cpp
  src/runtime/cpu_reference_backend.cpp
  src/runtime/hf_python_backend.cpp
  src/runtime/runtime_factory.cpp
  src/runtime/trt_backend.cpp
  src/runtime/trt/trt_common.cpp
  src/runtime/trt/trt_graph_ops.cpp
  src/runtime/trt/trt_engine_lifecycle.cpp
  src/runtime/trt/trt_decode_runtime.cpp
  src/runtime/trt/trt_backend_shared.cpp
  src/runtime/trt/kv_cache_step_state.cpp
  src/runtime/trt/trt_graph_builder.cpp
  src/runtime/trt/standard_decoder_graph_builder.cpp
  src/model/checkpoint_mapper.cpp
  src/model/standard_checkpoint_mapper.cpp
  src/models/qwen/registration.cpp
  src/models/qwen/checkpoint_mapper.cpp
  src/models/qwen/trt_model_populator.cpp
  src/models/llama/registration.cpp
  src/models/llama/checkpoint_mapper.cpp
  src/models/yi/registration.cpp
  src/models/yi/checkpoint_mapper.cpp
  src/models/mistral/registration.cpp
  src/models/mistral/checkpoint_mapper.cpp
  src/models/gemma/registration.cpp
  src/models/gemma/checkpoint_mapper.cpp
  src/models/internlm/registration.cpp
  src/models/internlm/checkpoint_mapper.cpp
  src/models/deepseek/registration.cpp
  src/models/deepseek/checkpoint_mapper.cpp
  src/models/baichuan/registration.cpp
  src/models/baichuan/checkpoint_mapper.cpp
  src/bundle/bundle_format.cpp
  src/bundle/bundle_api.cpp
  src/cabi/trtf_c.cpp
  src/utils/data_dir.cpp
  src/utils/text_parsers.cpp
  src/utils/json_helpers.cpp
  src/utils/tensor_math.cpp
  src/utils/trt/engine_cache.cpp
  src/tokenizer/toy_tokenizer.cpp
  src/tokenizer/hf_python_tokenizer.cpp
)

target_include_directories(trtf_core
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)

target_compile_definitions(trtf_core
  PUBLIC
    TRTF_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\"
    TRTF_VERSION_STRING=\"${PROJECT_VERSION}\"
)

target_compile_options(trtf_core
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_compile_definitions(trtf_core PUBLIC TRTF_HAS_TRT=1)
  target_include_directories(trtf_core SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(trtf_core PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
  message(STATUS "TensorRT backend enabled")
  message(STATUS "  TRT include: ${TRTF_TRT_INCLUDE_DIR}")
  message(STATUS "  TRT library: ${TRTF_TRT_LIBRARY}")
  message(STATUS "  CUDA include: ${TRTF_CUDA_INCLUDE_DIR}")
  message(STATUS "  CUDART lib : ${TRTF_CUDART_LIBRARY}")
else()
  target_compile_definitions(trtf_core PUBLIC TRTF_HAS_TRT=0)
  message(STATUS "TensorRT backend disabled (missing TensorRT or CUDA dev dependencies)")
endif()

add_executable(trtf_text_generation examples/text_generation.cpp)
target_link_libraries(trtf_text_generation PRIVATE trtf_core)

add_executable(trtf_load_model examples/load_model.cpp)
target_link_libraries(trtf_load_model PRIVATE trtf_core)

add_executable(trtf examples/trtf_cli.cpp)
target_link_libraries(trtf PRIVATE trtf_core)

enable_testing()

add_executable(test_tokenizer tests/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE trtf_core)
add_test(NAME test_tokenizer COMMAND test_tokenizer)

add_executable(test_pipeline tests/test_pipeline.cpp)
target_link_libraries(test_pipeline PRIVATE trtf_core)
add_test(NAME test_pipeline COMMAND test_pipeline)

add_executable(test_trt_smoke tests/test_trt_smoke.cpp)
target_link_libraries(test_trt_smoke PRIVATE trtf_core)
add_test(NAME test_trt_smoke COMMAND test_trt_smoke)

add_executable(test_model_loader tests/test_model_loader.cpp)
target_link_libraries(test_model_loader PRIVATE trtf_core)
add_test(NAME test_model_loader COMMAND test_model_loader)

add_executable(test_model_resolver tests/test_model_resolver.cpp)
target_link_libraries(test_model_resolver PRIVATE trtf_core)
add_test(NAME test_model_resolver COMMAND test_model_resolver)

add_executable(test_runtime_factory tests/test_runtime_factory.cpp)
target_link_libraries(test_runtime_factory PRIVATE trtf_core)
add_test(NAME test_runtime_factory COMMAND test_runtime_factory)

add_executable(test_extension_registry tests/test_extension_registry.cpp)
target_link_libraries(test_extension_registry PRIVATE trtf_core)
add_test(NAME test_extension_registry COMMAND test_extension_registry)

add_executable(test_hf_family_registry tests/test_hf_family_registry.cpp)
target_link_libraries(test_hf_family_registry PRIVATE trtf_core)
add_test(NAME test_hf_family_registry COMMAND test_hf_family_registry)

add_executable(test_qwen_family tests/test_qwen_family.cpp)
target_link_libraries(test_qwen_family PRIVATE trtf_core)
add_test(NAME test_qwen_family COMMAND test_qwen_family)

add_executable(test_llama_family tests/test_llama_family.cpp)
target_link_libraries(test_llama_family PRIVATE trtf_core)
add_test(NAME test_llama_family COMMAND test_llama_family)

add_executable(test_yi_family tests/test_yi_family.cpp)
target_link_libraries(test_yi_family PRIVATE trtf_core)
add_test(NAME test_yi_family COMMAND test_yi_family)

add_executable(test_mistral_family tests/test_mistral_family.cpp)
target_link_libraries(test_mistral_family PRIVATE trtf_core)
add_test(NAME test_mistral_family COMMAND test_mistral_family)

add_executable(test_gemma_family tests/test_gemma_family.cpp)
target_link_libraries(test_gemma_family PRIVATE trtf_core)
add_test(NAME test_gemma_family COMMAND test_gemma_family)

add_executable(test_internlm_family tests/test_internlm_family.cpp)
target_link_libraries(test_internlm_family PRIVATE trtf_core)
add_test(NAME test_internlm_family COMMAND test_internlm_family)

add_executable(test_deepseek_family tests/test_deepseek_family.cpp)
target_link_libraries(test_deepseek_family PRIVATE trtf_core)
add_test(NAME test_deepseek_family COMMAND test_deepseek_family)

add_executable(test_baichuan_family tests/test_baichuan_family.cpp)
target_link_libraries(test_baichuan_family PRIVATE trtf_core)
add_test(NAME test_baichuan_family COMMAND test_baichuan_family)

add_executable(test_tensor_math tests/test_tensor_math.cpp)
target_link_libraries(test_tensor_math PRIVATE trtf_core)
target_include_directories(test_tensor_math PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_tensor_math COMMAND test_tensor_math)

add_executable(test_json_helpers tests/test_json_helpers.cpp)
target_link_libraries(test_json_helpers PRIVATE trtf_core)
target_include_directories(test_json_helpers PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_json_helpers COMMAND test_json_helpers)

add_executable(test_text_parsers tests/test_text_parsers.cpp)
target_link_libraries(test_text_parsers PRIVATE trtf_core)
target_include_directories(test_text_parsers PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_text_parsers COMMAND test_text_parsers)

add_executable(test_decode_runtime tests/test_decode_runtime.cpp)
target_link_libraries(test_decode_runtime PRIVATE trtf_core)
target_include_directories(test_decode_runtime PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_decode_runtime SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_decode_runtime PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_decode_runtime COMMAND test_decode_runtime)

add_executable(test_engine_cache_key tests/test_engine_cache_key.cpp)
target_link_libraries(test_engine_cache_key PRIVATE trtf_core)
target_include_directories(test_engine_cache_key PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_engine_cache_key COMMAND test_engine_cache_key)

add_executable(test_kv_cache_step_state tests/test_kv_cache_step_state.cpp)
target_link_libraries(test_kv_cache_step_state PRIVATE trtf_core)
target_include_directories(test_kv_cache_step_state PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_kv_cache_step_state SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_kv_cache_step_state PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_kv_cache_step_state COMMAND test_kv_cache_step_state)

add_executable(test_extra_fields tests/test_extra_fields.cpp)
target_link_libraries(test_extra_fields PRIVATE trtf_core)
target_include_directories(test_extra_fields PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_extra_fields SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_extra_fields PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_extra_fields COMMAND test_extra_fields)

add_executable(test_data_dir tests/test_data_dir.cpp)
target_link_libraries(test_data_dir PRIVATE trtf_core)
target_include_directories(test_data_dir PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_data_dir COMMAND test_data_dir)

add_executable(test_bundle_format tests/test_bundle_format.cpp)
target_link_libraries(test_bundle_format PRIVATE trtf_core)
target_include_directories(test_bundle_format PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_bundle_format COMMAND test_bundle_format)

add_executable(test_c_abi_entry tests/test_c_abi_entry.cpp)
target_link_libraries(test_c_abi_entry PRIVATE trtf_core)
add_test(NAME test_c_abi_entry COMMAND test_c_abi_entry)

add_executable(test_pipeline_api tests/test_pipeline_api.cpp)
target_link_libraries(test_pipeline_api PRIVATE trtf_core)
add_test(NAME test_pipeline_api COMMAND test_pipeline_api)

add_executable(test_bundle_e2e tests/test_bundle_e2e.cpp)
target_link_libraries(test_bundle_e2e PRIVATE trtf_core)
add_test(NAME test_bundle_e2e COMMAND test_bundle_e2e)

add_executable(test_cli_args tests/test_cli_args.cpp)
target_link_libraries(test_cli_args PRIVATE trtf_core)
add_test(NAME test_cli_args COMMAND test_cli_args)

add_executable(test_trt_ops_gold tests/test_trt_graph_ops_gold.cpp)
target_link_libraries(test_trt_ops_gold PRIVATE trtf_core)
target_include_directories(test_trt_ops_gold PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_trt_ops_gold SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_trt_ops_gold PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_trt_ops_gold COMMAND test_trt_ops_gold)

# --- Install ---
include(CMakePackageConfigHelpers)

# Install library
install(TARGETS trtf_core
  EXPORT trtfTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install public headers
install(DIRECTORY include/trtf/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/trtf
  FILES_MATCHING PATTERN "*.h"
)

# Install CLI executable
install(TARGETS trtf
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Generate and install CMake config files
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/trtfConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/trtfConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/trtf
)

write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/trtfConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(EXPORT trtfTargets
  FILE trtfTargets.cmake
  NAMESPACE trtf::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/trtf
)

install(FILES
  "${PROJECT_BINARY_DIR}/trtfConfig.cmake"
  "${PROJECT_BINARY_DIR}/trtfConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/trtf
)
