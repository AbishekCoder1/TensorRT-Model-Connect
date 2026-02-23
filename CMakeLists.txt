cmake_minimum_required(VERSION 3.20)
project(trt_transformers_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)

option(TRTF_ENABLE_TRT "Enable TensorRT backend integration" ON)

set(_trtf_default_search_paths
  /usr/include/x86_64-linux-gnu
  /usr/include/aarch64-linux-gnu
  /usr/include
  /usr/lib/x86_64-linux-gnu
  /usr/lib/aarch64-linux-gnu
  /lib/x86_64-linux-gnu
  /lib/aarch64-linux-gnu
  /usr/local/cuda
  /usr/local/cuda/include
  /usr/local/cuda/lib64
  /usr/local/cuda/targets/sbsa-linux/include
  /usr/local/cuda/targets/sbsa-linux/lib
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

# --- Bundle-only runtime library ---
add_library(trtf_core
  src/bundle/bundle_format.cpp
  src/cabi/trtf_c.cpp
  src/cabi/fast_path_config.cpp
  src/cabi/bundle_helpers.cpp
  src/runtime/trt/trt_common.cpp
  src/runtime/trt/trt_engine_lifecycle.cpp
  src/runtime/trt/trt_decode_runtime.cpp
  src/runtime/trt/trt_backend_shared.cpp
  src/runtime/trt/device_kv_cache.cpp
  src/runtime/trt/mamba_step_state.cpp
  src/runtime/trt/mamba_decode_runtime.cpp
  src/runtime/trt/mamba_backend.cpp
  src/runtime/trt/rwkv_step_state.cpp
  src/runtime/trt/rwkv_decode_runtime.cpp
  src/runtime/trt/rwkv_backend.cpp
  src/runtime/trt/whisper_backend.cpp
  src/runtime/trt/vl_backend.cpp
  src/runtime/trt/image_preprocessor.cpp
  src/runtime/trt/vision_engine.cpp
  src/runtime/trt/segmentation_backend.cpp
  src/runtime/trt/bark_backend.cpp
  src/runtime/trt/diffusion_backend.cpp
  src/runtime/trt/diffusion_backend_base.cpp
  src/runtime/trt/wan_diffusion_backend.cpp
  src/runtime/trt/stb_impl.cpp
  src/tokenizer/hf_python_tokenizer.cpp
  src/tokenizer/vocab_tokenizer.cpp
  src/utils/data_dir.cpp
  src/utils/text_parsers.cpp
  src/utils/json_helpers.cpp
)

target_include_directories(trtf_core
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party/stb
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

add_executable(trtf examples/trtf_cli.cpp)
target_link_libraries(trtf PRIVATE trtf_core)

enable_testing()

# --- Tests ---

add_executable(test_bundle_format tests/cpp/test_bundle_format.cpp)
target_link_libraries(test_bundle_format PRIVATE trtf_core)
target_include_directories(test_bundle_format PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_bundle_format COMMAND test_bundle_format)

add_executable(test_c_abi_entry tests/cpp/test_c_abi_entry.cpp)
target_link_libraries(test_c_abi_entry PRIVATE trtf_core)
add_test(NAME test_c_abi_entry COMMAND test_c_abi_entry)

add_executable(test_fast_path_config tests/cpp/test_fast_path_config.cpp)
target_link_libraries(test_fast_path_config PRIVATE trtf_core)
target_include_directories(test_fast_path_config PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_fast_path_config COMMAND test_fast_path_config)

add_executable(test_text_parsers tests/cpp/test_text_parsers.cpp)
target_link_libraries(test_text_parsers PRIVATE trtf_core)
target_include_directories(test_text_parsers PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_text_parsers COMMAND test_text_parsers)

add_executable(test_json_helpers tests/cpp/test_json_helpers.cpp)
target_link_libraries(test_json_helpers PRIVATE trtf_core)
target_include_directories(test_json_helpers PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_json_helpers COMMAND test_json_helpers)

add_executable(test_data_dir tests/cpp/test_data_dir.cpp)
target_link_libraries(test_data_dir PRIVATE trtf_core)
target_include_directories(test_data_dir PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_data_dir COMMAND test_data_dir)

add_executable(test_decode_runtime tests/cpp/test_decode_runtime.cpp)
target_link_libraries(test_decode_runtime PRIVATE trtf_core)
target_include_directories(test_decode_runtime PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_decode_runtime SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_decode_runtime PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_decode_runtime COMMAND test_decode_runtime)

add_executable(test_pipeline_api tests/cpp/test_pipeline_api.cpp)
target_link_libraries(test_pipeline_api PRIVATE trtf_core)
add_test(NAME test_pipeline_api COMMAND test_pipeline_api)

add_executable(test_bundle_e2e tests/cpp/test_bundle_e2e.cpp)
target_link_libraries(test_bundle_e2e PRIVATE trtf_core)
add_test(NAME test_bundle_e2e COMMAND test_bundle_e2e)

add_executable(test_cli_args tests/cpp/test_cli_args.cpp)
target_link_libraries(test_cli_args PRIVATE trtf_core)
add_test(NAME test_cli_args COMMAND test_cli_args)

add_executable(test_image_preprocessor tests/cpp/test_image_preprocessor.cpp)
target_link_libraries(test_image_preprocessor PRIVATE trtf_core)
target_include_directories(test_image_preprocessor PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_image_preprocessor COMMAND test_image_preprocessor)

add_executable(test_cuda_buffer tests/cpp/test_cuda_buffer.cpp)
target_link_libraries(test_cuda_buffer PRIVATE trtf_core)
target_include_directories(test_cuda_buffer PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_cuda_buffer SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_cuda_buffer PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_cuda_buffer COMMAND test_cuda_buffer)

add_executable(test_cuda_stream tests/cpp/test_cuda_stream.cpp)
target_link_libraries(test_cuda_stream PRIVATE trtf_core)
target_include_directories(test_cuda_stream PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_cuda_stream SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_cuda_stream PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_cuda_stream COMMAND test_cuda_stream)

add_executable(test_device_kv_cache tests/cpp/test_device_kv_cache.cpp)
target_link_libraries(test_device_kv_cache PRIVATE trtf_core)
target_include_directories(test_device_kv_cache PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_device_kv_cache SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_device_kv_cache PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_device_kv_cache COMMAND test_device_kv_cache)

add_executable(test_trt_logger tests/cpp/test_trt_logger.cpp)
target_link_libraries(test_trt_logger PRIVATE trtf_core)
target_include_directories(test_trt_logger PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_trt_logger SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_trt_logger PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_trt_logger COMMAND test_trt_logger)

add_executable(test_trt_engine_lifecycle tests/cpp/test_trt_engine_lifecycle.cpp)
target_link_libraries(test_trt_engine_lifecycle PRIVATE trtf_core)
target_include_directories(test_trt_engine_lifecycle PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_trt_engine_lifecycle SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_trt_engine_lifecycle PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_trt_engine_lifecycle COMMAND test_trt_engine_lifecycle)

add_executable(test_vocab_tokenizer tests/cpp/test_vocab_tokenizer.cpp)
target_link_libraries(test_vocab_tokenizer PRIVATE trtf_core)
add_test(NAME test_vocab_tokenizer COMMAND test_vocab_tokenizer)

add_executable(test_hf_python_tokenizer tests/cpp/test_hf_python_tokenizer.cpp)
target_link_libraries(test_hf_python_tokenizer PRIVATE trtf_core)
target_include_directories(test_hf_python_tokenizer PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_hf_python_tokenizer COMMAND test_hf_python_tokenizer)

add_executable(test_bundle_helpers tests/cpp/test_bundle_helpers.cpp)
target_link_libraries(test_bundle_helpers PRIVATE trtf_core)
target_include_directories(test_bundle_helpers PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_bundle_helpers SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_bundle_helpers PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_bundle_helpers COMMAND test_bundle_helpers)

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
