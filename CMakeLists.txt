cmake_minimum_required(VERSION 3.20)
project(trt_transformers_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
  src/model/qwen3_trt_model_definition.cpp
  src/model/qwen3_decoder_model_loader.cpp
  src/runtime/cpu_reference_backend.cpp
  src/runtime/hf_python_backend.cpp
  src/runtime/runtime_factory.cpp
  src/runtime/trt_backend.cpp
  src/runtime/trt_backend_qwen.cpp
  src/runtime/trt/trt_common.cpp
  src/runtime/trt/trt_graph_ops.cpp
  src/runtime/trt/trt_engine_lifecycle.cpp
  src/runtime/trt/trt_decode_runtime.cpp
  src/runtime/trt/trt_backend_shared.cpp
  src/runtime/trt/trt_graph_builder.cpp
  src/models/qwen/registration.cpp
  src/utils/text_parsers.cpp
  src/utils/json_helpers.cpp
  src/utils/tensor_math.cpp
  src/utils/trt/engine_cache.cpp
  src/tokenizer/toy_tokenizer.cpp
  src/tokenizer/hf_python_tokenizer.cpp
)

target_include_directories(trtf_core
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
  PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)

target_compile_definitions(trtf_core
  PUBLIC
    TRTF_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\"
)

target_compile_options(trtf_core
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_compile_definitions(trtf_core PUBLIC TRTF_HAS_TRT=1)
  target_include_directories(trtf_core PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
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
