cmake_minimum_required(VERSION 3.20)
project(trt_transformers_cpp VERSION 0.1.0 LANGUAGES CXX)

# Enable CUDA for GPU kernels (magpie_kernels.cu)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)

option(TRTF_ENABLE_TRT "Enable TensorRT backend DSO integration" ON)
option(TRTF_BUILD_BACKEND_TRT "Build standard TensorRT backend DSO" ON)
set(TRTF_TRT_BACKEND_ABI "" CACHE STRING
  "TensorRT ABI suffix for the standard backend DSO, e.g. 10_15. Empty means auto-detect from headers.")

# CUDA is required by the core runtime. TensorRT is optional at core build time:
# concrete TRT runtimes live in backend DSOs.
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

find_path(TRTF_TRT_INCLUDE_DIR NvInferRuntime.h
  PATHS ${_trtf_default_search_paths}
  PATH_SUFFIXES "" include include/zapped_headers
)

find_library(TRTF_TRT_LIBRARY
  NAMES nvinfer libnvinfer.so.11 nvinfer.so.11 libnvinfer.so.10 nvinfer.so.10
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

function(_trtf_header_define_value OUT_VAR DEFINE_NAME)
  set(${OUT_VAR} "" PARENT_SCOPE)
  if(NOT TRTF_TRT_INCLUDE_DIR OR NOT EXISTS "${TRTF_TRT_INCLUDE_DIR}/NvInferVersion.h")
    return()
  endif()
  file(READ "${TRTF_TRT_INCLUDE_DIR}/NvInferVersion.h" _trtf_version_header)
  string(REGEX MATCH "#define[ \t]+${DEFINE_NAME}[ \t]+([A-Za-z0-9_]+)"
    _trtf_define_match "${_trtf_version_header}")
  if(NOT _trtf_define_match)
    return()
  endif()
  set(_trtf_value "${CMAKE_MATCH_1}")
  if(NOT _trtf_value MATCHES "^[0-9]+$")
    string(REGEX MATCH "#define[ \t]+${_trtf_value}[ \t]+([0-9]+)"
      _trtf_alias_match "${_trtf_version_header}")
    if(_trtf_alias_match)
      set(_trtf_value "${CMAKE_MATCH_1}")
    endif()
  endif()
  if(_trtf_value MATCHES "^[0-9]+$")
    set(${OUT_VAR} "${_trtf_value}" PARENT_SCOPE)
  endif()
endfunction()

if(NOT TRTF_TRT_BACKEND_ABI AND TRTF_TRT_INCLUDE_DIR)
  _trtf_header_define_value(_trtf_header_trt_major NV_TENSORRT_MAJOR)
  _trtf_header_define_value(_trtf_header_trt_minor NV_TENSORRT_MINOR)
  if(_trtf_header_trt_major AND _trtf_header_trt_minor)
    set(TRTF_TRT_BACKEND_ABI "${_trtf_header_trt_major}_${_trtf_header_trt_minor}"
      CACHE STRING "TensorRT ABI suffix for the standard backend DSO" FORCE)
  endif()
endif()

if(NOT TRTF_CUDA_INCLUDE_DIR OR NOT TRTF_CUDART_LIBRARY)
  message(FATAL_ERROR
    "CUDA dev dependencies are required but not found.\n"
    "  TRTF_CUDA_INCLUDE_DIR=${TRTF_CUDA_INCLUDE_DIR}\n"
    "  TRTF_CUDART_LIBRARY=${TRTF_CUDART_LIBRARY}")
endif()

set(_trtf_can_build_trt_backend FALSE)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY)
  set(_trtf_can_build_trt_backend TRUE)
endif()

# --- nlohmann/json for BPE tokenizer JSON parsing ---
include(FetchContent)
set(JSON_Install OFF CACHE INTERNAL "")
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# --- Bundle-only runtime library ---
# PIC needed because trtf_core is linked into backend DSOs (shared libs).
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(cmake/trtf_config_schemas.cmake)
include(cmake/trtf_pipeline_plugins.cmake)
add_library(trtf_core
  src/bundle/bundle_format.cpp
  src/bundle/bundle_view.cpp
  src/cabi/api/trtf_c.cpp
  # Core runtime infrastructure (TRT engine, CUDA, KV cache, decode)
  src/runtime/core/trt_common.cpp
  src/runtime/core/gpu_matmul.cpp
  src/runtime/core/cuda_common.cpp
  src/runtime/backend/backend_loader.cpp
  src/runtime/backend/trt_version.cpp
  src/runtime/core/trt_engine_lifecycle.cpp
  src/runtime/core/trt_decode_runtime.cpp
  src/runtime/core/sampler.cpp
  src/runtime/core/device_kv_cache.cpp
  src/runtime/core/device_tensor.cpp
  src/runtime/core/kv_cache.cpp
  src/runtime/core/triattention_kv_cache.cpp
  src/runtime/core/recurrent_state.cpp
  src/runtime/core/chat_template.cpp
  src/runtime/core/flow_match_euler_scheduler.cpp
  src/runtime/core/stb_impl.cpp
  # Domain-specific components
  src/runtime/domains/multimodal/image_preprocessor.cpp
  src/runtime/domains/multimodal/vision_engine.cpp
  src/runtime/domains/audio/audio_types.cpp
  src/runtime/domains/audio/mel_spectrogram.cpp
  src/runtime/domains/audio/audio_bundle_validation.cpp
  src/runtime/domains/diffusion/diffusion_preprocessor.cpp
  src/tokenizer/vocab_tokenizer.cpp
  src/tokenizer/ipa_tokenizer.cpp
  src/tokenizer/bpe_tokenizer.cpp
  src/tokenizer/wordpiece_tokenizer.cpp
  src/tokenizer/unigram_tokenizer.cpp
  # Utilities
  src/utils/data_dir.cpp
  src/utils/text_parsers.cpp
  src/utils/json_helpers.cpp
  src/utils/wav_reader.cpp
  src/utils/image_reader.cpp
  src/runtime/core/hybrid_state.cpp
  # Config (schema registry + layered resolution + CLI surface + schemas)
  src/runtime/config/schema_registry.cpp
  src/runtime/config/config_bundle.cpp
  src/runtime/config/cli_support.cpp
  ${TRTF_CONFIG_SCHEMA_REGISTRATION_SOURCE}
  ${TRTF_CONFIG_SCHEMA_SOURCES}
  # Registry (factory + plugin dispatch)
  src/runtime/registry/pipeline_factory.cpp
  src/runtime/registry/pipeline_registry.cpp
  src/runtime/registry/pipeline_plugin.cpp
  # Plugin shared helpers
  src/runtime/plugins/shared/plugin_helpers.cpp
  src/runtime/plugins/shared/diffusion_helpers.cpp
  src/runtime/plugins/shared/audio_helpers.cpp
  ${TRTF_PIPELINE_PLUGIN_REGISTRATION_SOURCE}
  ${TRTF_PIPELINE_PLUGIN_SOURCES}
  # Pipeline implementations
  src/runtime/pipelines/text_generation_pipeline.cpp
  src/runtime/pipelines/recurrent_pipeline.cpp
  src/runtime/pipelines/encoder_pipeline.cpp
  src/runtime/pipelines/patchtst_pipeline.cpp
  src/runtime/pipelines/patchtsmixer_pipeline.cpp
  src/runtime/pipelines/timesfm_pipeline.cpp
  src/runtime/pipelines/chronos_bolt_pipeline.cpp
  src/runtime/pipelines/segment_pipeline.cpp
  src/runtime/pipelines/sam_pipeline.cpp
  src/runtime/pipelines/vl_pipeline.cpp
  src/runtime/pipelines/flux_pipeline.cpp
  src/runtime/pipelines/wan_pipeline.cpp
  src/runtime/pipelines/z_image_pipeline.cpp
  src/runtime/pipelines/whisper_pipeline.cpp
  src/runtime/pipelines/rnnt_pipeline.cpp
  src/runtime/pipelines/bark_pipeline.cpp
  src/runtime/pipelines/magpie_pipeline.cpp
  src/runtime/pipelines/speech_pipeline.cpp
  src/runtime/pipelines/omni_pipeline.cpp
  src/runtime/pipelines/pixart_pipeline.cpp
  src/runtime/pipelines/pixart_torchtrt_pipeline.cpp
)

# Add CUDA kernel sources when CUDA compiler is available
if(CMAKE_CUDA_COMPILER)
  target_sources(trtf_core PRIVATE
    src/runtime/domains/audio/magpie_kernels.cu
    src/runtime/core/argmax_kernel.cu
    src/runtime/core/sparse_multinomial_kernel.cu
    src/runtime/core/triattention_kernels.cu
  )
  set_source_files_properties(
    src/runtime/domains/audio/magpie_kernels.cu
    src/runtime/core/argmax_kernel.cu
    src/runtime/core/sparse_multinomial_kernel.cu
    src/runtime/core/triattention_kernels.cu
    PROPERTIES LANGUAGE CUDA
  )
  target_include_directories(trtf_core PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_compile_definitions(trtf_core PUBLIC TRTF_HAS_CUDA_KERNELS=1)
endif()

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

# nlohmann/json: use include-only (header-only lib) to avoid export-set issues
target_include_directories(trtf_core SYSTEM PRIVATE
  ${nlohmann_json_SOURCE_DIR}/include
)

target_link_libraries(trtf_core PRIVATE ${CMAKE_DL_LIBS})

target_compile_definitions(trtf_core PRIVATE TRTF_HAS_TRT=0)
target_include_directories(trtf_core SYSTEM PRIVATE ${TRTF_CUDA_INCLUDE_DIR})
find_library(TRTF_CUBLAS_LIBRARY NAMES cublas libcublas.so
  PATHS ${_trtf_default_search_paths} PATH_SUFFIXES "" lib lib64)
target_link_libraries(trtf_core PRIVATE ${TRTF_CUDART_LIBRARY}
  $<$<BOOL:${TRTF_CUBLAS_LIBRARY}>:${TRTF_CUBLAS_LIBRARY}>)
message(STATUS "Core runtime: TensorRT-free (backend DSOs are dlopen-loaded)")
message(STATUS "  CUDA include: ${TRTF_CUDA_INCLUDE_DIR}")
message(STATUS "  CUDART lib : ${TRTF_CUDART_LIBRARY}")
if(_trtf_can_build_trt_backend)
  message(STATUS "  Standard TRT backend SDK detected: ${TRTF_TRT_LIBRARY}")
else()
  message(STATUS "  Standard TRT backend SDK not detected; libtrtf_backend_trt.so will be skipped")
endif()

# --- libtorch multinomial bridge (optional) ---
set(_trtf_torch_roots
  /opt/venv/lib/python3.12/site-packages/torch
  ${PROJECT_SOURCE_DIR}/.venv/lib/python3.12/site-packages/torch
)
if(DEFINED ENV{TRTF_TORCH_ROOT})
  list(INSERT _trtf_torch_roots 0 "$ENV{TRTF_TORCH_ROOT}")
endif()

find_path(TRTF_TORCH_INCLUDE_DIR ATen/ATen.h
  PATHS ${_trtf_torch_roots}
  PATH_SUFFIXES include
)
find_path(TRTF_TORCH_API_INCLUDE_DIR torch/torch.h
  PATHS ${_trtf_torch_roots}
  PATH_SUFFIXES include/torch/csrc/api/include
)
find_library(TRTF_TORCH_LIBRARY
  NAMES torch libtorch.so
  PATHS ${_trtf_torch_roots}
  PATH_SUFFIXES lib
)
find_library(TRTF_TORCH_CPU_LIBRARY
  NAMES torch_cpu libtorch_cpu.so
  PATHS ${_trtf_torch_roots}
  PATH_SUFFIXES lib
)
find_library(TRTF_TORCH_CUDA_LIBRARY
  NAMES torch_cuda libtorch_cuda.so
  PATHS ${_trtf_torch_roots}
  PATH_SUFFIXES lib
)
find_library(TRTF_C10_LIBRARY
  NAMES c10 libc10.so
  PATHS ${_trtf_torch_roots}
  PATH_SUFFIXES lib
)
find_library(TRTF_C10_CUDA_LIBRARY
  NAMES c10_cuda libc10_cuda.so
  PATHS ${_trtf_torch_roots}
  PATH_SUFFIXES lib
)

if(TRTF_TORCH_INCLUDE_DIR AND TRTF_TORCH_API_INCLUDE_DIR
   AND TRTF_TORCH_LIBRARY AND TRTF_TORCH_CPU_LIBRARY
   AND TRTF_TORCH_CUDA_LIBRARY AND TRTF_C10_LIBRARY AND TRTF_C10_CUDA_LIBRARY)
  get_filename_component(TRTF_TORCH_LIBRARY_DIR ${TRTF_TORCH_LIBRARY} DIRECTORY)
  target_compile_definitions(trtf_core PUBLIC TRTF_HAS_LIBTORCH_MULTINOMIAL=1)
  target_include_directories(trtf_core SYSTEM PRIVATE
    ${TRTF_TORCH_INCLUDE_DIR}
    ${TRTF_TORCH_API_INCLUDE_DIR}
  )
  target_link_libraries(trtf_core PRIVATE
    ${TRTF_TORCH_LIBRARY}
    ${TRTF_TORCH_CPU_LIBRARY}
    ${TRTF_TORCH_CUDA_LIBRARY}
    ${TRTF_C10_LIBRARY}
    ${TRTF_C10_CUDA_LIBRARY}
  )
  set_property(TARGET trtf_core APPEND PROPERTY BUILD_RPATH ${TRTF_TORCH_LIBRARY_DIR})
  message(STATUS "libtorch multinomial bridge enabled")
  message(STATUS "  Torch include: ${TRTF_TORCH_INCLUDE_DIR}")
  message(STATUS "  Torch API    : ${TRTF_TORCH_API_INCLUDE_DIR}")
  message(STATUS "  Torch lib dir: ${TRTF_TORCH_LIBRARY_DIR}")
else()
  target_compile_definitions(trtf_core PUBLIC TRTF_HAS_LIBTORCH_MULTINOMIAL=0)
  message(STATUS "libtorch multinomial bridge disabled")
endif()

# --- TVM-FFI kernel bridge plugin (optional) ---
option(TRTF_ENABLE_TVM_FFI "Enable TVM-FFI kernel bridge plugin" ON)

set(_trtf_has_tvm_ffi FALSE)

if(TRTF_ENABLE_TVM_FFI)
  find_path(TRTF_TVM_FFI_INCLUDE_DIR tvm/ffi/c_api.h
    PATHS ${_trtf_default_search_paths}
    PATH_SUFFIXES "" include
  )
  find_library(TRTF_TVM_FFI_LIBRARY
    NAMES tvm_ffi libtvm_ffi.so
    PATHS ${_trtf_default_search_paths}
    PATH_SUFFIXES "" lib lib64
  )

  if(TRTF_TVM_FFI_INCLUDE_DIR AND TRTF_TVM_FFI_LIBRARY)
    set(_trtf_has_tvm_ffi TRUE)
    target_sources(trtf_core PRIVATE
      src/plugins/tvm_ffi_module_loader.cpp
    )
    target_compile_definitions(trtf_core PUBLIC TRTF_HAS_TVM_FFI=1)
    target_include_directories(trtf_core SYSTEM PRIVATE ${TRTF_TVM_FFI_INCLUDE_DIR})
    target_link_libraries(trtf_core PRIVATE ${TRTF_TVM_FFI_LIBRARY})
    message(STATUS "TVM-FFI module loader enabled")
    message(STATUS "  TVM-FFI include: ${TRTF_TVM_FFI_INCLUDE_DIR}")
    message(STATUS "  TVM-FFI library: ${TRTF_TVM_FFI_LIBRARY}")
  endif()
endif()

if(NOT _trtf_has_tvm_ffi)
  target_compile_definitions(trtf_core PUBLIC TRTF_HAS_TVM_FFI=0)
  message(STATUS "TVM-FFI kernel bridge disabled")
endif()

# Build the TVM-FFI TensorRT plugin as a standalone shared library.
if(_trtf_has_tvm_ffi AND _trtf_can_build_trt_backend)
  add_library(trtf_tvm_ffi_plugin SHARED
    src/plugins/tvm_ffi_kernel_plugin.cpp
    src/plugins/tvm_ffi_kernel_creator.cpp
    src/plugins/tvm_ffi_module_loader.cpp
    src/utils/json_helpers.cpp
  )
  target_include_directories(trtf_tvm_ffi_plugin PRIVATE ${PROJECT_SOURCE_DIR}/src)
  target_include_directories(trtf_tvm_ffi_plugin SYSTEM PRIVATE
    ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR} ${TRTF_TVM_FFI_INCLUDE_DIR})
  target_link_libraries(trtf_tvm_ffi_plugin PRIVATE
    ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY} ${TRTF_TVM_FFI_LIBRARY})
  target_compile_definitions(trtf_tvm_ffi_plugin PRIVATE TRTF_HAS_TRT=1 TRTF_HAS_TVM_FFI=1)
  target_compile_options(trtf_tvm_ffi_plugin PRIVATE -Wall -Wextra -Wpedantic)
  message(STATUS "TVM-FFI plugin shared library: trtf_tvm_ffi_plugin.so")
endif()

# --- Backend DSO: standard TRT ---
if(TRTF_BUILD_BACKEND_TRT AND _trtf_can_build_trt_backend)
  add_library(trtf_backend_trt SHARED
    src/runtime/backend/trt_logger.cpp
    src/runtime/backend/trt_module_impl.cpp
    src/runtime/backend/trt_backend.cpp
  )
  target_include_directories(trtf_backend_trt
    PRIVATE
      ${PROJECT_SOURCE_DIR}/include
      ${PROJECT_SOURCE_DIR}/src
  )
  target_include_directories(trtf_backend_trt SYSTEM PRIVATE
    ${TRTF_TRT_INCLUDE_DIR}
    ${TRTF_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(trtf_backend_trt PRIVATE
    trtf_core ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY}
  )
  if(_trtf_has_tvm_ffi)
    target_sources(trtf_backend_trt PRIVATE
      src/plugins/tvm_ffi_kernel_plugin.cpp
      src/plugins/tvm_ffi_kernel_creator.cpp
    )
    target_include_directories(trtf_backend_trt SYSTEM PRIVATE ${TRTF_TVM_FFI_INCLUDE_DIR})
    target_link_libraries(trtf_backend_trt PRIVATE ${TRTF_TVM_FFI_LIBRARY})
  endif()
  target_compile_options(trtf_backend_trt PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_definitions(trtf_backend_trt PRIVATE
    TRTF_HAS_TRT=1
    TRTF_HAS_TVM_FFI=$<BOOL:${_trtf_has_tvm_ffi}>
    TRTF_TRT_BACKEND_ABI_STRING=\"${TRTF_TRT_BACKEND_ABI}\"
  )
  set_target_properties(trtf_backend_trt PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  if(TRTF_TRT_BACKEND_ABI)
    add_custom_command(TARGET trtf_backend_trt POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:trtf_backend_trt>
        "${CMAKE_BINARY_DIR}/libtrtf_backend_trt_${TRTF_TRT_BACKEND_ABI}.so"
      COMMENT "Writing TensorRT ABI backend alias libtrtf_backend_trt_${TRTF_TRT_BACKEND_ABI}.so"
    )
  endif()
  # Also add nlohmann_json include for trt_module_impl (uses DType from tensor.h)
  target_include_directories(trtf_backend_trt SYSTEM PRIVATE
    ${nlohmann_json_SOURCE_DIR}/include
  )
  message(STATUS "TRT backend DSO: enabled (libtrtf_backend_trt.so)")
  if(TRTF_TRT_BACKEND_ABI)
    message(STATUS "TRT backend DSO ABI alias: libtrtf_backend_trt_${TRTF_TRT_BACKEND_ABI}.so")
  endif()
elseif(TRTF_BUILD_BACKEND_TRT)
  message(STATUS "TRT backend DSO: disabled (missing libnvinfer or headers)")
endif()

# --- Backend DSO: TRT-RTX (optional) ---
option(TRTF_BUILD_BACKEND_RTX "Build TRT-RTX backend DSO" OFF)
if(TRTF_BUILD_BACKEND_RTX)
  find_library(TENSORRT_RTX_LIB tensorrt_rtx HINTS ${TRTF_RTX_LIBRARY_DIR})
  # Use TRTF_RTX_INCLUDE_DIR directly; find_path can pick up standard TRT first.
  set(TRT_RTX_INCLUDE_DIR "${TRTF_RTX_INCLUDE_DIR}")
  if(TENSORRT_RTX_LIB AND TRT_RTX_INCLUDE_DIR)
    add_library(trtf_backend_rtx SHARED
      src/runtime/backend/trt_logger.cpp
      src/runtime/backend/trt_module_impl.cpp
      src/runtime/backend/rtx_backend.cpp
    )
    target_include_directories(trtf_backend_rtx
      PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
    )
    target_include_directories(trtf_backend_rtx SYSTEM PRIVATE
      ${TRT_RTX_INCLUDE_DIR}
      ${TRTF_CUDA_INCLUDE_DIR}
    )
    target_link_libraries(trtf_backend_rtx PRIVATE
      trtf_core ${TENSORRT_RTX_LIB} ${TRTF_CUDART_LIBRARY}
    )
    if(_trtf_has_tvm_ffi)
      target_sources(trtf_backend_rtx PRIVATE
        src/plugins/tvm_ffi_kernel_plugin.cpp
        src/plugins/tvm_ffi_kernel_creator.cpp
      )
      target_include_directories(trtf_backend_rtx SYSTEM PRIVATE ${TRTF_TVM_FFI_INCLUDE_DIR})
      target_link_libraries(trtf_backend_rtx PRIVATE ${TRTF_TVM_FFI_LIBRARY})
    endif()
    target_compile_options(trtf_backend_rtx PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_definitions(trtf_backend_rtx PRIVATE
      TRTF_HAS_TRT=1
      TRTF_HAS_TVM_FFI=$<BOOL:${_trtf_has_tvm_ffi}>
    )
    set_target_properties(trtf_backend_rtx PROPERTIES
      OUTPUT_NAME trtf_backend_trt_rtx
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    target_include_directories(trtf_backend_rtx SYSTEM PRIVATE
      ${nlohmann_json_SOURCE_DIR}/include
    )
    message(STATUS "TRT-RTX backend DSO: enabled (libtrtf_backend_trt_rtx.so)")
  else()
    message(STATUS "TRT-RTX backend DSO: disabled (missing tensorrt_rtx lib or headers)")
  endif()
endif()

add_executable(trtf examples/trtf_cli.cpp)
target_link_libraries(trtf PRIVATE trtf_core)
target_include_directories(trtf PRIVATE ${PROJECT_SOURCE_DIR}/third_party/stb ${PROJECT_SOURCE_DIR}/src)

add_executable(trtf_dataset_benchmark examples/trtf_dataset_benchmark.cpp)
target_link_libraries(trtf_dataset_benchmark PRIVATE trtf_core)
target_include_directories(trtf_dataset_benchmark PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/stb
)

enable_testing()

# --- Test helper function ---
# Usage:
#   trtf_add_test(test_name)               -- links trtf_core, adds src/ include
#   trtf_add_test(test_name REQUIRES_TRT)  -- also links TensorRT test deps
function(trtf_add_test TEST_NAME)
  cmake_parse_arguments(ARG "NO_SRC_INCLUDE;REQUIRES_TRT" "" "EXTRA_INCLUDES" ${ARGN})
  if(ARG_REQUIRES_TRT AND NOT _trtf_can_build_trt_backend)
    message(STATUS "Skipping ${TEST_NAME}: TensorRT SDK not available")
    return()
  endif()
  add_executable(${TEST_NAME} tests/cpp/${TEST_NAME}.cpp)
  target_link_libraries(${TEST_NAME} PRIVATE trtf_core)
  if(NOT ARG_NO_SRC_INCLUDE)
    target_include_directories(${TEST_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src)
  endif()
  if(ARG_EXTRA_INCLUDES)
    target_include_directories(${TEST_NAME} PRIVATE ${ARG_EXTRA_INCLUDES})
  endif()
  target_include_directories(${TEST_NAME} SYSTEM PRIVATE ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(${TEST_NAME} PRIVATE ${TRTF_CUDART_LIBRARY})
  if(ARG_REQUIRES_TRT)
    target_compile_definitions(${TEST_NAME} PRIVATE TRTF_HAS_TRT=1)
    target_include_directories(${TEST_NAME} SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR})
    target_link_libraries(${TEST_NAME} PRIVATE ${TRTF_TRT_LIBRARY})
  else()
    target_compile_definitions(${TEST_NAME} PRIVATE TRTF_HAS_TRT=0)
  endif()
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# --- Tests ---

# Bundle / format
trtf_add_test(test_bundle_format)
trtf_add_test(test_bundle_e2e          NO_SRC_INCLUDE)
trtf_add_test(test_bundle_view)
trtf_add_test(test_pipeline_registry)
trtf_add_test(test_backend_loader)
trtf_add_test(test_trt_version)

# C ABI
trtf_add_test(test_c_abi_entry                NO_SRC_INCLUDE)
trtf_add_test(test_c_abi_runtime_regression   NO_SRC_INCLUDE)

# Config / parsing
trtf_add_test(test_text_parsers)
trtf_add_test(test_json_helpers)
trtf_add_test(test_data_dir)
trtf_add_test(test_cli_args            NO_SRC_INCLUDE)
trtf_add_test(test_chat_template)
trtf_add_test(test_config_schema_registry)
trtf_add_test(test_config_cli_support)
trtf_add_test(test_config_schemas_triattention)

# TRT runtime core
trtf_add_test(test_decode_runtime REQUIRES_TRT)
trtf_add_test(test_sampler)
trtf_add_test(test_plugin_helpers)
trtf_add_test(test_trt_logger REQUIRES_TRT)
if(TARGET trtf_backend_trt)
  target_link_libraries(test_trt_logger PRIVATE trtf_backend_trt)
endif()
trtf_add_test(test_trt_runtime_lifetime REQUIRES_TRT)
if(TARGET trtf_backend_trt)
  target_link_libraries(test_trt_runtime_lifetime PRIVATE trtf_backend_trt)
endif()
trtf_add_test(test_trt_engine_lifecycle)
trtf_add_test(test_trt_engine_lifecycle_fake_engine)
trtf_add_test(test_trt_module REQUIRES_TRT)
if(TARGET trtf_backend_trt)
  target_link_libraries(test_trt_module PRIVATE trtf_backend_trt)
endif()

# CUDA / device
trtf_add_test(test_cuda_buffer REQUIRES_TRT)
trtf_add_test(test_cuda_stream REQUIRES_TRT)
trtf_add_test(test_cuda_graph REQUIRES_TRT)
if(TARGET trtf_backend_trt)
  target_link_libraries(test_cuda_graph PRIVATE trtf_backend_trt)
endif()
trtf_add_test(test_device_kv_cache)
trtf_add_test(test_device_resources REQUIRES_TRT)
if(TARGET trtf_backend_trt)
  target_link_libraries(test_device_resources PRIVATE trtf_backend_trt)
endif()
trtf_add_test(test_device_tensor)
trtf_add_test(test_kv_cache_new)
trtf_add_test(test_triattention_kv_cache)
trtf_add_test(test_recurrent_state)

# Pipelines (public API)
trtf_add_test(test_pipeline_api                NO_SRC_INCLUDE)
trtf_add_test(test_text_generation_pipeline REQUIRES_TRT)
if(TARGET trtf_backend_trt)
  target_link_libraries(test_text_generation_pipeline PRIVATE trtf_backend_trt)
endif()
trtf_add_test(test_recurrent_pipeline REQUIRES_TRT)
if(TARGET trtf_backend_trt)
  target_link_libraries(test_recurrent_pipeline PRIVATE trtf_backend_trt)
endif()
trtf_add_test(test_encoder_pipeline REQUIRES_TRT)
if(TARGET trtf_backend_trt)
  target_link_libraries(test_encoder_pipeline PRIVATE trtf_backend_trt)
endif()
trtf_add_test(test_vl_pipeline REQUIRES_TRT)
if(TARGET trtf_backend_trt)
  target_link_libraries(test_vl_pipeline PRIVATE trtf_backend_trt)
endif()
trtf_add_test(test_diffusion_pipeline_new)
trtf_add_test(test_audio_pipeline_new REQUIRES_TRT)
if(TARGET trtf_backend_trt)
  target_link_libraries(test_audio_pipeline_new PRIVATE trtf_backend_trt)
endif()

# Tokenizers
trtf_add_test(test_vocab_tokenizer     NO_SRC_INCLUDE)
trtf_add_test(test_ipa_tokenizer)
trtf_add_test(test_bpe_tokenizer)
trtf_add_test(test_bpe_golden)
trtf_add_test(test_bpe_benchmark)
trtf_add_test(test_wordpiece_tokenizer)
trtf_add_test(test_wordpiece_golden)
trtf_add_test(test_unigram_tokenizer)
trtf_add_test(test_unigram_golden)
trtf_add_test(test_tokenizer_edge_cases)

# Image I/O
trtf_add_test(test_image_reader        NO_SRC_INCLUDE)
trtf_add_test(test_trtf_io             NO_SRC_INCLUDE)

# Vision / VL
trtf_add_test(test_image_preprocessor)
trtf_add_test(test_vl_decode_policy)
trtf_add_test(test_vision_execution_plan)

# Diffusion
trtf_add_test(test_diffusion_generation_plan)
trtf_add_test(test_diffusion_denoising_step_seam)
trtf_add_test(test_flow_match_scheduler)
trtf_add_test(test_diffusion_math)

# Recurrent (Mamba / RWKV)
trtf_add_test(test_recurrent_step_contracts)

# Audio — Whisper
trtf_add_test(test_audio_bundle_validation)
trtf_add_test(test_whisper_host_plan)
trtf_add_test(test_whisper_decode_policy)
trtf_add_test(test_rnnt_decode_policy)
trtf_add_test(test_rnnt_streaming_contract)
trtf_add_test(test_wav_reader)
trtf_add_test(test_audio_types)
trtf_add_test(test_mel_spectrogram)

# Audio — Bark
trtf_add_test(test_bark_generation_plan)

# Audio — Speech (PersonaPlex / Magpie)
trtf_add_test(test_speech_subprocess_seam      NO_SRC_INCLUDE)
trtf_add_test(test_speech_decode_stop_policy   NO_SRC_INCLUDE)
trtf_add_test(test_speech_generation_helpers)
trtf_add_test(test_speech_depth_plan)
trtf_add_test(test_speech_runtime_plan)
trtf_add_test(test_speech_temporal_embed_plan)
trtf_add_test(test_speech_mimi_decode_plan)
trtf_add_test(test_magpie_decode_policy)
trtf_add_test(test_magpie_decoder_plan)
trtf_add_test(test_magpie_text_completion_policy)
trtf_add_test(test_magpie_codec_plan)

# Audio — Omni
trtf_add_test(test_omni_audio_plan)

# Perception (segmentation, SAM, neural operator)
trtf_add_test(test_neural_operator_config)
trtf_add_test(test_sam_prompt_seam)
trtf_add_test(test_perception_preprocess_seams)

# Diffusion — conditioning
trtf_add_test(test_wan_generation_conditioning)

# IoMap tensor name mapping
trtf_add_test(test_io_map)

# Decoder config parsing (legacy name: test_torchtrt_decoder)
trtf_add_test(test_torchtrt_decoder)

# TVM-FFI plugin
trtf_add_test(test_tvm_ffi_plugin REQUIRES_TRT)
if(TARGET test_tvm_ffi_plugin AND _trtf_has_tvm_ffi)
  target_include_directories(test_tvm_ffi_plugin SYSTEM PRIVATE ${TRTF_TVM_FFI_INCLUDE_DIR})
  target_link_libraries(test_tvm_ffi_plugin PRIVATE ${TRTF_TVM_FFI_LIBRARY})
endif()

# TVM-FFI plugin V2 (self-contained test with IPluginV2DynamicExt)
trtf_add_test(test_tvm_ffi_plugin_v2 NO_SRC_INCLUDE REQUIRES_TRT)

# TVM-FFI module loader (loads FlashInfer .so from C++)
trtf_add_test(test_tvm_ffi_module_loader REQUIRES_TRT)
if(TARGET test_tvm_ffi_module_loader AND _trtf_has_tvm_ffi)
  target_include_directories(test_tvm_ffi_module_loader SYSTEM PRIVATE ${TRTF_TVM_FFI_INCLUDE_DIR})
  target_link_libraries(test_tvm_ffi_module_loader PRIVATE ${TRTF_TVM_FFI_LIBRARY})
endif()
if(TARGET test_tvm_ffi_plugin_v2 AND _trtf_has_tvm_ffi)
  target_include_directories(test_tvm_ffi_plugin_v2 SYSTEM PRIVATE ${TRTF_TVM_FFI_INCLUDE_DIR})
  target_link_libraries(test_tvm_ffi_plugin_v2 PRIVATE ${TRTF_TVM_FFI_LIBRARY})
endif()

# --- Install ---
include(CMakePackageConfigHelpers)

# Install library
install(TARGETS trtf_core
  EXPORT trtfTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(TARGET trtf_backend_trt)
  install(TARGETS trtf_backend_trt
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
  if(TRTF_TRT_BACKEND_ABI)
    install(FILES "${CMAKE_BINARY_DIR}/libtrtf_backend_trt_${TRTF_TRT_BACKEND_ABI}.so"
      DESTINATION ${CMAKE_INSTALL_LIBDIR}
      OPTIONAL
    )
  endif()
endif()

# Install public headers
install(DIRECTORY include/trtf/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/trtf
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
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
