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

# TensorRT is mandatory — the container image must provide TRT and CUDA.
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

if(NOT TRTF_TRT_INCLUDE_DIR OR NOT TRTF_TRT_LIBRARY)
  message(FATAL_ERROR
    "TensorRT is required but not found.\n"
    "  TRTF_TRT_INCLUDE_DIR=${TRTF_TRT_INCLUDE_DIR}\n"
    "  TRTF_TRT_LIBRARY=${TRTF_TRT_LIBRARY}\n"
    "Ensure the container image has libnvinfer-headers-dev and tensorrt_libs installed.")
endif()

if(NOT TRTF_CUDA_INCLUDE_DIR OR NOT TRTF_CUDART_LIBRARY)
  message(FATAL_ERROR
    "CUDA dev dependencies are required but not found.\n"
    "  TRTF_CUDA_INCLUDE_DIR=${TRTF_CUDA_INCLUDE_DIR}\n"
    "  TRTF_CUDART_LIBRARY=${TRTF_CUDART_LIBRARY}")
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
add_library(trtf_core
  src/bundle/bundle_format.cpp
  src/bundle/bundle_view.cpp
  src/cabi/api/trtf_c.cpp
  # Core runtime infrastructure (TRT engine, CUDA, KV cache, decode)
  src/runtime/core/trt_common.cpp
  src/runtime/core/gpu_matmul.cpp
  src/runtime/core/trt_engine_lifecycle.cpp
  src/runtime/core/trt_decode_runtime.cpp
  src/runtime/core/sampler.cpp
  src/runtime/core/device_kv_cache.cpp
  src/runtime/core/device_tensor.cpp
  src/runtime/core/trt_module.cpp
  src/runtime/core/kv_cache.cpp
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
  # Registry (factory + plugin dispatch)
  src/runtime/registry/pipeline_factory.cpp
  src/runtime/registry/pipeline_registry.cpp
  src/runtime/registry/pipeline_plugin.cpp
  # Plugin shared helpers
  src/runtime/plugins/shared/plugin_helpers.cpp
  src/runtime/plugins/shared/diffusion_helpers.cpp
  src/runtime/plugins/shared/audio_helpers.cpp
  src/runtime/plugins/force_link_plugins.cpp
  # Pipeline plugins (self-registering via REGISTER_PIPELINE_PLUGIN)
  src/runtime/plugins/decoder_plugin.cpp
  src/runtime/plugins/ssm_plugin.cpp
  src/runtime/plugins/rwkv_plugin.cpp
  src/runtime/plugins/hybrid_plugin.cpp
  src/runtime/plugins/encoder_plugin.cpp
  src/runtime/plugins/patchtst_plugin.cpp
  src/runtime/plugins/patchtsmixer_plugin.cpp
  src/runtime/plugins/timesfm_plugin.cpp
  src/runtime/plugins/chronos_bolt_plugin.cpp
  src/runtime/plugins/segmentation_plugin.cpp
  src/runtime/plugins/object_detection_plugin.cpp
  src/runtime/plugins/vl_plugin.cpp
  src/runtime/plugins/whisper_plugin.cpp
  src/runtime/plugins/bark_plugin.cpp
  src/runtime/plugins/magpie_plugin.cpp
  src/runtime/plugins/speech_plugin.cpp
  src/runtime/plugins/omni_plugin.cpp
  src/runtime/plugins/flux_plugin.cpp
  src/runtime/plugins/wan_plugin.cpp
  src/runtime/plugins/zimage_plugin.cpp
  src/runtime/plugins/t5_plugin.cpp
  src/runtime/plugins/marian_plugin.cpp
  src/runtime/plugins/seq2seq_plugin.cpp
  src/runtime/plugins/pixart_plugin.cpp
  src/runtime/plugins/pixart_torchtrt_plugin.cpp
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
  )
  set_source_files_properties(
    src/runtime/domains/audio/magpie_kernels.cu
    src/runtime/core/argmax_kernel.cu
    PROPERTIES LANGUAGE CUDA
  )
  target_include_directories(trtf_core PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_compile_definitions(trtf_core PRIVATE TRTF_HAS_CUDA_KERNELS=1)
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

option(TRTF_ENABLE_TRT "Enable TensorRT backend" ON)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_compile_definitions(trtf_core PUBLIC TRTF_HAS_TRT=1)
  target_include_directories(trtf_core SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  find_library(TRTF_CUBLAS_LIBRARY NAMES cublas libcublas.so
    PATHS ${_trtf_default_search_paths} PATH_SUFFIXES "" lib lib64)
  target_link_libraries(trtf_core PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY}
    $<$<BOOL:${TRTF_CUBLAS_LIBRARY}>:${TRTF_CUBLAS_LIBRARY}>)
  message(STATUS "TensorRT backend enabled")
  message(STATUS "  TRT include: ${TRTF_TRT_INCLUDE_DIR}")
  message(STATUS "  TRT library: ${TRTF_TRT_LIBRARY}")
  message(STATUS "  CUDA include: ${TRTF_CUDA_INCLUDE_DIR}")
  message(STATUS "  CUDART lib : ${TRTF_CUDART_LIBRARY}")
else()
  target_compile_definitions(trtf_core PUBLIC TRTF_HAS_TRT=0)
  message(STATUS "TensorRT backend disabled (missing TensorRT or CUDA dev dependencies)")
endif()

# --- TVM-FFI kernel bridge plugin (optional) ---
option(TRTF_ENABLE_TVM_FFI "Enable TVM-FFI kernel bridge plugin" ON)

set(_trtf_has_tvm_ffi FALSE)

if(TRTF_ENABLE_TVM_FFI AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY)
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
      src/plugins/tvm_ffi_kernel_plugin.cpp
      src/plugins/tvm_ffi_kernel_creator.cpp
      src/plugins/tvm_ffi_module_loader.cpp
    )
    target_compile_definitions(trtf_core PUBLIC TRTF_HAS_TVM_FFI=1)
    target_include_directories(trtf_core SYSTEM PRIVATE ${TRTF_TVM_FFI_INCLUDE_DIR})
    target_link_libraries(trtf_core PRIVATE ${TRTF_TVM_FFI_LIBRARY})
    message(STATUS "TVM-FFI kernel bridge enabled")
    message(STATUS "  TVM-FFI include: ${TRTF_TVM_FFI_INCLUDE_DIR}")
    message(STATUS "  TVM-FFI library: ${TRTF_TVM_FFI_LIBRARY}")
  endif()
endif()

if(NOT _trtf_has_tvm_ffi)
  target_compile_definitions(trtf_core PUBLIC TRTF_HAS_TVM_FFI=0)
  message(STATUS "TVM-FFI kernel bridge disabled")
endif()

# Build the TVM-FFI plugin as a standalone shared library for dynamic loading
# via IPluginRegistry::loadLibrary(). This avoids static-lib registry issues.
if(_trtf_has_tvm_ffi)
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

add_executable(trtf examples/trtf_cli.cpp)
target_link_libraries(trtf PRIVATE trtf_core)
target_include_directories(trtf PRIVATE ${PROJECT_SOURCE_DIR}/third_party/stb ${PROJECT_SOURCE_DIR}/src)

enable_testing()

# --- Test helper function ---
# Usage:
#   trtf_add_test(test_name)               -- links trtf_core, adds src/ include, adds TRT if available
#   trtf_add_test(test_name NO_SRC_INCLUDE) -- links trtf_core, no src/ include, adds TRT if available
function(trtf_add_test TEST_NAME)
  cmake_parse_arguments(ARG "NO_SRC_INCLUDE" "" "EXTRA_INCLUDES" ${ARGN})
  add_executable(${TEST_NAME} tests/cpp/${TEST_NAME}.cpp)
  target_link_libraries(${TEST_NAME} PRIVATE trtf_core)
  if(NOT ARG_NO_SRC_INCLUDE)
    target_include_directories(${TEST_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src)
  endif()
  if(ARG_EXTRA_INCLUDES)
    target_include_directories(${TEST_NAME} PRIVATE ${ARG_EXTRA_INCLUDES})
  endif()
  target_include_directories(${TEST_NAME} SYSTEM PRIVATE
    ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(${TEST_NAME} PRIVATE
    ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# --- Tests ---

# Bundle / format
trtf_add_test(test_bundle_format)
trtf_add_test(test_bundle_e2e          NO_SRC_INCLUDE)
trtf_add_test(test_bundle_view)
trtf_add_test(test_pipeline_registry)

# C ABI
trtf_add_test(test_c_abi_entry                NO_SRC_INCLUDE)
trtf_add_test(test_c_abi_runtime_regression   NO_SRC_INCLUDE)

# Config / parsing
trtf_add_test(test_text_parsers)
trtf_add_test(test_json_helpers)
trtf_add_test(test_data_dir)
trtf_add_test(test_cli_args            NO_SRC_INCLUDE)
trtf_add_test(test_chat_template)

# TRT runtime core
trtf_add_test(test_decode_runtime)
trtf_add_test(test_trt_logger)
trtf_add_test(test_trt_runtime_lifetime)
trtf_add_test(test_trt_engine_lifecycle)
trtf_add_test(test_trt_engine_lifecycle_fake_engine)
trtf_add_test(test_trt_module)

# CUDA / device
trtf_add_test(test_cuda_buffer)
trtf_add_test(test_cuda_stream)
trtf_add_test(test_cuda_graph)
trtf_add_test(test_device_kv_cache)
trtf_add_test(test_device_resources)
trtf_add_test(test_device_tensor)
trtf_add_test(test_kv_cache_new)
trtf_add_test(test_recurrent_state)

# Pipelines (public API)
trtf_add_test(test_pipeline_api                NO_SRC_INCLUDE)
trtf_add_test(test_text_generation_pipeline)
trtf_add_test(test_recurrent_pipeline)
trtf_add_test(test_encoder_pipeline)
trtf_add_test(test_vl_pipeline)
trtf_add_test(test_diffusion_pipeline_new)
trtf_add_test(test_audio_pipeline_new)

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
trtf_add_test(test_tvm_ffi_plugin)
if(_trtf_has_tvm_ffi)
  target_include_directories(test_tvm_ffi_plugin SYSTEM PRIVATE ${TRTF_TVM_FFI_INCLUDE_DIR})
  target_link_libraries(test_tvm_ffi_plugin PRIVATE ${TRTF_TVM_FFI_LIBRARY})
endif()

# TVM-FFI plugin V2 (self-contained test with IPluginV2DynamicExt)
trtf_add_test(test_tvm_ffi_plugin_v2 NO_SRC_INCLUDE)

# TVM-FFI module loader (loads FlashInfer .so from C++)
trtf_add_test(test_tvm_ffi_module_loader)
if(_trtf_has_tvm_ffi)
  target_include_directories(test_tvm_ffi_module_loader SYSTEM PRIVATE ${TRTF_TVM_FFI_INCLUDE_DIR})
  target_link_libraries(test_tvm_ffi_module_loader PRIVATE ${TRTF_TVM_FFI_LIBRARY})
endif()
if(_trtf_has_tvm_ffi)
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
