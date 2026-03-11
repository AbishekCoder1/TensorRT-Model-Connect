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
  src/cabi/api/trtf_c.cpp
  src/cabi/config/fast_path_config.cpp
  src/cabi/bundle/bundle_helpers.cpp
  src/runtime/trt/core/trt_common.cpp
  src/runtime/trt/core/trt_engine_lifecycle.cpp
  src/runtime/trt/core/trt_decode_runtime.cpp
  src/runtime/trt/core/device_kv_cache.cpp
  src/runtime/trt/recurrent/mamba_step_state.cpp
  src/runtime/trt/recurrent/mamba_decode_runtime.cpp
  src/runtime/trt/recurrent/mamba_backend.cpp
  src/runtime/trt/recurrent/rwkv_step_state.cpp
  src/runtime/trt/recurrent/rwkv_decode_runtime.cpp
  src/runtime/trt/recurrent/rwkv_backend.cpp
  src/runtime/trt/audio/whisper_backend.cpp
  src/runtime/trt/multimodal/vl_backend.cpp
  src/runtime/trt/multimodal/image_preprocessor.cpp
  src/runtime/trt/multimodal/vision_engine.cpp
  src/runtime/trt/encoder/encoder_backend.cpp
  src/runtime/trt/encoder/embedding_backend.cpp
  src/runtime/trt/encoder/reranking_backend.cpp
  src/runtime/trt/perception/segmentation_backend.cpp
  src/runtime/trt/perception/sam_backend.cpp
  src/runtime/trt/perception/detection_backend.cpp
  src/runtime/trt/perception/neural_operator_backend.cpp
  src/runtime/trt/recurrent/hybrid_backend.cpp
  src/runtime/trt/audio/bark_backend.cpp
  src/runtime/trt/audio/magpie_tts_backend.cpp
  src/runtime/trt/audio/speech_backend.cpp
  src/runtime/trt/diffusion/diffusion_backend.cpp
  src/runtime/trt/diffusion/diffusion_backend_base.cpp
  src/runtime/trt/diffusion/wan_diffusion_backend.cpp
  src/runtime/trt/diffusion/flux_diffusion_backend.cpp
  src/runtime/trt/diffusion/z_image_diffusion_backend.cpp
  src/runtime/trt/core/stb_impl.cpp
  src/tokenizer/hf_python_tokenizer.cpp
  src/tokenizer/vocab_tokenizer.cpp
  src/tokenizer/ipa_tokenizer.cpp
  src/utils/data_dir.cpp
  src/utils/text_parsers.cpp
  src/utils/json_helpers.cpp
  src/utils/wav_reader.cpp
  src/utils/image_reader.cpp
  src/runtime/trt/audio/mel_spectrogram.cpp
  src/runtime/trt/core/device_tensor.cpp
  src/runtime/trt/core/trt_module.cpp
  src/runtime/trt/core/kv_cache.cpp
  src/runtime/trt/core/recurrent_state.cpp
  src/runtime/trt/core/flow_match_euler_scheduler.cpp
  src/runtime/pipeline_factory.cpp
  src/runtime/pipelines/text_generation_pipeline.cpp
  src/runtime/pipelines/recurrent_pipeline.cpp
  src/runtime/pipelines/encoder_pipeline.cpp
  src/runtime/pipelines/vl_pipeline.cpp
  src/runtime/pipelines/diffusion_pipeline.cpp
  src/runtime/pipelines/audio_pipeline.cpp
  src/runtime/pipelines/audio_backend_factory.cpp
  src/runtime/pipelines/diffusion_backend_factory.cpp
  src/runtime/trt/audio/audio_bundle_validation.cpp
)

# Add CUDA kernel sources when CUDA compiler is available
if(CMAKE_CUDA_COMPILER)
  target_sources(trtf_core PRIVATE src/runtime/trt/audio/magpie_kernels.cu)
  set_source_files_properties(src/runtime/trt/audio/magpie_kernels.cu PROPERTIES
    LANGUAGE CUDA
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
target_include_directories(trtf PRIVATE ${PROJECT_SOURCE_DIR}/third_party/stb)

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
  if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY
     AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
    target_include_directories(${TEST_NAME} SYSTEM PRIVATE
      ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
    target_link_libraries(${TEST_NAME} PRIVATE
      ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
  endif()
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# --- Tests ---

# Bundle / format
trtf_add_test(test_bundle_format)
trtf_add_test(test_bundle_e2e          NO_SRC_INCLUDE)
trtf_add_test(test_bundle_helpers)

# C ABI
trtf_add_test(test_c_abi_entry                NO_SRC_INCLUDE)
trtf_add_test(test_c_abi_runtime_regression   NO_SRC_INCLUDE)

# Config / parsing
trtf_add_test(test_fast_path_config)
trtf_add_test(test_text_parsers)
trtf_add_test(test_json_helpers)
trtf_add_test(test_data_dir)
trtf_add_test(test_cli_args            NO_SRC_INCLUDE)

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
trtf_add_test(test_device_kv_cache)
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
trtf_add_test(test_hf_python_tokenizer)
trtf_add_test(test_ipa_tokenizer)

# Vision / VL
trtf_add_test(test_image_preprocessor)
trtf_add_test(test_vl_decode_policy)
trtf_add_test(test_vision_execution_plan)

# Diffusion
trtf_add_test(test_diffusion_generation_plan)
trtf_add_test(test_diffusion_denoising_step_seam)
trtf_add_test(test_flow_match_scheduler)

# Recurrent (Mamba / RWKV)
trtf_add_test(test_recurrent_step_contracts)

# Audio — Whisper
trtf_add_test(test_audio_bundle_validation)
trtf_add_test(test_whisper_host_plan)
trtf_add_test(test_whisper_decode_policy)
trtf_add_test(test_wav_reader)
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
