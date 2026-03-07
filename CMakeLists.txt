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
  src/runtime/pipeline/router.cpp
  src/runtime/adapters/io/audio_io_adapter.cpp
  src/runtime/adapters/io/detection_io_adapter.cpp
  src/runtime/adapters/io/image_io_adapter.cpp
  src/runtime/adapters/io/media_io_adapter.cpp
  src/runtime/adapters/bundle/bundle_port_adapter.cpp
  src/runtime/adapters/trt/trt_port_adapter.cpp
  src/runtime/builders/text/text_strategy_builder.cpp
  src/runtime/builders/encoder/encoder_strategy_builder.cpp
  src/runtime/builders/vision/vision_strategy_builder.cpp
  src/runtime/builders/audio/audio_bundle_validation.cpp
  src/runtime/builders/audio/audio_strategy_builder.cpp
  src/runtime/builders/diffusion/diffusion_strategy_builder.cpp
  src/runtime/services/common/runtime_service_ports.cpp
  src/runtime/services/text/generation_text_service.cpp
  src/runtime/services/vision/vision_runtime_services.cpp
  src/runtime/services/audio/audio_runtime_services.cpp
  src/runtime/trt/core/trt_common.cpp
  src/runtime/trt/core/trt_engine_lifecycle.cpp
  src/runtime/trt/core/trt_decode_runtime.cpp
  src/runtime/trt/core/trt_backend_shared.cpp
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
  src/runtime/trt/audio/omni_backend.cpp
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
  src/runtime/trt/audio/mel_spectrogram.cpp
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

enable_testing()

# --- Tests ---

add_executable(test_bundle_format tests/cpp/test_bundle_format.cpp)
target_link_libraries(test_bundle_format PRIVATE trtf_core)
target_include_directories(test_bundle_format PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_bundle_format COMMAND test_bundle_format)

add_executable(test_runtime_contracts tests/cpp/test_runtime_contracts.cpp)
target_include_directories(test_runtime_contracts PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME test_runtime_contracts COMMAND test_runtime_contracts)

add_executable(test_c_abi_entry tests/cpp/test_c_abi_entry.cpp)
target_link_libraries(test_c_abi_entry PRIVATE trtf_core)
add_test(NAME test_c_abi_entry COMMAND test_c_abi_entry)

add_executable(test_c_abi_runtime_regression tests/cpp/test_c_abi_runtime_regression.cpp)
target_link_libraries(test_c_abi_runtime_regression PRIVATE trtf_core)
add_test(NAME test_c_abi_runtime_regression COMMAND test_c_abi_runtime_regression)

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

add_executable(test_pipeline_router
  tests/cpp/test_pipeline_router.cpp
)
target_link_libraries(test_pipeline_router PRIVATE trtf_core)
target_include_directories(test_pipeline_router PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME test_pipeline_router COMMAND test_pipeline_router)

add_executable(test_runtime_io_adapters
  tests/cpp/test_runtime_io_adapters.cpp
)
target_link_libraries(test_runtime_io_adapters PRIVATE trtf_core)
target_include_directories(test_runtime_io_adapters PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_runtime_io_adapters COMMAND test_runtime_io_adapters)

add_executable(test_text_strategy_builder
  tests/cpp/test_text_strategy_builder.cpp
)
target_link_libraries(test_text_strategy_builder PRIVATE trtf_core)
target_include_directories(test_text_strategy_builder PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/cpp ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_text_strategy_builder COMMAND test_text_strategy_builder)

add_executable(test_encoder_strategy_builder
  tests/cpp/test_encoder_strategy_builder.cpp
)
target_link_libraries(test_encoder_strategy_builder PRIVATE trtf_core)
target_include_directories(test_encoder_strategy_builder PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/cpp ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_encoder_strategy_builder COMMAND test_encoder_strategy_builder)

add_executable(test_vision_strategy_builder
  tests/cpp/test_vision_strategy_builder.cpp
)
target_link_libraries(test_vision_strategy_builder PRIVATE trtf_core)
target_include_directories(test_vision_strategy_builder PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/cpp ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_vision_strategy_builder COMMAND test_vision_strategy_builder)

add_executable(test_audio_strategy_builder
  tests/cpp/test_audio_strategy_builder.cpp
)
target_link_libraries(test_audio_strategy_builder PRIVATE trtf_core)
target_include_directories(test_audio_strategy_builder PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/cpp ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_audio_strategy_builder COMMAND test_audio_strategy_builder)

add_executable(test_diffusion_strategy_builder
  tests/cpp/test_diffusion_strategy_builder.cpp
)
target_link_libraries(test_diffusion_strategy_builder PRIVATE trtf_core)
target_include_directories(test_diffusion_strategy_builder PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/cpp ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_diffusion_strategy_builder COMMAND test_diffusion_strategy_builder)

add_executable(test_diffusion_generation_plan tests/cpp/test_diffusion_generation_plan.cpp)
target_link_libraries(test_diffusion_generation_plan PRIVATE trtf_core)
target_include_directories(test_diffusion_generation_plan PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_diffusion_generation_plan SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_diffusion_generation_plan PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_diffusion_generation_plan COMMAND test_diffusion_generation_plan)

add_executable(test_diffusion_denoising_step_seam tests/cpp/test_diffusion_denoising_step_seam.cpp)
target_link_libraries(test_diffusion_denoising_step_seam PRIVATE trtf_core)
target_include_directories(test_diffusion_denoising_step_seam PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_diffusion_denoising_step_seam COMMAND test_diffusion_denoising_step_seam)

add_executable(test_recurrent_step_contracts tests/cpp/test_recurrent_step_contracts.cpp)
target_link_libraries(test_recurrent_step_contracts PRIVATE trtf_core)
target_include_directories(test_recurrent_step_contracts PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_recurrent_step_contracts COMMAND test_recurrent_step_contracts)

add_executable(test_audio_bundle_validation tests/cpp/test_audio_bundle_validation.cpp)
target_link_libraries(test_audio_bundle_validation PRIVATE trtf_core)
target_include_directories(test_audio_bundle_validation PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_audio_bundle_validation SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_audio_bundle_validation PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_audio_bundle_validation COMMAND test_audio_bundle_validation)

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

add_executable(test_vl_decode_policy tests/cpp/test_vl_decode_policy.cpp)
target_link_libraries(test_vl_decode_policy PRIVATE trtf_core)
target_include_directories(test_vl_decode_policy PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_vl_decode_policy COMMAND test_vl_decode_policy)

add_executable(test_vision_execution_plan tests/cpp/test_vision_execution_plan.cpp)
target_link_libraries(test_vision_execution_plan PRIVATE trtf_core)
target_include_directories(test_vision_execution_plan PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_vision_execution_plan COMMAND test_vision_execution_plan)

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

add_executable(test_trt_runtime_lifetime tests/cpp/test_trt_runtime_lifetime.cpp)
target_link_libraries(test_trt_runtime_lifetime PRIVATE trtf_core)
target_include_directories(test_trt_runtime_lifetime PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_trt_runtime_lifetime SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_trt_runtime_lifetime PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_trt_runtime_lifetime COMMAND test_trt_runtime_lifetime)

add_executable(test_trt_engine_lifecycle tests/cpp/test_trt_engine_lifecycle.cpp)
target_link_libraries(test_trt_engine_lifecycle PRIVATE trtf_core)
target_include_directories(test_trt_engine_lifecycle PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_trt_engine_lifecycle SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_trt_engine_lifecycle PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_trt_engine_lifecycle COMMAND test_trt_engine_lifecycle)

add_executable(test_trt_engine_lifecycle_fake_engine tests/cpp/test_trt_engine_lifecycle_fake_engine.cpp)
target_link_libraries(test_trt_engine_lifecycle_fake_engine PRIVATE trtf_core)
target_include_directories(test_trt_engine_lifecycle_fake_engine PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_trt_engine_lifecycle_fake_engine SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_trt_engine_lifecycle_fake_engine PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_trt_engine_lifecycle_fake_engine COMMAND test_trt_engine_lifecycle_fake_engine)

add_executable(test_vocab_tokenizer tests/cpp/test_vocab_tokenizer.cpp)
target_link_libraries(test_vocab_tokenizer PRIVATE trtf_core)
add_test(NAME test_vocab_tokenizer COMMAND test_vocab_tokenizer)

add_executable(test_hf_python_tokenizer tests/cpp/test_hf_python_tokenizer.cpp)
target_link_libraries(test_hf_python_tokenizer PRIVATE trtf_core)
target_include_directories(test_hf_python_tokenizer PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_hf_python_tokenizer COMMAND test_hf_python_tokenizer)

add_executable(test_speech_subprocess_seam tests/cpp/test_speech_subprocess_seam.cpp)
target_link_libraries(test_speech_subprocess_seam PRIVATE trtf_core)
add_test(NAME test_speech_subprocess_seam COMMAND test_speech_subprocess_seam)

add_executable(test_speech_decode_stop_policy tests/cpp/test_speech_decode_stop_policy.cpp)
target_link_libraries(test_speech_decode_stop_policy PRIVATE trtf_core)
add_test(NAME test_speech_decode_stop_policy COMMAND test_speech_decode_stop_policy)

add_executable(test_speech_generation_helpers tests/cpp/test_speech_generation_helpers.cpp)
target_link_libraries(test_speech_generation_helpers PRIVATE trtf_core)
target_include_directories(test_speech_generation_helpers PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_speech_generation_helpers COMMAND test_speech_generation_helpers)

add_executable(test_speech_depth_plan tests/cpp/test_speech_depth_plan.cpp)
target_link_libraries(test_speech_depth_plan PRIVATE trtf_core)
target_include_directories(test_speech_depth_plan PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_speech_depth_plan SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_speech_depth_plan PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_speech_depth_plan COMMAND test_speech_depth_plan)

add_executable(test_speech_runtime_plan tests/cpp/test_speech_runtime_plan.cpp)
target_link_libraries(test_speech_runtime_plan PRIVATE trtf_core)
target_include_directories(test_speech_runtime_plan PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_speech_runtime_plan SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_speech_runtime_plan PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_speech_runtime_plan COMMAND test_speech_runtime_plan)

add_executable(test_speech_temporal_embed_plan tests/cpp/test_speech_temporal_embed_plan.cpp)
target_link_libraries(test_speech_temporal_embed_plan PRIVATE trtf_core)
target_include_directories(test_speech_temporal_embed_plan PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_speech_temporal_embed_plan SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_speech_temporal_embed_plan PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_speech_temporal_embed_plan COMMAND test_speech_temporal_embed_plan)

add_executable(test_speech_mimi_decode_plan tests/cpp/test_speech_mimi_decode_plan.cpp)
target_link_libraries(test_speech_mimi_decode_plan PRIVATE trtf_core)
target_include_directories(test_speech_mimi_decode_plan PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_speech_mimi_decode_plan SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_speech_mimi_decode_plan PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_speech_mimi_decode_plan COMMAND test_speech_mimi_decode_plan)

add_executable(test_whisper_host_plan tests/cpp/test_whisper_host_plan.cpp)
target_link_libraries(test_whisper_host_plan PRIVATE trtf_core)
target_include_directories(test_whisper_host_plan PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_whisper_host_plan SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_whisper_host_plan PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_whisper_host_plan COMMAND test_whisper_host_plan)

add_executable(test_whisper_decode_policy tests/cpp/test_whisper_decode_policy.cpp)
target_link_libraries(test_whisper_decode_policy PRIVATE trtf_core)
target_include_directories(test_whisper_decode_policy PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_whisper_decode_policy COMMAND test_whisper_decode_policy)

add_executable(test_bark_generation_plan tests/cpp/test_bark_generation_plan.cpp)
target_link_libraries(test_bark_generation_plan PRIVATE trtf_core)
target_include_directories(test_bark_generation_plan PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_bark_generation_plan SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_bark_generation_plan PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_bark_generation_plan COMMAND test_bark_generation_plan)

add_executable(test_magpie_decode_policy tests/cpp/test_magpie_decode_policy.cpp)
target_link_libraries(test_magpie_decode_policy PRIVATE trtf_core)
target_include_directories(test_magpie_decode_policy PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_magpie_decode_policy COMMAND test_magpie_decode_policy)

add_executable(test_magpie_decoder_plan tests/cpp/test_magpie_decoder_plan.cpp)
target_link_libraries(test_magpie_decoder_plan PRIVATE trtf_core)
target_include_directories(test_magpie_decoder_plan PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_magpie_decoder_plan SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_magpie_decoder_plan PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_magpie_decoder_plan COMMAND test_magpie_decoder_plan)

add_executable(test_magpie_text_completion_policy tests/cpp/test_magpie_text_completion_policy.cpp)
target_link_libraries(test_magpie_text_completion_policy PRIVATE trtf_core)
target_include_directories(test_magpie_text_completion_policy PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_magpie_text_completion_policy SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_magpie_text_completion_policy PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_magpie_text_completion_policy COMMAND test_magpie_text_completion_policy)

add_executable(test_magpie_codec_plan tests/cpp/test_magpie_codec_plan.cpp)
target_link_libraries(test_magpie_codec_plan PRIVATE trtf_core)
target_include_directories(test_magpie_codec_plan PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_magpie_codec_plan SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_magpie_codec_plan PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_magpie_codec_plan COMMAND test_magpie_codec_plan)

add_executable(test_omni_audio_plan tests/cpp/test_omni_audio_plan.cpp)
target_link_libraries(test_omni_audio_plan PRIVATE trtf_core)
target_include_directories(test_omni_audio_plan PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_omni_audio_plan SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_omni_audio_plan PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_omni_audio_plan COMMAND test_omni_audio_plan)

add_executable(test_ipa_tokenizer tests/cpp/test_ipa_tokenizer.cpp)
target_link_libraries(test_ipa_tokenizer PRIVATE trtf_core)
target_include_directories(test_ipa_tokenizer PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_ipa_tokenizer COMMAND test_ipa_tokenizer)

add_executable(test_bundle_helpers tests/cpp/test_bundle_helpers.cpp)
target_link_libraries(test_bundle_helpers PRIVATE trtf_core)
target_include_directories(test_bundle_helpers PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_bundle_helpers SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_bundle_helpers PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_bundle_helpers COMMAND test_bundle_helpers)

add_executable(test_bundle_port_adapter tests/cpp/test_bundle_port_adapter.cpp)
target_link_libraries(test_bundle_port_adapter PRIVATE trtf_core)
target_include_directories(test_bundle_port_adapter PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_bundle_port_adapter SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_bundle_port_adapter PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_bundle_port_adapter COMMAND test_bundle_port_adapter)

add_executable(test_trt_port_adapter tests/cpp/test_trt_port_adapter.cpp)
target_link_libraries(test_trt_port_adapter PRIVATE trtf_core)
target_include_directories(test_trt_port_adapter PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_trt_port_adapter SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_trt_port_adapter PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_trt_port_adapter COMMAND test_trt_port_adapter)

add_executable(test_generation_text_service tests/cpp/test_generation_text_service.cpp)
target_link_libraries(test_generation_text_service PRIVATE trtf_core)
target_include_directories(test_generation_text_service PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_generation_text_service COMMAND test_generation_text_service)

add_executable(test_audio_runtime_services tests/cpp/test_audio_runtime_services.cpp)
target_link_libraries(test_audio_runtime_services PRIVATE trtf_core)
target_include_directories(test_audio_runtime_services PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_audio_runtime_services COMMAND test_audio_runtime_services)

add_executable(test_vision_runtime_services tests/cpp/test_vision_runtime_services.cpp)
target_link_libraries(test_vision_runtime_services PRIVATE trtf_core)
target_include_directories(test_vision_runtime_services PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_vision_runtime_services COMMAND test_vision_runtime_services)

add_executable(test_runtime_service_ports tests/cpp/test_runtime_service_ports.cpp)
target_link_libraries(test_runtime_service_ports PRIVATE trtf_core)
target_include_directories(test_runtime_service_ports PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_runtime_service_ports SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_runtime_service_ports PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_runtime_service_ports COMMAND test_runtime_service_ports)

add_executable(test_neural_operator_config tests/cpp/test_neural_operator_config.cpp)
target_link_libraries(test_neural_operator_config PRIVATE trtf_core)
target_include_directories(test_neural_operator_config PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_neural_operator_config COMMAND test_neural_operator_config)

add_executable(test_sam_prompt_seam tests/cpp/test_sam_prompt_seam.cpp)
target_link_libraries(test_sam_prompt_seam PRIVATE trtf_core)
target_include_directories(test_sam_prompt_seam PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_sam_prompt_seam SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_sam_prompt_seam PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_sam_prompt_seam COMMAND test_sam_prompt_seam)

add_executable(test_perception_preprocess_seams tests/cpp/test_perception_preprocess_seams.cpp)
target_link_libraries(test_perception_preprocess_seams PRIVATE trtf_core)
target_include_directories(test_perception_preprocess_seams PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_perception_preprocess_seams SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_perception_preprocess_seams PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_perception_preprocess_seams COMMAND test_perception_preprocess_seams)

add_executable(test_wan_generation_conditioning tests/cpp/test_wan_generation_conditioning.cpp)
target_link_libraries(test_wan_generation_conditioning PRIVATE trtf_core)
target_include_directories(test_wan_generation_conditioning PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(TRTF_ENABLE_TRT AND TRTF_TRT_INCLUDE_DIR AND TRTF_TRT_LIBRARY AND TRTF_CUDA_INCLUDE_DIR AND TRTF_CUDART_LIBRARY)
  target_include_directories(test_wan_generation_conditioning SYSTEM PRIVATE ${TRTF_TRT_INCLUDE_DIR} ${TRTF_CUDA_INCLUDE_DIR})
  target_link_libraries(test_wan_generation_conditioning PRIVATE ${TRTF_TRT_LIBRARY} ${TRTF_CUDART_LIBRARY})
endif()
add_test(NAME test_wan_generation_conditioning COMMAND test_wan_generation_conditioning)

add_executable(test_wav_reader tests/cpp/test_wav_reader.cpp)
target_link_libraries(test_wav_reader PRIVATE trtf_core)
target_include_directories(test_wav_reader PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_wav_reader COMMAND test_wav_reader)

add_executable(test_mel_spectrogram tests/cpp/test_mel_spectrogram.cpp)
target_link_libraries(test_mel_spectrogram PRIVATE trtf_core)
target_include_directories(test_mel_spectrogram PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_mel_spectrogram COMMAND test_mel_spectrogram)

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
