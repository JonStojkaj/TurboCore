set(_turbocore_generated_include "${PROJECT_BINARY_DIR}/generated/include")

configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/config.hpp.in"
  "${_turbocore_generated_include}/turbocore/config.hpp"
  @ONLY
)

# Library type follows BUILD_SHARED_LIBS (static by default).
add_library(turbocore)
add_library(TurboCore::turbocore ALIAS turbocore)

include(GenerateExportHeader)
generate_export_header(turbocore
  BASE_NAME TURBOCORE
  EXPORT_FILE_NAME "${_turbocore_generated_include}/turbocore/export.hpp"
)

target_sources(turbocore
  PRIVATE
    build_info.cpp
    compressor.cpp
    container.cpp
    cpu_features.cpp
    crc32c.cpp
    decompressor.cpp
    deflate_decoder.cpp
    deflate_encoder.cpp
    error_analysis.cpp
    huffman.cpp
    quantization.cpp
    transforms.cpp
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS
      "${PROJECT_SOURCE_DIR}/include"
      "${_turbocore_generated_include}"
    FILES
      "${PROJECT_SOURCE_DIR}/include/turbocore/build_info.hpp"
      "${PROJECT_SOURCE_DIR}/include/turbocore/compressor.hpp"
      "${PROJECT_SOURCE_DIR}/include/turbocore/decompressor.hpp"
      "${PROJECT_SOURCE_DIR}/include/turbocore/error.hpp"
      "${PROJECT_SOURCE_DIR}/include/turbocore/error_analysis.hpp"
      "${PROJECT_SOURCE_DIR}/include/turbocore/quantization.hpp"
      "${PROJECT_SOURCE_DIR}/include/turbocore/turbocore.hpp"
      "${_turbocore_generated_include}/turbocore/config.hpp"
      "${_turbocore_generated_include}/turbocore/export.hpp"
)
target_include_directories(turbocore PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(turbocore PUBLIC cxx_std_20)
set_target_properties(turbocore PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
)
turbocore_configure_target(turbocore)

# Backends are object libraries folded into turbocore. Their symbols are
# internal (hidden visibility) and only reached through runtime dispatch, so
# each backend can carry its own compile flags without leaking them into the
# rest of the library or into consumers.
function(_turbocore_add_backend name)
  add_library(${name} OBJECT ${ARGN})
  target_include_directories(${name} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/include"
    "${_turbocore_generated_include}"
  )
  if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${name} PRIVATE TURBOCORE_STATIC_DEFINE)
  endif()
  turbocore_configure_target(${name})
  target_sources(turbocore PRIVATE $<TARGET_OBJECTS:${name}>)
endfunction()

if(TURBOCORE_HAS_AVX2)
  _turbocore_add_backend(turbocore_avx2
    backends/avx2/probe_avx2.cpp
  )
  target_compile_features(turbocore_avx2 PRIVATE cxx_std_20)
  target_compile_options(turbocore_avx2 PRIVATE ${TURBOCORE_AVX2_FLAGS})
endif()

if(TURBOCORE_HAS_CUDA)
  _turbocore_add_backend(turbocore_cuda
    backends/cuda/probe_cuda.cu
  )
  target_link_libraries(turbocore_cuda PRIVATE CUDA::cudart)
  target_link_libraries(turbocore PRIVATE CUDA::cudart)
endif()
