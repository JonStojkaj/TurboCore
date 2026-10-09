cmake_minimum_required(VERSION 3.25...4.0)

project(TurboCore
  VERSION 0.1.0
  DESCRIPTION "High-performance FP32 / HDR image compression (scalar, AVX2, CUDA)"
  LANGUAGES CXX
)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(FATAL_ERROR
    "In-source builds are not supported. Use: cmake -S . -B build (or a preset).")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(TURBOCORE_BUILD_TESTS       "Build the GoogleTest unit tests"            ${PROJECT_IS_TOP_LEVEL})
option(TURBOCORE_BUILD_BENCHMARKS  "Build the Google Benchmark throughput suite" OFF)
option(TURBOCORE_BUILD_PYTHON      "Build the pybind11 Python extension"         OFF)
option(TURBOCORE_INSTALL           "Generate install and package-config rules"   ${PROJECT_IS_TOP_LEVEL})
option(TURBOCORE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors"          OFF)
option(TURBOCORE_ENABLE_SANITIZERS "Build everything with ASan + UBSan"          OFF)
option(TURBOCORE_USE_SYSTEM_DEPS
  "Use find_package() for GTest/benchmark/pybind11 instead of the pinned FetchContent copies" OFF)

# Backends are tri-state: AUTO enables the backend when the toolchain supports
# it, ON makes a missing toolchain a hard configure error, OFF never builds it.
set(TURBOCORE_ENABLE_AVX2 "AUTO" CACHE STRING "Build the AVX2 backend (AUTO, ON, OFF)")
set(TURBOCORE_ENABLE_CUDA "AUTO" CACHE STRING "Build the CUDA backend (AUTO, ON, OFF)")
set(TURBOCORE_CUDA_STANDARD "17" CACHE STRING "C++ standard used for CUDA translation units")
foreach(_backend_option IN ITEMS TURBOCORE_ENABLE_AVX2 TURBOCORE_ENABLE_CUDA)
  set_property(CACHE ${_backend_option} PROPERTY STRINGS AUTO ON OFF)
  if(NOT "${${_backend_option}}" MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR
      "${_backend_option} must be AUTO, ON or OFF (got '${${_backend_option}}').")
  endif()
endforeach()

if(TURBOCORE_ENABLE_SANITIZERS AND TURBOCORE_BUILD_PYTHON)
  message(FATAL_ERROR
    "TURBOCORE_ENABLE_SANITIZERS and TURBOCORE_BUILD_PYTHON cannot be combined: "
    "a sanitized extension module cannot be loaded by a non-sanitized interpreter.")
endif()

# ---------------------------------------------------------------------------
# Global build settings
# ---------------------------------------------------------------------------
get_property(_turbocore_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(PROJECT_IS_TOP_LEVEL AND NOT _turbocore_multi_config AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
  message(STATUS "TurboCore: CMAKE_BUILD_TYPE not set, defaulting to Release")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# Sanitizers are applied globally so that fetched dependencies are instrumented
# with the same runtime as the library (mixing causes false positives).
if(TURBOCORE_ENABLE_SANITIZERS)
  if(MSVC)
    add_compile_options(/fsanitize=address)
  else()
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
    add_link_options(-fsanitize=address,undefined)
  endif()
endif()

include(TurboCoreCompilerOptions)
include(TurboCoreDependencies)
include(TurboCoreSIMD)
include(TurboCoreCUDA)

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
add_subdirectory(src)

if(TURBOCORE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(TURBOCORE_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(TURBOCORE_BUILD_PYTHON)
  if(TURBOCORE_BUILD_TESTS)
    enable_testing()
  endif()
  add_subdirectory(python)
endif()

if(TURBOCORE_INSTALL)
  include(TurboCoreInstall)
endif()

# ---------------------------------------------------------------------------
# Configuration summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "TurboCore ${PROJECT_VERSION} configuration")
message(STATUS "  Compiler ........ ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build type ...... ${CMAKE_BUILD_TYPE}")
get_target_property(_turbocore_library_type turbocore TYPE)
message(STATUS "  Library type .... ${_turbocore_library_type}")
message(STATUS "  Scalar backend .. ON (always)")
message(STATUS "  AVX2 backend .... ${TURBOCORE_HAS_AVX2} (requested: ${TURBOCORE_ENABLE_AVX2})")
message(STATUS "  CUDA backend .... ${TURBOCORE_HAS_CUDA} (requested: ${TURBOCORE_ENABLE_CUDA})")
message(STATUS "  Tests ........... ${TURBOCORE_BUILD_TESTS}")
message(STATUS "  Benchmarks ...... ${TURBOCORE_BUILD_BENCHMARKS}")
message(STATUS "  Python module ... ${TURBOCORE_BUILD_PYTHON}")
message(STATUS "  Sanitizers ...... ${TURBOCORE_ENABLE_SANITIZERS}")
message(STATUS "")
