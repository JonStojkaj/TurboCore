# CUDA backend detection.
#
# Outputs:
#   TURBOCORE_HAS_CUDA  ON if the CUDA backend will be built
#
# check_language() probes for nvcc without failing the configure, so AUTO
# degrades cleanly on machines without the CUDA Toolkit (macOS, CI runners).
# This file must be include()d from the top-level CMakeLists.txt because
# enable_language() has to run in the directory scope shared by all targets.

set(TURBOCORE_HAS_CUDA OFF)

if(NOT TURBOCORE_ENABLE_CUDA STREQUAL "OFF")
  include(CheckLanguage)
  check_language(CUDA)

  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit 11.0 QUIET)
    if(CUDAToolkit_FOUND)
      set(TURBOCORE_HAS_CUDA ON)
      set(CMAKE_CUDA_STANDARD ${TURBOCORE_CUDA_STANDARD})
      set(CMAKE_CUDA_STANDARD_REQUIRED ON)
      set(CMAKE_CUDA_EXTENSIONS OFF)
      set(CMAKE_CUDA_VISIBILITY_PRESET hidden)
      message(STATUS "TurboCore: CUDA Toolkit ${CUDAToolkit_VERSION} (${CMAKE_CUDA_COMPILER})")
    endif()
  endif()

  if(NOT TURBOCORE_HAS_CUDA AND TURBOCORE_ENABLE_CUDA STREQUAL "ON")
    message(FATAL_ERROR
      "TURBOCORE_ENABLE_CUDA=ON but no usable CUDA Toolkit (>= 11.0) was found. "
      "Set CUDAToolkit_ROOT / CMAKE_CUDA_COMPILER, or use AUTO to build without CUDA.")
  endif()
endif()
