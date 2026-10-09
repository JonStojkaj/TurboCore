# Third-party dependencies (test, benchmark and binding tooling only -- the
# library itself has no external dependencies).
#
# By default every dependency is a pinned, hash-verified FetchContent download
# so that a build does not depend on what happens to be installed on the host.
# Set TURBOCORE_USE_SYSTEM_DEPS=ON to use find_package() instead (offline
# builds, distro packaging). There is deliberately no silent fallback between
# the two modes.
#
# Each macro is invoked from the subdirectory that needs the dependency, so a
# dependency is only resolved when the corresponding component is enabled.

include(FetchContent)

macro(turbocore_require_googletest)
  if(NOT TARGET GTest::gtest_main)
    if(TURBOCORE_USE_SYSTEM_DEPS)
      find_package(GTest REQUIRED)
    else()
      set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
      set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
      # Match the CRT of the project on MSVC instead of GoogleTest's static default.
      set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
      FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
        URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
        SYSTEM
      )
      FetchContent_MakeAvailable(googletest)
    endif()
  endif()
endmacro()

macro(turbocore_require_benchmark)
  if(NOT TARGET benchmark::benchmark)
    if(TURBOCORE_USE_SYSTEM_DEPS)
      find_package(benchmark REQUIRED)
    else()
      set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
      set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
      set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
      set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
      set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
      FetchContent_Declare(benchmark
        URL https://github.com/google/benchmark/archive/refs/tags/v1.9.4.tar.gz
        URL_HASH SHA256=b334658edd35efcf06a99d9be21e4e93e092bd5f95074c1673d5c8705d95c104
        SYSTEM
      )
      FetchContent_MakeAvailable(benchmark)
    endif()
  endif()
endmacro()

macro(turbocore_require_pybind11)
  if(NOT TARGET pybind11::module)
    if(TURBOCORE_USE_SYSTEM_DEPS)
      find_package(pybind11 CONFIG REQUIRED)
    else()
      set(PYBIND11_INSTALL OFF CACHE BOOL "" FORCE)
      set(PYBIND11_TEST OFF CACHE BOOL "" FORCE)
      FetchContent_Declare(pybind11
        URL https://github.com/pybind/pybind11/archive/refs/tags/v3.0.1.tar.gz
        URL_HASH SHA256=741633da746b7c738bb71f1854f957b9da660bcd2dce68d71949037f0969d0ca
        SYSTEM
      )
      FetchContent_MakeAvailable(pybind11)
    endif()
  endif()
endmacro()
