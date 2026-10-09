# AVX2 backend detection.
#
# Outputs:
#   TURBOCORE_HAS_AVX2    ON if the AVX2 backend will be built
#   TURBOCORE_AVX2_FLAGS  compile options for AVX2 translation units (list)
#
# Detection is a compile test rather than a CMAKE_SYSTEM_PROCESSOR match: the
# processor string is wrong for cross-architecture builds (for example
# CMAKE_OSX_ARCHITECTURES=x86_64 on Apple Silicon), while the compile test
# asks the actual target toolchain.
#
# The flags are applied ONLY to the AVX2 object library, never to the whole
# library, so the binary still runs on CPUs without AVX2 and the backend is
# selected at runtime via CPUID.
#
# The "AVX2 backend" targets the Haswell feature set: AVX2 + F16C (hardware
# FP32<->FP16 conversion) + BMI1/BMI2/LZCNT (bitstream packing). FMA is
# deliberately excluded so the compiler cannot contract a*b+c and break
# bit-exact parity with the scalar backend.

include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

set(TURBOCORE_HAS_AVX2 OFF)
set(TURBOCORE_AVX2_FLAGS "")

if(NOT TURBOCORE_ENABLE_AVX2 STREQUAL "OFF")
  if(MSVC)
    set(_turbocore_avx2_flags /arch:AVX2)
  else()
    set(_turbocore_avx2_flags -mavx2 -mf16c -mbmi -mbmi2 -mlzcnt)
  endif()

  cmake_push_check_state(RESET)
  list(JOIN _turbocore_avx2_flags " " CMAKE_REQUIRED_FLAGS)
  check_cxx_source_compiles([[
    #include <immintrin.h>
    int main() {
      __m256i v = _mm256_set1_epi32(1);
      v = _mm256_add_epi32(v, v);
      const __m128i half = _mm256_cvtps_ph(_mm256_setzero_ps(),
                                           _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
      const unsigned packed = _pext_u32(0xF0F0u, 0x0FF0u);
      return _mm256_extract_epi32(v, 0) + _mm_cvtsi128_si32(half) + static_cast<int>(packed);
    }
  ]] TURBOCORE_COMPILER_SUPPORTS_AVX2)
  cmake_pop_check_state()

  if(TURBOCORE_COMPILER_SUPPORTS_AVX2)
    set(TURBOCORE_HAS_AVX2 ON)
    set(TURBOCORE_AVX2_FLAGS ${_turbocore_avx2_flags})
  elseif(TURBOCORE_ENABLE_AVX2 STREQUAL "ON")
    message(FATAL_ERROR
      "TURBOCORE_ENABLE_AVX2=ON but the toolchain cannot compile AVX2/F16C/BMI2 code "
      "for this target (${CMAKE_SYSTEM_PROCESSOR}; CMAKE_OSX_ARCHITECTURES='${CMAKE_OSX_ARCHITECTURES}'). "
      "Use AUTO to build without the AVX2 backend.")
  endif()
endif()
