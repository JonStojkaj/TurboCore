// Compiled with the AVX2 backend flags. Keep the includes of this directory
// minimal and every helper in an anonymous namespace: an inline function from
// a shared header that gets instantiated here is emitted with AVX2
// instructions, and the linker may pick that copy for the scalar code path.

#include "backends/avx2/probe_avx2.hpp"

#include <immintrin.h>

#include <cstdint>

namespace turbocore::detail::avx2 {

bool toolchain_probe() noexcept {
  // volatile keeps the compiler from folding the whole probe at compile time.
  volatile std::uint32_t seed = 1;
  const std::uint32_t one = seed;

  // AVX2: 8 lanes of (1 + 2) summed.
  const __m256i a = _mm256_set1_epi32(static_cast<int>(one));
  const __m256i b = _mm256_set1_epi32(static_cast<int>(one * 2));
  alignas(32) std::uint32_t lanes[8];
  _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), _mm256_add_epi32(a, b));
  std::uint32_t sum = 0;
  for (const std::uint32_t lane : lanes) {
    sum += lane;
  }

  // F16C: 1.0f is 0x3C00 in binary16.
  const __m128i half = _mm256_cvtps_ph(_mm256_set1_ps(static_cast<float>(one)),
                                       _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  const auto half_bits = static_cast<std::uint32_t>(_mm_extract_epi16(half, 0));

  // BMI2: gather the bits selected by the mask.
  const std::uint32_t packed = _pext_u32(0xF0F0u * one, 0x0FF0u);

  return sum == 24 && half_bits == 0x3C00u && packed == 0x0Fu;
}

}  // namespace turbocore::detail::avx2
