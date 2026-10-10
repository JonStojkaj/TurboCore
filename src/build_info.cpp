#include "turbocore/build_info.hpp"

#include <bit>
#include <climits>
#include <limits>

#include "detail/kernels.hpp"

#if TURBOCORE_HAS_CUDA
#include "backends/cuda/probe_cuda.hpp"
#endif

// Platform assumptions the bit-level float code relies on. Failing here is
// preferable to producing silently corrupt streams on an exotic target.
static_assert(CHAR_BIT == 8, "TurboCore requires 8-bit bytes");
static_assert(sizeof(float) == 4, "TurboCore requires 32-bit float");
static_assert(std::numeric_limits<float>::is_iec559, "TurboCore requires IEEE 754 binary32");
static_assert(std::endian::native == std::endian::little, "TurboCore requires a little-endian target");

namespace turbocore {

BuildInfo build_info() noexcept {
  return BuildInfo{
      .version = TURBOCORE_VERSION_STRING,
      .avx2_compiled = TURBOCORE_HAS_AVX2 != 0,
      .cuda_compiled = TURBOCORE_HAS_CUDA != 0,
  };
}

RuntimeSupport runtime_support() noexcept {
  RuntimeSupport support{.avx2_usable = false, .cuda_device_count = 0};
  support.avx2_usable = detail::avx2_kernels() != nullptr;
#if TURBOCORE_HAS_CUDA
  support.cuda_device_count = detail::cuda::device_count();
#endif
  return support;
}

}  // namespace turbocore
