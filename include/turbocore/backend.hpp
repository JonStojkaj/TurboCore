#ifndef TURBOCORE_BACKEND_HPP
#define TURBOCORE_BACKEND_HPP

#include <cstdint>

#include "turbocore/export.hpp"

namespace turbocore {

/// Which implementation of the data-parallel kernels to run.
///
/// Every backend produces bit-identical results: the same compressed bytes,
/// the same quantized values. The choice affects speed only, so `Auto` is
/// right for everything except testing and benchmarking.
enum class Backend : std::uint8_t {
  Auto,    ///< the fastest backend usable on this machine
  Scalar,  ///< portable C++ reference implementation, always available
  Avx2,    ///< x86 AVX2 + F16C + BMI2; needs build support and a capable CPU
  /// The transform stage (quantization, predictor, byte shuffle) on an NVIDIA
  /// GPU, with entropy coding on CPU threads. Needs build support and a
  /// device. Never chosen by Auto: until entropy coding also runs on the
  /// device, the PCIe round trip costs more than these stages take on a CPU.
  Cuda,
};

/// True if `backend` was compiled in and the current CPU can run it.
[[nodiscard]] TURBOCORE_EXPORT bool is_backend_available(Backend backend) noexcept;

/// The concrete backend that Backend::Auto resolves to on this machine
/// (Scalar or Avx2).
[[nodiscard]] TURBOCORE_EXPORT Backend best_available_backend() noexcept;

/// "auto", "scalar", "avx2" or "cuda".
[[nodiscard]] TURBOCORE_EXPORT const char* backend_name(Backend backend) noexcept;

}  // namespace turbocore

#endif  // TURBOCORE_BACKEND_HPP
