#ifndef TURBOCORE_BACKENDS_AVX2_PROBE_AVX2_HPP
#define TURBOCORE_BACKENDS_AVX2_PROBE_AVX2_HPP

namespace turbocore::detail::avx2 {

/// Phase 0 toolchain probe: executes AVX2, F16C and BMI2 instructions and
/// checks their results. Proves that the per-backend compile flags and the
/// runtime dispatch gate work end to end. Only call after
/// cpu_supports_avx2_backend() returned true.
[[nodiscard]] bool toolchain_probe() noexcept;

}  // namespace turbocore::detail::avx2

#endif  // TURBOCORE_BACKENDS_AVX2_PROBE_AVX2_HPP
