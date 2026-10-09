#ifndef TURBOCORE_DETAIL_CPU_FEATURES_HPP
#define TURBOCORE_DETAIL_CPU_FEATURES_HPP

namespace turbocore::detail {

/// True if the CPU and the OS support everything the AVX2 backend is compiled
/// with: AVX2, F16C, BMI1, BMI2 and LZCNT, with YMM state enabled by the OS.
/// Always false on non-x86 targets.
[[nodiscard]] bool cpu_supports_avx2_backend() noexcept;

}  // namespace turbocore::detail

#endif  // TURBOCORE_DETAIL_CPU_FEATURES_HPP
