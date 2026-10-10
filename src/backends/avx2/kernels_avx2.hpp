#ifndef TURBOCORE_BACKENDS_AVX2_KERNELS_AVX2_HPP
#define TURBOCORE_BACKENDS_AVX2_KERNELS_AVX2_HPP

#include "detail/kernel_table.hpp"

namespace turbocore::detail::avx2 {

/// The AVX2 kernel table. Its functions may only be called after
/// cpu_supports_avx2_backend() returned true; use detail::avx2_kernels().
[[nodiscard]] const KernelTable& kernel_table() noexcept;

}  // namespace turbocore::detail::avx2

#endif  // TURBOCORE_BACKENDS_AVX2_KERNELS_AVX2_HPP
