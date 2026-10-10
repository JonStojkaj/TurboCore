#ifndef TURBOCORE_DETAIL_CRC32C_HPP
#define TURBOCORE_DETAIL_CRC32C_HPP

#include <cstdint>
#include <span>

namespace turbocore::detail {

/// CRC-32C (Castagnoli, the polynomial with hardware support on x86 and Arm).
/// Scalar slicing-by-8 reference implementation.
[[nodiscard]] std::uint32_t crc32c(std::span<const std::uint8_t> data) noexcept;

}  // namespace turbocore::detail

#endif  // TURBOCORE_DETAIL_CRC32C_HPP
