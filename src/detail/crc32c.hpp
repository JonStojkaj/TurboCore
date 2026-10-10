#ifndef TURBOCORE_DETAIL_CRC32C_HPP
#define TURBOCORE_DETAIL_CRC32C_HPP

#include <cstdint>
#include <span>

namespace turbocore::detail {

/// CRC-32C (Castagnoli, the polynomial with hardware support on x86 and Arm).
/// The scalar backend's implementation: Armv8 CRC instructions where the
/// compile target has them, crc32c_portable() otherwise.
[[nodiscard]] std::uint32_t crc32c(std::span<const std::uint8_t> data) noexcept;

/// Table-driven (slicing-by-8) implementation in portable C++; the
/// definition the hardware paths are tested against.
[[nodiscard]] std::uint32_t crc32c_portable(std::span<const std::uint8_t> data) noexcept;

}  // namespace turbocore::detail

#endif  // TURBOCORE_DETAIL_CRC32C_HPP
