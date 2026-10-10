#ifndef TURBOCORE_DETAIL_DEFLATE_TABLES_HPP
#define TURBOCORE_DETAIL_DEFLATE_TABLES_HPP

#include <array>
#include <cstddef>
#include <cstdint>

// Constants of the DEFLATE format, RFC 1951 section 3.2.

namespace turbocore::detail::deflate {

inline constexpr std::size_t kWindowSize = 32768;
inline constexpr std::size_t kMinMatch = 3;
inline constexpr std::size_t kMaxMatch = 258;

inline constexpr unsigned kEndOfBlock = 256;
inline constexpr std::size_t kLitLenSymbols = 286;       // usable literal/length symbols
inline constexpr std::size_t kFixedLitLenSymbols = 288;  // the fixed code also defines 286 and 287
inline constexpr std::size_t kDistSymbols = 30;
inline constexpr std::size_t kCodeLengthSymbols = 19;
inline constexpr std::size_t kLengthCodes = 29;

inline constexpr std::array<std::uint16_t, kLengthCodes> kLengthBase = {
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
inline constexpr std::array<std::uint8_t, kLengthCodes> kLengthExtraBits = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};

inline constexpr std::array<std::uint16_t, kDistSymbols> kDistBase = {
    1,   2,   3,   4,   5,   7,    9,    13,   17,   25,   33,   49,   65,    97,    129,
    193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
inline constexpr std::array<std::uint8_t, kDistSymbols> kDistExtraBits = {
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

/// Order in which code-length code lengths are transmitted.
inline constexpr std::array<std::uint8_t, kCodeLengthSymbols> kCodeLengthOrder = {
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

inline constexpr unsigned kFixedDistBits = 5;

/// Code lengths of the fixed literal/length code (BTYPE = 01).
constexpr std::array<std::uint8_t, kFixedLitLenSymbols> fixed_litlen_lengths() noexcept {
  std::array<std::uint8_t, kFixedLitLenSymbols> lengths{};
  for (std::size_t symbol = 0; symbol < lengths.size(); ++symbol) {
    lengths[symbol] = symbol < 144 ? 8 : symbol < 256 ? 9 : symbol < 280 ? 7 : 8;
  }
  return lengths;
}

}  // namespace turbocore::detail::deflate

#endif  // TURBOCORE_DETAIL_DEFLATE_TABLES_HPP
