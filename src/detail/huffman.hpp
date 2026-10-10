#ifndef TURBOCORE_DETAIL_HUFFMAN_HPP
#define TURBOCORE_DETAIL_HUFFMAN_HPP

#include <array>
#include <cstdint>
#include <span>

#include "detail/bit_io.hpp"

namespace turbocore::detail {

inline constexpr unsigned kMaxHuffmanBits = 15;
inline constexpr std::size_t kMaxHuffmanSymbols = 288;

/// Computes prefix-code lengths for `frequencies`, none longer than
/// `max_bits`. Symbols with frequency 0 get length 0. The result is a
/// complete code, except that a single used symbol gets length 1.
/// Requires lengths.size() == frequencies.size() and enough room in
/// 2^max_bits for the used symbols.
void build_code_lengths(std::span<const std::uint32_t> frequencies, unsigned max_bits,
                        std::span<std::uint8_t> lengths);

/// Assigns canonical codes (RFC 1951 section 3.2.2) for `lengths`. Each code
/// is stored bit-reversed so it can be handed to an LSB-first BitWriter.
void assign_reversed_codes(std::span<const std::uint8_t> lengths, std::span<std::uint16_t> codes);

/// Canonical Huffman decoder for alphabets of up to 288 symbols and codes of
/// up to 15 bits: a direct lookup for short codes, a canonical walk for the
/// rare long ones.
class HuffmanDecoder {
 public:
  /// Returns false if the lengths over-subscribe the code space. Incomplete
  /// codes are accepted; their unused bit patterns decode as an error.
  [[nodiscard]] bool build(std::span<const std::uint8_t> lengths) noexcept;

  /// Decodes one symbol, or returns -1 on an unassigned bit pattern. The
  /// caller must check reader.overrun() before trusting the symbol.
  [[nodiscard]] int decode(BitReader& reader) const noexcept;

 private:
  static constexpr unsigned kFastBits = 10;

  // (symbol << 4) | length for codes of at most kFastBits bits; 0 = not here.
  std::array<std::uint16_t, 1u << kFastBits> fast_{};
  std::array<std::uint16_t, kMaxHuffmanBits + 1> count_{};
  std::array<std::uint16_t, kMaxHuffmanSymbols> symbols_{};
};

}  // namespace turbocore::detail

#endif  // TURBOCORE_DETAIL_HUFFMAN_HPP
