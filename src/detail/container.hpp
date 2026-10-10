#ifndef TURBOCORE_DETAIL_CONTAINER_HPP
#define TURBOCORE_DETAIL_CONTAINER_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include "turbocore/decompressor.hpp"

// TurboCore container format. All integers are little-endian.
//
// Version 1 stores FP32 elements only. Version 2 has the same layout and adds
// the 16-bit element types. The writer emits the lowest version that can
// describe the stream, so lossless streams stay readable by version-1 readers.
//
//   Header (32 bytes)
//     0   4  magic "TCOR"
//     4   1  format version (1 or 2)
//     5   1  element type (ElementType); version 1 allows only Float32
//     6   1  predictor (turbocore::Predictor)
//     7   1  flags (bit 0: byte shuffle; other bits zero)
//     8   4  block_elements: elements per block
//     12  8  element_count
//     20  4  block_count = ceil(element_count / block_elements)
//     24  4  reserved, zero
//     28  4  CRC-32C of bytes [0, 28)
//
//   Block index (block_count entries of 12 bytes)
//     0   4  compressed size of the block in bytes
//     4   4  CRC-32C of the block's stored elements before predictor and
//            shuffle: the original FP32 data, or the quantized 16-bit values
//     8   1  method (BlockMethod)
//     9   3  reserved, zero
//
//   Block payloads, concatenated in order. A block's offset is the sum of the
//   compressed sizes before it; every block holds block_elements elements
//   except possibly the last.

namespace turbocore::detail {

inline constexpr std::array<std::uint8_t, 4> kMagic = {'T', 'C', 'O', 'R'};
inline constexpr std::uint8_t kFormatVersionFloat32 = 1;
inline constexpr std::uint8_t kFormatVersionQuantized = 2;
inline constexpr std::uint8_t kFlagByteShuffle = 0x01;

inline constexpr std::size_t kHeaderSize = 32;
inline constexpr std::size_t kHeaderCrcOffset = 28;
inline constexpr std::size_t kIndexEntrySize = 12;
inline constexpr std::size_t kMaxBlockElements = std::size_t{1} << 24;

/// The largest expansion a DEFLATE stream can achieve is 258 bytes per
/// 2 bits, i.e. 1032:1. Used to reject headers that claim more data than
/// the payload could possibly hold, before anything is allocated.
inline constexpr std::uint64_t kMaxExpansionRatio = 1032;

enum class ElementType : std::uint8_t {
  Float32 = 0,
  Float16 = 1,
  BFloat16 = 2,
};

[[nodiscard]] constexpr ElementType element_type_of(const std::optional<QuantFormat>& quantization) noexcept {
  if (!quantization.has_value()) {
    return ElementType::Float32;
  }
  return *quantization == QuantFormat::Float16 ? ElementType::Float16 : ElementType::BFloat16;
}

/// Bytes per stored element.
[[nodiscard]] constexpr std::size_t element_size_of(const std::optional<QuantFormat>& quantization) noexcept {
  return quantization.has_value() ? 2 : 4;
}

enum class BlockMethod : std::uint8_t {
  Stored = 0,   ///< transformed bytes, verbatim
  Deflate = 1,  ///< transformed bytes as one raw DEFLATE stream
};

template <std::unsigned_integral T>
[[nodiscard]] T load_le(const std::uint8_t* p) noexcept {
  std::uint64_t value = 0;
  for (std::size_t i = 0; i < sizeof(T); ++i) {
    value |= static_cast<std::uint64_t>(p[i]) << (8 * i);
  }
  return static_cast<T>(value);
}

template <std::unsigned_integral T>
void store_le(std::uint8_t* p, T value) noexcept {
  for (std::size_t i = 0; i < sizeof(T); ++i) {
    p[i] = static_cast<std::uint8_t>((static_cast<std::uint64_t>(value) >> (8 * i)) & 0xFFu);
  }
}

struct IndexEntry {
  std::uint32_t compressed_size;
  std::uint32_t crc;
  BlockMethod method;
};

/// Writes the 32-byte header, including its CRC, to `out`.
void write_header(std::span<std::uint8_t, kHeaderSize> out, const StreamInfo& info) noexcept;

void write_index_entry(std::uint8_t* out, const IndexEntry& entry) noexcept;

/// Validates the header and the size of the index against the stream length.
/// Throws turbocore::Error.
[[nodiscard]] StreamInfo parse_header(std::span<const std::uint8_t> stream);

/// Throws turbocore::Error(CorruptStream) on an unknown method or non-zero
/// reserved bytes.
[[nodiscard]] IndexEntry parse_index_entry(const std::uint8_t* in);

}  // namespace turbocore::detail

#endif  // TURBOCORE_DETAIL_CONTAINER_HPP
