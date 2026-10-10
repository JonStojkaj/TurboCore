#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "detail/bit_io.hpp"
#include "detail/codec.hpp"
#include "detail/deflate_tables.hpp"
#include "detail/huffman.hpp"

namespace turbocore::detail {

using namespace deflate;

namespace {

struct BlockDecoders {
  HuffmanDecoder litlen;
  HuffmanDecoder dist;
};

const BlockDecoders& fixed_decoders() noexcept {
  static const BlockDecoders decoders = [] {
    BlockDecoders d;
    static constexpr auto kFixedLitLen = fixed_litlen_lengths();
    std::array<std::uint8_t, kDistSymbols> dist_lengths{};
    dist_lengths.fill(static_cast<std::uint8_t>(kFixedDistBits));
    // Both fixed codes are valid by construction.
    (void)d.litlen.build(kFixedLitLen);
    (void)d.dist.build(dist_lengths);
    return d;
  }();
  return decoders;
}

/// Reads the code-length tables of a dynamic block (RFC 1951 section 3.2.7).
bool read_dynamic_decoders(BitReader& reader, BlockDecoders& decoders) noexcept {
  const std::size_t litlen_count = reader.read(5) + 257;
  const std::size_t dist_count = reader.read(5) + 1;
  const std::size_t code_length_count = reader.read(4) + 4;
  if (reader.overrun() || litlen_count > kLitLenSymbols || dist_count > kDistSymbols) {
    return false;
  }

  std::array<std::uint8_t, kCodeLengthSymbols> code_length_lengths{};
  for (std::size_t i = 0; i < code_length_count; ++i) {
    code_length_lengths[kCodeLengthOrder[i]] = static_cast<std::uint8_t>(reader.read(3));
  }
  HuffmanDecoder code_length_decoder;
  if (reader.overrun() || !code_length_decoder.build(code_length_lengths)) {
    return false;
  }

  std::array<std::uint8_t, kLitLenSymbols + kDistSymbols> lengths{};
  const std::size_t total = litlen_count + dist_count;
  for (std::size_t i = 0; i < total;) {
    const int symbol = code_length_decoder.decode(reader);
    if (symbol < 0 || reader.overrun()) {
      return false;
    }
    if (symbol < 16) {
      lengths[i++] = static_cast<std::uint8_t>(symbol);
      continue;
    }

    std::uint8_t value = 0;
    std::size_t repeat = 0;
    if (symbol == 16) {
      if (i == 0) {
        return false;  // nothing to repeat
      }
      value = lengths[i - 1];
      repeat = 3 + reader.read(2);
    } else if (symbol == 17) {
      repeat = 3 + reader.read(3);
    } else {
      repeat = 11 + reader.read(7);
    }
    if (reader.overrun() || repeat > total - i) {
      return false;
    }
    for (; repeat > 0; --repeat) {
      lengths[i++] = value;
    }
  }

  if (lengths[kEndOfBlock] == 0) {
    return false;  // the block could never terminate
  }
  const std::span<const std::uint8_t> all(lengths);
  return decoders.litlen.build(all.first(litlen_count)) && decoders.dist.build(all.subspan(litlen_count, dist_count));
}

/// Decodes the symbols of one Huffman-coded block into out[position...].
bool inflate_block(BitReader& reader, const BlockDecoders& decoders, std::span<std::uint8_t> out,
                   std::size_t& position) noexcept {
  for (;;) {
    const int symbol = decoders.litlen.decode(reader);
    if (symbol < 0 || reader.overrun()) {
      return false;
    }
    if (symbol < static_cast<int>(kEndOfBlock)) {
      if (position == out.size()) {
        return false;
      }
      out[position++] = static_cast<std::uint8_t>(symbol);
      continue;
    }
    if (symbol == static_cast<int>(kEndOfBlock)) {
      return true;
    }

    const auto length_code = static_cast<std::size_t>(symbol) - 257;
    if (length_code >= kLengthCodes) {
      return false;
    }
    const std::size_t length = kLengthBase[length_code] + reader.read(kLengthExtraBits[length_code]);

    const int dist_symbol = decoders.dist.decode(reader);
    if (dist_symbol < 0 || static_cast<std::size_t>(dist_symbol) >= kDistSymbols) {
      return false;
    }
    const auto dist_code = static_cast<std::size_t>(dist_symbol);
    const std::size_t distance = kDistBase[dist_code] + reader.read(kDistExtraBits[dist_code]);

    if (reader.overrun() || distance > position || length > out.size() - position) {
      return false;
    }
    // Byte by byte: source and destination overlap when distance < length,
    // which is how DEFLATE expresses runs.
    for (std::size_t i = 0; i < length; ++i, ++position) {
      out[position] = out[position - distance];
    }
  }
}

bool inflate_stored(BitReader& reader, std::span<std::uint8_t> out, std::size_t& position) noexcept {
  reader.align_to_byte();
  const std::uint32_t length = reader.read(16);
  const std::uint32_t length_complement = reader.read(16);
  if (reader.overrun() || (length ^ 0xFFFFu) != length_complement || length > out.size() - position) {
    return false;
  }
  for (std::uint32_t i = 0; i < length; ++i) {
    out[position++] = static_cast<std::uint8_t>(reader.read(8));
  }
  return !reader.overrun();
}

}  // namespace

bool DeflateCodec::decompress(std::span<const std::uint8_t> in, std::span<std::uint8_t> out) const noexcept {
  BitReader reader(in);
  std::size_t position = 0;

  bool final_block = false;
  while (!final_block) {
    final_block = reader.read(1) != 0;
    const std::uint32_t block_type = reader.read(2);
    if (reader.overrun()) {
      return false;
    }

    bool ok = false;
    switch (block_type) {
      case 0:
        ok = inflate_stored(reader, out, position);
        break;
      case 1:
        ok = inflate_block(reader, fixed_decoders(), out, position);
        break;
      case 2: {
        BlockDecoders decoders;
        ok = read_dynamic_decoders(reader, decoders) && inflate_block(reader, decoders, out, position);
        break;
      }
      default:
        break;  // block type 3 is reserved
    }
    if (!ok) {
      return false;
    }
  }

  return position == out.size() && reader.bytes_consumed() == in.size();
}

}  // namespace turbocore::detail
