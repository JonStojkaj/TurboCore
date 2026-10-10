#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "detail/bit_io.hpp"
#include "detail/codec.hpp"
#include "detail/deflate_tables.hpp"
#include "detail/huffman.hpp"

namespace turbocore::detail {

using namespace deflate;

namespace {

// ---------------------------------------------------------------------------
// LZ77
// ---------------------------------------------------------------------------

/// A literal byte (distance == 0, value = the byte) or a back-reference
/// (value = match length, distance in [1, 32768]).
struct Token {
  std::uint16_t value;
  std::uint16_t distance;
};

struct TokenStream {
  std::vector<Token> tokens;
  std::array<std::uint32_t, kLitLenSymbols> litlen_frequency{};
  std::array<std::uint32_t, kDistSymbols> dist_frequency{};
};

constexpr std::array<std::uint8_t, kMaxMatch + 1> make_length_code_table() noexcept {
  std::array<std::uint8_t, kMaxMatch + 1> table{};
  std::size_t code = 0;
  for (std::size_t length = kMinMatch; length <= kMaxMatch; ++length) {
    while (code + 1 < kLengthCodes && kLengthBase[code + 1] <= length) {
      ++code;
    }
    table[length] = static_cast<std::uint8_t>(code);
  }
  return table;
}
constexpr auto kLengthCode = make_length_code_table();

std::size_t distance_code(std::size_t distance) noexcept {
  const auto it = std::upper_bound(kDistBase.begin(), kDistBase.end(), distance);
  return static_cast<std::size_t>(it - kDistBase.begin()) - 1;
}

constexpr unsigned kHashBits = 15;

std::size_t hash3(const std::uint8_t* p) noexcept {
  const std::uint32_t v =
      static_cast<std::uint32_t>(p[0]) | static_cast<std::uint32_t>(p[1]) << 8 | static_cast<std::uint32_t>(p[2]) << 16;
  return (v * 0x9E3779B1u) >> (32 - kHashBits);
}

/// Greedy LZ77 with hash chains over a 32 KiB window.
TokenStream tokenize(std::span<const std::uint8_t> in, unsigned max_chain) {
  TokenStream stream;
  const std::uint8_t* data = in.data();
  const std::size_t size = in.size();
  stream.tokens.reserve(size / 2 + 16);

  // head[h]: most recent position whose 3-byte hash is h.
  // previous[p % window]: the position before p with the same hash.
  std::vector<std::int32_t> head(std::size_t{1} << kHashBits, -1);
  std::vector<std::int32_t> previous(kWindowSize, -1);

  const auto insert = [&](std::size_t position) {
    if (position + kMinMatch <= size) {
      const std::size_t h = hash3(data + position);
      previous[position % kWindowSize] = head[h];
      head[h] = static_cast<std::int32_t>(position);
    }
  };

  std::size_t position = 0;
  while (position < size) {
    std::size_t best_length = 0;
    std::size_t best_distance = 0;

    if (position + kMinMatch <= size) {
      const std::size_t limit = std::min(kMaxMatch, size - position);
      const std::uint8_t* current = data + position;
      std::int32_t candidate = head[hash3(current)];
      for (unsigned chain = max_chain; chain > 0 && candidate >= 0; --chain) {
        const auto candidate_position = static_cast<std::size_t>(candidate);
        const std::size_t distance = position - candidate_position;
        if (distance > kWindowSize) {
          break;
        }
        const std::uint8_t* match = data + candidate_position;
        // A longer match must agree at the byte that would extend the best one.
        if (best_length < limit && match[best_length] == current[best_length]) {
          std::size_t length = 0;
          while (length < limit && match[length] == current[length]) {
            ++length;
          }
          if (length > best_length) {
            best_length = length;
            best_distance = distance;
            if (length == limit) {
              break;
            }
          }
        }
        candidate = previous[candidate_position % kWindowSize];
      }
    }

    if (best_length >= kMinMatch) {
      const std::size_t length_code = kLengthCode[best_length];
      const std::size_t dist_code = distance_code(best_distance);
      stream.tokens.push_back(
          {static_cast<std::uint16_t>(best_length), static_cast<std::uint16_t>(best_distance)});
      ++stream.litlen_frequency[257 + length_code];
      ++stream.dist_frequency[dist_code];
      for (std::size_t i = 0; i < best_length; ++i) {
        insert(position + i);
      }
      position += best_length;
    } else {
      stream.tokens.push_back({data[position], 0});
      ++stream.litlen_frequency[data[position]];
      insert(position);
      ++position;
    }
  }

  ++stream.litlen_frequency[kEndOfBlock];
  return stream;
}

// ---------------------------------------------------------------------------
// Huffman code construction
// ---------------------------------------------------------------------------

/// Like build_code_lengths, but guarantees at least two coded symbols so the
/// result is always a complete code. Decoders differ in how they treat a
/// one-symbol or empty code; a complete one is accepted by all of them.
template <std::size_t N>
std::array<std::uint8_t, N> complete_code_lengths(std::array<std::uint32_t, N> frequencies, unsigned max_bits) {
  std::size_t used = 0;
  for (const std::uint32_t frequency : frequencies) {
    used += frequency != 0 ? 1u : 0u;
  }
  for (std::size_t symbol = 0; used < 2 && symbol < N; ++symbol) {
    if (frequencies[symbol] == 0) {
      frequencies[symbol] = 1;
      ++used;
    }
  }
  std::array<std::uint8_t, N> lengths{};
  build_code_lengths(frequencies, max_bits, lengths);
  return lengths;
}

template <std::size_t N>
std::uint64_t coded_bits(const std::array<std::uint32_t, N>& frequencies, std::span<const std::uint8_t> lengths) {
  std::uint64_t bits = 0;
  for (std::size_t symbol = 0; symbol < N; ++symbol) {
    bits += std::uint64_t{frequencies[symbol]} * lengths[symbol];
  }
  return bits;
}

// ---------------------------------------------------------------------------
// Dynamic block header (RFC 1951 section 3.2.7)
// ---------------------------------------------------------------------------

constexpr unsigned kMaxCodeLengthBits = 7;

struct CodeLengthSymbol {
  std::uint8_t symbol;      // 0-15 literal length, 16/17/18 repeat codes
  std::uint8_t extra;       // repeat count payload
  std::uint8_t extra_bits;  // width of `extra`
};

struct DynamicHeader {
  std::size_t litlen_count = 0;  // HLIT + 257
  std::size_t dist_count = 0;    // HDIST + 1
  std::size_t code_length_count = 0;  // HCLEN + 4
  std::array<std::uint8_t, kCodeLengthSymbols> code_length_lengths{};
  std::vector<CodeLengthSymbol> symbols;
  std::uint64_t bits = 0;  // size of the header after the 3 block-type bits
};

DynamicHeader make_dynamic_header(std::span<const std::uint8_t> litlen_lengths,
                                  std::span<const std::uint8_t> dist_lengths) {
  DynamicHeader header;

  header.litlen_count = litlen_lengths.size();
  while (header.litlen_count > 257 && litlen_lengths[header.litlen_count - 1] == 0) {
    --header.litlen_count;
  }
  header.dist_count = dist_lengths.size();
  while (header.dist_count > 1 && dist_lengths[header.dist_count - 1] == 0) {
    --header.dist_count;
  }

  // Both length tables are run-length coded as one sequence.
  std::vector<std::uint8_t> sequence(litlen_lengths.begin(),
                                     litlen_lengths.begin() + static_cast<std::ptrdiff_t>(header.litlen_count));
  sequence.insert(sequence.end(), dist_lengths.begin(),
                  dist_lengths.begin() + static_cast<std::ptrdiff_t>(header.dist_count));

  std::array<std::uint32_t, kCodeLengthSymbols> frequency{};
  const auto emit = [&](unsigned symbol, std::size_t extra, unsigned extra_bits) {
    header.symbols.push_back({static_cast<std::uint8_t>(symbol), static_cast<std::uint8_t>(extra),
                              static_cast<std::uint8_t>(extra_bits)});
    ++frequency[symbol];
  };

  for (std::size_t i = 0; i < sequence.size();) {
    const std::uint8_t value = sequence[i];
    std::size_t run = 1;
    while (i + run < sequence.size() && sequence[i + run] == value) {
      ++run;
    }
    i += run;

    if (value == 0) {
      while (run >= 11) {
        const std::size_t count = std::min<std::size_t>(run, 138);
        emit(18, count - 11, 7);
        run -= count;
      }
      if (run >= 3) {
        emit(17, run - 3, 3);
        run = 0;
      }
    } else {
      // Code 16 repeats the previous length, so the first one is explicit.
      emit(value, 0, 0);
      --run;
      while (run >= 3) {
        const std::size_t count = std::min<std::size_t>(run, 6);
        emit(16, count - 3, 2);
        run -= count;
      }
    }
    for (; run > 0; --run) {
      emit(value, 0, 0);
    }
  }

  header.code_length_lengths = complete_code_lengths(frequency, kMaxCodeLengthBits);

  header.code_length_count = kCodeLengthSymbols;
  while (header.code_length_count > 4 &&
         header.code_length_lengths[kCodeLengthOrder[header.code_length_count - 1]] == 0) {
    --header.code_length_count;
  }

  header.bits = 5 + 5 + 4 + 3 * header.code_length_count;
  for (const CodeLengthSymbol& s : header.symbols) {
    header.bits += std::uint64_t{header.code_length_lengths[s.symbol]} + s.extra_bits;
  }
  return header;
}

void write_dynamic_header(BitWriter& writer, const DynamicHeader& header) {
  writer.write(static_cast<std::uint32_t>(header.litlen_count - 257), 5);
  writer.write(static_cast<std::uint32_t>(header.dist_count - 1), 5);
  writer.write(static_cast<std::uint32_t>(header.code_length_count - 4), 4);
  for (std::size_t i = 0; i < header.code_length_count; ++i) {
    writer.write(header.code_length_lengths[kCodeLengthOrder[i]], 3);
  }

  std::array<std::uint16_t, kCodeLengthSymbols> codes{};
  assign_reversed_codes(header.code_length_lengths, codes);
  for (const CodeLengthSymbol& s : header.symbols) {
    writer.write(codes[s.symbol], header.code_length_lengths[s.symbol]);
    if (s.extra_bits != 0) {
      writer.write(s.extra, s.extra_bits);
    }
  }
}

// ---------------------------------------------------------------------------
// Block body
// ---------------------------------------------------------------------------

void write_tokens(BitWriter& writer, std::span<const Token> tokens, std::span<const std::uint8_t> litlen_lengths,
                  std::span<const std::uint8_t> dist_lengths) {
  std::array<std::uint16_t, kFixedLitLenSymbols> litlen_codes{};
  std::array<std::uint16_t, kDistSymbols> dist_codes{};
  assign_reversed_codes(litlen_lengths, std::span(litlen_codes).first(litlen_lengths.size()));
  assign_reversed_codes(dist_lengths, dist_codes);

  for (const Token& token : tokens) {
    if (token.distance == 0) {
      writer.write(litlen_codes[token.value], litlen_lengths[token.value]);
      continue;
    }
    const std::size_t length_code = kLengthCode[token.value];
    writer.write(litlen_codes[257 + length_code], litlen_lengths[257 + length_code]);
    if (kLengthExtraBits[length_code] != 0) {
      writer.write(static_cast<std::uint32_t>(token.value - kLengthBase[length_code]),
                   kLengthExtraBits[length_code]);
    }
    const std::size_t dist_code = distance_code(token.distance);
    writer.write(dist_codes[dist_code], dist_lengths[dist_code]);
    if (kDistExtraBits[dist_code] != 0) {
      writer.write(static_cast<std::uint32_t>(token.distance - kDistBase[dist_code]), kDistExtraBits[dist_code]);
    }
  }
  writer.write(litlen_codes[kEndOfBlock], litlen_lengths[kEndOfBlock]);
}

constexpr unsigned kBlockTypeFixed = 1;
constexpr unsigned kBlockTypeDynamic = 2;

}  // namespace

DeflateCodec::DeflateCodec(int level) noexcept
    : max_chain_(1u << static_cast<unsigned>(std::clamp(level, kMinLevel, kMaxLevel) + 1)) {}

void DeflateCodec::compress(std::span<const std::uint8_t> in, std::vector<std::uint8_t>& out) const {
  const TokenStream stream = tokenize(in, max_chain_);

  const auto litlen_lengths = complete_code_lengths(stream.litlen_frequency, kMaxHuffmanBits);
  const auto dist_lengths = complete_code_lengths(stream.dist_frequency, kMaxHuffmanBits);
  const DynamicHeader header = make_dynamic_header(litlen_lengths, dist_lengths);
  // The length/distance extra bits cost the same under either code, so they
  // are left out of the comparison.
  const std::uint64_t dynamic_bits =
      header.bits + coded_bits(stream.litlen_frequency, litlen_lengths) + coded_bits(stream.dist_frequency, dist_lengths);

  static constexpr auto kFixedLitLen = fixed_litlen_lengths();
  std::array<std::uint8_t, kDistSymbols> fixed_dist{};
  fixed_dist.fill(static_cast<std::uint8_t>(kFixedDistBits));
  const std::uint64_t fixed_bits =
      coded_bits(stream.litlen_frequency, kFixedLitLen) + coded_bits(stream.dist_frequency, fixed_dist);

  BitWriter writer(out);
  writer.write(1, 1);  // BFINAL: this is the only block
  if (fixed_bits <= dynamic_bits) {
    writer.write(kBlockTypeFixed, 2);
    write_tokens(writer, stream.tokens, kFixedLitLen, fixed_dist);
  } else {
    writer.write(kBlockTypeDynamic, 2);
    write_dynamic_header(writer, header);
    write_tokens(writer, stream.tokens, litlen_lengths, dist_lengths);
  }
  writer.flush();
}

}  // namespace turbocore::detail
