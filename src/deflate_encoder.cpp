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

/// The tokens of one DEFLATE block and the statistics needed to cost it.
/// The end-of-block symbol is not counted here.
struct TokenStream {
  std::vector<Token> tokens;
  std::array<std::uint32_t, kLitLenSymbols> litlen_frequency{};
  std::array<std::uint32_t, kDistSymbols> dist_frequency{};
  std::uint64_t extra_bits = 0;  // length + distance extra bits, independent of the Huffman codes

  void clear() noexcept {
    tokens.clear();
    litlen_frequency.fill(0);
    dist_frequency.fill(0);
    extra_bits = 0;
  }
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

/// Greedy LZ77 with hash chains over a 32 KiB window. The input can be
/// tokenized piecewise; the match history carries over between calls, so a
/// later piece can reference any earlier piece that was tokenized (pieces
/// that were never passed to tokenize() are simply not in the history).
class Matcher {
 public:
  Matcher(std::span<const std::uint8_t> in, unsigned max_chain)
      : data_(in.data()),
        size_(in.size()),
        max_chain_(max_chain),
        head_(std::size_t{1} << kHashBits, -1),
        previous_(kWindowSize, -1) {}

  /// Forgets all history.
  void reset() noexcept {
    std::fill(head_.begin(), head_.end(), -1);
    std::fill(previous_.begin(), previous_.end(), -1);
  }

  /// Appends tokens covering exactly [begin, end) to `stream`. Pieces must be
  /// tokenized in increasing order between resets.
  void tokenize(std::size_t begin, std::size_t end, TokenStream& stream) {
    std::size_t position = begin;
    while (position < end) {
      std::size_t best_length = 0;
      std::size_t best_distance = 0;

      if (position + kMinMatch <= end) {
        const std::size_t limit = std::min(kMaxMatch, end - position);
        const std::uint8_t* current = data_ + position;
        std::int32_t candidate = head_[hash3(current)];
        for (unsigned chain = max_chain_; chain > 0 && candidate >= 0; --chain) {
          const auto candidate_position = static_cast<std::size_t>(candidate);
          const std::size_t distance = position - candidate_position;
          if (distance > kWindowSize) {
            break;
          }
          const std::uint8_t* match = data_ + candidate_position;
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
          candidate = previous_[candidate_position % kWindowSize];
        }
      }

      if (best_length >= kMinMatch) {
        const std::size_t length_code = kLengthCode[best_length];
        const std::size_t dist_code = distance_code(best_distance);
        stream.tokens.push_back(
            {static_cast<std::uint16_t>(best_length), static_cast<std::uint16_t>(best_distance)});
        ++stream.litlen_frequency[257 + length_code];
        ++stream.dist_frequency[dist_code];
        stream.extra_bits += std::uint64_t{kLengthExtraBits[length_code]} + kDistExtraBits[dist_code];
        for (std::size_t i = 0; i < best_length; ++i) {
          insert(position + i);
        }
        position += best_length;
      } else {
        stream.tokens.push_back({data_[position], 0});
        ++stream.litlen_frequency[data_[position]];
        insert(position);
        ++position;
      }
    }
  }

 private:
  void insert(std::size_t position) noexcept {
    if (position + kMinMatch <= size_) {
      const std::size_t h = hash3(data_ + position);
      previous_[position % kWindowSize] = head_[h];
      head_[h] = static_cast<std::int32_t>(position);
    }
  }

  const std::uint8_t* data_;
  std::size_t size_;
  unsigned max_chain_;
  // head_[h]: most recent position whose 3-byte hash is h.
  // previous_[p % window]: the position before p with the same hash.
  std::vector<std::int32_t> head_;
  std::vector<std::int32_t> previous_;
};

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
// Block costing and emission
// ---------------------------------------------------------------------------

constexpr unsigned kBlockTypeStored = 0;
constexpr unsigned kBlockTypeFixed = 1;
constexpr unsigned kBlockTypeDynamic = 2;
constexpr std::uint64_t kBlockHeaderBits = 3;
constexpr std::size_t kMaxStoredBlock = 65535;

constexpr auto kFixedLitLen = fixed_litlen_lengths();

constexpr std::array<std::uint8_t, kDistSymbols> make_fixed_dist_lengths() noexcept {
  std::array<std::uint8_t, kDistSymbols> lengths{};
  for (std::uint8_t& length : lengths) {
    length = static_cast<std::uint8_t>(kFixedDistBits);
  }
  return lengths;
}
constexpr auto kFixedDist = make_fixed_dist_lengths();

/// A dynamic-Huffman encoding of a symbol distribution, with its exact size.
struct DynamicPlan {
  std::array<std::uint8_t, kLitLenSymbols> litlen_lengths{};
  std::array<std::uint8_t, kDistSymbols> dist_lengths{};
  DynamicHeader header;
  std::uint64_t bits = 0;  // whole block: type bits, tables, symbols, end-of-block, extra bits
};

/// `litlen_frequency` must already include the end-of-block symbol.
DynamicPlan plan_dynamic(const std::array<std::uint32_t, kLitLenSymbols>& litlen_frequency,
                         const std::array<std::uint32_t, kDistSymbols>& dist_frequency, std::uint64_t extra_bits) {
  DynamicPlan plan;
  plan.litlen_lengths = complete_code_lengths(litlen_frequency, kMaxHuffmanBits);
  plan.dist_lengths = complete_code_lengths(dist_frequency, kMaxHuffmanBits);
  plan.header = make_dynamic_header(plan.litlen_lengths, plan.dist_lengths);
  plan.bits = kBlockHeaderBits + plan.header.bits + coded_bits(litlen_frequency, plan.litlen_lengths) +
              coded_bits(dist_frequency, plan.dist_lengths) + extra_bits;
  return plan;
}

std::uint64_t fixed_block_bits(const std::array<std::uint32_t, kLitLenSymbols>& litlen_frequency,
                               const std::array<std::uint32_t, kDistSymbols>& dist_frequency,
                               std::uint64_t extra_bits) {
  return kBlockHeaderBits + coded_bits(litlen_frequency, kFixedLitLen) + coded_bits(dist_frequency, kFixedDist) +
         extra_bits;
}

/// Exact size of `size` bytes as stored blocks, starting `pending_bits` into a byte.
std::uint64_t stored_block_bits(unsigned pending_bits, std::size_t size) noexcept {
  const std::uint64_t chunks = (size + kMaxStoredBlock - 1) / kMaxStoredBlock;
  const std::uint64_t first_padding = (8 - (pending_bits + kBlockHeaderBits) % 8) % 8;
  // Every chunk after the first starts byte-aligned: 3 type bits + 5 padding.
  return kBlockHeaderBits + first_padding + (chunks - 1) * 8 + chunks * 32 + std::uint64_t{size} * 8;
}

void write_block_start(BitWriter& writer, bool final_block, unsigned type) {
  writer.write(final_block ? 1u : 0u, 1);
  writer.write(type, 2);
}

void write_stored(BitWriter& writer, std::span<const std::uint8_t> bytes, bool final_block) {
  std::size_t offset = 0;
  do {
    const std::size_t chunk = std::min(kMaxStoredBlock, bytes.size() - offset);
    const bool last_chunk = offset + chunk == bytes.size();
    write_block_start(writer, final_block && last_chunk, kBlockTypeStored);
    writer.align_to_byte();
    writer.write(static_cast<std::uint32_t>(chunk), 16);
    writer.write(static_cast<std::uint32_t>(chunk) ^ 0xFFFFu, 16);
    writer.append_bytes(bytes.subspan(offset, chunk));
    offset += chunk;
  } while (offset < bytes.size());
}

void write_literals(BitWriter& writer, std::span<const std::uint8_t> bytes, std::span<const std::uint8_t> litlen_lengths) {
  std::array<std::uint16_t, kFixedLitLenSymbols> codes{};
  assign_reversed_codes(litlen_lengths, std::span(codes).first(litlen_lengths.size()));
  for (const std::uint8_t byte : bytes) {
    writer.write(codes[byte], litlen_lengths[byte]);
  }
  writer.write(codes[kEndOfBlock], litlen_lengths[kEndOfBlock]);
}

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

// ---------------------------------------------------------------------------
// Per-segment encoding decision
// ---------------------------------------------------------------------------

/// Bytes at the start of each segment on which a trial match search is run.
constexpr std::size_t kProbeBytes = 8192;
/// A segment gets the full match search only if the trial predicts it will
/// shrink the whole stream by at least this many thousandths. The threshold
/// is on the stream, not the segment: a plane that codes to almost nothing
/// can be halved by LZ77 and still not be worth the search, while on a
/// highly compressible block every plane matters.
constexpr std::uint64_t kMinStreamSavingPermille = 20;
/// Segments are never shorter than this: below it the per-block Huffman
/// tables cost more than separate modelling gains.
constexpr std::size_t kMinSegmentBytes = 4096;
constexpr std::size_t kDefaultSegmentBytes = 65536;

/// Byte histogram plus one end-of-block symbol. Four interleaved tables: on
/// low-entropy planes consecutive bytes are usually equal, and incrementing
/// one counter repeatedly is limited by store-to-load forwarding.
std::array<std::uint32_t, kLitLenSymbols> literal_histogram(std::span<const std::uint8_t> bytes) noexcept {
  std::array<std::array<std::uint32_t, 256>, 4> partial{};
  const std::uint8_t* p = bytes.data();
  std::size_t remaining = bytes.size();
  for (; remaining >= 4; p += 4, remaining -= 4) {
    ++partial[0][p[0]];
    ++partial[1][p[1]];
    ++partial[2][p[2]];
    ++partial[3][p[3]];
  }
  for (; remaining > 0; ++p, --remaining) {
    ++partial[0][*p];
  }
  std::array<std::uint32_t, kLitLenSymbols> histogram{};
  for (std::size_t symbol = 0; symbol < 256; ++symbol) {
    histogram[symbol] = partial[0][symbol] + partial[1][symbol] + partial[2][symbol] + partial[3][symbol];
  }
  ++histogram[kEndOfBlock];
  return histogram;
}

constexpr std::array<std::uint32_t, kDistSymbols> kNoDistances{};

/// What is known about a segment before any full match search.
struct SegmentPlan {
  std::size_t begin = 0;
  std::size_t end = 0;
  DynamicPlan literal;                  // Huffman-coded literals, exact
  std::uint64_t literal_fixed_bits = 0;  // the same with the fixed code, exact
  std::uint64_t lz_estimate_bits = 0;    // LZ77 + Huffman, extrapolated from the probe
  bool search = true;
};

/// Costs one segment as literals and estimates what LZ77 would make of it by
/// running the match search on a probe at its start, with no history.
SegmentPlan plan_segment(std::span<const std::uint8_t> in, std::size_t begin, std::size_t end, Matcher& probe_matcher,
                         TokenStream& probe) {
  SegmentPlan plan;
  plan.begin = begin;
  plan.end = end;
  const auto literal_frequency = literal_histogram(in.subspan(begin, end - begin));
  plan.literal = plan_dynamic(literal_frequency, kNoDistances, 0);
  plan.literal_fixed_bits = fixed_block_bits(literal_frequency, kNoDistances, 0);

  const std::size_t probe_end = std::min(end, begin + kProbeBytes);
  probe_matcher.reset();
  probe.clear();
  probe_matcher.tokenize(begin, probe_end, probe);
  auto probe_frequency = probe.litlen_frequency;
  ++probe_frequency[kEndOfBlock];
  const std::uint64_t lz_probe_bits = plan_dynamic(probe_frequency, probe.dist_frequency, probe.extra_bits).bits;
  const std::uint64_t literal_probe_bits =
      plan_dynamic(literal_histogram(in.subspan(begin, probe_end - begin)), kNoDistances, 0).bits;
  // Scale the probe's LZ-to-literal ratio to the whole segment. Integer
  // arithmetic only: the decision must be identical on every platform.
  plan.lz_estimate_bits = plan.literal.bits * lz_probe_bits / literal_probe_bits;
  return plan;
}

/// Emits one segment as one or more DEFLATE blocks, choosing the smallest of
/// the available encodings. Every candidate is costed exactly, so the result
/// is never larger than Huffman-coded literals or stored bytes.
void encode_segment(BitWriter& writer, Matcher& matcher, TokenStream& lz, std::span<const std::uint8_t> in,
                    const SegmentPlan& plan, bool final_block) {
  const std::span<const std::uint8_t> bytes = in.subspan(plan.begin, plan.end - plan.begin);

  enum class Choice { Stored, LiteralFixed, LiteralDynamic, LzFixed, LzDynamic };
  // In order of decoding cost, so ties go to the cheaper decode.
  Choice choice = Choice::Stored;
  std::uint64_t best_bits = stored_block_bits(writer.pending_bits(), bytes.size());
  const auto consider = [&](Choice candidate, std::uint64_t bits) {
    if (bits < best_bits) {
      choice = candidate;
      best_bits = bits;
    }
  };
  consider(Choice::LiteralFixed, plan.literal_fixed_bits);
  consider(Choice::LiteralDynamic, plan.literal.bits);

  DynamicPlan lz_plan;
  if (plan.search) {
    lz.clear();
    matcher.tokenize(plan.begin, plan.end, lz);
    auto lz_frequency = lz.litlen_frequency;
    ++lz_frequency[kEndOfBlock];
    consider(Choice::LzFixed, fixed_block_bits(lz_frequency, lz.dist_frequency, lz.extra_bits));
    lz_plan = plan_dynamic(lz_frequency, lz.dist_frequency, lz.extra_bits);
    consider(Choice::LzDynamic, lz_plan.bits);
  }

  switch (choice) {
    case Choice::Stored:
      write_stored(writer, bytes, final_block);
      break;
    case Choice::LiteralFixed:
      write_block_start(writer, final_block, kBlockTypeFixed);
      write_literals(writer, bytes, kFixedLitLen);
      break;
    case Choice::LiteralDynamic:
      write_block_start(writer, final_block, kBlockTypeDynamic);
      write_dynamic_header(writer, plan.literal.header);
      write_literals(writer, bytes, plan.literal.litlen_lengths);
      break;
    case Choice::LzFixed:
      write_block_start(writer, final_block, kBlockTypeFixed);
      write_tokens(writer, lz.tokens, kFixedLitLen, kFixedDist);
      break;
    case Choice::LzDynamic:
      write_block_start(writer, final_block, kBlockTypeDynamic);
      write_dynamic_header(writer, lz_plan.header);
      write_tokens(writer, lz.tokens, lz_plan.litlen_lengths, lz_plan.dist_lengths);
      break;
  }
}

}  // namespace

DeflateCodec::DeflateCodec(int level) noexcept
    : max_chain_(1u << static_cast<unsigned>(std::clamp(level, kMinLevel, kMaxLevel) + 1)),
      adaptive_search_(level < kMaxLevel) {}

void DeflateCodec::compress(std::span<const std::uint8_t> in, std::vector<std::uint8_t>& out,
                            std::size_t segment_hint) const {
  out.reserve(out.size() + in.size() / 2 + 64);
  BitWriter writer(out);

  if (in.empty()) {
    // A stream needs at least one block: an empty fixed one is the smallest.
    write_block_start(writer, true, kBlockTypeFixed);
    write_literals(writer, in, kFixedLitLen);
    writer.flush();
    return;
  }

  // Segment boundaries follow the hint; short planes are grouped so that no
  // segment is smaller than kMinSegmentBytes.
  std::size_t segment = segment_hint != 0 ? segment_hint : kDefaultSegmentBytes;
  if (segment < kMinSegmentBytes) {
    segment *= (kMinSegmentBytes + segment - 1) / segment;
  }

  Matcher matcher(in, max_chain_);
  TokenStream tokens;

  // Pass 1: cost every segment as literals and probe it with the match
  // search, then decide which segments are worth searching in full.
  std::vector<SegmentPlan> plans;
  plans.reserve(in.size() / segment + 1);
  for (std::size_t begin = 0; begin < in.size(); begin += segment) {
    plans.push_back(plan_segment(in, begin, std::min(in.size(), begin + segment), matcher, tokens));
  }
  if (adaptive_search_) {
    std::uint64_t stream_estimate_bits = 0;
    for (const SegmentPlan& plan : plans) {
      stream_estimate_bits += std::min(plan.literal.bits, plan.lz_estimate_bits);
    }
    for (SegmentPlan& plan : plans) {
      const std::uint64_t saving =
          plan.literal.bits > plan.lz_estimate_bits ? plan.literal.bits - plan.lz_estimate_bits : 0;
      plan.search = saving * 1000 >= stream_estimate_bits * kMinStreamSavingPermille;
    }
  }

  // Pass 2: encode. The matcher's history spans all searched segments, so a
  // later plane can still reference an earlier one.
  matcher.reset();
  for (const SegmentPlan& plan : plans) {
    encode_segment(writer, matcher, tokens, in, plan, plan.end == in.size());
  }
  writer.flush();
}

}  // namespace turbocore::detail
