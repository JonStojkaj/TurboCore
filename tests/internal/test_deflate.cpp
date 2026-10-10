#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "detail/codec.hpp"

namespace {

using turbocore::detail::CodecHints;
using turbocore::detail::DeflateCodec;
using Bytes = std::vector<std::uint8_t>;

Bytes from_text(std::string_view text) { return Bytes(text.begin(), text.end()); }

Bytes random_bytes(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  Bytes v(n);
  for (std::uint8_t& x : v) {
    x = static_cast<std::uint8_t>(rng() & 0xFFu);
  }
  return v;
}

struct Case {
  std::string name;
  Bytes data;
};

std::vector<Case> cases() {
  Bytes text;
  for (int i = 0; i < 400; ++i) {
    const Bytes line = from_text("the quick brown fox jumps over the lazy dog " + std::to_string(i % 17) + "\n");
    text.insert(text.end(), line.begin(), line.end());
  }
  // Matches further back than the 32 KiB window must not be used.
  Bytes far_repeat = random_bytes(40000, 5);
  far_repeat.insert(far_repeat.end(), far_repeat.begin(), far_repeat.begin() + 40000);

  Bytes period_3;
  for (int i = 0; i < 1000; ++i) {
    period_3.insert(period_3.end(), {'a', 'b', 'c'});
  }

  return {
      {"empty", {}},
      {"one_byte", {0x42}},
      {"two_bytes", {0x42, 0x42}},
      {"run_of_three", {7, 7, 7}},                // shortest possible match
      {"run_of_258", Bytes(258, 0xAB)},           // longest single match, plus one literal
      {"run_of_259", Bytes(259, 0xAB)},
      {"zeros_100k", Bytes(100000, 0)},           // overlapping distance-1 copies
      {"period_3", period_3},
      {"text", text},
      {"random_70k", random_bytes(70000, 1)},     // incompressible
      {"far_repeat", far_repeat},
  };
}

class DeflateRoundTrip : public ::testing::TestWithParam<int> {};

TEST_P(DeflateRoundTrip, DecodesToTheOriginalBytes) {
  const DeflateCodec codec(GetParam());
  for (const Case& c : cases()) {
    SCOPED_TRACE(c.name);
    Bytes packed;
    codec.compress(c.data, packed);
    Bytes unpacked(c.data.size());
    ASSERT_TRUE(codec.decompress(packed, unpacked));
    EXPECT_EQ(unpacked, c.data);
  }
}

INSTANTIATE_TEST_SUITE_P(Levels, DeflateRoundTrip, ::testing::Values(1, 6, 9));

/// Four regions with very different statistics, like the byte planes of a
/// shuffled float block: constant, a small random alphabet, noise, text.
Bytes plane_like(std::size_t plane) {
  Bytes data(plane, 0x3F);
  std::mt19937 rng(21);
  for (std::size_t i = 0; i < plane; ++i) {
    data.push_back(static_cast<std::uint8_t>(0x40u + rng() % 12u));
  }
  const Bytes noise = random_bytes(plane, 22);
  data.insert(data.end(), noise.begin(), noise.end());
  while (data.size() < 4 * plane) {
    data.push_back(static_cast<std::uint8_t>("turbocore "[data.size() % 10]));
  }
  return data;
}

/// Upper bound for any input: every segment can fall back to stored blocks,
/// which cost 5 bytes of framing each. A segment is at least 4096 bytes and
/// needs two stored blocks when it is exactly one byte over the 65535 limit,
/// so framing stays below 10 bytes per 4096.
std::size_t stored_bound(std::size_t size) { return size + 10 * (size / 4096 + 1); }

TEST(Deflate, SegmentHintsNeverAffectCorrectness) {
  for (const std::size_t plane : {std::size_t{100}, std::size_t{5000}, std::size_t{65536}, std::size_t{70001}}) {
    const Bytes data = plane_like(plane);
    for (const int level : {1, 6, 9}) {
      const DeflateCodec codec(level);
      for (const std::size_t hint : {std::size_t{0}, plane, std::size_t{1}, std::size_t{7}, std::size_t{4096}, data.size(), data.size() * 2}) {
        SCOPED_TRACE("plane=" + std::to_string(plane) + " level=" + std::to_string(level) + " hint=" + std::to_string(hint));
        Bytes packed;
        codec.compress(data, packed, CodecHints{.segment_bytes = hint});
        Bytes unpacked(data.size());
        ASSERT_TRUE(codec.decompress(packed, unpacked));
        ASSERT_EQ(unpacked, data);
        EXPECT_LE(packed.size(), stored_bound(data.size()));
      }
    }
  }
}

// With the planes modelled separately, each one costs what it should: the
// constant plane almost nothing, the 12-symbol plane about log2(12) bits per
// byte, the noise plane its own size.
TEST(Deflate, PlanesAreModelledSeparately) {
  const std::size_t plane = 65536;
  const Bytes data = plane_like(plane);
  const DeflateCodec codec(6);

  Bytes with_hint;
  codec.compress(data, with_hint, CodecHints{.segment_bytes = plane});
  const double small_alphabet_bytes = static_cast<double>(plane) * 3.585 / 8.0;  // log2(12) bits per byte
  EXPECT_LT(static_cast<double>(with_hint.size()), 1.03 * (small_alphabet_bytes + static_cast<double>(plane)) + 3000.0);
}

TEST(Deflate, IncompressibleInputCostsOnlyStoredBlockFraming) {
  const Bytes noise = random_bytes(200000, 31);
  for (const int level : {1, 6, 9}) {
    Bytes packed;
    DeflateCodec(level).compress(noise, packed, CodecHints{.segment_bytes = 65536});
    EXPECT_LE(packed.size(), stored_bound(noise.size())) << level;
  }
}

// The search-skipping heuristic (levels below 9) must not cost much ratio on
// the data it was designed for, relative to searching everything.
TEST(Deflate, SkippingTheSearchCostsLittleRatio) {
  const Bytes data = plane_like(65536);
  Bytes adaptive;
  Bytes exhaustive;
  DeflateCodec(8).compress(data, adaptive, CodecHints{.segment_bytes = 65536});
  DeflateCodec(9).compress(data, exhaustive, CodecHints{.segment_bytes = 65536});
  EXPECT_LE(static_cast<double>(adaptive.size()), 1.02 * static_cast<double>(exhaustive.size()));
}

// Precomputed histograms are an optimisation only: with correct counts the
// output is byte-identical, and counts that do not cover the segments exactly
// are ignored.
TEST(Deflate, PrecomputedHistogramsDoNotChangeTheStream) {
  const std::size_t plane = 20000;
  const Bytes data = plane_like(plane);
  std::vector<std::uint32_t> histograms(4 * 256, 0);
  for (std::size_t i = 0; i < data.size(); ++i) {
    ++histograms[(i / plane) * 256 + data[i]];
  }
  for (const int level : {1, 6, 9}) {
    const DeflateCodec codec(level);
    Bytes without;
    Bytes with;
    Bytes with_wrong_count;
    codec.compress(data, without, CodecHints{.segment_bytes = plane});
    codec.compress(data, with, CodecHints{.segment_bytes = plane, .segment_histograms = histograms});
    codec.compress(data, with_wrong_count,
                   CodecHints{.segment_bytes = plane, .segment_histograms = std::span(histograms).first(3 * 256)});
    EXPECT_EQ(with, without) << level;
    EXPECT_EQ(with_wrong_count, without) << level;
  }
}

TEST(Deflate, CompressAppendsToExistingOutput) {
  const DeflateCodec codec;
  Bytes out = {0xDE, 0xAD};
  codec.compress(from_text("hello hello hello hello"), out);
  ASSERT_GT(out.size(), 2u);
  EXPECT_EQ(out[0], 0xDE);
  EXPECT_EQ(out[1], 0xAD);
}

TEST(Deflate, RepetitiveDataShrinks) {
  const DeflateCodec codec;
  Bytes packed;
  codec.compress(Bytes(100000, 0), packed);
  EXPECT_LT(packed.size(), 200u);
}

// Hand-assembled streams from RFC 1951, independent of our encoder.
TEST(Deflate, DecodesStoredBlock) {
  // BFINAL=1 BTYPE=00, pad to byte, LEN=3, NLEN=~3, then "abc".
  const Bytes stream = {0x01, 0x03, 0x00, 0xFC, 0xFF, 'a', 'b', 'c'};
  Bytes out(3);
  ASSERT_TRUE(DeflateCodec().decompress(stream, out));
  EXPECT_EQ(out, from_text("abc"));
}

TEST(Deflate, DecodesFixedHuffmanBlock) {
  // zlib's raw DEFLATE encoding of "a": fixed block, literal 'a', end of block.
  const Bytes stream = {0x4B, 0x04, 0x00};
  Bytes out(1);
  ASSERT_TRUE(DeflateCodec().decompress(stream, out));
  EXPECT_EQ(out, from_text("a"));
}

TEST(Deflate, RejectsMalformedStreams) {
  const DeflateCodec codec;
  Bytes out(3);

  EXPECT_FALSE(codec.decompress(Bytes{}, out)) << "empty input";
  EXPECT_FALSE(codec.decompress(Bytes{0x07}, out)) << "reserved block type 3";
  EXPECT_FALSE(codec.decompress(Bytes{0x01, 0x03, 0x00, 0xFF, 0xFF, 'a', 'b', 'c'}, out)) << "LEN/NLEN mismatch";
  EXPECT_FALSE(codec.decompress(Bytes{0x01, 0x03, 0x00, 0xFC, 0xFF, 'a', 'b'}, out)) << "truncated stored data";

  const Bytes stored = {0x01, 0x03, 0x00, 0xFC, 0xFF, 'a', 'b', 'c'};
  Bytes too_small(2);
  Bytes too_large(4);
  EXPECT_FALSE(codec.decompress(stored, too_small)) << "more data than expected";
  EXPECT_FALSE(codec.decompress(stored, too_large)) << "less data than expected";

  Bytes trailing = stored;
  trailing.push_back(0);
  EXPECT_FALSE(codec.decompress(trailing, out)) << "bytes after the final block";

  // Fixed block whose first symbol is a match: there is no history to copy from.
  // Bits: BFINAL=1, BTYPE=01, length code 257 (0000001), distance code 0 (00000).
  EXPECT_FALSE(codec.decompress(Bytes{0x03, 0x02, 0x00}, out)) << "distance before start of output";
}

// Throw arbitrary bytes at the decoder: it may accept or reject, but must
// return (the sanitizer presets turn any out-of-bounds access into a failure).
TEST(Deflate, RandomInputNeverCrashesTheDecoder) {
  const DeflateCodec codec;
  std::mt19937 rng(99);
  Bytes out(4096);
  for (int trial = 0; trial < 20000; ++trial) {
    const Bytes garbage = random_bytes(1 + rng() % 64, static_cast<std::uint32_t>(rng()));
    (void)codec.decompress(garbage, out);
  }
  SUCCEED();
}

// Corrupting a valid stream exercises far deeper decoder states than random bytes.
TEST(Deflate, MutatedValidStreamsNeverCrashTheDecoder) {
  const DeflateCodec codec;
  for (const Case& c : cases()) {
    if (c.data.size() > 20000) {
      continue;
    }
    Bytes packed;
    codec.compress(c.data, packed);
    Bytes out(c.data.size());
    for (std::size_t byte = 0; byte < packed.size(); ++byte) {
      const std::uint8_t saved = packed[byte];
      for (const std::uint8_t mask : {std::uint8_t{0x01}, std::uint8_t{0x10}, std::uint8_t{0xFF}}) {
        packed[byte] = saved ^ mask;
        (void)codec.decompress(packed, out);
      }
      packed[byte] = saved;
    }
  }
  SUCCEED();
}

}  // namespace
