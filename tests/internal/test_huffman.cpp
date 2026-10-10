#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "detail/bit_io.hpp"
#include "detail/huffman.hpp"

namespace {

using turbocore::detail::assign_reversed_codes;
using turbocore::detail::BitReader;
using turbocore::detail::BitWriter;
using turbocore::detail::build_code_lengths;
using turbocore::detail::HuffmanDecoder;

/// Kraft sum scaled by 2^max_bits: equals 2^max_bits for a complete code.
std::uint64_t kraft_sum(const std::vector<std::uint8_t>& lengths, unsigned max_bits) {
  std::uint64_t sum = 0;
  for (const std::uint8_t length : lengths) {
    if (length != 0) {
      sum += std::uint64_t{1} << (max_bits - length);
    }
  }
  return sum;
}

TEST(Huffman, UnusedSymbolsGetNoCode) {
  const std::vector<std::uint32_t> frequencies = {5, 0, 3, 0, 1};
  std::vector<std::uint8_t> lengths(frequencies.size());
  build_code_lengths(frequencies, 15, lengths);
  EXPECT_EQ(lengths[1], 0);
  EXPECT_EQ(lengths[3], 0);
  EXPECT_EQ(lengths[0], 1);  // the most frequent symbol gets the shortest code
  EXPECT_EQ(kraft_sum(lengths, 15), std::uint64_t{1} << 15);
}

TEST(Huffman, SingleSymbolGetsOneBit) {
  const std::vector<std::uint32_t> frequencies = {0, 0, 9, 0};
  std::vector<std::uint8_t> lengths(frequencies.size());
  build_code_lengths(frequencies, 15, lengths);
  EXPECT_EQ(lengths, (std::vector<std::uint8_t>{0, 0, 1, 0}));
}

// Fibonacci frequencies produce the deepest possible Huffman tree: 40 symbols
// would need a 39-bit code. The limiter must bring that down to the cap and
// still leave a complete prefix code.
TEST(Huffman, LengthLimitIsEnforcedAndCodeStaysComplete) {
  std::vector<std::uint32_t> frequencies = {1, 1};
  while (frequencies.size() < 40) {
    frequencies.push_back(frequencies[frequencies.size() - 1] + frequencies[frequencies.size() - 2]);
  }
  for (const unsigned max_bits : {7u, 9u, 15u}) {
    SCOPED_TRACE(max_bits);
    std::vector<std::uint8_t> lengths(frequencies.size());
    build_code_lengths(frequencies, max_bits, lengths);
    for (const std::uint8_t length : lengths) {
      EXPECT_GE(length, 1);
      EXPECT_LE(length, max_bits);
    }
    EXPECT_EQ(kraft_sum(lengths, max_bits), std::uint64_t{1} << max_bits);
    // Rarer symbols never get shorter codes than more frequent ones.
    for (std::size_t i = 1; i < lengths.size(); ++i) {
      EXPECT_GE(lengths[i - 1], lengths[i]);
    }
  }
}

// Encode with the canonical codes, decode with HuffmanDecoder. The Fibonacci
// alphabet has codes on both sides of the decoder's fast-table width.
TEST(Huffman, EncoderCodesRoundTripThroughDecoder) {
  std::vector<std::uint32_t> frequencies = {1, 1};
  while (frequencies.size() < 30) {
    frequencies.push_back(frequencies[frequencies.size() - 1] + frequencies[frequencies.size() - 2]);
  }
  std::vector<std::uint8_t> lengths(frequencies.size());
  build_code_lengths(frequencies, 15, lengths);
  std::vector<std::uint16_t> codes(frequencies.size());
  assign_reversed_codes(lengths, codes);

  std::vector<std::uint8_t> bytes;
  BitWriter writer(bytes);
  for (int pass = 0; pass < 3; ++pass) {
    for (std::size_t symbol = 0; symbol < codes.size(); ++symbol) {
      writer.write(codes[symbol], lengths[symbol]);
    }
  }
  writer.flush();

  HuffmanDecoder decoder;
  ASSERT_TRUE(decoder.build(lengths));
  BitReader reader(bytes);
  for (int pass = 0; pass < 3; ++pass) {
    for (std::size_t symbol = 0; symbol < codes.size(); ++symbol) {
      ASSERT_EQ(decoder.decode(reader), static_cast<int>(symbol));
      ASSERT_FALSE(reader.overrun());
    }
  }
}

TEST(Huffman, OverSubscribedLengthsAreRejected) {
  HuffmanDecoder decoder;
  EXPECT_FALSE(decoder.build(std::vector<std::uint8_t>{1, 1, 1}));
  EXPECT_TRUE(decoder.build(std::vector<std::uint8_t>{1, 2, 2}));
}

TEST(Huffman, UnassignedPatternOfIncompleteCodeFailsToDecode) {
  HuffmanDecoder decoder;
  // One symbol with the 1-bit code "0"; the pattern "1..." means nothing.
  ASSERT_TRUE(decoder.build(std::vector<std::uint8_t>{1}));
  const std::vector<std::uint8_t> ones(4, 0xFF);
  BitReader reader(ones);
  EXPECT_EQ(decoder.decode(reader), -1);
}

TEST(BitReader, ReadingPastTheEndLatchesOverrun) {
  const std::vector<std::uint8_t> bytes = {0xA5};
  BitReader reader(bytes);
  EXPECT_EQ(reader.read(4), 0x5u);
  EXPECT_EQ(reader.read(4), 0xAu);
  EXPECT_FALSE(reader.overrun());
  EXPECT_EQ(reader.read(1), 0u);
  EXPECT_TRUE(reader.overrun());
}

}  // namespace
