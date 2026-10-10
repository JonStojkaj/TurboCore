#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "test_data.hpp"
#include "turbocore/turbocore.hpp"

// The decoder's contract on untrusted input: it either raises
// turbocore::Error or returns exactly the original data. It never crashes,
// reads out of bounds (enforced by the sanitizer presets) or returns
// different data without complaint.

namespace {

using turbocore::Compressor;
using turbocore::Decompressor;
using turbocore::testing::bit_identical;

/// Decoded data, or nullopt if the decoder rejected the stream.
std::optional<std::vector<float>> try_decompress(std::span<const std::uint8_t> stream) {
  try {
    return Decompressor().decompress(stream);
  } catch (const turbocore::Error&) {
    return std::nullopt;
  }
}

struct Sample {
  std::vector<float> values;
  std::vector<std::uint8_t> stream;
};

/// Small enough to mutate exhaustively, large enough to hold several blocks
/// of both kinds (compressible -> DEFLATE, incompressible -> stored).
Sample make_sample() {
  std::vector<float> values = turbocore::testing::smooth_wave(600);
  const std::vector<float> noise = turbocore::testing::random_bit_patterns(200, 11);
  values.insert(values.end(), noise.begin(), noise.end());
  std::vector<std::uint8_t> stream = Compressor({.block_elements = 200}).compress(values);
  return {std::move(values), std::move(stream)};
}

TEST(Corruption, EveryTruncationIsRejected) {
  const Sample sample = make_sample();
  for (std::size_t length = 0; length < sample.stream.size(); ++length) {
    SCOPED_TRACE(length);
    EXPECT_FALSE(try_decompress(std::span(sample.stream).first(length)).has_value());
  }
}

TEST(Corruption, TrailingBytesAreRejected) {
  Sample sample = make_sample();
  sample.stream.push_back(0);
  EXPECT_FALSE(try_decompress(sample.stream).has_value());
}

TEST(Corruption, NoSingleBitFlipYieldsWrongData) {
  const Sample sample = make_sample();
  std::size_t rejected = 0;
  std::size_t harmless = 0;

  std::vector<std::uint8_t> mutated = sample.stream;
  for (std::size_t byte = 0; byte < mutated.size(); ++byte) {
    for (unsigned bit = 0; bit < 8; ++bit) {
      mutated[byte] ^= static_cast<std::uint8_t>(1u << bit);
      const auto decoded = try_decompress(mutated);
      mutated[byte] ^= static_cast<std::uint8_t>(1u << bit);

      if (!decoded.has_value()) {
        ++rejected;
      } else {
        // The only flips that may survive are ones that do not change the
        // decoded data, such as padding bits after a block's last symbol.
        ++harmless;
        ASSERT_TRUE(bit_identical(*decoded, sample.values)) << "byte " << byte << " bit " << bit;
      }
    }
  }
  EXPECT_GT(rejected, 0u);
  // Padding is at most 7 bits per DEFLATE block.
  EXPECT_LE(harmless, 7u * Decompressor::inspect(sample.stream).block_count);
}

TEST(Corruption, ErrorCodesDistinguishTheFailure) {
  const Sample sample = make_sample();
  const auto code_for = [](std::span<const std::uint8_t> stream) -> std::optional<turbocore::ErrorCode> {
    try {
      (void)Decompressor().decompress(stream);
      return std::nullopt;
    } catch (const turbocore::Error& error) {
      return error.code();
    }
  };

  std::vector<std::uint8_t> bad_magic = sample.stream;
  bad_magic[0] = 'X';
  EXPECT_EQ(code_for(bad_magic), turbocore::ErrorCode::UnsupportedFormat);

  std::vector<std::uint8_t> bad_version = sample.stream;
  bad_version[4] = 99;
  EXPECT_EQ(code_for(bad_version), turbocore::ErrorCode::UnsupportedFormat);

  // Bytes 4..7 of the first index entry are that block's data CRC.
  std::vector<std::uint8_t> bad_crc = sample.stream;
  bad_crc[32 + 4] ^= 0xFFu;
  EXPECT_EQ(code_for(bad_crc), turbocore::ErrorCode::ChecksumMismatch);

  EXPECT_EQ(code_for(std::span(sample.stream).first(10)), turbocore::ErrorCode::CorruptStream);
}

}  // namespace
