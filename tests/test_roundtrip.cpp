#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "test_data.hpp"
#include "turbocore/turbocore.hpp"

namespace {

using turbocore::CompressionOptions;
using turbocore::Compressor;
using turbocore::Decompressor;
using turbocore::Predictor;
using turbocore::testing::bit_identical;

std::vector<float> round_trip(const std::vector<float>& input, const CompressionOptions& options) {
  const std::vector<std::uint8_t> stream = Compressor(options).compress(input);
  return Decompressor().decompress(stream);
}

struct Dataset {
  std::string name;
  std::vector<float> values;
};

std::vector<Dataset> datasets() {
  namespace t = turbocore::testing;
  return {
      {"empty", {}},
      {"one_element", {3.14159F}},
      {"zeros", t::constant(5000, 0.0F)},
      {"constant", t::constant(5000, -273.15F)},
      {"ramp", t::ramp(10000)},
      {"smooth_wave", t::smooth_wave(70000)},
      {"uniform_noise", t::uniform_noise(20000, 1)},
      {"random_bit_patterns", t::random_bit_patterns(20000, 2)},
      {"special_values", t::special_values(4096)},
  };
}

// Every dataset under every combination of predictor, shuffle and level.
class LosslessRoundTrip : public ::testing::TestWithParam<std::tuple<Predictor, bool, int>> {};

TEST_P(LosslessRoundTrip, ReproducesEveryBitPattern) {
  const auto [predictor, byte_shuffle, level] = GetParam();
  // Small blocks so that every dataset spans several of them.
  const CompressionOptions options{
      .predictor = predictor, .byte_shuffle = byte_shuffle, .level = level, .block_elements = 4096};

  for (const Dataset& dataset : datasets()) {
    SCOPED_TRACE(dataset.name);
    EXPECT_TRUE(bit_identical(round_trip(dataset.values, options), dataset.values));
  }
}

INSTANTIATE_TEST_SUITE_P(AllOptions, LosslessRoundTrip,
                         ::testing::Combine(::testing::Values(Predictor::None, Predictor::Delta, Predictor::Xor),
                                            ::testing::Bool(), ::testing::Values(1, 6, 9)));

TEST(RoundTrip, DefaultOptions) {
  const std::vector<float> input = turbocore::testing::smooth_wave(200000);
  const std::vector<std::uint8_t> stream = Compressor().compress(input);
  EXPECT_TRUE(bit_identical(Decompressor().decompress(stream), input));
}

// Sizes on both sides of every block boundary, including a block of one element.
TEST(RoundTrip, EverySizeAroundBlockBoundaries) {
  const std::vector<float> source = turbocore::testing::random_bit_patterns(64, 7);
  for (const std::size_t block_elements : {std::size_t{1}, std::size_t{2}, std::size_t{7}, std::size_t{16}}) {
    for (std::size_t n = 0; n <= source.size(); ++n) {
      SCOPED_TRACE("block_elements=" + std::to_string(block_elements) + " n=" + std::to_string(n));
      const std::vector<float> input(source.begin(), source.begin() + static_cast<std::ptrdiff_t>(n));
      EXPECT_TRUE(bit_identical(round_trip(input, {.block_elements = block_elements}), input));
    }
  }
}

TEST(RoundTrip, HeaderDescribesTheStream) {
  const std::vector<float> input = turbocore::testing::ramp(10001);
  const CompressionOptions options{
      .predictor = Predictor::Xor, .byte_shuffle = false, .level = 3, .block_elements = 1000};
  const std::vector<std::uint8_t> stream = Compressor(options).compress(input);

  const turbocore::StreamInfo info = Decompressor::inspect(stream);
  EXPECT_EQ(info.element_count, 10001u);
  EXPECT_EQ(info.block_elements, 1000u);
  EXPECT_EQ(info.block_count, 11u);
  EXPECT_EQ(info.predictor, Predictor::Xor);
  EXPECT_FALSE(info.byte_shuffle);
}

TEST(RoundTrip, CompressionIsDeterministic) {
  const std::vector<float> input = turbocore::testing::smooth_wave(50000);
  EXPECT_EQ(Compressor().compress(input), Compressor().compress(input));
}

// Not a ratio benchmark, only a guard that the stages are actually wired up.
TEST(Compression, SmoothDataShrinks) {
  const std::vector<float> input = turbocore::testing::smooth_wave(200000);
  const std::size_t raw_bytes = input.size() * sizeof(float);
  EXPECT_LT(Compressor().compress(input).size(), raw_bytes * 3 / 4);
}

TEST(Compression, ConstantDataCollapses) {
  const std::vector<float> input = turbocore::testing::constant(200000, 1.5F);
  const std::size_t raw_bytes = input.size() * sizeof(float);
  EXPECT_LT(Compressor().compress(input).size(), raw_bytes / 100);
}

// Incompressible input falls back to stored blocks, so the only growth is the
// 32-byte header plus 12 bytes of index per block.
TEST(Compression, IncompressibleDataGrowsOnlyByFraming) {
  const std::vector<float> input = turbocore::testing::random_bit_patterns(100000, 3);
  const std::vector<std::uint8_t> stream = Compressor().compress(input);
  const turbocore::StreamInfo info = Decompressor::inspect(stream);
  EXPECT_LE(stream.size(), input.size() * sizeof(float) + 32 + 12 * std::size_t{info.block_count});
}

TEST(Options, OutOfRangeValuesAreRejected) {
  const auto expect_invalid = [](const CompressionOptions& options) {
    try {
      const Compressor compressor(options);
      ADD_FAILURE() << "expected turbocore::Error";
    } catch (const turbocore::Error& error) {
      EXPECT_EQ(error.code(), turbocore::ErrorCode::InvalidArgument);
    }
  };
  expect_invalid({.level = 0});
  expect_invalid({.level = 10});
  expect_invalid({.block_elements = 0});
  expect_invalid({.block_elements = (std::size_t{1} << 24) + 1});
  expect_invalid({.predictor = static_cast<Predictor>(3)});
}

}  // namespace
