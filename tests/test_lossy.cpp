#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <tuple>
#include <vector>

#include "test_data.hpp"
#include "turbocore/turbocore.hpp"

// End-to-end lossy compression. The contract is exact, not approximate:
// decompress(compress(x)) must equal quantize_round_trip(x, format) bit for
// bit, because everything after the quantization step is lossless.

namespace {

using turbocore::CompressionOptions;
using turbocore::Compressor;
using turbocore::Decompressor;
using turbocore::Predictor;
using turbocore::QuantFormat;
using turbocore::testing::bit_identical;

std::vector<float> mixed_data() {
  namespace t = turbocore::testing;
  std::vector<float> data = t::smooth_wave(30000);
  for (const auto& part : {t::uniform_noise(5000, 1), t::special_values(2000), t::random_bit_patterns(5000, 2)}) {
    data.insert(data.end(), part.begin(), part.end());
  }
  return data;
}

class LossyRoundTrip : public ::testing::TestWithParam<std::tuple<QuantFormat, Predictor, bool>> {};

TEST_P(LossyRoundTrip, DecodesToExactlyTheQuantizedValues) {
  const auto [format, predictor, byte_shuffle] = GetParam();
  const std::vector<float> data = mixed_data();
  const CompressionOptions options{
      .quantization = format, .predictor = predictor, .byte_shuffle = byte_shuffle, .block_elements = 4096};

  const std::vector<std::uint8_t> stream = Compressor(options).compress(data);
  const std::vector<float> decoded = Decompressor().decompress(stream);

  EXPECT_TRUE(bit_identical(decoded, turbocore::quantize_round_trip(data, format)));
}

INSTANTIATE_TEST_SUITE_P(AllOptions, LossyRoundTrip,
                         ::testing::Combine(::testing::Values(QuantFormat::Float16, QuantFormat::BFloat16),
                                            ::testing::Values(Predictor::None, Predictor::Delta, Predictor::Xor),
                                            ::testing::Bool()));

TEST(Lossy, EverySizeAroundBlockBoundaries) {
  const std::vector<float> source = turbocore::testing::uniform_noise(40, 9);
  for (const QuantFormat format : {QuantFormat::Float16, QuantFormat::BFloat16}) {
    for (std::size_t n = 0; n <= source.size(); ++n) {
      const std::vector<float> data(source.begin(), source.begin() + static_cast<std::ptrdiff_t>(n));
      const std::vector<std::uint8_t> stream = Compressor({.quantization = format, .block_elements = 7}).compress(data);
      ASSERT_TRUE(bit_identical(Decompressor().decompress(stream), turbocore::quantize_round_trip(data, format)))
          << "n=" << n;
    }
  }
}

TEST(Lossy, HeaderRecordsTheQuantization) {
  const std::vector<float> data = turbocore::testing::smooth_wave(1000);

  const auto lossless = Compressor().compress(data);
  EXPECT_FALSE(Decompressor::inspect(lossless).quantization.has_value());
  EXPECT_EQ(lossless[4], 1) << "lossless streams keep format version 1";

  for (const QuantFormat format : {QuantFormat::Float16, QuantFormat::BFloat16}) {
    const auto lossy = Compressor({.quantization = format}).compress(data);
    const turbocore::StreamInfo info = Decompressor::inspect(lossy);
    ASSERT_TRUE(info.quantization.has_value());
    EXPECT_EQ(*info.quantization, format);
    EXPECT_EQ(info.element_count, data.size());
    EXPECT_EQ(lossy[4], 2) << "16-bit element types need format version 2";
  }
}

// The decoded stream carries exactly the error analyze_quantization() predicts.
TEST(Lossy, ErrorMatchesTheAnalysisModule) {
  const std::vector<float> data = turbocore::testing::smooth_wave(100000);
  for (const QuantFormat format : {QuantFormat::Float16, QuantFormat::BFloat16}) {
    const std::vector<float> decoded = Decompressor().decompress(Compressor({.quantization = format}).compress(data));
    const turbocore::ErrorReport measured = turbocore::analyze_error(data, decoded);
    const turbocore::ErrorReport predicted = turbocore::analyze_quantization(data, format).error;

    EXPECT_EQ(measured.mean_squared_error, predicted.mean_squared_error);
    EXPECT_EQ(measured.psnr_db, predicted.psnr_db);
    EXPECT_EQ(measured.max_ulp_distance, predicted.max_ulp_distance);
    EXPECT_GT(measured.mean_squared_error, 0.0);  // it really is lossy
    EXPECT_GT(measured.psnr_db, 40.0);            // and still a faithful signal
  }
}

// Halving the element width must at least halve incompressible data, and the
// lossy stream must be smaller than the lossless one on noisy data, where
// lossless compression has little to work with.
TEST(Lossy, StreamsAreSmallerThanLossless) {
  const std::vector<float> noise = turbocore::testing::uniform_noise(100000, 4);
  const std::size_t raw_bytes = noise.size() * sizeof(float);
  const std::size_t lossless = Compressor().compress(noise).size();
  for (const QuantFormat format : {QuantFormat::Float16, QuantFormat::BFloat16}) {
    const std::size_t lossy = Compressor({.quantization = format}).compress(noise).size();
    EXPECT_LT(lossy, raw_bytes / 2 + 1024);
    EXPECT_LT(lossy, lossless * 2 / 3);
  }
}

TEST(Lossy, SpecialValuesSurviveTheContainer) {
  const float inf = std::numeric_limits<float>::infinity();
  const std::vector<float> data = {0.0F, -0.0F, inf, -inf, std::numeric_limits<float>::quiet_NaN(), 1e30F, -1e30F, 1e-30F};
  const std::vector<float> decoded =
      Decompressor().decompress(Compressor({.quantization = QuantFormat::Float16}).compress(data));

  ASSERT_EQ(decoded.size(), data.size());
  EXPECT_FALSE(std::signbit(decoded[0]));
  EXPECT_TRUE(std::signbit(decoded[1]) && decoded[1] == 0.0F);
  EXPECT_EQ(decoded[2], inf);
  EXPECT_EQ(decoded[3], -inf);
  EXPECT_TRUE(std::isnan(decoded[4]));
  EXPECT_EQ(decoded[5], inf);   // beyond the binary16 range
  EXPECT_EQ(decoded[6], -inf);
  EXPECT_EQ(decoded[7], 0.0F);  // below it
}

TEST(Lossy, CorruptStreamsAreRejected) {
  const std::vector<float> data = turbocore::testing::smooth_wave(500);
  const std::vector<std::uint8_t> stream =
      Compressor({.quantization = QuantFormat::BFloat16, .block_elements = 100}).compress(data);
  for (std::size_t length = 0; length < stream.size(); ++length) {
    EXPECT_THROW((void)Decompressor().decompress(std::span(stream).first(length)), turbocore::Error) << length;
  }
}

TEST(Lossy, UnknownFormatIsRejected) {
  EXPECT_THROW(Compressor({.quantization = static_cast<QuantFormat>(7)}), turbocore::Error);
}

}  // namespace
