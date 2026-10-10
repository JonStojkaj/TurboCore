#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "turbocore/turbocore.hpp"

namespace {

using turbocore::analyze_error;
using turbocore::analyze_quantization;
using turbocore::ErrorReport;
using turbocore::PeakMode;
using turbocore::QuantFormat;
using turbocore::QuantizationReport;

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// ---------------------------------------------------------------------------
// analyze_error: numeric metrics against hand-computed values
// ---------------------------------------------------------------------------

TEST(ErrorAnalysis, HandComputedMetrics) {
  const std::vector<float> reference = {1.0F, 2.0F, 3.0F, 4.0F};
  const std::vector<float> reconstructed = {1.0F, 2.0F, 3.0F, 5.0F};
  const ErrorReport r = analyze_error(reference, reconstructed);

  EXPECT_EQ(r.element_count, 4u);
  EXPECT_EQ(r.compared_count, 4u);
  EXPECT_EQ(r.bit_exact_count, 3u);
  EXPECT_DOUBLE_EQ(r.mean_squared_error, 0.25);      // (0 + 0 + 0 + 1) / 4
  EXPECT_DOUBLE_EQ(r.root_mean_squared_error, 0.5);
  EXPECT_DOUBLE_EQ(r.mean_absolute_error, 0.25);
  EXPECT_DOUBLE_EQ(r.max_absolute_error, 1.0);
  EXPECT_DOUBLE_EQ(r.max_relative_error, 0.25);      // 1 / 4
  EXPECT_DOUBLE_EQ(r.peak, 4.0);
  EXPECT_NEAR(r.psnr_db, 10.0 * std::log10(16.0 / 0.25), 1e-12);  // 18.06 dB
  // 4.0f is 0x40800000 and 5.0f is 0x40A00000: 2^21 floats apart.
  EXPECT_EQ(r.max_ulp_distance, std::uint64_t{1} << 21);
  EXPECT_DOUBLE_EQ(r.mean_ulp_distance, static_cast<double>(std::uint64_t{1} << 21) / 4.0);
}

TEST(ErrorAnalysis, PeakModesChangeOnlyPsnr) {
  const std::vector<float> reference = {1.0F, 2.0F, 3.0F, 4.0F};
  const std::vector<float> reconstructed = {1.0F, 2.0F, 3.0F, 5.0F};

  const ErrorReport range = analyze_error(reference, reconstructed, {.peak_mode = PeakMode::ReferenceRange});
  EXPECT_DOUBLE_EQ(range.peak, 3.0);
  EXPECT_NEAR(range.psnr_db, 10.0 * std::log10(9.0 / 0.25), 1e-12);

  const ErrorReport custom =
      analyze_error(reference, reconstructed, {.peak_mode = PeakMode::Custom, .custom_peak = 255.0});
  EXPECT_DOUBLE_EQ(custom.peak, 255.0);
  EXPECT_NEAR(custom.psnr_db, 10.0 * std::log10(255.0 * 255.0 / 0.25), 1e-12);
  EXPECT_DOUBLE_EQ(custom.mean_squared_error, range.mean_squared_error);
}

TEST(ErrorAnalysis, IdenticalDataHasInfinitePsnr) {
  const std::vector<float> data = {0.5F, -7.0F, 1e20F, 0.0F};
  const ErrorReport r = analyze_error(data, data);
  EXPECT_EQ(r.bit_exact_count, 4u);
  EXPECT_EQ(r.mean_squared_error, 0.0);
  EXPECT_EQ(r.max_ulp_distance, 0u);
  EXPECT_EQ(r.psnr_db, std::numeric_limits<double>::infinity());
}

TEST(ErrorAnalysis, PsnrSurvivesTheEndsOfTheFloatRange) {
  // peak^2 overflows double for neither, but the naive peak^2 / MSE quotient
  // is far outside what a float pipeline would handle; the result must still
  // be the mathematically correct finite number.
  const std::vector<float> huge = {3e38F, -3e38F};
  const std::vector<float> huge_off = {std::nextafter(3e38F, 0.0F), -3e38F};
  const ErrorReport r = analyze_error(huge, huge_off);
  EXPECT_TRUE(std::isfinite(r.psnr_db));
  const double expected = 20.0 * std::log10(static_cast<double>(3e38F)) - 10.0 * std::log10(r.mean_squared_error);
  EXPECT_NEAR(r.psnr_db, expected, 1e-9);
  EXPECT_GT(r.psnr_db, 100.0);  // one FP32 ULP of error on a 24-bit signal

  const float tiny = std::numeric_limits<float>::denorm_min();
  const ErrorReport small = analyze_error(std::vector<float>{4 * tiny, 0.0F}, std::vector<float>{3 * tiny, 0.0F});
  EXPECT_GT(small.mean_squared_error, 0.0);  // (2^-149)^2 / 2 does not underflow in double
  EXPECT_NEAR(small.psnr_db, 10.0 * std::log10(16.0 / 0.5), 1e-9);
}

// One large squared error followed by a million small ones. A plain double
// accumulator sitting at 1e16 cannot register +1 (its ULP there is 2), so it
// would report a sum of exactly 1e16 and lose all million terms.
TEST(ErrorAnalysis, SummationDoesNotLoseSmallTermsBehindALargeOne) {
  const std::size_t small_terms = 1'000'000;
  std::vector<float> reference(small_terms + 1, 0.0F);
  std::vector<float> reconstructed(small_terms + 1, 1.0F);  // squared error 1 each
  reconstructed[0] = 1e8F;                                   // squared error 1e16

  const ErrorReport r = analyze_error(reference, reconstructed, {.peak_mode = PeakMode::Custom, .custom_peak = 1.0});
  const double exact_sum = 1e16 + static_cast<double>(small_terms);
  EXPECT_DOUBLE_EQ(r.mean_squared_error, exact_sum / static_cast<double>(small_terms + 1));

  // Demonstrate that the naive sum really is wrong here, so this test would
  // catch a regression to it.
  double naive = 0.0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double e = static_cast<double>(reconstructed[i]) - static_cast<double>(reference[i]);
    naive += e * e;
  }
  EXPECT_NE(naive, exact_sum);
}

// ---------------------------------------------------------------------------
// analyze_error: ULP distance
// ---------------------------------------------------------------------------

TEST(UlpDistance, CountsRepresentableFloats) {
  using turbocore::ulp_distance;
  EXPECT_EQ(ulp_distance(1.0F, 1.0F), 0);
  EXPECT_EQ(ulp_distance(1.0F, std::nextafter(1.0F, 2.0F)), 1);
  EXPECT_EQ(ulp_distance(1.0F, std::nextafter(1.0F, 0.0F)), -1);
  // Crossing a power of two changes the size of a ULP, not the count.
  EXPECT_EQ(ulp_distance(std::nextafter(2.0F, 0.0F), std::nextafter(2.0F, 3.0F)), 2);
  // Both zeros occupy the same position.
  EXPECT_EQ(ulp_distance(0.0F, -0.0F), 0);
  const float tiny = std::numeric_limits<float>::denorm_min();
  EXPECT_EQ(ulp_distance(-tiny, tiny), 2);
  EXPECT_EQ(ulp_distance(0.0F, std::numeric_limits<float>::min()), 1 << 23);
  // The whole finite range.
  EXPECT_EQ(ulp_distance(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max()),
            2 * std::int64_t{0x7F7FFFFF});
  EXPECT_EQ(ulp_distance(std::numeric_limits<float>::max(), kInf), 1);
}

TEST(ErrorAnalysis, UlpDistanceIsUniformAcrossMagnitudes) {
  // The same one-float step at wildly different scales: absolute error
  // differs by 60 orders of magnitude, the ULP distance is 1 everywhere.
  const std::vector<float> reference = {1e-30F, 1.0F, 1e30F};
  std::vector<float> reconstructed;
  for (const float x : reference) {
    reconstructed.push_back(std::nextafter(x, kInf));
  }
  const ErrorReport r = analyze_error(reference, reconstructed);
  EXPECT_EQ(r.max_ulp_distance, 1u);
  EXPECT_DOUBLE_EQ(r.mean_ulp_distance, 1.0);
}

// ---------------------------------------------------------------------------
// analyze_error: special values
// ---------------------------------------------------------------------------

TEST(ErrorAnalysis, SpecialValuesAreCountedAndExcluded) {
  //                                     NaN kept  NaN lost  inf kept  inf flipped  overflow  NaN made  underflow  zero sign  normal
  const std::vector<float> reference = {kNaN,     kNaN,     kInf,     -kInf,       1e30F,    2.0F,     1e-30F,    0.0F,      8.0F};
  const std::vector<float> rebuilt = {kNaN,       1.0F,     kInf,     kInf,        kInf,     kNaN,     0.0F,      -0.0F,     8.5F};
  const ErrorReport r = analyze_error(reference, rebuilt);

  EXPECT_EQ(r.element_count, 9u);
  EXPECT_EQ(r.nan_reference_count, 2u);
  EXPECT_EQ(r.nan_preserved_count, 1u);
  EXPECT_EQ(r.infinity_reference_count, 2u);
  EXPECT_EQ(r.infinity_preserved_count, 1u);
  EXPECT_EQ(r.overflow_to_infinity_count, 1u);
  EXPECT_EQ(r.nan_introduced_count, 1u);
  EXPECT_EQ(r.underflow_to_zero_count, 1u);
  EXPECT_EQ(r.sign_mismatch_count, 1u);
  // Compared: the underflow, the zero pair and the normal pair.
  EXPECT_EQ(r.compared_count, 3u);

  // None of the non-finite pairs leaked into the numbers.
  EXPECT_TRUE(std::isfinite(r.mean_squared_error));
  EXPECT_DOUBLE_EQ(r.max_absolute_error, 0.5);
  EXPECT_DOUBLE_EQ(r.max_relative_error, 1.0);  // the value that underflowed lost 100 %
  EXPECT_DOUBLE_EQ(r.peak, 8.0);
}

TEST(ErrorAnalysis, NothingComparableIsUndefinedNotZero) {
  const std::vector<float> reference = {kNaN, kInf};
  const ErrorReport r = analyze_error(reference, reference);
  EXPECT_EQ(r.compared_count, 0u);
  EXPECT_TRUE(std::isnan(r.mean_squared_error));
  EXPECT_TRUE(std::isnan(r.psnr_db));
  EXPECT_TRUE(std::isnan(r.max_absolute_error));

  const ErrorReport empty = analyze_error({}, {});
  EXPECT_EQ(empty.element_count, 0u);
  EXPECT_TRUE(std::isnan(empty.mean_squared_error));
}

TEST(ErrorAnalysis, InvalidArgumentsAreRejected) {
  const std::vector<float> a(3);
  const std::vector<float> b(4);
  EXPECT_THROW((void)analyze_error(a, b), turbocore::Error);
  EXPECT_THROW((void)analyze_error(a, a, {.peak_mode = PeakMode::Custom, .custom_peak = 0.0}), turbocore::Error);
  EXPECT_THROW((void)analyze_error(a, a, {.peak_mode = PeakMode::Custom, .custom_peak = std::nan("")}),
               turbocore::Error);
}

// ---------------------------------------------------------------------------
// target_ulp
// ---------------------------------------------------------------------------

TEST(TargetUlp, SpacingOfEachFormat) {
  using turbocore::target_ulp;
  // binary16: 11 significant bits.
  EXPECT_EQ(target_ulp(1.0F, QuantFormat::Float16), std::ldexp(1.0, -10));
  EXPECT_EQ(target_ulp(1.999F, QuantFormat::Float16), std::ldexp(1.0, -10));
  EXPECT_EQ(target_ulp(2.0F, QuantFormat::Float16), std::ldexp(1.0, -9));
  EXPECT_EQ(target_ulp(-40000.0F, QuantFormat::Float16), 32.0);  // top binade: 2^15 * 2^-10
  // At and below the smallest normal (2^-14) the spacing is the subnormal step.
  EXPECT_EQ(target_ulp(std::ldexp(1.0F, -14), QuantFormat::Float16), std::ldexp(1.0, -24));
  EXPECT_EQ(target_ulp(1e-10F, QuantFormat::Float16), std::ldexp(1.0, -24));
  EXPECT_EQ(target_ulp(0.0F, QuantFormat::Float16), std::ldexp(1.0, -24));

  // BFloat16: 8 significant bits, FP32 exponent range.
  EXPECT_EQ(target_ulp(1.0F, QuantFormat::BFloat16), std::ldexp(1.0, -7));
  EXPECT_EQ(target_ulp(1e30F, QuantFormat::BFloat16), std::ldexp(1.0, 99 - 7));  // 1e30 is in [2^99, 2^100)
  EXPECT_EQ(target_ulp(std::numeric_limits<float>::denorm_min(), QuantFormat::BFloat16), std::ldexp(1.0, -133));
}

// ---------------------------------------------------------------------------
// analyze_quantization
// ---------------------------------------------------------------------------

std::vector<float> uniform(std::size_t n, float lo, float hi, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (float& x : v) {
    x = dist(rng);
  }
  return v;
}

std::uint64_t histogram_total(const QuantizationReport& r) {
  return std::accumulate(r.target_ulp_histogram.begin(), r.target_ulp_histogram.end(), std::uint64_t{0});
}

TEST(QuantizationAnalysis, CorrectRoundingNeverExceedsHalfATargetUlp) {
  const std::vector<float> data = uniform(500000, -1000.0F, 1000.0F, 1);
  for (const QuantFormat format : {QuantFormat::Float16, QuantFormat::BFloat16}) {
    const QuantizationReport r = analyze_quantization(data, format);
    EXPECT_EQ(r.format, format);
    EXPECT_EQ(r.error.compared_count, data.size());
    EXPECT_EQ(r.above_half_ulp_count, 0u);
    EXPECT_LE(r.max_target_ulp_error, 0.5);
    EXPECT_GT(r.max_target_ulp_error, 0.49);  // with 500k samples the bound is approached
    // Uniformly distributed rounding error on [0, 0.5] averages 0.25.
    EXPECT_NEAR(r.mean_target_ulp_error, 0.25, 0.005);
    EXPECT_EQ(histogram_total(r), r.error.compared_count);
    for (const std::uint64_t bin : r.target_ulp_histogram) {
      EXPECT_NEAR(static_cast<double>(bin), static_cast<double>(data.size()) / 8.0, 0.02 * static_cast<double>(data.size()));
    }
  }
}

TEST(QuantizationAnalysis, RelativeErrorBoundFollowsThePrecision) {
  // Half a ULP relative to the bottom of a binade: 2^-11 for binary16
  // (11-bit significand), 2^-8 for BFloat16 (8-bit significand).
  const std::vector<float> data = uniform(200000, 1.0F, 1000.0F, 2);
  EXPECT_LE(analyze_quantization(data, QuantFormat::Float16).error.max_relative_error, std::ldexp(1.0, -11));
  EXPECT_LE(analyze_quantization(data, QuantFormat::BFloat16).error.max_relative_error, std::ldexp(1.0, -8));
  // And binary16 really is the more precise of the two inside its range.
  EXPECT_GT(analyze_quantization(data, QuantFormat::Float16).error.psnr_db,
            analyze_quantization(data, QuantFormat::BFloat16).error.psnr_db + 15.0);
}

TEST(QuantizationAnalysis, RangeLimitsShowUpInTheCounts) {
  //                               fits   fits    overflows f16  underflows f16  NaN   inf
  const std::vector<float> data = {1.0F, -3.5F,  1e6F,          1e-10F,         kNaN, -kInf};

  const QuantizationReport f16 = analyze_quantization(data, QuantFormat::Float16);
  EXPECT_EQ(f16.error.overflow_to_infinity_count, 1u);
  EXPECT_EQ(f16.error.underflow_to_zero_count, 1u);
  EXPECT_EQ(f16.error.nan_preserved_count, 1u);
  EXPECT_EQ(f16.error.infinity_preserved_count, 1u);
  EXPECT_EQ(f16.error.nan_introduced_count, 0u);
  EXPECT_EQ(f16.error.compared_count, 3u);       // 1.0, -3.5 and the underflowed value
  EXPECT_EQ(f16.exactly_representable_count, 2u);
  EXPECT_EQ(f16.above_half_ulp_count, 0u);       // even the underflow is within half a subnormal step

  // BFloat16 has the FP32 exponent range: nothing overflows or underflows.
  const QuantizationReport bf16 = analyze_quantization(data, QuantFormat::BFloat16);
  EXPECT_EQ(bf16.error.overflow_to_infinity_count, 0u);
  EXPECT_EQ(bf16.error.underflow_to_zero_count, 0u);
  EXPECT_EQ(bf16.error.nan_preserved_count, 1u);
  EXPECT_EQ(bf16.error.infinity_preserved_count, 1u);
  EXPECT_EQ(bf16.error.compared_count, 4u);
}

TEST(QuantizationAnalysis, AlreadyQuantizedDataIsExact) {
  const std::vector<float> data = uniform(10000, -100.0F, 100.0F, 3);
  for (const QuantFormat format : {QuantFormat::Float16, QuantFormat::BFloat16}) {
    const std::vector<float> quantized = turbocore::quantize_round_trip(data, format);
    const QuantizationReport r = analyze_quantization(quantized, format);
    EXPECT_EQ(r.exactly_representable_count, quantized.size());
    EXPECT_EQ(r.error.bit_exact_count, quantized.size());
    EXPECT_EQ(r.max_target_ulp_error, 0.0);
    EXPECT_EQ(r.error.psnr_db, std::numeric_limits<double>::infinity());
  }
}

// The histogram exposes data that carries fewer bits than FP32 can hold:
// values on a 1/8 grid are exact in both formats here, so every error is 0.
TEST(QuantizationAnalysis, HistogramRevealsLowPrecisionData) {
  std::vector<float> grid(4096);
  for (std::size_t i = 0; i < grid.size(); ++i) {
    grid[i] = static_cast<float>(i % 64) / 8.0F;
  }
  const QuantizationReport r = analyze_quantization(grid, QuantFormat::BFloat16);
  EXPECT_EQ(r.target_ulp_histogram[0], grid.size());
  EXPECT_EQ(r.exactly_representable_count, grid.size());
}

}  // namespace
