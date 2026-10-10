#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "quantization_reference.hpp"
#include "turbocore/turbocore.hpp"

namespace {

using turbocore::bfloat16_bits_to_float32;
using turbocore::float16_bits_to_float32;
using turbocore::float32_to_bfloat16_bits;
using turbocore::float32_to_float16_bits;
using turbocore::QuantFormat;

constexpr float kInf = std::numeric_limits<float>::infinity();

float from_bits(std::uint32_t bits) { return std::bit_cast<float>(bits); }
std::uint32_t to_bits(float value) { return std::bit_cast<std::uint32_t>(value); }

std::uint16_t f16(float value) { return float32_to_float16_bits(value); }
std::uint16_t bf16(float value) { return float32_to_bfloat16_bits(value); }

bool is_nan_f16(std::uint16_t h) { return (h & 0x7FFFu) > 0x7C00u; }
bool is_nan_bf16(std::uint16_t h) { return (h & 0x7FFFu) > 0x7F80u; }

// ---------------------------------------------------------------------------
// binary16
// ---------------------------------------------------------------------------

TEST(Float16, ExactlyRepresentableValues) {
  EXPECT_EQ(f16(0.0F), 0x0000);
  EXPECT_EQ(f16(-0.0F), 0x8000);
  EXPECT_EQ(f16(1.0F), 0x3C00);
  EXPECT_EQ(f16(-2.0F), 0xC000);
  EXPECT_EQ(f16(0.5F), 0x3800);
  EXPECT_EQ(f16(65504.0F), 0x7BFF);                // largest finite
  EXPECT_EQ(f16(std::ldexp(1.0F, -14)), 0x0400);   // smallest normal
  EXPECT_EQ(f16(std::ldexp(1.0F, -24)), 0x0001);   // smallest subnormal
  EXPECT_EQ(f16(std::ldexp(1023.0F, -24)), 0x03FF);  // largest subnormal
  EXPECT_EQ(f16(kInf), 0x7C00);
  EXPECT_EQ(f16(-kInf), 0xFC00);
}

TEST(Float16, TiesRoundToEven) {
  // Spacing at 1.0 is 2^-10, so 1 + 2^-11 is exactly between 0x3C00 and 0x3C01.
  EXPECT_EQ(f16(1.0F + std::ldexp(1.0F, -11)), 0x3C00);  // tie -> even (down)
  EXPECT_EQ(f16(1.0F + std::ldexp(3.0F, -11)), 0x3C02);  // tie -> even (up)
  // One FP32 ULP either side of a tie is no longer a tie.
  EXPECT_EQ(f16(std::nextafter(1.0F + std::ldexp(1.0F, -11), 2.0F)), 0x3C01);
  EXPECT_EQ(f16(std::nextafter(1.0F + std::ldexp(3.0F, -11), 0.0F)), 0x3C01);
}

TEST(Float16, MantissaCarryIncrementsTheExponent) {
  // Just below 2.0 rounds up to 2.0: the mantissa overflows into the exponent.
  EXPECT_EQ(f16(std::nextafter(2.0F, 0.0F)), 0x4000);
}

TEST(Float16, OverflowBoundary) {
  // 65520 is halfway between the largest finite value (65504) and the next
  // step (65536, not representable); the tie goes to even, which is infinity.
  EXPECT_EQ(f16(std::nextafter(65520.0F, 0.0F)), 0x7BFF);
  EXPECT_EQ(f16(65520.0F), 0x7C00);
  EXPECT_EQ(f16(-65520.0F), 0xFC00);
  EXPECT_EQ(f16(65536.0F), 0x7C00);
  EXPECT_EQ(f16(std::numeric_limits<float>::max()), 0x7C00);
  EXPECT_EQ(f16(std::numeric_limits<float>::lowest()), 0xFC00);
}

TEST(Float16, GradualUnderflow) {
  const float min_subnormal = std::ldexp(1.0F, -24);
  // Halfway between 0 and the smallest subnormal: tie -> even -> zero.
  EXPECT_EQ(f16(std::ldexp(1.0F, -25)), 0x0000);
  EXPECT_EQ(f16(-std::ldexp(1.0F, -25)), 0x8000);  // sign survives underflow
  EXPECT_EQ(f16(std::nextafter(std::ldexp(1.0F, -25), 1.0F)), 0x0001);
  EXPECT_EQ(f16(std::nextafter(std::ldexp(1.0F, -25), 0.0F)), 0x0000);
  // 1.5 subnormal steps: tie between 1 and 2 -> even -> 2.
  EXPECT_EQ(f16(1.5F * min_subnormal), 0x0002);
  // 2.5 steps: tie between 2 and 3 -> even -> 2.
  EXPECT_EQ(f16(2.5F * min_subnormal), 0x0002);
  // Just below the smallest normal rounds up into it.
  EXPECT_EQ(f16(std::nextafter(std::ldexp(1.0F, -14), 0.0F)), 0x0400);
  // FP32 subnormals are far below the binary16 range.
  EXPECT_EQ(f16(std::numeric_limits<float>::denorm_min()), 0x0000);
  EXPECT_EQ(f16(-std::numeric_limits<float>::min()), 0x8000);
}

TEST(Float16, NanStaysNanWithSignAndTopPayload) {
  EXPECT_EQ(f16(from_bits(0x7FC00000u)), 0x7E00);  // default quiet NaN
  EXPECT_EQ(f16(from_bits(0xFFC00000u)), 0xFE00);  // sign kept
  // Signalling NaN whose payload is entirely in the dropped bits: must not
  // collapse to infinity (0x7C00).
  EXPECT_EQ(f16(from_bits(0x7F800001u)), 0x7E00);
  // Top 10 payload bits survive: mantissa 0x2AAAAA >> 13 = 0x155, plus quiet bit.
  EXPECT_EQ(f16(from_bits(0x7FAAAAAAu)), 0x7C00 | 0x0200 | 0x0155);
}

// Dequantization is exact, so quantizing its result must give back the same
// 16 bits, for every one of the 65536 patterns.
TEST(Float16, AllPatternsSurviveExpandAndRequantize) {
  for (std::uint32_t h = 0; h <= 0xFFFFu; ++h) {
    const auto half = static_cast<std::uint16_t>(h);
    const float expanded = float16_bits_to_float32(half);
    // The only change allowed is the quieting of a signalling NaN.
    const std::uint16_t expected = is_nan_f16(half) ? static_cast<std::uint16_t>(half | 0x0200u) : half;
    ASSERT_EQ(f16(expanded), expected) << std::hex << h;
  }
}

TEST(Float16, ExpansionMatchesTheValueDefinition) {
  for (std::uint32_t h = 0; h <= 0x7BFFu; ++h) {  // all non-negative finite patterns
    const int exponent_field = static_cast<int>(h >> 10);
    const int mantissa = static_cast<int>(h & 0x3FFu);
    const double expected = exponent_field == 0 ? std::ldexp(mantissa, -24)
                                                : std::ldexp(1024 + mantissa, exponent_field - 25);
    ASSERT_EQ(static_cast<double>(float16_bits_to_float32(static_cast<std::uint16_t>(h))), expected) << std::hex << h;
    ASSERT_EQ(static_cast<double>(float16_bits_to_float32(static_cast<std::uint16_t>(h | 0x8000u))), -expected);
  }
  EXPECT_EQ(float16_bits_to_float32(0x7C00), kInf);
  EXPECT_EQ(float16_bits_to_float32(0xFC00), -kInf);
  EXPECT_TRUE(std::signbit(float16_bits_to_float32(0x8000)));
  // Signalling NaN is quieted, payload shifted into the top of the FP32 mantissa.
  EXPECT_EQ(to_bits(float16_bits_to_float32(0x7C01)), 0x7FC02000u);
}

// ---------------------------------------------------------------------------
// BFloat16
// ---------------------------------------------------------------------------

TEST(BFloat16, ExactlyRepresentableValues) {
  EXPECT_EQ(bf16(0.0F), 0x0000);
  EXPECT_EQ(bf16(-0.0F), 0x8000);
  EXPECT_EQ(bf16(1.0F), 0x3F80);
  EXPECT_EQ(bf16(-2.0F), 0xC000);
  EXPECT_EQ(bf16(kInf), 0x7F80);
  EXPECT_EQ(bf16(-kInf), 0xFF80);
  EXPECT_EQ(bf16(from_bits(0x7F7F0000u)), 0x7F7F);  // largest finite BFloat16
}

TEST(BFloat16, TiesRoundToEven) {
  EXPECT_EQ(bf16(from_bits(0x3F808000u)), 0x3F80);  // tie -> even (down)
  EXPECT_EQ(bf16(from_bits(0x3F818000u)), 0x3F82);  // tie -> even (up)
  EXPECT_EQ(bf16(from_bits(0x3F808001u)), 0x3F81);  // just above the tie
  EXPECT_EQ(bf16(from_bits(0x3F817FFFu)), 0x3F81);  // just below the tie
  EXPECT_EQ(bf16(from_bits(0xBF818000u)), 0xBF82);  // rounding acts on the magnitude
}

TEST(BFloat16, RoundingIsNotTruncation) {
  // Truncation would give 0x3F80; the nearest value is 0x3F81.
  EXPECT_EQ(bf16(from_bits(0x3F80FFFFu)), 0x3F81);
}

TEST(BFloat16, CarryAndOverflow) {
  EXPECT_EQ(bf16(std::nextafter(2.0F, 0.0F)), 0x4000);  // carry into the exponent
  // FLT_MAX is closer to 2^128 than to the largest BFloat16 value.
  EXPECT_EQ(bf16(std::numeric_limits<float>::max()), 0x7F80);
  EXPECT_EQ(bf16(std::numeric_limits<float>::lowest()), 0xFF80);
  EXPECT_EQ(bf16(from_bits(0x7F7F7FFFu)), 0x7F7F);  // just below the overflow tie
  EXPECT_EQ(bf16(from_bits(0x7F7F8000u)), 0x7F80);  // the tie itself goes to even = infinity
}

TEST(BFloat16, SubnormalsKeepTheirRange) {
  // Unlike binary16, the FP32 subnormal range is still representable.
  EXPECT_EQ(bf16(from_bits(0x00010000u)), 0x0001);
  EXPECT_EQ(bf16(std::numeric_limits<float>::denorm_min()), 0x0000);  // below half a step
  EXPECT_EQ(bf16(from_bits(0x00008001u)), 0x0001);
  EXPECT_EQ(bf16(from_bits(0x007FFFFFu)), 0x0080);  // largest subnormal rounds up to the smallest normal
}

TEST(BFloat16, NanStaysNan) {
  EXPECT_EQ(bf16(from_bits(0x7FC00000u)), 0x7FC0);
  EXPECT_EQ(bf16(from_bits(0xFFC00000u)), 0xFFC0);
  // Payload only in the discarded half: truncation alone would yield 0x7F80, infinity.
  EXPECT_EQ(bf16(from_bits(0x7F800001u)), 0x7FC0);
  // A NaN must not be "rounded": 0x7FFFFFFF + carry would wrap into the sign bit.
  EXPECT_EQ(bf16(from_bits(0x7FFFFFFFu)), 0x7FFF);
  EXPECT_EQ(bf16(from_bits(0xFFFFFFFFu)), 0xFFFF);
}

TEST(BFloat16, AllPatternsSurviveExpandAndRequantize) {
  for (std::uint32_t h = 0; h <= 0xFFFFu; ++h) {
    const auto half = static_cast<std::uint16_t>(h);
    ASSERT_EQ(to_bits(bfloat16_bits_to_float32(half)), h << 16);
    const std::uint16_t expected = is_nan_bf16(half) ? static_cast<std::uint16_t>(half | 0x0040u) : half;
    ASSERT_EQ(bf16(bfloat16_bits_to_float32(half)), expected) << std::hex << h;
  }
}

// ---------------------------------------------------------------------------
// Both formats against the independent value-based model
// ---------------------------------------------------------------------------

class AgainstReference : public ::testing::TestWithParam<QuantFormat> {
 protected:
  void check(std::uint32_t bits) const {
    const float x = from_bits(bits);
    const float actual = GetParam() == QuantFormat::Float16
                             ? float16_bits_to_float32(float32_to_float16_bits(x))
                             : bfloat16_bits_to_float32(float32_to_bfloat16_bits(x));
    if (std::isnan(x)) {
      ASSERT_TRUE(std::isnan(actual)) << std::hex << bits;
      ASSERT_EQ(std::signbit(actual), std::signbit(x)) << std::hex << bits;
      return;
    }
    const float expected = std::isinf(x) ? x : turbocore::testing::reference_round_trip(x, GetParam());
    ASSERT_EQ(to_bits(actual), to_bits(expected)) << "input bits 0x" << std::hex << bits;
  }
};

// Every exponent, with the mantissas where rounding decisions flip: all-zero,
// all-one, and each single-bit and tie pattern around every bit position.
TEST_P(AgainstReference, StructuredSweepOfEveryBinade) {
  std::vector<std::uint32_t> mantissas = {0x000000u, 0x7FFFFFu, 0x555555u, 0x2AAAAAu};
  for (int bit = 0; bit < 23; ++bit) {
    const std::uint32_t one = std::uint32_t{1} << bit;
    for (const std::uint32_t m : {one, one - 1, one + 1, one | (one >> 1), 0x7FFFFFu ^ one, 0x7FFFFFu - one}) {
      mantissas.push_back(m & 0x7FFFFFu);
    }
  }
  for (std::uint32_t exponent = 0; exponent <= 0xFFu; ++exponent) {
    for (const std::uint32_t mantissa : mantissas) {
      for (const std::uint32_t sign : {0u, 0x80000000u}) {
        check(sign | (exponent << 23) | mantissa);
      }
    }
  }
}

TEST_P(AgainstReference, RandomBitPatterns) {
  std::mt19937 rng(2024);
  for (int i = 0; i < 2'000'000; ++i) {
    check(static_cast<std::uint32_t>(rng()));
  }
}

// Random patterns almost never hit the narrow binary16 range, so sample it
// (and its underflow and overflow edges) densely.
TEST_P(AgainstReference, RandomValuesAcrossTheFloat16Range) {
  std::mt19937 rng(7);
  // Exponent fields 100..145 cover 2^-27 .. 2^18: below the smallest
  // subnormal to above the overflow threshold.
  std::uniform_int_distribution<std::uint32_t> exponent(100, 145);
  for (int i = 0; i < 2'000'000; ++i) {
    check((static_cast<std::uint32_t>(rng()) & 0x807FFFFFu) | (exponent(rng) << 23));
  }
}

INSTANTIATE_TEST_SUITE_P(Formats, AgainstReference, ::testing::Values(QuantFormat::Float16, QuantFormat::BFloat16),
                         [](const auto& info) {
                           return info.param == QuantFormat::Float16 ? "Float16" : "BFloat16";
                         });

// ---------------------------------------------------------------------------
// Array API
// ---------------------------------------------------------------------------

TEST(QuantizeArrays, MatchesTheScalarFunctions) {
  const std::vector<float> in = {0.0F, -1.5F, 3.14159F, 1e-30F, 1e30F, kInf, from_bits(0x7FC00000u)};
  for (const QuantFormat format : {QuantFormat::Float16, QuantFormat::BFloat16}) {
    std::vector<std::uint16_t> packed(in.size());
    std::vector<float> out(in.size());
    turbocore::quantize(in, packed, format);
    turbocore::dequantize(packed, out, format);
    const std::vector<float> round_trip = turbocore::quantize_round_trip(in, format);

    for (std::size_t i = 0; i < in.size(); ++i) {
      const std::uint16_t expected =
          format == QuantFormat::Float16 ? float32_to_float16_bits(in[i]) : float32_to_bfloat16_bits(in[i]);
      EXPECT_EQ(packed[i], expected);
      EXPECT_EQ(to_bits(out[i]), to_bits(round_trip[i]));
    }
  }
}

TEST(QuantizeArrays, QuantizationIsIdempotent) {
  std::mt19937 rng(5);
  std::vector<float> in(100000);
  for (float& x : in) {
    x = from_bits(static_cast<std::uint32_t>(rng()));
  }
  for (const QuantFormat format : {QuantFormat::Float16, QuantFormat::BFloat16}) {
    const std::vector<float> once = turbocore::quantize_round_trip(in, format);
    const std::vector<float> twice = turbocore::quantize_round_trip(once, format);
    for (std::size_t i = 0; i < in.size(); ++i) {
      ASSERT_EQ(to_bits(once[i]), to_bits(twice[i])) << std::hex << to_bits(in[i]);
    }
  }
}

TEST(QuantizeArrays, SizeMismatchIsRejected) {
  const std::vector<float> in(4);
  std::vector<std::uint16_t> out(3);
  EXPECT_THROW(turbocore::quantize(in, out, QuantFormat::Float16), turbocore::Error);
  std::vector<float> back(5);
  EXPECT_THROW(turbocore::dequantize(out, back, QuantFormat::BFloat16), turbocore::Error);
}

}  // namespace
