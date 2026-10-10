#ifndef TURBOCORE_QUANTIZATION_HPP
#define TURBOCORE_QUANTIZATION_HPP

#include <cstdint>
#include <span>
#include <vector>

#include "turbocore/export.hpp"

// Lossy precision reduction of FP32 data to a 16-bit floating-point format.
//
// Both conversions round to nearest, ties to even (the IEEE 754 default), so
// the result is always the representable value closest to the input: the
// error is at most half a unit in the last place of the target format.
//
//                      sign  exponent  mantissa  precision  largest finite
//   IEEE binary32        1       8        23        24        3.4028e38
//   IEEE binary16        1       5        10        11        65504
//   BFloat16             1       8         7         8        3.3895e38
//
// Float16 keeps more mantissa but has a narrow range: magnitudes of 65520 or
// more overflow to infinity and magnitudes below 2^-24 lose precision or
// vanish. BFloat16 keeps the full FP32 range and only gives up mantissa.

namespace turbocore {

enum class QuantFormat : std::uint8_t {
  Float16,   ///< IEEE 754 binary16
  BFloat16,  ///< upper 16 bits of IEEE 754 binary32, correctly rounded
};

/// FP32 -> binary16 bit pattern.
///  - Overflow (|x| rounds to 65520 or more) gives +-infinity.
///  - Underflow is gradual through binary16 subnormals down to +-0.
///  - The sign is always preserved, including for zeros and NaNs.
///  - NaN stays NaN: the top 10 payload bits are kept and the quiet bit is
///    set, which is what hardware conversions (F16C, Arm FCVT) produce.
[[nodiscard]] TURBOCORE_EXPORT std::uint16_t float32_to_float16_bits(float value) noexcept;

/// binary16 bit pattern -> FP32. Exact for every non-NaN input. Signalling
/// NaNs are quieted, as hardware conversions do.
[[nodiscard]] TURBOCORE_EXPORT float float16_bits_to_float32(std::uint16_t bits) noexcept;

/// FP32 -> BFloat16 bit pattern.
///  - Only magnitudes that round past the largest BFloat16 value overflow.
///  - NaN stays NaN: the quiet bit is forced, because a NaN whose payload
///    lies entirely in the discarded 16 bits would otherwise become infinity.
[[nodiscard]] TURBOCORE_EXPORT std::uint16_t float32_to_bfloat16_bits(float value) noexcept;

/// BFloat16 bit pattern -> FP32: the 16 bits followed by 16 zero bits. Exact.
[[nodiscard]] TURBOCORE_EXPORT float bfloat16_bits_to_float32(std::uint16_t bits) noexcept;

/// Element-wise FP32 -> `format`.
/// Throws Error(InvalidArgument) if the spans differ in length.
TURBOCORE_EXPORT void quantize(std::span<const float> in, std::span<std::uint16_t> out, QuantFormat format);

/// Element-wise `format` -> FP32.
/// Throws Error(InvalidArgument) if the spans differ in length.
TURBOCORE_EXPORT void dequantize(std::span<const std::uint16_t> in, std::span<float> out, QuantFormat format);

/// quantize() followed by dequantize(): the FP32 values a consumer of the
/// quantized data would actually see.
[[nodiscard]] TURBOCORE_EXPORT std::vector<float> quantize_round_trip(std::span<const float> in, QuantFormat format);

}  // namespace turbocore

#endif  // TURBOCORE_QUANTIZATION_HPP
