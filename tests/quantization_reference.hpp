#ifndef TURBOCORE_TESTS_QUANTIZATION_REFERENCE_HPP
#define TURBOCORE_TESTS_QUANTIZATION_REFERENCE_HPP

#include <algorithm>
#include <cmath>
#include <limits>

#include "turbocore/quantization.hpp"

// An independent model of correct rounding to a small binary float format.
//
// The library quantizers manipulate bit fields. This model shares no code
// and no approach with them: it works on the *value*, in double precision,
// using only the definition of the format (precision p, exponent range
// [emin, emax]):
//
//     ulp     = 2^(max(floor(log2|x|), emin) - (p - 1))
//     rounded = nearbyint(x / ulp) * ulp        (ties to even)
//     result  = +-infinity if |rounded| >= 2^(emax + 1)
//
// Every step is exact in double: x / ulp divides by a power of two, and the
// quotient needs at most 24 significant bits. std::nearbyint rounds in the
// current rounding mode, which the tests never change from the default
// round-to-nearest-even.

namespace turbocore::testing {

struct FormatModel {
  int precision;
  int minimum_exponent;
  int maximum_exponent;
};

constexpr FormatModel model_of(QuantFormat format) {
  return format == QuantFormat::Float16 ? FormatModel{11, -14, 15} : FormatModel{8, -126, 127};
}

/// The FP32 value a correct quantize -> dequantize round trip must produce
/// for the finite input `x`.
inline float reference_round_trip(float x, QuantFormat format) {
  if (x == 0.0F) {
    return x;  // keeps the sign of zero
  }
  const FormatModel model = model_of(format);
  const auto value = static_cast<double>(x);
  const int exponent = std::max(std::ilogb(value), model.minimum_exponent);
  const double ulp = std::ldexp(1.0, exponent - (model.precision - 1));
  const double rounded = std::nearbyint(value / ulp) * ulp;  // -0.0 when a negative value rounds to zero
  if (std::fabs(rounded) >= std::ldexp(1.0, model.maximum_exponent + 1)) {
    return std::copysign(std::numeric_limits<float>::infinity(), x);
  }
  return static_cast<float>(rounded);  // exact: every target value is an FP32 value
}

}  // namespace turbocore::testing

#endif  // TURBOCORE_TESTS_QUANTIZATION_REFERENCE_HPP
