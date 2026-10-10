#include "turbocore/error_analysis.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <vector>

#include "detail/compensated_sum.hpp"
#include "turbocore/error.hpp"

namespace turbocore {

namespace {

constexpr std::uint32_t kSignBit = 0x80000000u;
constexpr std::uint32_t kMagnitudeMask = 0x7FFFFFFFu;
constexpr std::uint32_t kInfinityBits = 0x7F800000u;

// Classification on the bit pattern, so it cannot be optimised away or
// altered by floating-point compiler modes.
constexpr bool is_nan(std::uint32_t bits) noexcept { return (bits & kMagnitudeMask) > kInfinityBits; }
constexpr bool is_infinite(std::uint32_t bits) noexcept { return (bits & kMagnitudeMask) == kInfinityBits; }
constexpr bool is_zero(std::uint32_t bits) noexcept { return (bits & kMagnitudeMask) == 0; }

/// Position of a non-NaN float on the ordered float line. IEEE floats are
/// sign-magnitude, and for a fixed sign their bit patterns are ordered like
/// their values, so negating the magnitude for negative numbers gives a
/// monotonic integer with both zeros at 0.
constexpr std::int64_t ordered_index(std::uint32_t bits) noexcept {
  const auto magnitude = static_cast<std::int64_t>(bits & kMagnitudeMask);
  return (bits & kSignBit) != 0 ? -magnitude : magnitude;
}

struct FormatParameters {
  int precision;         // significand bits, including the implicit one
  int minimum_exponent;  // exponent of the smallest normal number
};

constexpr FormatParameters parameters(QuantFormat format) noexcept {
  return format == QuantFormat::Float16 ? FormatParameters{11, -14} : FormatParameters{8, -126};
}

double undefined() noexcept { return std::numeric_limits<double>::quiet_NaN(); }

}  // namespace

std::int64_t ulp_distance(float a, float b) noexcept {
  return ordered_index(std::bit_cast<std::uint32_t>(b)) - ordered_index(std::bit_cast<std::uint32_t>(a));
}

double target_ulp(float value, QuantFormat format) noexcept {
  const FormatParameters p = parameters(format);
  // The FP32 exponent field gives the binade directly. FP32 zeros and
  // subnormals (field 0) lie below the smallest normal of both target
  // formats, so they clamp to the subnormal spacing like everything else
  // down there.
  const int exponent_field = static_cast<int>((std::bit_cast<std::uint32_t>(value) >> 23) & 0xFFu);
  const int exponent = std::max(exponent_field - 127, p.minimum_exponent);
  return std::ldexp(1.0, exponent - (p.precision - 1));
}

ErrorReport analyze_error(std::span<const float> reference, std::span<const float> reconstructed,
                          const ErrorAnalysisOptions& options) {
  if (reference.size() != reconstructed.size()) {
    throw Error(ErrorCode::InvalidArgument, "reference and reconstructed data must have the same length");
  }
  if (options.peak_mode != PeakMode::MaxAbsReference && options.peak_mode != PeakMode::ReferenceRange &&
      options.peak_mode != PeakMode::Custom) {
    throw Error(ErrorCode::InvalidArgument, "unknown peak mode");
  }
  if (options.peak_mode == PeakMode::Custom && !(std::isfinite(options.custom_peak) && options.custom_peak > 0.0)) {
    throw Error(ErrorCode::InvalidArgument, "custom_peak must be finite and positive");
  }

  ErrorReport report;
  report.element_count = reference.size();
  report.peak_mode = options.peak_mode;

  detail::CompensatedSum squared_error_sum;
  detail::CompensatedSum absolute_error_sum;
  detail::CompensatedSum ulp_distance_sum;
  double reference_min = std::numeric_limits<double>::infinity();
  double reference_max = -std::numeric_limits<double>::infinity();
  double reference_max_abs = 0.0;

  for (std::size_t i = 0; i < reference.size(); ++i) {
    const auto ref_bits = std::bit_cast<std::uint32_t>(reference[i]);
    const auto rec_bits = std::bit_cast<std::uint32_t>(reconstructed[i]);

    if (ref_bits == rec_bits) {
      ++report.bit_exact_count;
    }

    if (is_nan(ref_bits)) {
      ++report.nan_reference_count;
      if (is_nan(rec_bits)) {
        ++report.nan_preserved_count;
      }
      continue;
    }
    if (is_infinite(ref_bits)) {
      ++report.infinity_reference_count;
      if (rec_bits == ref_bits) {
        ++report.infinity_preserved_count;
      }
      continue;
    }
    if (is_nan(rec_bits)) {
      ++report.nan_introduced_count;
      continue;
    }
    if (is_infinite(rec_bits)) {
      ++report.overflow_to_infinity_count;
      continue;
    }

    ++report.compared_count;
    if (((ref_bits ^ rec_bits) & kSignBit) != 0) {
      ++report.sign_mismatch_count;
    }
    if (!is_zero(ref_bits) && is_zero(rec_bits)) {
      ++report.underflow_to_zero_count;
    }

    // Both operands are floats, so their difference is exact in double
    // unless they are more than 2^29 apart in magnitude, where it is
    // correctly rounded.
    const auto ref = static_cast<double>(reference[i]);
    const double error = static_cast<double>(reconstructed[i]) - ref;
    const double absolute_error = std::fabs(error);

    squared_error_sum.add(error * error);
    absolute_error_sum.add(absolute_error);
    report.max_absolute_error = std::max(report.max_absolute_error, absolute_error);
    if (ref != 0.0) {
      report.max_relative_error = std::max(report.max_relative_error, absolute_error / std::fabs(ref));
    }

    const std::int64_t signed_distance = ordered_index(rec_bits) - ordered_index(ref_bits);
    const auto distance = static_cast<std::uint64_t>(signed_distance < 0 ? -signed_distance : signed_distance);
    report.max_ulp_distance = std::max(report.max_ulp_distance, distance);
    ulp_distance_sum.add(static_cast<double>(distance));

    reference_min = std::min(reference_min, ref);
    reference_max = std::max(reference_max, ref);
    reference_max_abs = std::max(reference_max_abs, std::fabs(ref));
  }

  if (report.compared_count == 0) {
    report.mean_squared_error = undefined();
    report.root_mean_squared_error = undefined();
    report.mean_absolute_error = undefined();
    report.max_absolute_error = undefined();
    report.max_relative_error = undefined();
    report.mean_ulp_distance = undefined();
    report.peak = options.peak_mode == PeakMode::Custom ? options.custom_peak : undefined();
    report.psnr_db = undefined();
    return report;
  }

  const auto count = static_cast<double>(report.compared_count);
  report.mean_squared_error = squared_error_sum.value() / count;
  report.root_mean_squared_error = std::sqrt(report.mean_squared_error);
  report.mean_absolute_error = absolute_error_sum.value() / count;
  report.mean_ulp_distance = ulp_distance_sum.value() / count;

  switch (options.peak_mode) {
    case PeakMode::MaxAbsReference:
      report.peak = reference_max_abs;
      break;
    case PeakMode::ReferenceRange:
      report.peak = reference_max - reference_min;
      break;
    case PeakMode::Custom:
      report.peak = options.custom_peak;
      break;
  }

  if (report.mean_squared_error == 0.0) {
    // A perfect reconstruction has unbounded PSNR whatever the peak is.
    report.psnr_db = std::numeric_limits<double>::infinity();
  } else {
    // 20 log10(peak) - 10 log10(MSE) rather than log10(peak^2 / MSE): the
    // square and the quotient can overflow or underflow for data near the
    // ends of the float range. A zero peak (all-zero reference) gives
    // -infinity, which is the honest answer.
    report.psnr_db = 20.0 * std::log10(report.peak) - 10.0 * std::log10(report.mean_squared_error);
  }
  return report;
}

QuantizationReport analyze_quantization(std::span<const float> reference, QuantFormat format,
                                        const ErrorAnalysisOptions& options) {
  const std::vector<float> reconstructed = quantize_round_trip(reference, format);

  QuantizationReport report;
  report.format = format;
  report.error = analyze_error(reference, reconstructed, options);

  detail::CompensatedSum ulp_error_sum;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const auto ref_bits = std::bit_cast<std::uint32_t>(reference[i]);
    const auto rec_bits = std::bit_cast<std::uint32_t>(reconstructed[i]);
    // Same "compared" class as analyze_error: both values finite.
    if ((ref_bits & kMagnitudeMask) >= kInfinityBits || (rec_bits & kMagnitudeMask) >= kInfinityBits) {
      continue;
    }

    // Exact: the difference of two nearby floats is exact in double, and the
    // divisor is a power of two.
    const double ulp_error = std::fabs(static_cast<double>(reconstructed[i]) - static_cast<double>(reference[i])) /
                             target_ulp(reference[i], format);

    ulp_error_sum.add(ulp_error);
    report.max_target_ulp_error = std::max(report.max_target_ulp_error, ulp_error);
    if (ulp_error == 0.0) {
      ++report.exactly_representable_count;
    }
    if (ulp_error > 0.5) {
      ++report.above_half_ulp_count;
      continue;
    }
    const double scaled = ulp_error * 2.0 * static_cast<double>(QuantizationReport::kHistogramBins);
    const auto bin = std::min(static_cast<std::size_t>(scaled), QuantizationReport::kHistogramBins - 1);
    ++report.target_ulp_histogram[bin];
  }

  if (report.error.compared_count == 0) {
    report.max_target_ulp_error = undefined();
    report.mean_target_ulp_error = undefined();
  } else {
    report.mean_target_ulp_error = ulp_error_sum.value() / static_cast<double>(report.error.compared_count);
  }
  return report;
}

}  // namespace turbocore
