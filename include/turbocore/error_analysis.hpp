#ifndef TURBOCORE_ERROR_ANALYSIS_HPP
#define TURBOCORE_ERROR_ANALYSIS_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "turbocore/export.hpp"
#include "turbocore/quantization.hpp"

// Error metrics for lossy FP32 pipelines.
//
// Every element pair (reference, reconstructed) falls into exactly one class:
//
//   reference NaN       -> counted under nan_*, excluded from numeric metrics
//   reference infinite  -> counted under infinity_*, excluded
//   reference finite, reconstructed NaN or infinite
//                       -> counted as nan_introduced / overflow_to_infinity,
//                          excluded (the error is not a number)
//   both finite         -> "compared": contributes to every numeric metric
//
// Non-finite values are therefore reported, never silently folded into an
// average they would turn into NaN or infinity.

namespace turbocore {

/// The signal peak used in PSNR = 10 * log10(peak^2 / MSE). Integer images
/// have a natural peak (255); floating-point data does not, so the choice is
/// explicit and recorded in the report.
enum class PeakMode : std::uint8_t {
  MaxAbsReference,  ///< largest |reference| among compared elements
  ReferenceRange,   ///< max(reference) - min(reference) among compared elements
  Custom,           ///< ErrorAnalysisOptions::custom_peak
};

struct ErrorAnalysisOptions {
  PeakMode peak_mode = PeakMode::MaxAbsReference;
  double custom_peak = 1.0;  ///< used only with PeakMode::Custom; must be finite and positive
};

struct ErrorReport {
  // --- Element accounting ---------------------------------------------------
  std::uint64_t element_count = 0;
  std::uint64_t compared_count = 0;   ///< both values finite
  std::uint64_t bit_exact_count = 0;  ///< identical bit patterns, any class

  std::uint64_t nan_reference_count = 0;
  std::uint64_t nan_preserved_count = 0;    ///< NaN in, NaN out (payload not compared)
  std::uint64_t nan_introduced_count = 0;   ///< finite in, NaN out
  std::uint64_t infinity_reference_count = 0;
  std::uint64_t infinity_preserved_count = 0;      ///< same-signed infinity out
  std::uint64_t overflow_to_infinity_count = 0;    ///< finite in, infinity out
  std::uint64_t underflow_to_zero_count = 0;       ///< finite non-zero in, zero out (still compared)
  std::uint64_t sign_mismatch_count = 0;           ///< compared pairs whose sign bits differ, e.g. +0 vs -0

  // --- Numeric metrics over compared elements -------------------------------
  // With compared_count == 0 these are NaN (undefined), not zero.
  double mean_squared_error = 0.0;
  double root_mean_squared_error = 0.0;
  double mean_absolute_error = 0.0;
  double max_absolute_error = 0.0;
  /// max |error| / |reference| over compared elements with reference != 0.
  double max_relative_error = 0.0;

  PeakMode peak_mode = PeakMode::MaxAbsReference;
  double peak = 0.0;
  /// In dB. +infinity when the mean squared error is exactly zero.
  double psnr_db = 0.0;

  // --- Error in FP32 units in the last place --------------------------------
  // The number of representable FP32 values between reference and
  // reconstruction: 0 means identical (or +0 vs -0), 1 means adjacent floats.
  // Unlike absolute or relative error this is uniform across magnitudes.
  std::uint64_t max_ulp_distance = 0;
  double mean_ulp_distance = 0.0;
};

/// Compares a reconstruction against its reference.
/// Throws Error(InvalidArgument) if the spans differ in length or the custom
/// peak is not finite and positive.
[[nodiscard]] TURBOCORE_EXPORT ErrorReport analyze_error(std::span<const float> reference,
                                                         std::span<const float> reconstructed,
                                                         const ErrorAnalysisOptions& options = {});

struct QuantizationReport {
  QuantFormat format = QuantFormat::Float16;

  /// Generic metrics of the quantize -> dequantize round trip.
  ErrorReport error;

  // --- Error in units in the last place of the *target* format ---------------
  // |error| divided by the spacing of target-format values around the
  // reference. Correct rounding guarantees at most 0.5 for every compared
  // element; above_half_ulp_count is that guarantee, checked on this data.
  double max_target_ulp_error = 0.0;
  double mean_target_ulp_error = 0.0;
  std::uint64_t above_half_ulp_count = 0;
  /// Reference values the target format represents exactly (zero error).
  std::uint64_t exactly_representable_count = 0;

  /// Distribution of the target-ULP error over compared elements, in
  /// kHistogramBins equal bins covering [0, 0.5]; bin i holds errors in
  /// [i, i + 1) / (2 * kHistogramBins), the last bin also holds exactly 0.5.
  /// Errors above 0.5 appear only in above_half_ulp_count.
  /// Uniformly spread mantissas give a flat histogram; structure in it means
  /// the data has fewer significant bits than FP32 can hold.
  static constexpr std::size_t kHistogramBins = 8;
  std::array<std::uint64_t, kHistogramBins> target_ulp_histogram{};
};

/// Quantizes `reference` to `format`, converts back, and measures the damage.
[[nodiscard]] TURBOCORE_EXPORT QuantizationReport analyze_quantization(std::span<const float> reference,
                                                                       QuantFormat format,
                                                                       const ErrorAnalysisOptions& options = {});

/// Spacing between adjacent `format` values in the binade containing `value`
/// (the spacing of subnormals for magnitudes below the smallest normal, zero
/// included). `value` must be finite.
[[nodiscard]] TURBOCORE_EXPORT double target_ulp(float value, QuantFormat format) noexcept;

/// Signed count of representable FP32 values from `a` to `b` on the ordered
/// float line, with +0 and -0 at the same position. Neither may be NaN.
[[nodiscard]] TURBOCORE_EXPORT std::int64_t ulp_distance(float a, float b) noexcept;

}  // namespace turbocore

#endif  // TURBOCORE_ERROR_ANALYSIS_HPP
