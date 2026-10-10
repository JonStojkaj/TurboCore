#ifndef TURBOCORE_DETAIL_COMPENSATED_SUM_HPP
#define TURBOCORE_DETAIL_COMPENSATED_SUM_HPP

#include <cmath>

namespace turbocore::detail {

/// Kahan-Babuska-Neumaier compensated summation.
///
/// A plain double accumulator loses every term smaller than half an ULP of
/// the running total, which for error metrics over 10^8+ elements, or data
/// with one large outlier, is most of the terms. This keeps the rounding
/// error of each addition in a second double and adds it back at the end, so
/// the result is accurate to about one ULP regardless of length or ordering.
///
/// Correctness depends on the compiler not reassociating floating-point
/// arithmetic; the build never enables -ffast-math.
class CompensatedSum {
 public:
  void add(double term) noexcept {
    const double total = sum_ + term;
    // Recover what the addition rounded away, from whichever operand was
    // large enough for the subtraction to be exact.
    if (std::fabs(sum_) >= std::fabs(term)) {
      compensation_ += (sum_ - total) + term;
    } else {
      compensation_ += (term - total) + sum_;
    }
    sum_ = total;
  }

  [[nodiscard]] double value() const noexcept { return sum_ + compensation_; }

 private:
  double sum_ = 0.0;
  double compensation_ = 0.0;
};

}  // namespace turbocore::detail

#endif  // TURBOCORE_DETAIL_COMPENSATED_SUM_HPP
