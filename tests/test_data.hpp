#ifndef TURBOCORE_TESTS_TEST_DATA_HPP
#define TURBOCORE_TESTS_TEST_DATA_HPP

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

// Deterministic FP32 datasets shared by the tests.

namespace turbocore::testing {

/// True if both arrays hold the same bit patterns. Deliberately not a float
/// comparison: NaN != NaN and +0 == -0 would both hide real differences.
inline bool bit_identical(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

inline std::vector<float> constant(std::size_t n, float value) { return std::vector<float>(n, value); }

inline std::vector<float> ramp(std::size_t n) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = static_cast<float>(i) * 0.25F;
  }
  return v;
}

/// A smooth signal, the kind of data the predictor is designed for.
inline std::vector<float> smooth_wave(std::size_t n) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double t = static_cast<double>(i) * 1e-3;
    v[i] = static_cast<float>(1000.0 + 40.0 * std::sin(t) + 3.0 * std::sin(17.0 * t));
  }
  return v;
}

inline std::vector<float> uniform_noise(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<float> v(n);
  for (float& x : v) {
    x = dist(rng);
  }
  return v;
}

/// Every one of the 2^32 bit patterns is equally likely: NaNs with arbitrary
/// payloads, infinities, subnormals and all. Incompressible by construction.
inline std::vector<float> random_bit_patterns(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<float> v(n);
  for (float& x : v) {
    x = std::bit_cast<float>(static_cast<std::uint32_t>(rng()));
  }
  return v;
}

/// The values a float pipeline is most likely to mangle, repeated to length n.
inline std::vector<float> special_values(std::size_t n) {
  using limits = std::numeric_limits<float>;
  const std::vector<float> specials = {
      0.0F,
      -0.0F,
      limits::infinity(),
      -limits::infinity(),
      limits::quiet_NaN(),
      -limits::quiet_NaN(),
      std::bit_cast<float>(std::uint32_t{0x7FC00001u}),  // quiet NaN with payload
      std::bit_cast<float>(std::uint32_t{0x7F800001u}),  // signalling NaN
      std::bit_cast<float>(std::uint32_t{0xFFBFFFFFu}),  // negative signalling NaN, full payload
      limits::denorm_min(),
      -limits::denorm_min(),
      std::bit_cast<float>(std::uint32_t{0x007FFFFFu}),  // largest subnormal
      limits::min(),
      limits::max(),
      limits::lowest(),
      limits::epsilon(),
      1.0F,
      std::nextafter(1.0F, 2.0F),
  };
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = specials[i % specials.size()];
  }
  return v;
}

}  // namespace turbocore::testing

#endif  // TURBOCORE_TESTS_TEST_DATA_HPP
