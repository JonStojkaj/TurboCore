#include "turbocore/quantization.hpp"

#include <bit>
#include <cstddef>

#include "turbocore/error.hpp"

// Scalar reference quantizers, written against the bit layouts rather than
// through compiler float16 types, so the rounding and special-value behaviour
// is spelled out here and identical on every platform. The AVX2 and CUDA
// versions are tested bit for bit against these.

namespace turbocore {

namespace {

// binary32 layout
constexpr std::uint32_t kF32ExponentMask = 0xFFu;
constexpr std::uint32_t kF32MantissaMask = 0x007FFFFFu;
constexpr std::uint32_t kF32ImplicitBit = 0x00800000u;
constexpr std::uint32_t kF32QuietBit = 0x00400000u;
constexpr std::uint32_t kF32InfinityBits = 0x7F800000u;
constexpr int kF32ExponentBias = 127;

// binary16 layout
constexpr std::uint32_t kF16SignBit = 0x8000u;
constexpr std::uint32_t kF16InfinityBits = 0x7C00u;
constexpr std::uint32_t kF16QuietBit = 0x0200u;
constexpr std::uint32_t kF16MantissaMask = 0x03FFu;
constexpr int kF16ExponentBias = 15;
constexpr int kF16ExponentMax = 31;  // exponent field of infinity and NaN
constexpr int kF16MantissaBits = 10;

// Mantissa bits dropped going from binary32 to binary16.
constexpr int kF16DroppedBits = 23 - kF16MantissaBits;

// BFloat16 layout
constexpr std::uint32_t kBF16QuietBit = 0x0040u;

/// Round to nearest, ties to even: whether to add one to `kept` after
/// discarding the low `dropped_bits` bits, whose value was `remainder`.
constexpr bool rounds_up(std::uint32_t kept, std::uint32_t remainder, int dropped_bits) noexcept {
  const std::uint32_t halfway = std::uint32_t{1} << (dropped_bits - 1);
  return remainder > halfway || (remainder == halfway && (kept & 1u) != 0);
}

void require_same_size(std::size_t a, std::size_t b) {
  if (a != b) {
    throw Error(ErrorCode::InvalidArgument, "input and output spans must have the same length");
  }
}

}  // namespace

std::uint16_t float32_to_float16_bits(float value) noexcept {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (bits >> 16) & kF16SignBit;
  const std::uint32_t exponent_field = (bits >> 23) & kF32ExponentMask;
  const std::uint32_t mantissa = bits & kF32MantissaMask;

  if (exponent_field == kF32ExponentMask) {
    if (mantissa == 0) {
      return static_cast<std::uint16_t>(sign | kF16InfinityBits);
    }
    return static_cast<std::uint16_t>(sign | kF16InfinityBits | kF16QuietBit | (mantissa >> kF16DroppedBits));
  }

  // The binary16 exponent field this value would have with unlimited range.
  const int half_exponent = static_cast<int>(exponent_field) - kF32ExponentBias + kF16ExponentBias;

  if (half_exponent >= kF16ExponentMax) {
    return static_cast<std::uint16_t>(sign | kF16InfinityBits);
  }

  if (half_exponent <= 0) {
    // Result is a binary16 subnormal or zero. Below 2^-25 (half the smallest
    // subnormal) everything rounds to zero; this also covers FP32 zeros and
    // subnormals.
    if (half_exponent < -kF16MantissaBits) {
      return static_cast<std::uint16_t>(sign);
    }
    // Subnormals have no implicit bit, so the full 24-bit significand is
    // shifted into place: one extra position for half_exponent == 0, and one
    // more for every step below that.
    const std::uint32_t significand = mantissa | kF32ImplicitBit;
    const int dropped = kF16DroppedBits + 1 - half_exponent;  // 14 to 24
    std::uint32_t half = significand >> dropped;
    const std::uint32_t remainder = significand & ((std::uint32_t{1} << dropped) - 1);
    if (rounds_up(half, remainder, dropped)) {
      ++half;  // may carry into the smallest normal number, which is the correct encoding
    }
    return static_cast<std::uint16_t>(sign | half);
  }

  std::uint32_t half = (static_cast<std::uint32_t>(half_exponent) << kF16MantissaBits) | (mantissa >> kF16DroppedBits);
  const std::uint32_t remainder = mantissa & ((std::uint32_t{1} << kF16DroppedBits) - 1);
  if (rounds_up(half, remainder, kF16DroppedBits)) {
    // A mantissa carry increments the exponent, and from the top binade it
    // lands exactly on the infinity encoding: overflow needs no special case.
    ++half;
  }
  return static_cast<std::uint16_t>(sign | half);
}

float float16_bits_to_float32(std::uint16_t bits) noexcept {
  const std::uint32_t sign = (std::uint32_t{bits} & kF16SignBit) << 16;
  const std::uint32_t exponent_field = (std::uint32_t{bits} >> kF16MantissaBits) & 0x1Fu;
  std::uint32_t mantissa = std::uint32_t{bits} & kF16MantissaMask;

  if (exponent_field == static_cast<std::uint32_t>(kF16ExponentMax)) {
    const std::uint32_t quiet = mantissa != 0 ? kF32QuietBit : 0u;
    return std::bit_cast<float>(sign | kF32InfinityBits | quiet | (mantissa << kF16DroppedBits));
  }

  if (exponent_field == 0) {
    if (mantissa == 0) {
      return std::bit_cast<float>(sign);
    }
    // Subnormal: value = mantissa * 2^-24. Normalise so the leading one
    // becomes the implicit bit; every binary16 subnormal is a normal FP32.
    const int shift = kF16MantissaBits + 1 - static_cast<int>(std::bit_width(mantissa));
    mantissa = (mantissa << shift) & kF16MantissaMask;
    const auto exponent = static_cast<std::uint32_t>(kF32ExponentBias - kF16ExponentBias + 1 - shift);
    return std::bit_cast<float>(sign | (exponent << 23) | (mantissa << kF16DroppedBits));
  }

  const std::uint32_t exponent = exponent_field + static_cast<std::uint32_t>(kF32ExponentBias - kF16ExponentBias);
  return std::bit_cast<float>(sign | (exponent << 23) | (mantissa << kF16DroppedBits));
}

std::uint16_t float32_to_bfloat16_bits(float value) noexcept {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  std::uint32_t top = bits >> 16;

  const bool is_nan = (bits & ~(std::uint32_t{1} << 31)) > kF32InfinityBits;
  if (is_nan) {
    return static_cast<std::uint16_t>(top | kBF16QuietBit);
  }

  // The format is the top half of binary32, so rounding is a plain integer
  // increment of that half: a mantissa carry moves into the exponent, and
  // from the largest finite values it lands on the infinity encoding.
  // Infinity itself has a zero remainder and is unchanged.
  if (rounds_up(top, bits & 0xFFFFu, 16)) {
    ++top;
  }
  return static_cast<std::uint16_t>(top);
}

float bfloat16_bits_to_float32(std::uint16_t bits) noexcept {
  return std::bit_cast<float>(std::uint32_t{bits} << 16);
}

void quantize(std::span<const float> in, std::span<std::uint16_t> out, QuantFormat format) {
  require_same_size(in.size(), out.size());
  switch (format) {
    case QuantFormat::Float16:
      for (std::size_t i = 0; i < in.size(); ++i) {
        out[i] = float32_to_float16_bits(in[i]);
      }
      return;
    case QuantFormat::BFloat16:
      for (std::size_t i = 0; i < in.size(); ++i) {
        out[i] = float32_to_bfloat16_bits(in[i]);
      }
      return;
  }
  throw Error(ErrorCode::InvalidArgument, "unknown quantization format");
}

void dequantize(std::span<const std::uint16_t> in, std::span<float> out, QuantFormat format) {
  require_same_size(in.size(), out.size());
  switch (format) {
    case QuantFormat::Float16:
      for (std::size_t i = 0; i < in.size(); ++i) {
        out[i] = float16_bits_to_float32(in[i]);
      }
      return;
    case QuantFormat::BFloat16:
      for (std::size_t i = 0; i < in.size(); ++i) {
        out[i] = bfloat16_bits_to_float32(in[i]);
      }
      return;
  }
  throw Error(ErrorCode::InvalidArgument, "unknown quantization format");
}

std::vector<float> quantize_round_trip(std::span<const float> in, QuantFormat format) {
  std::vector<float> out(in.size());
  switch (format) {
    case QuantFormat::Float16:
      for (std::size_t i = 0; i < in.size(); ++i) {
        out[i] = float16_bits_to_float32(float32_to_float16_bits(in[i]));
      }
      return out;
    case QuantFormat::BFloat16:
      for (std::size_t i = 0; i < in.size(); ++i) {
        out[i] = bfloat16_bits_to_float32(float32_to_bfloat16_bits(in[i]));
      }
      return out;
  }
  throw Error(ErrorCode::InvalidArgument, "unknown quantization format");
}

}  // namespace turbocore
