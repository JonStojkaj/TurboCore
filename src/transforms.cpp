#include "detail/transforms.hpp"

#include <cstddef>

namespace turbocore::detail {

void delta_encode(std::span<std::uint32_t> values) noexcept {
  // Backwards, so each element still sees its original predecessor.
  for (std::size_t i = values.size(); i-- > 1;) {
    values[i] -= values[i - 1];
  }
}

void delta_decode(std::span<std::uint32_t> values) noexcept {
  for (std::size_t i = 1; i < values.size(); ++i) {
    values[i] += values[i - 1];
  }
}

void xor_encode(std::span<std::uint32_t> values) noexcept {
  for (std::size_t i = values.size(); i-- > 1;) {
    values[i] ^= values[i - 1];
  }
}

void xor_decode(std::span<std::uint32_t> values) noexcept {
  for (std::size_t i = 1; i < values.size(); ++i) {
    values[i] ^= values[i - 1];
  }
}

void shuffle_bytes(std::span<const std::uint32_t> in, std::span<std::uint8_t> out) noexcept {
  const std::size_t n = in.size();
  for (std::size_t plane = 0; plane < 4; ++plane) {
    std::uint8_t* dst = out.data() + plane * n;
    const unsigned shift = static_cast<unsigned>(plane) * 8;
    for (std::size_t i = 0; i < n; ++i) {
      dst[i] = static_cast<std::uint8_t>((in[i] >> shift) & 0xFFu);
    }
  }
}

void unshuffle_bytes(std::span<const std::uint8_t> in, std::span<std::uint32_t> out) noexcept {
  const std::size_t n = out.size();
  const std::uint8_t* p0 = in.data();
  const std::uint8_t* p1 = p0 + n;
  const std::uint8_t* p2 = p1 + n;
  const std::uint8_t* p3 = p2 + n;
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = static_cast<std::uint32_t>(p0[i]) | static_cast<std::uint32_t>(p1[i]) << 8 |
             static_cast<std::uint32_t>(p2[i]) << 16 | static_cast<std::uint32_t>(p3[i]) << 24;
  }
}

}  // namespace turbocore::detail
