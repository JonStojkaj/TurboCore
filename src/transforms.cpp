#include "detail/transforms.hpp"

#include <concepts>
#include <cstddef>

namespace turbocore::detail {

namespace {

// The casts back to T make the modulo-2^width wrap explicit: for 16-bit
// elements the arithmetic itself is carried out in int.

template <std::unsigned_integral T>
void delta_encode_impl(std::span<T> values) noexcept {
  // Backwards, so each element still sees its original predecessor.
  for (std::size_t i = values.size(); i-- > 1;) {
    values[i] = static_cast<T>(values[i] - values[i - 1]);
  }
}

template <std::unsigned_integral T>
void delta_decode_impl(std::span<T> values) noexcept {
  for (std::size_t i = 1; i < values.size(); ++i) {
    values[i] = static_cast<T>(values[i] + values[i - 1]);
  }
}

template <std::unsigned_integral T>
void xor_encode_impl(std::span<T> values) noexcept {
  for (std::size_t i = values.size(); i-- > 1;) {
    values[i] = static_cast<T>(values[i] ^ values[i - 1]);
  }
}

template <std::unsigned_integral T>
void xor_decode_impl(std::span<T> values) noexcept {
  for (std::size_t i = 1; i < values.size(); ++i) {
    values[i] = static_cast<T>(values[i] ^ values[i - 1]);
  }
}

template <std::unsigned_integral T>
void shuffle_impl(std::span<const T> in, std::span<std::uint8_t> out) noexcept {
  const std::size_t n = in.size();
  for (std::size_t plane = 0; plane < sizeof(T); ++plane) {
    std::uint8_t* dst = out.data() + plane * n;
    const unsigned shift = static_cast<unsigned>(plane) * 8;
    for (std::size_t i = 0; i < n; ++i) {
      dst[i] = static_cast<std::uint8_t>((in[i] >> shift) & 0xFFu);
    }
  }
}

template <std::unsigned_integral T>
void unshuffle_impl(std::span<const std::uint8_t> in, std::span<T> out) noexcept {
  const std::size_t n = out.size();
  for (std::size_t i = 0; i < n; ++i) {
    std::uint32_t value = 0;
    for (std::size_t plane = 0; plane < sizeof(T); ++plane) {
      value |= static_cast<std::uint32_t>(in[plane * n + i]) << (plane * 8);
    }
    out[i] = static_cast<T>(value);
  }
}

}  // namespace

void delta_encode(std::span<std::uint32_t> values) noexcept { delta_encode_impl(values); }
void delta_encode(std::span<std::uint16_t> values) noexcept { delta_encode_impl(values); }
void delta_decode(std::span<std::uint32_t> values) noexcept { delta_decode_impl(values); }
void delta_decode(std::span<std::uint16_t> values) noexcept { delta_decode_impl(values); }

void xor_encode(std::span<std::uint32_t> values) noexcept { xor_encode_impl(values); }
void xor_encode(std::span<std::uint16_t> values) noexcept { xor_encode_impl(values); }
void xor_decode(std::span<std::uint32_t> values) noexcept { xor_decode_impl(values); }
void xor_decode(std::span<std::uint16_t> values) noexcept { xor_decode_impl(values); }

void shuffle_bytes(std::span<const std::uint32_t> in, std::span<std::uint8_t> out) noexcept { shuffle_impl(in, out); }
void shuffle_bytes(std::span<const std::uint16_t> in, std::span<std::uint8_t> out) noexcept { shuffle_impl(in, out); }
void unshuffle_bytes(std::span<const std::uint8_t> in, std::span<std::uint32_t> out) noexcept {
  unshuffle_impl(in, out);
}
void unshuffle_bytes(std::span<const std::uint8_t> in, std::span<std::uint16_t> out) noexcept {
  unshuffle_impl(in, out);
}

}  // namespace turbocore::detail
