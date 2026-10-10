#ifndef TURBOCORE_DETAIL_TRANSFORMS_HPP
#define TURBOCORE_DETAIL_TRANSFORMS_HPP

#include <cstdint>
#include <span>

// Scalar reference transforms. These are the kernels the AVX2 and CUDA
// backends reimplement, and their output is the definition those backends are
// tested against bit for bit.
//
// Each transform exists for 32-bit elements (FP32 data) and for 16-bit
// elements (FP16 / BFloat16 quantized data).

namespace turbocore::detail {

/// In place: v[i] <- v[i] - v[i-1] (modulo 2^width), v[0] unchanged.
void delta_encode(std::span<std::uint32_t> values) noexcept;
void delta_encode(std::span<std::uint16_t> values) noexcept;
/// Inverse of delta_encode: a prefix sum modulo 2^width.
void delta_decode(std::span<std::uint32_t> values) noexcept;
void delta_decode(std::span<std::uint16_t> values) noexcept;

/// In place: v[i] <- v[i] ^ v[i-1], v[0] unchanged.
void xor_encode(std::span<std::uint32_t> values) noexcept;
void xor_encode(std::span<std::uint16_t> values) noexcept;
void xor_decode(std::span<std::uint32_t> values) noexcept;
void xor_decode(std::span<std::uint16_t> values) noexcept;

/// Byte-plane transposition: out[k * n + i] = byte k (little-endian) of
/// in[i]. Requires out.size() == sizeof(element) * in.size().
void shuffle_bytes(std::span<const std::uint32_t> in, std::span<std::uint8_t> out) noexcept;
void shuffle_bytes(std::span<const std::uint16_t> in, std::span<std::uint8_t> out) noexcept;
/// Inverse of shuffle_bytes. Requires in.size() == sizeof(element) * out.size().
void unshuffle_bytes(std::span<const std::uint8_t> in, std::span<std::uint32_t> out) noexcept;
void unshuffle_bytes(std::span<const std::uint8_t> in, std::span<std::uint16_t> out) noexcept;

}  // namespace turbocore::detail

#endif  // TURBOCORE_DETAIL_TRANSFORMS_HPP
