#ifndef TURBOCORE_DETAIL_TRANSFORMS_HPP
#define TURBOCORE_DETAIL_TRANSFORMS_HPP

#include <cstdint>
#include <span>

// Scalar reference transforms. These are the kernels the AVX2 and CUDA
// backends reimplement, and their output is the definition those backends are
// tested against bit for bit.

namespace turbocore::detail {

/// In place: v[i] <- v[i] - v[i-1] (mod 2^32), v[0] unchanged.
void delta_encode(std::span<std::uint32_t> values) noexcept;
/// Inverse of delta_encode: a prefix sum modulo 2^32.
void delta_decode(std::span<std::uint32_t> values) noexcept;

/// In place: v[i] <- v[i] ^ v[i-1], v[0] unchanged.
void xor_encode(std::span<std::uint32_t> values) noexcept;
void xor_decode(std::span<std::uint32_t> values) noexcept;

/// Byte-plane transposition: out[k * n + i] = byte k (little-endian) of
/// in[i]. Requires out.size() == 4 * in.size().
void shuffle_bytes(std::span<const std::uint32_t> in, std::span<std::uint8_t> out) noexcept;
/// Inverse of shuffle_bytes. Requires in.size() == 4 * out.size().
void unshuffle_bytes(std::span<const std::uint8_t> in, std::span<std::uint32_t> out) noexcept;

}  // namespace turbocore::detail

#endif  // TURBOCORE_DETAIL_TRANSFORMS_HPP
