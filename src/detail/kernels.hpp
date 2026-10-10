#ifndef TURBOCORE_DETAIL_KERNELS_HPP
#define TURBOCORE_DETAIL_KERNELS_HPP

#include <cstdint>
#include <span>

#include "detail/kernel_table.hpp"
#include "turbocore/backend.hpp"
#include "turbocore/quantization.hpp"

// Backend selection, and span-typed overloads over a KernelTable so that
// code templated on the element width can call the right kernel by overload
// resolution.

namespace turbocore::detail {

[[nodiscard]] const KernelTable& scalar_kernels() noexcept;

/// The AVX2 table, or nullptr if the backend was not compiled in or the CPU
/// cannot run it. The CPU check runs once.
[[nodiscard]] const KernelTable* avx2_kernels() noexcept;

/// True if the GPU transform stage was compiled in and a device is present.
[[nodiscard]] bool cuda_available() noexcept;

/// The CPU kernels to use with `backend` (for Backend::Cuda: the best CPU
/// kernels, for the work that stays on the host).
/// Throws Error(BackendUnavailable) if `backend` cannot run here.
[[nodiscard]] const KernelTable& select_kernels(Backend backend);

inline void delta_encode(const KernelTable& k, std::span<std::uint32_t> v) noexcept {
  k.delta_encode_u32(v.data(), v.size());
}
inline void delta_encode(const KernelTable& k, std::span<std::uint16_t> v) noexcept {
  k.delta_encode_u16(v.data(), v.size());
}
inline void delta_decode(const KernelTable& k, std::span<std::uint32_t> v) noexcept {
  k.delta_decode_u32(v.data(), v.size());
}
inline void delta_decode(const KernelTable& k, std::span<std::uint16_t> v) noexcept {
  k.delta_decode_u16(v.data(), v.size());
}
inline void xor_encode(const KernelTable& k, std::span<std::uint32_t> v) noexcept {
  k.xor_encode_u32(v.data(), v.size());
}
inline void xor_encode(const KernelTable& k, std::span<std::uint16_t> v) noexcept {
  k.xor_encode_u16(v.data(), v.size());
}
inline void xor_decode(const KernelTable& k, std::span<std::uint32_t> v) noexcept {
  k.xor_decode_u32(v.data(), v.size());
}
inline void xor_decode(const KernelTable& k, std::span<std::uint16_t> v) noexcept {
  k.xor_decode_u16(v.data(), v.size());
}

inline void shuffle_bytes(const KernelTable& k, std::span<const std::uint32_t> in, std::span<std::uint8_t> out) noexcept {
  k.shuffle_u32(in.data(), out.data(), in.size());
}
inline void shuffle_bytes(const KernelTable& k, std::span<const std::uint16_t> in, std::span<std::uint8_t> out) noexcept {
  k.shuffle_u16(in.data(), out.data(), in.size());
}
inline void unshuffle_bytes(const KernelTable& k, std::span<const std::uint8_t> in,
                            std::span<std::uint32_t> out) noexcept {
  k.unshuffle_u32(in.data(), out.data(), out.size());
}
inline void unshuffle_bytes(const KernelTable& k, std::span<const std::uint8_t> in,
                            std::span<std::uint16_t> out) noexcept {
  k.unshuffle_u16(in.data(), out.data(), out.size());
}

/// Requires in.size() == out.size().
inline void quantize(const KernelTable& k, std::span<const float> in, std::span<std::uint16_t> out,
                     QuantFormat format) noexcept {
  (format == QuantFormat::Float16 ? k.quantize_f16 : k.quantize_bf16)(in.data(), out.data(), in.size());
}
inline void dequantize(const KernelTable& k, std::span<const std::uint16_t> in, std::span<float> out,
                       QuantFormat format) noexcept {
  (format == QuantFormat::Float16 ? k.dequantize_f16 : k.dequantize_bf16)(in.data(), out.data(), in.size());
}

}  // namespace turbocore::detail

#endif  // TURBOCORE_DETAIL_KERNELS_HPP
