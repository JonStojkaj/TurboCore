#ifndef TURBOCORE_DETAIL_KERNEL_TABLE_HPP
#define TURBOCORE_DETAIL_KERNEL_TABLE_HPP

#include <cstddef>
#include <cstdint>

// The data-parallel kernels of the library as a table of function pointers,
// one table per backend. A backend is selected once per operation and its
// kernels are then called on whole blocks, so the indirect call is paid per
// block, never per element.
//
// Contract for every backend: output must be bit-identical to the scalar
// table for every input, including lengths that are not a multiple of the
// vector width. Pointers have the natural alignment of their element type
// and nothing more: no kernel may assume vector alignment. No kernel reads
// or writes outside the ranges implied by its arguments.
//
// This header is included by translation units compiled with backend-specific
// instruction-set flags. It must contain declarations only: no inline
// functions and no standard-library templates.

namespace turbocore::detail {

struct KernelTable {
  const char* name;

  // Predictors, in place on `count` elements. See detail/transforms.hpp.
  void (*delta_encode_u32)(std::uint32_t* values, std::size_t count) noexcept;
  void (*delta_decode_u32)(std::uint32_t* values, std::size_t count) noexcept;
  void (*xor_encode_u32)(std::uint32_t* values, std::size_t count) noexcept;
  void (*xor_decode_u32)(std::uint32_t* values, std::size_t count) noexcept;
  void (*delta_encode_u16)(std::uint16_t* values, std::size_t count) noexcept;
  void (*delta_decode_u16)(std::uint16_t* values, std::size_t count) noexcept;
  void (*xor_encode_u16)(std::uint16_t* values, std::size_t count) noexcept;
  void (*xor_decode_u16)(std::uint16_t* values, std::size_t count) noexcept;

  // Byte-plane transposition of `count` elements; the byte buffer holds
  // count * sizeof(element) bytes and does not overlap the element buffer.
  void (*shuffle_u32)(const std::uint32_t* in, std::uint8_t* out, std::size_t count) noexcept;
  void (*unshuffle_u32)(const std::uint8_t* in, std::uint32_t* out, std::size_t count) noexcept;
  void (*shuffle_u16)(const std::uint16_t* in, std::uint8_t* out, std::size_t count) noexcept;
  void (*unshuffle_u16)(const std::uint8_t* in, std::uint16_t* out, std::size_t count) noexcept;

  // Element-wise conversions of `count` elements. See turbocore/quantization.hpp.
  void (*quantize_f16)(const float* in, std::uint16_t* out, std::size_t count) noexcept;
  void (*dequantize_f16)(const std::uint16_t* in, float* out, std::size_t count) noexcept;
  void (*quantize_bf16)(const float* in, std::uint16_t* out, std::size_t count) noexcept;
  void (*dequantize_bf16)(const std::uint16_t* in, float* out, std::size_t count) noexcept;

  // CRC-32C (Castagnoli) of `size` bytes.
  std::uint32_t (*crc32c)(const std::uint8_t* data, std::size_t size) noexcept;
};

}  // namespace turbocore::detail

#endif  // TURBOCORE_DETAIL_KERNEL_TABLE_HPP
