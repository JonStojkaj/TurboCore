#include "detail/kernels.hpp"

#include <span>

#include "detail/cpu_features.hpp"
#include "detail/crc32c.hpp"
#include "detail/transforms.hpp"
#include "turbocore/config.hpp"
#include "turbocore/error.hpp"

#if TURBOCORE_HAS_AVX2
#include "backends/avx2/kernels_avx2.hpp"
#include "backends/avx2/probe_avx2.hpp"
#endif

namespace turbocore {

namespace detail {

namespace {

// Adapters from the pointer + count kernel signatures to the span-based
// scalar reference functions.
namespace scalar {

void delta_encode_u32(std::uint32_t* v, std::size_t n) noexcept { delta_encode(std::span(v, n)); }
void delta_decode_u32(std::uint32_t* v, std::size_t n) noexcept { delta_decode(std::span(v, n)); }
void xor_encode_u32(std::uint32_t* v, std::size_t n) noexcept { xor_encode(std::span(v, n)); }
void xor_decode_u32(std::uint32_t* v, std::size_t n) noexcept { xor_decode(std::span(v, n)); }
void delta_encode_u16(std::uint16_t* v, std::size_t n) noexcept { delta_encode(std::span(v, n)); }
void delta_decode_u16(std::uint16_t* v, std::size_t n) noexcept { delta_decode(std::span(v, n)); }
void xor_encode_u16(std::uint16_t* v, std::size_t n) noexcept { xor_encode(std::span(v, n)); }
void xor_decode_u16(std::uint16_t* v, std::size_t n) noexcept { xor_decode(std::span(v, n)); }

void shuffle_u32(const std::uint32_t* in, std::uint8_t* out, std::size_t n) noexcept {
  shuffle_bytes(std::span(in, n), std::span(out, n * sizeof(std::uint32_t)));
}
void unshuffle_u32(const std::uint8_t* in, std::uint32_t* out, std::size_t n) noexcept {
  unshuffle_bytes(std::span(in, n * sizeof(std::uint32_t)), std::span(out, n));
}
void shuffle_u16(const std::uint16_t* in, std::uint8_t* out, std::size_t n) noexcept {
  shuffle_bytes(std::span(in, n), std::span(out, n * sizeof(std::uint16_t)));
}
void unshuffle_u16(const std::uint8_t* in, std::uint16_t* out, std::size_t n) noexcept {
  unshuffle_bytes(std::span(in, n * sizeof(std::uint16_t)), std::span(out, n));
}

void quantize_f16(const float* in, std::uint16_t* out, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = float32_to_float16_bits(in[i]);
  }
}
void dequantize_f16(const std::uint16_t* in, float* out, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = float16_bits_to_float32(in[i]);
  }
}
void quantize_bf16(const float* in, std::uint16_t* out, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = float32_to_bfloat16_bits(in[i]);
  }
}
void dequantize_bf16(const std::uint16_t* in, float* out, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = bfloat16_bits_to_float32(in[i]);
  }
}

std::uint32_t crc32c_bytes(const std::uint8_t* data, std::size_t size) noexcept { return crc32c({data, size}); }

}  // namespace scalar

constexpr KernelTable kScalarTable = {
    .name = "scalar",
    .delta_encode_u32 = scalar::delta_encode_u32,
    .delta_decode_u32 = scalar::delta_decode_u32,
    .xor_encode_u32 = scalar::xor_encode_u32,
    .xor_decode_u32 = scalar::xor_decode_u32,
    .delta_encode_u16 = scalar::delta_encode_u16,
    .delta_decode_u16 = scalar::delta_decode_u16,
    .xor_encode_u16 = scalar::xor_encode_u16,
    .xor_decode_u16 = scalar::xor_decode_u16,
    .shuffle_u32 = scalar::shuffle_u32,
    .unshuffle_u32 = scalar::unshuffle_u32,
    .shuffle_u16 = scalar::shuffle_u16,
    .unshuffle_u16 = scalar::unshuffle_u16,
    .quantize_f16 = scalar::quantize_f16,
    .dequantize_f16 = scalar::dequantize_f16,
    .quantize_bf16 = scalar::quantize_bf16,
    .dequantize_bf16 = scalar::dequantize_bf16,
    .crc32c = scalar::crc32c_bytes,
};

}  // namespace

const KernelTable& scalar_kernels() noexcept { return kScalarTable; }

const KernelTable* avx2_kernels() noexcept {
#if TURBOCORE_HAS_AVX2
  // The probe executes AVX2 instructions, so it must stay behind the CPUID check.
  static const bool usable = cpu_supports_avx2_backend() && avx2::toolchain_probe();
  return usable ? &avx2::kernel_table() : nullptr;
#else
  return nullptr;
#endif
}

const KernelTable& select_kernels(Backend backend) {
  switch (backend) {
    case Backend::Auto: {
      const KernelTable* avx2 = avx2_kernels();
      return avx2 != nullptr ? *avx2 : scalar_kernels();
    }
    case Backend::Scalar:
      return scalar_kernels();
    case Backend::Avx2: {
      const KernelTable* avx2 = avx2_kernels();
      if (avx2 == nullptr) {
        throw Error(ErrorCode::BackendUnavailable,
                    TURBOCORE_HAS_AVX2 ? "the AVX2 backend is not supported by this CPU"
                                       : "this build of TurboCore does not include the AVX2 backend");
      }
      return *avx2;
    }
  }
  throw Error(ErrorCode::InvalidArgument, "unknown backend");
}

}  // namespace detail

bool is_backend_available(Backend backend) noexcept {
  switch (backend) {
    case Backend::Auto:
    case Backend::Scalar:
      return true;
    case Backend::Avx2:
      return detail::avx2_kernels() != nullptr;
  }
  return false;
}

Backend best_available_backend() noexcept {
  return detail::avx2_kernels() != nullptr ? Backend::Avx2 : Backend::Scalar;
}

const char* backend_name(Backend backend) noexcept {
  switch (backend) {
    case Backend::Auto:
      return "auto";
    case Backend::Scalar:
      return "scalar";
    case Backend::Avx2:
      return "avx2";
  }
  return "unknown";
}

}  // namespace turbocore
