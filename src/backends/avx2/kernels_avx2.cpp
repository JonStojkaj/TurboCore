// AVX2 implementations of the kernels in detail/kernel_table.hpp.
//
// Compiled with the AVX2 backend flags (AVX2, F16C, BMI, BMI2, LZCNT; FMA is
// deliberately not enabled). Rules for this file:
//   - Every kernel is bit-identical to its scalar counterpart for all inputs.
//     Integer kernels are exact by construction; the float conversions rely
//     on IEEE-conformant F16C instructions and are verified exhaustively.
//   - Unaligned loads and stores throughout; nothing is assumed about the
//     caller's buffers.
//   - Lengths that are not a multiple of the vector width are finished with
//     scalar code, or through a padded stack buffer where that is the only
//     way to stay on the exact same instruction as the vector body.
//   - Everything except kernel_table() has internal linkage, and no
//     standard-library templates are used, so no code compiled with these
//     flags can be picked by the linker for the scalar path.

#include "backends/avx2/kernels_avx2.hpp"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace turbocore::detail::avx2 {

namespace {

inline __m256i load256(const void* p) noexcept { return _mm256_loadu_si256(static_cast<const __m256i*>(p)); }
inline void store256(void* p, __m256i v) noexcept { _mm256_storeu_si256(static_cast<__m256i*>(p), v); }
inline __m128i load128(const void* p) noexcept { return _mm_loadu_si128(static_cast<const __m128i*>(p)); }
inline void store128(void* p, __m128i v) noexcept { _mm_storeu_si128(static_cast<__m128i*>(p), v); }
inline void store64(void* p, __m128i v) noexcept { _mm_storel_epi64(static_cast<__m128i*>(p), v); }

// ---------------------------------------------------------------------------
// Predictors: encode
//
// v[i] <- v[i] op v[i-1]. Each output depends only on original inputs, so
// this is a plain element-wise operation between the array and itself
// shifted by one. Working from the end keeps the in-place update safe: a
// vector at index i reads [i-1, i+W) and every earlier store was at >= i+W.
// ---------------------------------------------------------------------------

void delta_encode_u32(std::uint32_t* v, std::size_t n) noexcept {
  std::size_t i = n;
  while (i >= 9) {
    i -= 8;
    store256(v + i, _mm256_sub_epi32(load256(v + i), load256(v + i - 1)));
  }
  while (i > 1) {
    --i;
    v[i] -= v[i - 1];
  }
}

void xor_encode_u32(std::uint32_t* v, std::size_t n) noexcept {
  std::size_t i = n;
  while (i >= 9) {
    i -= 8;
    store256(v + i, _mm256_xor_si256(load256(v + i), load256(v + i - 1)));
  }
  while (i > 1) {
    --i;
    v[i] ^= v[i - 1];
  }
}

void delta_encode_u16(std::uint16_t* v, std::size_t n) noexcept {
  std::size_t i = n;
  while (i >= 17) {
    i -= 16;
    store256(v + i, _mm256_sub_epi16(load256(v + i), load256(v + i - 1)));
  }
  while (i > 1) {
    --i;
    v[i] = static_cast<std::uint16_t>(v[i] - v[i - 1]);
  }
}

void xor_encode_u16(std::uint16_t* v, std::size_t n) noexcept {
  std::size_t i = n;
  while (i >= 17) {
    i -= 16;
    store256(v + i, _mm256_xor_si256(load256(v + i), load256(v + i - 1)));
  }
  while (i > 1) {
    --i;
    v[i] = static_cast<std::uint16_t>(v[i] ^ v[i - 1]);
  }
}

// ---------------------------------------------------------------------------
// Predictors: decode
//
// The inverse is a prefix scan, v[i] <- v[0] op v[1] op ... op v[i], which
// has a loop-carried dependency. Within one vector the scan is computed in
// log2(lanes) shift-and-combine steps; _mm256_slli_si256 shifts each 128-bit
// half separately, so that yields two independent half scans. The low half's
// total is then folded into the high half, and the previous vector's total
// (`carry`) into everything. Only the carry crosses iterations.
// ---------------------------------------------------------------------------

void delta_decode_u32(std::uint32_t* v, std::size_t n) noexcept {
  const __m256i last_lane = _mm256_set1_epi32(7);
  __m256i carry = _mm256_setzero_si256();
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256i x = load256(v + i);
    x = _mm256_add_epi32(x, _mm256_slli_si256(x, 4));
    x = _mm256_add_epi32(x, _mm256_slli_si256(x, 8));
    const __m128i low_total = _mm_shuffle_epi32(_mm256_castsi256_si128(x), 0xFF);
    x = _mm256_add_epi32(x, _mm256_inserti128_si256(_mm256_setzero_si256(), low_total, 1));
    x = _mm256_add_epi32(x, carry);
    store256(v + i, x);
    carry = _mm256_permutevar8x32_epi32(x, last_lane);
  }
  std::uint32_t running = i > 0 ? v[i - 1] : 0;
  for (; i < n; ++i) {
    running += v[i];
    v[i] = running;
  }
}

void xor_decode_u32(std::uint32_t* v, std::size_t n) noexcept {
  const __m256i last_lane = _mm256_set1_epi32(7);
  __m256i carry = _mm256_setzero_si256();
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256i x = load256(v + i);
    x = _mm256_xor_si256(x, _mm256_slli_si256(x, 4));
    x = _mm256_xor_si256(x, _mm256_slli_si256(x, 8));
    const __m128i low_total = _mm_shuffle_epi32(_mm256_castsi256_si128(x), 0xFF);
    x = _mm256_xor_si256(x, _mm256_inserti128_si256(_mm256_setzero_si256(), low_total, 1));
    x = _mm256_xor_si256(x, carry);
    store256(v + i, x);
    carry = _mm256_permutevar8x32_epi32(x, last_lane);
  }
  std::uint32_t running = i > 0 ? v[i - 1] : 0;
  for (; i < n; ++i) {
    running ^= v[i];
    v[i] = running;
  }
}

/// All eight 16-bit lanes set to lane 7 of `x`.
inline __m128i broadcast_last_u16(__m128i x) noexcept {
  return _mm_shuffle_epi32(_mm_shufflehi_epi16(x, 0xFF), 0xFF);
}

void delta_decode_u16(std::uint16_t* v, std::size_t n) noexcept {
  __m256i carry = _mm256_setzero_si256();
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m256i x = load256(v + i);
    x = _mm256_add_epi16(x, _mm256_slli_si256(x, 2));
    x = _mm256_add_epi16(x, _mm256_slli_si256(x, 4));
    x = _mm256_add_epi16(x, _mm256_slli_si256(x, 8));
    const __m128i low_total = broadcast_last_u16(_mm256_castsi256_si128(x));
    x = _mm256_add_epi16(x, _mm256_inserti128_si256(_mm256_setzero_si256(), low_total, 1));
    x = _mm256_add_epi16(x, carry);
    store256(v + i, x);
    carry = _mm256_broadcastsi128_si256(broadcast_last_u16(_mm256_extracti128_si256(x, 1)));
  }
  std::uint16_t running = i > 0 ? v[i - 1] : std::uint16_t{0};
  for (; i < n; ++i) {
    running = static_cast<std::uint16_t>(running + v[i]);
    v[i] = running;
  }
}

void xor_decode_u16(std::uint16_t* v, std::size_t n) noexcept {
  __m256i carry = _mm256_setzero_si256();
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m256i x = load256(v + i);
    x = _mm256_xor_si256(x, _mm256_slli_si256(x, 2));
    x = _mm256_xor_si256(x, _mm256_slli_si256(x, 4));
    x = _mm256_xor_si256(x, _mm256_slli_si256(x, 8));
    const __m128i low_total = broadcast_last_u16(_mm256_castsi256_si128(x));
    x = _mm256_xor_si256(x, _mm256_inserti128_si256(_mm256_setzero_si256(), low_total, 1));
    x = _mm256_xor_si256(x, carry);
    store256(v + i, x);
    carry = _mm256_broadcastsi128_si256(broadcast_last_u16(_mm256_extracti128_si256(x, 1)));
  }
  std::uint16_t running = i > 0 ? v[i - 1] : std::uint16_t{0};
  for (; i < n; ++i) {
    running = static_cast<std::uint16_t>(running ^ v[i]);
    v[i] = running;
  }
}

// ---------------------------------------------------------------------------
// Byte-plane shuffle
// ---------------------------------------------------------------------------

void shuffle_u32(const std::uint32_t* in, std::uint8_t* out, std::size_t n) noexcept {
  std::uint8_t* const plane0 = out;
  std::uint8_t* const plane1 = out + n;
  std::uint8_t* const plane2 = out + 2 * n;
  std::uint8_t* const plane3 = out + 3 * n;

  // Within each 128-bit half: gather byte 0 of the four elements, then byte
  // 1, byte 2, byte 3. Each group is then one 32-bit lane.
  const __m256i group_bytes = _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                                               0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
  // Bring the two halves' lanes for the same plane next to each other.
  const __m256i pair_planes = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256i x = _mm256_shuffle_epi8(load256(in + i), group_bytes);
    x = _mm256_permutevar8x32_epi32(x, pair_planes);
    const __m128i low = _mm256_castsi256_si128(x);       // plane 0 | plane 1, 8 bytes each
    const __m128i high = _mm256_extracti128_si256(x, 1);  // plane 2 | plane 3
    store64(plane0 + i, low);
    store64(plane1 + i, _mm_unpackhi_epi64(low, low));
    store64(plane2 + i, high);
    store64(plane3 + i, _mm_unpackhi_epi64(high, high));
  }
  for (; i < n; ++i) {
    const std::uint32_t value = in[i];
    plane0[i] = static_cast<std::uint8_t>(value & 0xFFu);
    plane1[i] = static_cast<std::uint8_t>((value >> 8) & 0xFFu);
    plane2[i] = static_cast<std::uint8_t>((value >> 16) & 0xFFu);
    plane3[i] = static_cast<std::uint8_t>(value >> 24);
  }
}

void unshuffle_u32(const std::uint8_t* in, std::uint32_t* out, std::size_t n) noexcept {
  const std::uint8_t* const plane0 = in;
  const std::uint8_t* const plane1 = in + n;
  const std::uint8_t* const plane2 = in + 2 * n;
  const std::uint8_t* const plane3 = in + 3 * n;

  std::size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    const __m256i b0 = load256(plane0 + i);
    const __m256i b1 = load256(plane1 + i);
    const __m256i b2 = load256(plane2 + i);
    const __m256i b3 = load256(plane3 + i);
    // Interleave bytes into 16-bit pairs, then pairs into 32-bit elements.
    // The unpack instructions work per 128-bit half, so each result holds
    // elements from two separate runs (noted as low half ; high half).
    const __m256i b01_low = _mm256_unpacklo_epi8(b0, b1);   // elements 0-7   ; 16-23
    const __m256i b01_high = _mm256_unpackhi_epi8(b0, b1);  // elements 8-15  ; 24-31
    const __m256i b23_low = _mm256_unpacklo_epi8(b2, b3);
    const __m256i b23_high = _mm256_unpackhi_epi8(b2, b3);
    const __m256i w0 = _mm256_unpacklo_epi16(b01_low, b23_low);    // 0-3   ; 16-19
    const __m256i w1 = _mm256_unpackhi_epi16(b01_low, b23_low);    // 4-7   ; 20-23
    const __m256i w2 = _mm256_unpacklo_epi16(b01_high, b23_high);  // 8-11  ; 24-27
    const __m256i w3 = _mm256_unpackhi_epi16(b01_high, b23_high);  // 12-15 ; 28-31
    store256(out + i, _mm256_permute2x128_si256(w0, w1, 0x20));
    store256(out + i + 8, _mm256_permute2x128_si256(w2, w3, 0x20));
    store256(out + i + 16, _mm256_permute2x128_si256(w0, w1, 0x31));
    store256(out + i + 24, _mm256_permute2x128_si256(w2, w3, 0x31));
  }
  for (; i < n; ++i) {
    out[i] = static_cast<std::uint32_t>(plane0[i]) | static_cast<std::uint32_t>(plane1[i]) << 8 |
             static_cast<std::uint32_t>(plane2[i]) << 16 | static_cast<std::uint32_t>(plane3[i]) << 24;
  }
}

void shuffle_u16(const std::uint16_t* in, std::uint8_t* out, std::size_t n) noexcept {
  std::uint8_t* const plane0 = out;
  std::uint8_t* const plane1 = out + n;

  // Within each half: the eight low bytes, then the eight high bytes.
  const __m256i group_bytes = _mm256_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
                                               0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m256i x = _mm256_shuffle_epi8(load256(in + i), group_bytes);
    x = _mm256_permute4x64_epi64(x, 0xD8);  // 64-bit lanes 0,2,1,3: plane 0 | plane 1
    store128(plane0 + i, _mm256_castsi256_si128(x));
    store128(plane1 + i, _mm256_extracti128_si256(x, 1));
  }
  for (; i < n; ++i) {
    plane0[i] = static_cast<std::uint8_t>(in[i] & 0xFFu);
    plane1[i] = static_cast<std::uint8_t>(in[i] >> 8);
  }
}

void unshuffle_u16(const std::uint8_t* in, std::uint16_t* out, std::size_t n) noexcept {
  const std::uint8_t* const plane0 = in;
  const std::uint8_t* const plane1 = in + n;

  std::size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    const __m256i b0 = load256(plane0 + i);
    const __m256i b1 = load256(plane1 + i);
    const __m256i low = _mm256_unpacklo_epi8(b0, b1);   // elements 0-7  ; 16-23
    const __m256i high = _mm256_unpackhi_epi8(b0, b1);  // elements 8-15 ; 24-31
    store256(out + i, _mm256_permute2x128_si256(low, high, 0x20));
    store256(out + i + 16, _mm256_permute2x128_si256(low, high, 0x31));
  }
  for (; i < n; ++i) {
    out[i] = static_cast<std::uint16_t>(plane0[i] | (plane1[i] << 8));
  }
}

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

constexpr int kRoundNearestEven = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

// VCVTPS2PH with an explicit rounding mode is the IEEE 754 binary32 ->
// binary16 conversion: ties to even, gradual underflow, overflow to
// infinity, NaN quieted with the top payload bits kept. That is exactly the
// behaviour the scalar reference implements by hand. The explicit mode means
// the result does not depend on the caller's MXCSR rounding mode.
inline __m128i quantize_f16_vector(__m256 x) noexcept { return _mm256_cvtps_ph(x, kRoundNearestEven); }

void quantize_f16(const float* in, std::uint16_t* out, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    store128(out + i, quantize_f16_vector(_mm256_loadu_ps(in + i)));
  }
  if (i < n) {
    // Same instruction for the tail, through zero-padded buffers.
    float padded_in[8] = {};
    std::uint16_t padded_out[8];
    std::memcpy(padded_in, in + i, (n - i) * sizeof(float));
    store128(padded_out, quantize_f16_vector(_mm256_loadu_ps(padded_in)));
    std::memcpy(out + i, padded_out, (n - i) * sizeof(std::uint16_t));
  }
}

void dequantize_f16(const std::uint16_t* in, float* out, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(out + i, _mm256_cvtph_ps(load128(in + i)));
  }
  if (i < n) {
    std::uint16_t padded_in[8] = {};
    float padded_out[8];
    std::memcpy(padded_in, in + i, (n - i) * sizeof(std::uint16_t));
    _mm256_storeu_ps(padded_out, _mm256_cvtph_ps(load128(padded_in)));
    std::memcpy(out + i, padded_out, (n - i) * sizeof(float));
  }
}

// BFloat16 is the top half of binary32, so the conversion is integer
// arithmetic on the bit pattern. Round to nearest even is
//     (bits + 0x7FFF + lsb_of_kept_half) >> 16
// which carries exactly when the discarded half is above the midpoint, or
// equal to it with an odd kept half. NaNs bypass the rounding (which would
// wrap for all-ones mantissas) and get the quiet bit forced instead.
inline __m128i quantize_bf16_vector(__m256i bits) noexcept {
  const __m256i top = _mm256_srli_epi32(bits, 16);
  const __m256i tie_to_even = _mm256_and_si256(top, _mm256_set1_epi32(1));
  const __m256i rounded =
      _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(bits, _mm256_set1_epi32(0x7FFF)), tie_to_even), 16);

  // After clearing the sign, NaNs are exactly the patterns above infinity;
  // both sides are non-negative so the signed compare is correct.
  const __m256i magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(0x7FFFFFFF));
  const __m256i is_nan = _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7F800000));
  const __m256i quieted = _mm256_or_si256(top, _mm256_set1_epi32(0x0040));

  const __m256i result = _mm256_blendv_epi8(rounded, quieted, is_nan);
  // Every lane is in [0, 0xFFFF], so the unsigned-saturating pack is exact.
  return _mm_packus_epi32(_mm256_castsi256_si128(result), _mm256_extracti128_si256(result, 1));
}

void quantize_bf16(const float* in, std::uint16_t* out, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    store128(out + i, quantize_bf16_vector(load256(in + i)));
  }
  for (; i < n; ++i) {
    std::uint32_t bits;
    std::memcpy(&bits, in + i, sizeof(bits));
    const std::uint32_t top = bits >> 16;
    const bool is_nan = (bits & 0x7FFFFFFFu) > 0x7F800000u;
    out[i] = static_cast<std::uint16_t>(is_nan ? (top | 0x0040u) : ((bits + 0x7FFFu + (top & 1u)) >> 16));
  }
}

void dequantize_bf16(const std::uint16_t* in, float* out, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    store256(out + i, _mm256_slli_epi32(_mm256_cvtepu16_epi32(load128(in + i)), 16));
  }
  for (; i < n; ++i) {
    const std::uint32_t bits = static_cast<std::uint32_t>(in[i]) << 16;
    std::memcpy(out + i, &bits, sizeof(bits));
  }
}

// ---------------------------------------------------------------------------
// CRC-32C: the SSE4.2 CRC32 instruction computes exactly this polynomial.
// ---------------------------------------------------------------------------

std::uint32_t crc32c(const std::uint8_t* data, std::size_t size) noexcept {
  std::uint64_t crc = 0xFFFFFFFFu;
  while (size >= 8) {
    std::uint64_t word;
    std::memcpy(&word, data, sizeof(word));
    crc = _mm_crc32_u64(crc, word);
    data += 8;
    size -= 8;
  }
  auto crc32 = static_cast<std::uint32_t>(crc);
  while (size > 0) {
    crc32 = _mm_crc32_u8(crc32, *data);
    ++data;
    --size;
  }
  return crc32 ^ 0xFFFFFFFFu;
}

constexpr KernelTable kTable = {
    .name = "avx2",
    .delta_encode_u32 = delta_encode_u32,
    .delta_decode_u32 = delta_decode_u32,
    .xor_encode_u32 = xor_encode_u32,
    .xor_decode_u32 = xor_decode_u32,
    .delta_encode_u16 = delta_encode_u16,
    .delta_decode_u16 = delta_decode_u16,
    .xor_encode_u16 = xor_encode_u16,
    .xor_decode_u16 = xor_decode_u16,
    .shuffle_u32 = shuffle_u32,
    .unshuffle_u32 = unshuffle_u32,
    .shuffle_u16 = shuffle_u16,
    .unshuffle_u16 = unshuffle_u16,
    .quantize_f16 = quantize_f16,
    .dequantize_f16 = dequantize_f16,
    .quantize_bf16 = quantize_bf16,
    .dequantize_bf16 = dequantize_bf16,
    .crc32c = crc32c,
};

}  // namespace

const KernelTable& kernel_table() noexcept { return kTable; }

}  // namespace turbocore::detail::avx2
