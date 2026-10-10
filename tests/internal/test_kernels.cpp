#include <gtest/gtest.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "backend_support.hpp"
#include "detail/kernels.hpp"

// Differential tests: every AVX2 kernel against its scalar counterpart, for
// every length around the vector widths and for buffers that are not aligned
// to the vector width. The requirement is bit-identical output, not
// approximate agreement.
//
// Each buffer is exactly the size the kernel may touch, placed at an offset
// inside a slightly larger allocation whose remaining bytes are a canary
// pattern. A kernel that writes out of bounds corrupts the canary
// (and trips ASan at the allocation edges).

namespace {

using turbocore::detail::KernelTable;

constexpr std::uint8_t kCanary = 0xC5;

/// Lengths that cover: empty, below one vector, every remainder modulo the
/// 8/16/32-element loop strides, and a few larger sizes.
std::vector<std::size_t> lengths() {
  std::vector<std::size_t> result;
  for (std::size_t n = 0; n <= 100; ++n) {
    result.push_back(n);
  }
  for (const std::size_t n : {127u, 128u, 129u, 255u, 256u, 257u, 1000u, 4099u}) {
    result.push_back(n);
  }
  return result;
}

/// An array of `count` T starting `shift` elements past a 64-byte boundary of
/// the allocation, with canaries on both sides. The array always has the
/// natural alignment of T (anything less is undefined behaviour in C++, not
/// something a kernel can be asked to tolerate), but for shift != 0 it is not
/// aligned to the 16- or 32-byte vector width.
template <typename T>
class Guarded {
 public:
  Guarded(std::size_t count, std::size_t shift)
      : count_(count),
        offset_(kPad + shift * sizeof(T)),
        storage_(offset_ + count * sizeof(T) + kPad, kCanary) {}

  T* data() { return reinterpret_cast<T*>(storage_.data() + offset_); }

  void fill_from(const std::vector<T>& values) {
    if (count_ > 0) {
      std::memcpy(storage_.data() + offset_, values.data(), count_ * sizeof(T));
    }
  }

  std::vector<T> contents() const {
    std::vector<T> values(count_);
    if (count_ > 0) {
      std::memcpy(values.data(), storage_.data() + offset_, count_ * sizeof(T));
    }
    return values;
  }

  bool canaries_intact() const {
    for (std::size_t i = 0; i < offset_; ++i) {
      if (storage_[i] != kCanary) {
        return false;
      }
    }
    for (std::size_t i = offset_ + count_ * sizeof(T); i < storage_.size(); ++i) {
      if (storage_[i] != kCanary) {
        return false;
      }
    }
    return true;
  }

 private:
  static constexpr std::size_t kPad = 64;
  std::size_t count_;
  std::size_t offset_;
  std::vector<std::uint8_t> storage_;
};

template <typename T>
std::vector<T> random_values(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<T> v(n);
  for (T& x : v) {
    x = static_cast<T>(rng());
  }
  return v;
}

/// Floats that stress the conversions: every special class at every lane
/// position, plus random bit patterns and values across the binary16 range.
std::vector<float> stress_floats(std::size_t n, std::uint32_t seed) {
  static const std::uint32_t specials[] = {
      0x00000000u, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u, 0xFFC00000u, 0x7F800001u, 0xFFBFFFFFu,
      0x7FFFFFFFu, 0xFFFFFFFFu, 0x00000001u, 0x007FFFFFu, 0x00800000u, 0x7F7FFFFFu, 0xFF7FFFFFu, 0x3F800000u,
      0x3F808000u, 0x3F818000u, 0x3F80FFFFu, 0x477FE000u, 0x477FF000u, 0x33000000u, 0x33000001u, 0x38800000u,
      0x387FFFFFu, 0x7F7F8000u, 0x7F7F7FFFu, 0x00008000u, 0x00008001u, 0x3F801000u, 0x3F803000u, 0xC77FF000u,
  };
  std::mt19937 rng(seed);
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::uint32_t bits = 0;
    switch (rng() % 3) {
      case 0:
        bits = specials[rng() % std::size(specials)];
        break;
      case 1:
        bits = static_cast<std::uint32_t>(rng());
        break;
      default:  // exponent fields 100..145: the binary16 range and both of its edges
        bits = (static_cast<std::uint32_t>(rng()) & 0x807FFFFFu) |
               ((100u + static_cast<std::uint32_t>(rng() % 46u)) << 23);
        break;
    }
    v[i] = std::bit_cast<float>(bits);
  }
  return v;
}

/// Compares float vectors by bit pattern (NaN payloads included).
std::vector<std::uint32_t> bits_of(const std::vector<float>& v) {
  std::vector<std::uint32_t> bits(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    bits[i] = std::bit_cast<std::uint32_t>(v[i]);
  }
  return bits;
}

class Avx2Kernels : public ::testing::Test {
 protected:
  void SetUp() override {
    TURBOCORE_NEEDS_AVX2();
    avx2_ = turbocore::detail::avx2_kernels();
    ASSERT_NE(avx2_, nullptr);
  }

  /// In-place kernel on T: both backends from the same input.
  template <typename T, typename Member>
  void check_in_place(Member kernel, const char* name) {
    for (const std::size_t n : lengths()) {
      for (const std::size_t misalignment : {std::size_t{0}, std::size_t{1}, std::size_t{3}}) {
        SCOPED_TRACE(std::string(name) + " n=" + std::to_string(n) + " misalignment=" + std::to_string(misalignment));
        const std::vector<T> input = random_values<T>(n, static_cast<std::uint32_t>(n * 31 + misalignment));

        Guarded<T> expected(n, 0);
        expected.fill_from(input);
        (scalar_.*kernel)(expected.data(), n);

        Guarded<T> actual(n, misalignment);
        actual.fill_from(input);
        (avx2_->*kernel)(actual.data(), n);

        ASSERT_EQ(actual.contents(), expected.contents());
        ASSERT_TRUE(actual.canaries_intact());
      }
    }
  }

  /// Out-of-place kernel from `in_count(n)` In to `out_count(n)` Out.
  template <typename In, typename Out, typename Member, typename MakeInput>
  void check_out_of_place(Member kernel, const char* name, std::size_t in_scale, std::size_t out_scale,
                          MakeInput make_input) {
    for (const std::size_t n : lengths()) {
      for (const std::size_t misalignment : {std::size_t{0}, std::size_t{1}, std::size_t{3}}) {
        SCOPED_TRACE(std::string(name) + " n=" + std::to_string(n) + " misalignment=" + std::to_string(misalignment));
        const std::vector<In> input = make_input(n * in_scale, static_cast<std::uint32_t>(n * 17 + misalignment));

        Guarded<In> in_scalar(n * in_scale, 0);
        in_scalar.fill_from(input);
        Guarded<Out> expected(n * out_scale, 0);
        (scalar_.*kernel)(in_scalar.data(), expected.data(), n);

        Guarded<In> in_avx2(n * in_scale, misalignment);
        in_avx2.fill_from(input);
        Guarded<Out> actual(n * out_scale, misalignment);
        (avx2_->*kernel)(in_avx2.data(), actual.data(), n);

        if constexpr (std::is_same_v<Out, float>) {
          ASSERT_EQ(bits_of(actual.contents()), bits_of(expected.contents()));
        } else {
          ASSERT_EQ(actual.contents(), expected.contents());
        }
        ASSERT_TRUE(actual.canaries_intact());
        ASSERT_EQ(in_avx2.contents().size(), input.size());
        ASSERT_TRUE(in_avx2.canaries_intact());
      }
    }
  }

  const KernelTable& scalar_ = turbocore::detail::scalar_kernels();
  const KernelTable* avx2_ = nullptr;
};

TEST_F(Avx2Kernels, PredictorsU32) {
  check_in_place<std::uint32_t>(&KernelTable::delta_encode_u32, "delta_encode_u32");
  check_in_place<std::uint32_t>(&KernelTable::delta_decode_u32, "delta_decode_u32");
  check_in_place<std::uint32_t>(&KernelTable::xor_encode_u32, "xor_encode_u32");
  check_in_place<std::uint32_t>(&KernelTable::xor_decode_u32, "xor_decode_u32");
}

TEST_F(Avx2Kernels, PredictorsU16) {
  check_in_place<std::uint16_t>(&KernelTable::delta_encode_u16, "delta_encode_u16");
  check_in_place<std::uint16_t>(&KernelTable::delta_decode_u16, "delta_decode_u16");
  check_in_place<std::uint16_t>(&KernelTable::xor_encode_u16, "xor_encode_u16");
  check_in_place<std::uint16_t>(&KernelTable::xor_decode_u16, "xor_decode_u16");
}

TEST_F(Avx2Kernels, Shuffles) {
  check_out_of_place<std::uint32_t, std::uint8_t>(&KernelTable::shuffle_u32, "shuffle_u32", 1, 4,
                                                  random_values<std::uint32_t>);
  check_out_of_place<std::uint8_t, std::uint32_t>(&KernelTable::unshuffle_u32, "unshuffle_u32", 4, 1,
                                                  random_values<std::uint8_t>);
  check_out_of_place<std::uint16_t, std::uint8_t>(&KernelTable::shuffle_u16, "shuffle_u16", 1, 2,
                                                  random_values<std::uint16_t>);
  check_out_of_place<std::uint8_t, std::uint16_t>(&KernelTable::unshuffle_u16, "unshuffle_u16", 2, 1,
                                                  random_values<std::uint8_t>);
}

TEST_F(Avx2Kernels, Quantization) {
  check_out_of_place<float, std::uint16_t>(&KernelTable::quantize_f16, "quantize_f16", 1, 1, stress_floats);
  check_out_of_place<float, std::uint16_t>(&KernelTable::quantize_bf16, "quantize_bf16", 1, 1, stress_floats);
  check_out_of_place<std::uint16_t, float>(&KernelTable::dequantize_f16, "dequantize_f16", 1, 1,
                                           random_values<std::uint16_t>);
  check_out_of_place<std::uint16_t, float>(&KernelTable::dequantize_bf16, "dequantize_bf16", 1, 1,
                                           random_values<std::uint16_t>);
}

// All 65536 half patterns, so every NaN payload and subnormal is covered.
TEST_F(Avx2Kernels, DequantizationOfEveryPattern) {
  std::vector<std::uint16_t> all(65536);
  for (std::size_t i = 0; i < all.size(); ++i) {
    all[i] = static_cast<std::uint16_t>(i);
  }
  std::vector<float> expected(all.size());
  std::vector<float> actual(all.size());

  scalar_.dequantize_f16(all.data(), expected.data(), all.size());
  avx2_->dequantize_f16(all.data(), actual.data(), all.size());
  EXPECT_EQ(bits_of(actual), bits_of(expected));

  scalar_.dequantize_bf16(all.data(), expected.data(), all.size());
  avx2_->dequantize_bf16(all.data(), actual.data(), all.size());
  EXPECT_EQ(bits_of(actual), bits_of(expected));
}

// A long carry chain: the prefix scans must carry correctly across every
// vector boundary, including wrap-around.
TEST_F(Avx2Kernels, PrefixScanCarriesAcrossManyVectors) {
  std::vector<std::uint32_t> ones32(100003, 0x01000001u);
  std::vector<std::uint32_t> expected32 = ones32;
  scalar_.delta_decode_u32(expected32.data(), expected32.size());
  avx2_->delta_decode_u32(ones32.data(), ones32.size());
  EXPECT_EQ(ones32, expected32);

  std::vector<std::uint16_t> ones16(100003, 0x0101);
  std::vector<std::uint16_t> expected16 = ones16;
  scalar_.delta_decode_u16(expected16.data(), expected16.size());
  avx2_->delta_decode_u16(ones16.data(), ones16.size());
  EXPECT_EQ(ones16, expected16);
}

TEST_F(Avx2Kernels, Crc32c) {
  const std::vector<std::uint8_t> data = random_values<std::uint8_t>(5000, 3);
  for (std::size_t offset = 0; offset < 9; ++offset) {
    for (std::size_t n = 0; n < 200; ++n) {
      ASSERT_EQ(avx2_->crc32c(data.data() + offset, n), scalar_.crc32c(data.data() + offset, n))
          << "offset=" << offset << " n=" << n;
    }
  }
  EXPECT_EQ(avx2_->crc32c(data.data(), data.size()), scalar_.crc32c(data.data(), data.size()));
  const std::uint8_t check[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
  EXPECT_EQ(avx2_->crc32c(check, sizeof(check)), 0xE3069283u);
}

// The scalar table must agree with the reference functions it wraps; this
// runs everywhere, so the dispatch layer is tested on machines without AVX2.
TEST(ScalarKernels, TableIsWiredToTheReferenceFunctions) {
  const KernelTable& k = turbocore::detail::scalar_kernels();
  EXPECT_STREQ(k.name, "scalar");

  std::vector<std::uint32_t> v = {10, 13, 13, 5};
  k.delta_encode_u32(v.data(), v.size());
  EXPECT_EQ(v, (std::vector<std::uint32_t>{10, 3, 0, 0xFFFFFFF8u}));
  k.delta_decode_u32(v.data(), v.size());
  EXPECT_EQ(v, (std::vector<std::uint32_t>{10, 13, 13, 5}));

  const float one = 1.0F;
  std::uint16_t half = 0;
  k.quantize_f16(&one, &half, 1);
  EXPECT_EQ(half, 0x3C00);
  k.quantize_bf16(&one, &half, 1);
  EXPECT_EQ(half, 0x3F80);

  const std::uint8_t check[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
  EXPECT_EQ(k.crc32c(check, sizeof(check)), 0xE3069283u);
}

}  // namespace
