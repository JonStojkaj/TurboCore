#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "detail/transforms.hpp"

namespace {

namespace d = turbocore::detail;

std::vector<std::uint32_t> random_words(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<std::uint32_t> v(n);
  for (std::uint32_t& x : v) {
    x = static_cast<std::uint32_t>(rng());
  }
  return v;
}

TEST(Transforms, DeltaMatchesItsDefinition) {
  std::vector<std::uint32_t> v = {10, 13, 13, 5, 0xFFFFFFFFu, 0};
  d::delta_encode(v);
  // Differences wrap modulo 2^32 instead of overflowing.
  EXPECT_EQ(v, (std::vector<std::uint32_t>{10, 3, 0, 0xFFFFFFF8u, 0xFFFFFFFAu, 1}));
}

TEST(Transforms, XorMatchesItsDefinition) {
  std::vector<std::uint32_t> v = {0xF0F0F0F0u, 0xF0F0F0F1u, 0x0F0F0F0Eu};
  d::xor_encode(v);
  EXPECT_EQ(v, (std::vector<std::uint32_t>{0xF0F0F0F0u, 0x00000001u, 0xFFFFFFFFu}));
}

TEST(Transforms, ShuffleGroupsBytePlanes) {
  const std::vector<std::uint32_t> in = {0x44332211u, 0x88776655u};
  std::vector<std::uint8_t> out(8);
  d::shuffle_bytes(in, out);
  EXPECT_EQ(out, (std::vector<std::uint8_t>{0x11, 0x55, 0x22, 0x66, 0x33, 0x77, 0x44, 0x88}));
}

TEST(Transforms, SixteenBitVariantsMatchTheirDefinition) {
  std::vector<std::uint16_t> delta = {10, 13, 5, 0xFFFF, 0};
  d::delta_encode(delta);
  EXPECT_EQ(delta, (std::vector<std::uint16_t>{10, 3, 0xFFF8, 0xFFFA, 1}));  // wraps modulo 2^16

  std::vector<std::uint16_t> x = {0xF0F0, 0xF0F1, 0x0F0E};
  d::xor_encode(x);
  EXPECT_EQ(x, (std::vector<std::uint16_t>{0xF0F0, 0x0001, 0xFFFF}));

  const std::vector<std::uint16_t> in = {0x2211, 0x4433, 0x6655};
  std::vector<std::uint8_t> planes(6);
  d::shuffle_bytes(in, planes);
  EXPECT_EQ(planes, (std::vector<std::uint8_t>{0x11, 0x33, 0x55, 0x22, 0x44, 0x66}));
}

TEST(Transforms, SixteenBitVariantsAreExactlyReversible) {
  for (const std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{1000}}) {
    SCOPED_TRACE(n);
    std::vector<std::uint16_t> original(n);
    std::mt19937 rng(43);
    for (std::uint16_t& value : original) {
      value = static_cast<std::uint16_t>(rng() & 0xFFFFu);
    }

    std::vector<std::uint16_t> v = original;
    d::delta_encode(v);
    d::delta_decode(v);
    EXPECT_EQ(v, original);

    d::xor_encode(v);
    d::xor_decode(v);
    EXPECT_EQ(v, original);

    std::vector<std::uint8_t> bytes(n * 2);
    std::vector<std::uint16_t> restored(n);
    d::shuffle_bytes(original, bytes);
    d::unshuffle_bytes(bytes, restored);
    EXPECT_EQ(restored, original);
  }
}

TEST(Transforms, EveryTransformIsExactlyReversible) {
  for (const std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{1000}}) {
    SCOPED_TRACE(n);
    const std::vector<std::uint32_t> original = random_words(n, 42);

    std::vector<std::uint32_t> v = original;
    d::delta_encode(v);
    d::delta_decode(v);
    EXPECT_EQ(v, original);

    d::xor_encode(v);
    d::xor_decode(v);
    EXPECT_EQ(v, original);

    std::vector<std::uint8_t> bytes(n * 4);
    std::vector<std::uint32_t> restored(n);
    d::shuffle_bytes(original, bytes);
    d::unshuffle_bytes(bytes, restored);
    EXPECT_EQ(restored, original);
  }
}

}  // namespace
