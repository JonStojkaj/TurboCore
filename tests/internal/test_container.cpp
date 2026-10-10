#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "detail/container.hpp"
#include "turbocore/turbocore.hpp"

namespace {

namespace d = turbocore::detail;

TEST(Container, HeaderRoundTrips) {
  const turbocore::StreamInfo written{
      .element_count = 0,
      .block_count = 0,
      .block_elements = 4096,
      .predictor = turbocore::Predictor::Xor,
      .byte_shuffle = true,
      .quantization = turbocore::QuantFormat::BFloat16,
  };
  std::vector<std::uint8_t> stream(d::kHeaderSize);
  d::write_header(std::span<std::uint8_t, d::kHeaderSize>(stream.data(), d::kHeaderSize), written);

  const turbocore::StreamInfo parsed = d::parse_header(stream);
  EXPECT_EQ(parsed.element_count, written.element_count);
  EXPECT_EQ(parsed.block_count, written.block_count);
  EXPECT_EQ(parsed.block_elements, written.block_elements);
  EXPECT_EQ(parsed.predictor, written.predictor);
  EXPECT_EQ(parsed.byte_shuffle, written.byte_shuffle);
  EXPECT_EQ(parsed.quantization, written.quantization);
}

TEST(Container, IntegersAreLittleEndian) {
  std::uint8_t bytes[8] = {};
  d::store_le<std::uint32_t>(bytes, 0x11223344u);
  EXPECT_EQ(bytes[0], 0x44);
  EXPECT_EQ(bytes[3], 0x11);
  EXPECT_EQ(d::load_le<std::uint32_t>(bytes), 0x11223344u);
  d::store_le<std::uint64_t>(bytes, 0x0102030405060708u);
  EXPECT_EQ(bytes[0], 0x08);
  EXPECT_EQ(d::load_le<std::uint64_t>(bytes), 0x0102030405060708u);
}

// A header with a valid checksum that claims 2^40 elements (4 TiB of floats)
// while carrying no payload. It must be rejected by arithmetic on the header,
// before the decoder allocates anything proportional to the claim.
TEST(Container, ImplausibleElementCountIsRejectedBeforeAllocating) {
  const turbocore::StreamInfo claim{
      .element_count = std::uint64_t{1} << 40,
      .block_count = 1u << 16,
      .block_elements = 1u << 24,
      .predictor = turbocore::Predictor::Delta,
      .byte_shuffle = true,
      .quantization = {},
  };
  // Header plus a complete (all-zero) index, so that only the payload is missing.
  std::vector<std::uint8_t> stream(d::kHeaderSize + std::size_t{claim.block_count} * d::kIndexEntrySize);
  d::write_header(std::span<std::uint8_t, d::kHeaderSize>(stream.data(), d::kHeaderSize), claim);

  try {
    (void)turbocore::Decompressor().decompress(stream);
    FAIL() << "expected turbocore::Error";
  } catch (const turbocore::Error& error) {
    EXPECT_EQ(error.code(), turbocore::ErrorCode::CorruptStream);
    EXPECT_NE(std::string(error.what()).find("implausible"), std::string::npos) << error.what();
  }
}

}  // namespace
