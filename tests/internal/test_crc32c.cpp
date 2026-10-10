#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "detail/crc32c.hpp"

namespace {

using turbocore::detail::crc32c;

std::uint32_t crc_of(std::string_view text) {
  return crc32c({reinterpret_cast<const std::uint8_t*>(text.data()), text.size()});
}

TEST(Crc32c, KnownVectors) {
  EXPECT_EQ(crc_of(""), 0x00000000u);
  EXPECT_EQ(crc_of("123456789"), 0xE3069283u);  // the standard check value
  // RFC 3720 (iSCSI) appendix B.4: 32 bytes of zeros and of 0xFF.
  EXPECT_EQ(crc32c(std::vector<std::uint8_t>(32, 0x00)), 0x8A9136AAu);
  EXPECT_EQ(crc32c(std::vector<std::uint8_t>(32, 0xFF)), 0x62A8AB43u);
}

// The 8-bytes-at-a-time path and the byte-at-a-time tail must agree for
// every split of the input between them.
TEST(Crc32c, SlicedPathMatchesBytewiseReference) {
  const auto bytewise = [](std::span<const std::uint8_t> data) {
    std::uint32_t crc = 0xFFFFFFFFu;
    for (const std::uint8_t byte : data) {
      crc ^= byte;
      for (int bit = 0; bit < 8; ++bit) {
        crc = (crc >> 1) ^ ((crc & 1u) != 0 ? 0x82F63B78u : 0u);
      }
    }
    return crc ^ 0xFFFFFFFFu;
  };

  std::vector<std::uint8_t> data(100);
  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<std::uint8_t>(i * 37 + 11);
  }
  for (std::size_t length = 0; length <= data.size(); ++length) {
    const std::span<const std::uint8_t> prefix(data.data(), length);
    EXPECT_EQ(crc32c(prefix), bytewise(prefix)) << "length " << length;
    EXPECT_EQ(turbocore::detail::crc32c_portable(prefix), bytewise(prefix)) << "length " << length;
  }
}

}  // namespace
