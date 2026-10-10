#include "detail/crc32c.hpp"

#include <array>
#include <cstddef>

namespace turbocore::detail {

namespace {

constexpr std::uint32_t kPolynomialReflected = 0x82F63B78u;

using SliceTables = std::array<std::array<std::uint32_t, 256>, 8>;

constexpr SliceTables make_tables() noexcept {
  SliceTables tables{};
  for (std::uint32_t i = 0; i < 256; ++i) {
    std::uint32_t crc = i;
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc >> 1) ^ ((crc & 1u) != 0 ? kPolynomialReflected : 0u);
    }
    tables[0][i] = crc;
  }
  for (std::size_t slice = 1; slice < 8; ++slice) {
    for (std::size_t i = 0; i < 256; ++i) {
      const std::uint32_t previous = tables[slice - 1][i];
      tables[slice][i] = (previous >> 8) ^ tables[0][previous & 0xFFu];
    }
  }
  return tables;
}

constexpr SliceTables kTables = make_tables();

}  // namespace

std::uint32_t crc32c(std::span<const std::uint8_t> data) noexcept {
  std::uint32_t crc = 0xFFFFFFFFu;
  const std::uint8_t* p = data.data();
  std::size_t remaining = data.size();

  while (remaining >= 8) {
    const std::uint32_t low = crc ^ (static_cast<std::uint32_t>(p[0]) | static_cast<std::uint32_t>(p[1]) << 8 |
                                     static_cast<std::uint32_t>(p[2]) << 16 | static_cast<std::uint32_t>(p[3]) << 24);
    crc = kTables[7][low & 0xFFu] ^ kTables[6][(low >> 8) & 0xFFu] ^ kTables[5][(low >> 16) & 0xFFu] ^
          kTables[4][low >> 24] ^ kTables[3][p[4]] ^ kTables[2][p[5]] ^ kTables[1][p[6]] ^ kTables[0][p[7]];
    p += 8;
    remaining -= 8;
  }
  while (remaining > 0) {
    crc = (crc >> 8) ^ kTables[0][(crc ^ *p) & 0xFFu];
    ++p;
    --remaining;
  }
  return crc ^ 0xFFFFFFFFu;
}

}  // namespace turbocore::detail
