#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "test_data.hpp"
#include "turbocore/turbocore.hpp"

// Threading must be invisible in the results: the same bytes out of the
// compressor and the same values (or the same error) out of the decompressor
// for every thread count.

namespace {

using turbocore::CompressionOptions;
using turbocore::Compressor;
using turbocore::Decompressor;
using turbocore::QuantFormat;
using turbocore::testing::bit_identical;

const unsigned kThreadCounts[] = {1, 2, 3, 8, 0};

std::vector<float> many_blocks() {
  namespace t = turbocore::testing;
  std::vector<float> data = t::smooth_wave(60000);
  for (const auto& part : {t::uniform_noise(20000, 1), t::special_values(3000), t::random_bit_patterns(17001, 2)}) {
    data.insert(data.end(), part.begin(), part.end());
  }
  return data;  // 100001 elements: 101 blocks of 1000, the last one short
}

TEST(Threads, StreamIsIdenticalForEveryThreadCount) {
  const std::vector<float> data = many_blocks();
  for (const std::optional<QuantFormat> quantization :
       {std::optional<QuantFormat>{}, std::optional{QuantFormat::Float16}}) {
    CompressionOptions options{.quantization = quantization, .block_elements = 1000, .threads = 1};
    const std::vector<std::uint8_t> reference = Compressor(options).compress(data);
    for (const unsigned threads : kThreadCounts) {
      options.threads = threads;
      ASSERT_EQ(Compressor(options).compress(data), reference) << "threads=" << threads;
    }
  }
}

TEST(Threads, DecodedDataIsIdenticalForEveryThreadCount) {
  const std::vector<float> data = many_blocks();
  const std::vector<std::uint8_t> stream = Compressor({.block_elements = 1000}).compress(data);
  for (const unsigned threads : kThreadCounts) {
    ASSERT_TRUE(bit_identical(Decompressor({.threads = threads}).decompress(stream), data)) << "threads=" << threads;
  }
}

TEST(Threads, FewerBlocksThanThreads) {
  const std::vector<float> data = turbocore::testing::smooth_wave(2500);
  for (const std::size_t block_elements : {std::size_t{1000}, std::size_t{100000}}) {
    const auto stream = Compressor({.block_elements = block_elements, .threads = 16}).compress(data);
    EXPECT_TRUE(bit_identical(Decompressor({.threads = 16}).decompress(stream), data));
  }
  EXPECT_TRUE(Decompressor({.threads = 16}).decompress(Compressor({.threads = 16}).compress({})).empty());
}

TEST(Threads, DamagedBlockIsReportedForEveryThreadCount) {
  const std::vector<float> data = many_blocks();
  std::vector<std::uint8_t> stream = Compressor({.block_elements = 1000}).compress(data);
  // Break the stored CRC of block 50 (index entries are 12 bytes, CRC at +4).
  stream[32 + 50 * 12 + 4] ^= 0xFFu;
  for (const unsigned threads : kThreadCounts) {
    try {
      (void)Decompressor({.threads = threads}).decompress(stream);
      ADD_FAILURE() << "expected turbocore::Error, threads=" << threads;
    } catch (const turbocore::Error& error) {
      EXPECT_EQ(error.code(), turbocore::ErrorCode::ChecksumMismatch) << "threads=" << threads;
    }
  }
}

}  // namespace
