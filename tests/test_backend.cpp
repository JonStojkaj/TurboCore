#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "backend_support.hpp"
#include "test_data.hpp"
#include "turbocore/turbocore.hpp"

// Backend parity through the public API: the backend may change how fast an
// answer is produced, never the answer.

namespace {

using turbocore::Backend;
using turbocore::CompressionOptions;
using turbocore::Compressor;
using turbocore::Decompressor;
using turbocore::Predictor;
using turbocore::QuantFormat;
using turbocore::testing::bit_identical;

TEST(Backend, ScalarAndAutoAreAlwaysAvailable) {
  EXPECT_TRUE(turbocore::is_backend_available(Backend::Scalar));
  EXPECT_TRUE(turbocore::is_backend_available(Backend::Auto));
  EXPECT_NE(turbocore::best_available_backend(), Backend::Auto);
  EXPECT_TRUE(turbocore::is_backend_available(turbocore::best_available_backend()));
  EXPECT_STREQ(turbocore::backend_name(Backend::Scalar), "scalar");
  EXPECT_STREQ(turbocore::backend_name(Backend::Avx2), "avx2");
}

TEST(Backend, AvailabilityAgreesWithRuntimeSupport) {
  EXPECT_EQ(turbocore::is_backend_available(Backend::Avx2), turbocore::runtime_support().avx2_usable);
  if (turbocore::is_backend_available(Backend::Avx2)) {
    EXPECT_EQ(turbocore::best_available_backend(), Backend::Avx2);
  }
}

TEST(Backend, RequestingAnUnavailableBackendFailsLoudly) {
  if (turbocore::is_backend_available(Backend::Avx2)) {
    GTEST_SKIP() << "AVX2 is available here, so there is no unavailable backend to request";
  }
  const auto expect_unavailable = [](auto&& action) {
    try {
      action();
      ADD_FAILURE() << "expected turbocore::Error";
    } catch (const turbocore::Error& error) {
      EXPECT_EQ(error.code(), turbocore::ErrorCode::BackendUnavailable);
    }
  };
  expect_unavailable([] { const Compressor c({.backend = Backend::Avx2}); });
  expect_unavailable([] { const Decompressor d({.backend = Backend::Avx2}); });
  expect_unavailable([] {
    const std::vector<float> in(4);
    std::vector<std::uint16_t> out(4);
    turbocore::quantize(in, out, QuantFormat::Float16, Backend::Avx2);
  });
}

struct Dataset {
  std::string name;
  std::vector<float> values;
};

std::vector<Dataset> datasets() {
  namespace t = turbocore::testing;
  return {
      {"empty", {}},
      {"one_element", {1.0F}},
      {"seven", t::ramp(7)},  // shorter than any vector
      {"smooth_wave", t::smooth_wave(70001)},
      {"uniform_noise", t::uniform_noise(20003, 1)},
      {"random_bit_patterns", t::random_bit_patterns(20005, 2)},
      {"special_values", t::special_values(4099)},
  };
}

std::vector<CompressionOptions> option_sets() {
  std::vector<CompressionOptions> sets;
  for (const std::optional<QuantFormat> quantization :
       {std::optional<QuantFormat>{}, std::optional{QuantFormat::Float16}, std::optional{QuantFormat::BFloat16}}) {
    for (const Predictor predictor : {Predictor::None, Predictor::Delta, Predictor::Xor}) {
      for (const bool byte_shuffle : {false, true}) {
        // 1000 elements per block: not a multiple of any vector width.
        sets.push_back({.quantization = quantization,
                        .predictor = predictor,
                        .byte_shuffle = byte_shuffle,
                        .level = 1,
                        .block_elements = 1000});
      }
    }
  }
  return sets;
}

TEST(BackendParity, CompressedStreamsAreByteIdentical) {
  TURBOCORE_NEEDS_AVX2();
  for (const Dataset& dataset : datasets()) {
    for (CompressionOptions options : option_sets()) {
      SCOPED_TRACE(dataset.name + " predictor=" + std::to_string(static_cast<int>(options.predictor)) +
                   " shuffle=" + std::to_string(options.byte_shuffle) +
                   " quantized=" + std::to_string(options.quantization.has_value()));
      options.backend = Backend::Scalar;
      const std::vector<std::uint8_t> scalar = Compressor(options).compress(dataset.values);
      options.backend = Backend::Avx2;
      const std::vector<std::uint8_t> avx2 = Compressor(options).compress(dataset.values);
      ASSERT_EQ(avx2, scalar);
    }
  }
}

TEST(BackendParity, EitherBackendDecodesTheSameValues) {
  TURBOCORE_NEEDS_AVX2();
  for (const Dataset& dataset : datasets()) {
    for (const CompressionOptions& options : option_sets()) {
      SCOPED_TRACE(dataset.name);
      const std::vector<std::uint8_t> stream = Compressor(options).compress(dataset.values);
      const std::vector<float> scalar = Decompressor({.backend = Backend::Scalar}).decompress(stream);
      const std::vector<float> avx2 = Decompressor({.backend = Backend::Avx2}).decompress(stream);
      ASSERT_TRUE(bit_identical(avx2, scalar));
      if (!options.quantization.has_value()) {
        ASSERT_TRUE(bit_identical(avx2, dataset.values));
      }
    }
  }
}

TEST(BackendParity, QuantizationIsBitIdentical) {
  TURBOCORE_NEEDS_AVX2();
  for (const Dataset& dataset : datasets()) {
    for (const QuantFormat format : {QuantFormat::Float16, QuantFormat::BFloat16}) {
      SCOPED_TRACE(dataset.name);
      const std::size_t n = dataset.values.size();
      std::vector<std::uint16_t> scalar(n);
      std::vector<std::uint16_t> avx2(n);
      turbocore::quantize(dataset.values, scalar, format, Backend::Scalar);
      turbocore::quantize(dataset.values, avx2, format, Backend::Avx2);
      ASSERT_EQ(avx2, scalar);

      ASSERT_TRUE(bit_identical(turbocore::quantize_round_trip(dataset.values, format, Backend::Avx2),
                                turbocore::quantize_round_trip(dataset.values, format, Backend::Scalar)));
    }
  }
}

// --- Backend::Cuda ----------------------------------------------------------
// These need a CUDA build and a GPU; elsewhere they are skipped. They check
// the backend's contract: byte-identical streams and bit-identical decoded
// values compared with the scalar backend.

TEST(CudaParity, CompressedStreamsAreByteIdentical) {
  TURBOCORE_NEEDS_CUDA();
  for (const Dataset& dataset : datasets()) {
    for (CompressionOptions options : option_sets()) {
      SCOPED_TRACE(dataset.name + " predictor=" + std::to_string(static_cast<int>(options.predictor)) +
                   " shuffle=" + std::to_string(options.byte_shuffle) +
                   " quantized=" + std::to_string(options.quantization.has_value()));
      options.backend = Backend::Scalar;
      const std::vector<std::uint8_t> scalar = Compressor(options).compress(dataset.values);
      options.backend = Backend::Cuda;
      ASSERT_EQ(Compressor(options).compress(dataset.values), scalar);
    }
  }
}

TEST(CudaParity, DecodesTheSameValues) {
  TURBOCORE_NEEDS_CUDA();
  for (const Dataset& dataset : datasets()) {
    for (const CompressionOptions& options : option_sets()) {
      SCOPED_TRACE(dataset.name);
      const std::vector<std::uint8_t> stream = Compressor(options).compress(dataset.values);
      const std::vector<float> scalar = Decompressor({.backend = Backend::Scalar}).decompress(stream);
      ASSERT_TRUE(bit_identical(Decompressor({.backend = Backend::Cuda}).decompress(stream), scalar));
    }
  }
}

// Default-sized blocks: the two-level prefix scan and the histogram hand-off
// only do real work once a block spans many scan tiles and a plane is large
// enough to be its own DEFLATE segment.
TEST(CudaParity, FullSizeBlocks) {
  TURBOCORE_NEEDS_CUDA();
  namespace t = turbocore::testing;
  std::vector<float> data = t::smooth_wave(150000);
  const std::vector<float> noise = t::random_bit_patterns(50001, 3);
  data.insert(data.end(), noise.begin(), noise.end());

  for (const std::optional<QuantFormat> quantization :
       {std::optional<QuantFormat>{}, std::optional{QuantFormat::Float16}, std::optional{QuantFormat::BFloat16}}) {
    for (const Predictor predictor : {Predictor::Delta, Predictor::Xor}) {
      CompressionOptions options{.quantization = quantization, .predictor = predictor, .level = 1};
      options.backend = Backend::Scalar;
      const std::vector<std::uint8_t> scalar = Compressor(options).compress(data);
      options.backend = Backend::Cuda;
      ASSERT_EQ(Compressor(options).compress(data), scalar);
      ASSERT_TRUE(bit_identical(Decompressor({.backend = Backend::Cuda}).decompress(scalar),
                                Decompressor({.backend = Backend::Scalar}).decompress(scalar)));
    }
  }
}

TEST(CudaParity, DamagedStreamsAreRejected) {
  TURBOCORE_NEEDS_CUDA();
  const std::vector<float> data = turbocore::testing::smooth_wave(5000);
  std::vector<std::uint8_t> stream = Compressor({.block_elements = 1000}).compress(data);
  stream[32 + 2 * 12 + 4] ^= 0xFFu;  // CRC of block 2
  try {
    (void)Decompressor({.backend = Backend::Cuda}).decompress(stream);
    ADD_FAILURE() << "expected turbocore::Error";
  } catch (const turbocore::Error& error) {
    EXPECT_EQ(error.code(), turbocore::ErrorCode::ChecksumMismatch);
  }
}

TEST(CudaParity, BlockSizeLimit) {
  TURBOCORE_NEEDS_CUDA();
  const std::size_t too_large = (std::size_t{1} << 19) + 1;
  EXPECT_THROW(Compressor({.block_elements = too_large, .backend = Backend::Cuda}), turbocore::Error);
  // A stream written with larger blocks is still decodable when the GPU is
  // requested: the decoder falls back to the CPU for it.
  const std::vector<float> data = turbocore::testing::ramp(1000);
  const auto stream = Compressor({.block_elements = too_large}).compress(data);
  EXPECT_TRUE(bit_identical(Decompressor({.backend = Backend::Cuda}).decompress(stream), data));
}

TEST(Backend, CudaIsNeverChosenAutomatically) {
  EXPECT_NE(turbocore::best_available_backend(), Backend::Cuda);
  EXPECT_STREQ(turbocore::backend_name(Backend::Cuda), "cuda");
  EXPECT_EQ(turbocore::is_backend_available(Backend::Cuda), turbocore::runtime_support().cuda_device_count > 0);
  if (!turbocore::is_backend_available(Backend::Cuda)) {
    EXPECT_THROW(Compressor({.backend = Backend::Cuda}), turbocore::Error);
    EXPECT_THROW(Decompressor({.backend = Backend::Cuda}), turbocore::Error);
  }
}

// The array API must equal the documented scalar functions on whatever
// backend Auto picks, which is the path every other test exercises.
TEST(BackendParity, AutoMatchesTheScalarDefinition) {
  const std::vector<float> data = turbocore::testing::random_bit_patterns(10007, 5);
  std::vector<std::uint16_t> packed(data.size());
  turbocore::quantize(data, packed, QuantFormat::Float16);
  for (std::size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(packed[i], turbocore::float32_to_float16_bits(data[i])) << i;
  }
  turbocore::quantize(data, packed, QuantFormat::BFloat16);
  for (std::size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(packed[i], turbocore::float32_to_bfloat16_bits(data[i])) << i;
  }
}

}  // namespace
