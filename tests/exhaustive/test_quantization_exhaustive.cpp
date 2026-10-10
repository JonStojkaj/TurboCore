// Checks the quantizers on every one of the 2^32 FP32 bit patterns.
//
// Two independent oracles:
//   1. The value-based rounding model in quantization_reference.hpp, for
//      both formats.
//   2. On AArch64, the CPU's own FP32 <-> binary16 conversion (FCVT), which
//      also pins down the exact NaN bit patterns.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "backend_support.hpp"
#include "quantization_reference.hpp"
#include "turbocore/turbocore.hpp"

namespace {

using turbocore::QuantFormat;

constexpr std::uint64_t kPatternCount = std::uint64_t{1} << 32;
constexpr std::uint64_t kNoFailure = ~std::uint64_t{0};

bool is_nan_bits(std::uint32_t bits) { return (bits & 0x7FFFFFFFu) > 0x7F800000u; }

/// Runs `passes(bits)` for every bit pattern across all hardware threads and
/// returns the smallest failing pattern, or kNoFailure.
template <typename Predicate>
std::uint64_t first_failure(Predicate passes) {
  const unsigned thread_count = std::max(1u, std::thread::hardware_concurrency());
  std::atomic<std::uint64_t> failure{kNoFailure};
  std::vector<std::thread> threads;
  for (unsigned t = 0; t < thread_count; ++t) {
    threads.emplace_back([&, t] {
      const std::uint64_t begin = kPatternCount * t / thread_count;
      const std::uint64_t end = kPatternCount * (t + 1) / thread_count;
      for (std::uint64_t pattern = begin; pattern < end; ++pattern) {
        if (!passes(static_cast<std::uint32_t>(pattern))) {
          std::uint64_t seen = failure.load();
          while (pattern < seen && !failure.compare_exchange_weak(seen, pattern)) {
          }
          return;
        }
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  return failure.load();
}

/// Round trip agrees with the model; NaN stays NaN with its sign.
bool matches_model(std::uint32_t bits, QuantFormat format) {
  const float x = std::bit_cast<float>(bits);
  const float actual = format == QuantFormat::Float16
                           ? turbocore::float16_bits_to_float32(turbocore::float32_to_float16_bits(x))
                           : turbocore::bfloat16_bits_to_float32(turbocore::float32_to_bfloat16_bits(x));
  const auto actual_bits = std::bit_cast<std::uint32_t>(actual);
  if (is_nan_bits(bits)) {
    return is_nan_bits(actual_bits) && ((actual_bits ^ bits) & 0x80000000u) == 0;
  }
  if ((bits & 0x7FFFFFFFu) == 0x7F800000u) {
    return actual_bits == bits;
  }
  return actual_bits == std::bit_cast<std::uint32_t>(turbocore::testing::reference_round_trip(x, format));
}

TEST(ExhaustiveQuantization, Float16MatchesTheRoundingModelForAllInputs) {
  const std::uint64_t failure = first_failure([](std::uint32_t bits) { return matches_model(bits, QuantFormat::Float16); });
  EXPECT_EQ(failure, kNoFailure) << "first failing FP32 pattern: 0x" << std::hex << failure;
}

TEST(ExhaustiveQuantization, BFloat16MatchesTheRoundingModelForAllInputs) {
  const std::uint64_t failure = first_failure([](std::uint32_t bits) { return matches_model(bits, QuantFormat::BFloat16); });
  EXPECT_EQ(failure, kNoFailure) << "first failing FP32 pattern: 0x" << std::hex << failure;
}

// The half-ULP guarantee, stated directly on the definition used by the
// error-analysis module rather than through the rounding model.
TEST(ExhaustiveQuantization, ErrorNeverExceedsHalfATargetUlp) {
  for (const QuantFormat format : {QuantFormat::Float16, QuantFormat::BFloat16}) {
    const std::uint64_t failure = first_failure([format](std::uint32_t bits) {
      const float x = std::bit_cast<float>(bits);
      const float y = format == QuantFormat::Float16
                          ? turbocore::float16_bits_to_float32(turbocore::float32_to_float16_bits(x))
                          : turbocore::bfloat16_bits_to_float32(turbocore::float32_to_bfloat16_bits(x));
      const auto y_bits = std::bit_cast<std::uint32_t>(y);
      if ((bits & 0x7FFFFFFFu) >= 0x7F800000u || (y_bits & 0x7FFFFFFFu) >= 0x7F800000u) {
        return true;  // non-finite pairs are outside the guarantee
      }
      const double error = static_cast<double>(y) - static_cast<double>(x);
      return (error < 0 ? -error : error) <= 0.5 * turbocore::target_ulp(x, format);
    });
    EXPECT_EQ(failure, kNoFailure) << "first failing FP32 pattern: 0x" << std::hex << failure;
  }
}

// The AVX2 quantizers against the scalar reference for every input, through
// the array API in vector-sized batches so each pattern is seen in a lane of
// a full vector. Runs on native x86, or under Rosetta 2 from an x86_64 build.
TEST(ExhaustiveQuantization, Avx2IsBitIdenticalToScalarForAllInputs) {
  TURBOCORE_NEEDS_AVX2();
  constexpr std::uint64_t kBatch = 1 << 12;
  for (const QuantFormat format : {QuantFormat::Float16, QuantFormat::BFloat16}) {
    const unsigned thread_count = std::max(1u, std::thread::hardware_concurrency());
    std::atomic<std::uint64_t> failure{kNoFailure};
    std::vector<std::thread> threads;
    for (unsigned t = 0; t < thread_count; ++t) {
      threads.emplace_back([&, t] {
        const std::uint64_t batches = kPatternCount / kBatch;
        std::vector<float> in(kBatch);
        std::vector<std::uint16_t> scalar(kBatch);
        std::vector<std::uint16_t> avx2(kBatch);
        for (std::uint64_t batch = batches * t / thread_count; batch < batches * (t + 1) / thread_count; ++batch) {
          for (std::uint64_t i = 0; i < kBatch; ++i) {
            in[i] = std::bit_cast<float>(static_cast<std::uint32_t>(batch * kBatch + i));
          }
          turbocore::quantize(in, scalar, format, turbocore::Backend::Scalar);
          turbocore::quantize(in, avx2, format, turbocore::Backend::Avx2);
          if (scalar != avx2) {
            for (std::uint64_t i = 0; i < kBatch; ++i) {
              if (scalar[i] != avx2[i]) {
                std::uint64_t seen = failure.load();
                const std::uint64_t pattern = batch * kBatch + i;
                while (pattern < seen && !failure.compare_exchange_weak(seen, pattern)) {
                }
                return;
              }
            }
          }
        }
      });
    }
    for (std::thread& thread : threads) {
      thread.join();
    }
    EXPECT_EQ(failure.load(), kNoFailure) << "first differing FP32 pattern: 0x" << std::hex << failure.load();
  }
}

#if defined(__aarch64__) && defined(__ARM_FP16_FORMAT_IEEE)

std::uint16_t hardware_float16_bits(float x) {
  const __fp16 half = static_cast<__fp16>(x);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &half, sizeof(bits));
  return bits;
}

float hardware_float16_expand(std::uint16_t bits) {
  __fp16 half;
  std::memcpy(&half, &bits, sizeof(half));
  return static_cast<float>(half);
}

TEST(ExhaustiveQuantization, Float16IsBitIdenticalToTheCpuConversion) {
  const std::uint64_t failure = first_failure([](std::uint32_t bits) {
    // volatile: force the conversion to execute on the CPU instead of being
    // folded by the compiler's own software model.
    volatile float x = std::bit_cast<float>(bits);
    return turbocore::float32_to_float16_bits(x) == hardware_float16_bits(x);
  });
  EXPECT_EQ(failure, kNoFailure) << "first failing FP32 pattern: 0x" << std::hex << failure;
}

TEST(ExhaustiveQuantization, Float16ExpansionIsBitIdenticalToTheCpuConversion) {
  for (std::uint32_t h = 0; h <= 0xFFFFu; ++h) {
    volatile std::uint16_t half = static_cast<std::uint16_t>(h);
    ASSERT_EQ(std::bit_cast<std::uint32_t>(turbocore::float16_bits_to_float32(half)),
              std::bit_cast<std::uint32_t>(hardware_float16_expand(half)))
        << std::hex << h;
  }
}

#endif  // AArch64 hardware oracle

}  // namespace
