// Per-kernel throughput for each backend, in bytes of element data per
// second, so every row is comparable with the memcpy ceiling.
//
// Each benchmark takes {backend, element count}. The sizes span L1 (16 KiB of
// FP32), L2 (256 KiB, the default block size) and DRAM (64 MiB), because the
// SIMD gain is largest while the data fits in cache and shrinks to the
// memory-bandwidth limit beyond it.
//
// Numbers from an emulated x86 build (Rosetta 2) say nothing about real AVX2
// hardware. Report only figures measured on native x86.

#include <benchmark/benchmark.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "detail/kernels.hpp"

namespace {

using turbocore::detail::KernelTable;

constexpr int kScalar = 0;
constexpr int kAvx2 = 1;

const KernelTable* table_for(benchmark::State& state) {
  const KernelTable* table =
      state.range(0) == kAvx2 ? turbocore::detail::avx2_kernels() : &turbocore::detail::scalar_kernels();
  if (table == nullptr) {
    state.SkipWithError("AVX2 backend not available on this machine");
    return nullptr;
  }
  state.SetLabel(table->name);
  return table;
}

template <typename T>
std::vector<T> random_data(std::size_t n) {
  std::mt19937 rng(42);
  std::vector<T> v(n);
  for (T& x : v) {
    x = static_cast<T>(rng());
  }
  return v;
}

std::vector<float> finite_floats(std::size_t n) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1000.0F, 1000.0F);
  std::vector<float> v(n);
  for (float& x : v) {
    x = dist(rng);
  }
  return v;
}

void finish(benchmark::State& state, std::size_t bytes) {
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * static_cast<std::int64_t>(bytes));
}

/// In-place kernels: benchmarked on their own output, which is fine because
/// their cost does not depend on the data.
template <typename T, typename Member>
void run_in_place(benchmark::State& state, Member kernel) {
  const KernelTable* table = table_for(state);
  if (table == nullptr) {
    return;
  }
  std::vector<T> data = random_data<T>(static_cast<std::size_t>(state.range(1)));
  for (auto _ : state) {
    (table->*kernel)(data.data(), data.size());
    benchmark::DoNotOptimize(data.data());
    benchmark::ClobberMemory();
  }
  finish(state, data.size() * sizeof(T));
}

template <typename In, typename Out, typename Member>
void run_out_of_place(benchmark::State& state, Member kernel, std::vector<In> in, std::size_t out_count,
                      std::size_t element_count, std::size_t element_bytes) {
  const KernelTable* table = table_for(state);
  if (table == nullptr) {
    return;
  }
  std::vector<Out> out(out_count);
  for (auto _ : state) {
    (table->*kernel)(in.data(), out.data(), element_count);
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  finish(state, element_count * element_bytes);
}

std::size_t count_of(benchmark::State& state) { return static_cast<std::size_t>(state.range(1)); }

void BM_DeltaEncodeU32(benchmark::State& s) { run_in_place<std::uint32_t>(s, &KernelTable::delta_encode_u32); }
void BM_DeltaDecodeU32(benchmark::State& s) { run_in_place<std::uint32_t>(s, &KernelTable::delta_decode_u32); }
void BM_XorEncodeU32(benchmark::State& s) { run_in_place<std::uint32_t>(s, &KernelTable::xor_encode_u32); }
void BM_XorDecodeU32(benchmark::State& s) { run_in_place<std::uint32_t>(s, &KernelTable::xor_decode_u32); }
void BM_DeltaEncodeU16(benchmark::State& s) { run_in_place<std::uint16_t>(s, &KernelTable::delta_encode_u16); }
void BM_DeltaDecodeU16(benchmark::State& s) { run_in_place<std::uint16_t>(s, &KernelTable::delta_decode_u16); }

void BM_ShuffleU32(benchmark::State& s) {
  const std::size_t n = count_of(s);
  run_out_of_place<std::uint32_t, std::uint8_t>(s, &KernelTable::shuffle_u32, random_data<std::uint32_t>(n), n * 4, n, 4);
}
void BM_UnshuffleU32(benchmark::State& s) {
  const std::size_t n = count_of(s);
  run_out_of_place<std::uint8_t, std::uint32_t>(s, &KernelTable::unshuffle_u32, random_data<std::uint8_t>(n * 4), n, n, 4);
}
void BM_ShuffleU16(benchmark::State& s) {
  const std::size_t n = count_of(s);
  run_out_of_place<std::uint16_t, std::uint8_t>(s, &KernelTable::shuffle_u16, random_data<std::uint16_t>(n), n * 2, n, 2);
}
void BM_UnshuffleU16(benchmark::State& s) {
  const std::size_t n = count_of(s);
  run_out_of_place<std::uint8_t, std::uint16_t>(s, &KernelTable::unshuffle_u16, random_data<std::uint8_t>(n * 2), n, n, 2);
}

// Quantization throughput is counted in FP32 input bytes; dequantization in
// FP32 output bytes.
void BM_QuantizeF16(benchmark::State& s) {
  const std::size_t n = count_of(s);
  run_out_of_place<float, std::uint16_t>(s, &KernelTable::quantize_f16, finite_floats(n), n, n, 4);
}
void BM_QuantizeBF16(benchmark::State& s) {
  const std::size_t n = count_of(s);
  run_out_of_place<float, std::uint16_t>(s, &KernelTable::quantize_bf16, finite_floats(n), n, n, 4);
}
void BM_DequantizeF16(benchmark::State& s) {
  const std::size_t n = count_of(s);
  run_out_of_place<std::uint16_t, float>(s, &KernelTable::dequantize_f16, random_data<std::uint16_t>(n), n, n, 4);
}
void BM_DequantizeBF16(benchmark::State& s) {
  const std::size_t n = count_of(s);
  run_out_of_place<std::uint16_t, float>(s, &KernelTable::dequantize_bf16, random_data<std::uint16_t>(n), n, n, 4);
}

void BM_Crc32c(benchmark::State& state) {
  const KernelTable* table = table_for(state);
  if (table == nullptr) {
    return;
  }
  const std::vector<std::uint8_t> data = random_data<std::uint8_t>(count_of(state) * 4);
  for (auto _ : state) {
    benchmark::DoNotOptimize(table->crc32c(data.data(), data.size()));
  }
  finish(state, data.size());
}

void backends_and_sizes(benchmark::internal::Benchmark* b) {
  b->ArgNames({"backend", "elements"});
  for (const int backend : {kScalar, kAvx2}) {
    for (const std::int64_t elements : {std::int64_t{1} << 12, std::int64_t{1} << 16, std::int64_t{1} << 24}) {
      b->Args({backend, elements});
    }
  }
}

BENCHMARK(BM_DeltaEncodeU32)->Apply(backends_and_sizes);
BENCHMARK(BM_DeltaDecodeU32)->Apply(backends_and_sizes);
BENCHMARK(BM_XorEncodeU32)->Apply(backends_and_sizes);
BENCHMARK(BM_XorDecodeU32)->Apply(backends_and_sizes);
BENCHMARK(BM_DeltaEncodeU16)->Apply(backends_and_sizes);
BENCHMARK(BM_DeltaDecodeU16)->Apply(backends_and_sizes);
BENCHMARK(BM_ShuffleU32)->Apply(backends_and_sizes);
BENCHMARK(BM_UnshuffleU32)->Apply(backends_and_sizes);
BENCHMARK(BM_ShuffleU16)->Apply(backends_and_sizes);
BENCHMARK(BM_UnshuffleU16)->Apply(backends_and_sizes);
BENCHMARK(BM_QuantizeF16)->Apply(backends_and_sizes);
BENCHMARK(BM_QuantizeBF16)->Apply(backends_and_sizes);
BENCHMARK(BM_DequantizeF16)->Apply(backends_and_sizes);
BENCHMARK(BM_DequantizeBF16)->Apply(backends_and_sizes);
BENCHMARK(BM_Crc32c)->Apply(backends_and_sizes);

}  // namespace
