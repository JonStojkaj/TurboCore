// End-to-end throughput of the scalar reference codec. bytes_per_second is
// always measured in uncompressed bytes, for compression and decompression
// alike, so the two are directly comparable with each other and with the
// memcpy ceiling in bench_memory_baseline.cpp. The "ratio" counter is
// uncompressed size / compressed size.

#include <benchmark/benchmark.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "turbocore/turbocore.hpp"

namespace {

constexpr std::size_t kElements = std::size_t{1} << 22;  // 16 MiB of FP32

enum class Data : int { Smooth, NoisySensor, RandomBits };

std::vector<float> make_data(Data kind) {
  std::vector<float> v(kElements);
  std::mt19937 rng(12345);
  switch (kind) {
    case Data::Smooth:
      // Slowly varying field: the best case for the predictor.
      for (std::size_t i = 0; i < v.size(); ++i) {
        const double t = static_cast<double>(i) * 1e-4;
        v[i] = static_cast<float>(1000.0 + 40.0 * std::sin(t) + 3.0 * std::sin(17.0 * t));
      }
      break;
    case Data::NoisySensor: {
      // Smooth signal plus noise: compressible exponents, random low mantissa.
      std::normal_distribution<double> noise(0.0, 0.05);
      for (std::size_t i = 0; i < v.size(); ++i) {
        const double t = static_cast<double>(i) * 1e-4;
        v[i] = static_cast<float>(20.0 + 5.0 * std::sin(t) + noise(rng));
      }
      break;
    }
    case Data::RandomBits:
      // Incompressible: measures the cost of trying and falling back to stored blocks.
      for (float& x : v) {
        x = std::bit_cast<float>(static_cast<std::uint32_t>(rng()));
      }
      break;
  }
  return v;
}

void set_counters(benchmark::State& state, std::size_t raw_bytes, std::size_t compressed_bytes) {
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * static_cast<std::int64_t>(raw_bytes));
  state.counters["ratio"] = static_cast<double>(raw_bytes) / static_cast<double>(compressed_bytes);
}

void BM_Compress(benchmark::State& state) {
  const std::vector<float> data = make_data(static_cast<Data>(state.range(0)));
  const turbocore::Compressor compressor(
      {.level = static_cast<int>(state.range(1)), .threads = static_cast<unsigned>(state.range(2))});
  std::size_t compressed_bytes = 0;
  for (auto _ : state) {
    const std::vector<std::uint8_t> stream = compressor.compress(data);
    compressed_bytes = stream.size();
    benchmark::DoNotOptimize(stream.data());
  }
  set_counters(state, data.size() * sizeof(float), compressed_bytes);
}

void BM_Decompress(benchmark::State& state) {
  const std::vector<float> data = make_data(static_cast<Data>(state.range(0)));
  const std::vector<std::uint8_t> stream =
      turbocore::Compressor({.level = static_cast<int>(state.range(1))}).compress(data);
  const turbocore::Decompressor decompressor({.threads = static_cast<unsigned>(state.range(2))});
  for (auto _ : state) {
    const std::vector<float> decoded = decompressor.decompress(stream);
    benchmark::DoNotOptimize(decoded.data());
  }
  set_counters(state, data.size() * sizeof(float), stream.size());
}

void apply_arguments(benchmark::internal::Benchmark* b) {
  // threads: 1 isolates the codec itself; 0 uses every hardware thread.
  b->ArgNames({"data", "level", "threads"})->Unit(benchmark::kMillisecond)->UseRealTime();
  for (const Data kind : {Data::Smooth, Data::NoisySensor, Data::RandomBits}) {
    for (const int level : {1, 6, 9}) {
      for (const int threads : {1, 0}) {
        b->Args({static_cast<int>(kind), level, threads});
      }
    }
  }
}

BENCHMARK(BM_Compress)->Apply(apply_arguments);
BENCHMARK(BM_Decompress)->Apply(apply_arguments);

}  // namespace
