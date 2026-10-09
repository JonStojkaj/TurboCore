// Memory-bandwidth baseline. A codec cannot be faster than copying its input,
// so this number is the ceiling every TurboCore throughput figure is read
// against. Google Benchmark reports it as bytes_per_second (GB/s).

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace {

void BM_MemcpyFp32(benchmark::State& state) {
  const auto count = static_cast<std::size_t>(state.range(0));
  std::vector<float> src(count);
  std::iota(src.begin(), src.end(), 0.0F);
  std::vector<float> dst(count);

  const std::size_t bytes = count * sizeof(float);
  for (auto _ : state) {
    std::memcpy(dst.data(), src.data(), bytes);
    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) *
                          static_cast<std::int64_t>(bytes));
}

// 64 KiB (fits L2) up to 256 MiB (DRAM-bound) of float data.
BENCHMARK(BM_MemcpyFp32)->RangeMultiplier(16)->Range(1 << 14, 1 << 26);

}  // namespace
