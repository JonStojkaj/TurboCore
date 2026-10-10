# TurboCore

A standalone C++20 library for high-performance compression of 32-bit
floating-point data and HDR images, with scalar, AVX2 and CUDA backends, a
floating-point error-analysis module and Python bindings.

**Status:** the scalar reference implementation works:

- lossless FP32 compression (delta/XOR predictor, byte shuffle, DEFLATE) in a
  checksummed block container; single-threaded and not yet fast;
- correctly rounded FP32 → FP16 / BFloat16 quantization with an error-analysis
  module (ULP error, MSE, PSNR, NaN/Inf accounting), selectable as a lossy
  mode of the compressor.

An AVX2 backend covers the predictors, byte shuffle, quantization and CRC. It
is verified bit-identical to the scalar code (under Rosetta 2) but has not been
benchmarked on native x86 yet, and the DEFLATE stage, which dominates run
time, is still scalar. The CUDA backend and the Python compression API are not
implemented.
See [plan.md](plan.md) for the architecture and roadmap.

## Usage

```cpp
#include <turbocore/turbocore.hpp>

std::vector<float> data = /* ... */;
std::vector<std::uint8_t> stream = turbocore::Compressor().compress(data);
std::vector<float> restored = turbocore::Decompressor().decompress(stream);
// restored is bit-identical to data, including NaN payloads and signed zeros.

// How much would storing this data as 16-bit floats cost?
turbocore::QuantizationReport report =
    turbocore::analyze_quantization(data, turbocore::QuantFormat::Float16);
// report.error.psnr_db, report.error.mean_squared_error,
// report.max_target_ulp_error (at most 0.5), report.error.overflow_to_infinity_count, ...

// Lossy mode: round to FP16, then compress the 16-bit values losslessly.
turbocore::Compressor lossy({.quantization = turbocore::QuantFormat::Float16});
std::vector<float> approx = turbocore::Decompressor().decompress(lossy.compress(data));
// approx == turbocore::quantize_round_trip(data, turbocore::QuantFormat::Float16), bit for bit.
```

## Building

Requires CMake ≥ 3.25 and a C++20 compiler. GoogleTest, Google Benchmark and
pybind11 are downloaded at configure time (pinned and hash-verified).

```sh
cmake --preset dev            # Debug, ASan + UBSan, warnings as errors
cmake --build --preset dev
ctest --preset dev
```

| Preset        | Purpose |
|---------------|---------|
| `dev`         | Debug with sanitizers |
| `release`     | Release, benchmarks and the Python module |
| `x86-rosetta` | x86_64 build on Apple Silicon to exercise the AVX2 backend under Rosetta 2 (correctness only) |
| `cuda`        | Release with the CUDA backend required |

## Options

| Option | Default | Meaning |
|--------|---------|---------|
| `TURBOCORE_ENABLE_AVX2` | `AUTO` | `AUTO` builds the backend if the toolchain can; `ON` fails the configure if it cannot; `OFF` skips it |
| `TURBOCORE_ENABLE_CUDA` | `AUTO` | Same, for the CUDA backend |
| `TURBOCORE_BUILD_TESTS` | `ON` when top-level | GoogleTest unit tests |
| `TURBOCORE_BUILD_BENCHMARKS` | `OFF` | Google Benchmark suite |
| `TURBOCORE_BUILD_PYTHON` | `OFF` | pybind11 extension |
| `TURBOCORE_WARNINGS_AS_ERRORS` | `OFF` | `-Werror` / `/WX` |
| `TURBOCORE_ENABLE_SANITIZERS` | `OFF` | ASan + UBSan |
| `TURBOCORE_USE_SYSTEM_DEPS` | `OFF` | Use `find_package` instead of the pinned downloads |
| `BUILD_SHARED_LIBS` | `OFF` | Build a shared library |

The AVX2 backend is compiled into its own object library and selected at
runtime by CPUID, so a binary built with it still runs on CPUs without AVX2.

## Python

```sh
cmake --preset release && cmake --build --preset release
PYTHONPATH=build/release/python python3 -c "import turbocore; print(turbocore.build_info())"
```
