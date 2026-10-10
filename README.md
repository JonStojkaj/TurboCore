# TurboCore

A standalone C++20 library for high-performance compression of 32-bit
floating-point data and HDR images, with scalar, AVX2 and CUDA backends, a
floating-point error-analysis module and Python bindings.

**Status:** the scalar reference codec works: lossless FP32 compression
(delta/XOR predictor, byte shuffle, DEFLATE) with a checksummed block
container. It is single-threaded and not yet fast. The AVX2 and CUDA backends,
the quantization/error-analysis module and the Python compression API are not
implemented. See [plan.md](plan.md) for the architecture and roadmap.

## Usage

```cpp
#include <turbocore/turbocore.hpp>

std::vector<float> data = /* ... */;
std::vector<std::uint8_t> stream = turbocore::Compressor().compress(data);
std::vector<float> restored = turbocore::Decompressor().decompress(stream);
// restored is bit-identical to data, including NaN payloads and signed zeros.
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
