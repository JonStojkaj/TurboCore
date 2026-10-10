# TurboCore

A standalone C++20 library for high-performance compression of 32-bit
floating-point data and HDR images, with scalar, AVX2 and CUDA backends, a
floating-point error-analysis module and Python bindings.

**Status**

- Lossless FP32 compression (delta/XOR predictor, byte shuffle, DEFLATE) in a
  checksummed block container, multi-threaded over blocks. The DEFLATE
  streams are standard RFC 1951 and cross-checked against zlib.
- Correctly rounded FP32 → FP16 / BFloat16 quantization with an error-analysis
  module (ULP error, MSE, PSNR, NaN/Inf accounting), selectable as a lossy
  mode of the compressor. Verified on all 2^32 inputs.
- AVX2 backend for the transforms, quantization and CRC, bit-identical to the
  scalar code. Verified under Rosetta 2 only; **not yet benchmarked on native
  x86**.
- CUDA backend for the transform stage, behind CMake guards. **Written without
  access to a GPU: never compiled by nvcc or run on a device, and not built
  or tested by the local build.** It is to be compiled and validated on a
  cloud GPU machine.
- Python API (NumPy) for compression and error analysis.

All performance figures so far come from one Apple Silicon development
machine and synthetic data. See [plan.md](plan.md) for numbers, architecture
and what is still open.

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
| `cuda`        | Release with the CUDA backend required (needs the CUDA Toolkit and a GPU) |

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
export PYTHONPATH=build/release/python
```

```python
import numpy as np, turbocore

x = np.fromfile("field.f32", dtype=np.float32)
stream = turbocore.compress(x)                       # lossless, all cores
assert np.array_equal(turbocore.decompress(stream), x)

report = turbocore.analyze_quantization(x, "float16")
print(report["error"]["psnr_db"], report["max_target_ulp_error"])
small = turbocore.compress(x, quantization="float16")  # lossy
```

Only float32 arrays are accepted; `decompress` returns a 1-D array (the shape
is not stored). There is no `pip install` packaging yet.
