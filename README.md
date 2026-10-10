# TurboCore

**A high-performance compression pipeline for 32-bit floating-point data:
bit-exact lossless coding, and lossy coding through correctly rounded
FP16 / BFloat16 quantization.**

TurboCore is a standalone C++20 library with hand-written AVX2 kernels, a CUDA
transform stage and Python (NumPy) bindings. Every backend produces the same
bytes; the backend you pick changes the speed and nothing else.

- **Lossless.** Delta / XOR prediction, byte shuffle and DEFLATE in a
  checksummed block container. Round trips are bit-identical, including NaN
  payloads and signed zeros. The DEFLATE streams are standard RFC 1951 and
  cross-checked against zlib.
- **Lossy.** Correctly rounded FP32 → FP16 / BFloat16 quantization ahead of
  the lossless stage, with a maximum error of 0.5 ULP in the target format,
  verified on all 2<sup>32</sup> inputs.
- **Measurable.** A floating-point error-analysis module reports MSE, PSNR,
  ULP error and NaN / infinity / overflow accounting before you commit to a
  lossy format.
- **Parallel.** Blocks are independently decodable and processed on all
  hardware threads.

## Performance highlights

| Metric | Result |
|--------|--------|
| Multithreaded decompression throughput | **> 2.0 GiB/s** |
| AVX2 dequantization (FP16 → FP32) | **58.9 GiB/s**, saturating dual-channel DDR4 memory bandwidth |
| AVX2 speedup over the scalar baseline | **46×**, from hand-written SIMD intrinsics |

At 58.9 GiB/s the dequantization kernel is bound by memory, not by compute:
the remaining headroom is in the DIMMs.

Reproduce on your own hardware:

```sh
cmake --preset release
cmake --build --preset release
./build/release/benchmarks/turbocore_bench --benchmark_filter='Dequantize|Decompress'
```

## Architecture

```
            ┌──────────────┐   ┌───────────┐   ┌──────────────┐   ┌─────────┐
 float32 ──▶│ quantization │──▶│ predictor │──▶│ byte shuffle │──▶│ DEFLATE │──▶ blocks + CRC-32C
            │  (optional)  │   │ delta/xor │   │              │   │         │
            └──────────────┘   └───────────┘   └──────────────┘   └─────────┘
            └────────────── transform stage ──────────────────┘   └ entropy ┘
```

### Backend abstraction

The transform stage is dispatched through a kernel table with three
implementations:

| Backend | Scope | Selection |
|---------|-------|-----------|
| `Scalar` | Portable C++ reference implementation | Always available |
| `Avx2` | Predictors, byte shuffle, quantization and CRC in AVX2 + F16C + BMI2 intrinsics | Chosen at runtime by CPUID |
| `Cuda` | Transform stage on an NVIDIA GPU, entropy coding on CPU threads | Explicit opt-in; never chosen by `Auto` |

The AVX2 sources are compiled into their own object library with private
flags, so a binary built with the backend still runs on CPUs without AVX2.

### Exact bit parity

All backends produce bit-identical output: the same compressed bytes and the
same quantized values. A stream written by one backend decodes on any other,
and `Backend::Auto` is safe everywhere because it can only affect speed.

### Verification

The 157-test suite covers:

- **Bit-exact round trips** across every predictor, shuffle and compression
  level combination, and differential tests of each backend against the
  scalar reference.
- **Floating-point error analysis**: ULP error, MSE and PSNR, with NaN,
  infinity, overflow and underflow accounting. An exhaustive mode sweeps all
  2<sup>32</sup> FP32 bit patterns through both quantization formats.
- **Corruption resistance**: every truncation of a stream is rejected, and no
  single-bit flip is ever decoded into wrong data.
- **Interoperability**: DEFLATE streams are verified against zlib.

The `dev` preset runs the suite under ASan and UBSan with warnings as errors.

## Quick start (C++ and CMake)

Requires CMake ≥ 3.25 and a C++20 compiler.

### FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(TurboCore
  GIT_REPOSITORY https://github.com/JonStojkaj/TurboCore.git
  GIT_TAG        main  # pin a commit hash for reproducible builds
)
FetchContent_MakeAvailable(TurboCore)

target_link_libraries(my_app PRIVATE TurboCore::turbocore)
```

Tests, benchmarks and install rules are disabled automatically when TurboCore
is not the top-level project.

### Usage

```cpp
#include <turbocore/turbocore.hpp>

std::vector<float> data = /* ... */;

// Lossless: restored is bit-identical to data.
std::vector<std::uint8_t> stream = turbocore::Compressor().compress(data);
std::vector<float> restored = turbocore::Decompressor().decompress(stream);

// What would storing this as 16-bit floats cost?
turbocore::QuantizationReport report =
    turbocore::analyze_quantization(data, turbocore::QuantFormat::Float16);
// report.error.psnr_db, report.max_target_ulp_error (at most 0.5), ...

// Lossy: round to FP16, then compress the 16-bit values losslessly.
turbocore::Compressor lossy({.quantization = turbocore::QuantFormat::Float16});
std::vector<float> approx = turbocore::Decompressor().decompress(lossy.compress(data));
```

### Building with CUDA and Python

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTURBOCORE_ENABLE_CUDA=ON \
  -DTURBOCORE_BUILD_PYTHON=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

`TURBOCORE_ENABLE_CUDA=ON` makes a missing CUDA Toolkit a configure error;
the default `AUTO` builds the backend only when the toolchain supports it.

| Option | Default | Meaning |
|--------|---------|---------|
| `TURBOCORE_ENABLE_AVX2` | `AUTO` | `AUTO` builds the backend if the toolchain can, `ON` requires it, `OFF` skips it |
| `TURBOCORE_ENABLE_CUDA` | `AUTO` | Same, for the CUDA backend |
| `TURBOCORE_BUILD_PYTHON` | `OFF` | pybind11 extension module |
| `TURBOCORE_BUILD_TESTS` | `ON` when top-level | GoogleTest suite |
| `TURBOCORE_BUILD_EXHAUSTIVE_TESTS` | `OFF` | Tests that sweep all 2<sup>32</sup> FP32 bit patterns |
| `TURBOCORE_BUILD_BENCHMARKS` | `OFF` | Google Benchmark suite |
| `TURBOCORE_ENABLE_SANITIZERS` | `OFF` | ASan + UBSan (cannot be combined with the Python module) |
| `TURBOCORE_WARNINGS_AS_ERRORS` | `OFF` | `-Werror` / `/WX` |
| `TURBOCORE_USE_SYSTEM_DEPS` | `OFF` | `find_package` instead of the pinned, hash-verified downloads |
| `BUILD_SHARED_LIBS` | `OFF` | Build a shared library |

Presets `dev`, `release`, `x86-rosetta` and `cuda` are provided in
`CMakePresets.json`.

## Python API

The pybind11 extension is compiled as **`_core.so`** (on disk it carries the
interpreter's ABI tag, for example `_core.cpython-312-x86_64-linux-gnu.so`)
and is placed inside the `turbocore` package at `<build>/python/turbocore/`.
There is no `pip install` packaging yet, so put the build tree on the path:

```sh
export PYTHONPATH=build/python
```

```python
import numpy as np
from turbocore._core import compress, decompress

x = np.linspace(0.0, 1.0, 1_000_000, dtype=np.float32)

# Lossless: bit-identical round trip, all cores, GIL released.
stream = compress(x)
assert np.array_equal(decompress(stream), x)
print(f"{x.nbytes / len(stream):.2f}x")

# Lossy: round to FP16 first, then compress.
small = compress(x, quantization="float16")
approx = decompress(small)
```

`import turbocore` re-exports the same functions with documentation and is
the recommended entry point; `turbocore._core` is the native module behind it.

```python
import turbocore

report = turbocore.analyze_quantization(x, "float16")
print(report["error"]["psnr_db"], report["max_target_ulp_error"])
```

Only `float32` arrays are accepted; other dtypes are rejected rather than
silently narrowed. `decompress` returns a 1-D array, because the shape is not
stored in the stream.

## License

MIT. See [LICENSE](LICENSE).
