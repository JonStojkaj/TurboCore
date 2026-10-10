# TurboCore — Architecture and Implementation Plan

TurboCore is a standalone C++20 library for compressing 32-bit floating-point
arrays and HDR images, with three interchangeable backends (scalar, AVX2,
CUDA), a floating-point error-analysis module for lossy quantization, and
Python bindings. No external compression library is used anywhere in the
library itself.

Status: **Phase 0 complete. Phase 1 core implemented** (lossless 1D FP32 codec,
single-threaded). **Phase 2 implemented** (quantizers, error metrics, lossy mode in the
container; exhaustively verified). **Phase 3 started**: AVX2 transform,
quantization and CRC kernels are implemented and verified bit-identical under
Rosetta 2, with no native x86 performance numbers yet. Phases 4–5 are not
started.

---

## 1. Goals and non-goals

**Goals**

- Lossless FP32 compression that beats general-purpose byte compressors on
  float data by exploiting float structure before entropy coding.
- An optional, explicitly lossy FP32 → FP16 / BFloat16 path whose error is
  measured and bounded, not assumed.
- One container format, decodable by every backend regardless of which backend
  wrote it.
- Throughput reported honestly in GB/s, end to end, including PCIe transfers
  for the GPU path.

**Non-goals**

- Matching the compression ratio of error-bounded scientific compressors
  (SZ, ZFP). TurboCore is a transform + DEFLATE design, not a predictive
  lossy codec.
- A general-purpose `.gz`/`.zip` tool. The DEFLATE stream is an internal
  building block.
- Multi-GPU, streaming-over-network, or Windows GPU support in the first pass.

---

## 2. Architecture

### 2.1 Compression pipeline

```
FP32 input
   │
   ├─(optional, lossy)─ quantize FP32 → FP16 | BF16        [Phase 2]
   │
   ▼
predictor on integer bit patterns (delta / XOR, 1D or 2D)   [Phase 1]
   ▼
byte-plane shuffle (group byte k of every element)          [Phase 1]
   ▼
LZ77 match finding  →  canonical Huffman coding             [Phase 1]
   ▼
block container (independent blocks + index + checksums)    [Phase 1]
```

Why each stage exists:

- **Predictor on bit patterns, not on float values.** `a - b` in floating
  point is not reversible. Reinterpreting each float as `uint32` and
  subtracting modulo 2³² (or XOR-ing) is exactly reversible for every input
  including NaN payloads, signed zeros and subnormals.
- **Byte-plane shuffle.** After prediction, sign/exponent bytes are highly
  repetitive while low mantissa bytes are near-random. Grouping them puts the
  compressible bytes next to each other, which is what an LZ77 window needs.
- **DEFLATE-style entropy stage.** Per-block Huffman tables are static once
  built, which makes blocks independent and therefore parallel.

### 2.2 Decision: DEFLATE only; no LZW, no separate RLE

- **LZW is dropped.** Its dictionary is adaptive and inherently serial; it
  cannot be split across GPU threads without restarting the dictionary, which
  destroys the ratio.
- **DEFLATE (LZ77 + canonical Huffman) is the entropy stage.** It separates
  match finding from entropy coding. Histogramming and Huffman bit-packing
  parallelise well; that is where the GPU work goes.
- **No separate RLE codec.** A run is an LZ77 match at distance 1, so DEFLATE
  already covers it (100 000 zero bytes encode to under 200 bytes). A
  dedicated RLE path is kept in reserve as a cheap GPU fast path for
  near-constant byte planes if Phase 4 measurements justify it.
- Blocks are **raw RFC 1951 streams**, which gives a free, independent test
  oracle: zlib (via the Python standard library, in the test suite only) must
  decode every block TurboCore writes, and TurboCore must decode what zlib
  writes. The library itself links no compression library.

### 2.3 Container format (TCF, version 1)

Implemented; the byte-level layout is documented in `src/detail/container.hpp`.

| Section      | Contents |
|--------------|----------|
| File header (32 bytes) | magic `TCOR`, format version, element type, predictor, flags (byte shuffle), elements per block, element count, block count, CRC-32C of the header |
| Block index (12 bytes per block) | compressed size, CRC-32C of the block's original bytes, method (stored or DEFLATE) |
| Blocks       | independent raw DEFLATE streams, or the transformed bytes verbatim if DEFLATE does not shrink them |

Independent blocks give random access, bounded memory, thread-level
parallelism on the CPU and block-level parallelism on the GPU. A block's
offset is the prefix sum of the compressed sizes before it. The predictor and
the LZ77 window reset at every block boundary. The block size (default 65 536
elements = 256 KiB, tunable) trades ratio against parallelism and is a
benchmark parameter, not a constant.

Version 2 adds the 16-bit element types (FP16, BFloat16) used by the lossy
mode; the layout is unchanged and lossless streams are still written as
version 1. Not yet in the format: array shape (needed for the 2D image
predictor).

### 2.4 Backends and dispatch

- Public API is object-oriented: `Compressor`, `Decompressor`, value types for
  options and stream info, and a single exception type `turbocore::Error`
  carrying an `ErrorCode`. Internally, an abstract `Codec` with a
  `DeflateCodec` implementation; decoders are `noexcept` and report failure by
  return value, and only the API boundary throws.
- Hot inner loops are **not** virtual. Each backend fills a table of function
  pointers (`KernelTable`) once at start-up; `Backend::Auto` picks the best
  usable table using `runtime_support()`.
- **Parity contract:**
  - Transforms and quantization are *bit-identical* across all backends.
  - Compressed bytes may differ between backends (GPU match finding makes
    different choices), but any stream decodes on any backend to identical
    data.

### 2.5 Rules that the build system already enforces

| Rule | Mechanism (Phase 0) |
|------|---------------------|
| AVX2 flags never leak into scalar code | AVX2 sources live in their own object library with private flags |
| Binary runs on non-AVX2 CPUs | CPUID + XCR0 check gates every AVX2 entry point |
| Bit-exact FP across backends and architectures | `-ffp-contract=off` on all targets; FMA not enabled for the AVX2 backend |
| No inline-function ODR leak from AVX2 TUs | Convention: AVX2 TUs include minimal headers, helpers in anonymous namespaces |
| CUDA TUs do not constrain the C++20 host code | `.cu` files compile as C++17 and only see `src/backends/cuda/*.hpp` |
| Reproducible dependencies | Pinned, SHA-256-verified FetchContent; system packages only by explicit opt-in |
| Unsupported platforms fail at compile time | `static_assert` on IEEE 754 binary32, 8-bit bytes, little-endian |

### 2.6 Repository layout

```
CMakeLists.txt            root build, options, summary
CMakePresets.json         dev / release / x86-rosetta / cuda
cmake/                    compiler options, deps, SIMD + CUDA detection, install
include/turbocore/        public headers
src/                      library sources
  detail/                 internal headers
  backends/avx2/          AVX2 object library
  backends/cuda/          CUDA object library (.cu)
tests/                    GoogleTest
benchmarks/               Google Benchmark
python/                   pybind11 extension + Python package
```

---

## 3. Verification environments

| Environment | What it can verify | Status |
|-------------|--------------------|--------|
| macOS arm64 (development machine) | scalar backend, Python module, sanitizers | working |
| macOS x86_64 under Rosetta 2 (`x86-rosetta` preset) | AVX2 backend **correctness** | working; needs `ROSETTA_ADVERTISE_AVX=1`, which the test preset sets |
| Linux x86_64 cloud instance, native | AVX2 backend **performance** | to be provisioned for the final benchmarks |
| Linux GPU VM (AWS / RunPod) or GitHub Actions | CUDA backend, build and run | to be provisioned; until then CUDA code is written locally and cannot be compiled |

Rosetta numbers are not performance evidence. Every AVX2 throughput figure in
the final report must come from native x86 hardware.

---

## Phase 0 — Build system and scaffolding (complete)

Delivered:

- [x] CMake ≥ 3.25 project with tri-state backend options (`AUTO`/`ON`/`OFF`);
      `ON` with a missing toolchain is a hard configure error.
- [x] AVX2 detection by compile test (correct for cross-architecture builds).
- [x] CUDA detection via `check_language(CUDA)` + `CUDAToolkit`.
- [x] GoogleTest 1.17.0, Google Benchmark 1.9.4, pybind11 3.0.1, pinned by hash.
- [x] Warning set, optional `-Werror`, ASan + UBSan preset.
- [x] Install rules and a `find_package(TurboCore)` config package.
- [x] Toolchain probes: a CPUID gate, an AVX2/F16C/BMI2 probe, a CUDA device
      count probe. These are placeholders that prove the plumbing; they are
      replaced by real kernels in Phases 3 and 4.
- [x] Smoke tests (C++ and Python) and a `memcpy` throughput baseline.

Verified on the development machine: `dev`, `release` and `x86-rosetta`
presets build with `-Werror` on TurboCore's own targets and pass (GoogleTest
emits one warning in its own sources under Apple Clang 21); static and shared builds
pass; an out-of-tree consumer links against the installed package.

**Not verified:** the CUDA path. `cmake/TurboCoreCUDA.cmake` and
`probe_cuda.cu` have never been run through `nvcc`. First action on a GPU
machine is `cmake --preset cuda && cmake --build --preset cuda && ctest --preset cuda`.

---

## Phase 1 — Scalar reference codec (core implemented)

The scalar implementation is the specification. Every later backend is tested
against it, so it is written for clarity and correctness first.

Done:

- [x] **Public API.** `Compressor`, `Decompressor`, `CompressionOptions`,
      `StreamInfo`, `Error`.
- [x] **Bit I/O.** LSB-first `BitWriter` / `BitReader` with 64-bit
      accumulators; the reader latches an overrun flag instead of reading out
      of bounds.
- [x] **CRC-32C.** Slicing-by-8 scalar implementation.
- [x] **Predictors.** 1D delta and XOR on `uint32` bit patterns, exactly
      reversible.
- [x] **Byte-plane shuffle** and unshuffle for 4-byte elements.
- [x] **Huffman.** Length-limited code lengths (Huffman tree plus Kraft-sum
      repair, 15-bit and 7-bit limits), canonical codes, table-driven decoder
      with a 10-bit fast path.
- [x] **LZ77.** Greedy hash-chain match finder, 32 KiB window; levels 1–9 map
      to chain depth.
- [x] **DEFLATE.** Encoder emits fixed or dynamic blocks, whichever is
      smaller. Decoder handles stored, fixed and dynamic blocks and multiple
      blocks per stream, and treats all input as untrusted.
- [x] **Container.** TCF v1 with per-block CRC and stored-block fallback.
- [x] **Tests.** Bit-exact round trips over every predictor × shuffle × level
      combination, on empty input, single elements, constants, ramps, noise,
      random bit patterns, NaN payloads (quiet and signalling), ±0, ±Inf and
      subnormals, and every size around block boundaries; zlib interop in both
      directions; exhaustive truncation and single-bit-flip tests on a whole
      stream; malformed-DEFLATE tests. All run under ASan + UBSan on arm64 and
      under Rosetta.
- [x] **Baseline benchmark** (`bench_codec.cpp`): GB/s and ratio per dataset
      and level.

Still open in Phase 1:

1. **Multi-threading over blocks** with `std::thread`. The codec is currently
   single-threaded; blocks are already independent, so this is mechanical.
2. **Lazy matching** in LZ77 (defer a match if the next position has a longer
   one), for better ratio at higher levels.
3. **Array shape in the header and a 2D left/up predictor** for images.
4. **HDR image I/O.** Minimal PFM reader/writer (trivial format, no
   dependency) so real HDR images can be used in tests and demos.
5. **A libFuzzer target** for the decoder.
6. **Encoder speed on noisy data.** Long hash chains on near-random byte
   planes are the bottleneck (see baseline numbers). Options: skip the LZ77
   search for planes whose histogram is near-uniform, or Huffman-only blocks.

Baseline numbers (Apple Silicon, scalar, single thread, 16 MiB of FP32,
default 256 KiB blocks; development-machine figures, not the final report):

| Dataset | Level | Ratio | Compress | Decompress |
|---------|-------|-------|----------|------------|
| Smooth synthetic field | 1 | 12.5 | 234 MiB/s | 323 MiB/s |
| Smooth synthetic field | 6 | 13.0 | 151 MiB/s | 350 MiB/s |
| Smooth signal + noise  | 1 | 1.51 | 70 MiB/s  | 191 MiB/s |
| Smooth signal + noise  | 6 | 1.57 | 19 MiB/s  | 223 MiB/s |
| Smooth signal + noise  | 9 | 1.59 | 4 MiB/s   | 233 MiB/s |
| Random bit patterns    | any | 1.00 | 61 MiB/s | 1.4 GiB/s |

The smooth field is a best case for the predictor and not representative of
real data. Throughput is two to three orders of magnitude below the `memcpy`
ceiling, as expected for a clarity-first scalar reference; closing that gap is
what Phases 3 and 4 are for.

Exit criteria for the rest of Phase 1:

- Open items 1–5 done, with the same tests passing under ASan + UBSan.
- Ratio and GB/s recorded on at least one real dataset (HDR image or public
  scientific data), not only synthetic ones.

---

## Phase 2 — Quantization and floating-point error analysis (math implemented)

Done before SIMD on purpose: the scalar quantizers defined here are the
bit-exact reference that the AVX2 (F16C) and CUDA versions must reproduce.

Status:

- [x] FP32 ↔ FP16 and FP32 ↔ BFloat16 scalar quantizers (`quantization.hpp`).
- [x] Error metrics: `analyze_error`, `analyze_quantization`, `target_ulp`,
      `ulp_distance` (`error_analysis.hpp`).
- [x] Exhaustive verification over all 2³² inputs against a value-based
      rounding model, and bit-for-bit against the CPU's own conversion on
      AArch64 (`exhaustive` CTest label, enabled in the `release` preset,
      about 16 s on 8+ cores).
- [x] **Container integration.** `CompressionOptions::quantization` selects
      FP16 or BFloat16; the stream then stores 16-bit elements (format
      version 2, with 16-bit predictor and 2-byte shuffle) and decompression
      returns exactly `quantize_round_trip(data, format)`. Lossless streams
      are still written as version 1.
- [ ] NumPy `float16` cross-check (moves to Phase 5 with the Python API).
- [ ] Written error report per benchmark dataset (needs real datasets).

### 2.1 Quantizers

**FP32 → FP16 (IEEE 754 binary16: 1 sign, 5 exponent, 10 mantissa bits)**

- Software implementation with round-to-nearest-even, written from the bit
  layout, not via a compiler `_Float16` cast.
- Overflow: finite values that round to magnitude ≥ 65520 become ±Inf.
- Underflow: gradual, through FP16 subnormals (smallest 2⁻²⁴) to ±0, sign
  preserved.
- NaN: stays NaN; sign and top 10 payload bits preserved; signalling NaNs are
  quieted (matches hardware F16C behaviour, which the AVX2 backend will use).

**FP32 → BFloat16 (top 16 bits of binary32: 1 sign, 8 exponent, 7 mantissa bits)**

- Round-to-nearest-even on the 16 dropped bits, not truncation.
- Same exponent range as FP32, so the only overflow is rounding the largest
  finite values up to ±Inf.
- NaN hazard handled explicitly: a NaN whose payload lives only in the low 16
  bits would truncate to ±Inf. The quiet bit is forced so NaN stays NaN.

Both directions (`quantize`, `dequantize`) are provided; dequantization is
exact.

### 2.2 Error metrics

`QuantizationReport analyze(reference, reconstructed)` returns:

- **ULP error**, in two units, because they answer different questions:
  - in *target-format* ULPs — correct rounding means this is ≤ 0.5 for every
    finite, non-overflowing input. This is an invariant the tests assert.
  - in *FP32* ULPs — the integer distance between the two values on the
    ordered float line, i.e. how many representable floats were skipped.
  - reported as max, mean and a histogram.
- **MSE** accumulated in `double` with compensated (Kahan–Neumaier) summation,
  so the metric is not dominated by its own rounding error on large arrays.
- **PSNR** = 10·log₁₀(peak² / MSE). HDR data has no fixed peak, so the peak
  definition is an explicit parameter (default: max |reference| over finite
  values; alternative: reference dynamic range). MSE = 0 reports +∞.
- **Max absolute** and **max relative** error.
- **Special-value accounting**, reported as counts and excluded from MSE/PSNR
  rather than silently poisoning them: NaN in, NaN preserved, ±Inf in,
  finite → Inf overflows, nonzero → zero underflows, sign-of-zero changes,
  values landing in the subnormal range.

### 2.3 Tests

- **Exhaustive:** all 2³² FP32 bit patterns through both quantizers, compared
  against an independent reference that rounds the *value* in double
  precision from the format definition (the library works on bit fields), and
  on AArch64 against the CPU's FCVT instruction. Built only with
  `TURBOCORE_BUILD_EXHAUSTIVE_TESTS=ON` (the `release` preset), under the
  CTest label `exhaustive`.
- Targeted cases: exact ties, the 65504/65520 overflow boundary, the FP16
  subnormal boundary, largest finite BF16 rounding, every NaN class.
- Metric tests against hand-computed values. A NumPy (`np.float16`)
  cross-check follows with the Python API in Phase 5.

Exit criteria:

- Exhaustive test passes for both formats.
- Target-format ULP error ≤ 0.5 holds for every finite non-overflowing input.
- A written error report (table + ULP histogram) for each benchmark dataset.

---

## Phase 3 — AVX2 backend (transform and quantization kernels implemented)

Status:

- [x] **Dispatch.** `KernelTable` of function pointers per backend, selected
      per operation through `Backend::{Auto, Scalar, Avx2}` on
      `CompressionOptions`, `Decompressor` and the array quantization API.
      Requesting an unavailable backend throws `Error(BackendUnavailable)`.
- [x] **Predictors** (delta and XOR, 32- and 16-bit): element-wise encode,
      in-register prefix scan with a carried lane for decode.
- [x] **Byte-plane shuffle / unshuffle** (4- and 2-byte elements).
- [x] **Quantization:** FP32 ↔ FP16 with F16C, FP32 ↔ BFloat16 with integer
      rounding arithmetic.
- [x] **CRC-32C** with the SSE4.2 `crc32` instruction.
- [x] **Verification:** every kernel bit-identical to scalar for all lengths
      0–100 and beyond, at three offsets from vector alignment, with
      out-of-bounds canaries; compressed streams byte-identical between
      backends for every option combination; AVX2 quantization identical to
      scalar for all 2³² inputs. All of it run under Rosetta 2.
- [x] **Multi-threading over blocks** (`CompressionOptions::threads`,
      `DecompressionOptions::threads`, `std::thread`). Blocks are compressed
      into per-block buffers and assembled in order, so the stream is
      byte-identical for every thread count; if several blocks are damaged the
      decoder reports the first one regardless of timing. Clean under
      ThreadSanitizer.
- [x] **Per-plane DEFLATE blocks with exact cost selection.** Each byte plane
      of a shuffled block is its own DEFLATE block with its own Huffman code,
      and is emitted as the cheapest of stored / Huffman-only / LZ77+Huffman.
      Streams stay valid RFC 1951 (zlib cross-check extended to cover it).
- [x] **Skipping the LZ77 search where it does not pay** (levels 1–8). The
      search runs on an 8 KiB probe of each plane; a plane is searched in full
      only if that predicts a saving of at least 2 % of the whole block.
      Level 9 always searches everything.
- [ ] **Native x86 measurements.** No AVX2 number exists yet: Rosetta timings
      are meaningless (emulated AVX2 runs *slower* than scalar there).
      `bench_kernels.cpp` is ready to run on the x86 cloud instance.
- [ ] Vectorised error metrics, SIMD inside the entropy stage (steps 5–7
      below), lazy matching.

**Where the time went.** The scalar transform kernels (auto-vectorised by the
compiler) run at 12–65 GiB/s, while the Phase 1 codec compressed at
0.02–0.23 GiB/s: the transforms were under 2 % of compression time, so
vectorising them could not move end-to-end throughput. The time was in the
LZ77 match search on byte planes where it finds nothing useful. That is what
the last two items above address, and threading multiplies the result.

Measured on the development machine (Apple Silicon, 6 performance + 4
efficiency cores, scalar backend, 16 MiB of FP32, level 6):

| Dataset | Phase 1 (1 thread) | Now, 1 thread | Now, all cores | Ratio before → now |
|---------|--------------------|---------------|----------------|--------------------|
| Smooth synthetic field, compress | 151 MiB/s | 130 MiB/s | 904 MiB/s | 12.98 → 13.35 |
| Smooth signal + noise, compress  | 19 MiB/s  | 132 MiB/s | 881 MiB/s | 1.573 → 1.656 |
| Random bit patterns, compress    | 61 MiB/s  | 589 MiB/s | 3.2 GiB/s | 1.00 |
| Smooth synthetic field, decompress | 350 MiB/s | 382 MiB/s | 2.8 GiB/s | |
| Smooth signal + noise, decompress  | 223 MiB/s | 295 MiB/s | 2.1 GiB/s | |
| Random bit patterns, decompress    | 1.4 GiB/s | 3.3 GiB/s | 18 GiB/s  | |

Notes on reading this table:

- The smooth dataset got slightly slower single-threaded (the probe pass is
  extra work where LZ77 is worth running anyway) and gained 3 % ratio from
  per-plane Huffman codes.
- On the noisy dataset the ratio went *up* although less searching is done,
  because separate Huffman codes per plane gain more than the skipped
  matches lose. Level 9 reaches 1.687 on it, at 4 MiB/s per thread.
- The skip decision is a heuristic. It can miss repetition that starts after
  the probe; level 9 exists for data where that matters.
- All datasets are synthetic. Real datasets are still an open item.

Steps:

1. **Dispatch.** `KernelTable` and `Backend::{Scalar, AVX2, CUDA, Auto}`,
   selected once using the existing CPUID gate. An environment variable forces
   a backend for testing.
2. **Delta / XOR predictor.** Forward is a plain 8-lane subtract. The inverse
   is a prefix sum; implement the in-register shift-and-add prefix sum with a
   carried lane.
3. **Byte-plane shuffle / unshuffle** with `vpshufb` + unpack/permute.
4. **Quantization.** FP32 ↔ FP16 with F16C (`vcvtps2ph` / `vcvtph2ps`);
   FP32 → BF16 with integer rounding arithmetic.
5. **Error metrics.** Vectorised accumulation in `double` lanes with the same
   summation order contract as scalar, so results stay bit-identical.
6. **Histogram.** Four-table unrolled histogram to break store-to-load
   dependencies.
7. **Entropy stage.** BMI2-assisted bit packing (`pext`/`pdep`, `lzcnt`) and
   vectorised match-length comparison. CRC-32C via `crc32` instructions.
8. **Tail and alignment handling** for every kernel: lengths 0–67, unaligned
   pointers.

Tests:

- Differential tests: every AVX2 kernel versus its scalar counterpart on
  random and adversarial inputs, all lengths 0–67, several misalignments.
  Required to be **bit-identical**.
- The Phase 2 exhaustive quantization test re-run against the AVX2 kernels.
- Full round-trip suite re-run with the backend forced to AVX2.

Exit criteria:

- All differential tests pass under Rosetta locally and on native x86 in CI.
- Per-kernel speed-up over scalar measured on native x86 and recorded.
  Expected shape (to be confirmed by measurement, not assumed): transforms
  and quantization gain most; LZ77 match finding gains least and will
  dominate the end-to-end profile.

---

## Phase 4 — CUDA backend

**Environment:** a Linux GPU VM (AWS / RunPod) or GitHub Actions. CUDA code is
written on the arm64 development machine, where it cannot be compiled, so it
is reviewed statically and kept behind the existing build-time gate until
deployment. The first step on the GPU machine is validating the untested
Phase 0 CUDA build path; until that has happened, no CUDA code counts as
working.

Realistic scope: full GPU DEFLATE at nvCOMP's level is a research-grade
problem. The phase is staged so that each stage is independently useful and
measurable, with an explicit fallback.

Steps:

1. **Device infrastructure.** RAII wrappers for device memory, streams and
   events; error translation into `Status`; a device buffer pool to keep
   `cudaMalloc` out of the hot path.
2. **PCIe transfer path.**
   - Pinned (page-locked) host buffers via `cudaHostAlloc`.
   - Chunked asynchronous pipeline: `cudaMemcpyAsync` on multiple streams so
     host→device copy, kernels, and device→host copy of consecutive chunks
     overlap.
   - Benchmark pageable vs pinned vs pinned + overlapped, and report transfer
     time separately from kernel time.
3. **Transform kernels.** Quantization, forward predictor and byte shuffle are
   embarrassingly parallel. The inverse predictor is a prefix sum, done as a
   per-block scan.
4. **Histogram kernel using shared memory.** Each thread block accumulates
   symbol counts into a `__shared__` table, then merges into the per-data-block
   global histogram. This is the textbook case where shared memory beats
   global atomics, and it is benchmarked both ways to show it.
5. **Huffman encode.** Code tables built on the CPU (tiny, serial). On the
   GPU: per-symbol code-length lookup → prefix sum of bit lengths → parallel
   scatter of codes to their bit offsets.
6. **LZ77 on the GPU.** One CUDA thread block per data block, hash table in
   shared memory. Shared memory is small (tens of KiB per block), which limits
   table size and therefore ratio — this trade-off is measured and reported.
7. **Decode.** Huffman decoding is serial within a block, so parallelism is
   one thread per block; evaluate storing sub-block entry points in the index
   to raise it.
8. **Fallback (decided by measurement):** if GPU LZ77 loses to the CPU on
   ratio or speed, ship a hybrid mode — GPU for quantization, transforms,
   histogram and Huffman; CPU threads for match finding.

Tests:

- Cross-backend matrix: compress on {scalar, AVX2, CUDA} × decompress on
  {scalar, AVX2, CUDA}; all nine combinations must reproduce the input.
- GPU transform and quantization kernels bit-identical to scalar.
- Tests skip visibly (not silently pass) when no device is present.
- Clean under `compute-sanitizer`.

Exit criteria:

- All nine cross-backend combinations pass.
- End-to-end GB/s **including transfers** reported next to kernel-only GB/s.
- Input-size crossover point where the GPU overtakes AVX2 identified and
  documented, including the case where it never does for small inputs.

---

## Phase 5 — Python API, benchmarks and release

Steps:

1. **Python API (pybind11).**
   - `compress(array, *, backend="auto", quantize=None, level=...) -> bytes`
   - `decompress(data, *, backend="auto") -> numpy.ndarray`
   - `analyze_quantization(array, target="fp16" | "bf16") -> QuantizationReport`
   - Zero-copy input through the buffer protocol; contiguity and dtype
     validated with clear errors; the GIL released around native calls.
2. **Packaging.** `pyproject.toml` with scikit-build-core so `pip install .`
   works; pytest suite; type stubs.
3. **Benchmark suite (Google Benchmark).**
   - Per-kernel and end-to-end, compress and decompress, every backend.
   - Datasets: synthetic (smooth field, turbulence-like, sparse, noise), HDR
     images (PFM), and public scientific float datasets (e.g. SDRBench).
   - Reported per run: GB/s, compression ratio, and for lossy modes PSNR and
     max ULP error — always alongside the `memcpy` ceiling.
   - Optional, benchmark-only comparison against zlib/zstd/nvCOMP for context.
     These never link into the library.
4. **CI.** GitHub Actions: Linux x86_64 (GCC + Clang, AVX2 native), macOS
   arm64, sanitizer job, exhaustive-quantization job; CUDA compile-only job in
   an `nvidia/cuda` container (no GPU needed to catch build breaks).
5. **Documentation.** README with a results table and methodology, format
   specification for TCF v1, API reference, and a short design write-up
   covering the decisions in section 2.

Exit criteria:

- `pip install .` then `import turbocore` round-trips a NumPy array.
- One command regenerates every number in the README results table.
- CI green on all jobs.

---

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| CUDA code written without a compiler | Bugs accumulate unseen until deployment | Keep kernels small and mirrored 1:1 on the scalar reference; add a CUDA compile-only CI job (`nvidia/cuda` container, no GPU needed) as early as possible |
| GPU LZ77 underperforms | Headline GPU number is weak | Hybrid fallback (Phase 4, step 8); report honestly either way |
| PCIe transfer dominates for small inputs | GPU slower end to end | Report the crossover size; `Auto` backend picks CPU below it |
| Low mantissa bytes are incompressible | Lossless ratio on noisy data is modest | Expected and stated; the lossy quantization path is the answer for those datasets |
| AVX2 numbers measured under Rosetta | Invalid performance claims | Rosetta is correctness-only; performance from native x86 only |
| Decoder fed malicious input | Memory-safety bugs | Bounds-checked reader, sanitizers, fuzzing from Phase 1 |

## Decisions log

| Question | Decision |
|----------|----------|
| Where does the CUDA backend run? | Dedicated Linux GPU VM (AWS / RunPod) or GitHub Actions, provisioned later. Code is written locally first. |
| LZW? | Dropped. Delta/XOR predictor followed by DEFLATE; RLE is subsumed by LZ77 (section 2.2). |
| Where do AVX2 performance numbers come from? | A native x86 Linux cloud instance. Rosetta 2 is sufficient for local correctness. |
