# TurboCore — Architecture and Implementation Plan

TurboCore is a standalone C++20 library for compressing 32-bit floating-point
arrays and HDR images, with three interchangeable backends (scalar, AVX2,
CUDA), a floating-point error-analysis module for lossy quantization, and
Python bindings. No external compression library is used anywhere in the
library itself.

Status: **Phase 0 complete.** Phases 1–5 are specified below and not started.

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

### 2.2 Decision: DEFLATE is the primary codec, LZW is secondary

The brief says "LZW / DEFLATE". The plan commits to **DEFLATE (LZ77 +
canonical Huffman) as the primary codec** and treats LZW as an optional second
codec behind the same interface.

- LZW's dictionary is adaptive and inherently serial; it cannot be split
  across GPU threads without restarting the dictionary, which destroys ratio.
- DEFLATE separates match finding from entropy coding. Histogramming and
  Huffman bit-packing parallelise well; that is where the GPU work goes.
- Emitting **raw RFC 1951 streams** per block gives a free, independent test
  oracle: Python's built-in `zlib.decompressobj(wbits=-15)` must be able to
  decode every block TurboCore writes. This is a test-only cross-check and
  does not violate the "no external compression library" rule for the library.

### 2.3 Container format (TCF, version 1)

| Section      | Contents |
|--------------|----------|
| File header  | magic `TCOR`, format version, element type (FP32/FP16/BF16), shape (up to 4 dims), predictor id, block size, block count |
| Block index  | per block: byte offset, compressed size, raw size, CRC-32C of the raw bytes |
| Blocks       | independent raw DEFLATE streams, or stored verbatim if incompressible |

Independent blocks give random access, bounded memory, thread-level
parallelism on the CPU and block-level parallelism on the GPU. The block size
(default 256 KiB, tunable) trades ratio against parallelism and is a benchmark
parameter, not a constant.

### 2.4 Backends and dispatch

- Public API is object-oriented: `Compressor`, `Decompressor`, an abstract
  `Codec`, and value types for options and reports.
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
| Linux x86_64, native | AVX2 backend **performance** | not available locally — needs CI or a remote machine |
| Linux x86_64 + NVIDIA GPU | CUDA backend, build and run | not available locally — **must be arranged before Phase 4** |

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

## Phase 1 — Scalar reference codec

The scalar implementation is the specification. Every later backend is tested
against it, so it is written for clarity and correctness first.

Steps:

1. **Core types.** `Span`-based buffer views, `Shape`, `Status`/error type
   (no exceptions across the hot path), `Options`.
2. **Bit I/O.** LSB-first `BitWriter` / `BitReader` with 64-bit accumulators
   and explicit bounds checking on the reader.
3. **CRC-32C.** Table-driven scalar implementation (hardware version comes in
   Phase 3).
4. **Predictors.** 1D delta and XOR on `uint32` bit patterns; 2D left/up
   predictor for images. Forward and inverse, each exactly reversible.
5. **Byte-plane shuffle** and unshuffle for 2- and 4-byte elements.
6. **Huffman.** Histogram → length-limited code lengths (package-merge, max
   15 bits) → canonical codes → table-driven decoder.
7. **LZ77.** Hash-chain match finder, 32 KiB window, lazy matching, with
   compression levels mapped to chain depth.
8. **DEFLATE block encoder/decoder.** Stored, fixed and dynamic blocks per
   RFC 1951. The decoder treats all input as untrusted.
9. **Container.** TCF header, block index, per-block CRC, `Compressor` /
   `Decompressor` classes, multi-threaded over blocks with `std::thread`.
10. **Optional: LZW codec** behind the `Codec` interface, variable code width
    9–16 bits, for a ratio/speed comparison against DEFLATE.
11. **HDR image I/O.** Minimal PFM reader/writer (trivial format, no
    dependency) so real HDR images can be used in tests and demos.

Tests:

- Round trip on: empty input, one element, constants, ramps, white noise,
  all-NaN, mixed NaN payloads, ±0, ±Inf, subnormals, sizes straddling block
  boundaries.
- Every emitted block decodes with Python `zlib` (raw mode) to the same bytes.
- TurboCore's decoder decodes streams produced by Python `zlib`.
- Corrupt-input tests: truncation, bit flips, bad lengths → clean error, no
  crash, no out-of-bounds read under ASan.
- A libFuzzer target for the decoder.

Exit criteria:

- All of the above pass under ASan + UBSan on arm64 and under Rosetta.
- Compression ratio and GB/s recorded for each pipeline stage on the benchmark
  datasets, to establish the baseline the later phases are measured against.

---

## Phase 2 — Quantization and floating-point error analysis

Done before SIMD on purpose: the scalar quantizers defined here are the
bit-exact reference that the AVX2 (F16C) and CUDA versions must reproduce.

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
  against an independent slow reference (integer arithmetic on widened
  values). Registered under a separate CTest label (`exhaustive`) so it runs
  on demand and in CI, not on every local build.
- Targeted cases: exact ties, the 65504/65520 overflow boundary, the FP16
  subnormal boundary, largest finite BF16 rounding, every NaN class.
- Metric tests against hand-computed values and against NumPy
  (`np.float16`) as an independent oracle.

Exit criteria:

- Exhaustive test passes for both formats.
- Target-format ULP error ≤ 0.5 holds for every finite non-overflowing input.
- A written error report (table + ULP histogram) for each benchmark dataset.

---

## Phase 3 — AVX2 backend

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

**Prerequisite:** access to a Linux machine with an NVIDIA GPU. Step 1 is
validating the untested Phase 0 CUDA build path.

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
| No NVIDIA hardware available | Phase 4 cannot be built or run | Arrange access before Phase 3 ends; CUDA compile-only CI job catches build breaks earlier |
| GPU LZ77 underperforms | Headline GPU number is weak | Hybrid fallback (Phase 4, step 8); report honestly either way |
| PCIe transfer dominates for small inputs | GPU slower end to end | Report the crossover size; `Auto` backend picks CPU below it |
| Low mantissa bytes are incompressible | Lossless ratio on noisy data is modest | Expected and stated; the lossy quantization path is the answer for those datasets |
| AVX2 numbers measured under Rosetta | Invalid performance claims | Rosetta is correctness-only; performance from native x86 only |
| Decoder fed malicious input | Memory-safety bugs | Bounds-checked reader, sanitizers, fuzzing from Phase 1 |

## Open questions

1. Which machine will run the CUDA backend (university cluster, cloud
   instance, personal workstation)? This gates Phase 4.
2. Is LZW required as a deliverable, or is DEFLATE-primary with LZW optional
   (section 2.2) acceptable?
3. Is a native x86 Linux machine available for AVX2 benchmarks, or will CI
   runners be the source of those numbers?
