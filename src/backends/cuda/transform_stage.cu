// CUDA implementation of the transform-stage contract in transform_stage.hpp.
//
// STATUS: written on a machine without the CUDA Toolkit. This file has never
// been compiled by nvcc or run on a GPU, and nothing on the development
// machine builds or tests it. The CudaParity tests in tests/test_backend.cpp
// are its check on the first GPU machine: streams must be byte-identical to
// the CPU backends, and decoded values bit-identical.
//
// Design
// ------
// All arithmetic is on integer bit patterns, including the float conversions,
// so results cannot depend on the device's floating-point mode and are
// bit-identical to the scalar reference by construction.
//
// Forward (one pass, no dependencies between elements):
//   forward_kernel     element i -> stored word (bits or quantized), minus /
//                      xor its predecessor's, written as shuffled bytes
//   histogram_kernel   one thread block per (data block, byte plane); counts
//                      accumulate in shared memory and are written out once
//
// Inverse:
//   unshuffle_kernel   bytes -> residual words
//   prefix scan        the inverse predictor is a prefix sum / prefix xor
//                      over each data block, done in three kernels:
//     scan_tiles_kernel    each thread block loads one tile of kScanTile
//                          residuals into shared memory, scans it there
//                          (Hillis-Steele, log2(kScanTile) steps), writes the
//                          tile-local result and the tile's total
//     scan_totals_kernel   one thread block per data block scans the tile
//                          totals in shared memory -> each tile's offset
//     apply_offsets_kernel adds each tile's offset to its elements
//   finalize_kernel    stored word -> FP32 bits (dequantize if 16-bit)
//
// Transfers go through page-locked host buffers with asynchronous copies on
// one stream; the array is processed in chunks of whole data blocks so device
// memory use is bounded. Chunks are not yet overlapped across streams: that
// is the next transfer optimisation and needs measurement on real hardware.

#include "backends/cuda/transform_stage.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <mutex>

namespace turbocore {
namespace detail {
namespace cuda {

namespace {

// ---------------------------------------------------------------------------
// Device functions: bit-level float conversions.
// Line-for-line ports of src/quantization.cpp; see there for the reasoning.
// ---------------------------------------------------------------------------

__device__ __forceinline__ bool rounds_up(std::uint32_t kept, std::uint32_t remainder, int dropped_bits) {
  const std::uint32_t halfway = 1u << (dropped_bits - 1);
  return remainder > halfway || (remainder == halfway && (kept & 1u) != 0u);
}

__device__ __forceinline__ std::uint32_t float32_to_float16_bits(std::uint32_t bits) {
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  const std::uint32_t exponent_field = (bits >> 23) & 0xFFu;
  const std::uint32_t mantissa = bits & 0x007FFFFFu;

  if (exponent_field == 0xFFu) {
    if (mantissa == 0u) {
      return sign | 0x7C00u;
    }
    return sign | 0x7C00u | 0x0200u | (mantissa >> 13);
  }

  const int half_exponent = static_cast<int>(exponent_field) - 127 + 15;
  if (half_exponent >= 31) {
    return sign | 0x7C00u;
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return sign;
    }
    const std::uint32_t significand = mantissa | 0x00800000u;
    const int dropped = 14 - half_exponent;  // 14 to 24
    std::uint32_t half = significand >> dropped;
    const std::uint32_t remainder = significand & ((1u << dropped) - 1u);
    if (rounds_up(half, remainder, dropped)) {
      ++half;
    }
    return sign | half;
  }

  std::uint32_t half = (static_cast<std::uint32_t>(half_exponent) << 10) | (mantissa >> 13);
  if (rounds_up(half, mantissa & 0x1FFFu, 13)) {
    ++half;
  }
  return sign | half;
}

__device__ __forceinline__ std::uint32_t float16_bits_to_float32(std::uint32_t half) {
  const std::uint32_t sign = (half & 0x8000u) << 16;
  const std::uint32_t exponent_field = (half >> 10) & 0x1Fu;
  std::uint32_t mantissa = half & 0x03FFu;

  if (exponent_field == 31u) {
    const std::uint32_t quiet = mantissa != 0u ? 0x00400000u : 0u;
    return sign | 0x7F800000u | quiet | (mantissa << 13);
  }
  if (exponent_field == 0u) {
    if (mantissa == 0u) {
      return sign;
    }
    // Normalise the subnormal: bit width of a non-zero 10-bit value is 32 - clz.
    const int shift = 11 - (32 - __clz(static_cast<int>(mantissa)));
    mantissa = (mantissa << shift) & 0x03FFu;
    const std::uint32_t exponent = static_cast<std::uint32_t>(127 - 15 + 1 - shift);
    return sign | (exponent << 23) | (mantissa << 13);
  }
  return sign | ((exponent_field + 112u) << 23) | (mantissa << 13);
}

__device__ __forceinline__ std::uint32_t float32_to_bfloat16_bits(std::uint32_t bits) {
  std::uint32_t top = bits >> 16;
  if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
    return top | 0x0040u;
  }
  if (rounds_up(top, bits & 0xFFFFu, 16)) {
    ++top;
  }
  return top;
}

/// The element as stored in the stream: FP32 bits, or the quantized 16 bits.
__device__ __forceinline__ std::uint32_t stored_word(std::uint32_t bits, int element) {
  if (element == 1) {
    return float32_to_float16_bits(bits);
  }
  if (element == 2) {
    return float32_to_bfloat16_bits(bits);
  }
  return bits;
}

__device__ __forceinline__ std::uint32_t combine(std::uint32_t a, std::uint32_t b, int predictor) {
  return predictor == 1 ? a + b : a ^ b;
}

/// Number of elements in the data block starting at `block_first`.
__device__ __forceinline__ std::size_t block_length(std::size_t block_first, std::size_t block_elements,
                                                    std::size_t count) {
  const std::size_t remaining = count - block_first;
  return remaining < block_elements ? remaining : block_elements;
}

// ---------------------------------------------------------------------------
// Forward kernels
// ---------------------------------------------------------------------------

constexpr int kElementwiseThreads = 256;

/// One thread per element.
__global__ void forward_kernel(const std::uint32_t* input, std::size_t count, std::size_t block_elements, int element,
                               int predictor, int byte_shuffle, std::uint16_t* words, std::uint8_t* transformed) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const std::size_t within = i % block_elements;
  const std::size_t block_first = i - within;
  const std::size_t n = block_length(block_first, block_elements, count);
  const int width = element == 0 ? 4 : 2;

  const std::uint32_t word = stored_word(input[i], element);
  std::uint32_t residual = word;
  if (predictor != 0 && within != 0) {
    // Recomputing the predecessor's stored word keeps this kernel free of
    // inter-thread dependencies; the conversion is a few integer operations.
    const std::uint32_t previous = stored_word(input[i - 1], element);
    residual = predictor == 1 ? word - previous : word ^ previous;
  }
  if (width == 2) {
    words[i] = static_cast<std::uint16_t>(word);
  }

  std::uint8_t* block = transformed + block_first * static_cast<std::size_t>(width);
  for (int k = 0; k < width; ++k) {
    const std::uint8_t byte = static_cast<std::uint8_t>((residual >> (8 * k)) & 0xFFu);
    if (byte_shuffle != 0) {
      block[static_cast<std::size_t>(k) * n + within] = byte;
    } else {
      block[within * static_cast<std::size_t>(width) + static_cast<std::size_t>(k)] = byte;
    }
  }
}

constexpr int kHistogramThreads = 256;

/// One thread block per (data block, byte plane). Requires byte_shuffle
/// layout and blockDim.x == 256, so that thread t owns histogram bin t when
/// the result is written out.
///
/// The counts are accumulated in shared memory: 256 counters that every
/// thread of the block updates are exactly what shared memory is for, and
/// keeping the atomics on-chip avoids global-memory atomic traffic for the
/// 64 Ki increments of a default-sized plane.
__global__ void histogram_kernel(const std::uint8_t* transformed, std::size_t count, std::size_t block_elements,
                                 int width, std::uint32_t* histograms) {
  __shared__ unsigned int bins[256];
  bins[threadIdx.x] = 0u;
  __syncthreads();

  const std::size_t segment = blockIdx.x;  // data block * width + plane
  const std::size_t data_block = segment / static_cast<std::size_t>(width);
  const std::size_t plane = segment % static_cast<std::size_t>(width);
  const std::size_t block_first = data_block * block_elements;
  const std::size_t n = block_length(block_first, block_elements, count);
  const std::uint8_t* bytes = transformed + block_first * static_cast<std::size_t>(width) + plane * n;

  for (std::size_t j = threadIdx.x; j < n; j += blockDim.x) {
    atomicAdd(&bins[bytes[j]], 1u);
  }
  __syncthreads();

  histograms[segment * 256 + threadIdx.x] = bins[threadIdx.x];
}

// ---------------------------------------------------------------------------
// Inverse kernels
// ---------------------------------------------------------------------------

/// One thread per element: reassemble the residual word from its bytes.
__global__ void unshuffle_kernel(const std::uint8_t* transformed, std::size_t count, std::size_t block_elements,
                                 int width, int byte_shuffle, std::uint32_t* work) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const std::size_t within = i % block_elements;
  const std::size_t block_first = i - within;
  const std::size_t n = block_length(block_first, block_elements, count);
  const std::uint8_t* block = transformed + block_first * static_cast<std::size_t>(width);

  std::uint32_t value = 0u;
  for (int k = 0; k < width; ++k) {
    const std::uint8_t byte = byte_shuffle != 0 ? block[static_cast<std::size_t>(k) * n + within]
                                                : block[within * static_cast<std::size_t>(width) +
                                                        static_cast<std::size_t>(k)];
    value |= static_cast<std::uint32_t>(byte) << (8 * k);
  }
  work[i] = value;
}

constexpr int kScanTileThreads = static_cast<int>(kScanTile);
constexpr int kTotalsThreads = static_cast<int>(kMaxTilesPerBlock);

/// Inclusive scan of `values[threadIdx.x]` over the thread block, in place in
/// shared memory. `buffers` is two arrays of blockDim.x entries; on entry the
/// first holds the input. Returns which of the two holds the result.
///
/// Hillis-Steele: in step s every element combines with the one 2^s to its
/// left, reading from one buffer and writing the other, so there are no
/// read/write races within a step. log2(blockDim.x) steps, all threads busy
/// in each, which suits the short fixed-size scans here better than the
/// work-efficient up/down-sweep variant.
__device__ __forceinline__ int scan_in_shared(std::uint32_t* first, std::uint32_t* second, int predictor) {
  std::uint32_t* source = first;
  std::uint32_t* destination = second;
  int result_in_second = 0;
  for (unsigned int offset = 1; offset < blockDim.x; offset <<= 1) {
    std::uint32_t value = source[threadIdx.x];
    if (threadIdx.x >= offset) {
      value = combine(value, source[threadIdx.x - offset], predictor);
    }
    destination[threadIdx.x] = value;
    __syncthreads();
    std::uint32_t* swap = source;
    source = destination;
    destination = swap;
    result_in_second ^= 1;
  }
  return result_in_second;
}

/// One thread block per tile of kScanTile elements; blockDim.x == kScanTile.
/// Tiles are numbered per data block (tiles_per_block each), so no tile ever
/// spans two data blocks. Positions past the end of a data block hold the
/// identity (0 for both + and ^) and so do not disturb the tile total.
__global__ void scan_tiles_kernel(std::uint32_t* work, std::size_t count, std::size_t block_elements,
                                  std::size_t tiles_per_block, int predictor, std::uint32_t* tile_totals) {
  __shared__ std::uint32_t buffer_a[kScanTileThreads];
  __shared__ std::uint32_t buffer_b[kScanTileThreads];

  const std::size_t tile = blockIdx.x;
  const std::size_t data_block = tile / tiles_per_block;
  const std::size_t tile_in_block = tile % tiles_per_block;
  const std::size_t block_first = data_block * block_elements;
  const std::size_t n = block_length(block_first, block_elements, count);
  const std::size_t within = tile_in_block * kScanTile + threadIdx.x;
  const bool valid = within < n;

  buffer_a[threadIdx.x] = valid ? work[block_first + within] : 0u;
  __syncthreads();

  const int in_b = scan_in_shared(buffer_a, buffer_b, predictor);
  const std::uint32_t scanned = in_b != 0 ? buffer_b[threadIdx.x] : buffer_a[threadIdx.x];

  if (valid) {
    work[block_first + within] = scanned;
  }
  if (threadIdx.x == blockDim.x - 1) {
    tile_totals[tile] = scanned;
  }
}

/// One thread block per data block; blockDim.x == kMaxTilesPerBlock. Turns
/// the tile totals into each tile's starting offset (an exclusive scan).
__global__ void scan_totals_kernel(const std::uint32_t* tile_totals, std::size_t tiles_per_block, int predictor,
                                   std::uint32_t* tile_offsets) {
  __shared__ std::uint32_t buffer_a[kTotalsThreads];
  __shared__ std::uint32_t buffer_b[kTotalsThreads];

  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * tiles_per_block;
  const bool valid = threadIdx.x < tiles_per_block;

  buffer_a[threadIdx.x] = valid ? tile_totals[base + threadIdx.x] : 0u;
  __syncthreads();

  const int in_b = scan_in_shared(buffer_a, buffer_b, predictor);
  // scan_in_shared ends with a __syncthreads(), so a neighbour's final value
  // is safe to read here.
  const std::uint32_t* inclusive = in_b != 0 ? buffer_b : buffer_a;
  if (valid) {
    tile_offsets[base + threadIdx.x] = threadIdx.x == 0 ? 0u : inclusive[threadIdx.x - 1];
  }
}

/// One thread per element: add the offset of the element's tile.
__global__ void apply_offsets_kernel(std::uint32_t* work, std::size_t count, std::size_t block_elements,
                                     std::size_t tiles_per_block, int predictor, const std::uint32_t* tile_offsets) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const std::size_t within = i % block_elements;
  const std::size_t data_block = i / block_elements;
  const std::size_t tile = data_block * tiles_per_block + within / kScanTile;
  work[i] = combine(work[i], tile_offsets[tile], predictor);
}

/// One thread per element: stored word -> FP32 bits (and the stored 16-bit
/// word itself, for the host's CRC check).
__global__ void finalize_kernel(const std::uint32_t* work, std::size_t count, int element, std::uint16_t* words,
                                std::uint32_t* output) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  if (element == 0) {
    output[i] = work[i];
    return;
  }
  // 16-bit arithmetic was carried out modulo 2^32; the low half is the
  // result modulo 2^16.
  const std::uint32_t word = work[i] & 0xFFFFu;
  words[i] = static_cast<std::uint16_t>(word);
  output[i] = element == 1 ? float16_bits_to_float32(word) : word << 16;
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

Status ok_status() noexcept {
  Status status;
  status.ok = true;
  status.message[0] = '\0';
  return status;
}

Status error_status(const char* what, cudaError_t error) noexcept {
  Status status;
  status.ok = false;
  std::snprintf(status.message, sizeof(status.message), "CUDA: %s: %s", what, cudaGetErrorString(error));
  return status;
}

Status argument_status(const char* what) noexcept {
  Status status;
  status.ok = false;
  std::snprintf(status.message, sizeof(status.message), "CUDA backend: %s", what);
  return status;
}

#define TURBOCORE_CUDA_TRY(call)                    \
  do {                                              \
    const cudaError_t turbocore_error_ = (call);    \
    if (turbocore_error_ != cudaSuccess) {          \
      return error_status(#call, turbocore_error_); \
    }                                               \
  } while (false)

/// Device memory that only grows, so repeated calls reuse the allocation.
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() { release(); }

  cudaError_t reserve(std::size_t bytes) noexcept {
    if (bytes <= capacity_) {
      return cudaSuccess;
    }
    release();
    const cudaError_t error = cudaMalloc(&pointer_, bytes);
    if (error == cudaSuccess) {
      capacity_ = bytes;
    } else {
      pointer_ = nullptr;
    }
    return error;
  }

  template <typename T>
  T* as() const noexcept {
    return static_cast<T*>(pointer_);
  }

 private:
  void release() noexcept {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
      pointer_ = nullptr;
      capacity_ = 0;
    }
  }

  void* pointer_ = nullptr;
  std::size_t capacity_ = 0;
};

/// Page-locked host memory. Copies between it and the device go by DMA
/// straight from these pages; from ordinary pageable memory the driver first
/// has to copy into a staging buffer of its own, synchronously.
class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  ~PinnedBuffer() { release(); }

  cudaError_t reserve(std::size_t bytes) noexcept {
    if (bytes <= capacity_) {
      return cudaSuccess;
    }
    release();
    const cudaError_t error = cudaHostAlloc(&pointer_, bytes, cudaHostAllocDefault);
    if (error == cudaSuccess) {
      capacity_ = bytes;
    } else {
      pointer_ = nullptr;
    }
    return error;
  }

  template <typename T>
  T* as() const noexcept {
    return static_cast<T*>(pointer_);
  }

 private:
  void release() noexcept {
    if (pointer_ != nullptr) {
      (void)cudaFreeHost(pointer_);
      pointer_ = nullptr;
      capacity_ = 0;
    }
  }

  void* pointer_ = nullptr;
  std::size_t capacity_ = 0;
};

/// Everything that lives across calls. One device, one stream; calls are
/// serialised by `mutex`.
struct Context {
  std::mutex mutex;
  cudaStream_t stream = nullptr;

  DeviceBuffer device_floats;       // FP32 in (forward) / out (inverse)
  DeviceBuffer device_transformed;  // shuffled bytes
  DeviceBuffer device_words;        // 16-bit stored elements
  DeviceBuffer device_work;         // 32-bit residuals / scan workspace
  DeviceBuffer device_histograms;
  DeviceBuffer device_tile_totals;
  DeviceBuffer device_tile_offsets;

  PinnedBuffer host_floats;
  PinnedBuffer host_transformed;
  PinnedBuffer host_words;
  PinnedBuffer host_histograms;

  cudaError_t ensure_stream() noexcept {
    return stream != nullptr ? cudaSuccess : cudaStreamCreate(&stream);
  }
};

Context& context() noexcept {
  // Intentionally never destroyed: tearing down CUDA objects from a static
  // destructor races with the runtime's own shutdown.
  static Context* instance = new Context();
  return *instance;
}

unsigned int grid_for(std::size_t items, int threads_per_block) noexcept {
  return static_cast<unsigned int>((items + static_cast<std::size_t>(threads_per_block) - 1) /
                                   static_cast<std::size_t>(threads_per_block));
}

/// Elements per device round trip: whole data blocks, about 64 MiB of FP32.
std::size_t chunk_elements_for(std::size_t block_elements) noexcept {
  const std::size_t target = std::size_t{1} << 24;
  const std::size_t blocks = target / block_elements;
  return (blocks == 0 ? 1 : blocks) * block_elements;
}

Status validate(const StageConfig& config) noexcept {
  if (config.block_elements < 1 || config.block_elements > kMaxBlockElements) {
    return argument_status("block_elements out of range");
  }
  return ok_status();
}

Status forward_chunk(Context& ctx, const StageConfig& config, const float* input, std::size_t count,
                     std::uint16_t* words, std::uint8_t* transformed, std::uint32_t* histograms) noexcept {
  const int element = static_cast<int>(config.element);
  const int predictor = static_cast<int>(config.predictor);
  const std::size_t width = element_bytes(config.element);
  const std::size_t float_bytes = count * sizeof(float);
  const std::size_t transformed_bytes = count * width;
  const std::size_t word_bytes = width == 2 ? count * sizeof(std::uint16_t) : 0;
  const std::size_t data_blocks = (count + config.block_elements - 1) / config.block_elements;
  const std::size_t histogram_bytes = histograms != nullptr ? data_blocks * width * 256 * sizeof(std::uint32_t) : 0;

  TURBOCORE_CUDA_TRY(ctx.device_floats.reserve(float_bytes));
  TURBOCORE_CUDA_TRY(ctx.device_transformed.reserve(transformed_bytes));
  TURBOCORE_CUDA_TRY(ctx.host_floats.reserve(float_bytes));
  TURBOCORE_CUDA_TRY(ctx.host_transformed.reserve(transformed_bytes));
  if (word_bytes != 0) {
    TURBOCORE_CUDA_TRY(ctx.device_words.reserve(word_bytes));
    TURBOCORE_CUDA_TRY(ctx.host_words.reserve(word_bytes));
  }
  if (histogram_bytes != 0) {
    TURBOCORE_CUDA_TRY(ctx.device_histograms.reserve(histogram_bytes));
    TURBOCORE_CUDA_TRY(ctx.host_histograms.reserve(histogram_bytes));
  }

  std::memcpy(ctx.host_floats.as<float>(), input, float_bytes);
  TURBOCORE_CUDA_TRY(cudaMemcpyAsync(ctx.device_floats.as<void>(), ctx.host_floats.as<void>(), float_bytes,
                                     cudaMemcpyHostToDevice, ctx.stream));

  const unsigned int element_grid = grid_for(count, kElementwiseThreads);
  forward_kernel<<<element_grid, kElementwiseThreads, 0, ctx.stream>>>(
      ctx.device_floats.as<std::uint32_t>(), count, config.block_elements, element, predictor,
      config.byte_shuffle ? 1 : 0, ctx.device_words.as<std::uint16_t>(), ctx.device_transformed.as<std::uint8_t>());
  TURBOCORE_CUDA_TRY(cudaGetLastError());

  if (histogram_bytes != 0) {
    const unsigned int segment_grid = static_cast<unsigned int>(data_blocks * width);
    histogram_kernel<<<segment_grid, kHistogramThreads, 0, ctx.stream>>>(
        ctx.device_transformed.as<std::uint8_t>(), count, config.block_elements, static_cast<int>(width),
        ctx.device_histograms.as<std::uint32_t>());
    TURBOCORE_CUDA_TRY(cudaGetLastError());
  }

  TURBOCORE_CUDA_TRY(cudaMemcpyAsync(ctx.host_transformed.as<void>(), ctx.device_transformed.as<void>(),
                                     transformed_bytes, cudaMemcpyDeviceToHost, ctx.stream));
  if (word_bytes != 0) {
    TURBOCORE_CUDA_TRY(cudaMemcpyAsync(ctx.host_words.as<void>(), ctx.device_words.as<void>(), word_bytes,
                                       cudaMemcpyDeviceToHost, ctx.stream));
  }
  if (histogram_bytes != 0) {
    TURBOCORE_CUDA_TRY(cudaMemcpyAsync(ctx.host_histograms.as<void>(), ctx.device_histograms.as<void>(),
                                       histogram_bytes, cudaMemcpyDeviceToHost, ctx.stream));
  }
  TURBOCORE_CUDA_TRY(cudaStreamSynchronize(ctx.stream));

  std::memcpy(transformed, ctx.host_transformed.as<std::uint8_t>(), transformed_bytes);
  if (word_bytes != 0) {
    std::memcpy(words, ctx.host_words.as<std::uint16_t>(), word_bytes);
  }
  if (histogram_bytes != 0) {
    std::memcpy(histograms, ctx.host_histograms.as<std::uint32_t>(), histogram_bytes);
  }
  return ok_status();
}

Status inverse_chunk(Context& ctx, const StageConfig& config, const std::uint8_t* transformed, std::size_t count,
                     std::uint16_t* words, float* output) noexcept {
  const int element = static_cast<int>(config.element);
  const int predictor = static_cast<int>(config.predictor);
  const std::size_t width = element_bytes(config.element);
  const std::size_t float_bytes = count * sizeof(float);
  const std::size_t transformed_bytes = count * width;
  const std::size_t word_bytes = width == 2 ? count * sizeof(std::uint16_t) : 0;
  const std::size_t data_blocks = (count + config.block_elements - 1) / config.block_elements;
  const std::size_t tiles_per_block = (config.block_elements + kScanTile - 1) / kScanTile;
  const std::size_t tiles = data_blocks * tiles_per_block;

  TURBOCORE_CUDA_TRY(ctx.device_floats.reserve(float_bytes));
  TURBOCORE_CUDA_TRY(ctx.device_transformed.reserve(transformed_bytes));
  TURBOCORE_CUDA_TRY(ctx.device_work.reserve(count * sizeof(std::uint32_t)));
  TURBOCORE_CUDA_TRY(ctx.host_floats.reserve(float_bytes));
  TURBOCORE_CUDA_TRY(ctx.host_transformed.reserve(transformed_bytes));
  if (word_bytes != 0) {
    TURBOCORE_CUDA_TRY(ctx.device_words.reserve(word_bytes));
    TURBOCORE_CUDA_TRY(ctx.host_words.reserve(word_bytes));
  }
  if (predictor != 0) {
    TURBOCORE_CUDA_TRY(ctx.device_tile_totals.reserve(tiles * sizeof(std::uint32_t)));
    TURBOCORE_CUDA_TRY(ctx.device_tile_offsets.reserve(tiles * sizeof(std::uint32_t)));
  }

  std::memcpy(ctx.host_transformed.as<std::uint8_t>(), transformed, transformed_bytes);
  TURBOCORE_CUDA_TRY(cudaMemcpyAsync(ctx.device_transformed.as<void>(), ctx.host_transformed.as<void>(),
                                     transformed_bytes, cudaMemcpyHostToDevice, ctx.stream));

  const unsigned int element_grid = grid_for(count, kElementwiseThreads);
  unshuffle_kernel<<<element_grid, kElementwiseThreads, 0, ctx.stream>>>(
      ctx.device_transformed.as<std::uint8_t>(), count, config.block_elements, static_cast<int>(width),
      config.byte_shuffle ? 1 : 0, ctx.device_work.as<std::uint32_t>());
  TURBOCORE_CUDA_TRY(cudaGetLastError());

  if (predictor != 0) {
    const unsigned int tile_grid = static_cast<unsigned int>(tiles);
    const unsigned int block_grid = static_cast<unsigned int>(data_blocks);
    scan_tiles_kernel<<<tile_grid, kScanTileThreads, 0, ctx.stream>>>(
        ctx.device_work.as<std::uint32_t>(), count, config.block_elements, tiles_per_block, predictor,
        ctx.device_tile_totals.as<std::uint32_t>());
    TURBOCORE_CUDA_TRY(cudaGetLastError());

    scan_totals_kernel<<<block_grid, kTotalsThreads, 0, ctx.stream>>>(
        ctx.device_tile_totals.as<std::uint32_t>(), tiles_per_block, predictor,
        ctx.device_tile_offsets.as<std::uint32_t>());
    TURBOCORE_CUDA_TRY(cudaGetLastError());

    apply_offsets_kernel<<<element_grid, kElementwiseThreads, 0, ctx.stream>>>(
        ctx.device_work.as<std::uint32_t>(), count, config.block_elements, tiles_per_block, predictor,
        ctx.device_tile_offsets.as<std::uint32_t>());
    TURBOCORE_CUDA_TRY(cudaGetLastError());
  }

  finalize_kernel<<<element_grid, kElementwiseThreads, 0, ctx.stream>>>(
      ctx.device_work.as<std::uint32_t>(), count, element, ctx.device_words.as<std::uint16_t>(),
      ctx.device_floats.as<std::uint32_t>());
  TURBOCORE_CUDA_TRY(cudaGetLastError());

  TURBOCORE_CUDA_TRY(cudaMemcpyAsync(ctx.host_floats.as<void>(), ctx.device_floats.as<void>(), float_bytes,
                                     cudaMemcpyDeviceToHost, ctx.stream));
  if (word_bytes != 0) {
    TURBOCORE_CUDA_TRY(cudaMemcpyAsync(ctx.host_words.as<void>(), ctx.device_words.as<void>(), word_bytes,
                                       cudaMemcpyDeviceToHost, ctx.stream));
  }
  TURBOCORE_CUDA_TRY(cudaStreamSynchronize(ctx.stream));

  std::memcpy(output, ctx.host_floats.as<float>(), float_bytes);
  if (word_bytes != 0) {
    std::memcpy(words, ctx.host_words.as<std::uint16_t>(), word_bytes);
  }
  return ok_status();
}

}  // namespace

Status forward(const StageConfig& config, const float* input, std::size_t count, std::uint16_t* words,
               std::uint8_t* transformed, std::uint32_t* histograms) noexcept {
  const Status valid = validate(config);
  if (!valid.ok) {
    return valid;
  }
  if (histograms != nullptr && !config.byte_shuffle) {
    return argument_status("histograms require byte_shuffle");
  }
  if (count == 0) {
    return ok_status();
  }

  Context& ctx = context();
  const std::lock_guard<std::mutex> lock(ctx.mutex);
  TURBOCORE_CUDA_TRY(ctx.ensure_stream());

  const std::size_t width = element_bytes(config.element);
  const std::size_t chunk = chunk_elements_for(config.block_elements);
  for (std::size_t first = 0; first < count; first += chunk) {
    const std::size_t n = count - first < chunk ? count - first : chunk;
    const std::size_t first_block = first / config.block_elements;
    const Status status =
        forward_chunk(ctx, config, input + first, n, words != nullptr ? words + first : nullptr,
                      transformed + first * width,
                      histograms != nullptr ? histograms + first_block * width * 256 : nullptr);
    if (!status.ok) {
      return status;
    }
  }
  return ok_status();
}

Status inverse(const StageConfig& config, const std::uint8_t* transformed, std::size_t count, std::uint16_t* words,
               float* output) noexcept {
  const Status valid = validate(config);
  if (!valid.ok) {
    return valid;
  }
  if (count == 0) {
    return ok_status();
  }

  Context& ctx = context();
  const std::lock_guard<std::mutex> lock(ctx.mutex);
  TURBOCORE_CUDA_TRY(ctx.ensure_stream());

  const std::size_t width = element_bytes(config.element);
  const std::size_t chunk = chunk_elements_for(config.block_elements);
  for (std::size_t first = 0; first < count; first += chunk) {
    const std::size_t n = count - first < chunk ? count - first : chunk;
    const Status status = inverse_chunk(ctx, config, transformed + first * width, n,
                                        words != nullptr ? words + first : nullptr, output + first);
    if (!status.ok) {
      return status;
    }
  }
  return ok_status();
}

int device_count() noexcept {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    return 0;
  }
  return count;
}

}  // namespace cuda
}  // namespace detail
}  // namespace turbocore
