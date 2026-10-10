#ifndef TURBOCORE_BACKENDS_CUDA_TRANSFORM_STAGE_HPP
#define TURBOCORE_BACKENDS_CUDA_TRANSFORM_STAGE_HPP

#include <cstddef>
#include <cstdint>

// The GPU transform stage: everything in the pipeline that is data-parallel,
// applied to a whole array in one device round trip.
//
//   forward:  FP32 -> [quantize] -> predictor -> byte shuffle (+ byte histograms)
//   inverse:  byte unshuffle -> inverse predictor -> [dequantize] -> FP32
//
// The entropy stage (DEFLATE) stays on CPU threads and consumes / produces
// the `transformed` buffer defined here.
//
// This header is the complete contract between the host code and the device
// implementation in transform_stage.cu. Every buffer it produces must be bit
// for bit what the CPU backends compute for the same stage, so that streams
// are byte-identical across backends.
//
// It is included by .cu files, so it must stay valid C++17 and free of
// standard-library templates.

namespace turbocore::detail::cuda {

/// Largest block the device implementation handles: its two-level prefix
/// scan covers kScanTile * kMaxTilesPerBlock elements per block.
constexpr std::size_t kScanTile = 512;
constexpr std::size_t kMaxTilesPerBlock = 1024;
constexpr std::size_t kMaxBlockElements = kScanTile * kMaxTilesPerBlock;

/// Values match detail::ElementType in the container format.
enum class Element : int { Float32 = 0, Float16 = 1, BFloat16 = 2 };
/// Values match turbocore::Predictor.
enum class PredictorOp : int { None = 0, Delta = 1, Xor = 2 };

constexpr std::size_t element_bytes(Element element) { return element == Element::Float32 ? 4 : 2; }

struct StageConfig {
  Element element;
  PredictorOp predictor;
  bool byte_shuffle;
  std::size_t block_elements;  // 1 to kMaxBlockElements
};

struct Status {
  bool ok;
  char message[224];  // empty when ok
};

// Layout of `transformed`, for `count` elements of width W = element_bytes():
// block b (elements [b * block_elements, ...), n_b of them) occupies bytes
// [b * block_elements * W, b * block_elements * W + n_b * W). Inside a block:
//   byte_shuffle:  byte k of element j is at  k * n_b + j
//   otherwise:     byte k of element j is at  j * W + k        (little-endian)
// The predictor restarts at every block boundary.

/// Forward transform of `count` elements.
///   input        count FP32 values
///   words        16-bit elements only: receives the count quantized values
///                (the stored elements the block CRCs are computed over).
///                Ignored for Float32, where the stored elements are `input`.
///   transformed  receives count * W bytes in the layout above
///   histograms   optional (may be null; requires byte_shuffle): receives,
///                for block b and plane k, 256 byte counts at
///                [(b * W + k) * 256, ...)
Status forward(const StageConfig& config, const float* input, std::size_t count, std::uint16_t* words,
               std::uint8_t* transformed, std::uint32_t* histograms) noexcept;

/// Inverse transform of `count` elements.
///   transformed  count * W bytes in the layout above
///   words        16-bit elements only: receives the count stored elements
///                after the inverse predictor. Ignored for Float32, where the
///                stored elements are the bits of `output`.
///   output       receives count FP32 values
Status inverse(const StageConfig& config, const std::uint8_t* transformed, std::size_t count, std::uint16_t* words,
               float* output) noexcept;

/// Number of usable devices; 0 if the runtime cannot be initialised.
int device_count() noexcept;

}  // namespace turbocore::detail::cuda

#endif  // TURBOCORE_BACKENDS_CUDA_TRANSFORM_STAGE_HPP
