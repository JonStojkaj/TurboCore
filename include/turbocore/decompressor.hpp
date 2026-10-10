#ifndef TURBOCORE_DECOMPRESSOR_HPP
#define TURBOCORE_DECOMPRESSOR_HPP

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "turbocore/compressor.hpp"
#include "turbocore/export.hpp"

namespace turbocore {

/// What a stream's header says about its contents.
struct StreamInfo {
  std::uint64_t element_count;
  std::uint32_t block_count;
  std::uint32_t block_elements;
  Predictor predictor;
  bool byte_shuffle;
  /// The 16-bit format the data was rounded to; empty for a lossless stream.
  std::optional<QuantFormat> quantization;
};

struct DecompressionOptions {
  /// Implementation to run. Any backend decodes streams written by any other.
  Backend backend = Backend::Auto;

  /// Worker threads; blocks are decoded independently, one per thread.
  /// 0 means one per hardware thread.
  unsigned threads = 0;
};

/// Decoder for streams produced by Compressor. All input is treated as
/// untrusted: malformed data raises Error and never reads or writes out of
/// bounds.
class TURBOCORE_EXPORT Decompressor {
 public:
  Decompressor();

  /// Throws Error(BackendUnavailable) if the requested backend cannot run here.
  explicit Decompressor(const DecompressionOptions& options);

  [[nodiscard]] const DecompressionOptions& options() const noexcept { return options_; }

  /// Validates and returns the header without decoding any block.
  /// Throws Error(UnsupportedFormat | CorruptStream).
  [[nodiscard]] static StreamInfo inspect(std::span<const std::uint8_t> stream);

  /// Always returns FP32: a quantized stream is expanded back to float.
  /// Throws Error(UnsupportedFormat | CorruptStream | ChecksumMismatch). If
  /// several blocks are damaged, the error reported is that of the first one.
  [[nodiscard]] std::vector<float> decompress(std::span<const std::uint8_t> stream) const;

 private:
  DecompressionOptions options_;
};

}  // namespace turbocore

#endif  // TURBOCORE_DECOMPRESSOR_HPP
