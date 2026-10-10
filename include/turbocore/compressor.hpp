#ifndef TURBOCORE_COMPRESSOR_HPP
#define TURBOCORE_COMPRESSOR_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "turbocore/backend.hpp"
#include "turbocore/export.hpp"
#include "turbocore/quantization.hpp"

namespace turbocore {

/// Decorrelation applied to the bit patterns of consecutive elements before
/// entropy coding. Both predictors work on the integer bit pattern, never on
/// the float value, so they are exactly reversible for every input including
/// NaN payloads, signed zeros and subnormals.
enum class Predictor : std::uint8_t {
  None = 0,
  Delta = 1,  ///< bits[i] - bits[i-1], modulo 2^(element width)
  Xor = 2,    ///< bits[i] ^ bits[i-1]
};

struct CompressionOptions {
  /// Lossy mode. When set, every value is first rounded to the given 16-bit
  /// format (see quantization.hpp) and the stream stores 16-bit elements;
  /// everything after that step is still lossless. Decompression returns the
  /// rounded values, i.e. exactly quantize_round_trip(data, format).
  /// analyze_quantization() reports what this costs for a given dataset.
  /// Empty (the default) means lossless FP32.
  std::optional<QuantFormat> quantization;

  Predictor predictor = Predictor::Delta;

  /// Regroup the data so that byte k of every element is contiguous. Sign and
  /// exponent bytes are then adjacent to each other, which is what the LZ77
  /// stage needs to find matches.
  bool byte_shuffle = true;

  /// 1 (fastest) to 9 (smallest). Controls how hard LZ77 searches for matches.
  int level = 6;

  /// Elements per independently decodable block (1 to 2^24). Predictor state
  /// and the LZ77 window reset at every block boundary.
  std::size_t block_elements = 65536;

  /// Implementation to run. Has no effect on the output bytes.
  Backend backend = Backend::Auto;

  /// Worker threads; blocks are compressed independently, one per thread.
  /// 0 means one per hardware thread. Has no effect on the output bytes.
  unsigned threads = 0;
};

/// FP32 compressor: [optional quantization ->] predictor -> byte shuffle ->
/// DEFLATE, in independent blocks, each protected by a CRC-32C of the stored
/// elements. Lossless unless CompressionOptions::quantization is set.
class TURBOCORE_EXPORT Compressor {
 public:
  Compressor();

  /// Throws Error(InvalidArgument) if the options are out of range, or
  /// Error(BackendUnavailable) if the requested backend cannot run here.
  explicit Compressor(const CompressionOptions& options);

  [[nodiscard]] const CompressionOptions& options() const noexcept { return options_; }

  /// Returns a self-describing stream. Decompressor::decompress() turns it
  /// back into the exact input bit patterns (lossless mode) or into the
  /// input rounded to the quantization format (lossy mode).
  [[nodiscard]] std::vector<std::uint8_t> compress(std::span<const float> data) const;

 private:
  CompressionOptions options_;
};

}  // namespace turbocore

#endif  // TURBOCORE_COMPRESSOR_HPP
