#ifndef TURBOCORE_COMPRESSOR_HPP
#define TURBOCORE_COMPRESSOR_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "turbocore/export.hpp"

namespace turbocore {

/// Decorrelation applied to the 32-bit patterns of consecutive floats before
/// entropy coding. Both predictors work on the integer bit pattern, never on
/// the float value, so they are exactly reversible for every input including
/// NaN payloads, signed zeros and subnormals.
enum class Predictor : std::uint8_t {
  None = 0,
  Delta = 1,  ///< bits[i] - bits[i-1], modulo 2^32
  Xor = 2,    ///< bits[i] ^ bits[i-1]
};

struct CompressionOptions {
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
};

/// Lossless FP32 compressor: predictor -> byte shuffle -> DEFLATE, in
/// independent blocks, each protected by a CRC-32C of the original data.
class TURBOCORE_EXPORT Compressor {
 public:
  Compressor();

  /// Throws Error(InvalidArgument) if the options are out of range.
  explicit Compressor(const CompressionOptions& options);

  [[nodiscard]] const CompressionOptions& options() const noexcept { return options_; }

  /// Returns a self-describing stream that Decompressor::decompress() turns
  /// back into the exact input bit patterns.
  [[nodiscard]] std::vector<std::uint8_t> compress(std::span<const float> data) const;

 private:
  CompressionOptions options_;
};

}  // namespace turbocore

#endif  // TURBOCORE_COMPRESSOR_HPP
