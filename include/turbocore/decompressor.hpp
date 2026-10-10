#ifndef TURBOCORE_DECOMPRESSOR_HPP
#define TURBOCORE_DECOMPRESSOR_HPP

#include <cstdint>
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
};

/// Decoder for streams produced by Compressor. All input is treated as
/// untrusted: malformed data raises Error and never reads or writes out of
/// bounds.
class TURBOCORE_EXPORT Decompressor {
 public:
  /// Validates and returns the header without decoding any block.
  /// Throws Error(UnsupportedFormat | CorruptStream).
  [[nodiscard]] static StreamInfo inspect(std::span<const std::uint8_t> stream);

  /// Throws Error(UnsupportedFormat | CorruptStream | ChecksumMismatch).
  [[nodiscard]] std::vector<float> decompress(std::span<const std::uint8_t> stream) const;
};

}  // namespace turbocore

#endif  // TURBOCORE_DECOMPRESSOR_HPP
