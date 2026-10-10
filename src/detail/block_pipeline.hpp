#ifndef TURBOCORE_DETAIL_BLOCK_PIPELINE_HPP
#define TURBOCORE_DETAIL_BLOCK_PIPELINE_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "detail/codec.hpp"
#include "detail/container.hpp"
#include "detail/crc32c.hpp"
#include "detail/transforms.hpp"
#include "turbocore/compressor.hpp"
#include "turbocore/error.hpp"

// The per-block lossless pipeline, shared by the 32-bit (FP32) and 16-bit
// (quantized) element widths:
//
//   encode: words -> CRC -> predictor -> byte shuffle -> codec -> payload
//   decode: payload -> codec -> byte unshuffle -> predictor -> CRC check -> words
//
// Both classes own their scratch buffers, so one instance reused across the
// blocks of a stream allocates only once.

namespace turbocore::detail {

template <std::unsigned_integral Word>
[[nodiscard]] std::span<const std::uint8_t> bytes_of(std::span<const Word> words) noexcept {
  return {reinterpret_cast<const std::uint8_t*>(words.data()), words.size_bytes()};
}

template <std::unsigned_integral Word>
class BlockEncoder {
 public:
  BlockEncoder(Predictor predictor, bool byte_shuffle, const Codec& codec) noexcept
      : predictor_(predictor), byte_shuffle_(byte_shuffle), codec_(codec) {}

  /// Appends the block's payload to `out` and returns its index entry.
  /// `words` is used as working storage and holds garbage afterwards.
  IndexEntry encode(std::span<Word> words, std::vector<std::uint8_t>& out) {
    const std::size_t byte_count = words.size_bytes();
    const std::uint32_t crc = crc32c(bytes_of<Word>(words));

    switch (predictor_) {
      case Predictor::None:
        break;
      case Predictor::Delta:
        delta_encode(words);
        break;
      case Predictor::Xor:
        xor_encode(words);
        break;
    }

    transformed_.resize(byte_count);
    if (byte_shuffle_) {
      shuffle_bytes(std::span<const Word>(words), transformed_);
    } else {
      std::memcpy(transformed_.data(), words.data(), byte_count);
    }

    packed_.clear();
    codec_.compress(transformed_, packed_);

    // Incompressible blocks are stored verbatim so the stream never grows by
    // more than the header and index.
    const bool use_deflate = packed_.size() < transformed_.size();
    const std::vector<std::uint8_t>& payload = use_deflate ? packed_ : transformed_;
    out.insert(out.end(), payload.begin(), payload.end());
    return IndexEntry{
        .compressed_size = static_cast<std::uint32_t>(payload.size()),
        .crc = crc,
        .method = use_deflate ? BlockMethod::Deflate : BlockMethod::Stored,
    };
  }

 private:
  Predictor predictor_;
  bool byte_shuffle_;
  const Codec& codec_;
  std::vector<std::uint8_t> transformed_;
  std::vector<std::uint8_t> packed_;
};

template <std::unsigned_integral Word>
class BlockDecoder {
 public:
  BlockDecoder(Predictor predictor, bool byte_shuffle, const Codec& codec) noexcept
      : predictor_(predictor), byte_shuffle_(byte_shuffle), codec_(codec) {}

  /// Fills `words` from the block's payload.
  /// Throws Error(CorruptStream | ChecksumMismatch).
  void decode(std::span<const std::uint8_t> payload, const IndexEntry& entry, std::span<Word> words) {
    const std::size_t byte_count = words.size_bytes();

    std::span<const std::uint8_t> transformed;
    if (entry.method == BlockMethod::Stored) {
      if (payload.size() != byte_count) {
        throw Error(ErrorCode::CorruptStream, "stored block has the wrong size");
      }
      transformed = payload;
    } else {
      inflated_.resize(byte_count);
      if (!codec_.decompress(payload, inflated_)) {
        throw Error(ErrorCode::CorruptStream, "block is not a valid DEFLATE stream of the expected size");
      }
      transformed = inflated_;
    }

    if (byte_shuffle_) {
      unshuffle_bytes(transformed, words);
    } else {
      std::memcpy(words.data(), transformed.data(), byte_count);
    }

    switch (predictor_) {
      case Predictor::None:
        break;
      case Predictor::Delta:
        delta_decode(words);
        break;
      case Predictor::Xor:
        xor_decode(words);
        break;
    }

    if (crc32c(bytes_of<Word>(words)) != entry.crc) {
      throw Error(ErrorCode::ChecksumMismatch, "block checksum mismatch");
    }
  }

 private:
  Predictor predictor_;
  bool byte_shuffle_;
  const Codec& codec_;
  std::vector<std::uint8_t> inflated_;
};

}  // namespace turbocore::detail

#endif  // TURBOCORE_DETAIL_BLOCK_PIPELINE_HPP
