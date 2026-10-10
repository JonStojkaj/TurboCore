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
#include "detail/kernels.hpp"
#include "turbocore/compressor.hpp"
#include "turbocore/error.hpp"

// The per-block lossless pipeline, shared by the 32-bit (FP32) and 16-bit
// (quantized) element widths:
//
//   encode: words -> CRC -> predictor -> byte shuffle -> | -> codec -> payload
//   decode: payload -> codec -> | -> byte unshuffle -> predictor -> CRC check -> words
//
// The part left of the bar is the transform stage: data-parallel, run through
// a KernelTable, and the part a GPU backend can take over for a whole stream
// at once. The part right of it is the entropy stage (BlockPacker /
// BlockUnpacker), which is serial within a block and runs one block per CPU
// thread.
//
// Every class here owns its scratch buffers and is meant to be created once
// per thread and reused for all the blocks that thread handles.

namespace turbocore::detail {

/// Entropy stage, encoding side.
class BlockPacker {
 public:
  explicit BlockPacker(const Codec& codec) noexcept : codec_(codec) {}

  /// Replaces the contents of `payload` with the block's payload and returns
  /// its index entry.
  IndexEntry pack(std::span<const std::uint8_t> transformed, std::uint32_t crc, const CodecHints& hints,
                  std::vector<std::uint8_t>& payload) const {
    payload.clear();
    codec_.compress(transformed, payload, hints);

    // Incompressible blocks are stored verbatim so the stream never grows by
    // more than the header and index.
    const bool use_deflate = payload.size() < transformed.size();
    if (!use_deflate) {
      payload.assign(transformed.begin(), transformed.end());
    }
    return IndexEntry{
        .compressed_size = static_cast<std::uint32_t>(payload.size()),
        .crc = crc,
        .method = use_deflate ? BlockMethod::Deflate : BlockMethod::Stored,
    };
  }

 private:
  const Codec& codec_;
};

/// Entropy stage, decoding side.
class BlockUnpacker {
 public:
  explicit BlockUnpacker(const Codec& codec) noexcept : codec_(codec) {}

  /// Writes the block's `transformed.size()` transformed bytes.
  /// Throws Error(CorruptStream).
  void unpack(std::span<const std::uint8_t> payload, const IndexEntry& entry,
              std::span<std::uint8_t> transformed) const {
    if (entry.method == BlockMethod::Stored) {
      if (payload.size() != transformed.size()) {
        throw Error(ErrorCode::CorruptStream, "stored block has the wrong size");
      }
      if (!payload.empty()) {
        std::memcpy(transformed.data(), payload.data(), payload.size());
      }
    } else if (!codec_.decompress(payload, transformed)) {
      throw Error(ErrorCode::CorruptStream, "block is not a valid DEFLATE stream of the expected size");
    }
  }

 private:
  const Codec& codec_;
};

template <std::unsigned_integral Word>
class BlockEncoder {
 public:
  BlockEncoder(const KernelTable& kernels, Predictor predictor, bool byte_shuffle, const Codec& codec) noexcept
      : kernels_(kernels), predictor_(predictor), byte_shuffle_(byte_shuffle), packer_(codec) {}

  /// Replaces the contents of `payload` with the block's payload and returns
  /// its index entry. `words` is used as working storage and holds garbage
  /// afterwards.
  IndexEntry encode(std::span<Word> words, std::vector<std::uint8_t>& payload) {
    const std::size_t byte_count = words.size_bytes();
    const std::uint32_t crc = kernels_.crc32c(reinterpret_cast<const std::uint8_t*>(words.data()), byte_count);

    switch (predictor_) {
      case Predictor::None:
        break;
      case Predictor::Delta:
        delta_encode(kernels_, words);
        break;
      case Predictor::Xor:
        xor_encode(kernels_, words);
        break;
    }

    transformed_.resize(byte_count);
    if (byte_shuffle_) {
      shuffle_bytes(kernels_, std::span<const Word>(words), transformed_);
    } else {
      std::memcpy(transformed_.data(), words.data(), byte_count);
    }
    // With the byte shuffle on, each plane of words.size() bytes has its own
    // statistics; tell the codec where the planes are.
    return packer_.pack(transformed_, crc, CodecHints{.segment_bytes = byte_shuffle_ ? words.size() : 0}, payload);
  }

 private:
  const KernelTable& kernels_;
  Predictor predictor_;
  bool byte_shuffle_;
  BlockPacker packer_;
  std::vector<std::uint8_t> transformed_;
};

template <std::unsigned_integral Word>
class BlockDecoder {
 public:
  BlockDecoder(const KernelTable& kernels, Predictor predictor, bool byte_shuffle, const Codec& codec) noexcept
      : kernels_(kernels), predictor_(predictor), byte_shuffle_(byte_shuffle), unpacker_(codec) {}

  /// Fills `words` from the block's payload.
  /// Throws Error(CorruptStream | ChecksumMismatch).
  void decode(std::span<const std::uint8_t> payload, const IndexEntry& entry, std::span<Word> words) {
    const std::size_t byte_count = words.size_bytes();
    transformed_.resize(byte_count);
    unpacker_.unpack(payload, entry, transformed_);

    if (byte_shuffle_) {
      unshuffle_bytes(kernels_, std::span<const std::uint8_t>(transformed_), words);
    } else {
      std::memcpy(words.data(), transformed_.data(), byte_count);
    }

    switch (predictor_) {
      case Predictor::None:
        break;
      case Predictor::Delta:
        delta_decode(kernels_, words);
        break;
      case Predictor::Xor:
        xor_decode(kernels_, words);
        break;
    }

    if (kernels_.crc32c(reinterpret_cast<const std::uint8_t*>(words.data()), byte_count) != entry.crc) {
      throw Error(ErrorCode::ChecksumMismatch, "block checksum mismatch");
    }
  }

 private:
  const KernelTable& kernels_;
  Predictor predictor_;
  bool byte_shuffle_;
  BlockUnpacker unpacker_;
  std::vector<std::uint8_t> transformed_;
};

}  // namespace turbocore::detail

#endif  // TURBOCORE_DETAIL_BLOCK_PIPELINE_HPP
