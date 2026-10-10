#include "turbocore/compressor.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

#include "detail/codec.hpp"
#include "detail/container.hpp"
#include "detail/crc32c.hpp"
#include "detail/transforms.hpp"
#include "turbocore/error.hpp"

namespace turbocore {

namespace {

const CompressionOptions& validated(const CompressionOptions& options) {
  if (options.predictor != Predictor::None && options.predictor != Predictor::Delta &&
      options.predictor != Predictor::Xor) {
    throw Error(ErrorCode::InvalidArgument, "unknown predictor");
  }
  if (options.level < detail::DeflateCodec::kMinLevel || options.level > detail::DeflateCodec::kMaxLevel) {
    throw Error(ErrorCode::InvalidArgument, "compression level must be between 1 and 9");
  }
  if (options.block_elements == 0 || options.block_elements > detail::kMaxBlockElements) {
    throw Error(ErrorCode::InvalidArgument, "block_elements must be between 1 and 2^24");
  }
  return options;
}

}  // namespace

Compressor::Compressor() = default;

Compressor::Compressor(const CompressionOptions& options) : options_(validated(options)) {}

std::vector<std::uint8_t> Compressor::compress(std::span<const float> data) const {
  using namespace detail;

  const std::size_t block_elements = options_.block_elements;
  const std::size_t block_count = data.size() / block_elements + (data.size() % block_elements != 0 ? 1 : 0);
  if (block_count > std::numeric_limits<std::uint32_t>::max()) {
    throw Error(ErrorCode::InvalidArgument, "input needs more blocks than the format allows; raise block_elements");
  }

  const std::size_t payload_offset = kHeaderSize + block_count * kIndexEntrySize;
  std::vector<std::uint8_t> out(payload_offset);
  write_header(std::span<std::uint8_t, kHeaderSize>(out.data(), kHeaderSize),
               StreamInfo{
                   .element_count = data.size(),
                   .block_count = static_cast<std::uint32_t>(block_count),
                   .block_elements = static_cast<std::uint32_t>(block_elements),
                   .predictor = options_.predictor,
                   .byte_shuffle = options_.byte_shuffle,
               });

  const DeflateCodec codec(options_.level);
  std::vector<std::uint32_t> words;       // bit patterns of the block
  std::vector<std::uint8_t> transformed;  // after predictor + shuffle
  std::vector<std::uint8_t> packed;       // after the codec

  for (std::size_t block = 0; block < block_count; ++block) {
    const std::span<const float> chunk =
        data.subspan(block * block_elements, std::min(block_elements, data.size() - block * block_elements));
    const std::size_t byte_count = chunk.size() * kElementSize;

    words.resize(chunk.size());
    std::memcpy(words.data(), chunk.data(), byte_count);
    const auto* word_bytes = reinterpret_cast<const std::uint8_t*>(words.data());
    const std::uint32_t crc = crc32c({word_bytes, byte_count});

    switch (options_.predictor) {
      case Predictor::None:
        break;
      case Predictor::Delta:
        delta_encode(words);
        break;
      case Predictor::Xor:
        xor_encode(words);
        break;
    }

    transformed.resize(byte_count);
    if (options_.byte_shuffle) {
      shuffle_bytes(words, transformed);
    } else {
      std::memcpy(transformed.data(), words.data(), byte_count);
    }

    packed.clear();
    codec.compress(transformed, packed);

    // Incompressible blocks are stored verbatim so the stream never grows by
    // more than the header and index.
    const bool use_deflate = packed.size() < transformed.size();
    const std::vector<std::uint8_t>& payload = use_deflate ? packed : transformed;
    out.insert(out.end(), payload.begin(), payload.end());
    write_index_entry(out.data() + kHeaderSize + block * kIndexEntrySize,
                      IndexEntry{
                          .compressed_size = static_cast<std::uint32_t>(payload.size()),
                          .crc = crc,
                          .method = use_deflate ? BlockMethod::Deflate : BlockMethod::Stored,
                      });
  }
  return out;
}

}  // namespace turbocore
