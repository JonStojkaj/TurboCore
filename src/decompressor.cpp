#include "turbocore/decompressor.hpp"

#include <algorithm>
#include <cstring>

#include "detail/codec.hpp"
#include "detail/container.hpp"
#include "detail/crc32c.hpp"
#include "detail/transforms.hpp"
#include "turbocore/error.hpp"

namespace turbocore {

StreamInfo Decompressor::inspect(std::span<const std::uint8_t> stream) { return detail::parse_header(stream); }

std::vector<float> Decompressor::decompress(std::span<const std::uint8_t> stream) const {
  using namespace detail;

  const StreamInfo info = parse_header(stream);
  const auto element_count = static_cast<std::size_t>(info.element_count);
  const std::size_t block_elements = info.block_elements;

  std::vector<float> out(element_count);
  const DeflateCodec codec;
  std::vector<std::uint8_t> inflated;
  std::vector<std::uint32_t> words;

  std::size_t offset = kHeaderSize + std::size_t{info.block_count} * kIndexEntrySize;
  for (std::size_t block = 0; block < info.block_count; ++block) {
    const IndexEntry entry = parse_index_entry(stream.data() + kHeaderSize + block * kIndexEntrySize);
    const std::size_t first_element = block * block_elements;
    const std::size_t count = std::min(block_elements, element_count - first_element);
    const std::size_t byte_count = count * kElementSize;

    if (entry.compressed_size > stream.size() - offset) {
      throw Error(ErrorCode::CorruptStream, "block extends past the end of the stream");
    }
    const std::span<const std::uint8_t> payload = stream.subspan(offset, entry.compressed_size);
    offset += entry.compressed_size;

    std::span<const std::uint8_t> transformed;
    if (entry.method == BlockMethod::Stored) {
      if (payload.size() != byte_count) {
        throw Error(ErrorCode::CorruptStream, "stored block has the wrong size");
      }
      transformed = payload;
    } else {
      inflated.resize(byte_count);
      if (!codec.decompress(payload, inflated)) {
        throw Error(ErrorCode::CorruptStream, "block is not a valid DEFLATE stream of the expected size");
      }
      transformed = inflated;
    }

    words.resize(count);
    if (info.byte_shuffle) {
      unshuffle_bytes(transformed, words);
    } else {
      std::memcpy(words.data(), transformed.data(), byte_count);
    }

    switch (info.predictor) {
      case Predictor::None:
        break;
      case Predictor::Delta:
        delta_decode(words);
        break;
      case Predictor::Xor:
        xor_decode(words);
        break;
    }

    const auto* word_bytes = reinterpret_cast<const std::uint8_t*>(words.data());
    if (crc32c({word_bytes, byte_count}) != entry.crc) {
      throw Error(ErrorCode::ChecksumMismatch, "block checksum mismatch");
    }
    std::memcpy(out.data() + first_element, words.data(), byte_count);
  }

  if (offset != stream.size()) {
    throw Error(ErrorCode::CorruptStream, "unexpected bytes after the last block");
  }
  return out;
}

}  // namespace turbocore
