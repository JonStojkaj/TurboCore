#include "detail/container.hpp"

#include <algorithm>

#include "detail/crc32c.hpp"
#include "turbocore/error.hpp"

namespace turbocore::detail {

void write_header(std::span<std::uint8_t, kHeaderSize> out, const StreamInfo& info) noexcept {
  std::fill(out.begin(), out.end(), std::uint8_t{0});
  std::copy(kMagic.begin(), kMagic.end(), out.begin());
  out[4] = kFormatVersion;
  out[5] = kElementTypeFloat32;
  out[6] = static_cast<std::uint8_t>(info.predictor);
  out[7] = info.byte_shuffle ? kFlagByteShuffle : std::uint8_t{0};
  store_le<std::uint32_t>(&out[8], info.block_elements);
  store_le<std::uint64_t>(&out[12], info.element_count);
  store_le<std::uint32_t>(&out[20], info.block_count);
  store_le<std::uint32_t>(&out[kHeaderCrcOffset], crc32c(out.first(kHeaderCrcOffset)));
}

void write_index_entry(std::uint8_t* out, const IndexEntry& entry) noexcept {
  store_le<std::uint32_t>(out, entry.compressed_size);
  store_le<std::uint32_t>(out + 4, entry.crc);
  out[8] = static_cast<std::uint8_t>(entry.method);
  out[9] = out[10] = out[11] = 0;
}

StreamInfo parse_header(std::span<const std::uint8_t> stream) {
  if (stream.size() < kHeaderSize) {
    throw Error(ErrorCode::CorruptStream, "stream is shorter than the TurboCore header");
  }
  const std::uint8_t* h = stream.data();
  if (!std::equal(kMagic.begin(), kMagic.end(), h)) {
    throw Error(ErrorCode::UnsupportedFormat, "not a TurboCore stream (bad magic)");
  }
  if (h[4] != kFormatVersion) {
    throw Error(ErrorCode::UnsupportedFormat, "unsupported TurboCore format version");
  }
  if (load_le<std::uint32_t>(h + kHeaderCrcOffset) != crc32c(stream.first(kHeaderCrcOffset))) {
    throw Error(ErrorCode::CorruptStream, "header checksum mismatch");
  }
  if (h[5] != kElementTypeFloat32) {
    throw Error(ErrorCode::UnsupportedFormat, "unsupported element type");
  }
  if (h[6] > static_cast<std::uint8_t>(Predictor::Xor) || (h[7] & ~kFlagByteShuffle) != 0 ||
      load_le<std::uint32_t>(h + 24) != 0) {
    throw Error(ErrorCode::CorruptStream, "invalid header fields");
  }

  StreamInfo info{};
  info.predictor = static_cast<Predictor>(h[6]);
  info.byte_shuffle = (h[7] & kFlagByteShuffle) != 0;
  info.block_elements = load_le<std::uint32_t>(h + 8);
  info.element_count = load_le<std::uint64_t>(h + 12);
  info.block_count = load_le<std::uint32_t>(h + 20);

  if (info.block_elements == 0 || info.block_elements > kMaxBlockElements) {
    throw Error(ErrorCode::CorruptStream, "invalid block size");
  }
  const std::uint64_t expected_blocks =
      info.element_count / info.block_elements + (info.element_count % info.block_elements != 0 ? 1 : 0);
  if (expected_blocks != info.block_count) {
    throw Error(ErrorCode::CorruptStream, "block count does not match element count");
  }

  const std::uint64_t index_bytes = std::uint64_t{info.block_count} * kIndexEntrySize;
  if (index_bytes > stream.size() - kHeaderSize) {
    throw Error(ErrorCode::CorruptStream, "stream is shorter than its block index");
  }
  // Refuse to allocate for an element count the payload cannot possibly hold.
  const std::uint64_t payload_bytes = stream.size() - kHeaderSize - index_bytes;
  if (info.element_count / (kMaxExpansionRatio / kElementSize) > payload_bytes) {
    throw Error(ErrorCode::CorruptStream, "element count is implausible for the stream size");
  }
  return info;
}

IndexEntry parse_index_entry(const std::uint8_t* in) {
  if (in[8] > static_cast<std::uint8_t>(BlockMethod::Deflate) || in[9] != 0 || in[10] != 0 || in[11] != 0) {
    throw Error(ErrorCode::CorruptStream, "invalid block index entry");
  }
  return IndexEntry{
      .compressed_size = load_le<std::uint32_t>(in),
      .crc = load_le<std::uint32_t>(in + 4),
      .method = static_cast<BlockMethod>(in[8]),
  };
}

}  // namespace turbocore::detail
