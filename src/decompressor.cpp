#include "turbocore/decompressor.hpp"

#include <algorithm>
#include <cstring>

#include "detail/block_pipeline.hpp"
#include "detail/codec.hpp"
#include "detail/container.hpp"
#include "detail/kernels.hpp"
#include "turbocore/error.hpp"

namespace turbocore {

Decompressor::Decompressor(Backend backend) : backend_(backend) {
  (void)detail::select_kernels(backend);  // throws if the backend cannot run here
}

StreamInfo Decompressor::inspect(std::span<const std::uint8_t> stream) { return detail::parse_header(stream); }

std::vector<float> Decompressor::decompress(std::span<const std::uint8_t> stream) const {
  using namespace detail;

  const StreamInfo info = parse_header(stream);
  const auto element_count = static_cast<std::size_t>(info.element_count);
  const std::size_t block_elements = info.block_elements;

  std::vector<float> out(element_count);
  const DeflateCodec codec;
  const KernelTable& kernels = select_kernels(backend_);

  // `store` turns a block's decoded elements back into output floats.
  const auto decode_blocks = [&]<typename Word>(std::vector<Word>& words, auto&& store) {
    BlockDecoder<Word> decoder(kernels, info.predictor, info.byte_shuffle, codec);
    std::size_t offset = kHeaderSize + std::size_t{info.block_count} * kIndexEntrySize;
    for (std::size_t block = 0; block < info.block_count; ++block) {
      const IndexEntry entry = parse_index_entry(stream.data() + kHeaderSize + block * kIndexEntrySize);
      if (entry.compressed_size > stream.size() - offset) {
        throw Error(ErrorCode::CorruptStream, "block extends past the end of the stream");
      }
      const std::span<const std::uint8_t> payload = stream.subspan(offset, entry.compressed_size);
      offset += entry.compressed_size;

      const std::size_t first = block * block_elements;
      words.resize(std::min(block_elements, element_count - first));
      decoder.decode(payload, entry, std::span<Word>(words));
      store(std::span<const Word>(words), std::span<float>(out).subspan(first, words.size()));
    }
    if (offset != stream.size()) {
      throw Error(ErrorCode::CorruptStream, "unexpected bytes after the last block");
    }
  };

  if (info.quantization.has_value()) {
    std::vector<std::uint16_t> words;
    decode_blocks(words, [&](std::span<const std::uint16_t> stored, std::span<float> chunk) {
      dequantize(kernels, stored, chunk, *info.quantization);
    });
  } else {
    std::vector<std::uint32_t> words;
    decode_blocks(words, [](std::span<const std::uint32_t> stored, std::span<float> chunk) {
      std::memcpy(chunk.data(), stored.data(), stored.size_bytes());
    });
  }
  return out;
}

}  // namespace turbocore
