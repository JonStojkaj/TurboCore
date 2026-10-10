#include "turbocore/decompressor.hpp"

#include <algorithm>
#include <cstring>
#include <type_traits>

#include "detail/block_pipeline.hpp"
#include "detail/codec.hpp"
#include "detail/container.hpp"
#include "detail/kernels.hpp"
#include "detail/parallel.hpp"
#include "turbocore/error.hpp"

namespace turbocore {

Decompressor::Decompressor() = default;

Decompressor::Decompressor(const DecompressionOptions& options) : options_(options) {
  (void)detail::select_kernels(options.backend);  // throws if the backend cannot run here
}

StreamInfo Decompressor::inspect(std::span<const std::uint8_t> stream) { return detail::parse_header(stream); }

std::vector<float> Decompressor::decompress(std::span<const std::uint8_t> stream) const {
  using namespace detail;

  const StreamInfo info = parse_header(stream);
  const auto element_count = static_cast<std::size_t>(info.element_count);
  const std::size_t block_elements = info.block_elements;
  const std::size_t block_count = info.block_count;

  // Locate every block first. A block's offset is the sum of the sizes
  // before it, so this pass is sequential; it also validates the index, so
  // the parallel pass below can never be pointed outside the stream.
  struct BlockLocation {
    std::size_t offset;
    IndexEntry entry;
  };
  std::vector<BlockLocation> locations(block_count);
  std::size_t offset = kHeaderSize + block_count * kIndexEntrySize;
  for (std::size_t block = 0; block < block_count; ++block) {
    const IndexEntry entry = parse_index_entry(stream.data() + kHeaderSize + block * kIndexEntrySize);
    if (entry.compressed_size > stream.size() - offset) {
      throw Error(ErrorCode::CorruptStream, "block extends past the end of the stream");
    }
    locations[block] = {offset, entry};
    offset += entry.compressed_size;
  }
  if (offset != stream.size()) {
    throw Error(ErrorCode::CorruptStream, "unexpected bytes after the last block");
  }

  std::vector<float> out(element_count);
  const DeflateCodec codec;
  const KernelTable& kernels = select_kernels(options_.backend);

  // Each block decodes into its own disjoint range of `out`.
  // `store` turns a block's decoded elements back into output floats.
  const auto decode_blocks = [&]<typename Word>(std::type_identity<Word>, auto store) {
    parallel_for(block_count, options_.threads, [&] {
      return [&, store, decoder = BlockDecoder<Word>(kernels, info.predictor, info.byte_shuffle, codec),
              words = std::vector<Word>()](std::size_t block) mutable {
        const BlockLocation& location = locations[block];
        const std::size_t first = block * block_elements;
        words.resize(std::min(block_elements, element_count - first));
        decoder.decode(stream.subspan(location.offset, location.entry.compressed_size), location.entry,
                       std::span<Word>(words));
        store(std::span<const Word>(words), std::span<float>(out).subspan(first, words.size()));
      };
    });
  };

  if (info.quantization.has_value()) {
    decode_blocks(std::type_identity<std::uint16_t>{},
                  [&](std::span<const std::uint16_t> stored, std::span<float> chunk) {
                    dequantize(kernels, stored, chunk, *info.quantization);
                  });
  } else {
    decode_blocks(std::type_identity<std::uint32_t>{},
                  [](std::span<const std::uint32_t> stored, std::span<float> chunk) {
                    std::memcpy(chunk.data(), stored.data(), stored.size_bytes());
                  });
  }
  return out;
}

}  // namespace turbocore
