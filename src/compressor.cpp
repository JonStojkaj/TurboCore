#include "turbocore/compressor.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <type_traits>

#include "detail/block_pipeline.hpp"
#include "detail/codec.hpp"
#include "detail/container.hpp"
#include "detail/kernels.hpp"
#include "detail/parallel.hpp"
#include "turbocore/decompressor.hpp"
#include "turbocore/error.hpp"

namespace turbocore {

namespace {

const CompressionOptions& validated(const CompressionOptions& options) {
  if (options.quantization.has_value() && *options.quantization != QuantFormat::Float16 &&
      *options.quantization != QuantFormat::BFloat16) {
    throw Error(ErrorCode::InvalidArgument, "unknown quantization format");
  }
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
  (void)detail::select_kernels(options.backend);  // throws if the backend cannot run here
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

  const DeflateCodec codec(options_.level);
  const KernelTable& kernels = select_kernels(options_.backend);

  // Blocks are independent, so each one is compressed on whichever thread
  // picks it up, into its own buffer. Assembling the buffers in block order
  // afterwards makes the stream identical for every thread count.
  struct EncodedBlock {
    std::vector<std::uint8_t> payload;
    IndexEntry entry{};
  };
  std::vector<EncodedBlock> blocks(block_count);

  // `load` turns a chunk of input floats into the block's stored elements;
  // everything after that is the same lossless pipeline for either width.
  const auto encode_blocks = [&]<typename Word>(std::type_identity<Word>, auto load) {
    parallel_for(block_count, options_.threads, [&] {
      return [&, load, encoder = BlockEncoder<Word>(kernels, options_.predictor, options_.byte_shuffle, codec),
              words = std::vector<Word>()](std::size_t block) mutable {
        const std::size_t first = block * block_elements;
        const std::span<const float> chunk = data.subspan(first, std::min(block_elements, data.size() - first));
        words.resize(chunk.size());
        load(chunk, std::span<Word>(words));
        blocks[block].entry = encoder.encode(words, blocks[block].payload);
      };
    });
  };

  if (options_.quantization.has_value()) {
    encode_blocks(std::type_identity<std::uint16_t>{},
                  [&](std::span<const float> chunk, std::span<std::uint16_t> stored) {
                    quantize(kernels, chunk, stored, *options_.quantization);
                  });
  } else {
    encode_blocks(std::type_identity<std::uint32_t>{},
                  [](std::span<const float> chunk, std::span<std::uint32_t> stored) {
                    std::memcpy(stored.data(), chunk.data(), chunk.size_bytes());
                  });
  }

  std::size_t total_size = kHeaderSize + block_count * kIndexEntrySize;
  for (const EncodedBlock& block : blocks) {
    total_size += block.payload.size();
  }
  std::vector<std::uint8_t> out;
  out.reserve(total_size);
  out.resize(kHeaderSize + block_count * kIndexEntrySize);
  write_header(std::span<std::uint8_t, kHeaderSize>(out.data(), kHeaderSize),
               StreamInfo{
                   .element_count = data.size(),
                   .block_count = static_cast<std::uint32_t>(block_count),
                   .block_elements = static_cast<std::uint32_t>(block_elements),
                   .predictor = options_.predictor,
                   .byte_shuffle = options_.byte_shuffle,
                   .quantization = options_.quantization,
               });
  for (std::size_t block = 0; block < block_count; ++block) {
    write_index_entry(out.data() + kHeaderSize + block * kIndexEntrySize, blocks[block].entry);
    out.insert(out.end(), blocks[block].payload.begin(), blocks[block].payload.end());
  }
  return out;
}

}  // namespace turbocore
