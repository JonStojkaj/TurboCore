#include "turbocore/compressor.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

#include "detail/block_pipeline.hpp"
#include "detail/codec.hpp"
#include "detail/container.hpp"
#include "detail/kernels.hpp"
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

  std::vector<std::uint8_t> out(kHeaderSize + block_count * kIndexEntrySize);
  write_header(std::span<std::uint8_t, kHeaderSize>(out.data(), kHeaderSize),
               StreamInfo{
                   .element_count = data.size(),
                   .block_count = static_cast<std::uint32_t>(block_count),
                   .block_elements = static_cast<std::uint32_t>(block_elements),
                   .predictor = options_.predictor,
                   .byte_shuffle = options_.byte_shuffle,
                   .quantization = options_.quantization,
               });

  const DeflateCodec codec(options_.level);
  const KernelTable& kernels = select_kernels(options_.backend);

  // `load` turns a chunk of input floats into the block's stored elements;
  // everything after that is the same lossless pipeline for either width.
  const auto encode_blocks = [&]<typename Word>(std::vector<Word>& words, auto&& load) {
    BlockEncoder<Word> encoder(kernels, options_.predictor, options_.byte_shuffle, codec);
    for (std::size_t block = 0; block < block_count; ++block) {
      const std::size_t first = block * block_elements;
      const std::span<const float> chunk = data.subspan(first, std::min(block_elements, data.size() - first));
      words.resize(chunk.size());
      load(chunk, std::span<Word>(words));
      const IndexEntry entry = encoder.encode(words, out);
      write_index_entry(out.data() + kHeaderSize + block * kIndexEntrySize, entry);
    }
  };

  if (options_.quantization.has_value()) {
    std::vector<std::uint16_t> words;
    encode_blocks(words, [&](std::span<const float> chunk, std::span<std::uint16_t> stored) {
      quantize(kernels, chunk, stored, *options_.quantization);
    });
  } else {
    std::vector<std::uint32_t> words;
    encode_blocks(words, [](std::span<const float> chunk, std::span<std::uint32_t> stored) {
      std::memcpy(stored.data(), chunk.data(), chunk.size_bytes());
    });
  }
  return out;
}

}  // namespace turbocore
