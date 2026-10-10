#include "detail/huffman.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace turbocore::detail {

namespace {

std::uint16_t reverse_bits(std::uint32_t code, unsigned length) noexcept {
  std::uint32_t reversed = 0;
  for (unsigned i = 0; i < length; ++i) {
    reversed = (reversed << 1) | ((code >> i) & 1u);
  }
  return static_cast<std::uint16_t>(reversed);
}

}  // namespace

void build_code_lengths(std::span<const std::uint32_t> frequencies, unsigned max_bits,
                        std::span<std::uint8_t> lengths) {
  std::fill(lengths.begin(), lengths.end(), std::uint8_t{0});

  // Used symbols, rarest first; ties broken by symbol for determinism.
  std::vector<std::uint32_t> order;
  for (std::size_t symbol = 0; symbol < frequencies.size(); ++symbol) {
    if (frequencies[symbol] != 0) {
      order.push_back(static_cast<std::uint32_t>(symbol));
    }
  }
  std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
    return frequencies[a] != frequencies[b] ? frequencies[a] < frequencies[b] : a < b;
  });

  const std::size_t leaf_count = order.size();
  if (leaf_count == 0) {
    return;
  }
  if (leaf_count == 1) {
    lengths[order[0]] = 1;
    return;
  }

  // Huffman tree by the two-queue method: leaves are already sorted and
  // internal nodes are created in non-decreasing weight order, so the two
  // lightest nodes are always at the front of one of the two queues.
  // Nodes [0, leaf_count) are leaves, the rest are internal; the last is the root.
  const std::size_t node_count = 2 * leaf_count - 1;
  std::vector<std::uint64_t> weight(node_count);
  std::vector<std::uint32_t> parent(node_count, 0);
  for (std::size_t i = 0; i < leaf_count; ++i) {
    weight[i] = frequencies[order[i]];
  }
  std::size_t next_leaf = 0;
  std::size_t next_internal = leaf_count;
  for (std::size_t created = leaf_count; created < node_count; ++created) {
    const auto take_lightest = [&]() {
      const bool leaf_available = next_leaf < leaf_count;
      const bool internal_available = next_internal < created;
      if (leaf_available && (!internal_available || weight[next_leaf] <= weight[next_internal])) {
        return next_leaf++;
      }
      return next_internal++;
    };
    const std::size_t a = take_lightest();
    const std::size_t b = take_lightest();
    weight[created] = weight[a] + weight[b];
    parent[a] = static_cast<std::uint32_t>(created);
    parent[b] = static_cast<std::uint32_t>(created);
  }

  // Depths, root first (a parent always has a higher index than its children).
  std::vector<std::uint32_t> depth(node_count, 0);
  for (std::size_t node = node_count - 1; node-- > 0;) {
    depth[node] = depth[parent[node]] + 1;
  }

  // Number of codes of each length; anything deeper than the limit is
  // clamped to it, which over-subscribes the code.
  std::vector<std::uint32_t> length_count(max_bits + 1, 0);
  for (std::size_t leaf = 0; leaf < leaf_count; ++leaf) {
    ++length_count[std::min<std::uint32_t>(depth[leaf], max_bits)];
  }

  // Repair the Kraft sum: repeatedly take one code off the longest length and
  // split the deepest shorter code into two children, until the code is
  // exactly complete again.
  std::uint64_t kraft = 0;
  for (unsigned length = 1; length <= max_bits; ++length) {
    kraft += static_cast<std::uint64_t>(length_count[length]) << (max_bits - length);
  }
  const std::uint64_t full = std::uint64_t{1} << max_bits;
  while (kraft > full) {
    --length_count[max_bits];
    for (unsigned length = max_bits - 1; length > 0; --length) {
      if (length_count[length] != 0) {
        --length_count[length];
        length_count[length + 1] += 2;
        break;
      }
    }
    --kraft;
  }

  // Longest codes go to the rarest symbols.
  std::size_t next = 0;
  for (unsigned length = max_bits; length > 0; --length) {
    for (std::uint32_t i = 0; i < length_count[length]; ++i) {
      lengths[order[next++]] = static_cast<std::uint8_t>(length);
    }
  }
}

void assign_reversed_codes(std::span<const std::uint8_t> lengths, std::span<std::uint16_t> codes) {
  std::array<std::uint32_t, kMaxHuffmanBits + 1> length_count{};
  for (const std::uint8_t length : lengths) {
    ++length_count[length];
  }
  length_count[0] = 0;

  std::array<std::uint32_t, kMaxHuffmanBits + 2> next_code{};
  std::uint32_t code = 0;
  for (unsigned length = 1; length <= kMaxHuffmanBits; ++length) {
    code = (code + length_count[length - 1]) << 1;
    next_code[length] = code;
  }

  for (std::size_t symbol = 0; symbol < lengths.size(); ++symbol) {
    const unsigned length = lengths[symbol];
    codes[symbol] = length != 0 ? reverse_bits(next_code[length]++, length) : std::uint16_t{0};
  }
}

bool HuffmanDecoder::build(std::span<const std::uint8_t> lengths) noexcept {
  count_.fill(0);
  fast_.fill(0);
  for (const std::uint8_t length : lengths) {
    ++count_[length];
  }
  count_[0] = 0;

  int unused = 1;
  for (unsigned length = 1; length <= kMaxHuffmanBits; ++length) {
    unused = unused * 2 - static_cast<int>(count_[length]);
    if (unused < 0) {
      return false;
    }
  }

  // Symbols sorted by (length, symbol): the canonical order.
  std::array<std::uint16_t, kMaxHuffmanBits + 2> offset{};
  for (unsigned length = 1; length <= kMaxHuffmanBits; ++length) {
    offset[length + 1] = static_cast<std::uint16_t>(offset[length] + count_[length]);
  }
  for (std::size_t symbol = 0; symbol < lengths.size(); ++symbol) {
    const unsigned length = lengths[symbol];
    if (length != 0) {
      symbols_[offset[length]++] = static_cast<std::uint16_t>(symbol);
    }
  }

  // Every kFastBits-bit window that starts with a short code maps to it.
  std::uint32_t code = 0;
  std::size_t index = 0;
  for (unsigned length = 1; length <= kFastBits; ++length) {
    for (unsigned i = 0; i < count_[length]; ++i) {
      const auto entry = static_cast<std::uint16_t>((std::uint32_t{symbols_[index++]} << 4) | length);
      for (std::uint32_t slot = reverse_bits(code, length); slot < fast_.size(); slot += 1u << length) {
        fast_[slot] = entry;
      }
      ++code;
    }
    code <<= 1;
  }
  return true;
}

int HuffmanDecoder::decode(BitReader& reader) const noexcept {
  const std::uint32_t bits = reader.peek(kMaxHuffmanBits);

  const std::uint16_t entry = fast_[bits & ((1u << kFastBits) - 1)];
  if (entry != 0) {
    reader.consume(entry & 0xFu);
    return entry >> 4;
  }

  // Canonical walk, one bit at a time, MSB of the code first.
  int code = 0;
  int first = 0;
  int index = 0;
  for (unsigned length = 1; length <= kMaxHuffmanBits; ++length) {
    code |= static_cast<int>((bits >> (length - 1)) & 1u);
    const int count = count_[length];
    if (code - count < first) {
      reader.consume(length);
      return symbols_[static_cast<std::size_t>(index + (code - first))];
    }
    index += count;
    first = (first + count) << 1;
    code <<= 1;
  }
  return -1;
}

}  // namespace turbocore::detail
