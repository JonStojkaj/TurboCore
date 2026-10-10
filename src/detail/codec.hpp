#ifndef TURBOCORE_DETAIL_CODEC_HPP
#define TURBOCORE_DETAIL_CODEC_HPP

#include <cstdint>
#include <span>
#include <vector>

namespace turbocore::detail {

/// A lossless byte-level entropy stage. Implementations are stateless across
/// calls, so one instance can serve many blocks (and, later, many threads).
class Codec {
 public:
  Codec() = default;
  Codec(const Codec&) = delete;
  Codec& operator=(const Codec&) = delete;
  virtual ~Codec() = default;

  /// Appends the compressed form of `in` to `out`.
  virtual void compress(std::span<const std::uint8_t> in, std::vector<std::uint8_t>& out) const = 0;

  /// Decodes `in`, which must expand to exactly out.size() bytes and be
  /// consumed completely. Returns false on any malformed input; never reads
  /// or writes outside the two spans.
  [[nodiscard]] virtual bool decompress(std::span<const std::uint8_t> in,
                                        std::span<std::uint8_t> out) const noexcept = 0;
};

/// Raw DEFLATE (RFC 1951, no zlib or gzip wrapper).
///
/// The encoder emits one final block per call, fixed or dynamic Huffman,
/// whichever is smaller, with greedy hash-chain LZ77. The decoder accepts any
/// conforming stream, including stored blocks and multiple blocks.
class DeflateCodec final : public Codec {
 public:
  static constexpr int kMinLevel = 1;
  static constexpr int kMaxLevel = 9;

  /// `level` must be in [kMinLevel, kMaxLevel]; it sets the LZ77 chain depth.
  explicit DeflateCodec(int level = 6) noexcept;

  /// Requires in.size() < 2^31.
  void compress(std::span<const std::uint8_t> in, std::vector<std::uint8_t>& out) const override;

  [[nodiscard]] bool decompress(std::span<const std::uint8_t> in,
                                std::span<std::uint8_t> out) const noexcept override;

 private:
  unsigned max_chain_;
};

}  // namespace turbocore::detail

#endif  // TURBOCORE_DETAIL_CODEC_HPP
