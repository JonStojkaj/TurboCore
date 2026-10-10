#ifndef TURBOCORE_DETAIL_CODEC_HPP
#define TURBOCORE_DETAIL_CODEC_HPP

#include <cstddef>
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
  ///
  /// `segment_hint` tells the codec that `in` is made of consecutive regions
  /// of that many bytes with different statistics (the byte planes of a
  /// shuffled block), so it can model each one separately. 0 means unknown.
  /// The hint affects only the size and speed of the result, never validity.
  virtual void compress(std::span<const std::uint8_t> in, std::vector<std::uint8_t>& out,
                        std::size_t segment_hint) const = 0;

  void compress(std::span<const std::uint8_t> in, std::vector<std::uint8_t>& out) const { compress(in, out, 0); }

  /// Decodes `in`, which must expand to exactly out.size() bytes and be
  /// consumed completely. Returns false on any malformed input; never reads
  /// or writes outside the two spans.
  [[nodiscard]] virtual bool decompress(std::span<const std::uint8_t> in,
                                        std::span<std::uint8_t> out) const noexcept = 0;
};

/// Raw DEFLATE (RFC 1951, no zlib or gzip wrapper).
///
/// The encoder writes one DEFLATE block per segment of the input, so every
/// byte plane gets its own Huffman code, and for each segment emits the
/// cheapest of: stored, Huffman-coded literals, or LZ77 + Huffman (greedy
/// hash chains, fixed or dynamic codes).
///
/// LZ77 match search is by far the most expensive step and often buys
/// nothing: the mantissa planes of float data are close to noise, and a plane
/// that already codes to a fraction of a bit per byte has little left to
/// give. At levels below kMaxLevel the encoder therefore first runs the
/// search on a short probe of every segment, estimates what it would save,
/// and searches a segment in full only if that is predicted to shrink the
/// whole stream by at least 2 %. kMaxLevel always searches everything.
///
/// The decoder accepts any conforming stream.
class DeflateCodec final : public Codec {
 public:
  static constexpr int kMinLevel = 1;
  static constexpr int kMaxLevel = 9;

  /// `level` must be in [kMinLevel, kMaxLevel]; it sets the LZ77 chain depth
  /// and whether the match search may be skipped.
  explicit DeflateCodec(int level = 6) noexcept;

  using Codec::compress;

  /// Requires in.size() < 2^31.
  void compress(std::span<const std::uint8_t> in, std::vector<std::uint8_t>& out,
                std::size_t segment_hint) const override;

  [[nodiscard]] bool decompress(std::span<const std::uint8_t> in,
                                std::span<std::uint8_t> out) const noexcept override;

 private:
  unsigned max_chain_;
  bool adaptive_search_;
};

}  // namespace turbocore::detail

#endif  // TURBOCORE_DETAIL_CODEC_HPP
