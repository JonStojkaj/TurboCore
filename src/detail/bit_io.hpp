#ifndef TURBOCORE_DETAIL_BIT_IO_HPP
#define TURBOCORE_DETAIL_BIT_IO_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace turbocore::detail {

/// LSB-first bit writer (the DEFLATE bit order) appending to a byte vector.
/// Bits are collected in a 64-bit accumulator and emitted four bytes at a
/// time, so the cost per symbol is a shift and an add, not a vector append.
class BitWriter {
 public:
  explicit BitWriter(std::vector<std::uint8_t>& out) noexcept : out_(out) {}
  BitWriter(const BitWriter&) = delete;
  BitWriter& operator=(const BitWriter&) = delete;

  /// Appends the low `count` bits of `value`. Requires count <= 32 and
  /// value < 2^count.
  void write(std::uint32_t value, unsigned count) {
    accumulator_ |= static_cast<std::uint64_t>(value) << fill_;
    fill_ += count;
    if (fill_ >= 32) {
      const std::size_t size = out_.size();
      out_.resize(size + 4);
      std::uint8_t* p = out_.data() + size;
      p[0] = static_cast<std::uint8_t>(accumulator_ & 0xFFu);
      p[1] = static_cast<std::uint8_t>((accumulator_ >> 8) & 0xFFu);
      p[2] = static_cast<std::uint8_t>((accumulator_ >> 16) & 0xFFu);
      p[3] = static_cast<std::uint8_t>((accumulator_ >> 24) & 0xFFu);
      accumulator_ >>= 32;
      fill_ -= 32;
    }
  }

  /// Bits written since the last byte boundary (0 to 7).
  [[nodiscard]] unsigned pending_bits() const noexcept { return fill_ % 8; }

  /// Pads with zero bits up to the next byte boundary and emits everything.
  void align_to_byte() {
    while (fill_ > 0) {
      out_.push_back(static_cast<std::uint8_t>(accumulator_ & 0xFFu));
      accumulator_ >>= 8;
      fill_ = fill_ > 8 ? fill_ - 8 : 0;
    }
  }

  /// Appends whole bytes. Requires a preceding align_to_byte().
  void append_bytes(std::span<const std::uint8_t> bytes) { out_.insert(out_.end(), bytes.begin(), bytes.end()); }

  /// Pads the final partial byte with zero bits. Must be called once at the end.
  void flush() { align_to_byte(); }

 private:
  std::vector<std::uint8_t>& out_;
  std::uint64_t accumulator_ = 0;
  unsigned fill_ = 0;
};

/// LSB-first bit reader over untrusted input. Reading past the end never
/// touches memory outside the span: it yields zero bits and latches
/// overrun(), which callers check before trusting what they decoded.
class BitReader {
 public:
  explicit BitReader(std::span<const std::uint8_t> in) noexcept : in_(in) {}

  /// Next `count` bits (count <= 32) without consuming them.
  [[nodiscard]] std::uint32_t peek(unsigned count) noexcept {
    refill();
    return static_cast<std::uint32_t>(accumulator_ & ((std::uint64_t{1} << count) - 1));
  }

  void consume(unsigned count) noexcept {
    if (count > fill_) {
      overrun_ = true;
      accumulator_ = 0;
      fill_ = 0;
      return;
    }
    accumulator_ >>= count;
    fill_ -= count;
  }

  [[nodiscard]] std::uint32_t read(unsigned count) noexcept {
    const std::uint32_t value = peek(count);
    consume(count);
    return value;
  }

  /// Discards bits up to the next byte boundary of the input.
  void align_to_byte() noexcept { consume(fill_ % 8); }

  [[nodiscard]] bool overrun() const noexcept { return overrun_; }

  /// Input bytes touched so far, counting a partially consumed byte as used.
  [[nodiscard]] std::size_t bytes_consumed() const noexcept { return position_ - fill_ / 8; }

 private:
  void refill() noexcept {
    while (fill_ <= 56 && position_ < in_.size()) {
      accumulator_ |= static_cast<std::uint64_t>(in_[position_++]) << fill_;
      fill_ += 8;
    }
  }

  std::span<const std::uint8_t> in_;
  std::size_t position_ = 0;
  std::uint64_t accumulator_ = 0;
  unsigned fill_ = 0;
  bool overrun_ = false;
};

}  // namespace turbocore::detail

#endif  // TURBOCORE_DETAIL_BIT_IO_HPP
