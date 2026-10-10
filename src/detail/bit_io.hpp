#ifndef TURBOCORE_DETAIL_BIT_IO_HPP
#define TURBOCORE_DETAIL_BIT_IO_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace turbocore::detail {

/// LSB-first bit writer (the DEFLATE bit order) appending to a byte vector.
class BitWriter {
 public:
  explicit BitWriter(std::vector<std::uint8_t>& out) noexcept : out_(out) {}

  /// Appends the low `count` bits of `value`. Requires count <= 32 and
  /// value < 2^count.
  void write(std::uint32_t value, unsigned count) {
    accumulator_ |= static_cast<std::uint64_t>(value) << fill_;
    fill_ += count;
    while (fill_ >= 8) {
      out_.push_back(static_cast<std::uint8_t>(accumulator_ & 0xFFu));
      accumulator_ >>= 8;
      fill_ -= 8;
    }
  }

  /// Pads the final partial byte with zero bits.
  void flush() {
    if (fill_ > 0) {
      out_.push_back(static_cast<std::uint8_t>(accumulator_ & 0xFFu));
      accumulator_ = 0;
      fill_ = 0;
    }
  }

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
