#ifndef TURBOCORE_ERROR_HPP
#define TURBOCORE_ERROR_HPP

#include <stdexcept>
#include <string>

#include "turbocore/export.hpp"

namespace turbocore {

enum class ErrorCode {
  InvalidArgument,    ///< The caller passed options or data the library cannot accept.
  UnsupportedFormat,  ///< Not a TurboCore stream, or a version this build cannot read.
  CorruptStream,      ///< The stream is structurally invalid or truncated.
  ChecksumMismatch,   ///< The stream decoded, but the data fails its CRC-32C.
};

/// The single exception type thrown across the public API.
class TURBOCORE_EXPORT Error : public std::runtime_error {
 public:
  Error(ErrorCode code, const std::string& message) : std::runtime_error(message), code_(code) {}

  [[nodiscard]] ErrorCode code() const noexcept { return code_; }

 private:
  ErrorCode code_;
};

}  // namespace turbocore

#endif  // TURBOCORE_ERROR_HPP
