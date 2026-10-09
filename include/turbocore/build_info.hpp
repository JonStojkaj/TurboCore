#ifndef TURBOCORE_BUILD_INFO_HPP
#define TURBOCORE_BUILD_INFO_HPP

#include <string_view>

#include "turbocore/config.hpp"
#include "turbocore/export.hpp"

namespace turbocore {

/// What this build of the library was compiled with.
struct BuildInfo {
  std::string_view version;
  bool avx2_compiled;
  bool cuda_compiled;
};

/// What the machine the library is currently running on can actually execute.
/// A backend is only usable if it was compiled in AND the hardware supports it.
struct RuntimeSupport {
  bool avx2_usable;
  int cuda_device_count;
};

[[nodiscard]] TURBOCORE_EXPORT BuildInfo build_info() noexcept;
[[nodiscard]] TURBOCORE_EXPORT RuntimeSupport runtime_support() noexcept;

}  // namespace turbocore

#endif  // TURBOCORE_BUILD_INFO_HPP
