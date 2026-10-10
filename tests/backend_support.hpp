#ifndef TURBOCORE_TESTS_BACKEND_SUPPORT_HPP
#define TURBOCORE_TESTS_BACKEND_SUPPORT_HPP

#include <gtest/gtest.h>

#include <cstdlib>
#include <string_view>

#include "turbocore/backend.hpp"

namespace turbocore::testing {

/// True in environments that exist to exercise the AVX2 backend (they set
/// TURBOCORE_REQUIRE_AVX2=1), where a missing backend must fail, not skip.
inline bool avx2_required() {
  const char* value = std::getenv("TURBOCORE_REQUIRE_AVX2");
  return value != nullptr && std::string_view(value) == "1";
}

}  // namespace turbocore::testing

/// Leaves the current test unless the AVX2 backend can run: as a failure
/// where the backend is required, as a visible skip everywhere else.
#define TURBOCORE_NEEDS_AVX2()                                                             \
  do {                                                                                     \
    if (!::turbocore::is_backend_available(::turbocore::Backend::Avx2)) {                  \
      if (::turbocore::testing::avx2_required()) {                                         \
        FAIL() << "TURBOCORE_REQUIRE_AVX2=1 but the AVX2 backend is not available";        \
      }                                                                                    \
      GTEST_SKIP() << "AVX2 backend not available on this machine";                        \
    }                                                                                      \
  } while (false)

/// Same for the CUDA backend (TURBOCORE_REQUIRE_CUDA=1).
#define TURBOCORE_NEEDS_CUDA()                                                             \
  do {                                                                                     \
    if (!::turbocore::is_backend_available(::turbocore::Backend::Cuda)) {                  \
      const char* turbocore_required_ = std::getenv("TURBOCORE_REQUIRE_CUDA");             \
      if (turbocore_required_ != nullptr && std::string_view(turbocore_required_) == "1") { \
        FAIL() << "TURBOCORE_REQUIRE_CUDA=1 but the CUDA backend is not available";        \
      }                                                                                    \
      GTEST_SKIP() << "CUDA backend not available on this machine";                        \
    }                                                                                      \
  } while (false)

#endif  // TURBOCORE_TESTS_BACKEND_SUPPORT_HPP
