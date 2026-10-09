#include <gtest/gtest.h>

#include <cstdlib>
#include <string_view>

#include "turbocore/turbocore.hpp"

namespace {

TEST(BuildInfo, VersionMatchesHeaderMacros) {
  const turbocore::BuildInfo info = turbocore::build_info();
  EXPECT_EQ(info.version, TURBOCORE_VERSION_STRING);
  EXPECT_FALSE(info.version.empty());
}

// Catches a consumer compiling against headers from a different build of the
// library than the one it links.
TEST(BuildInfo, CompiledBackendsMatchHeaderMacros) {
  const turbocore::BuildInfo info = turbocore::build_info();
  EXPECT_EQ(info.avx2_compiled, TURBOCORE_HAS_AVX2 != 0);
  EXPECT_EQ(info.cuda_compiled, TURBOCORE_HAS_CUDA != 0);
}

TEST(RuntimeSupport, UsableBackendsAreASubsetOfCompiledBackends) {
  const turbocore::BuildInfo info = turbocore::build_info();
  const turbocore::RuntimeSupport support = turbocore::runtime_support();
  if (!info.avx2_compiled) {
    EXPECT_FALSE(support.avx2_usable);
  }
  if (!info.cuda_compiled) {
    EXPECT_EQ(support.cuda_device_count, 0);
  }
  EXPECT_GE(support.cuda_device_count, 0);
}

// A build without AVX2, or a pre-Haswell CPU, is legitimate, so by default an
// unusable backend is a visible skip. Environments that exist to exercise the
// backend set TURBOCORE_REQUIRE_AVX2=1 to turn that skip into a failure, so
// they can never go green while silently testing nothing.
TEST(RuntimeSupport, Avx2BackendRunsWhenCompiled) {
  const char* require = std::getenv("TURBOCORE_REQUIRE_AVX2");
  const bool required = require != nullptr && std::string_view(require) == "1";
  const bool usable = turbocore::runtime_support().avx2_usable;
  if (required) {
    ASSERT_TRUE(turbocore::build_info().avx2_compiled);
    ASSERT_TRUE(usable);
    return;
  }
  if (!turbocore::build_info().avx2_compiled) {
    GTEST_SKIP() << "AVX2 backend not compiled into this build";
  }
  if (!usable) {
    GTEST_SKIP() << "AVX2 backend compiled but not usable on this CPU";
  }
}

}  // namespace
