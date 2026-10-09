turbocore_require_googletest()
include(GoogleTest)

add_executable(turbocore_tests
  test_build_info.cpp
)
target_link_libraries(turbocore_tests PRIVATE TurboCore::turbocore GTest::gtest_main)
turbocore_configure_target(turbocore_tests)

# PRE_TEST defers test discovery to `ctest`, so the build itself never has to
# execute the test binary (cross builds, GPU-less build nodes).
gtest_discover_tests(turbocore_tests
  DISCOVERY_MODE PRE_TEST
  PROPERTIES LABELS unit
)
