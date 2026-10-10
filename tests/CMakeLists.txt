turbocore_require_googletest()
include(GoogleTest)

# --- Public API tests: link only what a consumer would. ---------------------
add_executable(turbocore_tests
  test_build_info.cpp
  test_roundtrip.cpp
  test_corruption.cpp
)
target_link_libraries(turbocore_tests PRIVATE TurboCore::turbocore GTest::gtest_main)
turbocore_configure_target(turbocore_tests)

# PRE_TEST defers test discovery to `ctest`, so the build itself never has to
# execute the test binary (cross builds, GPU-less build nodes).
gtest_discover_tests(turbocore_tests
  DISCOVERY_MODE PRE_TEST
  PROPERTIES LABELS unit
)

# --- Internal tests: reach into src/detail. ---------------------------------
# Internal symbols have hidden visibility, so they are only linkable from a
# static build of the library.
if(NOT BUILD_SHARED_LIBS)
  add_executable(turbocore_internal_tests
    internal/test_container.cpp
    internal/test_crc32c.cpp
    internal/test_deflate.cpp
    internal/test_huffman.cpp
    internal/test_transforms.cpp
  )
  target_link_libraries(turbocore_internal_tests PRIVATE TurboCore::turbocore GTest::gtest_main)
  target_include_directories(turbocore_internal_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
  turbocore_configure_target(turbocore_internal_tests)
  gtest_discover_tests(turbocore_internal_tests
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS unit
  )

  # Independent oracle: every stream our encoder writes must be readable by
  # zlib, and our decoder must read what zlib writes. zlib is used here by the
  # test only (through the Python standard library); the library itself never
  # links a compression library.
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND AND NOT WIN32)
    add_executable(turbocore_deflate_cli tools/deflate_cli.cpp)
    target_link_libraries(turbocore_deflate_cli PRIVATE TurboCore::turbocore)
    target_include_directories(turbocore_deflate_cli PRIVATE "${PROJECT_SOURCE_DIR}/src")
    turbocore_configure_target(turbocore_deflate_cli)

    add_test(NAME interop.zlib
      COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/zlib_interop.py"
              $<TARGET_FILE:turbocore_deflate_cli>
    )
    set_tests_properties(interop.zlib PROPERTIES LABELS interop)
  else()
    message(STATUS "TurboCore: zlib interop test disabled (needs a Python 3 interpreter on a POSIX host)")
  endif()
endif()
