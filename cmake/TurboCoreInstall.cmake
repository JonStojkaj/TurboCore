# Install rules and the find_package(TurboCore) config package.

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_turbocore_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/TurboCore")

install(TARGETS turbocore
  EXPORT TurboCoreTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT TurboCoreTargets
  NAMESPACE TurboCore::
  DESTINATION ${_turbocore_cmake_dir}
)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/TurboCoreConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/TurboCoreConfig.cmake"
  INSTALL_DESTINATION ${_turbocore_cmake_dir}
)
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/TurboCoreConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMinorVersion
)
install(FILES
  "${PROJECT_BINARY_DIR}/TurboCoreConfig.cmake"
  "${PROJECT_BINARY_DIR}/TurboCoreConfigVersion.cmake"
  DESTINATION ${_turbocore_cmake_dir}
)
