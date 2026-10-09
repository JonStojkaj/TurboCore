find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
turbocore_require_pybind11()

# The extension is the private `turbocore._core` module; the public package is
# python/turbocore/. Both are assembled under <build>/python so that
# PYTHONPATH=<build>/python gives an importable package without installing.
set(_turbocore_py_package_dir "${PROJECT_BINARY_DIR}/python/turbocore")

pybind11_add_module(turbocore_python MODULE src/bindings.cpp)
target_link_libraries(turbocore_python PRIVATE TurboCore::turbocore)
set_target_properties(turbocore_python PROPERTIES
  OUTPUT_NAME _core
  LIBRARY_OUTPUT_DIRECTORY "${_turbocore_py_package_dir}"
)
foreach(_config IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set_target_properties(turbocore_python PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_${_config} "${_turbocore_py_package_dir}"
  )
endforeach()

add_custom_command(TARGET turbocore_python POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/turbocore/__init__.py"
    "${_turbocore_py_package_dir}/__init__.py"
  VERBATIM
)

if(TURBOCORE_BUILD_TESTS)
  add_test(NAME python.smoke
    COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tests/smoke_test.py"
  )
  set_tests_properties(python.smoke PROPERTIES
    ENVIRONMENT "PYTHONPATH=${PROJECT_BINARY_DIR}/python"
    LABELS python
  )
endif()
