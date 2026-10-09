# Per-target compiler configuration for TurboCore's own targets.
#
# Options are attached per target (never globally) so that fetched third-party
# code is not compiled with our warning set, and are restricted to C++ so that
# nvcc never receives host-compiler-only flags.

set(_turbocore_gnu_warnings
  -Wall
  -Wextra
  -Wpedantic
  -Wshadow
  -Wconversion
  -Wsign-conversion
  -Wdouble-promotion
  -Wold-style-cast
  -Wnon-virtual-dtor
  -Woverloaded-virtual
  -Wformat=2
  -Wundef
)
set(_turbocore_msvc_warnings /W4 /permissive- /utf-8)

function(turbocore_configure_target target)
  if(MSVC)
    set(_warnings ${_turbocore_msvc_warnings})
    set(_werror /WX)
    # /fp:precise forbids contraction and value-unsafe reassociation.
    set(_fp_flags /fp:precise)
  else()
    set(_warnings ${_turbocore_gnu_warnings})
    set(_werror -Werror)
    # Floating-point results must be bit-identical across backends and CPU
    # architectures, so the compiler may not fuse a*b+c into an FMA on its own.
    set(_fp_flags -ffp-contract=off)
  endif()

  if(TURBOCORE_WARNINGS_AS_ERRORS)
    list(APPEND _warnings ${_werror})
  endif()

  target_compile_options(${target} PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${_warnings};${_fp_flags}>"
  )
endfunction()
