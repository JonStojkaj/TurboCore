turbocore_require_benchmark()

add_executable(turbocore_bench
  bench_codec.cpp
  bench_memory_baseline.cpp
)
target_link_libraries(turbocore_bench PRIVATE TurboCore::turbocore benchmark::benchmark_main)
turbocore_configure_target(turbocore_bench)

if(CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE MATCHES "^(Release|RelWithDebInfo)$")
  message(WARNING
    "TurboCore benchmarks are being built with CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}; "
    "throughput numbers from a non-optimised build are meaningless.")
endif()
