#include "backends/cuda/probe_cuda.hpp"

#include <cuda_runtime.h>

namespace turbocore::detail::cuda {

int device_count() noexcept {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    return 0;
  }
  return count;
}

}  // namespace turbocore::detail::cuda
