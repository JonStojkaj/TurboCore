#ifndef TURBOCORE_BACKENDS_CUDA_PROBE_CUDA_HPP
#define TURBOCORE_BACKENDS_CUDA_PROBE_CUDA_HPP

// Headers in this directory are included by .cu files and must stay valid
// under TURBOCORE_CUDA_STANDARD (C++17 by default).

namespace turbocore::detail::cuda {

/// Phase 0 toolchain probe: number of CUDA devices visible to the process,
/// or 0 if the runtime cannot be initialised (no driver, no GPU).
int device_count() noexcept;

}  // namespace turbocore::detail::cuda

#endif  // TURBOCORE_BACKENDS_CUDA_PROBE_CUDA_HPP
