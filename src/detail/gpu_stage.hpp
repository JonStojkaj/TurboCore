#ifndef TURBOCORE_DETAIL_GPU_STAGE_HPP
#define TURBOCORE_DETAIL_GPU_STAGE_HPP

#include "turbocore/config.hpp"

// Host-side glue for the GPU transform stage. Everything here is compiled
// only in a CUDA build.

#if TURBOCORE_HAS_CUDA

#include <cstddef>
#include <optional>

#include "backends/cuda/transform_stage.hpp"
#include "turbocore/compressor.hpp"
#include "turbocore/error.hpp"

namespace turbocore::detail {

[[nodiscard]] inline cuda::StageConfig make_stage_config(const std::optional<QuantFormat>& quantization,
                                                         Predictor predictor, bool byte_shuffle,
                                                         std::size_t block_elements) noexcept {
  cuda::Element element = cuda::Element::Float32;
  if (quantization.has_value()) {
    element = *quantization == QuantFormat::Float16 ? cuda::Element::Float16 : cuda::Element::BFloat16;
  }
  return cuda::StageConfig{
      .element = element,
      .predictor = static_cast<cuda::PredictorOp>(static_cast<int>(predictor)),
      .byte_shuffle = byte_shuffle,
      .block_elements = block_elements,
  };
}

inline void require_ok(const cuda::Status& status) {
  if (!status.ok) {
    throw Error(ErrorCode::BackendFailure, status.message);
  }
}

}  // namespace turbocore::detail

#endif  // TURBOCORE_HAS_CUDA

#endif  // TURBOCORE_DETAIL_GPU_STAGE_HPP
