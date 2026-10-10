#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "turbocore/turbocore.hpp"

namespace py = pybind11;

namespace {

// ---------------------------------------------------------------------------
// Argument conversion. Bad values raise ValueError / TypeError naming the
// accepted choices, before any work is done.
// ---------------------------------------------------------------------------

turbocore::Backend parse_backend(const std::string& name) {
  if (name == "auto") return turbocore::Backend::Auto;
  if (name == "scalar") return turbocore::Backend::Scalar;
  if (name == "avx2") return turbocore::Backend::Avx2;
  if (name == "cuda") return turbocore::Backend::Cuda;
  throw py::value_error("backend must be 'auto', 'scalar', 'avx2' or 'cuda', not '" + name + "'");
}

turbocore::Predictor parse_predictor(const std::string& name) {
  if (name == "none") return turbocore::Predictor::None;
  if (name == "delta") return turbocore::Predictor::Delta;
  if (name == "xor") return turbocore::Predictor::Xor;
  throw py::value_error("predictor must be 'none', 'delta' or 'xor', not '" + name + "'");
}

turbocore::QuantFormat parse_format(const std::string& name) {
  if (name == "float16" || name == "fp16") return turbocore::QuantFormat::Float16;
  if (name == "bfloat16" || name == "bf16") return turbocore::QuantFormat::BFloat16;
  throw py::value_error("quantization format must be 'float16' or 'bfloat16', not '" + name + "'");
}

std::optional<turbocore::QuantFormat> parse_optional_format(const py::object& value) {
  if (value.is_none()) {
    return std::nullopt;
  }
  return parse_format(value.cast<std::string>());
}

const char* format_name(turbocore::QuantFormat format) {
  return format == turbocore::QuantFormat::Float16 ? "float16" : "bfloat16";
}

const char* predictor_name(turbocore::Predictor predictor) {
  switch (predictor) {
    case turbocore::Predictor::None:
      return "none";
    case turbocore::Predictor::Delta:
      return "delta";
    case turbocore::Predictor::Xor:
      return "xor";
  }
  return "unknown";
}

unsigned parse_threads(int threads) {
  if (threads < 0) {
    throw py::value_error("threads must be >= 0 (0 means one per hardware thread)");
  }
  return static_cast<unsigned>(threads);
}

/// `peak`: "max_abs", "range", or a positive number.
turbocore::ErrorAnalysisOptions parse_peak(const py::object& peak) {
  if (py::isinstance<py::str>(peak)) {
    const auto name = peak.cast<std::string>();
    if (name == "max_abs") return {.peak_mode = turbocore::PeakMode::MaxAbsReference};
    if (name == "range") return {.peak_mode = turbocore::PeakMode::ReferenceRange};
    throw py::value_error("peak must be 'max_abs', 'range' or a positive number, not '" + name + "'");
  }
  return {.peak_mode = turbocore::PeakMode::Custom, .custom_peak = peak.cast<double>()};
}

/// A contiguous float32 view of `object`, flattened. Only float32 arrays are
/// accepted: silently narrowing float64 would make a "lossless" compressor
/// lose data. A non-contiguous float32 array is copied, which changes
/// nothing about its values.
py::array_t<float, py::array::c_style> as_float32(const py::object& object, const char* argument) {
  if (!py::isinstance<py::array>(object)) {
    throw py::type_error(std::string(argument) + " must be a NumPy array of dtype float32");
  }
  const auto array = py::reinterpret_borrow<py::array>(object);
  if (!array.dtype().is(py::dtype::of<float>())) {
    throw py::type_error(std::string(argument) + " must have dtype float32, not " +
                         py::str(array.dtype()).cast<std::string>() +
                         "; convert explicitly with .astype(numpy.float32)");
  }
  return py::array_t<float, py::array::c_style>::ensure(array);
}

std::span<const float> span_of(const py::array_t<float, py::array::c_style>& array) {
  return {array.data(), static_cast<std::size_t>(array.size())};
}

/// The bytes of any object supporting the buffer protocol with 1-byte items
/// (bytes, bytearray, memoryview, uint8 arrays). The returned info keeps the
/// buffer alive and locked.
py::buffer_info byte_buffer(const py::buffer& buffer) {
  py::buffer_info info = buffer.request();
  if (info.itemsize != 1) {
    throw py::type_error("compressed data must be a bytes-like object");
  }
  return info;
}

std::span<const std::uint8_t> span_of(const py::buffer_info& info) {
  return {static_cast<const std::uint8_t*>(info.ptr), static_cast<std::size_t>(info.size)};
}

py::array_t<float> to_array(const std::vector<float>& values) {
  py::array_t<float> array(static_cast<py::ssize_t>(values.size()));
  if (!values.empty()) {
    std::memcpy(array.mutable_data(), values.data(), values.size() * sizeof(float));
  }
  return array;
}

// ---------------------------------------------------------------------------
// Reports as plain dicts: easy to print, log and turn into a DataFrame row.
// ---------------------------------------------------------------------------

py::dict to_dict(const turbocore::ErrorReport& r) {
  py::dict d;
  d["element_count"] = r.element_count;
  d["compared_count"] = r.compared_count;
  d["bit_exact_count"] = r.bit_exact_count;
  d["nan_reference_count"] = r.nan_reference_count;
  d["nan_preserved_count"] = r.nan_preserved_count;
  d["nan_introduced_count"] = r.nan_introduced_count;
  d["infinity_reference_count"] = r.infinity_reference_count;
  d["infinity_preserved_count"] = r.infinity_preserved_count;
  d["overflow_to_infinity_count"] = r.overflow_to_infinity_count;
  d["underflow_to_zero_count"] = r.underflow_to_zero_count;
  d["sign_mismatch_count"] = r.sign_mismatch_count;
  d["mean_squared_error"] = r.mean_squared_error;
  d["root_mean_squared_error"] = r.root_mean_squared_error;
  d["mean_absolute_error"] = r.mean_absolute_error;
  d["max_absolute_error"] = r.max_absolute_error;
  d["max_relative_error"] = r.max_relative_error;
  d["peak"] = r.peak;
  d["psnr_db"] = r.psnr_db;
  d["max_ulp_distance"] = r.max_ulp_distance;
  d["mean_ulp_distance"] = r.mean_ulp_distance;
  return d;
}

py::dict to_dict(const turbocore::QuantizationReport& r) {
  py::dict d;
  d["format"] = format_name(r.format);
  d["error"] = to_dict(r.error);
  d["max_target_ulp_error"] = r.max_target_ulp_error;
  d["mean_target_ulp_error"] = r.mean_target_ulp_error;
  d["above_half_ulp_count"] = r.above_half_ulp_count;
  d["exactly_representable_count"] = r.exactly_representable_count;
  py::list histogram;
  for (const std::uint64_t bin : r.target_ulp_histogram) {
    histogram.append(bin);
  }
  d["target_ulp_histogram"] = histogram;
  return d;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "TurboCore native extension";
  m.attr("__version__") = TURBOCORE_VERSION_STRING;

  // turbocore.TurboCoreError for stream and backend failures; bad arguments
  // become the ValueError a Python caller expects.
  static py::exception<turbocore::Error> turbocore_error(m, "TurboCoreError", PyExc_RuntimeError);
  py::register_exception_translator([](std::exception_ptr pointer) {
    try {
      if (pointer) {
        std::rethrow_exception(pointer);
      }
    } catch (const turbocore::Error& error) {
      if (error.code() == turbocore::ErrorCode::InvalidArgument) {
        PyErr_SetString(PyExc_ValueError, error.what());
      } else {
        PyErr_SetString(turbocore_error.ptr(), error.what());
      }
    }
  });

  m.def(
      "compress",
      [](const py::object& data, const py::object& quantization, const std::string& predictor, bool byte_shuffle,
         int level, std::size_t block_elements, const std::string& backend, int threads) {
        const auto array = as_float32(data, "data");
        const turbocore::Compressor compressor({
            .quantization = parse_optional_format(quantization),
            .predictor = parse_predictor(predictor),
            .byte_shuffle = byte_shuffle,
            .level = level,
            .block_elements = block_elements,
            .backend = parse_backend(backend),
            .threads = parse_threads(threads),
        });
        std::vector<std::uint8_t> stream;
        {
          // `array` keeps the input alive; no Python objects are touched here.
          const py::gil_scoped_release release;
          stream = compressor.compress(span_of(array));
        }
        return py::bytes(reinterpret_cast<const char*>(stream.data()), stream.size());
      },
      py::arg("data"), py::kw_only(), py::arg("quantization") = py::none(), py::arg("predictor") = "delta",
      py::arg("byte_shuffle") = true, py::arg("level") = 6, py::arg("block_elements") = 65536,
      py::arg("backend") = "auto", py::arg("threads") = 0);

  m.def(
      "decompress",
      [](const py::buffer& data, const std::string& backend, int threads) {
        const py::buffer_info info = byte_buffer(data);
        const turbocore::Decompressor decompressor(
            {.backend = parse_backend(backend), .threads = parse_threads(threads)});
        std::vector<float> values;
        {
          const py::gil_scoped_release release;
          values = decompressor.decompress(span_of(info));
        }
        return to_array(values);
      },
      py::arg("data"), py::kw_only(), py::arg("backend") = "auto", py::arg("threads") = 0);

  m.def(
      "inspect",
      [](const py::buffer& data) {
        const py::buffer_info buffer = byte_buffer(data);
        const turbocore::StreamInfo info = turbocore::Decompressor::inspect(span_of(buffer));
        py::dict d;
        d["element_count"] = info.element_count;
        d["block_count"] = info.block_count;
        d["block_elements"] = info.block_elements;
        d["predictor"] = predictor_name(info.predictor);
        d["byte_shuffle"] = info.byte_shuffle;
        if (info.quantization.has_value()) {
          d["quantization"] = format_name(*info.quantization);
        } else {
          d["quantization"] = py::none();
        }
        d["compressed_bytes"] = buffer.size;
        return d;
      },
      py::arg("data"));

  m.def(
      "quantize_round_trip",
      [](const py::object& data, const std::string& format, const std::string& backend) {
        const auto array = as_float32(data, "data");
        const turbocore::QuantFormat quant_format = parse_format(format);
        const turbocore::Backend selected = parse_backend(backend);
        std::vector<float> values;
        {
          const py::gil_scoped_release release;
          values = turbocore::quantize_round_trip(span_of(array), quant_format, selected);
        }
        return to_array(values);
      },
      py::arg("data"), py::arg("format"), py::kw_only(), py::arg("backend") = "auto");

  m.def(
      "analyze_error",
      [](const py::object& reference, const py::object& reconstructed, const py::object& peak) {
        const auto ref = as_float32(reference, "reference");
        const auto rec = as_float32(reconstructed, "reconstructed");
        const turbocore::ErrorAnalysisOptions options = parse_peak(peak);
        turbocore::ErrorReport report;
        {
          const py::gil_scoped_release release;
          report = turbocore::analyze_error(span_of(ref), span_of(rec), options);
        }
        return to_dict(report);
      },
      py::arg("reference"), py::arg("reconstructed"), py::kw_only(), py::arg("peak") = "max_abs");

  m.def(
      "analyze_quantization",
      [](const py::object& data, const std::string& format, const py::object& peak) {
        const auto array = as_float32(data, "data");
        const turbocore::QuantFormat quant_format = parse_format(format);
        const turbocore::ErrorAnalysisOptions options = parse_peak(peak);
        turbocore::QuantizationReport report;
        {
          const py::gil_scoped_release release;
          report = turbocore::analyze_quantization(span_of(array), quant_format, options);
        }
        return to_dict(report);
      },
      py::arg("data"), py::arg("format"), py::kw_only(), py::arg("peak") = "max_abs");

  m.def("build_info", [] {
    const turbocore::BuildInfo info = turbocore::build_info();
    py::dict d;
    d["version"] = std::string(info.version);
    d["avx2_compiled"] = info.avx2_compiled;
    d["cuda_compiled"] = info.cuda_compiled;
    return d;
  });

  m.def("runtime_support", [] {
    const turbocore::RuntimeSupport support = turbocore::runtime_support();
    py::dict d;
    d["avx2_usable"] = support.avx2_usable;
    d["cuda_device_count"] = support.cuda_device_count;
    d["best_backend"] = turbocore::backend_name(turbocore::best_available_backend());
    return d;
  });

  m.def(
      "is_backend_available",
      [](const std::string& backend) { return turbocore::is_backend_available(parse_backend(backend)); },
      py::arg("backend"));
}
