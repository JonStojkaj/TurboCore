#include <pybind11/pybind11.h>

#include <string>

#include "turbocore/turbocore.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() = "TurboCore native extension";

  m.attr("__version__") = TURBOCORE_VERSION_STRING;

  m.def(
      "build_info",
      [] {
        const turbocore::BuildInfo info = turbocore::build_info();
        py::dict result;
        result["version"] = std::string(info.version);
        result["avx2_compiled"] = info.avx2_compiled;
        result["cuda_compiled"] = info.cuda_compiled;
        return result;
      },
      "Backends compiled into this build of TurboCore.");

  m.def(
      "runtime_support",
      [] {
        const turbocore::RuntimeSupport support = turbocore::runtime_support();
        py::dict result;
        result["avx2_usable"] = support.avx2_usable;
        result["cuda_device_count"] = support.cuda_device_count;
        return result;
      },
      "Backends that can run on the current machine.");
}
