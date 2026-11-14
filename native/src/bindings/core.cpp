#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <stdexcept>

#include "bindings/bindings.hpp"
#include "bindings/numpy.hpp"
#include "core/intervals.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace cta::bindings {

void bind_core(nb::module_& m) {
  m.def(
      "nesting_parents",
      [](nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> starts,
         nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> ends) {
        if (starts.shape(0) != ends.shape(0))
          throw std::invalid_argument("starts and ends must have the same length");
        return to_numpy(cta::nesting_parents(starts.data(), ends.data(), starts.shape(0)));
      },
      "starts"_a, "ends"_a,
      "Enclosing range of each range, for ranges sorted by start (-1 for none).");
}

}  // namespace cta::bindings
