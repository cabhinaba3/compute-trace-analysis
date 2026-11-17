// The cta._native extension module. Everything it exposes is registered by the
// functions in bindings/, one per area.
#include <nanobind/nanobind.h>

#include "bindings/bindings.hpp"

NB_MODULE(_native, m) {
  m.doc() = "Native code for compute-trace-analysis: trace readers and synthetic trace generators.";
  cta::bindings::bind_synth(m);
  cta::bindings::bind_readers(m);
  cta::bindings::bind_analysis(m);
}
