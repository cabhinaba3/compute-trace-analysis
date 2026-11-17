// Registration of the Python-visible functions, one function per area. Names,
// argument names and defaults are the Python API: cta/_native.pyi and the
// cta.synth wrappers depend on them.
#pragma once

#include <nanobind/nanobind.h>

namespace cta::bindings {

void bind_synth(nanobind::module_& m);    // synthetic trace generators
void bind_readers(nanobind::module_& m);  // trace readers
void bind_analysis(nanobind::module_& m);  // DAG solver and interval helpers
void bind_sim(nanobind::module_& m);       // the simulator

}  // namespace cta::bindings
