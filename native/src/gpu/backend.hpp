// Optional GPU backends of the what-if scenario solver.
//
// Scenarios are independent, so a GPU gives each one a thread that walks the graph in
// topological order, with the same rules as the CPU solver (analysis/whatif.hpp): the later
// finish wins, then the greater depth, then the lowest vertex id. CUDA solves in 64-bit floats
// with no multiply-add fusion and agrees with the CPU to rounding; Metal has no 64-bit floats and
// solves in 32-bit ones, so it is approximate (about 1e-5 relative).
//
// A backend exists only if it was compiled in (CMake option CTA_GPU) and a device answers.
// Anything that goes wrong (no device, not enough memory, a driver error) throws
// std::runtime_error, and the caller solves on the CPU and keeps the message as a note.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "analysis/whatif.hpp"

namespace cta::gpu {

// What a backend needs: the graph in position order and the per-vertex columns by original id.
struct SolveInput {
  const analysis::DagPlan* plan = nullptr;
  const analysis::PreparedInputs* prepared = nullptr;
  const bool* is_comm = nullptr;
  const bool* is_compute = nullptr;
  const std::int64_t* size_bytes = nullptr;
  std::size_t num_groups = 0;
  const double* scale = nullptr;  // num_scenarios x num_groups, row-major
  std::size_t num_scenarios = 0;
};

// The backends compiled in whose device is present: "cuda" and/or "metal".
std::vector<std::string> available_backends();

// Solves on `backend` within `memory_bytes` of device memory (0 = what the device has free,
// with headroom). Throws std::invalid_argument for a name that is not a backend, and
// std::runtime_error with the reason when the backend cannot do it.
analysis::WhatIfResult solve_on(const std::string& backend, const SolveInput& input, std::uint64_t memory_bytes);

#if defined(CTA_HAVE_CUDA)
bool cuda_present();
analysis::WhatIfResult cuda_solve(const SolveInput& input, std::uint64_t memory_bytes);
#endif
#if defined(CTA_HAVE_METAL)
bool metal_present();
analysis::WhatIfResult metal_solve(const SolveInput& input, std::uint64_t memory_bytes);
#endif

}  // namespace cta::gpu
