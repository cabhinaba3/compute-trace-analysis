#include <cstdlib>
#include <stdexcept>

#include "gpu/backend.hpp"

namespace cta::gpu {

std::vector<std::string> available_backends() {
  std::vector<std::string> found;
#if defined(CTA_HAVE_CUDA)
  if (cuda_present()) found.emplace_back("cuda");
#endif
#if defined(CTA_HAVE_METAL)
  if (metal_present()) found.emplace_back("metal");
#endif
  return found;
}

analysis::WhatIfResult solve_on(const std::string& backend, const SolveInput& input, std::uint64_t memory_bytes) {
  if (backend != "cuda" && backend != "metal") throw std::invalid_argument("unknown device " + backend);
  // Test hook: makes every GPU solve fail, to exercise the CPU fallback.
  if (const char* fault = std::getenv("CTA_GPU_FAULT"); fault != nullptr && fault[0] != '\0' && fault[0] != '0')
    throw std::runtime_error(backend + " solve failed: injected fault (CTA_GPU_FAULT)");
#if defined(CTA_HAVE_CUDA)
  if (backend == "cuda") {
    if (!cuda_present()) throw std::runtime_error("cuda is not available: no CUDA device found");
    return cuda_solve(input, memory_bytes);
  }
#endif
#if defined(CTA_HAVE_METAL)
  if (backend == "metal") {
    if (!metal_present()) throw std::runtime_error("metal is not available: no Metal device found");
    return metal_solve(input, memory_bytes);
  }
#endif
  (void)input;
  (void)memory_bytes;
  throw std::runtime_error(backend + " is not available: this build has no " + backend + " support");
}

}  // namespace cta::gpu
