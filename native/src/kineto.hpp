#pragma once

#include <cstdint>
#include <string>

namespace cta {

// Same trace as the original Python generator (byte-identical JSON).
void write_kineto_trace(const std::string& path, int steps, double attn_us,
                        double gemm_us, const std::string& kernel_variant);

struct KinetoLargeSpec {
  std::uint64_t target_bytes = 200ull << 20;  // stop after the step that passes this
  std::uint64_t seed = 1;
  int rank = 0;
  int world_size = 1;
  int layers_per_step = 24;
  int child_cpu_ops_per_layer = 3;
  int python_events_per_layer = 2;
  int tensors_per_op = 3;         // size of the "Input Dims" args
  bool distributed_info_after = true;  // put distributedInfo after traceEvents
};

// Realistic mix: CPU ops with large args, runtime launches, kernels with long
// demangled names (braces and quotes included), memcpy/memset, flow events,
// python frames and metadata. Returns the number of events written.
std::uint64_t write_kineto_trace_large(const std::string& path,
                                       const KinetoLargeSpec& spec);

}  // namespace cta
