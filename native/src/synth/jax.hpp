#pragma once

#include <cstdint>
#include <string>

namespace cta::synth {

// Writes a small synthetic JAX/XProf Chrome-trace JSON (.json or .json.gz).
void write_jax_trace(const std::string& path, int steps = 3, int devices = 2, double gemm_us = 180.0,
                     double comm_us = 60.0, bool annotated = true);

// Writes the same synthetic JAX workload as an XPlane protobuf (.xplane.pb or .xplane.pb.gz).
void write_jax_xplane(const std::string& path, int steps = 3, int devices = 2, double gemm_us = 180.0,
                      double comm_us = 60.0, bool annotated = true);

// Writes the same synthetic JAX workload as a binary Perfetto trace (.pftrace or .pftrace.gz).
// When `legacy_chrome` is false, uses TrackDescriptor + TrackEvent + InternedData;
// when true, uses ChromeEventBundle packets.
void write_jax_pftrace(const std::string& path, int steps = 3, int devices = 2, double gemm_us = 180.0,
                       double comm_us = 60.0, bool annotated = true, bool legacy_chrome = false);

struct JaxLargeSpec {
  std::uint64_t target_bytes = 64ull << 20;  // uncompressed JSON bytes target
  std::uint64_t seed = 1;
  int devices = 2;
  int layers_per_step = 16;
  int python_events_per_layer = 4;
  bool annotated = true;
};

// Writes a large deterministic JAX/XProf Chrome-trace JSON (.json or .json.gz)
// with realistic host Python frames, launches, compute fusions, NCCL AllReduce,
// memcpy/memset, and derived XProf lines. Returns the number of events written.
std::uint64_t write_jax_trace_large(const std::string& path, const JaxLargeSpec& spec);

}  // namespace cta::synth
