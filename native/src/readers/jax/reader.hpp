// Reader for JAX/XProf traces: Chrome-trace JSON (.trace.json.gz,
// perfetto_trace.json.gz, .json, .json.gz), XPlane protobuf (.xplane.pb,
// .xplane.pb.gz) and binary Perfetto protobuf (.pftrace, .pftrace.gz,
// .perfetto-trace).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cta::readers {

// Where the time of one read went, in seconds.
struct JaxTiming {
  double read = 0, split = 0, submit_wait = 0, finish = 0;

  template <typename Visit>
  void for_each_column(Visit&& visit) {
    visit("read", read);
    visit("split", split);
    visit("submit_wait", submit_wait);
    visit("finish", finish);
  }
};

// Uniform dictionary-coded columns extracted from one host's JAX/XProf trace,
// regardless of whether the source file was Chrome-trace JSON, XPlane protobuf,
// or binary Perfetto protobuf.
//
// All times are integer nanoseconds. Rows are in trace order. Every `*_name`,
// `*_scope*`, `*_hlo_*`, `*_module` array holds indices into `names`, which
// lists each distinct string once in order of first appearance.
struct JaxColumns {
  std::string format;  // "json", "xplane", or "pftrace"
  std::vector<std::string> names;

  // Device ops (every event on a selected /device:* stream plus XLA:CPU thunks).
  // op_category: 0 compute, 1 comm, 2 memcpy (matches table_schemas.GPU_CATEGORIES).
  // op_launch_ns is NaN when the op's correlation_id matches no host launch.
  std::vector<std::int32_t> op_name;
  std::vector<std::int32_t> op_scope;
  std::vector<std::int32_t> op_scope_path;
  std::vector<std::int32_t> op_hlo_op;
  std::vector<std::int32_t> op_hlo_module;
  std::vector<std::int8_t> op_category;
  std::vector<std::int64_t> op_start, op_end, op_bytes, op_stream, op_device;
  std::vector<std::int64_t> op_correlation;
  std::vector<double> op_launch_ns;

  // Annotated step spans (jax.profiler.StepTraceAnnotation with step_num).
  std::vector<std::int64_t> step_id, step_start, step_end;

  // Host-side executions of compiled jitted modules (PJRT Execute spans).
  std::vector<std::int32_t> exec_module;
  std::vector<std::int64_t> exec_start, exec_end;

  // Host-side API calls (host events with correlation_id >= 0 and no hlo_op).
  // rcall_thread is a dense id of the host (pid, tid) pair.
  std::vector<std::int32_t> rcall_name;
  std::vector<std::int64_t> rcall_start, rcall_end, rcall_thread, rcall_correlation;

  JaxTiming timing;

  template <typename Visit>
  void for_each_column(Visit&& visit) {
    visit("format", format);
    visit("names", names);
    visit("op_name", op_name);
    visit("op_scope", op_scope);
    visit("op_scope_path", op_scope_path);
    visit("op_hlo_op", op_hlo_op);
    visit("op_hlo_module", op_hlo_module);
    visit("op_category", op_category);
    visit("op_start", op_start);
    visit("op_end", op_end);
    visit("op_bytes", op_bytes);
    visit("op_stream", op_stream);
    visit("op_device", op_device);
    visit("op_correlation", op_correlation);
    visit("op_launch_ns", op_launch_ns);
    visit("step_id", step_id);
    visit("step_start", step_start);
    visit("step_end", step_end);
    visit("exec_module", exec_module);
    visit("exec_start", exec_start);
    visit("exec_end", exec_end);
    visit("rcall_name", rcall_name);
    visit("rcall_start", rcall_start);
    visit("rcall_end", rcall_end);
    visit("rcall_thread", rcall_thread);
    visit("rcall_correlation", rcall_correlation);
    visit("timing", timing);
  }
};

// Reads a JAX/XProf trace into dictionary-coded columns.
//
// `format` is "auto" (detect from extension and header), "json", "xplane", or
// "pftrace". `threads` is the parser thread count (0 = all usable CPUs).
// `chunk_bytes` and `block_bytes` control streaming JSON chunking; output is
// independent of threads, chunk_bytes and block_bytes.
//
// Throws std::invalid_argument for malformed input or unknown format, and
// std::runtime_error when the file cannot be read.
JaxColumns read_jax(const std::string& path, const std::string& format = "auto", int threads = 0,
                    std::uint64_t chunk_bytes = 32ull << 20, std::uint64_t block_bytes = 8ull << 20);

}  // namespace cta::readers
