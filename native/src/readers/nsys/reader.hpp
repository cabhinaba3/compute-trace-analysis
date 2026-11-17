// Reader for Nsight Systems SQLite exports.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cta::readers {

// Total duration and call count per API name, from one of the CUDA runtime or
// OS runtime tables.
struct NsysSummary {
  std::vector<std::string> names;
  std::vector<std::int64_t> duration_ns, calls;

  // Calls visit(name, member) for every column, under the names Python sees.
  template <typename Visit>
  void for_each_column(Visit&& visit) {
    visit("names", names);
    visit("duration_ns", duration_ns);
    visit("calls", calls);
  }
};

// One CUDA runtime call per row, in table order. `name` indexes `names`
// (in order of first appearance); `thread` is the globalTid, -1 if unknown;
// `correlation` is the id that ties the call to the GPU work it launched, or
// -1.
struct NsysCalls {
  std::vector<std::string> names;
  std::vector<std::int32_t> name;
  std::vector<std::int64_t> start, end, thread, correlation;

  // Calls visit(name, member) for every column, under the names Python sees.
  template <typename Visit>
  void for_each_column(Visit&& visit) {
    visit("names", names);
    visit("name", name);
    visit("start", start);
    visit("end", end);
    visit("thread", thread);
    visit("correlation", correlation);
  }
};

// The GPU activities, launch times, NVTX ranges, GPU metric samples and API
// summaries of one export. Strings are dictionary-coded: `*_name` and `nvtx_text` hold indices
// into `names` and `nvtx_texts`, in order of first appearance.
struct NsysColumns {
  // GPU activities: kernels, then memcpys, then memsets, each in table order.
  // category: 0 compute, 1 comm (a kernel with "nccl" in its name), 2 memcpy.
  // launch_ns is NaN and launch_thread -1 for an activity whose correlation id
  // matches no CUDA runtime call (the first call with an id is the launch).
  std::vector<std::string> names;
  std::vector<std::int32_t> gpu_name;
  std::vector<std::int8_t> gpu_category;
  std::vector<std::int64_t> gpu_start, gpu_end, gpu_bytes, gpu_stream, gpu_device;
  std::vector<double> gpu_launch_ns;
  std::vector<std::int64_t> gpu_launch_thread;

  // Closed NVTX ranges in table order; thread is the globalTid, 0 if unknown.
  std::vector<std::string> nvtx_texts;
  std::vector<std::int32_t> nvtx_text;
  std::vector<std::int64_t> nvtx_start, nvtx_end, nvtx_thread;

  // Samples of the table GPU_METRICS joined with TARGET_INFO_GPU_METRICS.
  // `has_metrics` is false when the export lacks that form (older exports keep
  // the samples as text, which the caller reads).
  bool has_metrics = false;
  std::vector<std::string> metric_names;
  std::vector<std::int32_t> metric_name;
  std::vector<std::int64_t> metric_timestamp;
  std::vector<double> metric_value;

  // CUPTI_ACTIVITY_KIND_RUNTIME and OSRT_API, one row per API name in the order
  // SQLite groups them ("unknown" when the name id is not in StringIds). Empty
  // if the table or StringIds is missing.
  NsysSummary runtime, osrt;

  // Every CUPTI_ACTIVITY_KIND_RUNTIME call with a positive duration; empty if
  // the table is missing.
  NsysCalls runtime_calls;

  // Calls visit(name, member) for every column, under the names Python sees.
  template <typename Visit>
  void for_each_column(Visit&& visit) {
    visit("names", names);
    visit("gpu_name", gpu_name);
    visit("gpu_category", gpu_category);
    visit("gpu_start", gpu_start);
    visit("gpu_end", gpu_end);
    visit("gpu_bytes", gpu_bytes);
    visit("gpu_stream", gpu_stream);
    visit("gpu_device", gpu_device);
    visit("gpu_launch_ns", gpu_launch_ns);
    visit("gpu_launch_thread", gpu_launch_thread);
    visit("nvtx_texts", nvtx_texts);
    visit("nvtx_text", nvtx_text);
    visit("nvtx_start", nvtx_start);
    visit("nvtx_end", nvtx_end);
    visit("nvtx_thread", nvtx_thread);
    visit("has_metrics", has_metrics);
    visit("metric_names", metric_names);
    visit("metric_name", metric_name);
    visit("metric_timestamp", metric_timestamp);
    visit("metric_value", metric_value);
    visit("runtime", runtime);
    visit("osrt", osrt);
    visit("runtime_calls", runtime_calls);
  }
};

// Reads an export. The parts are read on separate connections at the same time.
// Throws std::runtime_error when the file cannot be opened or a query fails.
NsysColumns read_nsys(const std::string& path);

}  // namespace cta::readers
