// Reader for Nsight Systems SQLite exports.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cta {

// Total duration and call count per API name, from one of the CUDA runtime or
// OS runtime tables.
struct NsysSummary {
  std::vector<std::string> names;
  std::vector<std::int64_t> duration_ns, calls;
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
};

// Reads an export. The parts are read on separate connections at the same time.
// Throws std::runtime_error when the file cannot be opened or a query fails.
NsysColumns read_nsys_sqlite(const std::string& path);

}  // namespace cta
