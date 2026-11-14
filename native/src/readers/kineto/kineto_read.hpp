// Reader for PyTorch Profiler (Kineto) Chrome traces: plain or gzip-compressed
// JSON in, the columns the Python adapter needs out.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cta {

// What the reader extracts from one rank's trace. Names are dictionary-coded:
// every `*_name` array holds indices into `names`, which lists each distinct
// string once in order of first appearance in the file.
//
// All times are nanoseconds. Rows are in file order.
struct KinetoColumns {
  std::optional<std::int64_t> rank;  // distributedInfo.rank, when it is an integer
  std::vector<std::string> names;

  // GPU activities (kernels, memcpys, memsets). category: 0 compute, 1 comm
  // (an NCCL kernel), 2 memcpy. launch_ns is NaN and launch_thread -1 when the
  // activity's correlation id matches no CUDA runtime call.
  std::vector<std::int32_t> gpu_name;
  std::vector<std::int8_t> gpu_category;
  std::vector<std::int64_t> gpu_start, gpu_end, gpu_bytes, gpu_stream, gpu_device;
  std::vector<double> gpu_launch_ns;
  std::vector<std::int64_t> gpu_launch_thread;

  // CPU operators; thread is a dense id of the (pid, tid) pair.
  std::vector<std::int64_t> op_start, op_end;
  std::vector<std::int32_t> op_name;
  std::vector<std::int64_t> op_thread;

  // ProfilerStep ranges.
  std::vector<std::int64_t> step_id, step_start, step_end;

  // CUDA runtime calls aggregated by name, in order of first appearance.
  std::vector<std::int32_t> runtime_name;
  std::vector<std::int64_t> runtime_duration, runtime_calls;

  // Where the time went, in seconds: reading (and gunzipping) the file,
  // splitting it into chunks, waiting for room to hand a chunk to a parser,
  // and waiting for the parsers and the merge to finish after the last chunk.
  double read_seconds = 0, split_seconds = 0, submit_wait_seconds = 0, finish_seconds = 0;
};

// Reads a trace with `threads` parser threads (0 = all usable CPUs). The file
// is cut into pieces of about `chunk_bytes` at event boundaries, parsed in
// parallel, and merged in file order, so the result does not depend on either
// number. Throws std::invalid_argument for malformed JSON and
// std::runtime_error when the file cannot be read. The file is read in blocks
// of `block_bytes`; the result does not depend on that either (tests use tiny
// blocks to cut the text at every kind of position).
KinetoColumns read_kineto_columns(const std::string& path, int threads, std::uint64_t chunk_bytes,
                                  std::uint64_t block_bytes = 8u << 20);

}  // namespace cta
