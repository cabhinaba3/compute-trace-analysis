// Reader for PyTorch Profiler (Kineto) Chrome traces: plain or gzip-compressed
// JSON in, the columns the Python adapter needs out.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cta::readers {

// Where the time of one read went, in seconds: reading (and gunzipping) the
// file, splitting it into chunks, waiting for room to hand a chunk to a
// parser, and waiting for the parsers and the merge after the last chunk.
struct KinetoTiming {
  double read = 0, split = 0, submit_wait = 0, finish = 0;

  template <typename Visit>
  void for_each_column(Visit&& visit) {
    visit("read", read);
    visit("split", split);
    visit("submit_wait", submit_wait);
    visit("finish", finish);
  }
};

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

  // Collective metadata of the GPU rows that carry it (NCCL kernels of a PyTorch
  // trace): the row, the collective's name, its element counts, group size,
  // element type and process-group name.
  std::vector<std::int64_t> comm_row, comm_in, comm_out, comm_size;
  std::vector<std::int32_t> comm_op, comm_dtype, comm_group;

  // CPU operators; thread is a dense id of the (pid, tid) pair.
  std::vector<std::int64_t> op_start, op_end;
  std::vector<std::int32_t> op_name;
  std::vector<std::int64_t> op_thread;

  // ProfilerStep ranges.
  std::vector<std::int64_t> step_id, step_start, step_end;

  // CUDA runtime calls aggregated by name, in order of first appearance.
  std::vector<std::int32_t> runtime_name;
  std::vector<std::int64_t> runtime_duration, runtime_calls;

  // Every CUDA runtime call in file order: name, time, thread (dense id as
  // for op_thread) and correlation id (-1 when the call has none).
  std::vector<std::int32_t> rcall_name;
  std::vector<std::int64_t> rcall_start, rcall_end, rcall_thread, rcall_correlation;

  KinetoTiming timing;

  // Calls visit(name, member) for every column, under the names Python sees.
  template <typename Visit>
  void for_each_column(Visit&& visit) {
    visit("rank", rank);
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
    visit("comm_row", comm_row);
    visit("comm_op", comm_op);
    visit("comm_dtype", comm_dtype);
    visit("comm_group", comm_group);
    visit("comm_in", comm_in);
    visit("comm_out", comm_out);
    visit("comm_size", comm_size);
    visit("op_start", op_start);
    visit("op_end", op_end);
    visit("op_name", op_name);
    visit("op_thread", op_thread);
    visit("step_id", step_id);
    visit("step_start", step_start);
    visit("step_end", step_end);
    visit("runtime_name", runtime_name);
    visit("runtime_duration", runtime_duration);
    visit("runtime_calls", runtime_calls);
    visit("rcall_name", rcall_name);
    visit("rcall_start", rcall_start);
    visit("rcall_end", rcall_end);
    visit("rcall_thread", rcall_thread);
    visit("rcall_correlation", rcall_correlation);
    visit("timing", timing);
  }
};

// Reads a trace with `threads` parser threads (0 = all usable CPUs). The file
// is cut into pieces of about `chunk_bytes` at event boundaries, parsed in
// parallel, and merged in file order, so the result does not depend on either
// number. Throws std::invalid_argument for malformed JSON and
// std::runtime_error when the file cannot be read. The file is read in blocks
// of `block_bytes`; the result does not depend on that either (tests use tiny
// blocks to cut the text at every kind of position).
KinetoColumns read_kineto(const std::string& path, int threads, std::uint64_t chunk_bytes,
                                  std::uint64_t block_bytes = 8u << 20);

}  // namespace cta::readers
