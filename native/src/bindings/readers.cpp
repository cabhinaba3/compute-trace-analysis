#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "bindings/bindings.hpp"
#include "bindings/numpy.hpp"
#include "readers/kineto/kineto_read.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace cta::bindings {
namespace {

nb::dict kineto_columns_to_dict(cta::KinetoColumns&& c) {
  nb::dict d;
  if (c.rank) d["rank"] = *c.rank; else d["rank"] = nb::none();
  d["names"] = nb::cast(c.names);
  d["gpu_name"] = to_numpy(std::move(c.gpu_name));
  d["gpu_category"] = to_numpy(std::move(c.gpu_category));
  d["gpu_start"] = to_numpy(std::move(c.gpu_start));
  d["gpu_end"] = to_numpy(std::move(c.gpu_end));
  d["gpu_bytes"] = to_numpy(std::move(c.gpu_bytes));
  d["gpu_stream"] = to_numpy(std::move(c.gpu_stream));
  d["gpu_device"] = to_numpy(std::move(c.gpu_device));
  d["gpu_launch_ns"] = to_numpy(std::move(c.gpu_launch_ns));
  d["gpu_launch_thread"] = to_numpy(std::move(c.gpu_launch_thread));
  d["op_start"] = to_numpy(std::move(c.op_start));
  d["op_end"] = to_numpy(std::move(c.op_end));
  d["op_name"] = to_numpy(std::move(c.op_name));
  d["op_thread"] = to_numpy(std::move(c.op_thread));
  d["step_id"] = to_numpy(std::move(c.step_id));
  d["step_start"] = to_numpy(std::move(c.step_start));
  d["step_end"] = to_numpy(std::move(c.step_end));
  d["runtime_name"] = to_numpy(std::move(c.runtime_name));
  d["runtime_duration"] = to_numpy(std::move(c.runtime_duration));
  d["runtime_calls"] = to_numpy(std::move(c.runtime_calls));
  nb::dict timing;
  timing["read"] = c.read_seconds;
  timing["split"] = c.split_seconds;
  timing["submit_wait"] = c.submit_wait_seconds;
  timing["finish"] = c.finish_seconds;
  d["timing"] = timing;
  return d;
}

}  // namespace

void bind_readers(nb::module_& m) {
  m.def(
      "read_kineto_trace",
      [](const std::string& path, int threads, std::uint64_t chunk_bytes,
         std::uint64_t block_bytes) {
        cta::KinetoColumns columns;
        {
          nb::gil_scoped_release release;
          columns = cta::read_kineto_columns(path, threads, chunk_bytes, block_bytes);
        }
        return kineto_columns_to_dict(std::move(columns));
      },
      "path"_a, "threads"_a = 0, "chunk_bytes"_a = 32ull << 20, "block_bytes"_a = 8ull << 20,
      "Reads a Kineto trace (.json or .json.gz) into dictionary-coded columns.");
}

}  // namespace cta::bindings
