#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "bindings/bindings.hpp"
#include "bindings/numpy.hpp"
#include "readers/chakra/chakra_read.hpp"
#include "readers/goal/goal_read.hpp"
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

nb::dict chakra_columns_to_dict(cta::ChakraColumns&& c) {
  nb::dict d;
  d["names"] = nb::cast(c.names);
  d["comm_types"] = nb::cast(c.comm_types);
  d["rank"] = to_numpy(std::move(c.rank));
  d["node_id"] = to_numpy(std::move(c.node_id));
  d["name"] = to_numpy(std::move(c.name));
  d["op_class"] = to_numpy(std::move(c.op_class));
  d["comm_type"] = to_numpy(std::move(c.comm_type));
  d["duration_ns"] = to_numpy(std::move(c.duration_ns));
  d["size_bytes"] = to_numpy(std::move(c.size_bytes));
  d["num_ops"] = to_numpy(std::move(c.num_ops));
  d["src_rank"] = to_numpy(std::move(c.src_rank));
  d["dst_rank"] = to_numpy(std::move(c.dst_rank));
  d["message_tag"] = to_numpy(std::move(c.message_tag));
  d["edge_rank"] = to_numpy(std::move(c.edge_rank));
  d["edge_src"] = to_numpy(std::move(c.edge_src));
  d["edge_dst"] = to_numpy(std::move(c.edge_dst));
  d["edge_kind"] = to_numpy(std::move(c.edge_kind));
  return d;
}

nb::dict goal_columns_to_dict(cta::GoalTextColumns&& c) {
  nb::dict d;
  d["rank"] = to_numpy(std::move(c.rank));
  d["node_id"] = to_numpy(std::move(c.node_id));
  d["name_end"] = to_numpy(std::move(c.name_end));
  d["name_bytes"] = to_numpy(std::move(c.name_bytes));
  d["op_class"] = to_numpy(std::move(c.op_class));
  d["duration_ns"] = to_numpy(std::move(c.duration_ns));
  d["size_bytes"] = to_numpy(std::move(c.size_bytes));
  d["src_rank"] = to_numpy(std::move(c.src_rank));
  d["dst_rank"] = to_numpy(std::move(c.dst_rank));
  d["message_tag"] = to_numpy(std::move(c.message_tag));
  d["edge_rank"] = to_numpy(std::move(c.edge_rank));
  d["edge_src"] = to_numpy(std::move(c.edge_src));
  d["edge_dst"] = to_numpy(std::move(c.edge_dst));
  d["edge_kind"] = to_numpy(std::move(c.edge_kind));
  nb::dict timing;
  timing["scan"] = c.scan_seconds;
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

  m.def(
      "read_chakra_et",
      [](const std::vector<std::string>& paths, const std::vector<std::uint32_t>& ranks, int threads) {
        cta::ChakraColumns columns;
        {
          nb::gil_scoped_release release;
          columns = cta::read_chakra_files(paths, ranks, threads);
        }
        return chakra_columns_to_dict(std::move(columns));
      },
      "paths"_a, "ranks"_a, "threads"_a = 0,
      "Reads Chakra .et files (one rank each) into dictionary-coded node and edge columns.");

  m.def(
      "read_goal_text",
      [](const std::string& path, int threads, std::uint64_t chunk_bytes) {
        cta::GoalTextColumns columns;
        {
          nb::gil_scoped_release release;
          columns = cta::read_goal_text(path, threads, chunk_bytes);
        }
        return goal_columns_to_dict(std::move(columns));
      },
      "path"_a, "threads"_a = 0, "chunk_bytes"_a = 32ull << 20,
      "Reads a GOAL text schedule (.goal) into node and edge columns.");
}

}  // namespace cta::bindings
