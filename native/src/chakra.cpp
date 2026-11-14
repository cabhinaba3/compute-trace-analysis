#include "chakra.hpp"

#include <google/protobuf/util/delimited_message_util.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "common.hpp"
#include "et_def.pb.h"
#include "json_value.hpp"

namespace cta {
namespace {

void add_int_attr(ChakraProtoMsg::Node& node, const char* name, std::int64_t value) {
  ChakraProtoMsg::AttributeProto* a = node.add_attr();
  a->set_name(name);
  a->set_int64_val(value);
}

}  // namespace

std::vector<std::string> write_chakra_et_run(const std::string& out_dir, int world_size,
                                             std::int64_t compute_us, std::int64_t allreduce_bytes,
                                             int steps, int straggler_rank,
                                             double straggler_factor) {
  std::filesystem::create_directories(out_dir);
  std::vector<std::string> paths;
  for (int rank = 0; rank < world_size; ++rank) {
    const std::string path = out_dir + "/trace.rank" + std::to_string(rank) + ".et";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open " + path);

    ChakraProtoMsg::GlobalMetadata meta;
    meta.set_version("0.0.4");
    google::protobuf::util::SerializeDelimitedToOstream(meta, &out);

    const std::int64_t compute_duration =
        rank == straggler_rank ? static_cast<std::int64_t>(static_cast<double>(compute_us) * straggler_factor)
                               : compute_us;
    std::uint64_t node_id = 0;
    bool have_previous = false;
    std::uint64_t previous = 0;
    for (int step = 0; step < steps; ++step) {
      ChakraProtoMsg::Node compute;
      compute.set_id(node_id);
      compute.set_name("aten::mm_step" + std::to_string(step));
      compute.set_type(ChakraProtoMsg::COMP_NODE);
      compute.set_duration_micros(static_cast<std::uint64_t>(compute_duration));
      if (have_previous) compute.add_data_deps(previous);
      add_int_attr(compute, "num_ops", 2 * 1024 * 1024);
      add_int_attr(compute, "tensor_size", allreduce_bytes);
      google::protobuf::util::SerializeDelimitedToOstream(compute, &out);
      const std::uint64_t compute_id = node_id++;

      ChakraProtoMsg::Node allreduce;
      allreduce.set_id(node_id);
      allreduce.set_name("nccl:all_reduce_step" + std::to_string(step));
      allreduce.set_type(ChakraProtoMsg::COMM_COLL_NODE);
      allreduce.set_duration_micros(static_cast<std::uint64_t>(std::max<std::int64_t>(10, allreduce_bytes / 1000)));
      allreduce.add_data_deps(compute_id);
      add_int_attr(allreduce, "comm_type", 0);  // ALL_REDUCE
      add_int_attr(allreduce, "comm_size", allreduce_bytes);
      google::protobuf::util::SerializeDelimitedToOstream(allreduce, &out);
      previous = node_id++;
      have_previous = true;
    }
    out.close();
    if (!out) throw std::runtime_error("write failed: " + path);
    paths.push_back(path);
  }
  return paths;
}

void write_chakra_host_et(const std::string& path, int steps, double compute_us,
                          std::int64_t comm_bytes) {
  JsonArray nodes;
  std::int64_t node_id = 1;
  bool have_previous = false;
  std::int64_t previous = 0;
  for (int step = 0; step < steps; ++step) {
    JsonArray deps;
    if (have_previous) deps.emplace_back(previous);
    nodes.emplace_back(JsonObject{{"id", node_id},
                                  {"name", "aten::linear_" + std::to_string(step)},
                                  {"ctrl_deps", std::move(deps)},
                                  {"dur_us", compute_us}});
    const std::int64_t compute_id = node_id++;
    nodes.emplace_back(JsonObject{{"id", node_id},
                                  {"name", "c10d::all_reduce_" + std::to_string(step)},
                                  {"ctrl_deps", JsonArray{Json(compute_id)}},
                                  {"dur_us", static_cast<double>(comm_bytes) / 1000.0},
                                  {"comm_bytes", comm_bytes}});
    previous = node_id++;
    have_previous = true;
  }
  write_json_file(path,
                  Json(JsonObject{{"schema", "1.0.2-chakra.0.0.4"}, {"nodes", std::move(nodes)}}),
                  -1);
}

}  // namespace cta
