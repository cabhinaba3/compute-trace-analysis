// Reader for Chakra execution traces in protobuf form (.et files).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cta::readers {

// The nodes and dependencies of a set of .et files, one row per node or edge,
// files in the order given. Strings are dictionary-coded: `name` and
// `comm_type` hold indices into `names` and `comm_types`, listed in order of
// first appearance.
struct ChakraColumns {
  std::vector<std::string> names;
  std::vector<std::string> comm_types;

  // Nodes.
  std::vector<std::uint32_t> rank;
  std::vector<std::uint64_t> node_id;
  std::vector<std::int32_t> name;
  // 0 other, 1 memory, 2 compute, 3 comm_send, 4 comm_recv, 5 comm_coll,
  // 6 storage, 7 data_op.
  std::vector<std::uint8_t> op_class;
  std::vector<std::int32_t> comm_type;
  std::vector<std::uint64_t> duration_ns, size_bytes, num_ops;
  std::vector<std::int32_t> src_rank, dst_rank;
  std::vector<std::uint32_t> message_tag;

  // Dependencies: `edge_src` must finish before `edge_dst` starts.
  std::vector<std::uint32_t> edge_rank;
  std::vector<std::uint64_t> edge_src, edge_dst;
  std::vector<std::uint8_t> edge_kind;  // 0 data, 1 control

  // Calls visit(name, member) for every column, under the names Python sees.
  template <typename Visit>
  void for_each_column(Visit&& visit) {
    visit("names", names);
    visit("comm_types", comm_types);
    visit("rank", rank);
    visit("node_id", node_id);
    visit("name", name);
    visit("op_class", op_class);
    visit("comm_type", comm_type);
    visit("duration_ns", duration_ns);
    visit("size_bytes", size_bytes);
    visit("num_ops", num_ops);
    visit("src_rank", src_rank);
    visit("dst_rank", dst_rank);
    visit("message_tag", message_tag);
    visit("edge_rank", edge_rank);
    visit("edge_src", edge_src);
    visit("edge_dst", edge_dst);
    visit("edge_kind", edge_kind);
  }
};

// Reads the files on `threads` threads (0 = all usable CPUs), one task per
// file, and returns the rows in file order, so the result does not depend on
// the thread count. `ranks[i]` is the rank of `paths[i]`.
//
// Throws std::invalid_argument for a truncated or corrupt file and
// std::runtime_error when a file cannot be read.
ChakraColumns read_chakra(const std::vector<std::string>& paths,
                                const std::vector<std::uint32_t>& ranks, int threads);

}  // namespace cta::readers
