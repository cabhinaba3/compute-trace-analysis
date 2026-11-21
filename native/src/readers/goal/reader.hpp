// Reader for GOAL text schedules (.goal files).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cta::readers {

// The ops and dependencies of one GOAL text file, one row per op or edge, in
// file order. Op names are not dictionary-coded, because labels are unique:
// name i is name_bytes[name_end[i - 1] .. name_end[i]) (from 0 for i = 0).
// Seconds spent finding rank boundaries and waiting for the parsers.
struct GoalTiming {
  double scan = 0, finish = 0;

  // Calls visit(name, member) for every column, under the names Python sees.
  template <typename Visit>
  void for_each_column(Visit&& visit) {
    visit("scan", scan);
    visit("finish", finish);
  }
};

struct GoalColumns {
  std::vector<std::uint32_t> rank;
  std::vector<std::uint64_t> node_id;  // index of the op within its rank
  std::vector<std::uint64_t> name_end;
  std::vector<std::uint8_t> name_bytes;
  std::vector<std::uint8_t> op_class;  // 0 compute, 1 send, 2 recv
  std::vector<std::uint64_t> duration_ns, size_bytes;
  std::vector<std::int32_t> src_rank, dst_rank;
  std::vector<std::uint32_t> message_tag;
  std::vector<std::uint16_t> cpu, nic;  // "cpu N" and "nic N" of the op

  // Dependencies: `edge_src` must finish before `edge_dst` starts.
  std::vector<std::uint32_t> edge_rank;
  std::vector<std::uint64_t> edge_src, edge_dst;
  std::vector<std::uint8_t> edge_kind;  // 0 requires, 1 irequires

  GoalTiming timing;

  // Calls visit(name, member) for every column, under the names Python sees.
  template <typename Visit>
  void for_each_column(Visit&& visit) {
    visit("rank", rank);
    visit("node_id", node_id);
    visit("name_end", name_end);
    visit("name_bytes", name_bytes);
    visit("op_class", op_class);
    visit("duration_ns", duration_ns);
    visit("size_bytes", size_bytes);
    visit("src_rank", src_rank);
    visit("dst_rank", dst_rank);
    visit("message_tag", message_tag);
    visit("cpu", cpu);
    visit("nic", nic);
    visit("edge_rank", edge_rank);
    visit("edge_src", edge_src);
    visit("edge_dst", edge_dst);
    visit("edge_kind", edge_kind);
    visit("timing", timing);
  }
};

// Reads a schedule on `threads` threads (0 = all usable CPUs). The text is cut
// at rank headers into pieces of about `chunk_bytes`, which are parsed in
// parallel and merged in file order, so the result does not depend on either
// number. Throws std::invalid_argument for text that is not UTF-8 or holds a
// number the columns cannot hold, and std::runtime_error when the file cannot
// be read.
GoalColumns read_goal(const std::string& path, int threads, std::uint64_t chunk_bytes);

}  // namespace cta::readers
