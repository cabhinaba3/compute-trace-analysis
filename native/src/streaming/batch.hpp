// Shared canonical batch, enum, and provenance types for streaming graph analysis.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cta::streaming {

// Unified op-class codes matching Chakra's native reader so all formats share
// one numeric vocabulary across streams, caches, and solvers.
enum class OpClass : std::uint8_t {
  kOther = 0,
  kMemory = 1,
  kCompute = 2,
  kCommSend = 3,
  kCommRecv = 4,
  kCommColl = 5,
  kStorage = 6,
  kDataOp = 7,
};

inline bool is_comm_class(std::uint8_t c) {
  return c == static_cast<std::uint8_t>(OpClass::kCommSend) ||
         c == static_cast<std::uint8_t>(OpClass::kCommRecv) ||
         c == static_cast<std::uint8_t>(OpClass::kCommColl);
}

inline bool is_p2p_class(std::uint8_t c) {
  return c == static_cast<std::uint8_t>(OpClass::kCommSend) ||
         c == static_cast<std::uint8_t>(OpClass::kCommRecv);
}

inline bool is_compute_class(std::uint8_t c) { return c == static_cast<std::uint8_t>(OpClass::kCompute); }

// Unified dependency edge kinds.
enum class EdgeKind : std::uint8_t {
  kData = 0,         // Chakra data_deps
  kCtrl = 1,         // Chakra ctrl_deps
  kRequires = 2,     // GOAL requires (finish-to-start)
  kIrequires = 3,    // GOAL irequires (start-to-start in strict mode)
  kP2P = 4,          // Inter-rank matched send -> recv
  kCollBarrier = 5,  // Optional virtual collective barrier edge
};

enum class GoalSemantics : std::uint8_t {
  kCtaLegacy = 0,  // Match cta's existing dag.cpp & goal/reader.cpp behavior
  kStrict = 1,     // LogGOPSim/GOAL spec: start-to-start irequires + forward label refs
};

enum class CollectiveSync : std::uint8_t {
  kNone = 0,     // CTA default: intra-rank collective edges only
  kBarrier = 1,  // 2-node virtual barrier across participating ranks (2P edges)
};

// Contiguous SoA batch of execution-graph nodes (bounded per chunk).
struct NodeBatch {
  std::vector<std::uint32_t> rank;
  std::vector<std::uint64_t> node_id;
  std::vector<std::uint64_t> name_end;
  std::vector<std::uint8_t> name_bytes;
  std::vector<std::uint8_t> op_class;
  std::vector<std::uint64_t> comm_type_end;
  std::vector<std::uint8_t> comm_type_bytes;
  std::vector<std::uint64_t> duration_ns;
  std::vector<std::uint64_t> size_bytes;
  std::vector<std::uint64_t> num_ops;
  std::vector<std::int32_t> src_rank;
  std::vector<std::int32_t> dst_rank;
  std::vector<std::uint32_t> message_tag;
  std::vector<std::uint64_t> comm_group_end;
  std::vector<std::uint8_t> comm_group_bytes;

  std::size_t size() const { return node_id.size(); }
  bool empty() const { return node_id.empty(); }

  void clear() {
    rank.clear();
    node_id.clear();
    name_end.clear();
    name_bytes.clear();
    op_class.clear();
    comm_type_end.clear();
    comm_type_bytes.clear();
    duration_ns.clear();
    size_bytes.clear();
    num_ops.clear();
    src_rank.clear();
    dst_rank.clear();
    message_tag.clear();
    comm_group_end.clear();
    comm_group_bytes.clear();
  }

  void reserve(std::size_t n) {
    rank.reserve(n);
    node_id.reserve(n);
    name_end.reserve(n);
    op_class.reserve(n);
    comm_type_end.reserve(n);
    duration_ns.reserve(n);
    size_bytes.reserve(n);
    num_ops.reserve(n);
    src_rank.reserve(n);
    dst_rank.reserve(n);
    message_tag.reserve(n);
    comm_group_end.reserve(n);
  }

  void push(std::uint32_t r, std::uint64_t id, std::string_view name, OpClass oc, std::string_view comm_type,
            std::uint64_t dur_ns, std::uint64_t bytes, std::uint64_t ops, std::int32_t sr, std::int32_t dr,
            std::uint32_t tag, std::string_view comm_group = {}) {
    rank.push_back(r);
    node_id.push_back(id);
    name_bytes.insert(name_bytes.end(), name.begin(), name.end());
    name_end.push_back(name_bytes.size());
    op_class.push_back(static_cast<std::uint8_t>(oc));
    comm_type_bytes.insert(comm_type_bytes.end(), comm_type.begin(), comm_type.end());
    comm_type_end.push_back(comm_type_bytes.size());
    duration_ns.push_back(dur_ns);
    size_bytes.push_back(bytes);
    num_ops.push_back(ops);
    src_rank.push_back(sr);
    dst_rank.push_back(dr);
    message_tag.push_back(tag);
    comm_group_bytes.insert(comm_group_bytes.end(), comm_group.begin(), comm_group.end());
    comm_group_end.push_back(comm_group_bytes.size());
  }

  std::string_view name_at(std::size_t i) const {
    const std::uint64_t begin = i == 0 ? 0 : name_end[i - 1];
    return {reinterpret_cast<const char*>(name_bytes.data() + begin), static_cast<std::size_t>(name_end[i] - begin)};
  }

  std::string_view comm_type_at(std::size_t i) const {
    const std::uint64_t begin = i == 0 ? 0 : comm_type_end[i - 1];
    return {reinterpret_cast<const char*>(comm_type_bytes.data() + begin),
            static_cast<std::size_t>(comm_type_end[i] - begin)};
  }

  std::string_view comm_group_at(std::size_t i) const {
    const std::uint64_t begin = i == 0 ? 0 : comm_group_end[i - 1];
    return {reinterpret_cast<const char*>(comm_group_bytes.data() + begin),
            static_cast<std::size_t>(comm_group_end[i] - begin)};
  }
};

// Contiguous SoA batch of intra-rank dependency edges.
struct EdgeBatch {
  std::vector<std::uint32_t> rank;
  std::vector<std::uint64_t> src_id;
  std::vector<std::uint64_t> dst_id;
  std::vector<std::uint8_t> edge_kind;

  std::size_t size() const { return src_id.size(); }
  bool empty() const { return src_id.empty(); }

  void clear() {
    rank.clear();
    src_id.clear();
    dst_id.clear();
    edge_kind.clear();
  }

  void reserve(std::size_t n) {
    rank.reserve(n);
    src_id.reserve(n);
    dst_id.reserve(n);
    edge_kind.reserve(n);
  }

  void push(std::uint32_t r, std::uint64_t s, std::uint64_t d, EdgeKind kind) {
    rank.push_back(r);
    src_id.push_back(s);
    dst_id.push_back(d);
    edge_kind.push_back(static_cast<std::uint8_t>(kind));
  }
};

// One chunk yielded by a GraphSource; supports zero-copy Python dict conversion.
struct GraphChunk {
  NodeBatch nodes;
  EdgeBatch edges;

  void clear() {
    nodes.clear();
    edges.clear();
  }

  bool empty() const { return nodes.empty() && edges.empty(); }

  template <typename Visit>
  void for_each_column(Visit&& visit) {
    visit("rank", nodes.rank);
    visit("node_id", nodes.node_id);
    visit("name_end", nodes.name_end);
    visit("name_bytes", nodes.name_bytes);
    visit("op_class", nodes.op_class);
    visit("comm_type_end", nodes.comm_type_end);
    visit("comm_type_bytes", nodes.comm_type_bytes);
    visit("duration_ns", nodes.duration_ns);
    visit("size_bytes", nodes.size_bytes);
    visit("num_ops", nodes.num_ops);
    visit("src_rank", nodes.src_rank);
    visit("dst_rank", nodes.dst_rank);
    visit("message_tag", nodes.message_tag);
    visit("comm_group_end", nodes.comm_group_end);
    visit("comm_group_bytes", nodes.comm_group_bytes);
    visit("edge_rank", edges.rank);
    visit("edge_src", edges.src_id);
    visit("edge_dst", edges.dst_id);
    visit("edge_kind", edges.edge_kind);
  }
};

// Honest fidelity and diagnostic provenance attached to every streaming result.
struct StreamProvenance {
  std::string engine = "in_memory";
  bool exact = true;
  std::string slack_scope = "global_exact";     // "global_exact" | "window_local" | "unavailable"
  std::string goal_semantics = "cta_legacy";    // "cta_legacy" | "strict"
  std::string collective_sync = "none";         // "none" | "barrier"
  double makespan_lb = 0.0;
  double makespan_ub = 0.0;
  std::uint64_t evicted_dep_hits = 0;
  std::uint64_t lookahead_drops = 0;
  std::uint64_t straddling_edges = 0;
  std::uint64_t dead_end_sinks = 0;
  std::uint64_t unmatched_p2p_sends = 0;
  std::uint64_t unmatched_p2p_recvs = 0;
  bool has_cycle = false;
  std::uint64_t unordered_nodes = 0;
  std::vector<std::uint32_t> cycle_witness_rank;
  std::vector<std::uint64_t> cycle_witness_node_id;
  std::string quantile_sketch = "exact";        // "exact" | "ddsketch" | "kll" | "gk"
  double quantile_error_bound = 0.0;
  std::string delay_timing_basis = "none";      // "measured" | "modeled" | "none"

  template <typename Visit>
  void for_each_column(Visit&& visit) {
    visit("engine", engine);
    visit("exact", exact);
    visit("slack_scope", slack_scope);
    visit("goal_semantics", goal_semantics);
    visit("collective_sync", collective_sync);
    visit("makespan_lb", makespan_lb);
    visit("makespan_ub", makespan_ub);
    visit("evicted_dep_hits", evicted_dep_hits);
    visit("lookahead_drops", lookahead_drops);
    visit("straddling_edges", straddling_edges);
    visit("dead_end_sinks", dead_end_sinks);
    visit("unmatched_p2p_sends", unmatched_p2p_sends);
    visit("unmatched_p2p_recvs", unmatched_p2p_recvs);
    visit("has_cycle", has_cycle);
    visit("unordered_nodes", unordered_nodes);
    visit("cycle_witness_rank", cycle_witness_rank);
    visit("cycle_witness_node_id", cycle_witness_node_id);
    visit("quantile_sketch", quantile_sketch);
    visit("quantile_error_bound", quantile_error_bound);
    visit("delay_timing_basis", delay_timing_basis);
  }
};

}  // namespace cta::streaming
