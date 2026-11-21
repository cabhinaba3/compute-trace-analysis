// Pluggable GraphEngine interface, common solve options, and result structs.
// New out-of-core or streaming graph algorithms register under
// native/src/streaming/engines/<name>/ and implement GraphEngine.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "streaming/batch.hpp"
#include "streaming/sources/source.hpp"

namespace cta::streaming::engines {

// Nominal 400GbE LogGP parameters matching src/cta/analysis/dag.py.
constexpr double kNominalLatencyNs = 500.0;
constexpr double kNominalOverheadNs = 150.0;
constexpr double kNominalInvBwNsPerByte = 0.02;  // 50 GB/s

inline double nominal_comm_cost_ns(std::uint64_t size_bytes) {
  return kNominalLatencyNs + 2.0 * kNominalOverheadNs + static_cast<double>(size_bytes) * kNominalInvBwNsPerByte;
}

// Weight of one execution-graph node in ns, matching _node_weights_ns in dag.py.
inline double node_weight_ns(std::uint64_t duration_ns, std::uint8_t op_class, std::uint64_t size_bytes) {
  const double d = static_cast<double>(duration_ns);
  if (d > 0.0 || !is_comm_class(op_class)) return d;
  const double wire = nominal_comm_cost_ns(size_bytes);
  return is_p2p_class(op_class) ? 0.5 * wire : wire;
}

struct DagSolveOptions {
  std::string engine = "gridgraph";
  std::uint64_t memory_budget_bytes = 0;  // 0 = auto (60% available RAM)
  GoalSemantics goal_semantics = GoalSemantics::kCtaLegacy;
  CollectiveSync collective_sync = CollectiveSync::kNone;
  std::uint32_t world_size_hint = 0;
  std::unordered_map<std::string, std::uint32_t> coll_group_sizes;
  std::size_t window_size = 262144;    // per-rank active window for 1-pass frontier
  std::size_t max_lookahead = 65536;   // cap on unresolved future parents
  bool fail_on_cycle = false;
  bool extract_critical_path_ops = false;
  bool delay_attribution = false;
  std::string scratch_dir;
  std::size_t chunk_nodes = 524288;
};

struct DagStreamResult {
  std::uint64_t num_nodes = 0;
  std::uint64_t num_edges = 0;
  std::uint64_t unordered = 0;
  std::int32_t depth = 0;
  double makespan = 0.0;
  double critical_compute = 0.0;
  double critical_comm = 0.0;
  std::int64_t critical_comm_bytes = 0;
  double mean_slack = 0.0;
  double zero_slack_ratio = 0.0;

  // Optional critical-path op chain (source -> sink order).
  std::vector<std::uint32_t> cp_op_rank;
  std::vector<std::uint64_t> cp_op_node_id;
  std::vector<std::uint8_t> cp_op_class;
  std::vector<double> cp_op_weight_ns;
  std::vector<std::uint64_t> cp_op_size_bytes;
  std::vector<double> cp_op_earliest_start_ns;
  std::vector<double> cp_op_earliest_finish_ns;
  std::vector<std::string> cp_op_name;

  // Optional Scalasca backward wait-state & delay attribution table.
  std::vector<std::uint32_t> delay_rank;
  std::vector<std::string> delay_op_name;
  std::vector<double> delay_direct_wait_ns;
  std::vector<double> delay_propagated_ns;
  std::vector<std::uint64_t> delay_op_count;

  StreamProvenance provenance;

  template <typename Visit>
  void for_each_column(Visit&& visit) {
    visit("num_nodes", num_nodes);
    visit("num_edges", num_edges);
    visit("unordered", unordered);
    visit("depth", depth);
    visit("makespan", makespan);
    visit("critical_compute", critical_compute);
    visit("critical_comm", critical_comm);
    visit("critical_comm_bytes", critical_comm_bytes);
    visit("mean_slack", mean_slack);
    visit("zero_slack_ratio", zero_slack_ratio);
    visit("cp_op_rank", cp_op_rank);
    visit("cp_op_node_id", cp_op_node_id);
    visit("cp_op_class", cp_op_class);
    visit("cp_op_weight_ns", cp_op_weight_ns);
    visit("cp_op_size_bytes", cp_op_size_bytes);
    visit("cp_op_earliest_start_ns", cp_op_earliest_start_ns);
    visit("cp_op_earliest_finish_ns", cp_op_earliest_finish_ns);
    visit("cp_op_name", cp_op_name);
    visit("delay_rank", delay_rank);
    visit("delay_op_name", delay_op_name);
    visit("delay_direct_wait_ns", delay_direct_wait_ns);
    visit("delay_propagated_ns", delay_propagated_ns);
    visit("delay_op_count", delay_op_count);
    visit("provenance", provenance);
  }
};

// Abstract interface for pluggable streaming/out-of-core DAG engines.
class GraphEngine {
 public:
  virtual ~GraphEngine() = default;
  virtual std::string_view name() const = 0;
  virtual DagStreamResult solve_dag(GraphSource& source, const DagSolveOptions& opts) = 0;
};

// Registry factory for pluggable DAG engines ("frontier", "gridgraph", ...).
std::unique_ptr<GraphEngine> make_graph_engine(const std::string& name);
std::vector<std::string> available_graph_engines();

}  // namespace cta::streaming::engines
