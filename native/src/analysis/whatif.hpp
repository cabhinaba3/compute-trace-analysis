// What-if analysis of an execution graph: how long would it take if some of
// its vertices ran faster or slower?
//
// A DagPlan is the graph reduced to what repeated solving needs: vertices
// renumbered in a topological order and each vertex's distinct predecessors.
// It is built once. Each scenario then gives every group of vertices a time
// multiplier, and the solver finds the longest path again with those weights.
// Scenarios are solved in blocks of lanes that share one walk over the graph,
// and blocks run on separate threads.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cta::analysis {

class DagPlan {
 public:
  // `n` vertices (numbered 0..n-1, n < 2^32) and `m` edges
  // `edge_src[i] -> edge_dst[i]`; repeated edges count once. The order is a
  // FIFO Kahn pass seeded in ascending vertex order. If the graph has a cycle,
  // the vertices it holds back follow in index order and edges that point
  // backwards in that order are dropped, so results are approximate.
  //
  // Throws std::invalid_argument if an edge names a vertex >= n or n >= 2^32.
  DagPlan(std::size_t n, const std::uint32_t* edge_src, const std::uint32_t* edge_dst, std::size_t m);

  std::size_t num_vertices() const { return orig_of_.size(); }
  std::uint64_t num_edges() const { return num_edges_; }  // distinct edges
  std::uint64_t unordered() const { return unordered_; }  // vertices a cycle held back

  // Position of each vertex in the order, and the vertex at each position.
  const std::vector<std::uint32_t>& position_of() const { return pos_of_; }
  const std::vector<std::uint32_t>& vertex_at() const { return orig_of_; }
  // Predecessors (as positions, ascending) of the vertex at position p:
  // `preds[pred_ptr[p] .. pred_ptr[p + 1])`.
  const std::vector<std::uint64_t>& pred_ptr() const { return pred_ptr_; }
  const std::vector<std::uint32_t>& preds() const { return preds_; }

 private:
  std::uint64_t num_edges_ = 0;
  std::uint64_t unordered_ = 0;
  std::vector<std::uint32_t> pos_of_, orig_of_;
  std::vector<std::uint64_t> pred_ptr_;
  std::vector<std::uint32_t> preds_;
};

// One entry per scenario.
struct WhatIfResult {
  std::vector<double> makespan;
  std::vector<std::int32_t> depth;  // most vertices on a longest path
  std::vector<double> critical_compute;
  std::vector<double> critical_comm;
  std::vector<std::int64_t> critical_comm_bytes;
  std::vector<double> critical_by_group;  // scenarios x groups, row-major: critical-path weight per group
  int lanes = 0;                          // scenarios sharing one walk over the graph
  int workers = 0;                        // threads that solved blocks
  std::uint64_t bytes_per_worker = 0;
  std::string device = "cpu";  // where it ran: "cpu", "cuda" or "metal"
  std::string note;            // why a GPU was not used, when one was asked for
};

// What every solver reads, checked and put in position order: the weight and group of the
// vertex at each position.
struct PreparedInputs {
  std::vector<double> base;
  std::vector<std::uint32_t> group;
};

// Throws std::invalid_argument for a group >= num_groups or a multiplier that is negative or
// not finite.
PreparedInputs prepare_inputs(const DagPlan& plan, const double* weights, const std::uint32_t* group,
                              std::size_t num_groups, const double* scale, std::size_t num_scenarios);

// Solves `num_scenarios` scenarios. `weights`, `is_comm`, `is_compute`,
// `size_bytes` and `group` have one entry per vertex, in vertex order. Vertex
// v in scenario s weighs `weights[v] * scale[s * num_groups + group[v]]`.
//
// The longest path and its split follow the rules of solve_dag (core/dag.hpp)
// exactly: with every multiplier 1 the makespan, depth and critical-path
// compute, comm and bytes equal solve_dag's. `critical_by_group` adds up the
// weights on that path by group.
//
// `threads` is 0 for all usable CPUs. Each worker needs 12 bytes per vertex
// per lane; lanes and workers are chosen to fit `memory_budget_bytes` (0 = what
// the machine has available, about 60% of it).
//
// Throws std::invalid_argument for a group >= num_groups or a multiplier that
// is negative or not finite, and std::runtime_error if one worker with one
// lane does not fit the budget.
WhatIfResult solve_scenarios(const DagPlan& plan, const double* weights, const bool* is_comm, const bool* is_compute,
                             const std::int64_t* size_bytes, const std::uint32_t* group, std::size_t num_groups,
                             const double* scale, std::size_t num_scenarios, int threads,
                             std::uint64_t memory_budget_bytes);

}  // namespace cta::analysis
