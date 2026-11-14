// Critical path and slack of an execution graph.
#pragma once

#include <cstddef>
#include <cstdint>

namespace cta {

// What the solver found. All times are in the units of the weights.
struct DagResult {
  std::uint64_t num_edges = 0;  // distinct edges
  std::uint64_t unordered = 0;  // vertices a cycle kept out of the topological order
  std::int32_t depth = 0;       // most vertices on a longest path
  double makespan = 0;          // length of the critical path
  double critical_compute = 0;  // weight of the compute vertices on it
  double critical_comm = 0;     // weight of the comm vertices on it
  std::int64_t critical_comm_bytes = 0;
  double mean_slack = 0;
  double zero_slack_ratio = 0;  // fraction of vertices with (near-)zero slack
};

// Longest-path solver for a graph of `n` vertices (numbered 0..n-1, n < 2^32)
// and `m` edges `edge_src[i] -> edge_dst[i]`; repeated edges count once.
//
// A Kahn pass in FIFO order, seeded in ascending vertex order, finds each
// vertex's earliest finish; a backward pass finds its latest finish, and the
// difference is its slack. The critical path ends at the first vertex with the
// largest finish. Where predecessors tie on finish time and depth, the one
// with the lowest vertex id is the parent, so the path does not depend on the
// order of the edges.
//
// If the graph has a cycle, the vertices it holds back are left out of the
// forward pass and appended in index order to the backward pass, and
// `unordered` says how many there were; results are then approximate.
//
// Throws std::invalid_argument if an edge names a vertex >= n or n >= 2^32.
DagResult solve_dag(const double* weights, const bool* is_comm, const bool* is_compute,
                    const std::int64_t* size_bytes, std::size_t n, const std::uint32_t* edge_src,
                    const std::uint32_t* edge_dst, std::size_t m);

}  // namespace cta
