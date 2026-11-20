// Successor lists of an execution graph, shared by the critical-path solver
// and the what-if planner.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cta::analysis {

// Edges grouped by source vertex: the successors of vertex u are
// `successors[row_ptr[u] .. row_ptr[u + 1])`, sorted ascending and free of
// repeats.
struct SuccessorGraph {
  std::vector<std::uint64_t> row_ptr;  // n + 1 entries
  std::vector<std::uint32_t> successors;
};

// Builds the lists for `n` vertices (numbered 0..n-1) from `m` edges
// `edge_src[i] -> edge_dst[i]`; repeated edges count once.
//
// Throws std::invalid_argument if an edge names a vertex >= n.
SuccessorGraph build_successor_graph(std::size_t n, const std::uint32_t* edge_src, const std::uint32_t* edge_dst,
                                     std::size_t m);

}  // namespace cta::analysis
