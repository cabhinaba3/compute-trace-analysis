// The edges of an execution graph: intra-rank dependencies given by per-rank
// node ids, and the links between matching sends and receives.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cta {

struct EdgeList {
  std::vector<std::uint32_t> src, dst;  // vertex ids; an edge may repeat
  // Seconds spent finding vertices by node id, looking up dependencies, and
  // linking sends to receives.
  double index_seconds = 0, dependency_seconds = 0, link_seconds = 0;
};

// Vertices are the rows 0..n-1 of the node table. `rank` and `node_id` name a
// node within its rank; `is_send` and `is_recv` mark point-to-point ops, whose
// channel is (src_rank, dst_rank, tag).
//
// Returns an edge for every dependency (e_rank, e_src) -> (e_rank, e_dst)
// whose two ends exist (every pair of vertices when a (rank, node id) occurs on
// more than one row), then an edge from the k-th send of each channel to its
// k-th receive, in row order, for as many pairs as both sides have. The result
// does not depend on `threads` (0 = all usable CPUs). Throws
// std::invalid_argument if n >= 2^32 or the columns differ in length.
EdgeList build_dag_edges(std::size_t n, const std::uint64_t* rank, const std::uint64_t* node_id,
                         const bool* is_send, const bool* is_recv, const std::int64_t* src_rank,
                         const std::int64_t* dst_rank, const std::int64_t* tag, std::size_t m,
                         const std::uint64_t* e_rank, const std::uint64_t* e_src, const std::uint64_t* e_dst,
                         int threads);

}  // namespace cta
