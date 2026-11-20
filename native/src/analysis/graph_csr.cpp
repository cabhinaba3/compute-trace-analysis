#include "analysis/graph_csr.hpp"

#include <algorithm>
#include <stdexcept>

namespace cta::analysis {

SuccessorGraph build_successor_graph(std::size_t n, const std::uint32_t* edge_src, const std::uint32_t* edge_dst,
                                     std::size_t m) {
  SuccessorGraph graph;
  std::vector<std::uint64_t>& row_ptr = graph.row_ptr;
  std::vector<std::uint32_t>& successors = graph.successors;
  row_ptr.assign(n + 1, 0);
  for (std::size_t i = 0; i < m; ++i) {
    if (edge_src[i] >= n || edge_dst[i] >= n) throw std::invalid_argument("edge names a vertex outside the graph");
    ++row_ptr[edge_src[i] + 1];
  }
  for (std::size_t u = 0; u < n; ++u) row_ptr[u + 1] += row_ptr[u];
  successors.resize(m);
  {
    std::vector<std::uint64_t> cursor(row_ptr.begin(), row_ptr.end() - 1);
    for (std::size_t i = 0; i < m; ++i) successors[cursor[edge_src[i]]++] = edge_dst[i];
  }
  {
    std::uint64_t write = 0;
    std::uint64_t row_begin = 0;
    for (std::size_t u = 0; u < n; ++u) {
      const std::uint64_t row_end = row_ptr[u + 1];
      auto first = successors.begin() + static_cast<std::ptrdiff_t>(row_begin);
      auto last = successors.begin() + static_cast<std::ptrdiff_t>(row_end);
      std::sort(first, last);
      last = std::unique(first, last);
      const std::uint64_t kept = static_cast<std::uint64_t>(last - first);
      if (write != row_begin) std::copy(first, last, successors.begin() + static_cast<std::ptrdiff_t>(write));
      row_begin = row_end;
      row_ptr[u] = write;
      write += kept;
    }
    row_ptr[n] = write;
    successors.resize(write);
    successors.shrink_to_fit();
  }
  return graph;
}

}  // namespace cta::analysis
