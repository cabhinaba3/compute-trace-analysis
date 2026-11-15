#include "core/dag.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cta {

DagResult solve_dag(const double* weights, const bool* is_comm, const bool* is_compute,
                    const std::int64_t* size_bytes, std::size_t n, const std::uint32_t* edge_src,
                    const std::uint32_t* edge_dst, std::size_t m) {
  DagResult result;
  if (n == 0) return result;
  if (n > std::numeric_limits<std::uint32_t>::max()) throw std::invalid_argument("too many vertices");

  // --- Edges grouped by source; each row sorted and free of repeats.
  std::vector<std::uint64_t> row_ptr(n + 1, 0);
  for (std::size_t i = 0; i < m; ++i) {
    if (edge_src[i] >= n || edge_dst[i] >= n) throw std::invalid_argument("edge names a vertex outside the graph");
    ++row_ptr[edge_src[i] + 1];
  }
  for (std::size_t u = 0; u < n; ++u) row_ptr[u + 1] += row_ptr[u];
  std::vector<std::uint32_t> successors(m);
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
  result.num_edges = row_ptr[n];

  // --- Forward pass: Kahn's algorithm, FIFO, with the longest-path DP.
  std::vector<std::int32_t> in_degree(n, 0);
  for (std::uint32_t v : successors) ++in_degree[v];

  std::vector<double> earliest_start(n, 0.0);
  std::vector<double> earliest_finish(weights, weights + n);
  std::vector<std::int32_t> depth(n, 1);
  std::vector<std::int64_t> parent(n, -1);

  std::vector<std::uint32_t> order;  // doubles as the queue: [head, size) is waiting
  order.reserve(n);
  for (std::size_t u = 0; u < n; ++u)
    if (in_degree[u] == 0) order.push_back(static_cast<std::uint32_t>(u));
  for (std::size_t head = 0; head < order.size(); ++head) {
    const std::uint32_t u = order[head];
    const double u_finish = earliest_finish[u];
    const std::int32_t u_depth = depth[u];
    for (std::uint64_t e = row_ptr[u]; e < row_ptr[u + 1]; ++e) {
      const std::uint32_t v = successors[e];
      const bool later = u_finish > earliest_start[v];
      const bool tied = u_finish == earliest_start[v];
      if (later || (tied && (u_depth + 1 > depth[v] ||
                             (u_depth + 1 == depth[v] && parent[v] > static_cast<std::int64_t>(u))))) {
        earliest_start[v] = u_finish;
        earliest_finish[v] = u_finish + weights[v];
        depth[v] = u_depth + 1;
        parent[v] = u;
      }
      if (--in_degree[v] == 0) order.push_back(v);
    }
  }
  result.unordered = n - order.size();
  if (order.size() < n) {
    // A cycle: the vertices it holds back follow in index order.
    std::vector<bool> seen(n, false);
    for (std::uint32_t u : order) seen[u] = true;
    for (std::size_t u = 0; u < n; ++u)
      if (!seen[u]) order.push_back(static_cast<std::uint32_t>(u));
  }

  std::size_t sink = 0;
  for (std::size_t u = 1; u < n; ++u)
    if (earliest_finish[u] > earliest_finish[sink]) sink = u;
  const double makespan = earliest_finish[sink];
  result.makespan = makespan;
  result.depth = *std::max_element(depth.begin(), depth.end());

  // --- Backward pass: the latest finish that still meets the makespan.
  std::vector<double> latest_finish(n, makespan);
  for (std::size_t i = n; i-- > 0;) {
    const std::uint32_t u = order[i];
    if (row_ptr[u] == row_ptr[u + 1]) continue;
    double min_start = makespan;
    for (std::uint64_t e = row_ptr[u]; e < row_ptr[u + 1]; ++e) {
      const std::uint32_t v = successors[e];
      const double start = latest_finish[v] - weights[v];
      if (start < min_start) min_start = start;
    }
    latest_finish[u] = min_start;
  }

  const double tolerance = std::max(1e-6, makespan * 1e-9);
  long double slack_sum = 0;
  std::uint64_t zero_slack = 0;
  for (std::size_t u = 0; u < n; ++u) {
    const double slack = std::max(0.0, latest_finish[u] - earliest_finish[u]);
    slack_sum += slack;
    if (slack <= tolerance) ++zero_slack;
  }
  result.mean_slack = static_cast<double>(slack_sum / static_cast<long double>(n));
  result.zero_slack_ratio = static_cast<double>(zero_slack) / static_cast<double>(n);

  // --- The critical chain, from the sink back to a source (or to a repeat).
  std::vector<bool> visited(n, false);
  long double compute = 0, comm = 0;
  for (std::int64_t u = static_cast<std::int64_t>(sink); u != -1 && !visited[static_cast<std::size_t>(u)];
       u = parent[static_cast<std::size_t>(u)]) {
    const std::size_t k = static_cast<std::size_t>(u);
    visited[k] = true;
    if (is_compute[k]) compute += weights[k];
    if (is_comm[k]) {
      comm += weights[k];
      result.critical_comm_bytes += size_bytes[k];
    }
  }
  result.critical_compute = static_cast<double>(compute);
  result.critical_comm = static_cast<double>(comm);
  return result;
}

}  // namespace cta
