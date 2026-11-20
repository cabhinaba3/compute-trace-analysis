#include "analysis/whatif.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include "util/cpus.hpp"
#include "util/memory.hpp"
#include "analysis/graph_csr.hpp"

namespace cta::analysis {

DagPlan::DagPlan(std::size_t n, const std::uint32_t* edge_src, const std::uint32_t* edge_dst, std::size_t m) {
  if (n > std::numeric_limits<std::uint32_t>::max()) throw std::invalid_argument("too many vertices");
  if (n == 0) {
    pred_ptr_.assign(1, 0);
    return;
  }
  SuccessorGraph graph = build_successor_graph(n, edge_src, edge_dst, m);
  num_edges_ = graph.row_ptr[n];

  // Topological order: Kahn, FIFO, seeded in ascending vertex order.
  {
    std::vector<std::int32_t> in_degree(n, 0);
    for (std::uint32_t v : graph.successors) ++in_degree[v];
    orig_of_.reserve(n);
    for (std::size_t u = 0; u < n; ++u)
      if (in_degree[u] == 0) orig_of_.push_back(static_cast<std::uint32_t>(u));
    for (std::size_t head = 0; head < orig_of_.size(); ++head) {
      const std::uint32_t u = orig_of_[head];
      for (std::uint64_t e = graph.row_ptr[u]; e < graph.row_ptr[u + 1]; ++e)
        if (--in_degree[graph.successors[e]] == 0) orig_of_.push_back(graph.successors[e]);
    }
  }
  unordered_ = n - orig_of_.size();
  if (unordered_ != 0) {
    // A cycle: the vertices it holds back follow in index order.
    std::vector<bool> seen(n, false);
    for (std::uint32_t u : orig_of_) seen[u] = true;
    for (std::size_t u = 0; u < n; ++u)
      if (!seen[u]) orig_of_.push_back(static_cast<std::uint32_t>(u));
  }
  pos_of_.resize(n);
  for (std::size_t p = 0; p < n; ++p) pos_of_[orig_of_[p]] = static_cast<std::uint32_t>(p);

  // Predecessors by position. Filling the rows while walking sources in
  // position order leaves each row ascending.
  pred_ptr_.assign(n + 1, 0);
  for (std::size_t p = 0; p < n; ++p) {
    const std::uint32_t u = orig_of_[p];
    for (std::uint64_t e = graph.row_ptr[u]; e < graph.row_ptr[u + 1]; ++e) {
      const std::uint32_t q = pos_of_[graph.successors[e]];
      if (p < q) ++pred_ptr_[q + 1];
    }
  }
  for (std::size_t p = 0; p < n; ++p) pred_ptr_[p + 1] += pred_ptr_[p];
  preds_.resize(pred_ptr_[n]);
  std::vector<std::uint64_t> cursor(pred_ptr_.begin(), pred_ptr_.end() - 1);
  for (std::size_t p = 0; p < n; ++p) {
    const std::uint32_t u = orig_of_[p];
    for (std::uint64_t e = graph.row_ptr[u]; e < graph.row_ptr[u + 1]; ++e) {
      const std::uint32_t q = pos_of_[graph.successors[e]];
      if (p < q) preds_[cursor[q]++] = static_cast<std::uint32_t>(p);
    }
  }
}

// External linkage, so the compiler can pick between the copies of a hot loop when the library loads.
namespace detail {

// The hot loop is compiled twice on x86-64 (AVX2 and baseline) and the right
// copy is picked when the library loads. No FMA, so the answers do not depend
// on the CPU.
#if defined(__x86_64__) && defined(__has_attribute)
#if __has_attribute(target_clones)
#define CTA_MULTIARCH __attribute__((target_clones("avx2", "default")))
#endif
#endif
#ifndef CTA_MULTIARCH
#define CTA_MULTIARCH
#endif
#define CTA_ALWAYS_INLINE __attribute__((always_inline)) inline

// The graph in position order, as the passes read it.
struct Context {
  std::size_t n = 0;
  const std::uint64_t* pred_ptr = nullptr;
  const std::uint32_t* preds = nullptr;
  const std::uint32_t* orig_of = nullptr;
  const double* base = nullptr;          // weight of the vertex at each position
  const std::uint32_t* group = nullptr;  // group of the vertex at each position
};

// Per lane k and position p: finish time f[p * K + k] and the number of
// vertices on the longest path ending there, d[p * K + k].
template <int K>
CTA_ALWAYS_INLINE void forward_impl(const Context& c, const double* scale_t, double* f, std::int32_t* d, double* mk,
                                    std::uint32_t* sink, std::int32_t* max_depth) {
  for (int k = 0; k < K; ++k) {
    mk[k] = -1.0;
    sink[k] = 0;
    max_depth[k] = 0;
  }
  for (std::size_t v = 0; v < c.n; ++v) {
    const std::uint64_t first = c.pred_ptr[v], last = c.pred_ptr[v + 1];
    double best[K];
    std::int32_t depth[K];
    if (first == last) {
      for (int k = 0; k < K; ++k) {
        best[k] = 0.0;
        depth[k] = 0;
      }
    } else if (last - first == 1) {
      const std::size_t u = c.preds[first];
      for (int k = 0; k < K; ++k) {
        best[k] = f[u * K + k];
        depth[k] = d[u * K + k];
      }
    } else {
      {
        const std::size_t u = c.preds[first];
        for (int k = 0; k < K; ++k) best[k] = f[u * K + k];
      }
      for (std::uint64_t i = first + 1; i < last; ++i) {
        const std::size_t u = c.preds[i];
        for (int k = 0; k < K; ++k) best[k] = f[u * K + k] > best[k] ? f[u * K + k] : best[k];
      }
      // Among the predecessors that finish last: the deepest, then the lowest vertex id.
      std::uint32_t winner[K];
      for (int k = 0; k < K; ++k) {
        depth[k] = -1;
        winner[k] = std::numeric_limits<std::uint32_t>::max();
      }
      for (std::uint64_t i = first; i < last; ++i) {
        const std::size_t u = c.preds[i];
        const std::uint32_t id = c.orig_of[u];
        for (int k = 0; k < K; ++k) {
          if (f[u * K + k] != best[k]) continue;
          const std::int32_t du = d[u * K + k];
          if (du > depth[k] || (du == depth[k] && id < winner[k])) {
            depth[k] = du;
            winner[k] = id;
          }
        }
      }
    }
    const double base = c.base[v];
    const double* scale = scale_t + static_cast<std::size_t>(c.group[v]) * K;
    const std::uint32_t id = c.orig_of[v];
    for (int k = 0; k < K; ++k) {
      const double finish = best[k] + base * scale[k];
      const std::int32_t deep = depth[k] + 1;
      f[v * K + k] = finish;
      d[v * K + k] = deep;
      if (deep > max_depth[k]) max_depth[k] = deep;
      if (finish > mk[k] || (finish == mk[k] && id < c.orig_of[sink[k]])) {
        mk[k] = finish;
        sink[k] = static_cast<std::uint32_t>(v);
      }
    }
  }
}

using ForwardFn = void (*)(const Context&, const double*, double*, std::int32_t*, double*, std::uint32_t*,
                           std::int32_t*);

CTA_MULTIARCH void forward8(const Context& c, const double* scale_t, double* f, std::int32_t* d, double* mk,
                            std::uint32_t* sink, std::int32_t* max_depth) {
  forward_impl<8>(c, scale_t, f, d, mk, sink, max_depth);
}
CTA_MULTIARCH void forward4(const Context& c, const double* scale_t, double* f, std::int32_t* d, double* mk,
                            std::uint32_t* sink, std::int32_t* max_depth) {
  forward_impl<4>(c, scale_t, f, d, mk, sink, max_depth);
}
CTA_MULTIARCH void forward1(const Context& c, const double* scale_t, double* f, std::int32_t* d, double* mk,
                            std::uint32_t* sink, std::int32_t* max_depth) {
  forward_impl<1>(c, scale_t, f, d, mk, sink, max_depth);
}

struct Chain {
  long double compute = 0, comm = 0;
  std::int64_t comm_bytes = 0;
};

// Walks the critical chain of one lane from its sink back to a source, choosing
// at each step the predecessor the forward pass chose, and adds up its weights.
Chain trace_chain(const Context& c, const double* f, const std::int32_t* d, int lanes, int lane, std::uint32_t sink,
                  const double* scale, const bool* is_comm, const bool* is_compute, const std::int64_t* size_bytes,
                  long double* by_group) {
  Chain chain;
  std::size_t v = sink;
  for (;;) {
    const std::uint32_t g = c.group[v];
    const double w = c.base[v] * scale[g];
    by_group[g] += w;
    const std::uint32_t id = c.orig_of[v];
    if (is_compute[id]) chain.compute += w;
    if (is_comm[id]) {
      chain.comm += w;
      chain.comm_bytes += size_bytes[id];
    }
    const std::uint64_t first = c.pred_ptr[v], last = c.pred_ptr[v + 1];
    if (first == last) break;
    std::size_t win = c.preds[first];
    double win_f = f[win * lanes + lane];
    std::int32_t win_d = d[win * lanes + lane];
    for (std::uint64_t i = first + 1; i < last; ++i) {
      const std::size_t u = c.preds[i];
      const double fu = f[u * lanes + lane];
      const std::int32_t du = d[u * lanes + lane];
      if (fu > win_f || (fu == win_f && (du > win_d || (du == win_d && c.orig_of[u] < c.orig_of[win])))) {
        win = u;
        win_f = fu;
        win_d = du;
      }
    }
    v = win;
  }
  return chain;
}

}  // namespace detail

using namespace detail;

WhatIfResult solve_scenarios(const DagPlan& plan, const double* weights, const bool* is_comm, const bool* is_compute,
                             const std::int64_t* size_bytes, const std::uint32_t* group, std::size_t num_groups,
                             const double* scale, std::size_t num_scenarios, int threads,
                             std::uint64_t memory_budget_bytes) {
  const std::size_t n = plan.num_vertices();
  const std::size_t S = num_scenarios, G = num_groups;
  WhatIfResult result;
  result.makespan.assign(S, 0.0);
  result.depth.assign(S, 0);
  result.critical_compute.assign(S, 0.0);
  result.critical_comm.assign(S, 0.0);
  result.critical_comm_bytes.assign(S, 0);
  result.critical_by_group.assign(S * G, 0.0);
  if (S == 0 || n == 0) return result;

  for (std::size_t i = 0; i < S * G; ++i)
    if (!std::isfinite(scale[i]) || scale[i] < 0.0) throw std::invalid_argument("a multiplier is negative or not finite");

  // The graph's per-vertex inputs in position order.
  std::vector<double> base(n);
  std::vector<std::uint32_t> grp(n);
  const std::vector<std::uint32_t>& orig_of = plan.vertex_at();
  for (std::size_t p = 0; p < n; ++p) {
    const std::uint32_t id = orig_of[p];
    if (group[id] >= G) throw std::invalid_argument("a group id is outside 0..num_groups-1");
    base[p] = weights[id];
    grp[p] = group[id];
  }
  Context ctx;
  ctx.n = n;
  ctx.pred_ptr = plan.pred_ptr().data();
  ctx.preds = plan.preds().data();
  ctx.orig_of = orig_of.data();
  ctx.base = base.data();
  ctx.group = grp.data();

  // Lanes and workers: as many lanes as keep every thread busy, within memory.
  const int cpus = threads > 0 ? threads : usable_cpus();
  std::uint64_t budget = memory_budget_bytes;
  if (budget == 0) {
    const std::uint64_t available = available_memory_bytes();
    budget = available == 0 ? std::numeric_limits<std::uint64_t>::max() : available / 10 * 6;
  }
  const std::size_t target = std::min<std::size_t>(static_cast<std::size_t>(cpus), S);
  int lanes = 0;
  std::uint64_t per_worker = 0;
  for (int candidate : {8, 4, 1}) {
    const std::uint64_t bytes = 12ull * static_cast<std::uint64_t>(candidate) * n;
    if (bytes > budget) continue;
    if (candidate != 1 && (S + candidate - 1) / candidate < target) continue;
    lanes = candidate;
    per_worker = bytes;
    break;
  }
  if (lanes == 0)
    throw std::runtime_error("a what-if solve of " + std::to_string(n) + " vertices needs " +
                             std::to_string(12ull * n) + " bytes and the budget is " + std::to_string(budget));
  const std::size_t blocks = (S + lanes - 1) / lanes;
  const std::uint64_t fit = per_worker == 0 ? 1 : std::max<std::uint64_t>(1, budget / per_worker);
  const std::size_t workers = static_cast<std::size_t>(std::min<std::uint64_t>({static_cast<std::uint64_t>(cpus), blocks, fit}));
  result.lanes = lanes;
  result.workers = static_cast<int>(workers);
  result.bytes_per_worker = per_worker;

  const ForwardFn forward = lanes == 8 ? forward8 : lanes == 4 ? forward4 : forward1;
  std::atomic<std::size_t> next_block{0};
  std::exception_ptr error;
  std::mutex error_mutex;

  auto work = [&]() {
    try {
      auto f = std::make_unique_for_overwrite<double[]>(n * static_cast<std::size_t>(lanes));
      auto d = std::make_unique_for_overwrite<std::int32_t[]>(n * static_cast<std::size_t>(lanes));
      std::vector<double> scale_t(G * static_cast<std::size_t>(lanes));
      std::vector<long double> by_group(G);
      double mk[8];
      std::uint32_t sink[8];
      std::int32_t max_depth[8];
      for (;;) {
        const std::size_t block = next_block.fetch_add(1);
        if (block >= blocks) break;
        const std::size_t first = block * static_cast<std::size_t>(lanes);
        const std::size_t used = std::min<std::size_t>(static_cast<std::size_t>(lanes), S - first);
        for (std::size_t g = 0; g < G; ++g)
          for (std::size_t k = 0; k < static_cast<std::size_t>(lanes); ++k)
            scale_t[g * static_cast<std::size_t>(lanes) + k] = scale[(first + std::min(k, used - 1)) * G + g];
        forward(ctx, scale_t.data(), f.get(), d.get(), mk, sink, max_depth);
        for (std::size_t k = 0; k < used; ++k) {
          const std::size_t s = first + k;
          std::fill(by_group.begin(), by_group.end(), 0.0L);
          const Chain chain = trace_chain(ctx, f.get(), d.get(), lanes, static_cast<int>(k), sink[k], scale + s * G,
                                          is_comm, is_compute, size_bytes, by_group.data());
          result.makespan[s] = mk[k];
          result.depth[s] = max_depth[k];
          result.critical_compute[s] = static_cast<double>(chain.compute);
          result.critical_comm[s] = static_cast<double>(chain.comm);
          result.critical_comm_bytes[s] = chain.comm_bytes;
          for (std::size_t g = 0; g < G; ++g) result.critical_by_group[s * G + g] = static_cast<double>(by_group[g]);
        }
      }
    } catch (...) {
      next_block.store(blocks);
      std::lock_guard<std::mutex> lock(error_mutex);
      if (!error) error = std::current_exception();
    }
  };

  if (workers <= 1) {
    work();
  } else {
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (std::size_t i = 0; i < workers; ++i) pool.emplace_back(work);
    for (std::thread& t : pool) t.join();
  }
  if (error) std::rethrow_exception(error);
  return result;
}

}  // namespace cta::analysis
