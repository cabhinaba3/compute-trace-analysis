// CUDA backend of the what-if scenario solver: one GPU thread per scenario, 64-bit floats.
//
// The arithmetic is the CPU's: `finish = best + base * scale` as a multiply and then an add, each
// rounded (no fused multiply-add: -fmad=false, and __dmul_rn/__dadd_rn below), and the same
// tie-breaking (later finish, then greater depth, then lowest vertex id), so the makespan and depth
// agree with the CPU solver exactly; the critical-path sums are doubles here and long doubles
// there, so they agree to rounding.
//
// Layout: f[position * S + scenario] so a warp reads neighbouring words. Walks are cut into passes
// (positions for the forward walk, steps for the trace) so no launch runs long enough to trip a
// display driver's watchdog; thread state lives in device memory between passes.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu/backend.hpp"

namespace cta::gpu {
namespace {

constexpr std::uint32_t kPositionsPerPass = 1u << 16;
constexpr std::uint32_t kTraceStepsPerPass = 1u << 16;
constexpr int kBlock = 128;

#define CTA_CUDA(call)                                                                                    \
  do {                                                                                                    \
    const cudaError_t cta_status = (call);                                                                \
    if (cta_status != cudaSuccess)                                                                        \
      throw std::runtime_error(std::string("cuda error: ") + cudaGetErrorString(cta_status) + " in " + #call); \
  } while (0)

// Frees a device pointer when it goes out of scope, so an error never leaks device memory.
template <typename T>
struct Device {
  T* ptr = nullptr;
  Device() = default;
  explicit Device(std::size_t count) {
    CTA_CUDA(cudaMalloc(reinterpret_cast<void**>(&ptr), std::max<std::size_t>(count, 1) * sizeof(T)));
  }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  Device(Device&& o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
  ~Device() {
    if (ptr) cudaFree(ptr);
  }
  void upload(const T* host, std::size_t count) {
    CTA_CUDA(cudaMemcpy(ptr, host, count * sizeof(T), cudaMemcpyHostToDevice));
  }
  void download(T* host, std::size_t count) const {
    CTA_CUDA(cudaMemcpy(host, ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
  }
  void zero(std::size_t count) { CTA_CUDA(cudaMemset(ptr, 0, std::max<std::size_t>(count, 1) * sizeof(T))); }
};

template <typename T>
Device<T> uploaded(const T* host, std::size_t count) {
  Device<T> d(count);
  if (count != 0) d.upload(host, count);
  return d;
}

__global__ void forward_k(const std::uint64_t* pred_ptr, const std::uint32_t* preds, const std::uint32_t* orig_of,
                          const double* base, const std::uint32_t* grp, const double* scale_t, double* f,
                          std::int32_t* d, double* mk, std::uint32_t* sink, std::int32_t* maxd, std::uint32_t v0,
                          std::uint32_t v1, std::uint32_t S) {
  const std::uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= S) return;
  const std::uint64_t stride = S;
  double mk_s = mk[s];
  std::uint32_t sink_s = sink[s];
  std::int32_t maxd_s = maxd[s];
  for (std::uint32_t v = v0; v < v1; ++v) {
    const std::uint64_t first = pred_ptr[v], last = pred_ptr[v + 1];
    double best = 0.0;
    std::int32_t depth = 0;
    if (first == last) {
    } else if (last - first == 1) {
      const std::uint64_t u = preds[first];
      best = f[u * stride + s];
      depth = d[u * stride + s];
    } else {
      best = f[static_cast<std::uint64_t>(preds[first]) * stride + s];
      for (std::uint64_t i = first + 1; i < last; ++i) {
        const double fu = f[static_cast<std::uint64_t>(preds[i]) * stride + s];
        best = fu > best ? fu : best;
      }
      depth = -1;
      std::uint32_t winner = 0xFFFFFFFFu;
      for (std::uint64_t i = first; i < last; ++i) {
        const std::uint64_t u = preds[i];
        if (f[u * stride + s] != best) continue;
        const std::int32_t du = d[u * stride + s];
        const std::uint32_t id = orig_of[u];
        if (du > depth || (du == depth && id < winner)) {
          depth = du;
          winner = id;
        }
      }
    }
    const double finish = __dadd_rn(best, __dmul_rn(base[v], scale_t[static_cast<std::uint64_t>(grp[v]) * stride + s]));
    const std::int32_t deep = depth + 1;
    f[static_cast<std::uint64_t>(v) * stride + s] = finish;
    d[static_cast<std::uint64_t>(v) * stride + s] = deep;
    if (deep > maxd_s) maxd_s = deep;
    if (finish > mk_s || (finish == mk_s && orig_of[v] < orig_of[sink_s])) {
      mk_s = finish;
      sink_s = v;
    }
  }
  mk[s] = mk_s;
  sink[s] = sink_s;
  maxd[s] = maxd_s;
}

__global__ void trace_k(const std::uint64_t* pred_ptr, const std::uint32_t* preds, const std::uint32_t* orig_of,
                        const double* base, const std::uint32_t* grp, const double* scale_t, const double* f,
                        const std::int32_t* d, const std::uint8_t* is_comm, const std::uint8_t* is_compute,
                        const std::int64_t* size_bytes, std::uint32_t* cur, std::uint8_t* done, double* comp,
                        double* comm, std::int64_t* bytes, double* by_group, std::uint32_t S, std::uint32_t G,
                        std::uint32_t steps) {
  const std::uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= S || done[s] != 0) return;
  const std::uint64_t stride = S;
  std::uint32_t v = cur[s];
  double comp_s = comp[s], comm_s = comm[s];
  std::int64_t bytes_s = bytes[s];
  for (std::uint32_t step = 0; step < steps; ++step) {
    const std::uint32_t g = grp[v];
    const double w = __dmul_rn(base[v], scale_t[static_cast<std::uint64_t>(g) * stride + s]);
    by_group[static_cast<std::uint64_t>(s) * G + g] = __dadd_rn(by_group[static_cast<std::uint64_t>(s) * G + g], w);
    const std::uint32_t id = orig_of[v];
    if (is_compute[id] != 0) comp_s = __dadd_rn(comp_s, w);
    if (is_comm[id] != 0) {
      comm_s = __dadd_rn(comm_s, w);
      bytes_s += size_bytes[id];
    }
    const std::uint64_t first = pred_ptr[v], last = pred_ptr[v + 1];
    if (first == last) {
      done[s] = 1;
      break;
    }
    std::uint64_t win = preds[first];
    double win_f = f[win * stride + s];
    std::int32_t win_d = d[win * stride + s];
    for (std::uint64_t i = first + 1; i < last; ++i) {
      const std::uint64_t u = preds[i];
      const double fu = f[u * stride + s];
      const std::int32_t du = d[u * stride + s];
      if (fu > win_f || (fu == win_f && (du > win_d || (du == win_d && orig_of[u] < orig_of[win])))) {
        win = u;
        win_f = fu;
        win_d = du;
      }
    }
    v = static_cast<std::uint32_t>(win);
  }
  cur[s] = v;
  comp[s] = comp_s;
  comm[s] = comm_s;
  bytes[s] = bytes_s;
}

}  // namespace

bool cuda_present() {
  static const bool present = [] {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
  }();
  return present;
}

analysis::WhatIfResult cuda_solve(const SolveInput& in, std::uint64_t memory_bytes) {
  const std::size_t n = in.plan->num_vertices(), S = in.num_scenarios, G = in.num_groups;
  if (n >= std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error("cuda: too many vertices");
  const auto& pred_ptr = in.plan->pred_ptr();
  const auto& preds = in.plan->preds();
  const auto& orig_of = in.plan->vertex_at();

  std::size_t free_bytes = 0, total_bytes = 0;
  CTA_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
  std::uint64_t budget = memory_bytes != 0 ? memory_bytes : static_cast<std::uint64_t>(free_bytes) / 10 * 8;
  // The graph once; per scenario f (8 B) and d (4 B) per vertex plus the per-scenario outputs.
  const std::uint64_t fixed = pred_ptr.size() * 8ull + preds.size() * 4ull + n * 30ull;
  const std::uint64_t per_scenario = 12ull * n + 16ull * G + 64ull;
  if (budget <= fixed || (budget - fixed) < per_scenario)
    throw std::runtime_error("cuda: a solve of " + std::to_string(n) + " vertices needs about " +
                             std::to_string(fixed + per_scenario) + " bytes and the budget is " +
                             std::to_string(budget));
  std::uint64_t chunk = std::min<std::uint64_t>((budget - fixed) / per_scenario, std::uint64_t{S});
  chunk = std::min<std::uint64_t>(chunk, std::uint64_t{1} << 22);

  std::vector<double> base(in.prepared->base);
  std::vector<std::uint8_t> comm(n), compute(n);
  for (std::size_t i = 0; i < n; ++i) {
    comm[i] = in.is_comm[i] ? 1 : 0;
    compute[i] = in.is_compute[i] ? 1 : 0;
  }
  Device<std::uint64_t> d_pred_ptr = uploaded(pred_ptr.data(), pred_ptr.size());
  Device<std::uint32_t> d_preds = uploaded(preds.data(), preds.size());
  Device<std::uint32_t> d_orig = uploaded(orig_of.data(), orig_of.size());
  Device<double> d_base = uploaded(base.data(), n);
  Device<std::uint32_t> d_grp = uploaded(in.prepared->group.data(), n);
  Device<std::uint8_t> d_comm = uploaded(comm.data(), n);
  Device<std::uint8_t> d_compute = uploaded(compute.data(), n);
  Device<std::int64_t> d_size = uploaded(in.size_bytes, n);

  analysis::WhatIfResult result;
  result.makespan.assign(S, 0.0);
  result.depth.assign(S, 0);
  result.critical_compute.assign(S, 0.0);
  result.critical_comm.assign(S, 0.0);
  result.critical_comm_bytes.assign(S, 0);
  result.critical_by_group.assign(S * G, 0.0);
  result.device = "cuda";
  result.lanes = 1;
  result.workers = 1;
  result.bytes_per_worker = per_scenario * chunk + fixed;

  for (std::size_t first = 0; first < S; first += chunk) {
    const std::size_t count = std::min<std::size_t>(chunk, S - first);
    std::vector<double> scale_t(G * count);
    for (std::size_t g = 0; g < G; ++g)
      for (std::size_t s = 0; s < count; ++s) scale_t[g * count + s] = in.scale[(first + s) * G + g];
    Device<double> d_scale = uploaded(scale_t.data(), scale_t.size());
    Device<double> d_f(n * count);
    Device<std::int32_t> d_d(n * count);
    Device<double> d_mk(count);
    Device<std::uint32_t> d_sink(count), d_cur(count);
    Device<std::int32_t> d_maxd(count);
    Device<std::uint8_t> d_done(count);
    Device<double> d_comp(count), d_commf(count), d_by(count * G);
    Device<std::int64_t> d_bytes(count);
    std::vector<double> mk_init(count, -1.0);
    d_mk.upload(mk_init.data(), count);
    d_sink.zero(count);
    d_maxd.zero(count);
    d_done.zero(count);
    d_comp.zero(count);
    d_commf.zero(count);
    d_by.zero(count * G);
    d_bytes.zero(count);

    const unsigned blocks = static_cast<unsigned>((count + kBlock - 1) / kBlock);
    for (std::size_t v0 = 0; v0 < n; v0 += kPositionsPerPass) {
      const std::uint32_t v1 = static_cast<std::uint32_t>(std::min<std::size_t>(n, v0 + kPositionsPerPass));
      forward_k<<<blocks, kBlock>>>(d_pred_ptr.ptr, d_preds.ptr, d_orig.ptr, d_base.ptr, d_grp.ptr, d_scale.ptr,
                                    d_f.ptr, d_d.ptr, d_mk.ptr, d_sink.ptr, d_maxd.ptr,
                                    static_cast<std::uint32_t>(v0), v1, static_cast<std::uint32_t>(count));
      CTA_CUDA(cudaGetLastError());
    }
    CTA_CUDA(cudaDeviceSynchronize());

    CTA_CUDA(cudaMemcpy(d_cur.ptr, d_sink.ptr, count * sizeof(std::uint32_t), cudaMemcpyDeviceToDevice));
    std::vector<std::uint8_t> done(count);
    for (;;) {
      trace_k<<<blocks, kBlock>>>(d_pred_ptr.ptr, d_preds.ptr, d_orig.ptr, d_base.ptr, d_grp.ptr, d_scale.ptr,
                                  d_f.ptr, d_d.ptr, d_comm.ptr, d_compute.ptr, d_size.ptr, d_cur.ptr, d_done.ptr,
                                  d_comp.ptr, d_commf.ptr, d_bytes.ptr, d_by.ptr, static_cast<std::uint32_t>(count),
                                  static_cast<std::uint32_t>(G), kTraceStepsPerPass);
      CTA_CUDA(cudaGetLastError());
      d_done.download(done.data(), count);  // synchronizes
      if (std::all_of(done.begin(), done.end(), [](std::uint8_t x) { return x != 0; })) break;
    }

    std::vector<double> mk(count), comp(count), commf(count), by(count * G);
    std::vector<std::int32_t> maxd(count);
    std::vector<std::int64_t> bytes(count);
    d_mk.download(mk.data(), count);
    d_maxd.download(maxd.data(), count);
    d_comp.download(comp.data(), count);
    d_commf.download(commf.data(), count);
    d_bytes.download(bytes.data(), count);
    d_by.download(by.data(), count * G);
    for (std::size_t s = 0; s < count; ++s) {
      result.makespan[first + s] = mk[s];
      result.depth[first + s] = maxd[s];
      result.critical_compute[first + s] = comp[s];
      result.critical_comm[first + s] = commf[s];
      result.critical_comm_bytes[first + s] = bytes[s];
      for (std::size_t g = 0; g < G; ++g) result.critical_by_group[(first + s) * G + g] = by[s * G + g];
    }
  }
  return result;
}

}  // namespace cta::gpu
