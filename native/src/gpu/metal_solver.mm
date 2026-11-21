// Metal backend of the what-if scenario solver: one GPU thread per scenario, 32-bit floats.
//
// Apple GPUs have no 64-bit floats, so this is approximate (about 1e-5 relative; deep graphs sum
// more roundings). The rules are the CPU's: later finish, then greater depth, then lowest vertex
// id; a tie that the CPU sees in doubles can come out differently in floats.
//
// Layout: f[position * S + scenario] so the threads of a warp read neighbouring words. Long walks
// are cut into command buffers of kPositionsPerPass positions (and kTraceStepsPerPass trace steps)
// so no single dispatch runs long enough to trip the GPU watchdog; thread state lives in buffers
// between passes.
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include "gpu/backend.hpp"

namespace cta::gpu {
namespace {

constexpr std::uint32_t kPositionsPerPass = 1u << 16;
constexpr std::uint32_t kTraceStepsPerPass = 1u << 16;

NSString* const kShader = @R"MSL(
#pragma METAL fp contract(off)
#include <metal_stdlib>
using namespace metal;

struct Params {
  uint v0;
  uint v1;
  uint S;
  uint G;
  uint steps;
};

kernel void forward_k(device const ulong* pred_ptr [[buffer(0)]], device const uint* preds [[buffer(1)]],
                      device const uint* orig_of [[buffer(2)]], device const float* base [[buffer(3)]],
                      device const uint* grp [[buffer(4)]], device const float* scale_t [[buffer(5)]],
                      device float* f [[buffer(6)]], device int* d [[buffer(7)]], device float* mk [[buffer(8)]],
                      device uint* sink [[buffer(9)]], device int* maxd [[buffer(10)]],
                      constant Params& P [[buffer(11)]], uint s [[thread_position_in_grid]]) {
  if (s >= P.S) return;
  const ulong S = P.S;
  float mk_s = mk[s];
  uint sink_s = sink[s];
  int maxd_s = maxd[s];
  for (uint v = P.v0; v < P.v1; ++v) {
    const ulong first = pred_ptr[v], last = pred_ptr[v + 1];
    float best = 0.0f;
    int depth = 0;
    if (first == last) {
    } else if (last - first == 1) {
      const ulong u = preds[first];
      best = f[u * S + s];
      depth = d[u * S + s];
    } else {
      best = f[ulong(preds[first]) * S + s];
      for (ulong i = first + 1; i < last; ++i) best = max(best, f[ulong(preds[i]) * S + s]);
      depth = -1;
      uint winner = 0xFFFFFFFFu;
      for (ulong i = first; i < last; ++i) {
        const ulong u = preds[i];
        if (f[u * S + s] != best) continue;
        const int du = d[u * S + s];
        const uint id = orig_of[u];
        if (du > depth || (du == depth && id < winner)) {
          depth = du;
          winner = id;
        }
      }
    }
    const float finish = best + base[v] * scale_t[ulong(grp[v]) * S + s];
    const int deep = depth + 1;
    f[ulong(v) * S + s] = finish;
    d[ulong(v) * S + s] = deep;
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

kernel void trace_k(device const ulong* pred_ptr [[buffer(0)]], device const uint* preds [[buffer(1)]],
                    device const uint* orig_of [[buffer(2)]], device const float* base [[buffer(3)]],
                    device const uint* grp [[buffer(4)]], device const float* scale_t [[buffer(5)]],
                    device const float* f [[buffer(6)]], device const int* d [[buffer(7)]],
                    device const uchar* is_comm [[buffer(8)]], device const uchar* is_compute [[buffer(9)]],
                    device const long* size_bytes [[buffer(10)]], device uint* cur [[buffer(11)]],
                    device uchar* done [[buffer(12)]], device float* comp [[buffer(13)]],
                    device float* comm [[buffer(14)]], device long* bytes [[buffer(15)]],
                    device float* by_group [[buffer(16)]], constant Params& P [[buffer(17)]],
                    uint s [[thread_position_in_grid]]) {
  if (s >= P.S || done[s] != 0) return;
  const ulong S = P.S;
  uint v = cur[s];
  float comp_s = comp[s], comm_s = comm[s];
  long bytes_s = bytes[s];
  for (uint step = 0; step < P.steps; ++step) {
    const uint g = grp[v];
    const float w = base[v] * scale_t[ulong(g) * S + s];
    by_group[ulong(s) * P.G + g] += w;
    const uint id = orig_of[v];
    if (is_compute[id] != 0) comp_s += w;
    if (is_comm[id] != 0) {
      comm_s += w;
      bytes_s += size_bytes[id];
    }
    const ulong first = pred_ptr[v], last = pred_ptr[v + 1];
    if (first == last) {
      done[s] = 1;
      break;
    }
    ulong win = preds[first];
    float win_f = f[win * S + s];
    int win_d = d[win * S + s];
    for (ulong i = first + 1; i < last; ++i) {
      const ulong u = preds[i];
      const float fu = f[u * S + s];
      const int du = d[u * S + s];
      if (fu > win_f || (fu == win_f && (du > win_d || (du == win_d && orig_of[u] < orig_of[win])))) {
        win = u;
        win_f = fu;
        win_d = du;
      }
    }
    v = uint(win);
  }
  cur[s] = v;
  comp[s] = comp_s;
  comm[s] = comm_s;
  bytes[s] = bytes_s;
}
)MSL";

struct Params {
  std::uint32_t v0, v1, S, G, steps;
};

struct Metal {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> forward = nil;
  id<MTLComputePipelineState> trace = nil;
};

std::string describe(NSError* error) {
  return error == nil ? std::string("unknown error") : std::string([[error localizedDescription] UTF8String]);
}

Metal& metal() {
  static Metal m = [] {
    Metal made;
    made.device = MTLCreateSystemDefaultDevice();
    if (made.device == nil) return made;
    NSError* error = nil;
    MTLCompileOptions* options = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) {
      options.mathMode = MTLMathModeSafe;
    } else {
      options.fastMathEnabled = NO;
    }
    id<MTLLibrary> library = [made.device newLibraryWithSource:kShader options:options error:&error];
    if (library == nil) throw std::runtime_error("metal shader failed to compile: " + describe(error));
    for (int which = 0; which < 2; ++which) {
      id<MTLFunction> fn = [library newFunctionWithName:(which == 0 ? @"forward_k" : @"trace_k")];
      id<MTLComputePipelineState> pso = [made.device newComputePipelineStateWithFunction:fn error:&error];
      if (pso == nil) throw std::runtime_error("metal pipeline failed: " + describe(error));
      (which == 0 ? made.forward : made.trace) = pso;
    }
    made.queue = [made.device newCommandQueue];
    return made;
  }();
  return m;
}

id<MTLBuffer> shared_copy(id<MTLDevice> device, const void* data, std::size_t bytes) {
  id<MTLBuffer> buffer = [device newBufferWithBytes:data length:std::max<std::size_t>(bytes, 4)
                                            options:MTLResourceStorageModeShared];
  if (buffer == nil) throw std::runtime_error("metal could not allocate " + std::to_string(bytes) + " bytes");
  return buffer;
}

id<MTLBuffer> blank(id<MTLDevice> device, std::size_t bytes, MTLResourceOptions options) {
  id<MTLBuffer> buffer = [device newBufferWithLength:std::max<std::size_t>(bytes, 4) options:options];
  if (buffer == nil) throw std::runtime_error("metal could not allocate " + std::to_string(bytes) + " bytes");
  return buffer;
}

void run(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pso, std::size_t threads,
         void (^bind)(id<MTLComputeCommandEncoder>)) {
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pso];
  bind(enc);
  const NSUInteger width = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 64);
  [enc dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  if (cmd.status != MTLCommandBufferStatusCompleted)
    throw std::runtime_error("metal command failed: " + describe(cmd.error));
}

}  // namespace

bool metal_present() {
  static const bool present = [] {
    try {
      return metal().device != nil;
    } catch (...) {
      return false;
    }
  }();
  return present;
}

analysis::WhatIfResult metal_solve(const SolveInput& in, std::uint64_t memory_bytes) {
  @autoreleasepool {
    Metal& m = metal();
    if (m.device == nil) throw std::runtime_error("metal is not available: no Metal device found");
    const std::size_t n = in.plan->num_vertices(), S = in.num_scenarios, G = in.num_groups;
    if (n >= std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error("metal: too many vertices");

    // Memory: the graph once; per scenario f (4 B) and d (4 B) per vertex, plus the small outputs.
    std::uint64_t budget = memory_bytes;
    if (budget == 0) budget = static_cast<std::uint64_t>([m.device recommendedMaxWorkingSetSize]) / 10 * 8;
    const std::uint64_t fixed = (in.plan->pred_ptr().size() * 8ull) + (in.plan->preds().size() * 4ull) + n * 21ull;
    const std::uint64_t per_scenario = 8ull * n + 4ull * G + 64ull;
    if (budget <= fixed || (budget - fixed) < per_scenario)
      throw std::runtime_error("metal: a solve of " + std::to_string(n) + " vertices needs about " +
                               std::to_string(fixed + per_scenario) + " bytes and the budget is " +
                               std::to_string(budget));
    std::uint64_t chunk = (budget - fixed) / per_scenario;
    const std::uint64_t max_buffer = static_cast<std::uint64_t>([m.device maxBufferLength]);
    chunk = std::min<std::uint64_t>(chunk, max_buffer / std::max<std::uint64_t>(4ull * n, 1));
    chunk = std::min<std::uint64_t>({chunk, static_cast<std::uint64_t>(S), std::uint64_t{1} << 20});
    if (chunk == 0) throw std::runtime_error("metal: the graph does not fit one buffer of this device");

    // The graph, uploaded once.
    const auto& pred_ptr = in.plan->pred_ptr();
    const auto& preds = in.plan->preds();
    const auto& orig_of = in.plan->vertex_at();
    std::vector<float> base(n);
    for (std::size_t p = 0; p < n; ++p) base[p] = static_cast<float>(in.prepared->base[p]);
    std::vector<std::uint8_t> comm(n), compute(n);
    for (std::size_t i = 0; i < n; ++i) {
      comm[i] = in.is_comm[i] ? 1 : 0;
      compute[i] = in.is_compute[i] ? 1 : 0;
    }
    id<MTLBuffer> b_pred_ptr = shared_copy(m.device, pred_ptr.data(), pred_ptr.size() * 8);
    id<MTLBuffer> b_preds = shared_copy(m.device, preds.data(), preds.size() * 4);
    id<MTLBuffer> b_orig = shared_copy(m.device, orig_of.data(), orig_of.size() * 4);
    id<MTLBuffer> b_base = shared_copy(m.device, base.data(), n * 4);
    id<MTLBuffer> b_grp = shared_copy(m.device, in.prepared->group.data(), n * 4);
    id<MTLBuffer> b_comm = shared_copy(m.device, comm.data(), n);
    id<MTLBuffer> b_compute = shared_copy(m.device, compute.data(), n);
    id<MTLBuffer> b_size = shared_copy(m.device, in.size_bytes, n * 8);

    analysis::WhatIfResult result;
    result.makespan.assign(S, 0.0);
    result.depth.assign(S, 0);
    result.critical_compute.assign(S, 0.0);
    result.critical_comm.assign(S, 0.0);
    result.critical_comm_bytes.assign(S, 0);
    result.critical_by_group.assign(S * G, 0.0);
    result.device = "metal";
    result.lanes = 1;
    result.workers = 1;
    result.bytes_per_worker = per_scenario * chunk + fixed;

    for (std::size_t first = 0; first < S; first += chunk) {
      const std::size_t count = std::min<std::size_t>(chunk, S - first);
      std::vector<float> scale_t(G * count);
      for (std::size_t g = 0; g < G; ++g)
        for (std::size_t s = 0; s < count; ++s)
          scale_t[g * count + s] = static_cast<float>(in.scale[(first + s) * G + g]);
      id<MTLBuffer> b_scale = shared_copy(m.device, scale_t.data(), scale_t.size() * 4);
      id<MTLBuffer> b_f = blank(m.device, 4 * n * count, MTLResourceStorageModePrivate);
      id<MTLBuffer> b_d = blank(m.device, 4 * n * count, MTLResourceStorageModePrivate);
      id<MTLBuffer> b_mk = blank(m.device, 4 * count, MTLResourceStorageModeShared);
      id<MTLBuffer> b_sink = blank(m.device, 4 * count, MTLResourceStorageModeShared);
      id<MTLBuffer> b_maxd = blank(m.device, 4 * count, MTLResourceStorageModeShared);
      std::fill_n(static_cast<float*>([b_mk contents]), count, -1.0f);
      std::memset([b_sink contents], 0, 4 * count);
      std::memset([b_maxd contents], 0, 4 * count);

      for (std::size_t v0 = 0; v0 < n; v0 += kPositionsPerPass) {
        Params params{static_cast<std::uint32_t>(v0),
                      static_cast<std::uint32_t>(std::min<std::size_t>(n, v0 + kPositionsPerPass)),
                      static_cast<std::uint32_t>(count), static_cast<std::uint32_t>(G), 0};
        run(m.queue, m.forward, count, ^(id<MTLComputeCommandEncoder> enc) {
          const id<MTLBuffer> buffers[] = {b_pred_ptr, b_preds, b_orig, b_base, b_grp, b_scale, b_f, b_d, b_mk, b_sink, b_maxd};
          for (NSUInteger i = 0; i < 11; ++i) [enc setBuffer:buffers[i] offset:0 atIndex:i];
          [enc setBytes:&params length:sizeof params atIndex:11];
        });
      }

      // The critical chain of every scenario, from its sink back to a source.
      id<MTLBuffer> b_cur = blank(m.device, 4 * count, MTLResourceStorageModeShared);
      id<MTLBuffer> b_done = blank(m.device, count, MTLResourceStorageModeShared);
      id<MTLBuffer> b_comp = blank(m.device, 4 * count, MTLResourceStorageModeShared);
      id<MTLBuffer> b_commf = blank(m.device, 4 * count, MTLResourceStorageModeShared);
      id<MTLBuffer> b_bytes = blank(m.device, 8 * count, MTLResourceStorageModeShared);
      id<MTLBuffer> b_by = blank(m.device, 4 * count * G, MTLResourceStorageModeShared);
      std::memcpy([b_cur contents], [b_sink contents], 4 * count);
      std::memset([b_done contents], 0, count);
      std::memset([b_comp contents], 0, 4 * count);
      std::memset([b_commf contents], 0, 4 * count);
      std::memset([b_bytes contents], 0, 8 * count);
      std::memset([b_by contents], 0, 4 * count * G);
      Params params{0, 0, static_cast<std::uint32_t>(count), static_cast<std::uint32_t>(G), kTraceStepsPerPass};
      for (;;) {
        run(m.queue, m.trace, count, ^(id<MTLComputeCommandEncoder> enc) {
          const id<MTLBuffer> buffers[] = {b_pred_ptr, b_preds, b_orig, b_base, b_grp, b_scale, b_f, b_d,
                                           b_comm, b_compute, b_size, b_cur, b_done, b_comp, b_commf, b_bytes, b_by};
          for (NSUInteger i = 0; i < 17; ++i) [enc setBuffer:buffers[i] offset:0 atIndex:i];
          [enc setBytes:&params length:sizeof params atIndex:17];
        });
        const auto* done = static_cast<const std::uint8_t*>([b_done contents]);
        if (std::all_of(done, done + count, [](std::uint8_t x) { return x != 0; })) break;
      }

      const auto* mk = static_cast<const float*>([b_mk contents]);
      const auto* maxd = static_cast<const std::int32_t*>([b_maxd contents]);
      const auto* comp = static_cast<const float*>([b_comp contents]);
      const auto* commf = static_cast<const float*>([b_commf contents]);
      const auto* bytes = static_cast<const std::int64_t*>([b_bytes contents]);
      const auto* by = static_cast<const float*>([b_by contents]);
      for (std::size_t s = 0; s < count; ++s) {
        result.makespan[first + s] = static_cast<double>(mk[s]);
        result.depth[first + s] = maxd[s];
        result.critical_compute[first + s] = static_cast<double>(comp[s]);
        result.critical_comm[first + s] = static_cast<double>(commf[s]);
        result.critical_comm_bytes[first + s] = bytes[s];
        for (std::size_t g = 0; g < G; ++g) result.critical_by_group[(first + s) * G + g] = static_cast<double>(by[s * G + g]);
      }
    }
    return result;
  }
}

}  // namespace cta::gpu
