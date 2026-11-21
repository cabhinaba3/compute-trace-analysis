#include "streaming/engines/whatif/streaming_whatif.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "streaming/scratch.hpp"
#include "util/cpus.hpp"
#include "util/memory.hpp"

namespace cta::streaming::engines {
namespace {

#if defined(__x86_64__) && defined(__has_attribute)
#if __has_attribute(target_clones)
#define CTA_STREAM_MULTIARCH __attribute__((target_clones("avx2", "default")))
#endif
#endif
#ifndef CTA_STREAM_MULTIARCH
#define CTA_STREAM_MULTIARCH
#endif

constexpr std::uint32_t kNoPos = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint8_t kWinnerEscape = 0xFF;

template <typename T>
void write_exact(std::FILE* fp, const T* data, std::size_t count) {
  if (count == 0) return;
  if (std::fwrite(data, sizeof(T), count, fp) != count) throw std::runtime_error("streaming what-if write failed");
}

template <typename T>
void read_exact(std::FILE* fp, T* data, std::size_t count) {
  if (count == 0) return;
  if (std::fread(data, sizeof(T), count, fp) != count) throw std::runtime_error("streaming what-if read failed");
}

std::string to_upper_ascii(std::string_view s) {
  std::string out(s.size(), '\0');
  for (std::size_t i = 0; i < s.size(); ++i) {
    out[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[i])));
  }
  return out;
}

double ceil_log2_d(double n) { return std::ceil(std::log2(std::max(n, 1.0))); }

double collective_algo_cost_ns(const std::string& kind, double size_bytes, std::int64_t ranks,
                               const InterconnectSpec& ic, const std::string& algo) {
  if (ranks <= 1) return 0.0;
  const double n = static_cast<double>(ranks);
  const std::int64_t nodes_i =
      std::max<std::int64_t>(1, static_cast<std::int64_t>(std::ceil(n / static_cast<double>(ic.gpus_per_node))));
  const double k = static_cast<double>(nodes_i);
  const double share = (n - 1.0) / std::max(n, 1.0);
  const double log_n = ceil_log2_d(n);
  const double log_k = ceil_log2_d(k);

  const bool is_allreduce = (kind == "ALL_REDUCE");
  const bool is_gather = (kind == "ALL_GATHER" || kind == "REDUCE_SCATTER");
  const bool is_alltoall = (kind == "ALL_TO_ALL");
  const bool is_rooted = (kind == "BROADCAST" || kind == "REDUCE");
  const bool is_barrier = (kind == "BARRIER");
  const bool is_single = (kind == "SEND" || kind == "RECV");
  const bool is_other = !(is_allreduce || is_gather || is_alltoall || is_rooted || is_barrier || is_single);

  double steps = 0.0;
  double volume = 0.0;
  double inter = 0.0;

  if (is_single) {
    steps = 1.0;
    volume = 1.0;
  }
  if (algo == "ring") {
    if (is_allreduce) {
      steps = 2.0 * (n - 1.0);
      volume = 2.0 * share;
    } else if (is_gather || is_other || is_alltoall) {
      steps = n - 1.0;
      volume = share;
    } else if (is_rooted) {
      steps = n - 1.0;
      volume = 1.0;
    } else if (is_barrier) {
      steps = n - 1.0;
    }
    inter = k > 1.0 ? steps * k / std::max(n, 1.0) : 0.0;
  } else {
    if (is_allreduce) {
      steps = 2.0 * log_n;
      volume = 2.0 * log_n;
    } else if (is_gather || is_other) {
      steps = log_n;
      volume = share;
    } else if (is_alltoall) {
      steps = n - 1.0;
      volume = share;
    } else if (is_rooted) {
      steps = log_n;
      volume = log_n;
    } else if (is_barrier) {
      steps = log_n;
    }
    const double factor = is_allreduce ? 2.0 : 1.0;
    const double tree_inter = k > 1.0 ? factor * log_k : 0.0;
    const double ring_inter = k > 1.0 ? steps * k / std::max(n, 1.0) : 0.0;
    inter = is_alltoall ? ring_inter : tree_inter;
  }
  inter = std::min(inter, steps);
  const double intra = steps - inter;
  const double ns_per_byte = nodes_i > 1 ? ic.inter_ns_per_byte : ic.intra_ns_per_byte;
  return intra * ic.intra_alpha_ns + inter * ic.inter_alpha_ns + volume * size_bytes * ns_per_byte;
}

double single_node_interconnect_cost(const cache::TopoVertexMeta& vm, const std::vector<std::string>& comm_types,
                                     std::int64_t unique_ranks, const InterconnectSpec& ic,
                                     std::string* defaulted_kind = nullptr) {
  const std::uint8_t oc = vm.op_class;
  const bool is_p2p = (oc == static_cast<std::uint8_t>(OpClass::kCommSend) ||
                       oc == static_cast<std::uint8_t>(OpClass::kCommRecv));
  const bool is_coll = (oc == static_cast<std::uint8_t>(OpClass::kCommColl));
  if (!is_p2p && !is_coll) return 0.0;

  const double size = static_cast<double>(vm.size_bytes);
  const std::int64_t per_node = std::max<std::int64_t>(1, ic.gpus_per_node);

  if (is_p2p) {
    const std::int64_t src_node = vm.src_rank >= 0 ? static_cast<std::int64_t>(vm.src_rank) / per_node : -1;
    const std::int64_t dst_node = vm.dst_rank >= 0 ? static_cast<std::int64_t>(vm.dst_rank) / per_node : -1;
    const bool between = (src_node != dst_node) && (src_node >= 0) && (dst_node >= 0);
    const double alpha = between ? ic.inter_alpha_ns : ic.intra_alpha_ns;
    const double beta = between ? ic.inter_ns_per_byte : ic.intra_ns_per_byte;
    return 0.5 * (alpha + size * beta);
  }

  std::string kind;
  if (vm.comm_type_id >= 0 && static_cast<std::size_t>(vm.comm_type_id) < comm_types.size()) {
    kind = to_upper_ascii(comm_types[static_cast<std::size_t>(vm.comm_type_id)]);
  }
  std::int64_t ranks = std::max<std::int64_t>(unique_ranks, 1);
  auto it = ic.group_sizes.find(kind);
  if (it != ic.group_sizes.end()) {
    ranks = it->second;
  } else {
    auto star = ic.group_sizes.find("*");
    if (star != ic.group_sizes.end()) {
      ranks = star->second;
    } else if (unique_ranks > 1 && defaulted_kind != nullptr) {
      *defaulted_kind = kind;
    }
  }

  const std::int64_t nodes_i =
      std::max<std::int64_t>(1, static_cast<std::int64_t>(std::ceil(static_cast<double>(ranks) / per_node)));
  if (ic.algorithm == "flat") {
    const bool spans = nodes_i > 1;
    const double alpha = spans ? ic.inter_alpha_ns : ic.intra_alpha_ns;
    const double beta = spans ? ic.inter_ns_per_byte : ic.intra_ns_per_byte;
    return alpha + size * beta;
  }
  if (ic.algorithm == "auto") {
    const double r_cost = collective_algo_cost_ns(kind, size, ranks, ic, "ring");
    const double t_cost = collective_algo_cost_ns(kind, size, ranks, ic, "tree");
    return std::min(r_cost, t_cost);
  }
  return collective_algo_cost_ns(kind, size, ranks, ic, ic.algorithm);
}

// Per-vertex cross-block boundary payload for K lanes.
template <int K>
struct BoundaryRec {
  std::uint32_t orig_vid = 0;
  std::int32_t d[K]{};
  double f[K]{};
};

// Cross-block state store supporting BOTH:
//   1. Slot-recycled in-RAM pool using `last_use_pos` liveness (fast path when
//      max_cutwidth fits inside budget), AND
//   2. Direct-mapped paged disk spill (hard O(budget) RAM cap even for
//      adversarial Theta(V) cutwidth!).
template <int K>
class BoundaryStateStore {
 public:
  BoundaryStateStore(bool use_disk_spill, std::uint32_t num_blocks, std::uint32_t block_verts,
                     std::uint64_t page_cache_bytes, const std::string& spill_path)
      : use_spill_(use_disk_spill), block_verts_(std::max<std::uint32_t>(1, block_verts)) {
    if (use_spill_) {
      fp_ = std::fopen(spill_path.c_str(), "w+b");
      if (!fp_) throw std::runtime_error("cannot open what-if boundary spill file " + spill_path);
      const std::size_t rec_bytes = sizeof(BoundaryRec<K>);
      page_recs_ = std::max<std::size_t>(256, 65536 / rec_bytes);
      const std::size_t page_bytes = page_recs_ * rec_bytes;
      num_cache_slots_ = std::max<std::size_t>(4, static_cast<std::size_t>(page_cache_bytes / page_bytes));
      cache_tags_.assign(num_cache_slots_, kNoPos);
      cache_pages_.resize(num_cache_slots_);
      for (auto& p : cache_pages_) p.resize(page_recs_);
    } else {
      expire_after_block_.resize(num_blocks);
    }
  }

  ~BoundaryStateStore() {
    if (fp_) std::fclose(fp_);
  }

  void reset(std::uint32_t num_blocks) {
    if (use_spill_) {
      std::rewind(fp_);
      std::fill(cache_tags_.begin(), cache_tags_.end(), kNoPos);
      spilled_count_ = 0;
    } else {
      pos_to_slot_.clear();
      free_slots_.clear();
      pool_.clear();
      for (auto& v : expire_after_block_) v.clear();
      if (expire_after_block_.size() != num_blocks) expire_after_block_.resize(num_blocks);
    }
  }

  // Called once per finished block to persist all vertices needed by later blocks.
  void commit_block(std::uint32_t block_idx, std::uint32_t pos_begin, const cache::TopoBlock& block,
                    const double* block_f, const std::int32_t* block_d) {
    const std::size_t len = block.size();
    const std::uint32_t pos_end = block.pos_end;
    if (use_spill_) {
      // Write contiguous block records sequentially to flat spill file indexed by `pos`.
      if (std::fseek(fp_, static_cast<long>(static_cast<std::uint64_t>(pos_begin) * sizeof(BoundaryRec<K>)),
                     SEEK_SET) != 0)
        throw std::runtime_error("boundary spill seek failed");
      std::vector<BoundaryRec<K>> tmp(len);
      for (std::size_t i = 0; i < len; ++i) {
        tmp[i].orig_vid = block.vertices[i].orig_vid;
        for (int k = 0; k < K; ++k) {
          tmp[i].f[k] = block_f[i * K + k];
          tmp[i].d[k] = block_d[i * K + k];
        }
      }
      write_exact(fp_, tmp.data(), len);
      std::fflush(fp_);
      spilled_count_ = std::max<std::uint64_t>(spilled_count_, pos_end);
      // Invalidate any cached pages overlapped by this write.
      const std::uint32_t p0 = pos_begin / static_cast<std::uint32_t>(page_recs_);
      const std::uint32_t p1 = (pos_end == 0 ? 0 : (pos_end - 1) / static_cast<std::uint32_t>(page_recs_));
      for (std::uint32_t pg = p0; pg <= p1; ++pg) {
        const std::size_t slot = pg % num_cache_slots_;
        if (cache_tags_[slot] == pg) cache_tags_[slot] = kNoPos;
      }
      return;
    }

    // Slot-recycled in-RAM boundary pool: keep ONLY vertices whose last_use_pos >= pos_end.
    for (std::size_t i = 0; i < len; ++i) {
      const std::uint32_t last_use = block.vertices[i].last_use_pos;
      if (last_use < pos_end) continue;
      const std::uint32_t pos = pos_begin + static_cast<std::uint32_t>(i);
      std::uint32_t slot = 0;
      if (!free_slots_.empty()) {
        slot = free_slots_.back();
        free_slots_.pop_back();
      } else {
        slot = static_cast<std::uint32_t>(pool_.size());
        pool_.emplace_back();
      }
      BoundaryRec<K>& rec = pool_[slot];
      rec.orig_vid = block.vertices[i].orig_vid;
      for (int k = 0; k < K; ++k) {
        rec.f[k] = block_f[i * K + k];
        rec.d[k] = block_d[i * K + k];
      }
      pos_to_slot_[pos] = slot;
      const std::uint32_t expire_b = std::min<std::uint32_t>(
          static_cast<std::uint32_t>(expire_after_block_.size() - 1), last_use / block_verts_);
      expire_after_block_[expire_b].emplace_back(pos, slot);
    }

    // Recycle slots whose final consumer resided in `block_idx`.
    auto& exp = expire_after_block_[block_idx];
    for (const auto& [pos, slot] : exp) {
      pos_to_slot_.erase(pos);
      free_slots_.push_back(slot);
    }
    exp.clear();
  }

  const BoundaryRec<K>& lookup(std::uint32_t pos) {
    if (!use_spill_) {
      auto it = pos_to_slot_.find(pos);
      if (it == pos_to_slot_.end()) throw std::runtime_error("missing live boundary vertex in what-if pool");
      return pool_[it->second];
    }
    const std::uint32_t pg = pos / static_cast<std::uint32_t>(page_recs_);
    const std::size_t off = pos - static_cast<std::uint32_t>(pg * page_recs_);
    const std::size_t slot = pg % num_cache_slots_;
    if (cache_tags_[slot] != pg) {
      const std::uint64_t rec_start = static_cast<std::uint64_t>(pg) * page_recs_;
      const std::size_t count =
          static_cast<std::size_t>(std::min<std::uint64_t>(page_recs_, spilled_count_ - rec_start));
      if (std::fseek(fp_, static_cast<long>(rec_start * sizeof(BoundaryRec<K>)), SEEK_SET) != 0)
        throw std::runtime_error("boundary spill read seek failed");
      read_exact(fp_, cache_pages_[slot].data(), count);
      cache_tags_[slot] = pg;
    }
    return cache_pages_[slot][off];
  }

 private:
  bool use_spill_ = false;
  std::uint32_t block_verts_ = 1;
  // In-RAM slot-recycled pool state:
  std::vector<BoundaryRec<K>> pool_;
  std::vector<std::uint32_t> free_slots_;
  std::unordered_map<std::uint32_t, std::uint32_t> pos_to_slot_;
  std::vector<std::vector<std::pair<std::uint32_t, std::uint32_t>>> expire_after_block_;
  // Disk-backed direct-mapped page cache state:
  std::FILE* fp_ = nullptr;
  std::size_t page_recs_ = 0;
  std::size_t num_cache_slots_ = 0;
  std::uint64_t spilled_count_ = 0;
  std::vector<std::uint32_t> cache_tags_;
  std::vector<std::vector<BoundaryRec<K>>> cache_pages_;
};

// Escape entry when a vertex has >= 256 predecessors and winning pred index >= 255.
struct EscapeKey {
  std::uint32_t pos = 0;
  std::uint8_t lane = 0;
  bool operator==(const EscapeKey& o) const { return pos == o.pos && lane == o.lane; }
};

struct EscapeKeyHash {
  std::size_t operator()(const EscapeKey& k) const {
    return static_cast<std::size_t>((static_cast<std::uint64_t>(k.pos) << 4) ^ k.lane);
  }
};

// Hot SIMD-friendly K-lane forward kernel over one topological block.
// Compiled with -ffp-contract=off (and AVX2 multiarch on x86-64) for exact
// cross-CPU bit parity with native/src/analysis/whatif.cpp!
template <int K>
CTA_STREAM_MULTIARCH void forward_block_impl(
    const cache::TopoBlock& block, const double* base_w, const std::uint32_t* block_grp, const double* scale_t,
    BoundaryStateStore<K>& boundary, double* block_f, std::int32_t* block_d, double* mk, std::uint32_t* sink_pos,
    std::uint32_t* sink_orig_vid, std::int32_t* max_depth, std::vector<std::uint8_t>& block_winners,
    std::unordered_map<EscapeKey, std::uint32_t, EscapeKeyHash>& escape_winners) {
  const std::uint32_t pos_begin = block.pos_begin;
  const std::size_t len = block.size();
  block_winners.clear();

  for (std::size_t v = 0; v < len; ++v) {
    const std::uint64_t first = block.pred_ptr[v];
    const std::uint64_t last = block.pred_ptr[v + 1];
    const std::uint64_t deg = last - first;

    double best_f[K];
    std::int32_t best_d[K];

    if (deg == 0) {
      for (int k = 0; k < K; ++k) {
        best_f[k] = 0.0;
        best_d[k] = 0;
      }
    } else if (deg == 1) {
      const std::uint32_t u = block.preds[first];
      if (u >= pos_begin) {
        const std::size_t lu = u - pos_begin;
        for (int k = 0; k < K; ++k) {
          best_f[k] = block_f[lu * K + k];
          best_d[k] = block_d[lu * K + k];
        }
      } else {
        const auto& rec = boundary.lookup(u);
        for (int k = 0; k < K; ++k) {
          best_f[k] = rec.f[k];
          best_d[k] = rec.d[k];
        }
      }
    } else {
      // deg >= 2: exact tie-breaking (f DESC, d DESC, orig_vid ASC) matching whatif.cpp.
      std::uint32_t best_vid[K];
      std::uint64_t best_idx[K];
      std::uint32_t best_pos[K];
      for (int k = 0; k < K; ++k) {
        best_f[k] = -1.0;
        best_d[k] = -1;
        best_vid[k] = std::numeric_limits<std::uint32_t>::max();
        best_idx[k] = 0;
        best_pos[k] = 0;
      }
      for (std::uint64_t i = first; i < last; ++i) {
        const std::uint32_t u = block.preds[i];
        const std::uint64_t local_idx = i - first;
        if (u >= pos_begin) {
          const std::size_t lu = u - pos_begin;
          const std::uint32_t uid = block.vertices[lu].orig_vid;
          const double* uf = block_f + lu * K;
          const std::int32_t* ud = block_d + lu * K;
          for (int k = 0; k < K; ++k) {
            const double fu = uf[k];
            const std::int32_t du = ud[k];
            if (fu > best_f[k] ||
                (fu == best_f[k] && (du > best_d[k] || (du == best_d[k] && uid < best_vid[k])))) {
              best_f[k] = fu;
              best_d[k] = du;
              best_vid[k] = uid;
              best_idx[k] = local_idx;
              best_pos[k] = u;
            }
          }
        } else {
          const auto& rec = boundary.lookup(u);
          const std::uint32_t uid = rec.orig_vid;
          for (int k = 0; k < K; ++k) {
            const double fu = rec.f[k];
            const std::int32_t du = rec.d[k];
            if (fu > best_f[k] ||
                (fu == best_f[k] && (du > best_d[k] || (du == best_d[k] && uid < best_vid[k])))) {
              best_f[k] = fu;
              best_d[k] = du;
              best_vid[k] = uid;
              best_idx[k] = local_idx;
              best_pos[k] = u;
            }
          }
        }
      }
      // Append K compact 1-byte winner codes for this join vertex (0xFF -> escape map).
      const std::uint32_t cur_pos = pos_begin + static_cast<std::uint32_t>(v);
      for (int k = 0; k < K; ++k) {
        if (best_idx[k] < kWinnerEscape) {
          block_winners.push_back(static_cast<std::uint8_t>(best_idx[k]));
        } else {
          block_winners.push_back(kWinnerEscape);
          escape_winners[EscapeKey{cur_pos, static_cast<std::uint8_t>(k)}] = best_pos[k];
        }
      }
    }

    const cache::TopoVertexMeta& vm = block.vertices[v];
    const bool is_virt = (vm.flags & cache::kFlagIsVirtual) != 0;
    const std::int32_t d_inc = is_virt ? 0 : 1;
    const double bw = base_w[v];
    const double* scale = scale_t + static_cast<std::size_t>(block_grp[v]) * K;
    const std::uint32_t id = vm.orig_vid;
    const std::uint32_t cur_pos = pos_begin + static_cast<std::uint32_t>(v);

    for (int k = 0; k < K; ++k) {
      const double finish = best_f[k] + bw * scale[k];
      const std::int32_t deep = best_d[k] + d_inc;
      block_f[v * K + k] = finish;
      block_d[v * K + k] = deep;
      if (deep > max_depth[k]) max_depth[k] = deep;
      if (!is_virt && (finish > mk[k] || (finish == mk[k] && id < sink_orig_vid[k]))) {
        mk[k] = finish;
        sink_pos[k] = cur_pos;
        sink_orig_vid[k] = id;
      }
    }
  }
}

template <int K>
void run_worker_batches(const std::string& cache_path, const std::string& groups_path,
                        const std::uint32_t* inline_groups, std::size_t num_groups, const double* scale,
                        std::size_t num_scenarios, const std::vector<RepriceStep>& reprice_steps,
                        bool use_boundary_spill, std::uint64_t boundary_budget_bytes, const ScratchDir& scratch,
                        int worker_id, std::atomic<std::size_t>& next_batch, std::size_t total_batches,
                        StreamingWhatIfResult& result, std::exception_ptr& error, std::mutex& error_mutex) {
  try {
    cache::GraphCacheReader reader(cache_path);
    const auto& hdr = reader.header();
    const std::uint32_t num_blocks = hdr.num_blocks;
    const std::uint32_t block_verts = std::max<std::uint32_t>(1, hdr.block_vertices);
    const std::int64_t unique_ranks = static_cast<std::int64_t>(reader.rank_stats().size());

    std::FILE* grp_fp = nullptr;
    if (inline_groups == nullptr) {
      grp_fp = std::fopen(groups_path.c_str(), "rb");
      if (!grp_fp) throw std::runtime_error("cannot open what-if groups file " + groups_path);
    }
    struct FileGuard {
      std::FILE* fp = nullptr;
      ~FileGuard() {
        if (fp) std::fclose(fp);
      }
    } grp_guard{grp_fp};

    const std::string win_path = scratch.file("whatif_win_w" + std::to_string(worker_id) + ".bin");
    std::FILE* win_fp = std::fopen(win_path.c_str(), "w+b");
    if (!win_fp) throw std::runtime_error("cannot open winner log scratch file " + win_path);
    FileGuard win_guard{win_fp};

    const std::string spill_path = scratch.file("whatif_bnd_w" + std::to_string(worker_id) + ".bin");
    BoundaryStateStore<K> boundary(use_boundary_spill, num_blocks, block_verts, boundary_budget_bytes, spill_path);

    cache::TopoBlock block;
    std::vector<double> base_w;
    std::vector<std::uint32_t> block_grp;
    std::vector<double> block_f(static_cast<std::size_t>(block_verts) * K);
    std::vector<std::int32_t> block_d(static_cast<std::size_t>(block_verts) * K);
    std::vector<double> scale_t(num_groups * K);
    std::vector<std::uint8_t> block_winners;
    std::vector<std::uint64_t> win_offsets(num_blocks, 0);
    std::vector<std::uint64_t> win_sizes(num_blocks, 0);
    std::unordered_map<EscapeKey, std::uint32_t, EscapeKeyHash> escape_winners;

    auto read_block_groups = [&](std::uint32_t pos_begin, std::size_t len) {
      block_grp.resize(len);
      if (inline_groups != nullptr) {
        std::memcpy(block_grp.data(), inline_groups + pos_begin, len * sizeof(std::uint32_t));
      } else {
        if (std::fseek(grp_fp, static_cast<long>(static_cast<std::uint64_t>(pos_begin) * sizeof(std::uint32_t)),
                       SEEK_SET) != 0)
          throw std::runtime_error("groups file seek failed");
        read_exact(grp_fp, block_grp.data(), len);
      }
      for (std::size_t i = 0; i < len; ++i) {
        if (block_grp[i] >= num_groups) throw std::invalid_argument("a group id is outside 0..num_groups-1");
      }
    };

    auto compute_block_weights = [&]() {
      const std::size_t len = block.size();
      base_w.resize(len);
      if (reprice_steps.empty()) {
        for (std::size_t i = 0; i < len; ++i) base_w[i] = block.vertices[i].weight_ns;
      } else {
        for (std::size_t i = 0; i < len; ++i) {
          base_w[i] = effective_vertex_weight_ns(block.vertices[i], reader.comm_types(), unique_ranks, reprice_steps);
        }
      }
    };

    for (;;) {
      const std::size_t batch = next_batch.fetch_add(1);
      if (batch >= total_batches) break;

      const std::size_t first_s = batch * K;
      const std::size_t used = std::min<std::size_t>(K, num_scenarios - first_s);
      for (std::size_t g = 0; g < num_groups; ++g) {
        for (int k = 0; k < K; ++k) {
          const std::size_t s = first_s + std::min<std::size_t>(static_cast<std::size_t>(k), used - 1);
          scale_t[g * K + k] = scale[s * num_groups + g];
        }
      }

      boundary.reset(num_blocks);
      escape_winners.clear();
      // Rewind & reuse winner_log file for this batch (zero cumulative disk growth!).
      if (std::fseek(win_fp, 0, SEEK_SET) != 0) throw std::runtime_error("winner log rewind failed");
      std::uint64_t win_cursor = 0;

      double mk[K];
      std::uint32_t sink_pos[K];
      std::uint32_t sink_orig_vid[K];
      std::int32_t max_depth[K];
      for (int k = 0; k < K; ++k) {
        mk[k] = -1.0;
        sink_pos[k] = 0;
        sink_orig_vid[k] = std::numeric_limits<std::uint32_t>::max();
        max_depth[k] = 0;
      }

      // Forward sweep over topological blocks 0 .. num_blocks - 1.
      for (std::uint32_t b = 0; b < num_blocks; ++b) {
        reader.read_block(b, block);
        const std::size_t len = block.size();
        if (len == 0) {
          win_offsets[b] = win_cursor;
          win_sizes[b] = 0;
          continue;
        }
        if (block_f.size() < len * K) {
          block_f.resize(len * K);
          block_d.resize(len * K);
        }
        read_block_groups(block.pos_begin, len);
        compute_block_weights();

        forward_block_impl<K>(block, base_w.data(), block_grp.data(), scale_t.data(), boundary, block_f.data(),
                              block_d.data(), mk, sink_pos, sink_orig_vid, max_depth, block_winners, escape_winners);

        win_offsets[b] = win_cursor;
        win_sizes[b] = block_winners.size();
        if (!block_winners.empty()) {
          write_exact(win_fp, block_winners.data(), block_winners.size());
          win_cursor += block_winners.size();
        }
        boundary.commit_block(b, block.pos_begin, block, block_f.data(), block_d.data());
      }
      std::fflush(win_fp);

      // Simultaneous single reverse sweep across touched blocks for ALL `used` lanes!
      std::uint32_t cur_pos[K];
      bool active[K];
      long double chain_compute[K]{};
      long double chain_comm[K]{};
      std::int64_t chain_comm_bytes[K]{};
      std::vector<long double> by_group(used * num_groups, 0.0L);

      for (std::size_t k = 0; k < used; ++k) {
        cur_pos[k] = sink_pos[k];
        active[k] = (hdr.num_real_vertices > 0 && mk[k] >= 0.0);
      }

      std::vector<std::uint32_t> join_slot;
      for (std::int64_t bi = static_cast<std::int64_t>(num_blocks) - 1; bi >= 0; --bi) {
        const std::uint32_t b = static_cast<std::uint32_t>(bi);
        const auto& idx = reader.block_index()[b];
        bool needed = false;
        for (std::size_t k = 0; k < used; ++k) {
          if (active[k] && cur_pos[k] >= idx.pos_begin && cur_pos[k] < idx.pos_end) {
            needed = true;
            break;
          }
        }
        if (!needed) continue;

        reader.read_block(b, block);
        const std::size_t len = block.size();
        read_block_groups(block.pos_begin, len);
        compute_block_weights();

        // Load compact 1-byte winner log for this block and build local join offset map.
        block_winners.resize(static_cast<std::size_t>(win_sizes[b]));
        if (!block_winners.empty()) {
          if (std::fseek(win_fp, static_cast<long>(win_offsets[b]), SEEK_SET) != 0)
            throw std::runtime_error("winner log read seek failed");
          read_exact(win_fp, block_winners.data(), block_winners.size());
        }
        join_slot.resize(len);
        std::uint32_t jcnt = 0;
        for (std::size_t i = 0; i < len; ++i) {
          if (block.pred_ptr[i + 1] - block.pred_ptr[i] >= 2) {
            join_slot[i] = jcnt++;
          } else {
            join_slot[i] = kNoPos;
          }
        }

        // Advance every lane currently inside this block until it leaves or reaches a source.
        for (std::size_t k = 0; k < used; ++k) {
          const std::size_t s = first_s + k;
          const double* sc = scale + s * num_groups;
          while (active[k] && cur_pos[k] >= block.pos_begin && cur_pos[k] < block.pos_end) {
            const std::size_t local = cur_pos[k] - block.pos_begin;
            const cache::TopoVertexMeta& vm = block.vertices[local];
            const std::uint32_t g = block_grp[local];
            const double w = base_w[local] * sc[g];
            if ((vm.flags & cache::kFlagIsVirtual) == 0) {
              by_group[k * num_groups + g] += static_cast<long double>(w);
              if (vm.flags & cache::kFlagIsCompute) chain_compute[k] += static_cast<long double>(w);
              if (vm.flags & cache::kFlagIsComm) {
                chain_comm[k] += static_cast<long double>(w);
                chain_comm_bytes[k] += static_cast<std::int64_t>(vm.size_bytes);
              }
            }
            const std::uint64_t first = block.pred_ptr[local];
            const std::uint64_t last = block.pred_ptr[local + 1];
            const std::uint64_t deg = last - first;
            if (deg == 0) {
              active[k] = false;
              break;
            } else if (deg == 1) {
              cur_pos[k] = block.preds[first];
            } else {
              const std::uint32_t js = join_slot[local];
              const std::uint8_t code = block_winners[static_cast<std::size_t>(js) * K + k];
              if (code < kWinnerEscape) {
                cur_pos[k] = block.preds[first + code];
              } else {
                auto eit = escape_winners.find(EscapeKey{cur_pos[k], static_cast<std::uint8_t>(k)});
                if (eit == escape_winners.end()) throw std::runtime_error("missing escape winner in what-if log");
                cur_pos[k] = eit->second;
              }
            }
          }
        }
      }

      for (std::size_t k = 0; k < used; ++k) {
        const std::size_t s = first_s + k;
        result.makespan[s] = mk[k] < 0.0 ? 0.0 : mk[k];
        result.depth[s] = max_depth[k];
        result.critical_compute[s] = static_cast<double>(chain_compute[k]);
        result.critical_comm[s] = static_cast<double>(chain_comm[k]);
        result.critical_comm_bytes[s] = chain_comm_bytes[k];
        for (std::size_t g = 0; g < num_groups; ++g) {
          result.critical_by_group[s * num_groups + g] = static_cast<double>(by_group[k * num_groups + g]);
        }
      }
    }
  } catch (...) {
    next_batch.store(total_batches);
    std::lock_guard<std::mutex> lock(error_mutex);
    if (!error) error = std::current_exception();
  }
}

}  // namespace

double effective_vertex_weight_ns(const cache::TopoVertexMeta& vm, const std::vector<std::string>& comm_types,
                                  std::int64_t unique_ranks, const std::vector<RepriceStep>& steps) {
  double w = vm.weight_ns;
  if (steps.empty() || (vm.flags & cache::kFlagIsComm) == 0) return w;
  const bool recorded = vm.duration_ns > 0;
  for (const RepriceStep& step : steps) {
    const double cost_new = single_node_interconnect_cost(vm, comm_types, unique_ranks, step.target);
    if (!recorded) {
      w = cost_new;
    } else if (step.has_base) {
      const double cost_base = single_node_interconnect_cost(vm, comm_types, unique_ranks, step.base);
      const double ratio = cost_base > 0.0 ? (cost_new / cost_base) : 1.0;
      w *= ratio;
    }
  }
  return w;
}

RepriceAudit inspect_reprice_warnings(const cache::GraphCacheReader& reader, const RepriceStep& step) {
  RepriceAudit audit;
  const auto& hdr = reader.header();
  const std::int64_t unique_ranks = static_cast<std::int64_t>(reader.rank_stats().size());
  audit.has_recorded_comm_without_base = (hdr.has_recorded_comm_times != 0) && !step.has_base;

  if (unique_ranks <= 1) return audit;
  // Scan collective op comm_type_ids block-by-block only if needed to discover
  // defaulted collective types (exact parity with `interconnect.node_cost_ns`).
  const bool target_has_star = step.target.group_sizes.count("*") > 0;
  const bool base_has_star = !step.has_base || step.base.group_sizes.count("*") > 0;
  if (target_has_star && base_has_star) return audit;

  std::set<std::string> defaulted;
  cache::TopoBlock block;
  for (std::uint32_t b = 0; b < hdr.num_blocks; ++b) {
    reader.read_block(b, block);
    for (const auto& vm : block.vertices) {
      if (vm.op_class != static_cast<std::uint8_t>(OpClass::kCommColl)) continue;
      std::string kind;
      if (vm.comm_type_id >= 0 && static_cast<std::size_t>(vm.comm_type_id) < reader.comm_types().size()) {
        kind = to_upper_ascii(reader.comm_types()[static_cast<std::size_t>(vm.comm_type_id)]);
      }
      if (!target_has_star && step.target.group_sizes.count(kind) == 0) defaulted.insert(kind);
      if (step.has_base && !base_has_star && step.base.group_sizes.count(kind) == 0) defaulted.insert(kind);
    }
  }
  audit.defaulted_collectives.assign(defaulted.begin(), defaulted.end());
  return audit;
}

BlockMetaView read_block_meta(const cache::GraphCacheReader& reader, std::uint32_t block_idx,
                              const std::vector<RepriceStep>& reprice_steps) {
  cache::TopoBlock block;
  reader.read_block(block_idx, block);
  const std::size_t len = block.size();
  const std::int64_t unique_ranks = static_cast<std::int64_t>(reader.rank_stats().size());

  BlockMetaView out;
  out.rank.resize(len);
  out.op_class.resize(len);
  out.comm_type_id.resize(len);
  out.name_id.resize(len);
  out.local_names.resize(len);
  out.weight_ns.resize(len);
  out.is_comm.resize(len);
  out.is_compute.resize(len);
  out.is_virtual.resize(len);
  out.size_bytes.resize(len);

  for (std::size_t i = 0; i < len; ++i) {
    const auto& vm = block.vertices[i];
    out.rank[i] = vm.rank;
    out.op_class[i] = vm.op_class;
    out.comm_type_id[i] = vm.comm_type_id;
    out.name_id[i] = vm.name_id;
    if (vm.name_id < 0) {
      out.local_names[i] = std::string(block.name_of(i, reader.names()));
    }
    out.weight_ns[i] = effective_vertex_weight_ns(vm, reader.comm_types(), unique_ranks, reprice_steps);
    out.is_comm[i] = (vm.flags & cache::kFlagIsComm) != 0 ? 1 : 0;
    out.is_compute[i] = (vm.flags & cache::kFlagIsCompute) != 0 ? 1 : 0;
    out.is_virtual[i] = (vm.flags & cache::kFlagIsVirtual) != 0 ? 1 : 0;
    out.size_bytes[i] = vm.size_bytes;
  }
  return out;
}

StreamingWhatIfResult solve_whatif_streaming(const std::string& cache_path, const std::string& groups_path,
                                             const std::uint32_t* inline_groups, std::size_t inline_groups_len,
                                             std::size_t num_groups, const double* scale, std::size_t num_scenarios,
                                             const std::vector<RepriceStep>& reprice_steps, int threads,
                                             std::uint64_t memory_budget_bytes, const std::string& scratch_dir) {
  cache::GraphCacheReader reader(cache_path);
  const auto& hdr = reader.header();
  const std::size_t S = num_scenarios;
  const std::size_t G = num_groups;

  StreamingWhatIfResult result;
  result.makespan.assign(S, 0.0);
  result.depth.assign(S, 0);
  result.critical_compute.assign(S, 0.0);
  result.critical_comm.assign(S, 0.0);
  result.critical_comm_bytes.assign(S, 0);
  result.critical_by_group.assign(S * G, 0.0);
  if (S == 0 || hdr.num_real_vertices == 0) return result;

  if (inline_groups != nullptr && inline_groups_len != hdr.num_total_vertices) {
    throw std::invalid_argument("inline_groups length does not match graph cache vertex count");
  }
  for (std::size_t i = 0; i < S * G; ++i) {
    if (!std::isfinite(scale[i]) || scale[i] < 0.0) {
      throw std::invalid_argument("a multiplier is negative or not finite");
    }
  }

  const int cpus = threads > 0 ? threads : usable_cpus();
  const std::uint64_t budget = resolve_memory_budget(memory_budget_bytes);
  const std::size_t target = std::min<std::size_t>(static_cast<std::size_t>(cpus), S);

  // Per-worker RAM for lane width K:
  //   block buffers: ~12 * K * block_vertices + 80 * block_vertices
  //   boundary state: min(max_cutwidth * (32 + 12*K), boundary_cap)
  const std::uint64_t bverts = std::max<std::uint64_t>(1, hdr.block_vertices);
  const std::uint64_t cut = hdr.max_cutwidth;

  int lanes = 1;
  std::uint64_t per_worker = 0;
  bool use_spill = false;
  const std::uint64_t boundary_budget = std::max<std::uint64_t>(16ull * 1024, budget / 4);

  for (int candidate : {8, 4, 1}) {
    const std::uint64_t block_ram = (12ull * static_cast<std::uint64_t>(candidate) + 80ull) * bverts;
    const std::uint64_t pool_ram = cut * (64ull + 12ull * static_cast<std::uint64_t>(candidate));
    const bool need_spill = (pool_ram > boundary_budget);
    const std::uint64_t bnd_ram = need_spill ? std::min<std::uint64_t>(boundary_budget, 4ull << 20) : pool_ram;
    const std::uint64_t total_ram = block_ram + bnd_ram;
    if (candidate != 1 && total_ram > budget) continue;
    if (candidate != 1 && (S + static_cast<std::size_t>(candidate) - 1) / static_cast<std::size_t>(candidate) < target)
      continue;
    lanes = candidate;
    per_worker = total_ram;
    use_spill = need_spill;
    break;
  }

  const std::size_t total_batches = (S + static_cast<std::size_t>(lanes) - 1) / static_cast<std::size_t>(lanes);
  const std::uint64_t fit = per_worker == 0 ? 1 : std::max<std::uint64_t>(1, budget / per_worker);
  const std::size_t workers = static_cast<std::size_t>(
      std::min<std::uint64_t>({static_cast<std::uint64_t>(cpus), static_cast<std::uint64_t>(total_batches), fit}));
  const std::uint64_t per_worker_bnd =
      std::max<std::uint64_t>(16ull * 1024, boundary_budget / std::max<std::size_t>(1, workers));

  result.lanes = lanes;
  result.workers = static_cast<int>(workers);
  result.bytes_per_worker = per_worker;
  result.used_boundary_spill = use_spill;

  ScratchDir scratch(scratch_dir);
  std::atomic<std::size_t> next_batch{0};
  std::exception_ptr error;
  std::mutex error_mutex;

  auto launch = [&](int wid) {
    if (lanes == 8) {
      run_worker_batches<8>(cache_path, groups_path, inline_groups, G, scale, S, reprice_steps, use_spill,
                            per_worker_bnd, scratch, wid, next_batch, total_batches, result, error, error_mutex);
    } else if (lanes == 4) {
      run_worker_batches<4>(cache_path, groups_path, inline_groups, G, scale, S, reprice_steps, use_spill,
                            per_worker_bnd, scratch, wid, next_batch, total_batches, result, error, error_mutex);
    } else {
      run_worker_batches<1>(cache_path, groups_path, inline_groups, G, scale, S, reprice_steps, use_spill,
                            per_worker_bnd, scratch, wid, next_batch, total_batches, result, error, error_mutex);
    }
  };

  if (workers <= 1) {
    launch(0);
  } else {
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (std::size_t w = 0; w < workers; ++w) pool.emplace_back(launch, static_cast<int>(w));
    for (std::thread& t : pool) t.join();
  }
  if (error) std::rethrow_exception(error);
  return result;
}

}  // namespace cta::streaming::engines
