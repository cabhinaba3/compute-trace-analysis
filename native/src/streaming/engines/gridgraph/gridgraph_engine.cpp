#include "streaming/engines/gridgraph/gridgraph_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "streaming/attribution/scalasca_delay.hpp"
#include "streaming/cache/graph_cache.hpp"
#include "streaming/scratch.hpp"
#include "util/compensated_sum.hpp"
#include "util/memory.hpp"

namespace cta::streaming::engines {
namespace {

constexpr std::uint32_t kNoPos = std::numeric_limits<std::uint32_t>::max();

template <typename T>
void write_exact(std::FILE* fp, const T* data, std::size_t count) {
  if (count == 0) return;
  if (std::fwrite(data, sizeof(T), count, fp) != count) throw std::runtime_error("gridgraph scratch write failed");
}

template <typename T>
void read_exact(std::FILE* fp, T* data, std::size_t count) {
  if (count == 0) return;
  if (std::fread(data, sizeof(T), count, fp) != count) throw std::runtime_error("gridgraph scratch read failed");
}

// Per-vertex Pass 1 summary written sequentially to scratch (32 bytes/vertex).
struct ForwardRec {
  double es = 0.0;
  double ef = 0.0;
  std::int32_t depth = 1;
  std::uint32_t parent_pos = kNoPos;
  std::uint32_t orig_vid = 0;
  std::uint32_t pad = 0;
};

// Reverse delta pushed across block boundaries during Pass 2.
struct RevSlackDelta {
  std::uint32_t target_pos = 0;
  std::uint32_t pad = 0;
  double bound_lf = 0.0;
};

struct RevDelayDelta {
  std::uint32_t target_pos = 0;
  std::uint32_t pad = 0;
  double delay_ns = 0.0;
};

// Bounded-RAM cache for cross-block forward DP records (LRU/windowed block
// reads from forward.bin so even adversarial Theta(V) cutwidth stays strictly
// inside memory_budget_bytes!).
class ForwardStateStore {
 public:
  ForwardStateStore(const std::string& path, std::uint32_t block_verts, std::size_t max_cached_blocks)
      : path_(path), block_verts_(block_verts), max_cached_(std::max<std::size_t>(2, max_cached_blocks)) {
    fp_ = std::fopen(path.c_str(), "w+b");
    if (!fp_) throw std::runtime_error("cannot open forward scratch file " + path);
  }
  ~ForwardStateStore() {
    if (fp_) std::fclose(fp_);
  }

  void append_block(std::uint32_t block_idx, const std::vector<ForwardRec>& recs) {
    if (std::fseek(fp_, 0, SEEK_END) != 0) throw std::runtime_error("forward scratch seek failed");
    write_exact(fp_, recs.data(), recs.size());
    std::fflush(fp_);
    cache_block(block_idx, recs);
  }

  ForwardRec read_pos(std::uint32_t pos) {
    const std::uint32_t b = pos / block_verts_;
    const std::uint32_t off = pos - b * block_verts_;
    auto it = cached_.find(b);
    if (it != cached_.end()) return it->second[off];
    // Load the target block (or single record if cache budget is tight).
    ForwardRec rec{};
    if (std::fseek(fp_, static_cast<long>(static_cast<std::uint64_t>(pos) * sizeof(ForwardRec)), SEEK_SET) != 0)
      throw std::runtime_error("forward scratch seek failed");
    read_exact(fp_, &rec, 1);
    return rec;
  }

  std::vector<ForwardRec> read_block(std::uint32_t block_idx, std::uint32_t len) {
    auto it = cached_.find(block_idx);
    if (it != cached_.end()) return it->second;
    std::vector<ForwardRec> out(len);
    const std::uint64_t offset = static_cast<std::uint64_t>(block_idx) * block_verts_ * sizeof(ForwardRec);
    if (std::fseek(fp_, static_cast<long>(offset), SEEK_SET) != 0)
      throw std::runtime_error("forward scratch block seek failed");
    read_exact(fp_, out.data(), len);
    return out;
  }

 private:
  void cache_block(std::uint32_t block_idx, const std::vector<ForwardRec>& recs) {
    if (cached_.size() >= max_cached_ && !lru_.empty()) {
      cached_.erase(lru_.front());
      lru_.erase(lru_.begin());
    }
    cached_[block_idx] = recs;
    lru_.push_back(block_idx);
  }

  std::string path_;
  std::FILE* fp_ = nullptr;
  std::uint32_t block_verts_ = 1;
  std::size_t max_cached_ = 2;
  std::unordered_map<std::uint32_t, std::vector<ForwardRec>> cached_;
  std::vector<std::uint32_t> lru_;
};

template <typename Delta>
class DeltaSpillBuckets {
 public:
  DeltaSpillBuckets(const ScratchDir& scratch, const std::string& prefix, std::uint32_t num_blocks)
      : paths_(num_blocks), bufs_(num_blocks), counts_(num_blocks, 0) {
    for (std::uint32_t b = 0; b < num_blocks; ++b) paths_[b] = scratch.file(prefix + "_" + std::to_string(b) + ".bin");
  }

  void push(std::uint32_t block_idx, const Delta& d) {
    auto& buf = bufs_[block_idx];
    buf.push_back(d);
    ++counts_[block_idx];
    if (buf.size() >= 4096) flush_one(block_idx);
  }

  std::vector<Delta> take_all(std::uint32_t block_idx) {
    flush_one(block_idx);
    std::vector<Delta> out;
    if (counts_[block_idx] == 0) return out;
    out.resize(static_cast<std::size_t>(counts_[block_idx]));
    std::FILE* fp = std::fopen(paths_[block_idx].c_str(), "rb");
    if (!fp) throw std::runtime_error("cannot read reverse delta bucket " + paths_[block_idx]);
    read_exact(fp, out.data(), out.size());
    std::fclose(fp);
    std::remove(paths_[block_idx].c_str());
    counts_[block_idx] = 0;
    return out;
  }

 private:
  void flush_one(std::uint32_t b) {
    if (bufs_[b].empty()) return;
    std::FILE* fp = std::fopen(paths_[b].c_str(), "ab");
    if (!fp) throw std::runtime_error("cannot write reverse delta bucket " + paths_[b]);
    write_exact(fp, bufs_[b].data(), bufs_[b].size());
    std::fclose(fp);
    bufs_[b].clear();
  }

  std::vector<std::string> paths_;
  std::vector<std::vector<Delta>> bufs_;
  std::vector<std::uint64_t> counts_;
};

class GridGraphEngine final : public GraphEngine {
 public:
  std::string_view name() const override { return "gridgraph"; }

  DagStreamResult solve_dag(GraphSource& source, const DagSolveOptions& opts) override {
    ScratchDir scratch(opts.scratch_dir);
    const std::string cache_path = scratch.file("temp.cta_graph");
    cache::build_graph_cache(source, cache_path, opts);
    cache::GraphCacheReader reader(cache_path);
    return solve_from_graph_cache(reader, opts);
  }
};

}  // namespace

DagStreamResult solve_from_graph_cache(const cache::GraphCacheReader& reader, const DagSolveOptions& opts) {
  const auto& hdr = reader.header();
  DagStreamResult res;
  res.num_nodes = hdr.num_real_vertices;
  res.num_edges = hdr.num_edges;
  res.unordered = hdr.unordered;
  res.provenance.engine = "gridgraph";
  res.provenance.goal_semantics =
      hdr.goal_semantics == static_cast<std::uint8_t>(GoalSemantics::kStrict) ? "strict" : "cta_legacy";
  res.provenance.collective_sync =
      hdr.collective_sync == static_cast<std::uint8_t>(CollectiveSync::kBarrier) ? "barrier" : "none";
  res.provenance.unmatched_p2p_sends = hdr.unmatched_p2p_sends;
  res.provenance.unmatched_p2p_recvs = hdr.unmatched_p2p_recvs;
  res.provenance.has_cycle = hdr.unordered > 0;
  res.provenance.unordered_nodes = hdr.unordered;
  res.provenance.cycle_witness_rank = reader.cycle_witness_rank();
  res.provenance.cycle_witness_node_id = reader.cycle_witness_node_id();

  if (hdr.num_real_vertices == 0) {
    res.provenance.exact = true;
    res.provenance.slack_scope = "global_exact";
    return res;
  }

  const bool strict_ireq = (opts.goal_semantics == GoalSemantics::kStrict ||
                            hdr.goal_semantics == static_cast<std::uint8_t>(GoalSemantics::kStrict));
  const std::uint64_t budget = resolve_memory_budget(opts.memory_budget_bytes);
  ScratchDir scratch(opts.scratch_dir);

  const std::size_t block_rec_bytes = std::max<std::size_t>(1, hdr.block_vertices) * sizeof(ForwardRec);
  const std::size_t max_cached_blocks =
      std::max<std::size_t>(2, std::min<std::size_t>(hdr.num_blocks, (budget / 4) / block_rec_bytes));
  ForwardStateStore fwd_store(scratch.file("forward.bin"), hdr.block_vertices, max_cached_blocks);

  // Pass 1: Forward sweep over topological blocks 0 .. num_blocks - 1.
  // Slot-recycled boundary map keeps cross-block live values in RAM up to budget,
  // falling back seamlessly to `fwd_store.read_pos(u)` when cutwidth is adversarial!
  const std::size_t max_boundary_entries = std::max<std::size_t>(1024, (budget / 6) / (sizeof(ForwardRec) + 24));
  std::unordered_map<std::uint32_t, ForwardRec> cross_boundary;
  std::unordered_map<std::uint32_t, std::uint32_t> boundary_last_use;

  double makespan = -1.0;
  std::uint32_t sink_pos = kNoPos;
  std::uint32_t sink_orig_vid = kNoPos;
  std::int32_t max_depth = 0;

  cache::TopoBlock block;
  for (std::uint32_t b = 0; b < hdr.num_blocks; ++b) {
    reader.read_block(b, block);
    const std::size_t len = block.size();
    std::vector<ForwardRec> brecs(len);

    for (std::size_t i = 0; i < len; ++i) {
      const std::uint32_t p = block.pos_begin + static_cast<std::uint32_t>(i);
      const cache::TopoVertexMeta& vm = block.vertices[i];
      const bool is_virt = (vm.flags & cache::kFlagIsVirtual) != 0;
      const std::int32_t depth_inc = is_virt ? 0 : 1;

      double best_start = 0.0;
      std::int32_t best_pred_depth = 0;
      std::uint32_t best_parent_pos = kNoPos;
      std::uint32_t best_parent_orig = kNoPos;

      const std::uint64_t e_begin = block.pred_ptr[i];
      const std::uint64_t e_end = block.pred_ptr[i + 1];
      for (std::uint64_t e = e_begin; e < e_end; ++e) {
        const std::uint32_t u = block.preds[e];
        const std::uint8_t kind = block.pred_kind[e];

        ForwardRec urec{};
        if (u >= block.pos_begin) {
          urec = brecs[u - block.pos_begin];
        } else {
          auto bit = cross_boundary.find(u);
          if (bit != cross_boundary.end()) {
            urec = bit->second;
            auto lit = boundary_last_use.find(u);
            if (lit != boundary_last_use.end() && lit->second <= p) {
              cross_boundary.erase(bit);
              boundary_last_use.erase(lit);
            }
          } else {
            urec = fwd_store.read_pos(u);
          }
        }

        const bool use_start = (strict_ireq && kind == static_cast<std::uint8_t>(EdgeKind::kIrequires));
        const double cand_time = use_start ? urec.es : urec.ef;
        const bool later = cand_time > best_start;
        const bool tied = (cand_time == best_start) && (best_parent_pos != kNoPos || cand_time == 0.0);
        if (best_parent_pos == kNoPos || later ||
            (tied && (urec.depth > best_pred_depth ||
                      (urec.depth == best_pred_depth && urec.orig_vid < best_parent_orig)))) {
          best_start = cand_time;
          best_pred_depth = urec.depth;
          best_parent_pos = u;
          best_parent_orig = urec.orig_vid;
        }
      }

      ForwardRec cur;
      cur.es = best_start;
      cur.ef = best_start + vm.weight_ns;
      cur.depth = best_pred_depth + depth_inc;
      cur.parent_pos = best_parent_pos;
      cur.orig_vid = vm.orig_vid;
      brecs[i] = cur;

      // Retain in cross-block boundary map only if consumed by a future block AND within RAM cap.
      if (vm.last_use_pos >= block.pos_end && cross_boundary.size() < max_boundary_entries) {
        cross_boundary[p] = cur;
        boundary_last_use[p] = vm.last_use_pos;
      }

      if (!is_virt) {
        if (cur.depth > max_depth) max_depth = cur.depth;
        if (cur.ef > makespan || (cur.ef == makespan && vm.orig_vid < sink_orig_vid)) {
          makespan = cur.ef;
          sink_pos = p;
          sink_orig_vid = vm.orig_vid;
        }
      }
    }

    fwd_store.append_block(b, brecs);
  }
  cross_boundary.clear();
  boundary_last_use.clear();

  if (makespan < 0.0) makespan = 0.0;
  res.makespan = makespan;
  res.depth = max_depth;
  res.provenance.makespan_lb = makespan;
  res.provenance.makespan_ub = makespan;

  // Backtrack critical chain sink -> source (exact long double order matching dag.cpp!).
  if (sink_pos != kNoPos) {
    struct ChainEntry {
      std::uint32_t pos = 0;
      double es = 0.0;
      double ef = 0.0;
    };
    std::vector<ChainEntry> chain_pos;
    std::unordered_set<std::uint32_t> visited;
    for (std::uint32_t p = sink_pos; p != kNoPos && visited.insert(p).second;) {
      const ForwardRec fr = fwd_store.read_pos(p);
      chain_pos.push_back({p, fr.es, fr.ef});
      p = fr.parent_pos;
    }

    // Group chain positions by block to minimize disk reads while preserving
    // exact sink -> source summation order for cp_compute / cp_comm.
    std::unordered_map<std::uint32_t, std::vector<std::size_t>> by_block;
    for (std::size_t k = 0; k < chain_pos.size(); ++k) {
      by_block[chain_pos[k].pos / hdr.block_vertices].push_back(k);
    }
    struct ChainNodeMeta {
      std::uint32_t rank = 0;
      std::uint64_t node_id = 0;
      std::uint8_t op_class = 0;
      std::uint16_t flags = 0;
      double weight_ns = 0.0;
      std::uint64_t size_bytes = 0;
      std::string name;
    };
    std::vector<ChainNodeMeta> metas(chain_pos.size());
    for (const auto& [bidx, indices] : by_block) {
      reader.read_block(bidx, block);
      for (std::size_t k : indices) {
        const std::size_t li = chain_pos[k].pos - block.pos_begin;
        const auto& vm = block.vertices[li];
        metas[k] = {vm.rank,
                    vm.node_id,
                    vm.op_class,
                    vm.flags,
                    vm.weight_ns,
                    vm.size_bytes,
                    opts.extract_critical_path_ops ? std::string(block.name_of(li, reader.names())) : std::string()};
      }
    }

    long double cp_comp = 0.0L, cp_comm = 0.0L;
    std::int64_t cp_bytes = 0;
    for (std::size_t k = 0; k < metas.size(); ++k) {
      const auto& m = metas[k];
      if ((m.flags & cache::kFlagIsVirtual) != 0) continue;
      if ((m.flags & cache::kFlagIsCompute) != 0) cp_comp += m.weight_ns;
      if ((m.flags & cache::kFlagIsComm) != 0) {
        cp_comm += m.weight_ns;
        cp_bytes += static_cast<std::int64_t>(m.size_bytes);
      }
    }
    res.critical_compute = static_cast<double>(cp_comp);
    res.critical_comm = static_cast<double>(cp_comm);
    res.critical_comm_bytes = cp_bytes;

    if (opts.extract_critical_path_ops) {
      for (std::size_t k = metas.size(); k-- > 0;) {
        const auto& m = metas[k];
        if ((m.flags & cache::kFlagIsVirtual) != 0) continue;
        res.cp_op_rank.push_back(m.rank);
        res.cp_op_node_id.push_back(m.node_id);
        res.cp_op_class.push_back(m.op_class);
        res.cp_op_weight_ns.push_back(m.weight_ns);
        res.cp_op_size_bytes.push_back(m.size_bytes);
        res.cp_op_earliest_start_ns.push_back(chain_pos[k].es);
        res.cp_op_earliest_finish_ns.push_back(chain_pos[k].ef);
        res.cp_op_name.push_back(m.name);
      }
    }
  }

  // Pass 2: Reverse block sweep for exact slack (Neumaier compensated sum)
  // + optional Scalasca backward wait-state & delay root-cause attribution.
  DeltaSpillBuckets<RevSlackDelta> slack_deltas(scratch, "rslack", hdr.num_blocks);
  std::unique_ptr<DeltaSpillBuckets<RevDelayDelta>> delay_deltas;
  attribution::ScalascaDelayAccumulator delay_acc;
  if (opts.delay_attribution) {
    delay_deltas = std::make_unique<DeltaSpillBuckets<RevDelayDelta>>(scratch, "rdelay", hdr.num_blocks);
  }

  // Note: for strict irequires (LF[u] <= LF[v] - w[v] + w[u]), a cross-block
  // edge pushes LF[v] - w[v] (latest_start of v) as an upper bound on LS[u],
  // and when block(u) pops the delta we add +w[u] to convert LS[u] -> LF[u]!
  // Let's keep a separate LS-delta spill bucket when strict_ireq is active so
  // we NEVER need random reads of w[u] across blocks!
  std::unique_ptr<DeltaSpillBuckets<RevSlackDelta>> strict_ls_deltas;
  if (strict_ireq) {
    strict_ls_deltas = std::make_unique<DeltaSpillBuckets<RevSlackDelta>>(scratch, "rslack_ls", hdr.num_blocks);
  }

  const double tol = std::max(1e-6, makespan * 1e-9);
  NeumaierSum slack_sum;
  std::uint64_t zero_slack = 0;

  for (std::uint32_t b = hdr.num_blocks; b-- > 0;) {
    reader.read_block(b, block);
    const std::size_t len = block.size();
    const std::vector<ForwardRec> brecs = fwd_store.read_block(b, static_cast<std::uint32_t>(len));
    std::vector<double> lf(len, makespan);

    // Apply incoming cross-block LF bounds from blocks > b.
    for (const auto& d : slack_deltas.take_all(b)) {
      const std::size_t off = d.target_pos - block.pos_begin;
      if (d.bound_lf < lf[off]) lf[off] = d.bound_lf;
    }
    if (strict_ls_deltas) {
      for (const auto& d : strict_ls_deltas->take_all(b)) {
        const std::size_t off = d.target_pos - block.pos_begin;
        const double bound_lf = d.bound_lf + block.vertices[off].weight_ns;
        if (bound_lf < lf[off]) lf[off] = bound_lf;
      }
    }

    std::vector<double> prop_delay;
    if (opts.delay_attribution) {
      prop_delay.assign(len, 0.0);
      for (const auto& d : delay_deltas->take_all(b)) {
        prop_delay[d.target_pos - block.pos_begin] += d.delay_ns;
      }
    }

    for (std::size_t i = len; i-- > 0;) {
      const cache::TopoVertexMeta& vm = block.vertices[i];
      const ForwardRec& fr = brecs[i];
      const bool is_virt = (vm.flags & cache::kFlagIsVirtual) != 0;

      if (!is_virt) {
        const double slack = std::max(0.0, lf[i] - fr.ef);
        slack_sum.add(slack);
        if (slack <= tol) ++zero_slack;
      }

      const double my_ls = lf[i] - vm.weight_ns;
      const std::uint64_t e_begin = block.pred_ptr[i];
      const std::uint64_t e_end = block.pred_ptr[i + 1];

      // Propagate backward slack bounds to all predecessors u < p.
      for (std::uint64_t e = e_begin; e < e_end; ++e) {
        const std::uint32_t u = block.preds[e];
        const std::uint8_t kind = block.pred_kind[e];
        const bool is_strict_ireq_edge =
            (strict_ireq && kind == static_cast<std::uint8_t>(EdgeKind::kIrequires));
        if (u >= block.pos_begin) {
          const std::size_t ui = u - block.pos_begin;
          const double bound = is_strict_ireq_edge ? (my_ls + block.vertices[ui].weight_ns) : my_ls;
          if (bound < lf[ui]) lf[ui] = bound;
        } else {
          const std::uint32_t ub = u / hdr.block_vertices;
          if (is_strict_ireq_edge) {
            strict_ls_deltas->push(ub, RevSlackDelta{u, 0, my_ls});
          } else {
            slack_deltas.push(ub, RevSlackDelta{u, 0, my_ls});
          }
        }
      }

      // Optional Scalasca backward wait-state & delay propagation (Böhme et al., IPDPS '12).
      if (opts.delay_attribution) {
        double incoming_delay = prop_delay[i];
        bool has_sync_pred = false;
        double max_local_ef = 0.0;
        bool has_local_pred = false;
        double barrier_wait_sum = 0.0;
        for (std::uint64_t e = e_begin; e < e_end; ++e) {
          const std::uint8_t kind = block.pred_kind[e];
          const std::uint32_t u = block.preds[e];
          const ForwardRec ufr = (u >= block.pos_begin) ? brecs[u - block.pos_begin] : fwd_store.read_pos(u);
          if (is_virt) {
            // Virtual collective barrier B_in / B_out: sum arrival skew of all
            // non-critical early-arriving ranks and credit the straggler (parent_pos).
            if (u != fr.parent_pos && fr.es > ufr.ef) {
              barrier_wait_sum += (fr.es - ufr.ef);
            }
          } else if (kind == static_cast<std::uint8_t>(EdgeKind::kP2P) ||
                     kind == static_cast<std::uint8_t>(EdgeKind::kCollBarrier)) {
            has_sync_pred = true;
          } else {
            if (!has_local_pred || ufr.ef > max_local_ef) {
              max_local_ef = ufr.ef;
              has_local_pred = true;
            }
          }
        }
        if (is_virt && barrier_wait_sum > 0.0) {
          incoming_delay += barrier_wait_sum;
        } else if (has_sync_pred && has_local_pred && fr.es > max_local_ef) {
          const double direct_wait = fr.es - max_local_ef;
          delay_acc.record_direct_wait(vm.rank, block.name_of(i, reader.names()), direct_wait);
          incoming_delay += direct_wait;
        }
        if (incoming_delay > 0.0) {
          if (fr.parent_pos != kNoPos) {
            // Propagate backward along the critical predecessor that delayed this node.
            if (fr.parent_pos >= block.pos_begin) {
              prop_delay[fr.parent_pos - block.pos_begin] += incoming_delay;
            } else {
              delay_deltas->push(fr.parent_pos / hdr.block_vertices, RevDelayDelta{fr.parent_pos, 0, incoming_delay});
            }
          }
          // Credit non-waiting compute/comm ops as root-cause delay sources.
          if (!is_virt && !has_sync_pred) {
            delay_acc.record_root_cause(vm.rank, block.name_of(i, reader.names()), incoming_delay);
          }
        }
      }
    }
  }

  res.mean_slack = static_cast<double>(slack_sum.value() / static_cast<long double>(hdr.num_real_vertices));
  res.zero_slack_ratio = static_cast<double>(zero_slack) / static_cast<double>(hdr.num_real_vertices);
  res.provenance.slack_scope = "global_exact";
  res.provenance.exact = (hdr.unordered == 0);
  if (opts.delay_attribution) {
    delay_acc.finalize(res, hdr.has_recorded_comm_times != 0);
  }
  return res;
}

std::unique_ptr<GraphEngine> make_gridgraph_engine() { return std::make_unique<GridGraphEngine>(); }

}  // namespace cta::streaming::engines
