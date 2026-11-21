// GraphChi Parallel Sliding Windows (PSW) DAG Engine
// (Aapo Kyrola, Guy Blelloch, Carlos Guestrin — USENIX OSDI 2012).
//
// Partitions topologically ordered vertices into P intervals [start_p, end_p),
// builds P source-sorted incoming-edge shards on disk, and executes Parallel
// Sliding Windows sequentially across shards for both forward longest-path DP
// and reverse-interval backward slack computation within bounded RAM.
#include "streaming/engines/graphchi/graphchi_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "streaming/cache/graph_cache.hpp"
#include "streaming/scratch.hpp"
#include "util/compensated_sum.hpp"
#include "util/memory.hpp"

namespace cta::streaming::engines {
namespace {

constexpr std::uint32_t kNoPos = std::numeric_limits<std::uint32_t>::max();

struct ShardEdge {
  std::uint32_t src = 0;
  std::uint32_t dst = 0;
  std::uint8_t kind = 0;
  std::uint8_t pad[3]{};
};

struct VertexState {
  double es = 0.0;
  double ef = 0.0;
  double lf = 0.0;
  double weight = 0.0;
  std::int32_t depth = 1;
  std::uint32_t parent_pos = kNoPos;
  std::uint32_t orig_vid = 0;
  std::uint32_t rank = 0;
  std::uint64_t node_id = 0;
  std::uint64_t size_bytes = 0;
  std::uint16_t flags = 0;
  std::uint8_t op_class = 0;
  std::uint8_t pad = 0;
};

template <typename T>
void write_all(std::FILE* fp, const T* data, std::size_t n) {
  if (n > 0 && std::fwrite(data, sizeof(T), n, fp) != n) throw std::runtime_error("graphchi shard write failed");
}

template <typename T>
void read_all(std::FILE* fp, T* data, std::size_t n) {
  if (n > 0 && std::fread(data, sizeof(T), n, fp) != n) throw std::runtime_error("graphchi shard read failed");
}

class GraphChiEngine final : public GraphEngine {
 public:
  std::string_view name() const override { return "graphchi"; }

  DagStreamResult solve_dag(GraphSource& source, const DagSolveOptions& opts) override {
    reject_delay_attribution(opts, "graphchi");
    ScratchDir scratch(opts.scratch_dir);
    const std::string cache_path = scratch.file("graphchi.cta_graph");
    cache::build_graph_cache(source, cache_path, opts);
    cache::GraphCacheReader reader(cache_path);

    const auto& hdr = reader.header();
    DagStreamResult res;
    res.num_nodes = hdr.num_real_vertices;
    res.num_edges = hdr.num_edges;
    res.unordered = hdr.unordered;
    res.provenance.engine = "graphchi";
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
    const std::uint32_t P = hdr.num_blocks;
    const std::uint32_t interval_len = hdr.block_vertices;

    // Build P source-sorted GraphChi incoming-edge shards on scratch disk +
    // P vertex-interval state files.
    std::vector<std::string> shard_paths(P), vstate_paths(P);
    std::vector<std::uint64_t> shard_counts(P, 0);
    std::vector<std::uint32_t> interval_sizes(P, 0);
    cache::TopoBlock block;

    for (std::uint32_t p = 0; p < P; ++p) {
      reader.read_block(p, block);
      interval_sizes[p] = static_cast<std::uint32_t>(block.size());

      std::vector<VertexState> vs(block.size());
      std::vector<ShardEdge> shard;
      shard.reserve(block.preds.size());
      for (std::size_t i = 0; i < block.size(); ++i) {
        const auto& vm = block.vertices[i];
        vs[i].weight = vm.weight_ns;
        vs[i].ef = vm.weight_ns;
        vs[i].depth = (vm.flags & cache::kFlagIsVirtual) ? 0 : 1;
        vs[i].orig_vid = vm.orig_vid;
        vs[i].rank = vm.rank;
        vs[i].node_id = vm.node_id;
        vs[i].size_bytes = vm.size_bytes;
        vs[i].flags = vm.flags;
        vs[i].op_class = vm.op_class;
        const std::uint32_t dst_pos = block.pos_begin + static_cast<std::uint32_t>(i);
        for (std::uint64_t e = block.pred_ptr[i]; e < block.pred_ptr[i + 1]; ++e) {
          shard.push_back({block.preds[e], dst_pos, block.pred_kind[e], {}});
        }
      }
      // GraphChi invariant: each shard is sorted by source vertex id (`src`)
      // so sliding windows across shards read strictly sequentially!
      std::sort(shard.begin(), shard.end(), [](const ShardEdge& a, const ShardEdge& b) {
        if (a.src != b.src) return a.src < b.src;
        return a.dst < b.dst;
      });
      shard_paths[p] = scratch.file("gc_shard_" + std::to_string(p) + ".bin");
      vstate_paths[p] = scratch.file("gc_vstate_" + std::to_string(p) + ".bin");
      shard_counts[p] = shard.size();
      std::FILE* sfp = std::fopen(shard_paths[p].c_str(), "wb");
      write_all(sfp, shard.data(), shard.size());
      std::fclose(sfp);
      std::FILE* vfp = std::fopen(vstate_paths[p].c_str(), "wb");
      write_all(vfp, vs.data(), vs.size());
      std::fclose(vfp);
    }

    // Forward PSW pass: for each interval p = 0 .. P-1, load interval p + its
    // incoming shard_p, plus sliding windows from already-solved intervals q < p.
    // Because shards are source-sorted, we stream interval q's values into a
    // compact edge-value staging file (classic GraphChi sliding-window push).
    struct EdgeUpdate {
      std::uint32_t dst = 0;
      std::uint32_t src_pos = 0;
      std::uint32_t src_orig = 0;
      std::int32_t src_depth = 0;
      double cand_time = 0.0;
    };
    std::vector<std::string> fwd_msg_paths(P);
    std::vector<std::uint64_t> fwd_msg_counts(P, 0);
    for (std::uint32_t p = 0; p < P; ++p) fwd_msg_paths[p] = scratch.file("gc_fmsg_" + std::to_string(p) + ".bin");

    // Track current sliding-window read cursor in each shard `q` (monotonically
    // advances left-to-right as source interval `p` sweeps 0 .. P-1!).
    std::vector<std::uint64_t> shard_cursor(P, 0);

    double makespan = -1.0;
    std::uint32_t sink_pos = kNoPos;
    std::uint32_t sink_orig = kNoPos;
    std::int32_t max_depth = 0;

    for (std::uint32_t p = 0; p < P; ++p) {
      const std::uint32_t p_begin = p * interval_len;
      const std::uint32_t p_end = p_begin + interval_sizes[p];
      std::vector<VertexState> vs(interval_sizes[p]);
      {
        std::FILE* vfp = std::fopen(vstate_paths[p].c_str(), "rb");
        read_all(vfp, vs.data(), vs.size());
        std::fclose(vfp);
      }

      // 1. Apply incoming cross-interval sliding-window messages pushed by
      //    intervals q < p with exact orig_vid tie-breaking.
      std::vector<std::uint32_t> parent_orig(interval_sizes[p], kNoPos);
      auto relax_into = [&](std::uint32_t dst_pos, std::uint32_t src_pos, std::uint32_t src_orig,
                            std::int32_t src_depth, double cand_time) {
        const std::size_t di = dst_pos - p_begin;
        VertexState& v = vs[di];
        const std::int32_t depth_inc = (v.flags & cache::kFlagIsVirtual) ? 0 : 1;
        const bool later = cand_time > v.es;
        const bool tied = (cand_time == v.es);
        if (v.parent_pos == kNoPos || later ||
            (tied && (src_depth + depth_inc > v.depth ||
                      (src_depth + depth_inc == v.depth && src_orig < parent_orig[di])))) {
          v.es = cand_time;
          v.ef = cand_time + v.weight;
          v.depth = src_depth + depth_inc;
          v.parent_pos = src_pos;
          parent_orig[di] = src_orig;
        }
      };

      if (fwd_msg_counts[p] > 0) {
        std::vector<EdgeUpdate> msgs(static_cast<std::size_t>(fwd_msg_counts[p]));
        std::FILE* mfp = std::fopen(fwd_msg_paths[p].c_str(), "rb");
        read_all(mfp, msgs.data(), msgs.size());
        std::fclose(mfp);
        for (const auto& m : msgs) relax_into(m.dst, m.src_pos, m.src_orig, m.src_depth, m.cand_time);
      }

      // 2. Load intra-interval edges from shard_p (sliding window [p_begin, p_end))
      //    and evaluate in ascending topological order inside interval p!
      if (shard_counts[p] > 0) {
        std::vector<ShardEdge> shard(static_cast<std::size_t>(shard_counts[p]));
        std::FILE* sfp = std::fopen(shard_paths[p].c_str(), "rb");
        read_all(sfp, shard.data(), shard.size());
        std::fclose(sfp);
        // Intra-interval edges satisfy p_begin <= src < dst < p_end; sort by dst for exact sequential pass.
        std::vector<ShardEdge> intra;
        for (const auto& e : shard)
          if (e.src >= p_begin) intra.push_back(e);
        std::sort(intra.begin(), intra.end(), [](const ShardEdge& a, const ShardEdge& b) {
          if (a.dst != b.dst) return a.dst < b.dst;
          return a.src < b.src;
        });
        for (const auto& e : intra) {
          const VertexState& u = vs[e.src - p_begin];
          const bool use_start = (strict_ireq && e.kind == static_cast<std::uint8_t>(EdgeKind::kIrequires));
          relax_into(e.dst, e.src, u.orig_vid, u.depth, use_start ? u.es : u.ef);
        }
      }

      // Update global makespan/sink & save finalized interval p state to disk.
      for (std::size_t i = 0; i < vs.size(); ++i) {
        if ((vs[i].flags & cache::kFlagIsVirtual) == 0) {
          if (vs[i].depth > max_depth) max_depth = vs[i].depth;
          if (vs[i].ef > makespan || (vs[i].ef == makespan && vs[i].orig_vid < sink_orig)) {
            makespan = vs[i].ef;
            sink_pos = p_begin + static_cast<std::uint32_t>(i);
            sink_orig = vs[i].orig_vid;
          }
        }
      }
      {
        std::FILE* vfp = std::fopen(vstate_paths[p].c_str(), "wb");
        write_all(vfp, vs.data(), vs.size());
        std::fclose(vfp);
      }

      // 3. Sliding-window push: advance the sequential read cursor of every
      //    downstream shard q > p over its source-sorted slice [p_begin, p_end)!
      for (std::uint32_t q = p + 1; q < P; ++q) {
        if (shard_cursor[q] >= shard_counts[q]) continue;
        std::FILE* sfp = std::fopen(shard_paths[q].c_str(), "rb");
        std::fseek(sfp, static_cast<long>(shard_cursor[q] * sizeof(ShardEdge)), SEEK_SET);
        std::vector<EdgeUpdate> out_msgs;
        ShardEdge chunk_buf[4096];
        bool done_window = false;
        while (!done_window && shard_cursor[q] < shard_counts[q]) {
          const std::size_t want = static_cast<std::size_t>(std::min<std::uint64_t>(4096, shard_counts[q] - shard_cursor[q]));
          read_all(sfp, chunk_buf, want);
          for (std::size_t k = 0; k < want; ++k) {
            if (chunk_buf[k].src >= p_end) {
              done_window = true;
              break;
            }
            ++shard_cursor[q];
            if (chunk_buf[k].src < p_begin) continue;
            const VertexState& u = vs[chunk_buf[k].src - p_begin];
            const bool use_start =
                (strict_ireq && chunk_buf[k].kind == static_cast<std::uint8_t>(EdgeKind::kIrequires));
            out_msgs.push_back({chunk_buf[k].dst, chunk_buf[k].src, u.orig_vid, u.depth, use_start ? u.es : u.ef});
          }
        }
        std::fclose(sfp);
        if (!out_msgs.empty()) {
          std::FILE* mfp = std::fopen(fwd_msg_paths[q].c_str(), "ab");
          write_all(mfp, out_msgs.data(), out_msgs.size());
          std::fclose(mfp);
          fwd_msg_counts[q] += out_msgs.size();
        }
      }
    }

    if (makespan < 0.0) makespan = 0.0;
    res.makespan = makespan;
    res.depth = max_depth;
    res.provenance.makespan_lb = makespan;
    res.provenance.makespan_ub = makespan;

    // Backtrack critical chain sink -> source across vertex-state interval files.
    if (sink_pos != kNoPos) {
      std::vector<VertexState> chain;
      std::unordered_set<std::uint32_t> seen;
      long double cp_comp = 0.0L, cp_comm = 0.0L;
      std::int64_t cp_bytes = 0;
      for (std::uint32_t pos = sink_pos; pos != kNoPos && seen.insert(pos).second;) {
        const std::uint32_t iv = pos / interval_len;
        const std::uint32_t off = pos - iv * interval_len;
        VertexState st{};
        std::FILE* vfp = std::fopen(vstate_paths[iv].c_str(), "rb");
        std::fseek(vfp, static_cast<long>(static_cast<std::uint64_t>(off) * sizeof(VertexState)), SEEK_SET);
        read_all(vfp, &st, 1);
        std::fclose(vfp);
        if ((st.flags & cache::kFlagIsVirtual) == 0) {
          if (st.flags & cache::kFlagIsCompute) cp_comp += st.weight;
          if (st.flags & cache::kFlagIsComm) {
            cp_comm += st.weight;
            cp_bytes += static_cast<std::int64_t>(st.size_bytes);
          }
          chain.push_back(st);
        }
        pos = st.parent_pos;
      }
      res.critical_compute = static_cast<double>(cp_comp);
      res.critical_comm = static_cast<double>(cp_comm);
      res.critical_comm_bytes = cp_bytes;
      if (opts.extract_critical_path_ops) {
        std::reverse(chain.begin(), chain.end());
        for (const auto& st : chain) {
          res.cp_op_rank.push_back(st.rank);
          res.cp_op_node_id.push_back(st.node_id);
          res.cp_op_class.push_back(st.op_class);
          res.cp_op_weight_ns.push_back(st.weight);
          res.cp_op_size_bytes.push_back(st.size_bytes);
          res.cp_op_earliest_start_ns.push_back(st.es);
          res.cp_op_earliest_finish_ns.push_back(st.ef);
          res.cp_op_name.emplace_back("");
        }
      }
    }

    // Reverse PSW pass for exact slack: sweep intervals p = P-1 down to 0,
    // streaming shard_p backward to push LF bounds to intervals q <= p.
    struct RevBound {
      std::uint32_t target = 0;
      std::uint8_t is_strict_ireq = 0;
      std::uint8_t pad[3]{};
      double ls_val = 0.0;
    };
    std::vector<std::string> rev_paths(P);
    std::vector<std::uint64_t> rev_counts(P, 0);
    for (std::uint32_t p = 0; p < P; ++p) rev_paths[p] = scratch.file("gc_rmsg_" + std::to_string(p) + ".bin");

    const double tol = std::max(1e-6, makespan * 1e-9);
    NeumaierSum slack_sum;
    std::uint64_t zero_slack = 0;

    for (std::uint32_t p = P; p-- > 0;) {
      const std::uint32_t p_begin = p * interval_len;
      std::vector<VertexState> vs(interval_sizes[p]);
      {
        std::FILE* vfp = std::fopen(vstate_paths[p].c_str(), "rb");
        read_all(vfp, vs.data(), vs.size());
        std::fclose(vfp);
      }
      for (auto& st : vs) st.lf = makespan;
      if (rev_counts[p] > 0) {
        std::vector<RevBound> rbs(static_cast<std::size_t>(rev_counts[p]));
        std::FILE* rfp = std::fopen(rev_paths[p].c_str(), "rb");
        read_all(rfp, rbs.data(), rbs.size());
        std::fclose(rfp);
        for (const auto& rb : rbs) {
          VertexState& u = vs[rb.target - p_begin];
          const double cand = rb.is_strict_ireq ? (rb.ls_val + u.weight) : rb.ls_val;
          if (cand < u.lf) u.lf = cand;
        }
      }

      // Load shard_p and process in descending `dst` order for intra-interval edges.
      std::vector<std::vector<RevBound>> push_out(p);
      if (shard_counts[p] > 0) {
        std::vector<ShardEdge> shard(static_cast<std::size_t>(shard_counts[p]));
        std::FILE* sfp = std::fopen(shard_paths[p].c_str(), "rb");
        read_all(sfp, shard.data(), shard.size());
        std::fclose(sfp);
        std::sort(shard.begin(), shard.end(), [](const ShardEdge& a, const ShardEdge& b) { return a.dst > b.dst; });
        for (const auto& e : shard) {
          const VertexState& v = vs[e.dst - p_begin];
          const double v_ls = v.lf - v.weight;
          const bool s_ireq = (strict_ireq && e.kind == static_cast<std::uint8_t>(EdgeKind::kIrequires));
          if (e.src >= p_begin) {
            VertexState& u = vs[e.src - p_begin];
            const double cand = s_ireq ? (v_ls + u.weight) : v_ls;
            if (cand < u.lf) u.lf = cand;
          } else {
            push_out[e.src / interval_len].push_back({e.src, static_cast<std::uint8_t>(s_ireq ? 1 : 0), {}, v_ls});
          }
        }
      }
      for (std::uint32_t q = 0; q < p; ++q) {
        if (push_out[q].empty()) continue;
        std::FILE* rfp = std::fopen(rev_paths[q].c_str(), "ab");
        write_all(rfp, push_out[q].data(), push_out[q].size());
        std::fclose(rfp);
        rev_counts[q] += push_out[q].size();
      }
      for (const auto& st : vs) {
        if (st.flags & cache::kFlagIsVirtual) continue;
        const double slack = std::max(0.0, st.lf - st.ef);
        slack_sum.add(slack);
        if (slack <= tol) ++zero_slack;
      }
    }

    res.mean_slack = static_cast<double>(slack_sum.value() / static_cast<long double>(hdr.num_real_vertices));
    res.zero_slack_ratio = static_cast<double>(zero_slack) / static_cast<double>(hdr.num_real_vertices);
    res.provenance.slack_scope = "global_exact";
    res.provenance.exact = (hdr.unordered == 0);
    return res;
  }
};

}  // namespace

std::unique_ptr<GraphEngine> make_graphchi_engine() { return std::make_unique<GraphChiEngine>(); }

}  // namespace cta::streaming::engines
