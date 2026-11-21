// X-Stream Edge-Centric Scatter-Gather DAG Engine
// (Amitabha Roy, Ivo Mihailovic, Willy Zwaenepoel — ACM SOSP 2013).
//
// Streams unsorted edge partitions sequentially in scatter-gather passes over
// streaming vertex partitions, using topological partition order so deep
// execution DAGs converge in 1 forward scatter-gather sweep + 1 reverse sweep!
#include "streaming/engines/xstream/xstream_engine.hpp"

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

namespace cta::streaming::engines {
namespace {

constexpr std::uint32_t kNoPos = std::numeric_limits<std::uint32_t>::max();

struct XEdge {
  std::uint32_t src = 0;
  std::uint32_t dst = 0;
  std::uint8_t kind = 0;
  std::uint8_t pad[3]{};
};

struct XUpdate {
  std::uint32_t dst = 0;
  std::uint32_t src_pos = 0;
  std::uint32_t src_orig = 0;
  std::int32_t src_depth = 0;
  double val = 0.0;
};

struct XVertex {
  double es = 0.0;
  double ef = 0.0;
  double lf = 0.0;
  double weight = 0.0;
  std::int32_t depth = 1;
  std::uint32_t parent_pos = kNoPos;
  std::uint32_t parent_orig = kNoPos;
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
  if (n > 0 && std::fwrite(data, sizeof(T), n, fp) != n) throw std::runtime_error("xstream write failed");
}

template <typename T>
void read_all(std::FILE* fp, T* data, std::size_t n) {
  if (n > 0 && std::fread(data, sizeof(T), n, fp) != n) throw std::runtime_error("xstream read failed");
}

class XStreamEngine final : public GraphEngine {
 public:
  std::string_view name() const override { return "xstream"; }

  DagStreamResult solve_dag(GraphSource& source, const DagSolveOptions& opts) override {
    ScratchDir scratch(opts.scratch_dir);
    const std::string cache_path = scratch.file("xstream.cta_graph");
    cache::build_graph_cache(source, cache_path, opts);
    cache::GraphCacheReader reader(cache_path);

    const auto& hdr = reader.header();
    DagStreamResult res;
    res.num_nodes = hdr.num_real_vertices;
    res.num_edges = hdr.num_edges;
    res.unordered = hdr.unordered;
    res.provenance.engine = "xstream";
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
    const std::uint32_t part_len = hdr.block_vertices;

    // Build P streaming partitions: unsorted outgoing edge lists per source
    // partition (X-Stream edge-centric layout) + vertex state files.
    std::vector<std::string> v_paths(P), e_src_paths(P), e_dst_paths(P);
    std::vector<std::uint64_t> e_src_counts(P, 0), e_dst_counts(P, 0);
    std::vector<std::uint32_t> part_sizes(P, 0);
    for (std::uint32_t p = 0; p < P; ++p) {
      v_paths[p] = scratch.file("xs_v_" + std::to_string(p) + ".bin");
      e_src_paths[p] = scratch.file("xs_esrc_" + std::to_string(p) + ".bin");
      e_dst_paths[p] = scratch.file("xs_edst_" + std::to_string(p) + ".bin");
    }

    cache::TopoBlock block;
    for (std::uint32_t p = 0; p < P; ++p) {
      reader.read_block(p, block);
      part_sizes[p] = static_cast<std::uint32_t>(block.size());
      std::vector<XVertex> vs(block.size());
      std::vector<std::vector<XEdge>> by_src_part(p + 1);
      std::vector<XEdge> dst_edges;
      dst_edges.reserve(block.preds.size());

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
          const XEdge xe{block.preds[e], dst_pos, block.pred_kind[e], {}};
          by_src_part[xe.src / part_len].push_back(xe);
          dst_edges.push_back(xe);
        }
      }
      std::FILE* vfp = std::fopen(v_paths[p].c_str(), "wb");
      write_all(vfp, vs.data(), vs.size());
      std::fclose(vfp);

      std::FILE* dfp = std::fopen(e_dst_paths[p].c_str(), "wb");
      write_all(dfp, dst_edges.data(), dst_edges.size());
      std::fclose(dfp);
      e_dst_counts[p] = dst_edges.size();

      for (std::uint32_t sp = 0; sp <= p; ++sp) {
        if (by_src_part[sp].empty()) continue;
        std::FILE* sfp = std::fopen(e_src_paths[sp].c_str(), "ab");
        write_all(sfp, by_src_part[sp].data(), by_src_part[sp].size());
        std::fclose(sfp);
        e_src_counts[sp] += by_src_part[sp].size();
      }
    }

    // Forward edge-centric Scatter-Gather in topological partition order 0 .. P-1.
    std::vector<std::string> upd_paths(P);
    std::vector<std::uint64_t> upd_counts(P, 0);
    for (std::uint32_t p = 0; p < P; ++p) upd_paths[p] = scratch.file("xs_upd_" + std::to_string(p) + ".bin");

    double makespan = -1.0;
    std::uint32_t sink_pos = kNoPos;
    std::uint32_t sink_orig = kNoPos;
    std::int32_t max_depth = 0;

    for (std::uint32_t p = 0; p < P; ++p) {
      const std::uint32_t p_begin = p * part_len;
      std::vector<XVertex> vs(part_sizes[p]);
      {
        std::FILE* vfp = std::fopen(v_paths[p].c_str(), "rb");
        read_all(vfp, vs.data(), vs.size());
        std::fclose(vfp);
      }

      auto apply_update = [&](const XUpdate& u) {
        XVertex& v = vs[u.dst - p_begin];
        const std::int32_t depth_inc = (v.flags & cache::kFlagIsVirtual) ? 0 : 1;
        const bool later = u.val > v.es;
        const bool tied = (u.val == v.es);
        if (v.parent_pos == kNoPos || later ||
            (tied && (u.src_depth + depth_inc > v.depth ||
                      (u.src_depth + depth_inc == v.depth && u.src_orig < v.parent_orig)))) {
          v.es = u.val;
          v.ef = u.val + v.weight;
          v.depth = u.src_depth + depth_inc;
          v.parent_pos = u.src_pos;
          v.parent_orig = u.src_orig;
        }
      };

      // Gather phase: stream incoming cross-partition updates into partition p.
      if (upd_counts[p] > 0) {
        std::vector<XUpdate> upds(static_cast<std::size_t>(upd_counts[p]));
        std::FILE* ufp = std::fopen(upd_paths[p].c_str(), "rb");
        read_all(ufp, upds.data(), upds.size());
        std::fclose(ufp);
        for (const auto& u : upds) apply_update(u);
      }

      // Intra-partition edge pass (edges in e_dst_paths[p] are already in dst order).
      if (e_dst_counts[p] > 0) {
        std::vector<XEdge> edges(static_cast<std::size_t>(e_dst_counts[p]));
        std::FILE* dfp = std::fopen(e_dst_paths[p].c_str(), "rb");
        read_all(dfp, edges.data(), edges.size());
        std::fclose(dfp);
        for (const auto& e : edges) {
          if (e.src < p_begin) continue;
          const XVertex& src = vs[e.src - p_begin];
          const bool use_start = (strict_ireq && e.kind == static_cast<std::uint8_t>(EdgeKind::kIrequires));
          apply_update({e.dst, e.src, src.orig_vid, src.depth, use_start ? src.es : src.ef});
        }
      }

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
        std::FILE* vfp = std::fopen(v_paths[p].c_str(), "wb");
        write_all(vfp, vs.data(), vs.size());
        std::fclose(vfp);
      }

      // Scatter phase: stream unsorted outgoing edges from partition p to downstream partitions q > p!
      if (e_src_counts[p] > 0) {
        std::vector<XEdge> edges(static_cast<std::size_t>(e_src_counts[p]));
        std::FILE* sfp = std::fopen(e_src_paths[p].c_str(), "rb");
        read_all(sfp, edges.data(), edges.size());
        std::fclose(sfp);
        std::vector<std::vector<XUpdate>> scatter(P);
        for (const auto& e : edges) {
          const std::uint32_t q = e.dst / part_len;
          if (q <= p) continue;
          const XVertex& src = vs[e.src - p_begin];
          const bool use_start = (strict_ireq && e.kind == static_cast<std::uint8_t>(EdgeKind::kIrequires));
          scatter[q].push_back({e.dst, e.src, src.orig_vid, src.depth, use_start ? src.es : src.ef});
        }
        for (std::uint32_t q = p + 1; q < P; ++q) {
          if (scatter[q].empty()) continue;
          std::FILE* ufp = std::fopen(upd_paths[q].c_str(), "ab");
          write_all(ufp, scatter[q].data(), scatter[q].size());
          std::fclose(ufp);
          upd_counts[q] += scatter[q].size();
        }
      }
    }

    if (makespan < 0.0) makespan = 0.0;
    res.makespan = makespan;
    res.depth = max_depth;
    res.provenance.makespan_lb = makespan;
    res.provenance.makespan_ub = makespan;

    // Critical chain backtrack sink -> source.
    if (sink_pos != kNoPos) {
      std::vector<XVertex> chain;
      std::unordered_set<std::uint32_t> seen;
      long double cp_comp = 0.0L, cp_comm = 0.0L;
      std::int64_t cp_bytes = 0;
      for (std::uint32_t pos = sink_pos; pos != kNoPos && seen.insert(pos).second;) {
        const std::uint32_t part = pos / part_len;
        const std::uint32_t off = pos - part * part_len;
        XVertex st{};
        std::FILE* vfp = std::fopen(v_paths[part].c_str(), "rb");
        std::fseek(vfp, static_cast<long>(static_cast<std::uint64_t>(off) * sizeof(XVertex)), SEEK_SET);
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
    }

    // Reverse scatter-gather pass for exact slack.
    struct XRevUpdate {
      std::uint32_t dst_src = 0;
      std::uint8_t is_strict_ireq = 0;
      std::uint8_t pad[3]{};
      double ls_val = 0.0;
    };
    std::vector<std::string> rev_paths(P);
    std::vector<std::uint64_t> rev_counts(P, 0);
    for (std::uint32_t p = 0; p < P; ++p) rev_paths[p] = scratch.file("xs_rupd_" + std::to_string(p) + ".bin");

    const double tol = std::max(1e-6, makespan * 1e-9);
    NeumaierSum slack_sum;
    std::uint64_t zero_slack = 0;

    for (std::uint32_t p = P; p-- > 0;) {
      const std::uint32_t p_begin = p * part_len;
      std::vector<XVertex> vs(part_sizes[p]);
      {
        std::FILE* vfp = std::fopen(v_paths[p].c_str(), "rb");
        read_all(vfp, vs.data(), vs.size());
        std::fclose(vfp);
      }
      for (auto& st : vs) st.lf = makespan;
      if (rev_counts[p] > 0) {
        std::vector<XRevUpdate> rupds(static_cast<std::size_t>(rev_counts[p]));
        std::FILE* rfp = std::fopen(rev_paths[p].c_str(), "rb");
        read_all(rfp, rupds.data(), rupds.size());
        std::fclose(rfp);
        for (const auto& u : rupds) {
          XVertex& v = vs[u.dst_src - p_begin];
          const double cand = u.is_strict_ireq ? (u.ls_val + v.weight) : u.ls_val;
          if (cand < v.lf) v.lf = cand;
        }
      }
      if (e_dst_counts[p] > 0) {
        std::vector<XEdge> edges(static_cast<std::size_t>(e_dst_counts[p]));
        std::FILE* dfp = std::fopen(e_dst_paths[p].c_str(), "rb");
        read_all(dfp, edges.data(), edges.size());
        std::fclose(dfp);
        std::vector<std::vector<XRevUpdate>> scatter(p);
        // Walk edges in reverse `dst` order (e_dst_paths[p] was written in ascending `dst` order).
        for (std::size_t k = edges.size(); k-- > 0;) {
          const XEdge& e = edges[k];
          const XVertex& v = vs[e.dst - p_begin];
          const double v_ls = v.lf - v.weight;
          const bool s_ireq = (strict_ireq && e.kind == static_cast<std::uint8_t>(EdgeKind::kIrequires));
          if (e.src >= p_begin) {
            XVertex& u = vs[e.src - p_begin];
            const double cand = s_ireq ? (v_ls + u.weight) : v_ls;
            if (cand < u.lf) u.lf = cand;
          } else {
            scatter[e.src / part_len].push_back({e.src, static_cast<std::uint8_t>(s_ireq ? 1 : 0), {}, v_ls});
          }
        }
        for (std::uint32_t q = 0; q < p; ++q) {
          if (scatter[q].empty()) continue;
          std::FILE* rfp = std::fopen(rev_paths[q].c_str(), "ab");
          write_all(rfp, scatter[q].data(), scatter[q].size());
          std::fclose(rfp);
          rev_counts[q] += scatter[q].size();
        }
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

std::unique_ptr<GraphEngine> make_xstream_engine() { return std::make_unique<XStreamEngine>(); }

}  // namespace cta::streaming::engines
