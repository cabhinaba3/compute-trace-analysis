#include "streaming/engines/frontier/frontier_engine.hpp"

#include <fcntl.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/util/delimited_message_util.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "et_def.pb.h"
#include "streaming/scratch.hpp"
#include "util/compensated_sum.hpp"
#include "util/util.hpp"

namespace cta::streaming::engines {
namespace {

constexpr std::uint32_t kNoParent = std::numeric_limits<std::uint32_t>::max();
constexpr std::int8_t kGoalOpSend = 1;
constexpr std::int8_t kGoalOpRecv = 2;

struct ChannelKey {
  std::int32_t src = -1;
  std::int32_t dst = -1;
  std::uint32_t tag = 0;
  bool operator==(const ChannelKey& o) const { return src == o.src && dst == o.dst && tag == o.tag; }
};

struct ChannelKeyHash {
  std::size_t operator()(const ChannelKey& c) const {
    return static_cast<std::size_t>(mix64(static_cast<std::uint64_t>(static_cast<std::uint32_t>(c.src)) *
                                              0x9e3779b97f4a7c15ULL ^
                                          mix64(static_cast<std::uint64_t>(static_cast<std::uint32_t>(c.dst)) *
                                                    0xc2b2ae3d27d4eb4fULL ^
                                                static_cast<std::uint64_t>(c.tag))));
  }
};

struct ChannelSeqKey {
  std::int32_t src = -1;
  std::int32_t dst = -1;
  std::uint32_t tag = 0;
  std::uint32_t seq = 0;
  bool operator==(const ChannelSeqKey& o) const {
    return src == o.src && dst == o.dst && tag == o.tag && seq == o.seq;
  }
};

struct ChannelSeqKeyHash {
  std::size_t operator()(const ChannelSeqKey& c) const {
    const std::uint64_t h1 = mix64(static_cast<std::uint64_t>(static_cast<std::uint32_t>(c.src)) *
                                       0x9e3779b97f4a7c15ULL ^
                                   static_cast<std::uint64_t>(static_cast<std::uint32_t>(c.dst)));
    const std::uint64_t h2 =
        mix64(static_cast<std::uint64_t>(c.tag) * 0xc2b2ae3d27d4eb4fULL ^ static_cast<std::uint64_t>(c.seq));
    return static_cast<std::size_t>(mix64(h1 ^ h2));
  }
};

struct FrontierDP {
  double es_lb = 0.0;
  double ef_lb = 0.0;
  double es_ub = 0.0;
  double ef_ub = 0.0;
  std::int32_t depth = 1;
  std::uint32_t parent_vid = kNoParent;
  double cp_compute = 0.0;
  double cp_comm = 0.0;
  std::int64_t cp_comm_bytes = 0;
};

// Deterministic tie-breaking matching native/src/analysis/dag.cpp:
// prefer larger finish time, then deeper path, then lowest original vertex id.
bool beats_parent(double cand_finish, std::int32_t cand_depth, std::uint32_t cand_vid, double cur_start,
                  std::int32_t cur_depth, std::uint32_t cur_parent) {
  if (cand_finish > cur_start) return true;
  if (cand_finish < cur_start) return false;
  if (cand_depth + 1 > cur_depth) return true;
  if (cand_depth + 1 < cur_depth) return false;
  return cur_parent == kNoParent || cand_vid < cur_parent;
}

// Compact Bloom filter for sound eviction-membership checks on sparse/non-
// monotonic 64-bit node IDs: zero false negatives (if !might_contain, the id
// was definitely never evicted).
class EvictionBloom {
 public:
  explicit EvictionBloom(std::size_t num_bits = 1u << 19)  // 64 KiB per rank
      : bits_((num_bits + 63) / 64, 0), mask_(bits_.size() * 64 - 1) {}

  void insert(std::uint64_t id) {
    const std::uint64_t h1 = mix64(id ^ 0x243f6a8885a308d3ULL);
    const std::uint64_t h2 = mix64(id ^ 0x13198a2e03707344ULL) | 1ULL;
    for (int k = 0; k < 4; ++k) {
      const std::size_t bit = static_cast<std::size_t>((h1 + static_cast<std::uint64_t>(k) * h2) & mask_);
      bits_[bit >> 6] |= (std::uint64_t{1} << (bit & 63));
    }
  }

  bool might_contain(std::uint64_t id) const {
    const std::uint64_t h1 = mix64(id ^ 0x243f6a8885a308d3ULL);
    const std::uint64_t h2 = mix64(id ^ 0x13198a2e03707344ULL) | 1ULL;
    for (int k = 0; k < 4; ++k) {
      const std::size_t bit = static_cast<std::size_t>((h1 + static_cast<std::uint64_t>(k) * h2) & mask_);
      if ((bits_[bit >> 6] & (std::uint64_t{1} << (bit & 63))) == 0) return false;
    }
    return true;
  }

 private:
  std::vector<std::uint64_t> bits_;
  std::size_t mask_;
};

// Per-rank Certified Eviction-Epoch tracker.
// Stores coarse (min_id, max_id, min_ef_lb, max_ef_ub, dense_contiguous)
// buckets of 4096 evicted nodes (~2 MB per 1B nodes) + Bloom guard for sparse
// or non-monotonic 64-bit IDs.
class EvictionTracker {
 public:
  static constexpr std::uint32_t kBucketSize = 4096;

  void record(std::uint64_t id, double ef_lb, double ef_ub) {
    if (!has_evicted_) {
      has_evicted_ = true;
      last_id_ = id;
    } else {
      if (id <= last_id_) monotonic_ = false;
      last_id_ = id;
    }
    if (ef_ub > global_max_ub_) global_max_ub_ = ef_ub;
    bloom_.insert(id);

    if (buckets_.empty() || buckets_.back().count >= kBucketSize) {
      Bucket b;
      b.min_id = b.max_id = id;
      b.min_ef_lb = ef_lb;
      b.max_ef_ub = ef_ub;
      b.count = 1;
      b.dense_contiguous = true;
      buckets_.push_back(b);
    } else {
      Bucket& b = buckets_.back();
      if (id != b.max_id + 1) b.dense_contiguous = false;
      if (id < b.min_id) b.min_id = id;
      if (id > b.max_id) b.max_id = id;
      if (ef_lb < b.min_ef_lb) b.min_ef_lb = ef_lb;
      if (ef_ub > b.max_ef_ub) b.max_ef_ub = ef_ub;
      ++b.count;
    }
  }

  // Returns true if `id` was (or might have been) evicted, writing mathematically
  // sound [lb, ub] bounds on the evicted parent's earliest finish.
  bool lookup_evicted(std::uint64_t id, double& out_lb, double& out_ub) const {
    if (!has_evicted_) return false;
    if (monotonic_) {
      auto it = std::lower_bound(buckets_.begin(), buckets_.end(), id,
                                 [](const Bucket& b, std::uint64_t k) { return b.max_id < k; });
      if (it == buckets_.end() || id < it->min_id) return false;
      if (it->dense_contiguous) {
        // Provably existed in this contiguous bucket!
        out_lb = it->min_ef_lb;
        out_ub = it->max_ef_ub;
        return true;
      }
      // Sparse monotonic bucket: Bloom negative proves it was never evicted.
      if (!bloom_.might_contain(id)) return false;
      // Possible Bloom false positive for a dangling ID -> lb must stay 0.0 for soundness.
      out_lb = 0.0;
      out_ub = it->max_ef_ub;
      return true;
    }
    if (!bloom_.might_contain(id)) return false;
    out_lb = 0.0;
    out_ub = global_max_ub_;
    return true;
  }

  bool empty() const { return !has_evicted_; }

 private:
  struct Bucket {
    std::uint64_t min_id = 0;
    std::uint64_t max_id = 0;
    double min_ef_lb = 0.0;
    double max_ef_ub = 0.0;
    std::uint32_t count = 0;
    bool dense_contiguous = true;
  };

  bool has_evicted_ = false;
  bool monotonic_ = true;
  std::uint64_t last_id_ = 0;
  double global_max_ub_ = 0.0;
  std::vector<Bucket> buckets_;
  EvictionBloom bloom_;
};

// Lightweight on-disk append-only parent log for CP op chain extraction.
struct ParentLogRecord {
  std::uint32_t parent_vid = kNoParent;
  std::uint32_t rank = 0;
  std::uint64_t node_id = 0;
  std::uint64_t size_bytes = 0;
  double weight_ns = 0.0;
  double es_ns = 0.0;
  double ef_ns = 0.0;
  std::uint8_t op_class = 0;
  char name[47]{};
};

class DiskParentLog {
 public:
  explicit DiskParentLog(const std::string& path) : path_(path) {
    fp_ = std::fopen(path.c_str(), "w+b");
    if (!fp_) throw std::runtime_error("cannot open parent log " + path);
  }
  ~DiskParentLog() {
    if (fp_) std::fclose(fp_);
  }

  void write(std::uint32_t vid, const ParentLogRecord& rec) {
    if (vid == next_vid_) {
      if (std::fwrite(&rec, sizeof(rec), 1, fp_) != 1) throw std::runtime_error("parent log write failed");
      ++next_vid_;
      return;
    }
    if (std::fseek(fp_, static_cast<long>(static_cast<std::uint64_t>(vid) * sizeof(rec)), SEEK_SET) != 0)
      throw std::runtime_error("parent log seek failed");
    if (std::fwrite(&rec, sizeof(rec), 1, fp_) != 1) throw std::runtime_error("parent log write failed");
    if (vid >= next_vid_) next_vid_ = vid + 1;
    std::fseek(fp_, 0, SEEK_END);
  }

  ParentLogRecord read(std::uint32_t vid) {
    ParentLogRecord rec{};
    if (std::fseek(fp_, static_cast<long>(static_cast<std::uint64_t>(vid) * sizeof(rec)), SEEK_SET) != 0)
      throw std::runtime_error("parent log seek failed");
    if (std::fread(&rec, sizeof(rec), 1, fp_) != 1) throw std::runtime_error("parent log read failed");
    return rec;
  }

  void flush() {
    if (fp_) std::fflush(fp_);
  }

 private:
  std::string path_;
  std::FILE* fp_ = nullptr;
  std::uint32_t next_vid_ = 0;
};

void fill_cp_op_chain_from_log(DiskParentLog& log, std::uint32_t sink_vid, std::uint64_t total_nodes,
                               DagStreamResult& out) {
  log.flush();
  std::vector<ParentLogRecord> chain;
  std::unordered_set<std::uint32_t> seen;
  long double cp_comp = 0.0L, cp_comm = 0.0L;
  std::int64_t cp_bytes = 0;
  for (std::uint32_t u = sink_vid; u != kNoParent && seen.insert(u).second && chain.size() <= total_nodes;) {
    ParentLogRecord rec = log.read(u);
    if (is_compute_class(rec.op_class)) cp_comp += rec.weight_ns;
    if (is_comm_class(rec.op_class)) {
      cp_comm += rec.weight_ns;
      cp_bytes += static_cast<std::int64_t>(rec.size_bytes);
    }
    chain.push_back(rec);
    u = rec.parent_vid;
  }
  out.critical_compute = static_cast<double>(cp_comp);
  out.critical_comm = static_cast<double>(cp_comm);
  out.critical_comm_bytes = cp_bytes;

  std::reverse(chain.begin(), chain.end());  // source -> sink order
  out.cp_op_rank.reserve(chain.size());
  out.cp_op_node_id.reserve(chain.size());
  out.cp_op_class.reserve(chain.size());
  out.cp_op_weight_ns.reserve(chain.size());
  out.cp_op_size_bytes.reserve(chain.size());
  out.cp_op_earliest_start_ns.reserve(chain.size());
  out.cp_op_earliest_finish_ns.reserve(chain.size());
  out.cp_op_name.reserve(chain.size());
  for (const auto& rec : chain) {
    out.cp_op_rank.push_back(rec.rank);
    out.cp_op_node_id.push_back(rec.node_id);
    out.cp_op_class.push_back(rec.op_class);
    out.cp_op_weight_ns.push_back(rec.weight_ns);
    out.cp_op_size_bytes.push_back(rec.size_bytes);
    out.cp_op_earliest_start_ns.push_back(rec.es_ns);
    out.cp_op_earliest_finish_ns.push_back(rec.ef_ns);
    out.cp_op_name.emplace_back(rec.name);
  }
}

void copy_name_trunc(char (&dst)[47], std::string_view src) {
  const std::size_t n = std::min<std::size_t>(src.size(), sizeof(dst) - 1);
  std::memcpy(dst, src.data(), n);
  dst[n] = '\0';
}

// -----------------------------------------------------------------------------
// Engine B2: 1-pass Windowed Frontier over any GraphSource (or multi-rank stream)
// -----------------------------------------------------------------------------
struct WindowNode {
  std::uint32_t vid = 0;
  std::uint32_t rank = 0;
  std::uint64_t node_id = 0;
  std::uint8_t op_class = 0;
  double weight_ns = 0.0;
  std::uint64_t size_bytes = 0;
  std::uint32_t window_epoch = 0;
  std::string name;
  FrontierDP dp;
  bool has_local_child = false;
  bool on_barrier_path = false;
};

struct UnresolvedNode {
  std::uint32_t vid = 0;
  std::uint32_t rank = 0;
  std::uint64_t node_id = 0;
  std::uint8_t op_class = 0;
  double weight_ns = 0.0;
  std::uint64_t size_bytes = 0;
  std::int32_t src_rank = -1;
  std::int32_t dst_rank = -1;
  std::uint32_t tag = 0;
  std::uint32_t p2p_seq = 0;
  std::string name;
  std::string comm_type;
  std::string comm_group;
  std::vector<std::pair<std::uint64_t, std::uint8_t>> missing_deps;  // (parent_id, edge_kind)
  // Already-resolved best incoming bounds before remaining missing parents arrive.
  double best_start_lb = 0.0;
  double best_start_ub = 0.0;
  std::int32_t best_depth = 0;
  std::uint32_t best_parent = kNoParent;
  double best_cp_compute = 0.0;
  double best_cp_comm = 0.0;
  std::int64_t best_cp_bytes = 0;
  bool waiting_p2p = false;
  bool waiting_barrier = false;
};

struct CollBarrierKey {
  std::string group;
  std::string comm_type;
  std::uint32_t seq = 0;
  bool operator==(const CollBarrierKey& o) const {
    return seq == o.seq && comm_type == o.comm_type && group == o.group;
  }
};

struct CollBarrierKeyHash {
  std::size_t operator()(const CollBarrierKey& k) const {
    std::size_t h = std::hash<std::string>{}(k.group);
    h ^= std::hash<std::string>{}(k.comm_type) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= static_cast<std::size_t>(mix64(k.seq));
    return h;
  }
};

struct WindowEdge {
  std::uint32_t u_vid = 0;
  std::uint32_t v_vid = 0;
  std::uint8_t kind = 0;
};

struct WindowBarrierState {
  std::uint32_t expected_ranks = 0;
  std::vector<std::uint32_t> participant_vids;
  double max_pre_lb = 0.0;
  double max_pre_ub = 0.0;
  std::int32_t max_pre_depth = 0;
  std::uint32_t max_pre_vid = kNoParent;
  double max_pre_cp_compute = 0.0;
  double max_pre_cp_comm = 0.0;
  std::int64_t max_pre_cp_bytes = 0;

  double max_post_lb = 0.0;
  double max_post_ub = 0.0;
  std::int32_t max_post_depth = 0;
  std::uint32_t max_post_vid = kNoParent;
  double max_post_cp_compute = 0.0;
  double max_post_cp_comm = 0.0;
  std::int64_t max_post_cp_bytes = 0;
};

// Stateful incremental 1-pass windowed frontier solver.
class StreamFrontierSolver {
 public:
  explicit StreamFrontierSolver(const DagSolveOptions& opts) : opts_(opts) {
    if (opts_.extract_critical_path_ops) {
      scratch_ = std::make_unique<ScratchDir>(opts_.scratch_dir);
      parent_log_ = std::make_unique<DiskParentLog>(scratch_->file("parent.bin"));
    }
    res_.provenance.engine = "frontier";
    res_.provenance.goal_semantics = opts_.goal_semantics == GoalSemantics::kStrict ? "strict" : "cta_legacy";
    res_.provenance.collective_sync = opts_.collective_sync == CollectiveSync::kBarrier ? "barrier" : "none";
  }

  void set_expected_world_size(std::uint32_t ws) { expected_world_size_ = ws; }

  void feed_chunk(const GraphChunk& chunk) {
    const NodeBatch& nb = chunk.nodes;
    const EdgeBatch& eb = chunk.edges;

    // Group edges in this chunk by (rank, dst_id) so each arriving node sees its
    // incoming intra-rank edges immediately (for Chakra .et/.json where edges
    // accompany the destination node), while also handling GOAL text where
    // edges may arrive in later lines/chunks.
    std::unordered_map<std::uint64_t, std::vector<std::pair<std::uint64_t, std::uint8_t>>> incoming_in_chunk;
    std::unordered_set<std::uint64_t> nodes_in_chunk;
    nodes_in_chunk.reserve(nb.size() * 2 + 1);
    for (std::size_t i = 0; i < nb.size(); ++i) {
      nodes_in_chunk.insert(rank_node_key(nb.rank[i], nb.node_id[i]));
    }
    std::vector<std::size_t> post_edges;
    for (std::size_t e = 0; e < eb.size(); ++e) {
      const std::uint64_t dst_k = rank_node_key(eb.rank[e], eb.dst_id[e]);
      if (nodes_in_chunk.count(dst_k)) {
        incoming_in_chunk[dst_k].emplace_back(eb.src_id[e], eb.edge_kind[e]);
      } else {
        post_edges.push_back(e);
      }
    }

    for (std::size_t i = 0; i < nb.size(); ++i) {
      const std::uint32_t r = nb.rank[i];
      const std::uint64_t nid = nb.node_id[i];
      const std::uint8_t oc = nb.op_class[i];
      const double w = node_weight_ns(nb.duration_ns[i], oc, nb.size_bytes[i]);
      const std::uint32_t vid = next_vid_++;
      ++res_.num_nodes;
      seen_ranks_.insert(r);

      std::uint32_t p2p_seq = 0;
      if (oc == static_cast<std::uint8_t>(OpClass::kCommSend) && nb.dst_rank[i] >= 0) {
        p2p_seq = send_seq_[ChannelKey{nb.src_rank[i], nb.dst_rank[i], nb.message_tag[i]}]++;
      } else if (oc == static_cast<std::uint8_t>(OpClass::kCommRecv) && nb.src_rank[i] >= 0) {
        p2p_seq = recv_seq_[ChannelKey{nb.src_rank[i], nb.dst_rank[i], nb.message_tag[i]}]++;
      }

      auto it = incoming_in_chunk.find(rank_node_key(r, nid));
      std::vector<std::pair<std::uint64_t, std::uint8_t>> deps;
      if (it != incoming_in_chunk.end()) deps = std::move(it->second);

      ingest_node(vid, r, nid, oc, w, nb.size_bytes[i], nb.src_rank[i], nb.dst_rank[i], nb.message_tag[i], p2p_seq,
                  nb.name_at(i), nb.comm_type_at(i), nb.comm_group_at(i), deps);
    }

    // Apply any edges whose destination arrived in an earlier chunk (e.g., GOAL text).
    for (std::size_t e : post_edges) {
      apply_late_edge(eb.rank[e], eb.src_id[e], eb.dst_id[e], eb.edge_kind[e]);
    }
  }

  DagStreamResult finish() {
    // 1. Strip truly-dangling parent IDs (not present in any unresolved node or
    //    parked barrier) and unmatched P2P waits, then cascade-release in exact
    //    topological order so downstream children wake AFTER their parents.
    std::unordered_set<std::uint64_t> pending_keys;
    pending_keys.reserve(unresolved_.size() * 2 + barrier_parked_.size() * 2 + 1);
    for (const auto& [vid, un] : unresolved_) pending_keys.insert(rank_node_key(un.rank, un.node_id));
    for (const auto& [vid, un] : barrier_parked_) pending_keys.insert(rank_node_key(un.rank, un.node_id));

    std::vector<std::uint32_t> ready_to_release;
    for (auto& [vid, un] : unresolved_) {
      for (auto it = un.missing_deps.begin(); it != un.missing_deps.end();) {
        if (!pending_keys.count(rank_node_key(un.rank, it->first))) {
          // Provably dangling parent: ignored per CTA semantics.
          it = un.missing_deps.erase(it);
        } else {
          ++it;
        }
      }
      if (un.waiting_p2p) {
        // Sender never arrived in the entire trace -> provably unmatched recv.
        const ChannelSeqKey ck{un.src_rank, un.dst_rank, un.tag, un.p2p_seq};
        waiting_p2p_recvs_.erase(ck);
        un.waiting_p2p = false;
        ++res_.provenance.unmatched_p2p_recvs;
      }
      if (un.missing_deps.empty()) ready_to_release.push_back(vid);
    }
    std::sort(ready_to_release.begin(), ready_to_release.end());
    for (std::uint32_t vid : ready_to_release) {
      auto it = unresolved_.find(vid);
      if (it == unresolved_.end()) continue;
      UnresolvedNode un = std::move(it->second);
      unresolved_.erase(it);
      finalize_node(un);
    }

    // 2. Flush any partial barrier groups at EOF (may wake downstream waiters).
    while (!active_barriers_.empty()) {
      auto bit = active_barriers_.begin();
      WindowBarrierState bs = std::move(bit->second);
      active_barriers_.erase(bit);
      release_barrier_participants(bs, false);
    }

    // 3. Anything still in unresolved_ after dangling-dep stripping and barrier
    //    flushing is held back by a true dependency cycle!
    if (!unresolved_.empty()) {
      res_.unordered += unresolved_.size();
      res_.provenance.has_cycle = true;
      res_.provenance.unordered_nodes = res_.unordered;
      std::vector<std::uint32_t> rem_vids;
      rem_vids.reserve(unresolved_.size());
      for (const auto& [vid, _] : unresolved_) rem_vids.push_back(vid);
      std::sort(rem_vids.begin(), rem_vids.end());
      if (opts_.fail_on_cycle) throw std::runtime_error("cycle detected in streaming frontier graph");
      for (std::uint32_t vid : rem_vids) {
        auto it = unresolved_.find(vid);
        if (it == unresolved_.end()) continue;
        UnresolvedNode un = std::move(it->second);
        unresolved_.erase(it);
        finalize_node(un);
      }
    }

    // 4. Count unmatched P2P sends left in pending_sends_.
    res_.provenance.unmatched_p2p_sends += pending_sends_.size();

    // 4. Finalize critical path & provenance.
    if (res_.num_nodes > 0) {
      res_.makespan = global_makespan_lb_;
      res_.depth = global_max_depth_;
      res_.provenance.makespan_lb = global_makespan_lb_;
      res_.provenance.makespan_ub = global_makespan_ub_;

      if (parent_log_ && global_sink_vid_ != kNoParent) {
        fill_cp_op_chain_from_log(*parent_log_, global_sink_vid_, res_.num_nodes, res_);
      } else {
        res_.critical_compute = global_cp_compute_;
        res_.critical_comm = global_cp_comm_;
        res_.critical_comm_bytes = global_cp_comm_bytes_;
      }

      if (computed_window_slack_ && window_slack_nodes_ > 0) {
        // Check remaining nodes in the final window as well.
        compute_final_window_slack();
        res_.mean_slack = static_cast<double>(window_slack_sum_.value() / static_cast<long double>(window_slack_nodes_));
        res_.zero_slack_ratio =
            static_cast<double>(window_zero_slack_) / static_cast<double>(window_slack_nodes_);
        const bool provably_global =
            (res_.provenance.straddling_edges == 0 && res_.provenance.dead_end_sinks == 0 &&
             res_.provenance.evicted_dep_hits == 0 && res_.provenance.lookahead_drops == 0);
        res_.provenance.slack_scope = provably_global ? "global_exact" : "window_local";
      } else {
        res_.provenance.slack_scope = "unavailable";
      }

      res_.provenance.exact =
          (res_.provenance.evicted_dep_hits == 0 && res_.provenance.lookahead_drops == 0 && !res_.provenance.has_cycle &&
           res_.provenance.slack_scope != "window_local");
    } else {
      res_.provenance.exact = true;
      res_.provenance.slack_scope = "global_exact";
    }
    return std::move(res_);
  }

 private:
  static std::uint64_t rank_node_key(std::uint32_t r, std::uint64_t nid) {
    return mix64(static_cast<std::uint64_t>(r) * 0x9e3779b97f4a7c15ULL ^ mix64(nid));
  }

  struct RankWindow {
    std::deque<std::uint64_t> order;
    std::unordered_map<std::uint64_t, WindowNode> nodes;
    EvictionTracker evicted;
  };

  void relax_from_pred(UnresolvedNode& dst, double pred_val_lb, double pred_val_ub, std::int32_t pred_depth,
                       std::uint32_t pred_vid, double pred_cp_comp, double pred_cp_comm, std::int64_t pred_cp_bytes) {
    if (pred_val_ub > dst.best_start_ub) dst.best_start_ub = pred_val_ub;
    if (beats_parent(pred_val_lb, pred_depth, pred_vid, dst.best_start_lb, dst.best_depth, dst.best_parent)) {
      dst.best_start_lb = pred_val_lb;
      dst.best_depth = pred_depth + 1;
      dst.best_parent = pred_vid;
      dst.best_cp_compute = pred_cp_comp;
      dst.best_cp_comm = pred_cp_comm;
      dst.best_cp_bytes = pred_cp_bytes;
    }
  }

  void ingest_node(std::uint32_t vid, std::uint32_t rank, std::uint64_t nid, std::uint8_t oc, double w,
                   std::uint64_t size_bytes, std::int32_t src_rank, std::int32_t dst_rank, std::uint32_t tag,
                   std::uint32_t p2p_seq, std::string_view name, std::string_view comm_type,
                   std::string_view comm_group, const std::vector<std::pair<std::uint64_t, std::uint8_t>>& raw_deps) {
    UnresolvedNode un;
    un.vid = vid;
    un.rank = rank;
    un.node_id = nid;
    un.op_class = oc;
    un.weight_ns = w;
    un.size_bytes = size_bytes;
    un.src_rank = src_rank;
    un.dst_rank = dst_rank;
    un.tag = tag;
    un.p2p_seq = p2p_seq;
    un.name.assign(name.data(), name.size());
    un.comm_type.assign(comm_type.data(), comm_type.size());
    un.comm_group.assign(comm_group.data(), comm_group.size());
    un.best_depth = 1;

    RankWindow& rw = rank_windows_[rank];

    // Deduplicate incoming intra-rank deps for this node (keeping finish-to-start
    // over strict irequires if both exist between the same pair).
    std::unordered_map<std::uint64_t, std::uint8_t> dedup_deps;
    for (const auto& [src_id, kind] : raw_deps) {
      auto it = dedup_deps.find(src_id);
      if (it == dedup_deps.end()) {
        dedup_deps.emplace(src_id, kind);
      } else if (it->second == static_cast<std::uint8_t>(EdgeKind::kIrequires) &&
                 kind != static_cast<std::uint8_t>(EdgeKind::kIrequires)) {
        it->second = kind;
      }
    }

    for (const auto& [src_id, kind] : dedup_deps) {
      auto it = rw.nodes.find(src_id);
      if (it != rw.nodes.end()) {
        WindowNode& pred = it->second;
        pred.has_local_child = true;
        if (opts_.collective_sync == CollectiveSync::kBarrier && pred.window_epoch < current_window_epoch_ &&
            !pred.on_barrier_path) {
          ++res_.provenance.straddling_edges;
        }
        ++res_.num_edges;
        if (opts_.collective_sync == CollectiveSync::kBarrier) {
          window_edges_.push_back({pred.vid, vid, kind});
        }
        const bool strict_ireq = (opts_.goal_semantics == GoalSemantics::kStrict &&
                                  kind == static_cast<std::uint8_t>(EdgeKind::kIrequires));
        const double p_lb = strict_ireq ? pred.dp.es_lb : pred.dp.ef_lb;
        const double p_ub = strict_ireq ? pred.dp.es_ub : pred.dp.ef_ub;
        relax_from_pred(un, p_lb, p_ub, pred.dp.depth, pred.vid, pred.dp.cp_compute, pred.dp.cp_comm,
                        pred.dp.cp_comm_bytes);
      } else {
        double ev_lb = 0.0, ev_ub = 0.0;
        if (rw.evicted.lookup_evicted(src_id, ev_lb, ev_ub)) {
          ++res_.provenance.evicted_dep_hits;
          ++res_.num_edges;
          if (opts_.collective_sync == CollectiveSync::kBarrier) ++res_.provenance.straddling_edges;
          relax_from_pred(un, ev_lb, ev_ub, 1, kNoParent, 0.0, 0.0, 0);
        } else {
          un.missing_deps.emplace_back(src_id, kind);
        }
      }
    }

    // Check P2P incoming dependency for comm_recv.
    if (oc == static_cast<std::uint8_t>(OpClass::kCommRecv) && src_rank >= 0) {
      const ChannelSeqKey ck{src_rank, dst_rank, tag, p2p_seq};
      auto sit = pending_sends_.find(ck);
      if (sit != pending_sends_.end()) {
        const P2PSendState st = sit->second;
        pending_sends_.erase(sit);
        ++res_.num_edges;
        if (opts_.collective_sync == CollectiveSync::kBarrier) {
          window_edges_.push_back({st.vid, vid, static_cast<std::uint8_t>(EdgeKind::kP2P)});
        }
        relax_from_pred(un, st.ef_lb, st.ef_ub, st.depth, st.vid, st.cp_compute, st.cp_comm, st.cp_comm_bytes);
      } else {
        un.waiting_p2p = true;
      }
    }

    if (!un.missing_deps.empty() || un.waiting_p2p) {
      // Park until missing parents or matching P2P send arrives (capped by max_lookahead).
      const std::uint32_t uvid = un.vid;
      for (const auto& [pid, kind] : un.missing_deps) {
        waiting_on_parent_[rank][pid].push_back(uvid);
      }
      if (un.waiting_p2p) {
        waiting_p2p_recvs_[ChannelSeqKey{src_rank, dst_rank, tag, p2p_seq}] = uvid;
      }
      unresolved_order_.push_back(uvid);
      unresolved_.emplace(uvid, std::move(un));
      enforce_lookahead_cap();
      return;
    }

    finalize_node(un);
  }

  void enforce_lookahead_cap() {
    while (unresolved_.size() > opts_.max_lookahead && !unresolved_order_.empty()) {
      const std::uint32_t oldest = unresolved_order_.front();
      unresolved_order_.pop_front();
      auto it = unresolved_.find(oldest);
      if (it == unresolved_.end()) continue;
      UnresolvedNode un = std::move(it->second);
      unresolved_.erase(it);
      if (!un.missing_deps.empty()) {
        ++res_.provenance.lookahead_drops;
        un.best_start_ub = std::max(un.best_start_ub, global_makespan_ub_);
      }
      if (un.waiting_p2p) {
        waiting_p2p_recvs_.erase(ChannelSeqKey{un.src_rank, un.dst_rank, un.tag, un.p2p_seq});
        ++res_.provenance.lookahead_drops;
        un.best_start_ub = std::max(un.best_start_ub, global_makespan_ub_);
      }
      finalize_node(un);
    }
  }

  void finalize_node(UnresolvedNode& un) {
    // Handle optional 2-node virtual collective barriers.
    if (opts_.collective_sync == CollectiveSync::kBarrier &&
        un.op_class == static_cast<std::uint8_t>(OpClass::kCommColl) && !un.waiting_barrier) {
      const std::uint32_t gsize = collective_group_size(un.comm_type);
      if (gsize > 1) {
        const std::string grp = un.comm_group.empty() ? "world" : un.comm_group;
        const std::uint32_t seq = coll_seq_[un.rank][grp + ":" + un.comm_type]++;
        const CollBarrierKey bk{grp, un.comm_type, seq};
        WindowBarrierState& bs = active_barriers_[bk];
        bs.expected_ranks = gsize;
        bs.participant_vids.push_back(un.vid);
        if (un.best_start_ub > bs.max_pre_ub) bs.max_pre_ub = un.best_start_ub;
        if (beats_parent(un.best_start_lb, un.best_depth - 1, un.best_parent, bs.max_pre_lb, bs.max_pre_depth,
                         bs.max_pre_vid)) {
          bs.max_pre_lb = un.best_start_lb;
          bs.max_pre_depth = un.best_depth;
          bs.max_pre_vid = un.best_parent;
          bs.max_pre_cp_compute = un.best_cp_compute;
          bs.max_pre_cp_comm = un.best_cp_comm;
          bs.max_pre_cp_bytes = un.best_cp_bytes;
        }
        un.waiting_barrier = true;
        barrier_parked_[un.vid] = std::move(un);
        if (bs.participant_vids.size() >= bs.expected_ranks) {
          WindowBarrierState done = std::move(bs);
          active_barriers_.erase(bk);
          release_barrier_participants(done, done.expected_ranks >= current_world_size());
        }
        return;
      }
    }

    commit_node(un);
  }

  void release_barrier_participants(WindowBarrierState& bs, bool is_global_cut) {
    // Virtual barrier B_in adds 1 edge from each rank's pre-state, and B_out
    // synchronizes completion across all participating ranks (2 * G edges).
    res_.num_edges += 2 * static_cast<std::uint64_t>(bs.participant_vids.size());

    // First compute each participant's post-collective finish after B_in.
    for (std::uint32_t pvid : bs.participant_vids) {
      auto it = barrier_parked_.find(pvid);
      if (it == barrier_parked_.end()) continue;
      UnresolvedNode& un = it->second;
      if (bs.max_pre_ub > un.best_start_ub) un.best_start_ub = bs.max_pre_ub;
      if (beats_parent(bs.max_pre_lb, bs.max_pre_depth - 1, bs.max_pre_vid, un.best_start_lb, un.best_depth,
                       un.best_parent)) {
        un.best_start_lb = bs.max_pre_lb;
        un.best_depth = bs.max_pre_depth;
        un.best_parent = bs.max_pre_vid;
        un.best_cp_compute = bs.max_pre_cp_compute;
        un.best_cp_comm = bs.max_pre_cp_comm;
        un.best_cp_bytes = bs.max_pre_cp_bytes;
      }
      const double ef_lb = un.best_start_lb + un.weight_ns;
      const double ef_ub = un.best_start_ub + un.weight_ns;
      if (ef_ub > bs.max_post_ub) bs.max_post_ub = ef_ub;
      if (beats_parent(ef_lb, un.best_depth, un.vid, bs.max_post_lb, bs.max_post_depth, bs.max_post_vid)) {
        bs.max_post_lb = ef_lb;
        bs.max_post_depth = un.best_depth + 1;
        bs.max_post_vid = un.vid;
        bs.max_post_cp_compute = un.best_cp_compute;
        bs.max_post_cp_comm = un.best_cp_comm + un.weight_ns;
        bs.max_post_cp_bytes = un.best_cp_bytes + static_cast<std::int64_t>(un.size_bytes);
      }
    }

    std::vector<WindowNode> committed_barriers;
    std::vector<std::uint32_t> barrier_vids;
    for (std::uint32_t pvid : bs.participant_vids) {
      auto it = barrier_parked_.find(pvid);
      if (it == barrier_parked_.end()) continue;
      UnresolvedNode un = std::move(it->second);
      barrier_parked_.erase(it);
      // Commit barrier nodes first WITHOUT waking step t+1 waiters yet so
      // window_nodes_ contains strictly window t during cut certification.
      WindowNode wn = commit_node(un, /*override_ef_lb=*/bs.max_post_lb, /*override_ef_ub=*/bs.max_post_ub,
                                  /*is_barrier_out=*/true, bs.max_post_cp_compute, bs.max_post_cp_comm,
                                  bs.max_post_cp_bytes, bs.max_post_vid, /*defer_wake=*/true);
      committed_barriers.push_back(std::move(wn));
      barrier_vids.push_back(pvid);
    }

    if (is_global_cut && !barrier_vids.empty()) {
      compute_window_slack_at_barrier(barrier_vids, bs.max_post_lb);
      ++current_window_epoch_;
    }

    // Now wake step t+1 waiters downstream of each barrier participant.
    for (const WindowNode& wn : committed_barriers) {
      wake_intra_rank_waiters(wn);
    }
  }

  WindowNode commit_node(const UnresolvedNode& un, double override_ef_lb = -1.0, double override_ef_ub = -1.0,
                         bool is_barrier_out = false, double override_cp_comp = -1.0, double override_cp_comm = -1.0,
                         std::int64_t override_cp_bytes = -1, std::uint32_t override_b_vid = kNoParent,
                         bool defer_wake = false) {
    WindowNode wn;
    wn.vid = un.vid;
    wn.rank = un.rank;
    wn.node_id = un.node_id;
    wn.op_class = un.op_class;
    wn.weight_ns = un.weight_ns;
    wn.size_bytes = un.size_bytes;
    wn.window_epoch = current_window_epoch_;
    wn.name = un.name;
    wn.on_barrier_path = is_barrier_out;
    wn.dp.es_lb = un.best_start_lb;
    wn.dp.es_ub = un.best_start_ub;
    wn.dp.ef_lb = override_ef_lb >= 0.0 ? override_ef_lb : (un.best_start_lb + un.weight_ns);
    wn.dp.ef_ub = override_ef_ub >= 0.0 ? override_ef_ub : (un.best_start_ub + un.weight_ns);
    wn.dp.depth = un.best_depth;
    wn.dp.parent_vid = un.best_parent;
    wn.dp.cp_compute =
        override_cp_comp >= 0.0 ? override_cp_comp : (un.best_cp_compute + (is_compute_class(un.op_class) ? un.weight_ns : 0.0));
    wn.dp.cp_comm =
        override_cp_comm >= 0.0 ? override_cp_comm : (un.best_cp_comm + (is_comm_class(un.op_class) ? un.weight_ns : 0.0));
    wn.dp.cp_comm_bytes =
        override_cp_bytes >= 0
            ? override_cp_bytes
            : (un.best_cp_bytes + (is_comm_class(un.op_class) ? static_cast<std::int64_t>(un.size_bytes) : 0));
    if (is_barrier_out && override_b_vid != kNoParent && override_b_vid != un.vid) {
      wn.dp.parent_vid = override_b_vid;
    }

    if (wn.dp.ef_ub > global_makespan_ub_) global_makespan_ub_ = wn.dp.ef_ub;
    if (wn.dp.depth > global_max_depth_) global_max_depth_ = wn.dp.depth;
    if (wn.dp.ef_lb > global_makespan_lb_ ||
        (wn.dp.ef_lb == global_makespan_lb_ && (global_sink_vid_ == kNoParent || wn.vid < global_sink_vid_))) {
      global_makespan_lb_ = wn.dp.ef_lb;
      global_sink_vid_ = wn.vid;
      global_cp_compute_ = wn.dp.cp_compute;
      global_cp_comm_ = wn.dp.cp_comm;
      global_cp_comm_bytes_ = wn.dp.cp_comm_bytes;
    }

    if (parent_log_) {
      ParentLogRecord rec;
      rec.parent_vid = wn.dp.parent_vid;
      rec.rank = wn.rank;
      rec.node_id = wn.node_id;
      rec.size_bytes = wn.size_bytes;
      rec.weight_ns = wn.weight_ns;
      rec.es_ns = wn.dp.es_lb;
      rec.ef_ns = wn.dp.ef_lb;
      rec.op_class = wn.op_class;
      copy_name_trunc(rec.name, wn.name);
      parent_log_->write(wn.vid, rec);
    }

    if (opts_.collective_sync == CollectiveSync::kBarrier) {
      window_nodes_[wn.vid] = {wn.vid, wn.weight_ns, wn.dp.ef_lb, wn.window_epoch, is_barrier_out, false};
    }

    // Store into rank's active window and evict oldest if exceeding window_size.
    RankWindow& rw = rank_windows_[wn.rank];
    rw.nodes[wn.node_id] = wn;
    rw.order.push_back(wn.node_id);
    while (rw.order.size() > opts_.window_size) {
      const std::uint64_t ev_id = rw.order.front();
      rw.order.pop_front();
      auto eit = rw.nodes.find(ev_id);
      if (eit != rw.nodes.end()) {
        rw.evicted.record(ev_id, eit->second.dp.ef_lb, eit->second.dp.ef_ub);
        rw.nodes.erase(eit);
      }
    }

    // If this node is a P2P send, publish or wake waiting matching recv.
    if (wn.op_class == static_cast<std::uint8_t>(OpClass::kCommSend) && un.dst_rank >= 0) {
      const ChannelSeqKey ck{un.src_rank, un.dst_rank, un.tag, un.p2p_seq};
      auto wit = waiting_p2p_recvs_.find(ck);
      if (wit != waiting_p2p_recvs_.end()) {
        const std::uint32_t rvid = wit->second;
        waiting_p2p_recvs_.erase(wit);
        auto uit = unresolved_.find(rvid);
        if (uit != unresolved_.end()) {
          ++res_.num_edges;
          if (opts_.collective_sync == CollectiveSync::kBarrier) {
            window_edges_.push_back({wn.vid, rvid, static_cast<std::uint8_t>(EdgeKind::kP2P)});
          }
          UnresolvedNode& recv_un = uit->second;
          recv_un.waiting_p2p = false;
          relax_from_pred(recv_un, wn.dp.ef_lb, wn.dp.ef_ub, wn.dp.depth, wn.vid, wn.dp.cp_compute, wn.dp.cp_comm,
                          wn.dp.cp_comm_bytes);
          if (recv_un.missing_deps.empty()) {
            UnresolvedNode ready = std::move(recv_un);
            unresolved_.erase(uit);
            finalize_node(ready);
          }
        }
      } else {
        pending_sends_[ck] = {wn.vid, wn.dp.ef_lb, wn.dp.ef_ub, wn.dp.depth, wn.dp.cp_compute, wn.dp.cp_comm,
                              wn.dp.cp_comm_bytes};
      }
    }

    // Wake any intra-rank children waiting on this node_id (unless deferred by a barrier cut).
    if (!defer_wake) wake_intra_rank_waiters(wn);
    return wn;
  }

  void wake_intra_rank_waiters(const WindowNode& wn) {
    auto& rank_wait = waiting_on_parent_[wn.rank];
    auto pit = rank_wait.find(wn.node_id);
    if (pit == rank_wait.end()) return;
    std::vector<std::uint32_t> waiters = std::move(pit->second);
    rank_wait.erase(pit);
    for (std::uint32_t cvid : waiters) {
      auto uit = unresolved_.find(cvid);
      if (uit == unresolved_.end()) continue;
      UnresolvedNode& child = uit->second;
      for (auto mit = child.missing_deps.begin(); mit != child.missing_deps.end();) {
        if (mit->first == wn.node_id) {
          const bool strict_ireq = (opts_.goal_semantics == GoalSemantics::kStrict &&
                                    mit->second == static_cast<std::uint8_t>(EdgeKind::kIrequires));
          ++res_.num_edges;
          if (opts_.collective_sync == CollectiveSync::kBarrier) {
            window_edges_.push_back({wn.vid, cvid, mit->second});
          }
          relax_from_pred(child, strict_ireq ? wn.dp.es_lb : wn.dp.ef_lb, strict_ireq ? wn.dp.es_ub : wn.dp.ef_ub,
                          wn.dp.depth, wn.vid, wn.dp.cp_compute, wn.dp.cp_comm, wn.dp.cp_comm_bytes);
          mit = child.missing_deps.erase(mit);
        } else {
          ++mit;
        }
      }
      if (child.missing_deps.empty() && !child.waiting_p2p) {
        UnresolvedNode ready = std::move(child);
        unresolved_.erase(uit);
        finalize_node(ready);
      }
    }
  }

  void apply_late_edge(std::uint32_t rank, std::uint64_t src_id, std::uint64_t dst_id, std::uint8_t kind) {
    RankWindow& rw = rank_windows_[rank];
    auto dit = rw.nodes.find(dst_id);
    if (dit == rw.nodes.end()) return;
    auto sit = rw.nodes.find(src_id);
    if (sit == rw.nodes.end()) {
      double lb = 0.0, ub = 0.0;
      if (rw.evicted.lookup_evicted(src_id, lb, ub)) ++res_.provenance.evicted_dep_hits;
      return;
    }
    ++res_.num_edges;
    WindowNode& pred = sit->second;
    WindowNode& dst = dit->second;
    pred.has_local_child = true;
    const bool strict_ireq =
        (opts_.goal_semantics == GoalSemantics::kStrict && kind == static_cast<std::uint8_t>(EdgeKind::kIrequires));
    const double p_lb = strict_ireq ? pred.dp.es_lb : pred.dp.ef_lb;
    const double p_ub = strict_ireq ? pred.dp.es_ub : pred.dp.ef_ub;
    if (p_ub > dst.dp.es_ub) {
      dst.dp.es_ub = p_ub;
      dst.dp.ef_ub = p_ub + dst.weight_ns;
      if (dst.dp.ef_ub > global_makespan_ub_) global_makespan_ub_ = dst.dp.ef_ub;
    }
    if (beats_parent(p_lb, pred.dp.depth, pred.vid, dst.dp.es_lb, dst.dp.depth, dst.dp.parent_vid)) {
      dst.dp.es_lb = p_lb;
      dst.dp.ef_lb = p_lb + dst.weight_ns;
      dst.dp.depth = pred.dp.depth + 1;
      dst.dp.parent_vid = pred.vid;
      dst.dp.cp_compute = pred.dp.cp_compute + (is_compute_class(dst.op_class) ? dst.weight_ns : 0.0);
      dst.dp.cp_comm = pred.dp.cp_comm + (is_comm_class(dst.op_class) ? dst.weight_ns : 0.0);
      dst.dp.cp_comm_bytes =
          pred.dp.cp_comm_bytes + (is_comm_class(dst.op_class) ? static_cast<std::int64_t>(dst.size_bytes) : 0);
      if (dst.dp.depth > global_max_depth_) global_max_depth_ = dst.dp.depth;
      if (dst.dp.ef_lb > global_makespan_lb_) {
        global_makespan_lb_ = dst.dp.ef_lb;
        global_sink_vid_ = dst.vid;
        global_cp_compute_ = dst.dp.cp_compute;
        global_cp_comm_ = dst.dp.cp_comm;
        global_cp_comm_bytes_ = dst.dp.cp_comm_bytes;
      }
    }
  }

  std::uint32_t current_world_size() const {
    if (expected_world_size_ > 0) return expected_world_size_;
    if (opts_.world_size_hint > 0) return opts_.world_size_hint;
    return static_cast<std::uint32_t>(std::max<std::size_t>(1, seen_ranks_.size()));
  }

  std::uint32_t collective_group_size(const std::string& comm_type) const {
    auto it = opts_.coll_group_sizes.find(comm_type);
    if (it != opts_.coll_group_sizes.end()) return it->second;
    auto all = opts_.coll_group_sizes.find("*");
    if (all != opts_.coll_group_sizes.end()) return all->second;
    return current_world_size();
  }

  struct MiniWinNode {
    std::uint32_t vid = 0;
    double weight = 0.0;
    double ef = 0.0;
    std::uint32_t epoch = 0;
    bool is_barrier = false;
    bool slack_counted = false;
  };

  void compute_window_slack_at_barrier(const std::vector<std::uint32_t>& barrier_vids, double window_makespan) {
    if (window_nodes_.empty()) return;
    computed_window_slack_ = true;

    // Build reverse adjacency & reachability from barrier_vids over current window.
    std::unordered_map<std::uint32_t, std::vector<std::pair<std::uint32_t, std::uint8_t>>> preds_map;
    std::unordered_map<std::uint32_t, std::vector<std::pair<std::uint32_t, std::uint8_t>>> succs_map;
    for (const WindowEdge& we : window_edges_) {
      if (!window_nodes_.count(we.u_vid) || !window_nodes_.count(we.v_vid)) continue;
      preds_map[we.v_vid].emplace_back(we.u_vid, we.kind);
      succs_map[we.u_vid].emplace_back(we.v_vid, we.kind);
    }

    // Reverse BFS from barrier_vids to certify whether any dead-end local sinks exist.
    std::unordered_set<std::uint32_t> reaches_barrier;
    std::deque<std::uint32_t> q;
    for (std::uint32_t bv : barrier_vids) {
      reaches_barrier.insert(bv);
      q.push_back(bv);
    }
    while (!q.empty()) {
      const std::uint32_t v = q.front();
      q.pop_front();
      auto pit = preds_map.find(v);
      if (pit == preds_map.end()) continue;
      for (const auto& [u, _] : pit->second) {
        if (reaches_barrier.insert(u).second) q.push_back(u);
      }
    }

    // Reverse topological pass (sorted by descending ef then descending vid).
    std::vector<std::uint32_t> order;
    order.reserve(window_nodes_.size());
    for (const auto& [vid, _] : window_nodes_) order.push_back(vid);
    std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
      const auto& na = window_nodes_[a];
      const auto& nb = window_nodes_[b];
      if (na.ef != nb.ef) return na.ef > nb.ef;
      return a > b;
    });

    std::unordered_map<std::uint32_t, double> lf;
    for (std::uint32_t vid : order) lf[vid] = window_makespan;
    for (std::uint32_t u : order) {
      const auto& nu = window_nodes_[u];
      auto sit = succs_map.find(u);
      if (sit != succs_map.end() && !sit->second.empty()) {
        double min_lf = window_makespan;
        for (const auto& [v, kind] : sit->second) {
          const auto& nv = window_nodes_[v];
          const bool strict_ireq =
              (opts_.goal_semantics == GoalSemantics::kStrict && kind == static_cast<std::uint8_t>(EdgeKind::kIrequires));
          const double bound = strict_ireq ? (lf[v] - nv.weight + nu.weight) : (lf[v] - nv.weight);
          if (bound < min_lf) min_lf = bound;
        }
        lf[u] = min_lf;
      }
      if (!reaches_barrier.count(u)) ++res_.provenance.dead_end_sinks;
    }

    const double tol = std::max(1e-6, window_makespan * 1e-9);
    for (auto& [vid, node] : window_nodes_) {
      if (node.slack_counted) continue;
      const double slack = std::max(0.0, lf[vid] - node.ef);
      window_slack_sum_.add(slack);
      if (slack <= tol) ++window_zero_slack_;
      ++window_slack_nodes_;
      node.slack_counted = true;
    }

    // Keep only the barrier nodes so the next window can see edges leaving B_out.
    std::unordered_map<std::uint32_t, MiniWinNode> kept;
    for (std::uint32_t bv : barrier_vids) {
      auto it = window_nodes_.find(bv);
      if (it != window_nodes_.end()) kept.emplace(bv, it->second);
    }
    window_nodes_.swap(kept);
    window_edges_.clear();
  }

  void compute_final_window_slack() {
    // Nodes after the last barrier (excluding already-counted barrier nodes).
    std::vector<std::uint32_t> fresh;
    for (const auto& [vid, n] : window_nodes_) {
      if (!n.slack_counted) fresh.push_back(vid);
    }
    if (fresh.empty()) return;
    const double mk = global_makespan_lb_;
    std::unordered_map<std::uint32_t, std::vector<std::pair<std::uint32_t, std::uint8_t>>> succs_map;
    for (const WindowEdge& we : window_edges_) {
      if (window_nodes_.count(we.u_vid) && window_nodes_.count(we.v_vid)) {
        succs_map[we.u_vid].emplace_back(we.v_vid, we.kind);
      }
    }
    std::sort(fresh.begin(), fresh.end(), [&](std::uint32_t a, std::uint32_t b) {
      return window_nodes_[a].ef > window_nodes_[b].ef;
    });
    std::unordered_map<std::uint32_t, double> lf;
    for (std::uint32_t u : fresh) lf[u] = mk;
    for (std::uint32_t u : fresh) {
      auto sit = succs_map.find(u);
      if (sit == succs_map.end()) continue;
      double min_lf = mk;
      for (const auto& [v, kind] : sit->second) {
        if (!lf.count(v)) continue;
        const bool strict_ireq =
            (opts_.goal_semantics == GoalSemantics::kStrict && kind == static_cast<std::uint8_t>(EdgeKind::kIrequires));
        const double bound =
            strict_ireq ? (lf[v] - window_nodes_[v].weight + window_nodes_[u].weight) : (lf[v] - window_nodes_[v].weight);
        if (bound < min_lf) min_lf = bound;
      }
      lf[u] = min_lf;
    }
    const double tol = std::max(1e-6, mk * 1e-9);
    for (std::uint32_t u : fresh) {
      const double slack = std::max(0.0, lf[u] - window_nodes_[u].ef);
      window_slack_sum_.add(slack);
      if (slack <= tol) ++window_zero_slack_;
      ++window_slack_nodes_;
    }
  }

  struct P2PSendState {
    std::uint32_t vid = 0;
    double ef_lb = 0.0;
    double ef_ub = 0.0;
    std::int32_t depth = 1;
    double cp_compute = 0.0;
    double cp_comm = 0.0;
    std::int64_t cp_comm_bytes = 0;
  };

  DagSolveOptions opts_;
  DagStreamResult res_;
  std::unique_ptr<ScratchDir> scratch_;
  std::unique_ptr<DiskParentLog> parent_log_;
  std::uint32_t expected_world_size_ = 0;
  std::uint32_t next_vid_ = 0;
  std::uint32_t current_window_epoch_ = 0;

  std::unordered_set<std::uint32_t> seen_ranks_;
  std::unordered_map<std::uint32_t, RankWindow> rank_windows_;
  std::unordered_map<ChannelKey, std::uint32_t, ChannelKeyHash> send_seq_, recv_seq_;
  std::unordered_map<ChannelSeqKey, P2PSendState, ChannelSeqKeyHash> pending_sends_;
  std::unordered_map<ChannelSeqKey, std::uint32_t, ChannelSeqKeyHash> waiting_p2p_recvs_;

  std::unordered_map<std::uint32_t, UnresolvedNode> unresolved_;
  std::deque<std::uint32_t> unresolved_order_;
  std::unordered_map<std::uint32_t, std::unordered_map<std::uint64_t, std::vector<std::uint32_t>>> waiting_on_parent_;

  std::unordered_map<std::uint32_t, std::unordered_map<std::string, std::uint32_t>> coll_seq_;
  std::unordered_map<CollBarrierKey, WindowBarrierState, CollBarrierKeyHash> active_barriers_;
  std::unordered_map<std::uint32_t, UnresolvedNode> barrier_parked_;

  std::unordered_map<std::uint32_t, MiniWinNode> window_nodes_;
  std::vector<WindowEdge> window_edges_;
  bool computed_window_slack_ = false;
  NeumaierSum window_slack_sum_;
  std::uint64_t window_zero_slack_ = 0;
  std::uint64_t window_slack_nodes_ = 0;

  double global_makespan_lb_ = 0.0;
  double global_makespan_ub_ = 0.0;
  std::int32_t global_max_depth_ = 0;
  std::uint32_t global_sink_vid_ = kNoParent;
  double global_cp_compute_ = 0.0;
  double global_cp_comm_ = 0.0;
  std::int64_t global_cp_comm_bytes_ = 0;
};

class FrontierEngine final : public GraphEngine {
 public:
  std::string_view name() const override { return "frontier"; }

  DagStreamResult solve_dag(GraphSource& source, const DagSolveOptions& opts) override {
    StreamFrontierSolver solver(opts);
    GraphChunk chunk;
    const std::size_t step = opts.chunk_nodes > 0 ? opts.chunk_nodes : 65536;
    while (source.next_chunk(chunk, step)) solver.feed_chunk(chunk);
    return solver.finish();
  }
};

}  // namespace

std::unique_ptr<GraphEngine> make_frontier_engine() { return std::make_unique<FrontierEngine>(); }

// -----------------------------------------------------------------------------
// Engine B1: Exact O(frontier) multi-rank Kahn directly over mmapped GOAL .bin
// -----------------------------------------------------------------------------
DagStreamResult solve_goal_bin_frontier(const std::string& bin_path, const DagSolveOptions& opts) {
  GoalBinFile bin(bin_path);
  DagStreamResult res;
  res.provenance.engine = "frontier";
  res.provenance.slack_scope = "unavailable";
  res.provenance.goal_semantics = opts.goal_semantics == GoalSemantics::kStrict ? "strict" : "cta_legacy";
  res.provenance.collective_sync = "none";

  const std::uint32_t nranks = bin.num_ranks();
  std::vector<std::uint32_t> rank_base(nranks + 1, 0);
  std::uint64_t total_nodes = 0;
  for (std::uint32_t r = 0; r < nranks; ++r) {
    rank_base[r] = static_cast<std::uint32_t>(total_nodes);
    total_nodes += bin.rank_view(r).num_nodes;
  }
  rank_base[nranks] = static_cast<std::uint32_t>(total_nodes);
  if (total_nodes > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("GOAL binary exceeds 2^32 vertices");
  res.num_nodes = total_nodes;
  if (total_nodes == 0) {
    res.provenance.exact = true;
    res.provenance.slack_scope = "global_exact";
    return res;
  }

  // Row-order P2P channel indexing (per-rank sequential scan guarantees k-th
  // send in file order pairs with k-th recv in file order, never pop order).
  std::unordered_map<ChannelKey, std::vector<std::uint32_t>, ChannelKeyHash> chan_sends, chan_recvs;
  for (std::uint32_t r = 0; r < nranks; ++r) {
    const GoalBinRankView& rv = bin.rank_view(r);
    for (std::uint32_t i = 0; i < rv.num_nodes; ++i) {
      const GoalBinNode nr = rv.node(i);
      const std::uint32_t vid = rank_base[r] + i;
      if (nr.op_type == kGoalOpSend) {
        chan_sends[ChannelKey{static_cast<std::int32_t>(r), static_cast<std::int32_t>(nr.peer), nr.tag}].push_back(vid);
      } else if (nr.op_type == kGoalOpRecv) {
        chan_recvs[ChannelKey{static_cast<std::int32_t>(nr.peer), static_cast<std::int32_t>(r), nr.tag}].push_back(vid);
      }
    }
  }
  std::unordered_map<std::uint32_t, std::uint32_t> send_to_recv;
  std::unordered_set<std::uint32_t> matched_recvs;
  for (auto& [ch, sv] : chan_sends) {
    auto rit = chan_recvs.find(ch);
    const std::size_t rcnt = rit == chan_recvs.end() ? 0 : rit->second.size();
    const std::size_t pairs = std::min(sv.size(), rcnt);
    res.provenance.unmatched_p2p_sends += (sv.size() - pairs);
    for (std::size_t k = 0; k < pairs; ++k) {
      const std::uint32_t s = sv[k];
      const std::uint32_t d = rit->second[k];
      send_to_recv[s] = d;
      matched_recvs.insert(d);
    }
  }
  for (const auto& [ch, rv] : chan_recvs) {
    auto sit = chan_sends.find(ch);
    const std::size_t scnt = sit == chan_sends.end() ? 0 : sit->second.size();
    if (rv.size() > scnt) res.provenance.unmatched_p2p_recvs += (rv.size() - scnt);
  }
  chan_sends.clear();
  chan_recvs.clear();

  // Active-frontier structures: remaining_in_deg is initialized lazily from
  // GoalBinNode::deps_cnt + (1 if matched_recv else 0) when a node is first
  // touched; frontier DP states are erased as soon as a node finishes firing!
  std::unordered_map<std::uint32_t, std::uint32_t> rem_in_deg;
  std::unordered_map<std::uint32_t, FrontierDP> active_dp;
  std::deque<std::uint32_t> ready;

  std::unique_ptr<ScratchDir> scratch;
  std::unique_ptr<DiskParentLog> parent_log;
  if (opts.extract_critical_path_ops) {
    scratch = std::make_unique<ScratchDir>(opts.scratch_dir);
    parent_log = std::make_unique<DiskParentLog>(scratch->file("parent.bin"));
  }

  auto vid_rank_local = [&](std::uint32_t vid) -> std::pair<std::uint32_t, std::uint32_t> {
    auto it = std::upper_bound(rank_base.begin(), rank_base.end(), vid);
    const std::uint32_t r = static_cast<std::uint32_t>((it - rank_base.begin()) - 1);
    return {r, vid - rank_base[r]};
  };

  auto weight_and_class = [&](const GoalBinNode& nr) -> std::pair<double, std::uint8_t> {
    if (nr.op_type == kGoalOpSend) {
      return {node_weight_ns(0, static_cast<std::uint8_t>(OpClass::kCommSend), nr.size),
              static_cast<std::uint8_t>(OpClass::kCommSend)};
    }
    if (nr.op_type == kGoalOpRecv) {
      return {node_weight_ns(0, static_cast<std::uint8_t>(OpClass::kCommRecv), nr.size),
              static_cast<std::uint8_t>(OpClass::kCommRecv)};
    }
    return {static_cast<double>(nr.size), static_cast<std::uint8_t>(OpClass::kCompute)};
  };

  // Seed ready queue with true global roots: deps_cnt == 0 AND not a matched recv.
  for (std::uint32_t r = 0; r < nranks; ++r) {
    const GoalBinRankView& rv = bin.rank_view(r);
    for (std::uint32_t i = 0; i < rv.num_roots; ++i) {
      const std::uint32_t local_id = rv.root(i);
      if (local_id >= rv.num_nodes) continue;
      const std::uint32_t vid = rank_base[r] + local_id;
      if (matched_recvs.count(vid)) continue;  // waits for its matching P2P send!
      const GoalBinNode nr = rv.node(local_id);
      const auto [w, oc] = weight_and_class(nr);
      FrontierDP dp;
      dp.es_lb = dp.es_ub = 0.0;
      dp.ef_lb = dp.ef_ub = w;
      dp.depth = 1;
      dp.cp_compute = is_compute_class(oc) ? w : 0.0;
      dp.cp_comm = is_comm_class(oc) ? w : 0.0;
      dp.cp_comm_bytes = is_comm_class(oc) ? static_cast<std::int64_t>(nr.size) : 0;
      active_dp[vid] = dp;
      ready.push_back(vid);
    }
  }

  double makespan = 0.0;
  std::int32_t max_depth = 0;
  std::uint32_t sink_vid = kNoParent;
  double best_cp_comp = 0.0, best_cp_comm = 0.0;
  std::int64_t best_cp_bytes = 0;
  std::uint64_t processed = 0;
  std::vector<bool> fired(total_nodes, false);

  auto relax_edge = [&](std::uint32_t u_vid, const FrontierDP& u_dp, std::uint32_t v_vid, bool strict_ireq,
                        std::uint32_t raw_mult) {
    const auto [vr, vl] = vid_rank_local(v_vid);
    const GoalBinNode vnr = bin.rank_view(vr).node(vl);
    const auto [vw, voc] = weight_and_class(vnr);

    auto dit = rem_in_deg.find(v_vid);
    if (dit == rem_in_deg.end()) {
      const std::uint32_t init_deg = vnr.deps_cnt + (matched_recvs.count(v_vid) ? 1u : 0u);
      dit = rem_in_deg.emplace(v_vid, init_deg).first;
    }
    auto [dp_it, inserted] = active_dp.try_emplace(v_vid);
    FrontierDP& v_dp = dp_it->second;
    if (inserted) {
      v_dp.ef_lb = v_dp.ef_ub = vw;
      v_dp.cp_compute = is_compute_class(voc) ? vw : 0.0;
      v_dp.cp_comm = is_comm_class(voc) ? vw : 0.0;
      v_dp.cp_comm_bytes = is_comm_class(voc) ? static_cast<std::int64_t>(vnr.size) : 0;
    }

    const double u_val = strict_ireq ? u_dp.es_lb : u_dp.ef_lb;
    if (beats_parent(u_val, u_dp.depth, u_vid, v_dp.es_lb, v_dp.depth, v_dp.parent_vid)) {
      v_dp.es_lb = v_dp.es_ub = u_val;
      v_dp.ef_lb = v_dp.ef_ub = u_val + vw;
      v_dp.depth = u_dp.depth + 1;
      v_dp.parent_vid = u_vid;
      v_dp.cp_compute = u_dp.cp_compute + (is_compute_class(voc) ? vw : 0.0);
      v_dp.cp_comm = u_dp.cp_comm + (is_comm_class(voc) ? vw : 0.0);
      v_dp.cp_comm_bytes = u_dp.cp_comm_bytes + (is_comm_class(voc) ? static_cast<std::int64_t>(vnr.size) : 0);
    }

    dit->second = (dit->second > raw_mult) ? (dit->second - raw_mult) : 0u;
    if (dit->second == 0) {
      rem_in_deg.erase(dit);
      ready.push_back(v_vid);
    }
  };

  while (!ready.empty()) {
    const std::uint32_t u = ready.front();
    ready.pop_front();
    if (fired[u]) continue;
    fired[u] = true;
    ++processed;

    const FrontierDP u_dp = active_dp[u];
    active_dp.erase(u);  // O(frontier) eviction immediately on fire!

    const auto [ur, ul] = vid_rank_local(u);
    const GoalBinRankView& rv = bin.rank_view(ur);
    const GoalBinNode nr = rv.node(ul);
    const auto [uw, uoc] = weight_and_class(nr);

    if (u_dp.depth > max_depth) max_depth = u_dp.depth;
    if (u_dp.ef_lb > makespan || (u_dp.ef_lb == makespan && (sink_vid == kNoParent || u < sink_vid))) {
      makespan = u_dp.ef_lb;
      sink_vid = u;
      best_cp_comp = u_dp.cp_compute;
      best_cp_comm = u_dp.cp_comm;
      best_cp_bytes = u_dp.cp_comm_bytes;
    }

    if (parent_log) {
      ParentLogRecord rec;
      rec.parent_vid = u_dp.parent_vid;
      rec.rank = ur;
      rec.node_id = ul;
      rec.size_bytes = is_comm_class(uoc) ? nr.size : 0;
      rec.weight_ns = uw;
      rec.es_ns = u_dp.es_lb;
      rec.ef_ns = u_dp.ef_lb;
      rec.op_class = uoc;
      copy_name_trunc(rec.name, uoc == static_cast<std::uint8_t>(OpClass::kCommSend) ? "comm_send"
                                : uoc == static_cast<std::uint8_t>(OpClass::kCommRecv) ? "comm_recv"
                                                                                       : "compute");
      parent_log->write(u, rec);
    }

    // Deduplicate outgoing intra-rank edges while preserving raw multiplicity
    // so `deps_cnt` reaches 0 accurately even if txt2bin emitted duplicates.
    struct OutTarget {
      std::uint32_t raw_mult = 0;
      bool has_requires = false;
    };
    std::unordered_map<std::uint32_t, OutTarget> out_targets;
    if (nr.num_deps > 0) {
      for (std::uint32_t k = 0; k < nr.num_deps; ++k) {
        const std::uint32_t dst_local = rv.appendix(static_cast<std::size_t>(nr.deps_apdx) + k);
        if (dst_local >= rv.num_nodes) continue;
        OutTarget& ot = out_targets[rank_base[ur] + dst_local];
        ++ot.raw_mult;
        ot.has_requires = true;
      }
    }
    if (nr.num_startdeps > 0) {
      for (std::uint32_t k = 0; k < nr.num_startdeps; ++k) {
        const std::uint32_t dst_local = rv.appendix(static_cast<std::size_t>(nr.startdeps_apdx) + k);
        if (dst_local >= rv.num_nodes) continue;
        OutTarget& ot = out_targets[rank_base[ur] + dst_local];
        ++ot.raw_mult;
      }
    }
    for (const auto& [v_vid, ot] : out_targets) {
      ++res.num_edges;
      const bool strict_ireq = (opts.goal_semantics == GoalSemantics::kStrict && !ot.has_requires);
      relax_edge(u, u_dp, v_vid, strict_ireq, ot.raw_mult);
    }

    // Inter-rank P2P edge (only if this send has a row-order matched recv).
    auto sit = send_to_recv.find(u);
    if (sit != send_to_recv.end()) {
      ++res.num_edges;
      relax_edge(u, u_dp, sit->second, false, 1);
    }
  }

  res.unordered = total_nodes - processed;
  res.makespan = makespan;
  res.depth = max_depth;
  res.provenance.makespan_lb = makespan;
  res.provenance.makespan_ub = makespan;
  if (res.unordered > 0) {
    res.provenance.has_cycle = true;
    res.provenance.unordered_nodes = res.unordered;
    res.provenance.exact = false;
    // Extract a concrete sample cycle witness over the unvisited subgraph.
    std::unordered_map<std::uint32_t, std::uint32_t> unvisited_pred;
    for (std::uint32_t u = 0; u < total_nodes; ++u) {
      if (fired[u]) continue;
      const auto [ur, ul] = vid_rank_local(u);
      const GoalBinRankView& rv = bin.rank_view(ur);
      const GoalBinNode nr = rv.node(ul);
      for (std::uint32_t k = 0; k < nr.num_deps; ++k) {
        const std::uint32_t dl = rv.appendix(static_cast<std::size_t>(nr.deps_apdx) + k);
        if (dl < rv.num_nodes && !fired[rank_base[ur] + dl]) unvisited_pred.try_emplace(rank_base[ur] + dl, u);
      }
      for (std::uint32_t k = 0; k < nr.num_startdeps; ++k) {
        const std::uint32_t dl = rv.appendix(static_cast<std::size_t>(nr.startdeps_apdx) + k);
        if (dl < rv.num_nodes && !fired[rank_base[ur] + dl]) unvisited_pred.try_emplace(rank_base[ur] + dl, u);
      }
      auto sit = send_to_recv.find(u);
      if (sit != send_to_recv.end() && !fired[sit->second]) unvisited_pred.try_emplace(sit->second, u);
    }
    if (!unvisited_pred.empty()) {
      std::uint32_t cur = unvisited_pred.begin()->first;
      std::unordered_map<std::uint32_t, std::size_t> seen_at;
      std::vector<std::uint32_t> path;
      while (true) {
        auto it = seen_at.find(cur);
        if (it != seen_at.end()) {
          for (std::size_t idx = path.size(); idx-- > it->second;) {
            const auto [wr, wl] = vid_rank_local(path[idx]);
            res.provenance.cycle_witness_rank.push_back(wr);
            res.provenance.cycle_witness_node_id.push_back(wl);
          }
          break;
        }
        seen_at[cur] = path.size();
        path.push_back(cur);
        auto pit = unvisited_pred.find(cur);
        if (pit == unvisited_pred.end()) break;
        cur = pit->second;
      }
    }
    if (opts.fail_on_cycle) throw std::runtime_error("cycle detected in GOAL binary graph");
  } else {
    res.provenance.exact = true;
  }

  if (parent_log && sink_vid != kNoParent) {
    fill_cp_op_chain_from_log(*parent_log, sink_vid, total_nodes, res);
  } else {
    res.critical_compute = best_cp_comp;
    res.critical_comm = best_cp_comm;
    res.critical_comm_bytes = best_cp_bytes;
  }
  return res;
}

// -----------------------------------------------------------------------------
// Multi-rank lockstep Chakra .et frontier reader (FD-multiplexed round-robin)
// -----------------------------------------------------------------------------
DagStreamResult solve_chakra_et_frontier(const std::vector<std::string>& paths,
                                         const std::vector<std::uint32_t>& ranks, const DagSolveOptions& opts) {
  if (paths.size() != ranks.size()) throw std::invalid_argument("paths and ranks must have the same length");
  StreamFrontierSolver solver(opts);
  solver.set_expected_world_size(static_cast<std::uint32_t>(paths.size()));

  // Open rank streams in FD-bounded waves if paths.size() > safe_fd_limit().
  const std::size_t fd_cap = safe_fd_limit();
  const std::size_t step_nodes = std::max<std::size_t>(1, std::min<std::size_t>(opts.window_size / 8, 2048));

  if (paths.size() <= fd_cap) {
    std::vector<std::unique_ptr<GraphSource>> sources;
    sources.reserve(paths.size());
    for (std::size_t i = 0; i < paths.size(); ++i) {
      sources.push_back(open_chakra_et_stream({paths[i]}, {ranks[i]}));
    }
    std::vector<bool> done(paths.size(), false);
    std::size_t remaining = paths.size();
    GraphChunk chunk;
    while (remaining > 0) {
      for (std::size_t i = 0; i < paths.size(); ++i) {
        if (done[i]) continue;
        if (sources[i]->next_chunk(chunk, step_nodes)) {
          solver.feed_chunk(chunk);
        } else {
          done[i] = true;
          --remaining;
        }
      }
    }
  } else {
    // Fallback for huge rank counts (> RLIMIT_NOFILE / 2): stream sequentially via ChakraEtSource.
    auto src = open_chakra_et_stream(paths, ranks);
    GraphChunk chunk;
    while (src->next_chunk(chunk, opts.chunk_nodes)) solver.feed_chunk(chunk);
  }
  return solver.finish();
}

}  // namespace cta::streaming::engines
