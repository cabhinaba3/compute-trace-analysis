#include "synth/training.hpp"

#include <google/protobuf/util/delimited_message_util.h>

#include <algorithm>
#include <atomic>
#include <exception>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

#include "util/cpus.hpp"
#include "util/format.hpp"
#include "util/scoped_join.hpp"
#include "util/sink.hpp"
#include "et_def.pb.h"
#include "synth/goal_format.hpp"

namespace cta::synth {
namespace {

// Dependencies of the next node: none (-1) or one earlier node.
struct After {
  int v[1];
  int n;
  explicit After(int x) : n(x >= 0 ? 1 : 0) { v[0] = x; }
  operator std::span<const int>() const { return {v, static_cast<std::size_t>(n)}; }
};

enum Kind : int { kTpF, kTpB, kMoeF1, kMoeF2, kMoeB1, kMoeB2, kPpF, kPpB, kDp };
constexpr std::uint64_t kTagLimit = 1ull << 31;

// Tags of the phases of one ring all-reduce are consecutive, so each collective
// gets a span of 2 * (largest group size) tags (a ring has 2 * (g - 1) phases).
std::uint64_t phase_span(const TrainingSpec& s) {
  return 2ull * static_cast<std::uint64_t>(std::max({s.dp, s.tp, s.ep_size}));
}

std::uint64_t tag_of(const TrainingSpec& s, int step, int mb, int layer, Kind kind) {
  const std::uint64_t slot =
      (static_cast<std::uint64_t>(step) * s.microbatches + mb) * (s.layers_per_stage + 1) + layer;
  return (slot * 16 + kind) * phase_span(s);
}

// GOAL has only calc/send/recv, so collectives become ring all-reduce phases and
// a pairwise all-to-all. Derived supplies calc, send and recv (each returns the
// new node's id in its rank).
template <typename Derived>
struct GoalCollectives {
  int allreduce(const std::vector<int>& group, int idx, std::int64_t bytes, std::uint64_t tag,
                std::span<const int> dep) {
    auto& d = static_cast<Derived&>(*this);
    const int g = static_cast<int>(group.size());
    if (g <= 1) return dep.empty() ? -1 : dep[0];
    const std::int64_t chunk = bytes / g;
    const int next = group[(idx + 1) % g], prev_rank = group[(idx + g - 1) % g];
    int previous = dep.empty() ? -1 : dep[0];
    for (int phase = 0; phase < 2 * (g - 1); ++phase) {
      const std::uint64_t t = tag + static_cast<std::uint64_t>(phase);
      d.send(next, chunk, t, After(previous));
      previous = d.recv(prev_rank, chunk, t, After(previous));
    }
    return previous;
  }
  int alltoall(const std::vector<int>& group, int idx, std::int64_t bytes, std::uint64_t tag,
               std::span<const int> dep) {
    auto& d = static_cast<Derived&>(*this);
    const int g = static_cast<int>(group.size());
    if (g <= 1) return dep.empty() ? -1 : dep[0];
    std::vector<int> pending;
    for (int i = 0; i < g; ++i) {
      if (i == idx) continue;
      pending.push_back(d.send(group[i], bytes, tag, dep));
      pending.push_back(d.recv(group[i], bytes, tag, dep));
    }
    return d.calc(0, pending);  // barrier
  }
};

// The schedule of one rank, reported to a builder. Builder methods return the id
// of the node they add; they take the nodes it must wait for.
template <typename B>
void training_rank(const TrainingSpec& s, int rank, B& b) {
  const int t = rank % s.tp, p = (rank / s.tp) % s.pp, d = rank / (s.tp * s.pp);
  auto rank_of = [&](int dd, int pp_, int tt) { return (dd * s.pp + pp_) * s.tp + tt; };
  std::vector<int> tpg, dpg, epg;
  for (int k = 0; k < s.tp; ++k) tpg.push_back(rank_of(d, p, k));
  for (int k = 0; k < s.dp; ++k) dpg.push_back(rank_of(k, p, t));
  const int ep_start = d - d % s.ep_size;
  for (int k = 0; k < s.ep_size; ++k) epg.push_back(rank_of(ep_start + k, p, t));
  const int L = s.layers_per_stage, M = s.microbatches;
  auto is_moe = [&](int layer) { return s.moe_every > 0 && layer % s.moe_every == s.moe_every - 1; };
  auto bucket_of = [&](int layer) { return (L - 1 - layer) * s.dp_buckets / L; };

  int prev = -1;
  for (int step = 0; step < s.steps; ++step) {
    for (int mb = 0; mb < M; ++mb) {  // forward
      if (p > 0) prev = b.recv(rank_of(d, p - 1, t), s.pp_bytes, tag_of(s, step, mb, L, kPpF), After(prev));
      for (int layer = 0; layer < L; ++layer) {
        prev = b.calc(s.layer_fwd_ns, After(prev));
        if (s.tp > 1) prev = b.allreduce(tpg, t, s.tp_bytes, tag_of(s, step, mb, layer, kTpF), After(prev));
        if (is_moe(layer)) {
          prev = b.alltoall(epg, d % s.ep_size, s.moe_bytes, tag_of(s, step, mb, layer, kMoeF1), After(prev));
          prev = b.calc(s.expert_ns, After(prev));
          prev = b.alltoall(epg, d % s.ep_size, s.moe_bytes, tag_of(s, step, mb, layer, kMoeF2), After(prev));
        }
      }
      if (p < s.pp - 1) b.send(rank_of(d, p + 1, t), s.pp_bytes, tag_of(s, step, mb, L, kPpF), After(prev));
    }
    std::vector<int> dp_done;
    for (int mb = M - 1; mb >= 0; --mb) {  // backward, last microbatch first
      if (p < s.pp - 1) prev = b.recv(rank_of(d, p + 1, t), s.pp_bytes, tag_of(s, step, mb, L, kPpB), After(prev));
      for (int layer = L - 1; layer >= 0; --layer) {
        prev = b.calc(s.layer_bwd_ns, After(prev));
        if (s.tp > 1) prev = b.allreduce(tpg, t, s.tp_bytes, tag_of(s, step, mb, layer, kTpB), After(prev));
        if (is_moe(layer)) {
          prev = b.alltoall(epg, d % s.ep_size, s.moe_bytes, tag_of(s, step, mb, layer, kMoeB1), After(prev));
          prev = b.calc(s.expert_ns, After(prev));
          prev = b.alltoall(epg, d % s.ep_size, s.moe_bytes, tag_of(s, step, mb, layer, kMoeB2), After(prev));
        }
        // A gradient bucket is complete after the last microbatch passes its layers.
        if (mb == 0 && s.dp > 1 && (layer == 0 || bucket_of(layer - 1) != bucket_of(layer))) {
          dp_done.push_back(b.allreduce(dpg, d, s.dp_bytes / s.dp_buckets,
                                        tag_of(s, step, 0, bucket_of(layer), kDp), After(prev)));
        }
      }
      if (p > 0) b.send(rank_of(d, p - 1, t), s.pp_bytes, tag_of(s, step, mb, L, kPpB), After(prev));
    }
    std::vector<int> waits = dp_done;
    if (prev >= 0) waits.push_back(prev);
    prev = b.calc(s.optimizer_ns, waits);
  }
}

// ---- GOAL text ------------------------------------------------------------------

struct TextBuilder : GoalCollectives<TextBuilder> {
  Sink& sink;
  std::string line;
  int next = 0;
  explicit TextBuilder(Sink& s) : sink(s) {}
  void deps(int id, std::span<const int> dep) {
    for (int x : dep) {
      line.push_back('l'); append_int(line, id + 1); line.append(" requires l");
      append_int(line, x + 1); line.push_back('\n');
    }
    sink.put(line);
    line.clear();
  }
  int calc(std::int64_t ns, std::span<const int> dep) {
    const int id = next++;
    line.push_back('l'); append_int(line, id + 1); line.append(": calc ");
    append_int(line, ns); line.append(" cpu 0\n");
    deps(id, dep);
    return id;
  }
  int send(int peer, std::int64_t bytes, std::uint64_t tag, std::span<const int> dep) {
    const int id = next++;
    line.push_back('l'); append_int(line, id + 1); line.append(": send ");
    append_int(line, bytes); line.append("b to "); append_int(line, peer);
    line.append(" tag "); append_int(line, static_cast<std::int64_t>(tag));
    line.append(" cpu 0 nic 0\n");
    deps(id, dep);
    return id;
  }
  int recv(int peer, std::int64_t bytes, std::uint64_t tag, std::span<const int> dep) {
    const int id = next++;
    line.push_back('l'); append_int(line, id + 1); line.append(": recv ");
    append_int(line, bytes); line.append("b from "); append_int(line, peer);
    line.append(" tag "); append_int(line, static_cast<std::int64_t>(tag));
    line.append(" cpu 0 nic 0\n");
    deps(id, dep);
    return id;
  }
};

// ---- GOAL binary ----------------------------------------------------------------

struct BinBuilder : GoalCollectives<BinBuilder> {
  RankGraph& g;
  explicit BinBuilder(RankGraph& graph) : g(graph) {}
  int add(NodeRec rec, std::span<const int> dep) {
    const std::uint32_t id = static_cast<std::uint32_t>(g.nodes.size());
    g.nodes.push_back(rec);
    for (int x : dep) g.edges.emplace_back(static_cast<std::uint32_t>(x), id);
    return static_cast<int>(id);
  }
  int calc(std::int64_t ns, std::span<const int> dep) { return add({Op::kCalc, ns, 0, 0}, dep); }
  int send(int peer, std::int64_t bytes, std::uint64_t tag, std::span<const int> dep) {
    return add({Op::kSend, bytes, static_cast<std::uint32_t>(peer), static_cast<std::uint32_t>(tag)}, dep);
  }
  int recv(int peer, std::int64_t bytes, std::uint64_t tag, std::span<const int> dep) {
    return add({Op::kRecv, bytes, static_cast<std::uint32_t>(peer), static_cast<std::uint32_t>(tag)}, dep);
  }
};

// ---- Chakra ---------------------------------------------------------------------

class ChakraBuilder {
 public:
  ChakraBuilder(std::ostream& out, int rank) : out_(out), rank_(rank) {}
  int calc(std::int64_t ns, std::span<const int> dep) {
    auto& n = prep_node("compute", ChakraProtoMsg::COMP_NODE, dep);
    n.set_duration_micros(static_cast<std::uint64_t>(ns / 1000));
    return emit(n);
  }
  int send(int peer, std::int64_t bytes, std::uint64_t tag, std::span<const int> dep) {
    auto& n = prep_node("send", ChakraProtoMsg::COMM_SEND_NODE, dep);
    p2p_attrs(n, rank_, peer, bytes, tag);
    return emit(n);
  }
  int recv(int peer, std::int64_t bytes, std::uint64_t tag, std::span<const int> dep) {
    auto& n = prep_node("recv", ChakraProtoMsg::COMM_RECV_NODE, dep);
    p2p_attrs(n, peer, rank_, bytes, tag);
    return emit(n);
  }
  int allreduce(const std::vector<int>& group, int, std::int64_t bytes, std::uint64_t,
                std::span<const int> dep) {
    return group.size() <= 1 ? last(dep) : collective("all_reduce", 0, bytes, dep, pg_name_of(group));
  }
  int alltoall(const std::vector<int>& group, int, std::int64_t bytes, std::uint64_t,
               std::span<const int> dep) {
    return group.size() <= 1 ? last(dep) : collective("all_to_all", 6, bytes, dep, pg_name_of(group));
  }
  std::uint64_t count() const { return next_; }

 private:
  static int last(std::span<const int> dep) { return dep.empty() ? -1 : dep[0]; }
  // The process group's name, the same on every member: "pg_<first>_<stride>_<count>"
  // (the groups are arithmetic progressions of ranks).
  static std::string pg_name_of(const std::vector<int>& group) {
    const int stride = group.size() > 1 ? group[1] - group[0] : 1;
    return "pg_" + std::to_string(group[0]) + "_" + std::to_string(stride) + "_" + std::to_string(group.size());
  }
  ChakraProtoMsg::Node& prep_node(const char* name, ChakraProtoMsg::NodeType type, std::span<const int> dep) {
    node_.Clear();
    node_.set_id(next_);
    node_.set_name(std::string(name) + "_" + std::to_string(next_));
    node_.set_type(type);
    for (int x : dep) node_.add_data_deps(static_cast<std::uint64_t>(x));
    return node_;
  }
  static void attr(ChakraProtoMsg::Node& n, const char* name, std::int64_t v) {
    auto* a = n.add_attr();
    a->set_name(name);
    a->set_int64_val(v);
  }
  static void p2p_attrs(ChakraProtoMsg::Node& n, int src, int dst, std::int64_t bytes, std::uint64_t tag) {
    attr(n, "comm_src", src);
    attr(n, "comm_dst", dst);
    attr(n, "comm_size", bytes);
    attr(n, "comm_tag", static_cast<std::int64_t>(tag));
  }
  int collective(const char* name, int comm_type, std::int64_t bytes, std::span<const int> dep,
                 const std::string& pg_name) {
    auto& n = prep_node(name, ChakraProtoMsg::COMM_COLL_NODE, dep);
    n.set_duration_micros(static_cast<std::uint64_t>(std::max<std::int64_t>(10, bytes / 1000)));
    attr(n, "comm_type", comm_type);
    attr(n, "comm_size", bytes);
    auto* pg = n.add_attr();
    pg->set_name("pg_name");
    pg->set_string_val(pg_name);
    return emit(n);
  }
  int emit(const ChakraProtoMsg::Node& n) {
    google::protobuf::util::SerializeDelimitedToOstream(n, &out_);
    return static_cast<int>(next_++);
  }
  std::ostream& out_;
  int rank_;
  std::uint64_t next_ = 0;
  ChakraProtoMsg::Node node_;
};


}  // namespace

void TrainingSpec::validate() const {
  auto bad = [](const std::string& m) { throw std::invalid_argument(m); };
  if (dp < 1 || pp < 1 || tp < 1) bad("dp, pp and tp must be at least 1");
  if (microbatches < 1 || layers_per_stage < 1 || steps < 1) bad("microbatches, layers_per_stage and steps must be at least 1");
  if (dp_buckets < 1 || dp_buckets > layers_per_stage) bad("dp_buckets must be between 1 and layers_per_stage");
  if (moe_every < 0 || ep_size < 1 || dp % ep_size != 0) bad("ep_size must divide dp, and moe_every must be >= 0");
  if (static_cast<std::uint64_t>(steps) * microbatches * (layers_per_stage + 1) * 16 * phase_span(*this) >= kTagLimit)
    bad("steps * microbatches * layers_per_stage is too large for 31-bit message tags");
}

std::uint64_t write_goal_training(const std::string& path, const TrainingSpec& spec) {
  spec.validate();
  Sink sink(path);
  sink.put("num_ranks " + std::to_string(spec.ranks()) + "\n");
  std::uint64_t nodes = 0;
  for (int rank = 0; rank < spec.ranks(); ++rank) {
    sink.put("\nrank " + std::to_string(rank) + " {\n");
    TextBuilder b(sink);
    training_rank(spec, rank, b);
    nodes += static_cast<std::uint64_t>(b.next);
    sink.put("}\n");
  }
  sink.close();
  return nodes;
}

std::uint64_t write_goal_training_bin(const std::string& path, const TrainingSpec& spec) {
  spec.validate();
  return goal_bin::write_file(path, spec.ranks(), [&](int rank, RankGraph& g) {
    BinBuilder b(g);
    training_rank(spec, rank, b);
  });
}

std::uint64_t write_chakra_training(const std::string& out_dir, const TrainingSpec& spec,
                                    int threads) {
  spec.validate();
  std::filesystem::create_directories(out_dir);
  const int workers = std::min(spec.ranks(), threads > 0 ? threads : usable_cpus());
  std::atomic<int> next_rank{0};
  std::atomic<std::uint64_t> total{0};
  std::exception_ptr failure;
  std::mutex failure_mutex;
  auto work = [&] {
    try {
      for (int rank = next_rank++; rank < spec.ranks(); rank = next_rank++) {
        const std::string path = out_dir + "/trace.rank" + std::to_string(rank) + ".et";
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot open " + path);
        ChakraProtoMsg::GlobalMetadata meta;
        meta.set_version("0.0.4");
        google::protobuf::util::SerializeDelimitedToOstream(meta, &out);
        ChakraBuilder b(out, rank);
        training_rank(spec, rank, b);
        total += b.count();
        out.close();
        if (!out) throw std::runtime_error("write failed: " + path);
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(failure_mutex);
      if (!failure) failure = std::current_exception();
      next_rank = spec.ranks();  // stop the other workers
    }
  };
  ScopedJoin pool;
  for (int i = 0; i < workers; ++i) pool.emplace(work);
  pool.join_all();
  if (failure) std::rethrow_exception(failure);
  return total;
}

}  // namespace cta::synth
