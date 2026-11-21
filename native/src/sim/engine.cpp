#include "sim/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <initializer_list>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>

#include "sim/flow.hpp"
#include "sim/lowering.hpp"
#include "sim/noise.hpp"
#include "sim/topology.hpp"
#include "util/util.hpp"

namespace cta::sim {

void SimParams::check() const {
  auto bad = [](const std::string& what) { throw std::invalid_argument("simulation parameters: " + what); };
  auto choice = [&](const char* name, const std::string& value, std::initializer_list<const char*> allowed) {
    if (std::none_of(allowed.begin(), allowed.end(), [&](const char* a) { return value == a; }))
      bad(std::string(name) + " cannot be '" + value + "'");
  };
  choice("model", model, {"loggops", "alpha_beta", "flow"});
  choice("collectives.completion", completion, {"all", "outputs"});
  choice("collectives.all_to_all_size", all_to_all_size, {"total", "per_peer"});
  choice("collectives.cost", collective_cost, {"model", "recorded"});
  choice("noise.distribution", noise_distribution, {"fixed", "exponential"});
  for (const std::string& a : algorithms) parse_algorithm(a);
  const LogGOPS in = intra_loggops();
  for (double x : {loggops.L, loggops.o, loggops.g, loggops.G, loggops.O, in.L, in.o, in.g, in.G, in.O, alpha, beta,
                   intra_alpha.value_or(0), intra_beta.value_or(0), reduce_ns_per_byte, noise_frequency,
                   noise_duration})
    if (x < 0) bad("costs, rates and durations must not be negative");
  for (double s : nic_scale)
    if (!(s > 0)) bad("NIC scale factors must be positive");
}

namespace {

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

// An activity holds and waits for resources: an op starting, or, with kMsg
// set, a receiver processing message k.
constexpr u32 kMsg = 1u << 31;
// In a ready cause: the dependency's start rather than its end.
constexpr u32 kStarted = 1u << 31;
constexpr std::size_t kListLimit = 10000;

Time ns(double x) { return std::llround(x); }
double minus_one(u64 bytes) { return bytes > 0 ? static_cast<double>(bytes - 1) : 0.0; }  // LogGP's s - 1

enum class Model : u8 { kLogGOPS, kAlphaBeta, kFlow };
Model model_of(const std::string& name) {
  return name == "flow" ? Model::kFlow : name == "alpha_beta" ? Model::kAlphaBeta : Model::kLogGOPS;
}

enum Ev : u8 { kDone, kFlowDone, kArrive, kFlowStart, kWake, kReady };
struct Event {
  Time t;
  u32 id;
  u8 type;
  bool operator>(const Event& o) const { return std::tie(t, type, id) > std::tie(o.t, o.type, o.id); }
};

enum class ResKind : u8 { kNone, kCpu, kNicSend, kNicRecv };
struct Waiter {
  Time since;
  u32 act;
  auto operator<=>(const Waiter&) const = default;
};
struct Resource {
  Time free_at = 0, hold_start = 0;
  u32 holder = kNone;
  ResKind kind = ResKind::kCpu;
  bool wake = false;
  std::priority_queue<Waiter, std::vector<Waiter>, std::greater<Waiter>> waiters;
};
// Who held the resource an activity waited for last.
struct Wait {
  u32 holder = kNone;
  ResKind kind = ResKind::kNone;
  Time hold_start = 0;
};

// A message, with the times its critical-path segments start and end.
struct Message {
  u32 op = kNone;
  bool rendezvous = false;
  Segment drain = Segment::kTransfer;  // the receiver's per-byte time
  Time cpu_end = kNever;               // the send's CPU time ends
  Time flight = kNever;  // the data leaves (start + o), or alpha-beta's latency ends
  Time ideal = kNever, data_done = kNever;  // flow: last byte out without contention, and in fact
  Time arrive = kNever, proc_start = kNever, proc_o_end = kNever, proc_end = kNever, proc_noise = 0;
  Wait wait;
};

struct MatchKey {
  u32 dst, src;
  u64 tag;
  bool operator==(const MatchKey&) const = default;
};
struct MatchKeyHash {
  std::size_t operator()(const MatchKey& k) const { return mix64(k.tag ^ mix64((u64{k.dst} << 32) | k.src)); }
};
using MatchQueues = std::unordered_map<MatchKey, std::vector<u32>, MatchKeyHash>;

u32 take(MatchQueues& queues, const MatchKey& key) {
  const auto it = queues.find(key);
  if (it == queues.end()) return kNone;
  const u32 v = it->second.front();
  it->second.erase(it->second.begin());
  if (it->second.empty()) queues.erase(it);
  return v;
}

class Pairs {
 public:
  explicit Pairs(const SimParams& p) : p_(p), intra_(p.intra_loggops()), model_(model_of(p.model)) {}
  Model model() const { return model_; }
  bool same_node(u32 a, u32 b) const { return p_.ranks_per_node > 1 && a / p_.ranks_per_node == b / p_.ranks_per_node; }
  const LogGOPS& loggops(u32 a, u32 b) const { return p_.has_intra_loggops() && same_node(a, b) ? intra_ : p_.loggops; }
  double scale(u32 a, u32 b) const {
    auto of = [&](u32 r) { return r < p_.nic_scale.size() ? p_.nic_scale[r] : 1.0; };
    return std::min(of(a), of(b));
  }
  double G(u32 a, u32 b) const { return loggops(a, b).G / scale(a, b); }
  double alpha(u32 a, u32 b) const { return same_node(a, b) && p_.intra_alpha ? *p_.intra_alpha : p_.alpha; }
  double beta(u32 a, u32 b) const {
    return (same_node(a, b) && p_.intra_beta ? *p_.intra_beta : p_.beta) / scale(a, b);
  }
  FlowNetwork* flow = nullptr;

 private:
  const SimParams& p_;
  LogGOPS intra_;
  Model model_;
};

class ModelCost final : public CostModel {
 public:
  ModelCost(const SimParams& p, const Pairs& pairs) : p_(p), pairs_(pairs) {}
  Time send_busy(u32 src, u32 dst, u64 bytes) const override {
    if (pairs_.model() == Model::kAlphaBeta) return 0;
    const LogGOPS& l = pairs_.loggops(src, dst);
    return ns(l.o + minus_one(bytes) * l.O);
  }
  Time deliver(u32 src, u32 dst, u64 bytes) const override {
    const double b1 = minus_one(bytes);
    const LogGOPS& l = pairs_.loggops(src, dst);
    switch (pairs_.model()) {
      case Model::kAlphaBeta:
        return ns(pairs_.alpha(src, dst) + static_cast<double>(bytes) * pairs_.beta(src, dst));
      case Model::kLogGOPS:
        return ns(l.o) + ns(l.L) + ns(l.o + std::max(b1 * l.O, b1 * pairs_.G(src, dst)));
      case Model::kFlow:
        return ns(l.o) + transfer_time(static_cast<double>(bytes), pairs_.flow->bottleneck(src, dst)) +
               pairs_.flow->latency(src, dst) + ns(l.o + b1 * l.O);
    }
    return 0;
  }
  Time reduce(u64 bytes) const override { return ns(static_cast<double>(bytes) * p_.reduce_ns_per_byte); }

 private:
  const SimParams& p_;
  const Pairs& pairs_;
};

std::vector<Time> evaluate(CollSpec spec, std::span<const u32> members, const SimParams& p, const CostModel& cost,
                           bool* fell_back) {
  const u32 node_size = node_size_of(members, p.ranks_per_node);
  spec.size = static_cast<u32>(members.size());
  spec.algorithm = resolve_algorithm(spec.type, algorithms_of(p)[static_cast<std::size_t>(spec.type)], spec.size,
                                     node_size, fell_back);
  spec.node_size = std::max<u32>(1, node_size);
  spec.chunks = std::max<u32>(1, p.chunks);
  spec.reduce = p.reduce_ns_per_byte > 0;
  spec.all_to_all_per_peer = p.all_to_all_size == "per_peer";
  return evaluate_collective(spec, members, cost, p.completion == "outputs");
}

class Engine {
 public:
  Engine(const GraphView& g, const Groups& groups, const SimParams& p);
  SimColumns run();

 private:
  void push(Ev type, u32 id, Time t) { events_.push({t, id, type}); }
  u32 cpu_id(u32 rank, u32 cpu) const { return cpu_base_[rank] + std::min(cpu, cpu_base_[rank + 1] - cpu_base_[rank] - 1); }
  u32 nic_id(u32 rank, u32 nic, bool recv) const {
    return total_cpus_ + (recv ? total_nics_ : 0) + nic_base_[rank] +
           std::min(nic, nic_base_[rank + 1] - nic_base_[rank] - 1);
  }
  int required(u32 act, u32 out[2]) const;
  Time since(u32 act) const { return act & kMsg ? msgs_[act & ~kMsg].arrive : ready_[act]; }
  void request(u32 act);
  void enqueue(u32 act, Time since, u32 res);
  void grant(u32 act, const u32* res, int count);
  void occupy(u32 res, u32 act, Time until);
  void on_wake(u32 res);
  void on_ready(u32 op);
  void start_op(u32 op);
  void on_done(u32 op);
  void begin_calc(u32 op);
  void begin_send(u32 op);
  void begin_proc(u32 m);
  void flow_changes();
  void complete_recv(u32 recv, u32 send);
  void coll_arrive(u32 op);
  void critical_path(SimColumns& out) const;

  const GraphView& g_;
  const Groups& groups_;
  const SimParams& p_;
  Model model_;
  u32 n_ = 0, ranks_ = 0, total_cpus_ = 0, total_nics_ = 0;
  Successors succ_;
  std::vector<u32> cpu_base_, nic_base_;
  std::vector<Resource> res_;
  std::unique_ptr<Topology> topology_;
  std::unique_ptr<FlowNetwork> flow_;
  Pairs pairs_;
  ModelCost cost_;
  std::unique_ptr<OsNoise> os_noise_;
  std::priority_queue<Event, std::vector<Event>, std::greater<Event>> events_;
  Time now_ = 0;
  FlowNetwork::Changes changes_;

  std::vector<Time> ready_, start_, end_, noise_;
  std::vector<u32> pend_req_, pend_ireq_, cause_, matched_, msg_of_;
  std::vector<Wait> wait_;
  std::vector<Message> msgs_;
  MatchQueues posted_, unexpected_;

  Instances inst_;
  std::vector<u64> inst_begin_;
  std::vector<u32> inst_ops_, inst_arrived_, inst_last_;
  std::vector<std::unordered_map<u32, u32>> positions_;
  std::unordered_map<u64, std::pair<std::vector<Time>, bool>> costs_;

  std::vector<Time> compute_ns_, noise_ns_, overhead_ns_, nic_ns_, wait_ns_, sync_ns_;
  std::vector<u64> bytes_sent_, bytes_received_, messages_;
  RunStats stats_;
};

Engine::Engine(const GraphView& g, const Groups& groups, const SimParams& p)
    : g_(g), groups_(groups), p_(p), model_(model_of(p.model)), pairs_(p), cost_(p, pairs_) {
  check(g, groups);
  p.check();
  n_ = static_cast<u32>(g.size());
  ranks_ = std::max<u32>(1, p.world_size);
  for (u32 i = 0; i < n_; ++i) {
    ranks_ = std::max(ranks_, g.rank[i] + 1);
    if (g.op_kind(i) == OpKind::kSend || g.op_kind(i) == OpKind::kRecv)
      ranks_ = std::max(ranks_, static_cast<u32>(g.peer[i]) + 1);
  }
  for (u32 m : groups.members) ranks_ = std::max(ranks_, m + 1);

  std::vector<u32> cpus(ranks_, 1), nics(ranks_, 1);
  for (u32 i = 0; i < n_; ++i) {
    cpus[g.rank[i]] = std::max<u32>(cpus[g.rank[i]], g.cpu[i] + 1u);
    nics[g.rank[i]] = std::max<u32>(nics[g.rank[i]], g.nic[i] + 1u);
  }
  cpu_base_.assign(ranks_ + 1, 0);
  nic_base_.assign(ranks_ + 1, 0);
  for (u32 r = 0; r < ranks_; ++r) {
    cpu_base_[r + 1] = cpu_base_[r] + cpus[r];
    nic_base_[r + 1] = nic_base_[r] + nics[r];
  }
  total_cpus_ = cpu_base_[ranks_];
  total_nics_ = nic_base_[ranks_];
  res_.resize(total_cpus_ + 2 * std::size_t{total_nics_});
  for (u32 i = total_cpus_; i < res_.size(); ++i)
    res_[i].kind = i < total_cpus_ + total_nics_ ? ResKind::kNicSend : ResKind::kNicRecv;

  msg_of_.assign(n_, kNone);
  for (u32 i = 0; i < n_; ++i)
    if (g.op_kind(i) == OpKind::kSend) {
      msg_of_[i] = static_cast<u32>(msgs_.size());
      msgs_.emplace_back().op = i;
    }
  succ_ = build_successors(g);
  pend_req_.assign(n_, 0);
  pend_ireq_.assign(n_, 0);
  for (std::size_t e = 0; e < g.edges(); ++e) ++(g.edge_kind[e] ? pend_ireq_ : pend_req_)[g.edge_dst[e]];
  stats_.unordered_ops = count_unordered(g, succ_);

  if (model_ == Model::kFlow) {
    topology_ = std::make_unique<Topology>(p, ranks_);
    flow_ = std::make_unique<FlowNetwork>(*topology_, msgs_.size());
    pairs_.flow = flow_.get();
  }
  os_noise_ = std::make_unique<OsNoise>(p.noise_frequency / 1e9, p.noise_duration, p.noise_distribution == "exponential",
                                     p.seed, total_cpus_);

  inst_ = number_instances(g, groups.count());
  inst_begin_.assign(inst_.base.back() + 1, 0);
  positions_.resize(groups.count());
  for (std::size_t grp = 0; grp < groups.count(); ++grp)
    for (u32 k = 0; k < groups.of(grp).size(); ++k) positions_[grp].emplace(groups.of(grp)[k], k);
  for (u32 i = 0; i < n_; ++i) {
    if (g.op_kind(i) != OpKind::kColl) continue;
    if (!positions_[static_cast<std::size_t>(g.coll_group[i])].count(g.rank[i]))
      throw std::invalid_argument("rank " + std::to_string(g.rank[i]) + " is not a member of collective group " +
                                  std::to_string(g.coll_group[i]));
    ++inst_begin_[inst_.id(i, g) + 1];
  }
  for (u64 k = 0; k + 1 < inst_begin_.size(); ++k) inst_begin_[k + 1] += inst_begin_[k];
  inst_ops_.resize(inst_begin_.back());
  std::vector<u64> fill(inst_begin_.begin(), inst_begin_.end() - 1);
  for (u32 i = 0; i < n_; ++i)
    if (g.op_kind(i) == OpKind::kColl) inst_ops_[fill[inst_.id(i, g)]++] = i;
  inst_arrived_.assign(inst_.base.back(), 0);
  inst_last_.assign(inst_.base.back(), kNone);

  for (auto* v : {&ready_, &start_, &end_}) v->assign(n_, kNever);
  noise_.assign(n_, 0);
  cause_.assign(n_, kNone);
  matched_.assign(n_, kNone);
  wait_.assign(n_, Wait{});
  for (auto* v : {&compute_ns_, &noise_ns_, &overhead_ns_, &nic_ns_, &wait_ns_, &sync_ns_}) v->assign(ranks_, 0);
  for (auto* v : {&bytes_sent_, &bytes_received_, &messages_}) v->assign(ranks_, 0);
  stats_.ops = n_;
  stats_.ranks = ranks_;
}

// --- Resources: an activity starts when all it needs are free and nobody who
// waited longer is queued for them.

int Engine::required(u32 act, u32 out[2]) const {
  if (act & kMsg) {  // the receiver's CPU and NIC, at the send's indices
    if (model_ == Model::kAlphaBeta) return 0;
    const u32 s = msgs_[act & ~kMsg].op, dst = static_cast<u32>(g_.peer[s]);
    out[0] = cpu_id(dst, g_.cpu[s]);
    out[1] = nic_id(dst, g_.nic[s], true);
    return model_ == Model::kLogGOPS ? 2 : 1;
  }
  if (g_.op_kind(act) == OpKind::kSend && model_ == Model::kAlphaBeta) return 0;
  out[0] = cpu_id(g_.rank[act], g_.cpu[act]);
  out[1] = nic_id(g_.rank[act], g_.nic[act], false);
  return g_.op_kind(act) == OpKind::kSend && model_ == Model::kLogGOPS ? 2 : 1;
}

void Engine::request(u32 act) {
  u32 res[2];
  const int count = required(act, res);
  int block = -1;
  for (int k = 0; k < count; ++k)
    if (res_[res[k]].free_at > std::max(now_, block < 0 ? now_ : res_[res[block]].free_at)) block = k;
  for (int k = 0; k < count && block < 0; ++k)
    if (!res_[res[k]].waiters.empty()) block = k;
  if (block < 0) grant(act, res, count);
  else enqueue(act, since(act), res[block]);
}

void Engine::enqueue(u32 act, Time since_t, u32 id) {
  Resource& r = res_[id];
  r.waiters.push({since_t, act});
  if (!r.wake) {
    r.wake = true;
    push(kWake, id, std::max(r.free_at, now_));
  }
}

void Engine::grant(u32 act, const u32* res, int count) {
  if (count > 0 && since(act) < now_) {
    const u32* last = std::max_element(res, res + count, [&](u32 a, u32 b) { return res_[a].free_at < res_[b].free_at; });
    const Wait w{res_[*last].holder, res_[*last].kind, res_[*last].hold_start};
    (act & kMsg ? msgs_[act & ~kMsg].wait : wait_[act]) = w;
  }
  if (act & kMsg) begin_proc(act & ~kMsg);
  else if (g_.op_kind(act) == OpKind::kCalc) begin_calc(act);
  else begin_send(act);
}

void Engine::occupy(u32 id, u32 act, Time until) {
  res_[id].free_at = until;
  res_[id].holder = act;
  res_[id].hold_start = now_;
}

void Engine::on_wake(u32 id) {
  Resource& r = res_[id];
  r.wake = false;
  while (!r.waiters.empty() && r.free_at <= now_) {
    const Waiter w = r.waiters.top();
    r.waiters.pop();
    u32 res[2];
    const int count = required(w.act, res);
    int other = -1;
    for (int k = 0; k < count && other < 0; ++k) {
      const Resource& o = res_[res[k]];
      if (res[k] != id && (o.free_at > now_ || (!o.waiters.empty() && o.waiters.top() < w))) other = k;
    }
    if (other < 0) grant(w.act, res, count);
    else enqueue(w.act, w.since, res[other]);
  }
  if (!r.waiters.empty() && !r.wake) {
    r.wake = true;
    push(kWake, id, std::max(r.free_at, now_));
  }
}

// --- Ops.

void Engine::on_ready(u32 op) {
  ready_[op] = now_;
  switch (g_.op_kind(op)) {
    case OpKind::kNop:
      start_op(op);
      push(kDone, op, now_);
      break;
    case OpKind::kCalc:
    case OpKind::kSend:
      request(op);
      break;
    case OpKind::kRecv: {
      start_op(op);
      const u32 send = take(unexpected_, {g_.rank[op], static_cast<u32>(g_.peer[op]), g_.tag[op]});
      if (send != kNone) complete_recv(op, send);
      else posted_[{g_.rank[op], static_cast<u32>(g_.peer[op]), g_.tag[op]}].push_back(op);
      break;
    }
    case OpKind::kColl:
      coll_arrive(op);
      break;
  }
}

void Engine::start_op(u32 op) {
  start_[op] = now_;
  for (u64 k = succ_.offsets[op]; k < succ_.offsets[op + 1]; ++k) {
    const u32 t = succ_.targets[k];
    if (succ_.kinds[k] && --pend_ireq_[t] == 0 && pend_req_[t] == 0) {
      cause_[t] = op | kStarted;
      push(kReady, t, now_);
    }
  }
}

void Engine::on_done(u32 op) {
  end_[op] = now_;
  ++stats_.finished_ops;
  for (u64 k = succ_.offsets[op]; k < succ_.offsets[op + 1]; ++k) {
    const u32 t = succ_.targets[k];
    if (!succ_.kinds[k] && --pend_req_[t] == 0 && pend_ireq_[t] == 0) {
      cause_[t] = op;
      push(kReady, t, now_);
    }
  }
}

void Engine::begin_calc(u32 op) {
  start_op(op);
  const u32 rank = g_.rank[op], cpu = cpu_id(rank, g_.cpu[op]);
  const Time d = g_.duration[op];
  noise_[op] = os_noise_->stretch(cpu, now_, d);
  occupy(cpu, op, now_ + d + noise_[op]);
  compute_ns_[rank] += d;
  noise_ns_[rank] += noise_[op];
  wait_ns_[rank] += now_ - ready_[op];
  push(kDone, op, now_ + d + noise_[op]);
}

void Engine::begin_send(u32 op) {
  start_op(op);
  Message& m = msgs_[msg_of_[op]];
  const u32 src = g_.rank[op], dst = static_cast<u32>(g_.peer[op]);
  const u64 bytes = g_.bytes[op];
  wait_ns_[src] += now_ - ready_[op];
  bytes_sent_[src] += bytes;
  ++messages_[src];
  ++stats_.messages;
  if (model_ == Model::kAlphaBeta) {
    m.cpu_end = now_;
    m.flight = now_ + ns(pairs_.alpha(src, dst));
    push(kArrive, op, now_ + ns(pairs_.alpha(src, dst) + static_cast<double>(bytes) * pairs_.beta(src, dst)));
    push(kDone, op, now_);
    return;
  }
  const LogGOPS& l = pairs_.loggops(src, dst);
  const u32 cpu = cpu_id(src, g_.cpu[op]);
  const Time busy = ns(l.o + minus_one(bytes) * l.O);
  noise_[op] = os_noise_->stretch(cpu, now_, busy);
  m.cpu_end = now_ + busy + noise_[op];
  m.flight = now_ + ns(l.o);
  m.rendezvous = bytes > p_.eager_limit;
  occupy(cpu, op, m.cpu_end);
  overhead_ns_[src] += busy;
  if (model_ == Model::kLogGOPS) {
    const Time gap = ns(l.g + minus_one(bytes) * pairs_.G(src, dst));
    occupy(nic_id(src, g_.nic[op], false), op, now_ + gap);
    nic_ns_[src] += gap;
    push(kArrive, op, m.flight + ns(l.L));
  } else {
    push(kFlowStart, op, m.flight);
  }
  if (!m.rendezvous) push(kDone, op, m.cpu_end);
}

void Engine::flow_changes() {
  for (const auto& [id, t] : changes_)
    if (t != kNever) push(kFlowDone, msgs_[id].op, t);
  changes_.clear();
}

void Engine::begin_proc(u32 id) {
  Message& m = msgs_[id];
  const u32 s = m.op, src = g_.rank[s], dst = static_cast<u32>(g_.peer[s]);
  m.proc_start = m.proc_o_end = m.proc_end = now_;
  if (model_ != Model::kAlphaBeta) {
    const LogGOPS& l = pairs_.loggops(src, dst);
    const double b1 = minus_one(g_.bytes[s]), per_byte_G = b1 * pairs_.G(src, dst);
    const u32 cpu = cpu_id(dst, g_.cpu[s]);
    Time busy = ns(l.o + b1 * l.O);
    if (model_ == Model::kLogGOPS) {
      busy = ns(l.o + std::max(b1 * l.O, per_byte_G));
      m.proc_o_end = now_ + ns(l.o);
      m.drain = per_byte_G >= b1 * l.O ? Segment::kTransfer : Segment::kOverhead;
      const Time gap = ns(l.g + per_byte_G);
      occupy(nic_id(dst, g_.nic[s], true), id | kMsg, now_ + gap);
      nic_ns_[dst] += gap;
    } else {
      m.proc_o_end = now_ + busy;
    }
    m.proc_noise = os_noise_->stretch(cpu, now_, model_ == Model::kLogGOPS ? ns(l.o) : busy);
    m.proc_end = now_ + busy + m.proc_noise;
    occupy(cpu, id | kMsg, m.proc_end);
    overhead_ns_[dst] += busy;
  }
  const u32 recv = take(posted_, {dst, src, g_.tag[s]});
  if (recv != kNone) complete_recv(recv, s);
  else unexpected_[{dst, src, g_.tag[s]}].push_back(s);
}

void Engine::complete_recv(u32 recv, u32 send) {
  const Message& m = msgs_[msg_of_[send]];
  matched_[recv] = send;
  matched_[send] = recv;
  push(kDone, recv, std::max(start_[recv], m.proc_end));
  bytes_received_[g_.rank[recv]] += g_.bytes[send];
  if (m.rendezvous) push(kDone, send, std::max({m.cpu_end, m.proc_start, start_[recv]}));
}

void Engine::coll_arrive(u32 op) {
  const u64 inst = inst_.id(op, g_);
  inst_last_[inst] = op;
  const auto grp = static_cast<u32>(g_.coll_group[op]);
  const auto members = groups_.of(grp);
  if (++inst_arrived_[inst] < members.size()) return;
  const auto ops = std::span<const u32>(inst_ops_).subspan(inst_begin_[inst], inst_begin_[inst + 1] - inst_begin_[inst]);
  std::vector<Time> finish;
  if (p_.collective_cost == "recorded") {
    Time shortest = kNever;
    for (u32 x : ops) shortest = std::min<Time>(shortest, g_.duration[x]);
    finish.assign(members.size(), shortest);
  } else {
    const u64 key = mix64(g_.bytes[op] ^ mix64((u64{grp} << 32) ^ (u64(std::max(0, g_.coll_root[op])) << 8) ^
                                                static_cast<u64>(g_.coll_type[op])));
    auto [it, fresh] = costs_.try_emplace(key);
    if (fresh) {
      CollSpec spec;
      spec.type = static_cast<CollType>(g_.coll_type[op]);
      spec.bytes = g_.bytes[op];
      spec.root = static_cast<u32>(std::max(0, g_.coll_root[op])) % members.size();
      it->second.first = evaluate(spec, members, p_, cost_, &it->second.second);
    }
    finish = it->second.first;
    stats_.collective_fallbacks += it->second.second;
  }
  for (u32 x : ops) {
    start_op(x);
    matched_[x] = op;
    sync_ns_[g_.rank[x]] += now_ - ready_[x];
    push(kDone, x, now_ + finish[positions_[grp].at(g_.rank[x])]);
  }
}

SimColumns Engine::run() {
  const auto wall_start = std::chrono::steady_clock::now();
  for (u32 i = 0; i < n_; ++i)
    if (pend_req_[i] == 0 && pend_ireq_[i] == 0) push(kReady, i, 0);
  while (!events_.empty()) {
    const Event e = events_.top();
    events_.pop();
    now_ = e.t;
    if (p_.max_events && stats_.events >= p_.max_events) {
      stats_.stopped_early = true;
      break;
    }
    ++stats_.events;
    switch (e.type) {
      case kDone: on_done(e.id); break;
      case kFlowDone: {
        const u32 id = msg_of_[e.id];
        if (!flow_->due(id, now_)) break;  // superseded by a re-share
        flow_->finish(id, now_, changes_);
        flow_changes();
        msgs_[id].data_done = now_;
        push(kArrive, e.id, now_ + flow_->latency(g_.rank[e.id], static_cast<u32>(g_.peer[e.id])));
        break;
      }
      case kArrive:
        msgs_[msg_of_[e.id]].arrive = now_;
        request(msg_of_[e.id] | kMsg);
        break;
      case kFlowStart: {
        Message& m = msgs_[msg_of_[e.id]];
        const u32 src = g_.rank[e.id], dst = static_cast<u32>(g_.peer[e.id]);
        m.ideal = now_ + transfer_time(static_cast<double>(g_.bytes[e.id]), flow_->bottleneck(src, dst));
        if (g_.bytes[e.id] == 0) {
          m.data_done = now_;
          push(kArrive, e.id, now_ + flow_->latency(src, dst));
        } else {
          flow_->start(msg_of_[e.id], src, dst, g_.bytes[e.id], now_, changes_);
          flow_changes();
        }
        break;
      }
      case kWake: on_wake(e.id); break;
      case kReady: on_ready(e.id); break;
    }
  }

  SimColumns out;
  out.rank_finish_ns.assign(ranks_, 0);
  for (u32 i = 0; i < n_; ++i) {
    if (end_[i] == kNever) {
      if (out.left_ops.size() < kListLimit) out.left_ops.push_back(i);
      continue;
    }
    stats_.makespan_ns = std::max(stats_.makespan_ns, end_[i]);
    out.rank_finish_ns[g_.rank[i]] = std::max(out.rank_finish_ns[g_.rank[i]], end_[i]);
  }
  for (auto [queues, list] : {std::pair{&posted_, &out.left_recvs}, std::pair{&unexpected_, &out.left_sends}}) {
    for (const auto& [key, ops] : *queues)
      for (u32 op : ops)
        if (list->size() < kListLimit) list->push_back(op);
    std::sort(list->begin(), list->end());
  }
  for (u64 k = 0; k < inst_arrived_.size(); ++k)
    if (inst_arrived_[k] > 0 && out.left_collectives.size() < kListLimit &&
        inst_arrived_[k] < groups_.of(static_cast<std::size_t>(g_.coll_group[inst_last_[k]])).size())
      out.left_collectives.push_back(k);
  for (const Message& m : msgs_)
    if (m.arrive != kNever) {
      out.msg_send.push_back(m.op);
      out.msg_arrive_ns.push_back(m.arrive);
      out.msg_done_ns.push_back(m.proc_end);
    }
  out.rank_compute_ns = compute_ns_;
  out.rank_noise_ns = noise_ns_;
  out.rank_overhead_ns = overhead_ns_;
  out.rank_nic_ns = nic_ns_;
  out.rank_wait_ns = wait_ns_;
  out.rank_sync_ns = sync_ns_;
  out.rank_bytes_sent = bytes_sent_;
  out.rank_bytes_received = bytes_received_;
  out.rank_messages = messages_;
  critical_path(out);
  if (topology_) {
    for (const Link& l : topology_->links()) {
      out.link_kind.push_back(static_cast<u8>(l.kind));
      out.link_src.push_back(l.from);
      out.link_dst.push_back(l.to);
      out.link_capacity_gbs.push_back(l.capacity);
      out.link_latency_ns.push_back(l.latency);
    }
    out.link_bytes = flow_->link_bytes;
    out.link_busy_ns = flow_->link_busy;
    out.link_max_flows = flow_->link_max_flows;
    stats_.flow_reshares = flow_->reshares;
  }
  out.op_ready_ns = std::move(ready_);
  out.op_start_ns = std::move(start_);
  out.op_end_ns = std::move(end_);
  out.op_noise_ns = std::move(noise_);
  out.op_matched = std::move(matched_);
  stats_.wall_seconds = duration_seconds(std::chrono::steady_clock::now() - wall_start);
  out.stats = stats_;
  return out;
}

// Walks back from the op that finished last. Each step explains the stretch
// that ends where the previous one began (an op's work, a message's
// processing or flight, or the activity whose resource it waited for), then
// moves to what made that stretch start when it did.
void Engine::critical_path(SimColumns& out) const {
  u32 last = kNone;
  for (u32 i = 0; i < n_; ++i)
    if (end_[i] != kNever && (last == kNone || end_[i] > end_[last])) last = i;
  if (last == kNone) return;

  struct Seg {
    u32 op, rank;
    Segment kind;
    Time start, end;
  };
  std::vector<Seg> segs;
  Time cursor = end_[last];
  auto emit = [&](u32 op, u32 rank, Segment kind, Time a, Time b) {
    b = std::min(b, cursor);
    a = std::min(std::max<Time>(a, 0), b);
    if (b < cursor) segs.push_back({op, rank, Segment::kIdle, b, cursor});
    if (b > a) segs.push_back({op, rank, kind, a, b});
    cursor = a;
  };

  enum class At { kBody, kStart, kProc, kProcStart, kFlight, kHold };
  At at = At::kBody;
  u32 id = last;
  Wait hold;
  for (u64 steps = 0, limit = 6 * (u64{n_} + msgs_.size()) + 16; cursor > 0 && steps < limit; ++steps) {
    switch (at) {
      case At::kBody: {
        const u32 op = id, rank = g_.rank[op];
        at = At::kStart;
        if (g_.op_kind(op) == OpKind::kCalc) {
          emit(op, rank, Segment::kNoise, start_[op] + g_.duration[op], end_[op]);
          emit(op, rank, Segment::kCompute, start_[op], start_[op] + g_.duration[op]);
        } else if (g_.op_kind(op) == OpKind::kSend) {
          const Message& m = msgs_[msg_of_[op]];
          if (m.rendezvous && end_[op] > m.cpu_end && matched_[op] != kNone) {
            const u32 recv = matched_[op];
            if (m.proc_start >= start_[recv]) {
              at = At::kProcStart;
              id = msg_of_[op];
            } else {
              id = recv;
            }
          } else {
            emit(op, rank, Segment::kNoise, m.cpu_end - noise_[op], m.cpu_end);
            emit(op, rank, Segment::kOverhead, start_[op], m.cpu_end - noise_[op]);
          }
        } else if (g_.op_kind(op) == OpKind::kRecv && matched_[op] != kNone && end_[op] > start_[op]) {
          at = At::kProc;
          id = msg_of_[matched_[op]];
        } else if (g_.op_kind(op) == OpKind::kColl) {
          emit(op, rank, Segment::kCollective, start_[op], end_[op]);
          id = matched_[op];  // the member that arrived last
        }
        break;
      }
      case At::kStart: {
        const u32 op = id;
        if (start_[op] > ready_[op] && wait_[op].holder != kNone) {
          hold = wait_[op];
          at = At::kHold;
        } else if (cause_[op] == kNone) {
          emit(op, g_.rank[op], Segment::kIdle, 0, ready_[op]);
          cursor = 0;
        } else {
          id = cause_[op] & ~kStarted;
          at = cause_[op] & kStarted ? At::kStart : At::kBody;
        }
        break;
      }
      case At::kProc: {
        const Message& m = msgs_[id];
        const u32 who = matched_[m.op] != kNone ? matched_[m.op] : m.op, dst = static_cast<u32>(g_.peer[m.op]);
        emit(who, dst, Segment::kNoise, m.proc_end - m.proc_noise, m.proc_end);
        emit(who, dst, m.drain, m.proc_o_end, m.proc_end - m.proc_noise);
        emit(who, dst, Segment::kOverhead, m.proc_start, m.proc_o_end);
        at = At::kProcStart;
        break;
      }
      case At::kProcStart: {
        const Message& m = msgs_[id];
        if (m.proc_start > m.arrive && m.wait.holder != kNone) {
          hold = m.wait;
          at = At::kHold;
        } else {
          at = At::kFlight;
        }
        break;
      }
      case At::kFlight: {
        const Message& m = msgs_[id];
        const u32 s = m.op, src = g_.rank[s];
        if (model_ == Model::kAlphaBeta) {
          emit(s, src, Segment::kTransfer, m.flight, m.arrive);
          emit(s, src, Segment::kLatency, start_[s], m.flight);
        } else if (model_ == Model::kLogGOPS) {
          emit(s, src, Segment::kLatency, m.flight, m.arrive);
          emit(s, src, Segment::kOverhead, start_[s], m.flight);
        } else {
          emit(s, src, Segment::kLatency, m.data_done, m.arrive);
          emit(s, src, Segment::kContention, m.ideal, m.data_done);
          emit(s, src, Segment::kTransfer, m.flight, m.ideal);
          emit(s, src, Segment::kOverhead, start_[s], m.flight);
        }
        id = s;
        at = At::kStart;
        break;
      }
      case At::kHold: {
        const Segment kind = hold.kind == ResKind::kCpu ? Segment::kOverhead : Segment::kTransfer;
        if (hold.holder & kMsg) {
          const u32 s = msgs_[hold.holder & ~kMsg].op;
          emit(matched_[s] != kNone ? matched_[s] : s, static_cast<u32>(g_.peer[s]), kind, hold.hold_start, cursor);
          id = hold.holder & ~kMsg;
          at = At::kProcStart;
        } else {
          const u32 h = hold.holder;
          if (g_.op_kind(h) == OpKind::kCalc) {
            emit(h, g_.rank[h], Segment::kNoise, hold.hold_start + g_.duration[h], cursor);
            emit(h, g_.rank[h], Segment::kCompute, hold.hold_start, hold.hold_start + g_.duration[h]);
          } else {
            emit(h, g_.rank[h], kind, hold.hold_start, cursor);
          }
          id = h;
          at = At::kStart;
        }
        break;
      }
    }
  }
  if (cursor > 0) segs.push_back({last, g_.rank[last], Segment::kIdle, 0, cursor});
  for (auto it = segs.rbegin(); it != segs.rend(); ++it) {
    out.cp_op.push_back(it->op);
    out.cp_rank.push_back(it->rank);
    out.cp_segment.push_back(static_cast<u8>(it->kind));
    out.cp_start_ns.push_back(it->start);
    out.cp_end_ns.push_back(it->end);
  }
}

}  // namespace

SimColumns simulate(const GraphView& g, const Groups& groups, const SimParams& params) {
  return Engine(g, groups, params).run();
}

std::vector<Time> collective_finish_times(CollSpec spec, std::span<const std::uint32_t> members,
                                          const SimParams& params) {
  params.check();
  Pairs pairs(params);
  std::unique_ptr<Topology> topology;
  std::unique_ptr<FlowNetwork> flow;
  if (pairs.model() == Model::kFlow) {
    u32 ranks = params.world_size;
    for (u32 m : members) ranks = std::max(ranks, m + 1);
    topology = std::make_unique<Topology>(params, ranks);
    flow = std::make_unique<FlowNetwork>(*topology, 0);
    pairs.flow = flow.get();
  }
  bool fell_back = false;
  return evaluate(spec, members, params, ModelCost(params, pairs), &fell_back);
}

}  // namespace cta::sim
