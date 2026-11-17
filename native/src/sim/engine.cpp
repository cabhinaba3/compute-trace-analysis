#include "sim/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "sim/collectives.hpp"
#include "sim/flow.hpp"
#include "sim/lowering.hpp"
#include "sim/noise.hpp"
#include "sim/topology.hpp"
#include "util/util.hpp"

namespace cta::sim {
namespace {

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

// Activities hold and wait for resources: an op starting, a receiver
// processing a message, or a sender sending a rendezvous message's data.
// Their ids keep the kind in the top two bits.
enum ActKind : u32 { kActOp = 0, kActMsg = 1, kActData = 2 };
constexpr u32 kIndexMask = (1u << 30) - 1;
u32 activity(ActKind kind, u32 index) { return (static_cast<u32>(kind) << 30) | index; }
ActKind kind_of(u32 act) { return static_cast<ActKind>(act >> 30); }
u32 index_of(u32 act) { return act & kIndexMask; }
// Marks a ready cause that is a dependency's start rather than its end.
constexpr u32 kStartedBit = 1u << 31;
constexpr std::size_t kListLimit = 10000;

Time ns(double x) { return static_cast<Time>(std::llround(x)); }
// (s - 1) of the LogGP formulas; a message of 0 bytes costs what one of 1 does.
double minus_one(u64 bytes) { return bytes > 0 ? static_cast<double>(bytes - 1) : 0.0; }

enum EventType : u8 { kEvDone, kEvFlowDone, kEvArrive, kEvFlowStart, kEvRts, kEvCts, kEvWake, kEvReady };
struct Event {
  Time t;
  u32 id;
  u8 type;
};
struct EventLater {
  bool operator()(const Event& a, const Event& b) const {
    if (a.t != b.t) return a.t > b.t;
    if (a.type != b.type) return a.type > b.type;
    return a.id > b.id;
  }
};

enum class ResKind : u8 { kNone = 0, kCpu = 1, kNicSend = 2, kNicRecv = 3 };

struct Waiter {
  Time since;
  u32 act;
  bool operator>(const Waiter& o) const { return since != o.since ? since > o.since : act > o.act; }
  bool operator<(const Waiter& o) const { return o > *this; }
};
struct Resource {
  Time free_at = 0, hold_start = 0;
  u32 holder = kNone;
  ResKind kind = ResKind::kCpu;
  bool wake = false;
  std::priority_queue<Waiter, std::vector<Waiter>, std::greater<Waiter>> waiters;
};

// Why an activity started late: who held the resource it waited for last.
struct Wait {
  u32 holder = kNone;
  ResKind kind = ResKind::kNone;
  Time hold_start = 0;
};

struct Message {
  u32 op = kNone;
  bool rendezvous = false;  // completes when matched (kSync)
  bool rts = false;         // goes through RTS/CTS
  Time cpu_end = kNever, inject = kNever, data_done = kNever, arrive = kNever;
  Time proc_start = kNever, proc_end = kNever, proc_noise = 0, data_noise = 0;
  Time rts_arrive = kNever, rts_match = kNever, cts_arrive = kNever, data_start = kNever;
  Wait proc_wait, data_wait;
};

struct MatchKey {
  u32 dst, src;
  u64 tag;
  bool operator==(const MatchKey&) const = default;
};
struct MatchKeyHash {
  std::size_t operator()(const MatchKey& k) const {
    return static_cast<std::size_t>(mix64(k.tag ^ mix64((u64{k.dst} << 32) | k.src)));
  }
};
using MatchQueues = std::unordered_map<MatchKey, std::vector<u32>, MatchKeyHash>;

u32 pop_front(MatchQueues& queues, MatchQueues::iterator it) {
  const u32 v = it->second.front();
  it->second.erase(it->second.begin());
  if (it->second.empty()) queues.erase(it);
  return v;
}

// The per-pair message parameters of the three network models.
class PairParams {
 public:
  PairParams(const SimParams& p) : p_(p) {}
  void set_flow(FlowNetwork* flow) { flow_ = flow; }
  FlowNetwork* flow() const { return flow_; }

  bool same_node(u32 a, u32 b) const { return p_.ranks_per_node > 1 && a / p_.ranks_per_node == b / p_.ranks_per_node; }
  const LogGOPS& loggops(u32 a, u32 b) const {
    return p_.has_intra_loggops && same_node(a, b) ? p_.intra_loggops : p_.loggops;
  }
  double scale(u32 a, u32 b) const {
    auto of = [&](u32 r) { return r < p_.nic_scale.size() ? p_.nic_scale[r] : 1.0; };
    return std::min(of(a), of(b));
  }
  double G(u32 a, u32 b) const { return loggops(a, b).G / scale(a, b); }
  double alpha(u32 a, u32 b) const { return p_.has_intra_alpha_beta && same_node(a, b) ? p_.intra_alpha : p_.alpha; }
  double beta(u32 a, u32 b) const {
    return (p_.has_intra_alpha_beta && same_node(a, b) ? p_.intra_beta : p_.beta) / scale(a, b);
  }
  // Flight time of a control message (RTS, CTS).
  Time control_latency(u32 a, u32 b) const {
    if (p_.model == NetworkModel::kFlow) return flow_->latency(a, b);
    return ns(loggops(a, b).L);
  }

 private:
  const SimParams& p_;
  FlowNetwork* flow_ = nullptr;
};

// Contention-free message costs for analytically costed collectives.
class ModelCost final : public CostModel {
 public:
  ModelCost(const SimParams& p, const PairParams& pairs) : p_(p), pairs_(pairs) {}
  Time send_busy(u32 src, u32 dst, u64 bytes) const override {
    if (p_.model == NetworkModel::kAlphaBeta) return 0;
    const LogGOPS& l = pairs_.loggops(src, dst);
    return ns(l.o + minus_one(bytes) * l.O);
  }
  Time deliver(u32 src, u32 dst, u64 bytes) const override {
    const double b1 = minus_one(bytes);
    switch (p_.model) {
      case NetworkModel::kAlphaBeta:
        return ns(pairs_.alpha(src, dst) + static_cast<double>(bytes) * pairs_.beta(src, dst));
      case NetworkModel::kLogGOPS: {
        const LogGOPS& l = pairs_.loggops(src, dst);
        return ns(l.o) + ns(l.L) + ns(l.o + std::max(b1 * l.O, b1 * pairs_.G(src, dst)));
      }
      case NetworkModel::kFlow: {
        const LogGOPS& l = pairs_.loggops(src, dst);
        FlowNetwork* flow = pairs_.flow();
        return ns(l.o) + transfer_time(static_cast<double>(bytes), flow->bottleneck(src, dst)) +
               flow->latency(src, dst) + ns(l.o + b1 * l.O);
      }
    }
    return 0;
  }
  Time reduce(u64 bytes) const override {
    return ns(static_cast<double>(bytes) * p_.collectives.reduce_ns_per_byte);
  }

 private:
  const SimParams& p_;
  const PairParams& pairs_;
};

void check_params(const SimParams& p) {
  auto bad = [](const std::string& what) { throw std::invalid_argument("simulation parameters: " + what); };
  for (const LogGOPS* l : {&p.loggops, &p.intra_loggops})
    if (l->L < 0 || l->o < 0 || l->g < 0 || l->G < 0 || l->O < 0) bad("LogGOPS parameters must not be negative");
  if (p.alpha < 0 || p.beta < 0 || p.intra_alpha < 0 || p.intra_beta < 0) bad("alpha and beta must not be negative");
  if (p.noise.frequency < 0 || p.noise.duration < 0) bad("noise frequency and duration must not be negative");
  if (p.collectives.reduce_ns_per_byte < 0) bad("reduce_ns_per_byte must not be negative");
  for (double s : p.nic_scale)
    if (!(s > 0)) bad("NIC scale factors must be positive");
}

struct CostKey {
  u64 bytes;
  u32 group, root;
  int type;
  bool operator==(const CostKey&) const = default;
};
struct CostKeyHash {
  std::size_t operator()(const CostKey& k) const {
    return static_cast<std::size_t>(mix64(k.bytes ^ mix64((u64{k.group} << 32) ^ (u64{k.root} << 8) ^ static_cast<u64>(k.type))));
  }
};
struct CostEntry {
  std::vector<Time> finish;
  bool fell_back = false;
};

std::vector<Time> evaluate(const CollSpec& wanted, std::span<const u32> members, const SimParams& p,
                           const CostModel& cost, bool* fell_back) {
  CollSpec spec = wanted;
  const u32 node_size = node_size_of(members, p.ranks_per_node);
  spec.algorithm = resolve_algorithm(spec.type, p.collectives.algorithms[static_cast<std::size_t>(spec.type)],
                                     spec.size, node_size, fell_back);
  spec.node_size = node_size == 0 ? 1 : node_size;
  spec.chunks = p.collectives.chunks;
  spec.reduce = p.collectives.reduce_ns_per_byte > 0;
  spec.all_to_all_per_peer = p.collectives.all_to_all_per_peer;
  return evaluate_collective(spec, members, cost, p.collectives.data_completion);
}

class Engine {
 public:
  Engine(const GraphView& g, const Groups& groups, const SimParams& p);
  SimColumns run();

 private:
  // --- Setup and bookkeeping.
  void push(EventType type, u32 id, Time t) { events_.push({t, id, static_cast<u8>(type)}); }
  u32 cpu_id(u32 rank, u32 cpu) const {
    return cpu_base_[rank] + std::min(cpu, cpu_base_[rank + 1] - cpu_base_[rank] - 1);
  }
  u32 nic_offset(u32 rank, u32 nic) const {
    return nic_base_[rank] + std::min(nic, nic_base_[rank + 1] - nic_base_[rank] - 1);
  }
  u32 nic_send_id(u32 rank, u32 nic) const { return total_cpus_ + nic_offset(rank, nic); }
  u32 nic_recv_id(u32 rank, u32 nic) const { return total_cpus_ + total_nics_ + nic_offset(rank, nic); }
  bool loggops() const { return p_.model == NetworkModel::kLogGOPS; }
  bool alpha_beta() const { return p_.model == NetworkModel::kAlphaBeta; }
  bool uses_rts(u32 op) const {
    return !alpha_beta() && p_.rendezvous == Rendezvous::kRtsCts && g_.bytes[op] > p_.S;
  }

  // --- Resources.
  int required(u32 act, u32 out[2]) const;
  Time since(u32 act) const;
  void set_wait(u32 act, const Wait& w);
  void request(u32 act);
  void enqueue(u32 act, Time since, u32 res);
  void grant(u32 act, const u32* res, int count);
  void occupy(u32 res, u32 act, Time until);
  void on_wake(u32 res);
  void begin(u32 act);

  // --- Ops and messages.
  void on_ready(u32 op);
  void on_done(u32 op);
  void start_op(u32 op);
  void begin_calc(u32 op);
  void begin_send(u32 op);
  Time send_data(u32 op, Message& m, u32 holder);
  void on_flow_start(u32 op);
  void on_flow_done(u32 op);
  void schedule_flow_changes();
  void begin_proc(u32 m);
  void deliver(u32 m);
  void complete_recv(u32 recv, u32 send);
  void post_recv(u32 op);
  void on_rts(u32 op);
  void rts_matched(u32 send);
  void begin_data(u32 m);
  void link(u32 recv, u32 send) {
    matched_[recv] = send;
    matched_[send] = recv;
  }

  // --- Collectives.
  void coll_arrive(u32 op);
  void complete_instance(u64 inst, u32 op);

  // --- Results.
  void critical_path(SimColumns& out) const;

  const GraphView& g_;
  const Groups& groups_;
  const SimParams& p_;
  u32 n_ = 0, ranks_ = 0, total_cpus_ = 0, total_nics_ = 0;
  Successors succ_;
  std::vector<u32> cpu_base_, nic_base_;
  std::vector<Resource> res_;
  std::unique_ptr<Topology> topology_;
  std::unique_ptr<FlowNetwork> flow_;
  PairParams pairs_;
  ModelCost cost_;
  std::unique_ptr<OsNoise> noise_;
  std::priority_queue<Event, std::vector<Event>, EventLater> events_;
  Time now_ = 0;
  FlowNetwork::Changes changes_;

  std::vector<Time> ready_, start_, end_, noise_ns_;
  std::vector<u32> pend_req_, pend_ireq_, cause_, matched_, msg_of_;
  std::vector<Wait> wait_;
  std::vector<Message> msgs_;
  MatchQueues posted_, unexpected_;

  Instances inst_;
  std::vector<u64> inst_begin_;
  std::vector<u32> inst_ops_, inst_arrived_, inst_last_;
  std::vector<std::unordered_map<u32, u32>> positions_;
  std::unordered_map<CostKey, CostEntry, CostKeyHash> cost_cache_;

  std::vector<Time> compute_ns_, noise_rank_, overhead_, nic_busy_, wait_ns_, sync_ns_;
  std::vector<u64> bytes_sent_, bytes_recv_, messages_;
  RunStats stats_;
};

Engine::Engine(const GraphView& g, const Groups& groups, const SimParams& p)
    : g_(g), groups_(groups), p_(p), pairs_(p), cost_(p, pairs_) {
  groups.check();
  g.check(groups.count());
  check_params(p);
  n_ = static_cast<u32>(g.size());

  ranks_ = p.world_size;
  for (u32 i = 0; i < n_; ++i) {
    ranks_ = std::max(ranks_, g.rank[i] + 1);
    if (g.op_kind(i) == OpKind::kSend || g.op_kind(i) == OpKind::kRecv)
      ranks_ = std::max(ranks_, static_cast<u32>(g.peer[i]) + 1);
  }
  for (u32 m : groups.members) ranks_ = std::max(ranks_, m + 1);
  if (ranks_ == 0) ranks_ = 1;

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
  res_.resize(total_cpus_ + 2 * static_cast<std::size_t>(total_nics_));
  for (u32 i = 0; i < res_.size(); ++i)
    res_[i].kind = i < total_cpus_ ? ResKind::kCpu : i < total_cpus_ + total_nics_ ? ResKind::kNicSend : ResKind::kNicRecv;

  msg_of_.assign(n_, kNone);
  for (u32 i = 0; i < n_; ++i)
    if (g.op_kind(i) == OpKind::kSend) {
      msg_of_[i] = static_cast<u32>(msgs_.size());
      msgs_.emplace_back().op = i;
    }

  succ_ = build_successors(g);
  pend_req_.assign(n_, 0);
  pend_ireq_.assign(n_, 0);
  for (std::size_t e = 0; e < g.edges(); ++e)
    ++(g.edge_kind[e] == static_cast<u8>(DepKind::kIrequires) ? pend_ireq_ : pend_req_)[g.edge_dst[e]];
  stats_.unordered_ops = count_unordered(g, succ_);

  if (p.model == NetworkModel::kFlow) {
    topology_ = Topology::make(p.topology, ranks_, p.ranks_per_node, p.nic_scale, p.seed);
    flow_ = std::make_unique<FlowNetwork>(*topology_, msgs_.size());
    pairs_.set_flow(flow_.get());
  }
  noise_ = std::make_unique<OsNoise>(p.noise, p.seed, total_cpus_);

  inst_ = number_instances(g, groups.count());
  const u64 instances = inst_.base.back();
  inst_begin_.assign(instances + 1, 0);
  positions_.resize(groups.count());
  for (std::size_t grp = 0; grp < groups.count(); ++grp) {
    const auto members = groups.of(grp);
    for (u32 k = 0; k < members.size(); ++k) positions_[grp].emplace(members[k], k);
  }
  for (u32 i = 0; i < n_; ++i) {
    if (g.op_kind(i) != OpKind::kColl) continue;
    const auto grp = static_cast<std::size_t>(g.coll_group[i]);
    if (!positions_[grp].count(g.rank[i]))
      throw std::invalid_argument("rank " + std::to_string(g.rank[i]) + " runs a collective of group " +
                                  std::to_string(grp) + " but is not one of its members");
    ++inst_begin_[inst_.id(i, g) + 1];
  }
  for (u64 k = 0; k < instances; ++k) inst_begin_[k + 1] += inst_begin_[k];
  inst_ops_.resize(inst_begin_.back());
  {
    std::vector<u64> fill(inst_begin_.begin(), inst_begin_.end() - 1);
    for (u32 i = 0; i < n_; ++i)
      if (g.op_kind(i) == OpKind::kColl) inst_ops_[fill[inst_.id(i, g)]++] = i;
  }
  inst_arrived_.assign(instances, 0);
  inst_last_.assign(instances, kNone);

  ready_.assign(n_, kNever);
  start_.assign(n_, kNever);
  end_.assign(n_, kNever);
  noise_ns_.assign(n_, 0);
  cause_.assign(n_, kNone);
  matched_.assign(n_, kNone);
  wait_.assign(n_, Wait{});
  for (auto* v : {&compute_ns_, &noise_rank_, &overhead_, &nic_busy_, &wait_ns_, &sync_ns_}) v->assign(ranks_, 0);
  for (auto* v : {&bytes_sent_, &bytes_recv_, &messages_}) v->assign(ranks_, 0);
  stats_.ops = n_;
  stats_.ranks = ranks_;
}

// --- Resources ---------------------------------------------------------------

int Engine::required(u32 act, u32 out[2]) const {
  const u32 i = index_of(act);
  switch (kind_of(act)) {
    case kActOp: {
      if (g_.op_kind(i) == OpKind::kCalc) {
        out[0] = cpu_id(g_.rank[i], g_.cpu[i]);
        return 1;
      }
      if (alpha_beta()) return 0;
      out[0] = cpu_id(g_.rank[i], g_.cpu[i]);
      if (!loggops() || uses_rts(i)) return 1;  // flow sends and RTS need only the CPU
      out[1] = nic_send_id(g_.rank[i], g_.nic[i]);
      return 2;
    }
    case kActMsg: {
      if (alpha_beta()) return 0;
      const u32 s = msgs_[i].op, dst = static_cast<u32>(g_.peer[s]);
      out[0] = cpu_id(dst, g_.cpu[s]);
      if (!loggops()) return 1;
      out[1] = nic_recv_id(dst, g_.nic[s]);
      return 2;
    }
    case kActData: {
      const u32 s = msgs_[i].op;
      out[0] = cpu_id(g_.rank[s], g_.cpu[s]);
      if (!loggops()) return 1;
      out[1] = nic_send_id(g_.rank[s], g_.nic[s]);
      return 2;
    }
  }
  return 0;
}

Time Engine::since(u32 act) const {
  const u32 i = index_of(act);
  switch (kind_of(act)) {
    case kActOp: return ready_[i];
    case kActMsg: return msgs_[i].arrive;
    case kActData: return msgs_[i].cts_arrive;
  }
  return now_;
}

void Engine::set_wait(u32 act, const Wait& w) {
  const u32 i = index_of(act);
  switch (kind_of(act)) {
    case kActOp: wait_[i] = w; break;
    case kActMsg: msgs_[i].proc_wait = w; break;
    case kActData: msgs_[i].data_wait = w; break;
  }
}

void Engine::request(u32 act) {
  u32 res[2];
  const int count = required(act, res);
  int block = -1;
  Time latest = now_;
  for (int k = 0; k < count; ++k)
    if (res_[res[k]].free_at > latest) {
      latest = res_[res[k]].free_at;
      block = k;
    }
  if (block < 0)
    for (int k = 0; k < count; ++k)
      if (!res_[res[k]].waiters.empty()) {
        block = k;
        break;
      }
  if (block < 0) grant(act, res, count);
  else enqueue(act, since(act), res[block]);
}

void Engine::enqueue(u32 act, Time since_t, u32 res) {
  Resource& r = res_[res];
  r.waiters.push({since_t, act});
  if (!r.wake) {
    r.wake = true;
    push(kEvWake, res, std::max(r.free_at, now_));
  }
}

void Engine::grant(u32 act, const u32* res, int count) {
  if (count > 0 && since(act) < now_) {
    int last = 0;
    for (int k = 1; k < count; ++k)
      if (res_[res[k]].free_at > res_[res[last]].free_at) last = k;
    const Resource& r = res_[res[last]];
    set_wait(act, {r.holder, r.kind, r.hold_start});
  }
  begin(act);
}

void Engine::occupy(u32 res, u32 act, Time until) {
  Resource& r = res_[res];
  r.free_at = until;
  r.holder = act;
  r.hold_start = now_;
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
    for (int k = 0; k < count; ++k) {
      if (res[k] == id) continue;
      const Resource& o = res_[res[k]];
      if (o.free_at > now_ || (!o.waiters.empty() && o.waiters.top() < w)) {
        other = k;
        break;
      }
    }
    if (other < 0) grant(w.act, res, count);
    else enqueue(w.act, w.since, res[other]);
  }
  if (!r.waiters.empty() && !r.wake) {
    r.wake = true;
    push(kEvWake, id, std::max(r.free_at, now_));
  }
}

void Engine::begin(u32 act) {
  const u32 i = index_of(act);
  switch (kind_of(act)) {
    case kActOp:
      if (g_.op_kind(i) == OpKind::kCalc) begin_calc(i);
      else begin_send(i);
      break;
    case kActMsg: begin_proc(i); break;
    case kActData: begin_data(i); break;
  }
}

// --- Ops -----------------------------------------------------------------------

void Engine::on_ready(u32 op) {
  ready_[op] = now_;
  switch (g_.op_kind(op)) {
    case OpKind::kNop:
      start_op(op);
      push(kEvDone, op, now_);
      break;
    case OpKind::kCalc:
    case OpKind::kSend:
      request(activity(kActOp, op));
      break;
    case OpKind::kRecv:
      start_op(op);
      post_recv(op);
      break;
    case OpKind::kColl:
      coll_arrive(op);
      break;
  }
}

void Engine::start_op(u32 op) {
  start_[op] = now_;
  for (std::size_t k = succ_.begin(op); k < succ_.end(op); ++k) {
    if (succ_.kinds[k] != static_cast<u8>(DepKind::kIrequires)) continue;
    const u32 t = succ_.targets[k];
    if (--pend_ireq_[t] == 0 && pend_req_[t] == 0) {
      cause_[t] = op | kStartedBit;
      push(kEvReady, t, now_);
    }
  }
}

void Engine::on_done(u32 op) {
  end_[op] = now_;
  ++stats_.finished_ops;
  for (std::size_t k = succ_.begin(op); k < succ_.end(op); ++k) {
    if (succ_.kinds[k] != static_cast<u8>(DepKind::kRequires)) continue;
    const u32 t = succ_.targets[k];
    if (--pend_req_[t] == 0 && pend_ireq_[t] == 0) {
      cause_[t] = op;
      push(kEvReady, t, now_);
    }
  }
}

void Engine::begin_calc(u32 op) {
  start_op(op);
  const u32 rank = g_.rank[op], cpu = cpu_id(rank, g_.cpu[op]);
  const Time d = g_.duration[op];
  const Time noise = noise_->stretch(cpu, now_, d);
  noise_ns_[op] = noise;
  occupy(cpu, activity(kActOp, op), now_ + d + noise);
  compute_ns_[rank] += d;
  noise_rank_[rank] += noise;
  wait_ns_[rank] += now_ - ready_[op];
  push(kEvDone, op, now_ + d + noise);
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
  if (alpha_beta()) {
    m.cpu_end = now_;
    push(kEvArrive, op, now_ + ns(pairs_.alpha(src, dst) + static_cast<double>(bytes) * pairs_.beta(src, dst)));
    push(kEvDone, op, now_);
    return;
  }
  const LogGOPS& l = pairs_.loggops(src, dst);
  if (uses_rts(op)) {
    m.rts = true;
    const u32 cpu = cpu_id(src, g_.cpu[op]);
    const Time o = ns(l.o), noise = noise_->stretch(cpu, now_, o);
    noise_ns_[op] = noise;
    occupy(cpu, activity(kActOp, op), now_ + o + noise);
    overhead_[src] += o;
    push(kEvRts, op, now_ + o + pairs_.control_latency(src, dst));
    return;
  }
  m.rendezvous = bytes > p_.S;
  noise_ns_[op] = send_data(op, m, activity(kActOp, op));
  if (!m.rendezvous) push(kEvDone, op, m.cpu_end);
}

Time Engine::send_data(u32 op, Message& m, u32 holder) {
  const u32 src = g_.rank[op], dst = static_cast<u32>(g_.peer[op]);
  const u64 bytes = g_.bytes[op];
  const LogGOPS& l = pairs_.loggops(src, dst);
  const u32 cpu = cpu_id(src, g_.cpu[op]);
  const Time busy = ns(l.o + minus_one(bytes) * l.O), noise = noise_->stretch(cpu, now_, busy);
  m.cpu_end = now_ + busy + noise;
  occupy(cpu, holder, m.cpu_end);
  overhead_[src] += busy;
  if (loggops()) {
    const Time gap = ns(l.g + minus_one(bytes) * pairs_.G(src, dst));
    occupy(nic_send_id(src, g_.nic[op]), holder, now_ + gap);
    nic_busy_[src] += gap;
    push(kEvArrive, op, now_ + ns(l.o) + ns(l.L));
  } else {
    m.inject = now_ + ns(l.o);
    push(kEvFlowStart, op, m.inject);
  }
  return noise;
}

void Engine::schedule_flow_changes() {
  for (const auto& [id, t] : changes_)
    if (t != kNever) push(kEvFlowDone, msgs_[id].op, t);
}

void Engine::on_flow_start(u32 op) {
  const u32 id = msg_of_[op], src = g_.rank[op], dst = static_cast<u32>(g_.peer[op]);
  if (g_.bytes[op] == 0) {
    msgs_[id].data_done = now_;
    push(kEvArrive, op, now_ + flow_->latency(src, dst));
    return;
  }
  changes_.clear();
  flow_->start(id, src, dst, g_.bytes[op], now_, changes_);
  schedule_flow_changes();
}

void Engine::on_flow_done(u32 op) {
  const u32 id = msg_of_[op];
  if (!flow_->due(id, now_)) return;  // superseded by a later re-share
  changes_.clear();
  flow_->finish(id, now_, changes_);
  schedule_flow_changes();
  msgs_[id].data_done = now_;
  push(kEvArrive, op, now_ + flow_->latency(g_.rank[op], static_cast<u32>(g_.peer[op])));
}

void Engine::begin_proc(u32 id) {
  Message& m = msgs_[id];
  const u32 s = m.op, src = g_.rank[s], dst = static_cast<u32>(g_.peer[s]);
  const u64 bytes = g_.bytes[s];
  m.proc_start = now_;
  m.proc_end = now_;
  if (!alpha_beta()) {
    const LogGOPS& l = pairs_.loggops(src, dst);
    const u32 cpu = cpu_id(dst, g_.cpu[s]), act = activity(kActMsg, id);
    const double b1 = minus_one(bytes);
    Time busy = 0, noise = 0;
    if (loggops()) {
      busy = ns(l.o + std::max(b1 * l.O, b1 * pairs_.G(src, dst)));
      noise = noise_->stretch(cpu, now_, ns(l.o));
      const Time gap = ns(l.g + b1 * pairs_.G(src, dst));
      occupy(nic_recv_id(dst, g_.nic[s]), act, now_ + gap);
      nic_busy_[dst] += gap;
    } else {
      busy = ns(l.o + b1 * l.O);
      noise = noise_->stretch(cpu, now_, busy);
    }
    m.proc_end = now_ + busy + noise;
    m.proc_noise = noise;
    occupy(cpu, act, m.proc_end);
    overhead_[dst] += busy;
  }
  deliver(id);
}

void Engine::deliver(u32 id) {
  const u32 s = msgs_[id].op;
  if (matched_[s] != kNone) {  // matched by its RTS
    complete_recv(matched_[s], s);
    return;
  }
  const MatchKey key{static_cast<u32>(g_.peer[s]), g_.rank[s], g_.tag[s]};
  if (auto it = posted_.find(key); it != posted_.end()) {
    const u32 recv = pop_front(posted_, it);
    link(recv, s);
    complete_recv(recv, s);
  } else {
    unexpected_[key].push_back(s);
  }
}

void Engine::complete_recv(u32 recv, u32 send) {
  const Message& m = msgs_[msg_of_[send]];
  push(kEvDone, recv, std::max(start_[recv], m.proc_end));
  bytes_recv_[g_.rank[recv]] += g_.bytes[send];
  if (m.rendezvous) push(kEvDone, send, std::max(m.cpu_end, std::max(m.proc_start, start_[recv])));
}

void Engine::post_recv(u32 op) {
  const MatchKey key{g_.rank[op], static_cast<u32>(g_.peer[op]), g_.tag[op]};
  if (auto it = unexpected_.find(key); it != unexpected_.end()) {
    const u32 send = pop_front(unexpected_, it);
    link(op, send);
    if (msgs_[msg_of_[send]].rts) rts_matched(send);
    else complete_recv(op, send);
  } else {
    posted_[key].push_back(op);
  }
}

void Engine::on_rts(u32 op) {
  msgs_[msg_of_[op]].rts_arrive = now_;
  const MatchKey key{static_cast<u32>(g_.peer[op]), g_.rank[op], g_.tag[op]};
  if (auto it = posted_.find(key); it != posted_.end()) {
    const u32 recv = pop_front(posted_, it);
    link(recv, op);
    rts_matched(op);
  } else {
    unexpected_[key].push_back(op);
  }
}

void Engine::rts_matched(u32 send) {
  Message& m = msgs_[msg_of_[send]];
  const u32 src = g_.rank[send], dst = static_cast<u32>(g_.peer[send]);
  m.rts_match = now_;
  m.cts_arrive = now_ + ns(pairs_.loggops(dst, src).o) + pairs_.control_latency(dst, src);
  push(kEvCts, send, m.cts_arrive);
}

void Engine::begin_data(u32 id) {
  Message& m = msgs_[id];
  m.data_start = now_;
  m.data_noise = send_data(m.op, m, activity(kActData, id));
  push(kEvDone, m.op, m.cpu_end);
}

// --- Collectives -----------------------------------------------------------------

void Engine::coll_arrive(u32 op) {
  const u64 inst = inst_.id(op, g_);
  ++inst_arrived_[inst];
  inst_last_[inst] = op;
  const auto group = static_cast<std::size_t>(g_.coll_group[op]);
  if (inst_arrived_[inst] == groups_.of(group).size()) complete_instance(inst, op);
}

void Engine::complete_instance(u64 inst, u32 op) {
  const auto group = static_cast<u32>(g_.coll_group[op]);
  const auto members = groups_.of(group);
  const auto ops = std::span<const u32>(inst_ops_).subspan(inst_begin_[inst], inst_begin_[inst + 1] - inst_begin_[inst]);
  std::vector<Time> finish;
  if (p_.collectives.cost == CollectiveCost::kRecorded) {
    Time shortest = kNever;
    for (u32 x : ops) shortest = std::min<Time>(shortest, g_.duration[x]);
    finish.assign(members.size(), shortest);
  } else {
    const CostKey key{g_.bytes[op], group, g_.coll_root[op] > 0 ? static_cast<u32>(g_.coll_root[op]) : 0u,
                      g_.coll_type[op]};
    auto it = cost_cache_.find(key);
    if (it == cost_cache_.end()) {
      CollSpec spec;
      spec.type = static_cast<CollType>(g_.coll_type[op]);
      spec.size = static_cast<u32>(members.size());
      spec.bytes = key.bytes;
      spec.root = spec.size ? key.root % spec.size : 0;
      CostEntry entry;
      entry.finish = evaluate(spec, members, p_, cost_, &entry.fell_back);
      it = cost_cache_.emplace(key, std::move(entry)).first;
    }
    finish = it->second.finish;
    stats_.collective_fallbacks += it->second.fell_back ? 1 : 0;
  }
  for (u32 x : ops) {
    start_op(x);
    matched_[x] = inst_last_[inst];
    sync_ns_[g_.rank[x]] += now_ - ready_[x];
    push(kEvDone, x, now_ + finish[positions_[group].at(g_.rank[x])]);
  }
}

// --- The run -------------------------------------------------------------------

SimColumns Engine::run() {
  const auto wall_start = std::chrono::steady_clock::now();
  for (u32 i = 0; i < n_; ++i)
    if (pend_req_[i] == 0 && pend_ireq_[i] == 0) push(kEvReady, i, 0);
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
      case kEvDone: on_done(e.id); break;
      case kEvFlowDone: on_flow_done(e.id); break;
      case kEvArrive:
        msgs_[msg_of_[e.id]].arrive = now_;
        request(activity(kActMsg, msg_of_[e.id]));
        break;
      case kEvFlowStart: on_flow_start(e.id); break;
      case kEvRts: on_rts(e.id); break;
      case kEvCts: request(activity(kActData, msg_of_[e.id])); break;
      case kEvWake: on_wake(e.id); break;
      case kEvReady: on_ready(e.id); break;
    }
  }

  SimColumns out;
  for (u32 i = 0; i < n_; ++i)
    if (end_[i] != kNever) stats_.makespan_ns = std::max(stats_.makespan_ns, end_[i]);
  out.rank_finish_ns.assign(ranks_, 0);
  for (u32 i = 0; i < n_; ++i)
    if (end_[i] != kNever) out.rank_finish_ns[g_.rank[i]] = std::max(out.rank_finish_ns[g_.rank[i]], end_[i]);
  for (u32 i = 0; i < n_ && out.unfinished.size() < kListLimit; ++i)
    if (end_[i] == kNever) out.unfinished.push_back(i);
  for (const auto& [key, ops] : posted_)
    for (u32 op : ops)
      if (out.unmatched_recvs.size() < kListLimit) out.unmatched_recvs.push_back(op);
  for (const auto& [key, ops] : unexpected_)
    for (u32 op : ops)
      if (out.unmatched_sends.size() < kListLimit) out.unmatched_sends.push_back(op);
  std::sort(out.unmatched_recvs.begin(), out.unmatched_recvs.end());
  std::sort(out.unmatched_sends.begin(), out.unmatched_sends.end());
  for (u64 k = 0; k < inst_arrived_.size() && out.incomplete_instances.size() < kListLimit; ++k)
    if (inst_arrived_[k] > 0 && inst_begin_[k + 1] - inst_begin_[k] != 0 && inst_last_[k] != kNone &&
        inst_arrived_[k] < groups_.of(static_cast<std::size_t>(g_.coll_group[inst_last_[k]])).size())
      out.incomplete_instances.push_back(k);

  for (const Message& m : msgs_) {
    if (m.arrive == kNever) continue;
    out.msg_send.push_back(m.op);
    out.msg_arrive_ns.push_back(m.arrive);
    out.msg_done_ns.push_back(m.proc_end);
  }
  out.rank_compute_ns = compute_ns_;
  out.rank_noise_ns = noise_rank_;
  out.rank_overhead_ns = overhead_;
  out.rank_nic_ns = nic_busy_;
  out.rank_wait_ns = wait_ns_;
  out.rank_sync_ns = sync_ns_;
  out.rank_bytes_sent = bytes_sent_;
  out.rank_bytes_received = bytes_recv_;
  out.rank_messages = messages_;

  critical_path(out);

  if (topology_) {
    const auto& links = topology_->links();
    for (u32 l = 0; l < links.size(); ++l) {
      out.link_kind.push_back(static_cast<u8>(links[l].kind));
      out.link_from.push_back(links[l].from);
      out.link_to.push_back(links[l].to);
      out.link_capacity.push_back(links[l].capacity);
      out.link_latency_ns.push_back(links[l].latency);
    }
    out.link_bytes = flow_->link_bytes();
    out.link_busy_ns = flow_->link_busy();
    out.link_max_flows = flow_->link_max_flows();
    stats_.flow_reshares = flow_->reshares();
  }

  out.ready_ns = std::move(ready_);
  out.start_ns = std::move(start_);
  out.end_ns = std::move(end_);
  out.noise_ns = std::move(noise_ns_);
  out.matched = std::move(matched_);
  stats_.wall_seconds = duration_seconds(std::chrono::steady_clock::now() - wall_start);
  out.stats = stats_;
  return out;
}

// --- Critical path -----------------------------------------------------------------
//
// Walks back from the op that finished last. Each step explains a stretch of
// time ending where the previous one began: an op's own work, a message's
// processing and flight, or the activity whose resource it waited for; then it
// moves to whatever made that stretch start when it did (a dependency, a
// message's sender, the last member of a collective).

void Engine::critical_path(SimColumns& out) const {
  u32 last = kNone;
  Time latest = -1;
  for (u32 i = 0; i < n_; ++i)
    if (end_[i] != kNever && end_[i] > latest) {
      latest = end_[i];
      last = i;
    }
  if (last == kNone) return;

  struct Seg {
    u32 op, rank;
    Segment kind;
    Time start, end;
  };
  std::vector<Seg> segs;
  Time cursor = latest;
  auto emit = [&](u32 op, u32 rank, Segment kind, Time a, Time b) {
    b = std::min(b, cursor);
    a = std::min(std::max<Time>(a, 0), b);
    if (b < cursor) segs.push_back({op, rank, Segment::kIdle, b, cursor});
    if (b > a) segs.push_back({op, rank, kind, a, b});
    cursor = a;
  };

  enum class At { kOpBody, kOpStart, kProcBody, kProcStart, kFlight, kDataStart, kHold, kDone };
  At at = At::kOpBody;
  u32 id = last;
  Wait hold;
  u32 hold_owner = kNone;  // the activity that waited (for its rank)
  const u64 limit = 8 * (static_cast<u64>(n_) + msgs_.size()) + 16;
  for (u64 steps = 0; at != At::kDone && steps < limit && cursor > 0; ++steps) {
    switch (at) {
      case At::kOpBody: {
        const u32 op = id, rank = g_.rank[op];
        switch (g_.op_kind(op)) {
          case OpKind::kCalc:
            emit(op, rank, Segment::kNoise, start_[op] + g_.duration[op], end_[op]);
            emit(op, rank, Segment::kCompute, start_[op], start_[op] + g_.duration[op]);
            at = At::kOpStart;
            break;
          case OpKind::kNop:
            at = At::kOpStart;
            break;
          case OpKind::kSend: {
            const Message& m = msgs_[msg_of_[op]];
            if (alpha_beta()) {
              at = At::kOpStart;
            } else if (m.rts) {
              const u32 dst = static_cast<u32>(g_.peer[op]);
              const LogGOPS& l = pairs_.loggops(rank, dst);
              const Time busy = ns(l.o + minus_one(g_.bytes[op]) * l.O);
              emit(op, rank, Segment::kNoise, m.data_start + busy, end_[op]);
              emit(op, rank, Segment::kOverhead, m.data_start, m.data_start + busy);
              at = At::kDataStart;
              id = msg_of_[op];
            } else if (m.rendezvous && end_[op] > m.cpu_end && matched_[op] != kNone) {
              const u32 recv = matched_[op];
              if (m.proc_start >= start_[recv]) {
                at = At::kProcStart;
                id = msg_of_[op];
              } else {
                at = At::kOpStart;
                id = recv;
              }
            } else {
              const u32 dst = static_cast<u32>(g_.peer[op]);
              const LogGOPS& l = pairs_.loggops(rank, dst);
              const Time busy = ns(l.o + minus_one(g_.bytes[op]) * l.O);
              emit(op, rank, Segment::kNoise, start_[op] + busy, end_[op]);
              emit(op, rank, Segment::kOverhead, start_[op], start_[op] + busy);
              at = At::kOpStart;
            }
            break;
          }
          case OpKind::kRecv: {
            const u32 send = matched_[op];
            if (send == kNone || end_[op] <= start_[op]) {
              at = At::kOpStart;
            } else {
              at = At::kProcBody;
              id = msg_of_[send];
            }
            break;
          }
          case OpKind::kColl:
            emit(op, rank, Segment::kCollective, start_[op], end_[op]);
            at = At::kOpStart;
            id = matched_[op] != kNone ? matched_[op] : op;
            break;
        }
        break;
      }
      case At::kOpStart: {
        const u32 op = id;
        if (start_[op] > ready_[op] && wait_[op].holder != kNone) {
          hold = wait_[op];
          hold_owner = op;
          at = At::kHold;
        } else if (cause_[op] == kNone) {
          emit(op, g_.rank[op], Segment::kIdle, 0, ready_[op]);
          at = At::kDone;
        } else if (cause_[op] & kStartedBit) {
          id = cause_[op] & ~kStartedBit;
          at = At::kOpStart;
        } else {
          id = cause_[op];
          at = At::kOpBody;
        }
        break;
      }
      case At::kProcBody: {
        const Message& m = msgs_[id];
        const u32 s = m.op, src = g_.rank[s], dst = static_cast<u32>(g_.peer[s]);
        const u32 who = matched_[s] != kNone ? matched_[s] : s;
        if (!alpha_beta()) {
          const LogGOPS& l = pairs_.loggops(src, dst);
          const double b1 = minus_one(g_.bytes[s]);
          if (loggops()) {
            const Time o = ns(l.o), busy = ns(l.o + std::max(b1 * l.O, b1 * pairs_.G(src, dst)));
            emit(who, dst, Segment::kNoise, m.proc_start + busy, m.proc_end);
            emit(who, dst, b1 * pairs_.G(src, dst) >= b1 * l.O ? Segment::kTransfer : Segment::kOverhead,
                 m.proc_start + o, m.proc_start + busy);
            emit(who, dst, Segment::kOverhead, m.proc_start, m.proc_start + o);
          } else {
            const Time busy = ns(l.o + b1 * l.O);
            emit(who, dst, Segment::kNoise, m.proc_start + busy, m.proc_end);
            emit(who, dst, Segment::kOverhead, m.proc_start, m.proc_start + busy);
          }
        }
        at = At::kProcStart;
        break;
      }
      case At::kProcStart: {
        const Message& m = msgs_[id];
        if (m.proc_start > m.arrive && m.proc_wait.holder != kNone) {
          hold = m.proc_wait;
          hold_owner = m.op;
          at = At::kHold;
        } else {
          at = At::kFlight;
        }
        break;
      }
      case At::kFlight: {
        const Message& m = msgs_[id];
        const u32 s = m.op, src = g_.rank[s], dst = static_cast<u32>(g_.peer[s]);
        const Time begin = m.rts ? m.data_start : start_[s];
        if (alpha_beta()) {
          const Time alpha = ns(pairs_.alpha(src, dst));
          emit(s, src, Segment::kTransfer, begin + alpha, m.arrive);
          emit(s, src, Segment::kLatency, begin, begin + alpha);
        } else if (loggops()) {
          const Time o = ns(pairs_.loggops(src, dst).o);
          emit(s, src, Segment::kLatency, begin + o, m.arrive);
          emit(s, src, Segment::kOverhead, begin, begin + o);
        } else {
          const Time ideal = transfer_time(static_cast<double>(g_.bytes[s]), flow_->bottleneck(src, dst));
          emit(s, src, Segment::kLatency, m.data_done, m.arrive);
          emit(s, src, Segment::kContention, m.inject + ideal, m.data_done);
          emit(s, src, Segment::kTransfer, m.inject, m.inject + ideal);
          emit(s, src, Segment::kOverhead, begin, m.inject);
        }
        if (m.rts) {
          at = At::kDataStart;
        } else {
          at = At::kOpStart;
          id = s;
        }
        break;
      }
      case At::kDataStart: {
        const Message& m = msgs_[id];
        const u32 s = m.op, src = g_.rank[s];
        if (m.data_start > m.cts_arrive && m.data_wait.holder != kNone) {
          hold = m.data_wait;
          hold_owner = s;
          at = At::kHold;
          break;
        }
        emit(s, src, Segment::kLatency, m.rts_match, m.cts_arrive);
        if (m.rts_match > m.rts_arrive && matched_[s] != kNone) {
          id = matched_[s];
          at = At::kOpStart;
        } else {
          const Time o = ns(pairs_.loggops(src, static_cast<u32>(g_.peer[s])).o);
          emit(s, src, Segment::kLatency, start_[s] + o, m.rts_arrive);
          emit(s, src, Segment::kOverhead, start_[s], start_[s] + o);
          id = s;
          at = At::kOpStart;
        }
        break;
      }
      case At::kHold: {
        const u32 h = index_of(hold.holder);
        const Segment on_res = hold.kind == ResKind::kCpu ? Segment::kOverhead : Segment::kTransfer;
        const Time until = cursor;
        (void)hold_owner;
        switch (kind_of(hold.holder)) {
          case kActOp:
            if (g_.op_kind(h) == OpKind::kCalc) {
              emit(h, g_.rank[h], Segment::kNoise, hold.hold_start + g_.duration[h], until);
              emit(h, g_.rank[h], Segment::kCompute, hold.hold_start, hold.hold_start + g_.duration[h]);
            } else {
              emit(h, g_.rank[h], on_res, hold.hold_start, until);
            }
            id = h;
            at = At::kOpStart;
            break;
          case kActMsg: {
            const u32 s = msgs_[h].op;
            emit(matched_[s] != kNone ? matched_[s] : s, static_cast<u32>(g_.peer[s]), on_res, hold.hold_start, until);
            id = h;
            at = At::kProcStart;
            break;
          }
          case kActData:
            emit(msgs_[h].op, g_.rank[msgs_[h].op], on_res, hold.hold_start, until);
            id = h;
            at = At::kDataStart;
            break;
        }
        break;
      }
      case At::kDone:
        break;
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
  Engine engine(g, groups, params);
  return engine.run();
}

std::vector<Time> collective_finish_times(const CollSpec& spec, std::span<const std::uint32_t> members,
                                          const SimParams& params) {
  check_params(params);
  PairParams pairs(params);
  std::unique_ptr<Topology> topology;
  std::unique_ptr<FlowNetwork> flow;
  if (params.model == NetworkModel::kFlow) {
    u32 ranks = params.world_size;
    for (u32 m : members) ranks = std::max(ranks, m + 1);
    topology = Topology::make(params.topology, ranks, params.ranks_per_node, params.nic_scale, params.seed);
    flow = std::make_unique<FlowNetwork>(*topology, 0);
    pairs.set_flow(flow.get());
  }
  ModelCost cost(params, pairs);
  bool fell_back = false;
  return evaluate(spec, members, params, cost, &fell_back);
}

}  // namespace cta::sim
