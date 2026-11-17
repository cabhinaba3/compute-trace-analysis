// The discrete-event simulator.
//
// Semantics, in the order the engine applies them:
//
// * An op is ready when every op it requires has finished and every op it
//   irequires has started. Ops without dependencies are ready at time 0.
// * A calc needs its CPU (cpu index of its rank) for its duration, stretched
//   by OS noise. A NOP finishes when ready. A recv is posted when ready and
//   finishes when its message has been processed, or when it is posted if the
//   message was processed before. A collective (not lowered) starts when the
//   last member of its instance (the k-th collective of a group on each
//   member) is ready, and member i finishes after the algorithm's
//   contention-free time for member i, or after the shortest duration any
//   member recorded (CollectiveParams::cost).
// * Ops that need busy resources queue for them; a free resource goes to the
//   activity that has waited longest, ties to the lowest op id. Events at one
//   time run in the order: completions, flow completions, arrivals, flow
//   starts, RTS, CTS, resource wake-ups, ready ops; ties by id. So a run does
//   not depend on the order of anything but the ops.
// * LogGOPS (as LogGOPSim): a send of s bytes needs its CPU and its NIC; the
//   CPU is busy o + (s-1)O (plus noise), the NIC g + (s-1)G; the first byte
//   reaches the receiver at start + o + L. The receiver then needs a CPU and
//   a NIC (the send's indices) for o + max((s-1)O, (s-1)G) and g + (s-1)G.
//   An eager send (s <= S) finishes when its CPU is free; a larger one when
//   the receiver matches it (kSync: start of processing or the recv's post,
//   whichever is later) or, with kRtsCts, when its data has been sent after
//   an RTS/CTS exchange (each control message costs o + L, CPU time not
//   charged). Ranks of one node use the intra-node parameters if given.
// * Alpha-beta: a message is complete at the receiver alpha + s * beta after
//   its send started; nothing is shared and the send finishes at once.
// * Flow: the send's CPU is busy o + (s-1)O; the data enters the network at
//   start + o, shares link bandwidth with the other flows, and reaches the
//   receiver the route's latency after its last byte left; the receiver's
//   CPU then spends o + (s-1)O. Eager threshold and rendezvous as LogGOPS.
// * NIC scale factors divide G (LogGOPS) and beta and scale NIC link
//   bandwidth (flow); a message uses the smaller factor of its two ends.
#pragma once

#include <cstdint>
#include <vector>

#include "sim/graph.hpp"
#include "sim/params.hpp"
#include "sim/types.hpp"

namespace cta::sim {

struct RunStats {
  std::int64_t makespan_ns = 0;
  std::uint64_t events = 0, ops = 0, finished_ops = 0, messages = 0;
  std::uint64_t unordered_ops = 0;      // ops on or behind a dependency cycle
  std::uint64_t collective_fallbacks = 0;
  std::uint64_t flow_reshares = 0;
  std::uint32_t ranks = 0;
  bool stopped_early = false;           // hit max_events
  double wall_seconds = 0;

  template <typename Visit>
  void for_each_column(Visit&& visit) {
    visit("makespan_ns", makespan_ns);
    visit("events", events);
    visit("ops", ops);
    visit("finished_ops", finished_ops);
    visit("messages", messages);
    visit("unordered_ops", unordered_ops);
    visit("collective_fallbacks", collective_fallbacks);
    visit("flow_reshares", flow_reshares);
    visit("ranks", ranks);
    visit("stopped_early", stopped_early);
    visit("wall_seconds", wall_seconds);
  }
};

struct SimColumns {
  // Per op; kNever (int64 max) for what never happened.
  std::vector<std::int64_t> ready_ns, start_ns, end_ns, noise_ns;
  // A recv's send, a send's recv, a collective's last-arriving member.
  std::vector<std::uint32_t> matched;

  // Per message that was sent: the send op, when it reached the receiver and
  // when the receiver had processed it.
  std::vector<std::uint32_t> msg_send;
  std::vector<std::int64_t> msg_arrive_ns, msg_done_ns;

  // Per rank.
  std::vector<std::int64_t> rank_finish_ns, rank_compute_ns, rank_noise_ns, rank_overhead_ns, rank_nic_ns,
      rank_wait_ns, rank_sync_ns;
  std::vector<std::uint64_t> rank_bytes_sent, rank_bytes_received, rank_messages;

  // The critical path, from time 0 to the end, as segments (see Segment).
  std::vector<std::uint32_t> cp_op, cp_rank;
  std::vector<std::uint8_t> cp_segment;
  std::vector<std::int64_t> cp_start_ns, cp_end_ns;

  // Links of the flow model's topology.
  std::vector<std::uint8_t> link_kind;
  std::vector<std::uint32_t> link_from, link_to, link_max_flows;
  std::vector<double> link_capacity;
  std::vector<std::int64_t> link_latency_ns, link_busy_ns;
  std::vector<std::uint64_t> link_bytes;

  // What did not finish: ops, recvs never matched, sends never received,
  // collective instances some member never reached (at most 10000 each).
  std::vector<std::uint32_t> unfinished, unmatched_recvs, unmatched_sends;
  std::vector<std::uint64_t> incomplete_instances;

  RunStats stats;

  template <typename Visit>
  void for_each_column(Visit&& visit) {
    visit("ready_ns", ready_ns);
    visit("start_ns", start_ns);
    visit("end_ns", end_ns);
    visit("noise_ns", noise_ns);
    visit("matched", matched);
    visit("msg_send", msg_send);
    visit("msg_arrive_ns", msg_arrive_ns);
    visit("msg_done_ns", msg_done_ns);
    visit("rank_finish_ns", rank_finish_ns);
    visit("rank_compute_ns", rank_compute_ns);
    visit("rank_noise_ns", rank_noise_ns);
    visit("rank_overhead_ns", rank_overhead_ns);
    visit("rank_nic_ns", rank_nic_ns);
    visit("rank_wait_ns", rank_wait_ns);
    visit("rank_sync_ns", rank_sync_ns);
    visit("rank_bytes_sent", rank_bytes_sent);
    visit("rank_bytes_received", rank_bytes_received);
    visit("rank_messages", rank_messages);
    visit("cp_op", cp_op);
    visit("cp_rank", cp_rank);
    visit("cp_segment", cp_segment);
    visit("cp_start_ns", cp_start_ns);
    visit("cp_end_ns", cp_end_ns);
    visit("link_kind", link_kind);
    visit("link_from", link_from);
    visit("link_to", link_to);
    visit("link_max_flows", link_max_flows);
    visit("link_capacity", link_capacity);
    visit("link_latency_ns", link_latency_ns);
    visit("link_busy_ns", link_busy_ns);
    visit("link_bytes", link_bytes);
    visit("unfinished", unfinished);
    visit("unmatched_recvs", unmatched_recvs);
    visit("unmatched_sends", unmatched_sends);
    visit("incomplete_instances", incomplete_instances);
    visit("stats", stats);
  }
};

// Runs the graph. Throws std::invalid_argument for an invalid graph, groups
// or parameters.
SimColumns simulate(const GraphView& g, const Groups& groups, const SimParams& params);

// When each member of a collective finishes, starting at 0, under the
// parameters' network model without contention (the analytical level).
std::vector<Time> collective_finish_times(const CollSpec& spec, std::span<const std::uint32_t> members,
                                          const SimParams& params);

}  // namespace cta::sim
