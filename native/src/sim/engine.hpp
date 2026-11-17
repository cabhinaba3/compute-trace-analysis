// The discrete-event simulator.
//
// * An op is ready once every op it requires has finished and every op it
//   irequires has started; ops without dependencies are ready at 0.
// * A calc holds its CPU for its duration plus OS noise. A NOP finishes when
//   ready. A recv is posted when ready and finishes once its message has been
//   processed. A collective that was not lowered starts when the last member
//   of its instance (the k-th collective of a group on each member) is ready;
//   member i then takes the algorithm's contention-free time for i, or the
//   shortest duration any member recorded.
// * Activities queue for busy CPUs and NICs; a freed resource goes to the one
//   that has waited longest, ties to the lowest id. Events at one time run as
//   completions, flow completions, arrivals, flow starts, wake-ups, ready ops,
//   ties by id, so a run depends only on its ops and parameters.
// * LogGOPS, as LogGOPSim: a send of s bytes holds its CPU for o + (s-1)O and
//   its NIC for g + (s-1)G, and its first byte reaches the receiver o + L
//   after it started; the receiver's CPU and NIC (the send's indices) then take
//   o + max((s-1)O, (s-1)G) and g + (s-1)G. A send of up to S bytes finishes
//   with its CPU time, a larger one when matched (the later of the message's
//   processing and the recv's post). Same-node pairs use the intra values.
// * Alpha-beta: a message is at the receiver alpha + s beta after its send
//   started; nothing is shared, and the send finishes at once.
// * Flow: the send holds its CPU for o + (s-1)O; the data enters the network
//   at start + o, shares bandwidth with other flows, and arrives the route's
//   latency after its last byte left; the receiver's CPU then takes
//   o + (s-1)O. S as for LogGOPS.
// * NIC scale factors divide G and beta and scale NIC links; a message uses
//   the smaller factor of its two ends.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "sim/collectives.hpp"
#include "sim/graph.hpp"
#include "sim/params.hpp"

namespace cta::sim {

struct RunStats {
  std::int64_t makespan_ns = 0;
  std::uint64_t events = 0, ops = 0, finished_ops = 0, messages = 0, unordered_ops = 0;
  std::uint64_t collective_fallbacks = 0, flow_reshares = 0;
  std::uint32_t ranks = 0;
  bool stopped_early = false;  // hit max_events
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

// Columns of one table each per prefix: op_ (per op; kNever for what never
// happened), msg_ (per message sent), rank_, cp_ (the critical path from 0 to
// the end, as Segment-coded stretches), link_ (flow model) and left_ (what did
// not finish: ops, unmatched recvs and sends, collective instances some member
// never reached; at most 10000 each).
struct SimColumns {
  std::vector<std::int64_t> op_ready_ns, op_start_ns, op_end_ns, op_noise_ns;
  std::vector<std::uint32_t> op_matched;  // a recv's send, a send's recv, a collective's last arrival
  std::vector<std::uint32_t> msg_send;
  std::vector<std::int64_t> msg_arrive_ns, msg_done_ns;
  std::vector<std::int64_t> rank_finish_ns, rank_compute_ns, rank_noise_ns, rank_overhead_ns, rank_nic_ns,
      rank_wait_ns, rank_sync_ns;
  std::vector<std::uint64_t> rank_bytes_sent, rank_bytes_received, rank_messages;
  std::vector<std::uint32_t> cp_op, cp_rank;
  std::vector<std::uint8_t> cp_segment;
  std::vector<std::int64_t> cp_start_ns, cp_end_ns;
  std::vector<std::uint8_t> link_kind;
  std::vector<std::uint32_t> link_src, link_dst, link_max_flows;
  std::vector<double> link_capacity_gbs;
  std::vector<std::int64_t> link_latency_ns, link_busy_ns;
  std::vector<std::uint64_t> link_bytes;
  std::vector<std::uint32_t> left_ops, left_recvs, left_sends;
  std::vector<std::uint64_t> left_collectives;
  RunStats stats;

  template <typename Visit>
  void for_each_column(Visit&& visit) {
    visit("op_ready_ns", op_ready_ns);
    visit("op_start_ns", op_start_ns);
    visit("op_end_ns", op_end_ns);
    visit("op_noise_ns", op_noise_ns);
    visit("op_matched", op_matched);
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
    visit("link_src", link_src);
    visit("link_dst", link_dst);
    visit("link_max_flows", link_max_flows);
    visit("link_capacity_gbs", link_capacity_gbs);
    visit("link_latency_ns", link_latency_ns);
    visit("link_busy_ns", link_busy_ns);
    visit("link_bytes", link_bytes);
    visit("left_ops", left_ops);
    visit("left_recvs", left_recvs);
    visit("left_sends", left_sends);
    visit("left_collectives", left_collectives);
    visit("stats", stats);
  }
};

// Throws std::invalid_argument for an invalid graph or parameters.
SimColumns simulate(const GraphView& g, const Groups& groups, const SimParams& params);

// Each member's contention-free finish time of one collective (the
// analytical level), with the parameters' algorithm for its type.
std::vector<Time> collective_finish_times(CollSpec spec, std::span<const std::uint32_t> members,
                                          const SimParams& params);

}  // namespace cta::sim
