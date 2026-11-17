// The flow model's network: every message in transit is a flow along its
// route, and the flows that share links split their bandwidth max-min fairly
// (progressive filling). A flow's rate changes only when a flow that shares a
// link with it, directly or through other flows, starts or ends; only those
// flows are re-shared.
#pragma once

#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sim/topology.hpp"
#include "sim/types.hpp"

namespace cta::sim {

// Nanoseconds until `bytes` have gone at `rate` bytes/ns, rounded up; a hair
// of slack keeps exact quotients from rounding up past themselves.
inline Time transfer_time(double bytes, double rate) {
  if (bytes <= 0) return 0;
  return static_cast<Time>(std::ceil(bytes / rate - 1e-6));
}

class FlowNetwork {
 public:
  // (flow, new finish time) for each flow whose finish time changed.
  using Changes = std::vector<std::pair<std::uint32_t, Time>>;

  // `max_flows` bounds the flow ids.
  FlowNetwork(const Topology& topology, std::size_t max_flows);

  // Starts flow `id` of `bytes` bytes from rank src to rank dst at `now`.
  void start(std::uint32_t id, std::uint32_t src, std::uint32_t dst, std::uint64_t bytes, Time now, Changes& changes);
  // Ends flow `id` (its last byte left at `now`) and re-shares its links.
  void finish(std::uint32_t id, Time now, Changes& changes);
  // Whether flow `id` is in transit and due to finish at `t`.
  bool due(std::uint32_t id, Time t) const { return flows_[id].active && flows_[id].finish == t; }

  // Sum of the latencies, and the smallest capacity, of the route src -> dst.
  Time latency(std::uint32_t src, std::uint32_t dst) { return route(src, dst).latency; }
  double bottleneck(std::uint32_t src, std::uint32_t dst) { return route(src, dst).bottleneck; }

  // Per link: bytes carried, time with at least one flow, most flows at once.
  const std::vector<std::uint64_t>& link_bytes() const { return link_bytes_; }
  const std::vector<Time>& link_busy() const { return link_busy_; }
  const std::vector<std::uint32_t>& link_max_flows() const { return link_max_flows_; }
  std::uint64_t reshares() const { return reshares_; }

 private:
  struct Route {
    std::uint32_t begin, size;
    Time latency;
    double bottleneck;
  };
  struct Flow {
    std::uint32_t route_begin = 0, route_size = 0;
    double remaining = 0, rate = 0;
    Time last = 0, finish = kNever;
    bool active = false;
  };

  const Route& route(std::uint32_t src, std::uint32_t dst);
  void reshare(std::uint32_t route_begin, std::uint32_t route_size, Time now, Changes& changes);

  const Topology& topology_;
  std::vector<Flow> flows_;
  std::vector<std::vector<std::uint32_t>> link_flows_;
  std::unordered_map<std::uint64_t, Route> routes_;
  std::vector<std::uint32_t> pool_;  // route links

  // Scratch for reshare, valid for the links and flows of the current epoch.
  std::uint32_t epoch_ = 0;
  std::vector<std::uint32_t> link_epoch_, flow_epoch_, link_version_, unfrozen_;
  std::vector<double> capacity_left_;
  std::vector<char> frozen_;
  std::vector<std::uint32_t> component_links_, component_flows_, stack_;

  std::vector<std::uint64_t> link_bytes_;
  std::vector<Time> link_busy_, busy_since_;
  std::vector<std::uint32_t> link_max_flows_;
  std::uint64_t reshares_ = 0;
};

}  // namespace cta::sim
