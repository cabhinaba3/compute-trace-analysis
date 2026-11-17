// Messages in flight as flows sharing link bandwidth max-min fairly
// (progressive filling). A start or end re-shares only the flows connected to
// it through shared links.
#pragma once

#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sim/topology.hpp"
#include "sim/types.hpp"

namespace cta::sim {

// Nanoseconds for `bytes` at `rate` bytes/ns, rounded up; the slack keeps an
// exact quotient from rounding past itself.
inline Time transfer_time(double bytes, double rate) {
  return bytes <= 0 ? 0 : static_cast<Time>(std::ceil(bytes / rate - 1e-6));
}

class FlowNetwork {
 public:
  using Changes = std::vector<std::pair<std::uint32_t, Time>>;  // flow, its new finish time

  FlowNetwork(Topology& topology, std::size_t max_flows);

  void start(std::uint32_t id, std::uint32_t src, std::uint32_t dst, std::uint64_t bytes, Time now, Changes& changes);
  void finish(std::uint32_t id, Time now, Changes& changes);
  bool due(std::uint32_t id, Time t) const { return flows_[id].active && flows_[id].finish == t; }

  Time latency(std::uint32_t src, std::uint32_t dst) { return route(src, dst).latency; }
  double bottleneck(std::uint32_t src, std::uint32_t dst) { return route(src, dst).bottleneck; }

  std::vector<std::uint64_t> link_bytes;
  std::vector<Time> link_busy;
  std::vector<std::uint32_t> link_max_flows;
  std::uint64_t reshares = 0;

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
  void reshare(const Flow& changed, Time now, Changes& changes);

  Topology& topology_;
  std::vector<Flow> flows_;
  std::vector<std::vector<std::uint32_t>> link_flows_;
  std::unordered_map<std::uint64_t, Route> routes_;
  std::vector<std::uint32_t> pool_;  // the links of every route

  // Scratch of reshare: marks are valid for the current epoch.
  std::uint32_t epoch_ = 0;
  std::vector<std::uint32_t> link_epoch_, flow_epoch_, link_version_, unfrozen_;
  std::vector<double> capacity_left_;
  std::vector<char> frozen_;
  std::vector<std::uint32_t> links_, flows_seen_, stack_;
  std::vector<Time> busy_since_;
};

}  // namespace cta::sim
