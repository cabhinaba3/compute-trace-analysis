#include "sim/flow.hpp"

#include <algorithm>
#include <limits>
#include <queue>

namespace cta::sim {
namespace {

struct Share {
  double share;
  std::uint32_t link, version;
  bool operator>(const Share& o) const { return share != o.share ? share > o.share : link > o.link; }
};

}  // namespace

FlowNetwork::FlowNetwork(const Topology& topology, std::size_t max_flows) : topology_(topology) {
  const std::size_t links = topology.links().size();
  flows_.resize(max_flows);
  link_flows_.resize(links);
  link_epoch_.assign(links, 0);
  link_version_.assign(links, 0);
  unfrozen_.assign(links, 0);
  capacity_left_.assign(links, 0);
  flow_epoch_.assign(max_flows, 0);
  frozen_.assign(max_flows, 0);
  link_bytes_.assign(links, 0);
  link_busy_.assign(links, 0);
  busy_since_.assign(links, 0);
  link_max_flows_.assign(links, 0);
}

const FlowNetwork::Route& FlowNetwork::route(std::uint32_t src, std::uint32_t dst) {
  const std::uint64_t key = (static_cast<std::uint64_t>(src) << 32) | dst;
  auto it = routes_.find(key);
  if (it != routes_.end()) return it->second;
  Route r{static_cast<std::uint32_t>(pool_.size()), 0, 0, std::numeric_limits<double>::infinity()};
  topology_.route(src, dst, pool_);
  r.size = static_cast<std::uint32_t>(pool_.size() - r.begin);
  for (std::uint32_t k = r.begin; k < r.begin + r.size; ++k) {
    const Link& link = topology_.links()[pool_[k]];
    r.latency += link.latency;
    r.bottleneck = std::min(r.bottleneck, link.capacity);
  }
  return routes_.emplace(key, r).first->second;
}

void FlowNetwork::start(std::uint32_t id, std::uint32_t src, std::uint32_t dst, std::uint64_t bytes, Time now,
                        Changes& changes) {
  const Route& r = route(src, dst);
  Flow& f = flows_[id];
  f = Flow{r.begin, r.size, static_cast<double>(bytes), 0, now, kNever, true};
  for (std::uint32_t k = r.begin; k < r.begin + r.size; ++k) {
    const std::uint32_t l = pool_[k];
    if (link_flows_[l].empty()) busy_since_[l] = now;
    link_flows_[l].push_back(id);
    link_bytes_[l] += bytes;
    link_max_flows_[l] = std::max(link_max_flows_[l], static_cast<std::uint32_t>(link_flows_[l].size()));
  }
  reshare(r.begin, r.size, now, changes);
}

void FlowNetwork::finish(std::uint32_t id, Time now, Changes& changes) {
  Flow& f = flows_[id];
  f.active = false;
  for (std::uint32_t k = f.route_begin; k < f.route_begin + f.route_size; ++k) {
    const std::uint32_t l = pool_[k];
    auto& on_link = link_flows_[l];
    const auto it = std::find(on_link.begin(), on_link.end(), id);
    if (it != on_link.end()) {
      *it = on_link.back();
      on_link.pop_back();
    }
    if (on_link.empty()) link_busy_[l] += now - busy_since_[l];
  }
  reshare(f.route_begin, f.route_size, now, changes);
}

void FlowNetwork::reshare(std::uint32_t route_begin, std::uint32_t route_size, Time now, Changes& changes) {
  ++reshares_;
  ++epoch_;
  component_links_.clear();
  component_flows_.clear();
  stack_.clear();
  for (std::uint32_t k = route_begin; k < route_begin + route_size; ++k) {
    const std::uint32_t l = pool_[k];
    if (link_epoch_[l] != epoch_) {
      link_epoch_[l] = epoch_;
      stack_.push_back(l);
    }
  }
  // The links and flows connected to the route through shared links.
  while (!stack_.empty()) {
    const std::uint32_t l = stack_.back();
    stack_.pop_back();
    component_links_.push_back(l);
    for (std::uint32_t id : link_flows_[l]) {
      if (flow_epoch_[id] == epoch_) continue;
      flow_epoch_[id] = epoch_;
      component_flows_.push_back(id);
      const Flow& f = flows_[id];
      for (std::uint32_t k = f.route_begin; k < f.route_begin + f.route_size; ++k) {
        const std::uint32_t l2 = pool_[k];
        if (link_epoch_[l2] != epoch_) {
          link_epoch_[l2] = epoch_;
          stack_.push_back(l2);
        }
      }
    }
  }
  for (std::uint32_t id : component_flows_) {
    Flow& f = flows_[id];
    f.remaining = std::max(0.0, f.remaining - f.rate * static_cast<double>(now - f.last));
    f.last = now;
    frozen_[id] = 0;
  }
  // Progressive filling: the link offering the smallest fair share fixes the
  // rate of its unfixed flows, which then use up capacity on their other links.
  std::priority_queue<Share, std::vector<Share>, std::greater<Share>> shares;
  for (std::uint32_t l : component_links_) {
    capacity_left_[l] = topology_.links()[l].capacity;
    unfrozen_[l] = static_cast<std::uint32_t>(link_flows_[l].size());
    ++link_version_[l];
    if (unfrozen_[l] > 0) shares.push({capacity_left_[l] / unfrozen_[l], l, link_version_[l]});
  }
  while (!shares.empty()) {
    const Share top = shares.top();
    shares.pop();
    if (top.version != link_version_[top.link] || unfrozen_[top.link] == 0) continue;
    const double rate = std::max(0.0, top.share);
    for (std::uint32_t id : link_flows_[top.link]) {
      if (frozen_[id]) continue;
      frozen_[id] = 1;
      Flow& f = flows_[id];
      f.rate = rate;
      for (std::uint32_t k = f.route_begin; k < f.route_begin + f.route_size; ++k) {
        const std::uint32_t l = pool_[k];
        capacity_left_[l] -= rate;
        --unfrozen_[l];
        ++link_version_[l];
        if (unfrozen_[l] > 0) shares.push({std::max(0.0, capacity_left_[l]) / unfrozen_[l], l, link_version_[l]});
      }
    }
  }
  for (std::uint32_t id : component_flows_) {
    Flow& f = flows_[id];
    const Time finish = f.rate > 0 ? now + transfer_time(f.remaining, f.rate) : kNever;
    if (finish != f.finish) {
      f.finish = finish;
      changes.emplace_back(id, finish);
    }
  }
}

}  // namespace cta::sim
