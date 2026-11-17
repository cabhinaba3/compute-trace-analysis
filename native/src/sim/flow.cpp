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

FlowNetwork::FlowNetwork(Topology& topology, std::size_t max_flows) : topology_(topology) {
  const std::size_t links = topology.links().size();
  link_bytes.assign(links, 0);
  link_busy.assign(links, 0);
  link_max_flows.assign(links, 0);
  flows_.resize(max_flows);
  link_flows_.resize(links);
  link_epoch_.assign(links, 0);
  link_version_.assign(links, 0);
  unfrozen_.assign(links, 0);
  capacity_left_.assign(links, 0);
  busy_since_.assign(links, 0);
  flow_epoch_.assign(max_flows, 0);
  frozen_.assign(max_flows, 0);
}

const FlowNetwork::Route& FlowNetwork::route(std::uint32_t src, std::uint32_t dst) {
  auto [it, fresh] = routes_.try_emplace((std::uint64_t{src} << 32) | dst);
  if (fresh) {
    Route& r = it->second;
    r = {static_cast<std::uint32_t>(pool_.size()), 0, 0, std::numeric_limits<double>::infinity()};
    topology_.route(src, dst, pool_);
    r.size = static_cast<std::uint32_t>(pool_.size() - r.begin);
    for (std::uint32_t k = r.begin; k < r.begin + r.size; ++k) {
      r.latency += topology_.links()[pool_[k]].latency;
      r.bottleneck = std::min(r.bottleneck, topology_.links()[pool_[k]].capacity);
    }
  }
  return it->second;
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
    link_bytes[l] += bytes;
    link_max_flows[l] = std::max(link_max_flows[l], static_cast<std::uint32_t>(link_flows_[l].size()));
  }
  reshare(f, now, changes);
}

void FlowNetwork::finish(std::uint32_t id, Time now, Changes& changes) {
  Flow& f = flows_[id];
  f.active = false;
  for (std::uint32_t k = f.route_begin; k < f.route_begin + f.route_size; ++k) {
    auto& on_link = link_flows_[pool_[k]];
    std::erase(on_link, id);
    if (on_link.empty()) link_busy[pool_[k]] += now - busy_since_[pool_[k]];
  }
  reshare(f, now, changes);
}

void FlowNetwork::reshare(const Flow& changed, Time now, Changes& changes) {
  ++reshares;
  ++epoch_;
  links_.clear();
  flows_seen_.clear();
  stack_.clear();
  auto visit_link = [&](std::uint32_t l) {
    if (link_epoch_[l] != epoch_) {
      link_epoch_[l] = epoch_;
      stack_.push_back(l);
    }
  };
  for (std::uint32_t k = changed.route_begin; k < changed.route_begin + changed.route_size; ++k) visit_link(pool_[k]);
  while (!stack_.empty()) {
    const std::uint32_t l = stack_.back();
    stack_.pop_back();
    links_.push_back(l);
    for (std::uint32_t id : link_flows_[l]) {
      if (flow_epoch_[id] == epoch_) continue;
      flow_epoch_[id] = epoch_;
      flows_seen_.push_back(id);
      for (std::uint32_t k = flows_[id].route_begin; k < flows_[id].route_begin + flows_[id].route_size; ++k)
        visit_link(pool_[k]);
    }
  }
  for (std::uint32_t id : flows_seen_) {
    Flow& f = flows_[id];
    f.remaining = std::max(0.0, f.remaining - f.rate * static_cast<double>(now - f.last));
    f.last = now;
    frozen_[id] = 0;
  }
  // The link with the smallest fair share fixes its unfixed flows' rate.
  std::priority_queue<Share, std::vector<Share>, std::greater<Share>> shares;
  auto offer = [&](std::uint32_t l) {
    ++link_version_[l];
    if (unfrozen_[l] > 0) shares.push({std::max(0.0, capacity_left_[l]) / unfrozen_[l], l, link_version_[l]});
  };
  for (std::uint32_t l : links_) {
    capacity_left_[l] = topology_.links()[l].capacity;
    unfrozen_[l] = static_cast<std::uint32_t>(link_flows_[l].size());
    offer(l);
  }
  while (!shares.empty()) {
    const Share top = shares.top();
    shares.pop();
    if (top.version != link_version_[top.link] || unfrozen_[top.link] == 0) continue;
    for (std::uint32_t id : link_flows_[top.link]) {
      if (frozen_[id]) continue;
      frozen_[id] = 1;
      Flow& f = flows_[id];
      f.rate = top.share;
      for (std::uint32_t k = f.route_begin; k < f.route_begin + f.route_size; ++k) {
        capacity_left_[pool_[k]] -= f.rate;
        --unfrozen_[pool_[k]];
        offer(pool_[k]);
      }
    }
  }
  for (std::uint32_t id : flows_seen_) {
    Flow& f = flows_[id];
    const Time finish = f.rate > 0 ? now + transfer_time(f.remaining, f.rate) : kNever;
    if (finish != f.finish) {
      f.finish = finish;
      changes.emplace_back(id, finish);
    }
  }
}

}  // namespace cta::sim
