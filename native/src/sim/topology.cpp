#include "sim/topology.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "util/util.hpp"

namespace cta::sim {
namespace {

using u32 = std::uint32_t;

u32 ceil_div(u32 a, u32 b) { return (a + b - 1) / b; }

}  // namespace

u32 Topology::add_vertices(u32 n) {
  vertices_ += n;
  out_.resize(vertices_);
  in_.resize(vertices_);
  return vertices_ - n;
}

u32 Topology::add_link(double capacity, Time latency, LinkKind kind, u32 from, u32 to) {
  if (!(capacity > 0)) throw std::invalid_argument("link bandwidths must be positive");
  const auto id = static_cast<u32>(links_.size());
  links_.push_back({capacity, latency, kind, from, to});
  out_[from].push_back(id);
  in_[to].push_back(id);
  return id;
}

Topology::Topology(const SimParams& p, u32 ranks) : per_node_(std::max<u32>(1, p.ranks_per_node)) {
  const Time lat = std::llround(p.link_latency);
  const double bw = p.link_bandwidth;
  const u32 nodes = ceil_div(ranks, per_node_);
  add_vertices(ranks);
  switch_of_.resize(ranks);
  auto both_ways = [&](double capacity, LinkKind kind, u32 a, u32 b) {
    add_link(capacity, lat, kind, a, b);
    add_link(capacity, lat, kind, b, a);
  };

  if (p.topology == "switch") {
    const u32 crossbar = add_vertices(1);
    for (u32 r = 0; r < ranks; ++r) switch_of_[r] = crossbar;
  } else if (p.topology == "fat_tree") {
    if (p.hosts_per_leaf == 0 || p.spines == 0 || !(p.oversubscription > 0))
      throw std::invalid_argument("fat_tree needs hosts_per_leaf, spines and oversubscription");
    const u32 leaves = ceil_div(ranks, p.hosts_per_leaf);
    const u32 leaf0 = add_vertices(leaves), spine0 = add_vertices(p.spines);
    for (u32 r = 0; r < ranks; ++r) switch_of_[r] = leaf0 + r / p.hosts_per_leaf;
    const double uplink = bw * p.hosts_per_leaf / (p.spines * p.oversubscription);
    for (u32 l = 0; l < leaves; ++l)
      for (u32 s = 0; s < p.spines; ++s) both_ways(uplink, LinkKind::kFabric, leaf0 + l, spine0 + s);
  } else if (p.topology == "torus" || p.topology == "ring") {
    std::vector<u32> dims = p.dims;
    if (dims.empty() && p.topology == "ring") dims = {nodes};
    u32 routers = 1;
    for (u32 d : dims) routers *= d;
    if (dims.empty() || routers < nodes)
      throw std::invalid_argument("torus dims of " + std::to_string(routers) + " routers do not cover " +
                                  std::to_string(nodes) + " nodes");
    const u32 r0 = add_vertices(routers);
    for (u32 r = 0; r < ranks; ++r) switch_of_[r] = r0 + r / per_node_;
    for (u32 v = 0, stride = 1; v < routers; ++v, stride = 1)
      for (u32 d : dims) {
        const u32 c = (v / stride) % d;
        if (d > 1) both_ways(bw, LinkKind::kFabric, r0 + v, r0 + v - c * stride + (c + 1) % d * stride);
        stride *= d;
      }
  } else if (p.topology == "dragonfly") {
    const u32 a = p.routers_per_group, h = p.hosts_per_router;
    if (a == 0 || h == 0) throw std::invalid_argument("dragonfly needs hosts_per_router and routers_per_group");
    const u32 groups = ceil_div(ceil_div(ranks, h), a), r0 = add_vertices(groups * a);
    for (u32 r = 0; r < ranks; ++r) switch_of_[r] = r0 + r / h;
    for (u32 g = 0; g < groups; ++g)
      for (u32 i = 0; i < a; ++i)
        for (u32 j = i + 1; j < a; ++j) both_ways(bw, LinkKind::kFabric, r0 + g * a + i, r0 + g * a + j);
    // Group g reaches group k > g through router k - 1 of g and router g of k.
    const double global = p.global_bandwidth > 0 ? p.global_bandwidth : bw;
    for (u32 g = 0; g < groups; ++g)
      for (u32 k = g + 1; k < groups; ++k)
        both_ways(global, LinkKind::kGlobal, r0 + g * a + (k - 1) % a, r0 + k * a + g % a);
  } else {
    throw std::invalid_argument("unknown topology '" + p.topology + "'");
  }

  for (u32 r = 0; r < ranks; ++r) {
    const double scale = r < p.nic_scale.size() ? p.nic_scale[r] : 1.0;
    if (!(scale > 0)) throw std::invalid_argument("NIC scale factors must be positive");
    nic_up_.push_back(add_link(bw * scale, lat, LinkKind::kNicUp, r, switch_of_[r]));
    nic_down_.push_back(add_link(bw * scale, lat, LinkKind::kNicDown, switch_of_[r], r));
  }
  if (per_node_ > 1 && p.intra_bandwidth > 0) {
    const u32 node0 = add_vertices(nodes);
    const Time intra_lat = std::llround(p.intra_latency);
    for (u32 r = 0; r < ranks; ++r) {
      intra_up_.push_back(add_link(p.intra_bandwidth, intra_lat, LinkKind::kIntraUp, r, node0 + r / per_node_));
      intra_down_.push_back(add_link(p.intra_bandwidth, intra_lat, LinkKind::kIntraDown, node0 + r / per_node_, r));
    }
  }
  for (u32 l = 0; l < links_.size() && p.degraded_fraction > 0; ++l) {
    if (links_[l].kind == LinkKind::kIntraUp || links_[l].kind == LinkKind::kIntraDown) continue;
    if (static_cast<double>(mix64(p.seed ^ mix64(l + 0x51ed27)) >> 11) * 0x1.0p-53 < p.degraded_fraction)
      links_[l].capacity *= p.degraded_factor;
  }
}

const std::vector<u32>& Topology::hops_to(u32 target) {
  auto [it, fresh] = hops_.try_emplace(target);
  if (!fresh) return it->second;
  std::vector<u32>& hops = it->second;
  hops.assign(vertices_, kNone);
  hops[target] = 0;
  std::vector<u32> queue{target};
  for (std::size_t q = 0; q < queue.size(); ++q)
    for (u32 l : in_[queue[q]]) {
      const u32 from = links_[l].from;
      if (from >= switch_of_.size() && hops[from] == kNone) {  // through switches only
        hops[from] = hops[queue[q]] + 1;
        queue.push_back(from);
      }
    }
  return hops;
}

void Topology::route(u32 src, u32 dst, std::vector<u32>& out) {
  if (!intra_up_.empty() && src / per_node_ == dst / per_node_) {
    out.push_back(intra_up_[src]);
    out.push_back(intra_down_[dst]);
    return;
  }
  out.push_back(nic_up_[src]);
  const u32 target = switch_of_[dst];
  const auto& hops = hops_to(target);
  std::uint64_t hash = mix64((std::uint64_t{src} << 32) | dst);
  std::vector<u32> next;
  for (u32 v = switch_of_[src]; v != target; hash = mix64(hash)) {
    next.clear();
    for (u32 l : out_[v])
      if (hops[links_[l].to] + 1 == hops[v]) next.push_back(l);
    if (next.empty()) throw std::logic_error("switch " + std::to_string(v) + " cannot reach its destination");
    const u32 l = next[hash % next.size()];
    out.push_back(l);
    v = links_[l].to;
  }
  out.push_back(nic_down_[dst]);
}

}  // namespace cta::sim
