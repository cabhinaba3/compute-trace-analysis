#include "sim/topology.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "util/util.hpp"

namespace cta::sim {
namespace {

using u32 = std::uint32_t;

u32 ceil_div(u32 a, u32 b) { return (a + b - 1) / b; }
Time latency_of(double ns) { return static_cast<Time>(std::llround(ns)); }

class SwitchTopology : public Topology {
 public:
  SwitchTopology(const TopologyParams& p, u32 ranks, u32 per_node) : Topology(ranks, per_node) {
    add_vertices(ranks);
    const u32 sw = add_vertices(1);
    attach(p, std::vector<u32>(ranks, sw));
  }

 protected:
  void fabric_route(u32, u32, std::vector<u32>&) const override {}
};

class LeafSpineTopology : public Topology {
 public:
  LeafSpineTopology(const TopologyParams& p, u32 ranks, u32 per_node) : Topology(ranks, per_node) {
    hosts_ = p.hosts_per_leaf ? p.hosts_per_leaf : 16;
    spines_ = p.spines ? p.spines : 8;
    if (p.oversubscription <= 0) throw std::invalid_argument("fat_tree oversubscription must be positive");
    add_vertices(ranks);
    const u32 leaves = ceil_div(ranks, hosts_);
    const u32 leaf0 = add_vertices(leaves), spine0 = add_vertices(spines_);
    std::vector<u32> leaf_of(ranks);
    for (u32 r = 0; r < ranks; ++r) leaf_of[r] = leaf0 + r / hosts_;
    attach(p, leaf_of);
    const double uplink = p.link_bandwidth * hosts_ / (spines_ * p.oversubscription);
    up0_ = static_cast<u32>(links_.size());
    for (u32 l = 0; l < leaves; ++l)
      for (u32 s = 0; s < spines_; ++s) add_link(uplink, latency_of(p.link_latency), LinkKind::kFabric, leaf0 + l, spine0 + s);
    down0_ = static_cast<u32>(links_.size());
    for (u32 l = 0; l < leaves; ++l)
      for (u32 s = 0; s < spines_; ++s) add_link(uplink, latency_of(p.link_latency), LinkKind::kFabric, spine0 + s, leaf0 + l);
  }

 protected:
  void fabric_route(u32 src, u32 dst, std::vector<u32>& out) const override {
    const u32 ls = src / hosts_, ld = dst / hosts_;
    if (ls == ld) return;
    const u32 s = static_cast<u32>(mix64((static_cast<std::uint64_t>(src) << 32) | dst) % spines_);
    out.push_back(up0_ + ls * spines_ + s);
    out.push_back(down0_ + ld * spines_ + s);
  }

 private:
  u32 hosts_, spines_, up0_ = 0, down0_ = 0;
};

class TorusTopology : public Topology {
 public:
  TorusTopology(const TopologyParams& p, u32 ranks, u32 per_node, bool ring) : Topology(ranks, per_node) {
    const u32 nodes = ceil_div(ranks, per_node);
    dims_ = p.dims;
    if (dims_.empty()) {
      if (!ring) throw std::invalid_argument("a torus needs its dims");
      dims_ = {nodes};
    }
    if (ring && dims_.size() != 1) throw std::invalid_argument("a ring has one dimension");
    u32 routers = 1;
    for (u32 d : dims_) {
      if (d == 0) throw std::invalid_argument("torus dims must be positive");
      routers *= d;
    }
    if (routers < nodes)
      throw std::invalid_argument("torus of " + std::to_string(routers) + " routers is too small for " +
                                  std::to_string(nodes) + " nodes");
    add_vertices(ranks);
    router0_ = add_vertices(routers);
    std::vector<u32> router_of(ranks);
    for (u32 r = 0; r < ranks; ++r) router_of[r] = router0_ + r / per_node;
    attach(p, router_of);
    const u32 nd = static_cast<u32>(dims_.size());
    link_of_.assign(static_cast<std::size_t>(routers) * nd * 2, kNone);
    for (u32 v = 0; v < routers; ++v)
      for (u32 d = 0; d < nd; ++d)
        for (u32 dir = 0; dir < 2; ++dir) {
          if (dims_[d] < 2) continue;
          link_of_[(static_cast<std::size_t>(v) * nd + d) * 2 + dir] =
              add_link(p.link_bandwidth, latency_of(p.link_latency), LinkKind::kFabric, router0_ + v,
                       router0_ + step(v, d, dir));
        }
  }

 protected:
  void fabric_route(u32 src, u32 dst, std::vector<u32>& out) const override {
    u32 v = src / per_node_;
    const u32 target = dst / per_node_;
    const u32 nd = static_cast<u32>(dims_.size());
    u32 stride = 1;
    for (u32 d = 0; d < nd; ++d) {
      const u32 size = dims_[d];
      const u32 from = (v / stride) % size, to = (target / stride) % size;
      const u32 forward = (to + size - from) % size;
      const bool plus = forward <= size - forward;
      for (u32 hop = 0, hops = plus ? forward : size - forward; hop < hops; ++hop) {
        out.push_back(link_of_[(static_cast<std::size_t>(v) * nd + d) * 2 + (plus ? 0 : 1)]);
        v = step(v, d, plus ? 0 : 1);
      }
      stride *= size;
    }
  }

 private:
  // The router one hop from v along dimension d (dir 0: +1, dir 1: -1).
  u32 step(u32 v, u32 d, u32 dir) const {
    u32 stride = 1;
    for (u32 k = 0; k < d; ++k) stride *= dims_[k];
    const u32 size = dims_[d], c = (v / stride) % size;
    const u32 next = dir == 0 ? (c + 1) % size : (c + size - 1) % size;
    return v + (next - c) * stride;  // unsigned wrap-around gives the right result
  }

  std::vector<u32> dims_;
  u32 router0_ = 0;
  std::vector<u32> link_of_;  // (router, dim, dir) -> link
};

class DragonflyTopology : public Topology {
 public:
  DragonflyTopology(const TopologyParams& p, u32 ranks, u32 per_node) : Topology(ranks, per_node) {
    hosts_ = p.hosts_per_router ? p.hosts_per_router : 4;
    per_group_ = p.routers_per_group ? p.routers_per_group : 8;
    const u32 routers_used = ceil_div(ranks, hosts_);
    groups_ = ceil_div(routers_used, per_group_);
    add_vertices(ranks);
    router0_ = add_vertices(groups_ * per_group_);
    std::vector<u32> router_of(ranks);
    for (u32 r = 0; r < ranks; ++r) router_of[r] = router0_ + r / hosts_;
    attach(p, router_of);
    const Time lat = latency_of(p.link_latency);
    local_.assign(static_cast<std::size_t>(groups_) * per_group_ * per_group_, kNone);
    for (u32 g = 0; g < groups_; ++g)
      for (u32 i = 0; i < per_group_; ++i)
        for (u32 j = 0; j < per_group_; ++j)
          if (i != j)
            local_[(static_cast<std::size_t>(g) * per_group_ + i) * per_group_ + j] =
                add_link(p.link_bandwidth, lat, LinkKind::kFabric, router0_ + g * per_group_ + i,
                         router0_ + g * per_group_ + j);
    const double global_bw = p.global_bandwidth > 0 ? p.global_bandwidth : p.link_bandwidth;
    global_.assign(static_cast<std::size_t>(groups_) * groups_, kNone);
    for (u32 g = 0; g < groups_; ++g)
      for (u32 h = 0; h < groups_; ++h)
        if (g != h)
          global_[static_cast<std::size_t>(g) * groups_ + h] =
              add_link(global_bw, lat, LinkKind::kGlobal, router0_ + gateway(g, h), router0_ + gateway(h, g));
  }

 protected:
  void fabric_route(u32 src, u32 dst, std::vector<u32>& out) const override {
    const u32 rs = src / hosts_, rd = dst / hosts_;
    if (rs == rd) return;
    const u32 gs = rs / per_group_, gd = rd / per_group_;
    if (gs == gd) {
      out.push_back(local(rs, rd));
      return;
    }
    const u32 x = gateway(gs, gd), y = gateway(gd, gs);
    if (x != rs) out.push_back(local(rs, x));
    out.push_back(global_[static_cast<std::size_t>(gs) * groups_ + gd]);
    if (y != rd) out.push_back(local(y, rd));
  }

 private:
  // The router of group g that holds g's global link to group h.
  u32 gateway(u32 g, u32 h) const { return g * per_group_ + (h < g ? h : h - 1) % per_group_; }
  u32 local(u32 a, u32 b) const {
    const u32 g = a / per_group_;
    return local_[(static_cast<std::size_t>(g) * per_group_ + a % per_group_) * per_group_ + b % per_group_];
  }

  u32 hosts_, per_group_, groups_ = 0, router0_ = 0;
  std::vector<u32> local_, global_;
};

}  // namespace

std::uint32_t Topology::add_link(double capacity, Time latency, LinkKind kind, std::uint32_t from, std::uint32_t to) {
  if (!(capacity > 0)) throw std::invalid_argument("link bandwidths must be positive");
  links_.push_back({capacity, latency, kind, from, to});
  return static_cast<std::uint32_t>(links_.size() - 1);
}

std::uint32_t Topology::add_vertices(std::uint32_t n) {
  const std::uint32_t first = vertices_;
  vertices_ += n;
  return first;
}

void Topology::attach(const TopologyParams& p, const std::vector<std::uint32_t>& switch_of_rank) {
  const Time lat = latency_of(p.link_latency);
  nic_up_.resize(ranks_);
  nic_down_.resize(ranks_);
  for (std::uint32_t r = 0; r < ranks_; ++r) {
    nic_up_[r] = add_link(p.link_bandwidth, lat, LinkKind::kNicUp, r, switch_of_rank[r]);
    nic_down_[r] = add_link(p.link_bandwidth, lat, LinkKind::kNicDown, switch_of_rank[r], r);
  }
  if (per_node_ > 1 && p.intra_bandwidth > 0) {
    intra_ = true;
    const std::uint32_t node0 = add_vertices(ceil_div(ranks_, per_node_));
    const Time intra_lat = latency_of(p.intra_latency);
    intra_up_.resize(ranks_);
    intra_down_.resize(ranks_);
    for (std::uint32_t r = 0; r < ranks_; ++r) {
      intra_up_[r] = add_link(p.intra_bandwidth, intra_lat, LinkKind::kIntraUp, r, node0 + r / per_node_);
      intra_down_[r] = add_link(p.intra_bandwidth, intra_lat, LinkKind::kIntraDown, node0 + r / per_node_, r);
    }
  }
}

void Topology::route(std::uint32_t src, std::uint32_t dst, std::vector<std::uint32_t>& out) const {
  if (intra_ && src / per_node_ == dst / per_node_) {
    out.push_back(intra_up_[src]);
    out.push_back(intra_down_[dst]);
    return;
  }
  out.push_back(nic_up_[src]);
  fabric_route(src, dst, out);
  out.push_back(nic_down_[dst]);
}

std::unique_ptr<Topology> Topology::make(const TopologyParams& params, std::uint32_t ranks,
                                         std::uint32_t ranks_per_node, const std::vector<double>& nic_scale,
                                         std::uint64_t seed) {
  const std::uint32_t per_node = ranks_per_node == 0 ? 1 : ranks_per_node;
  std::unique_ptr<Topology> t;
  if (params.kind == "switch") t = std::make_unique<SwitchTopology>(params, ranks, per_node);
  else if (params.kind == "fat_tree") t = std::make_unique<LeafSpineTopology>(params, ranks, per_node);
  else if (params.kind == "torus") t = std::make_unique<TorusTopology>(params, ranks, per_node, false);
  else if (params.kind == "ring") t = std::make_unique<TorusTopology>(params, ranks, per_node, true);
  else if (params.kind == "dragonfly") t = std::make_unique<DragonflyTopology>(params, ranks, per_node);
  else throw std::invalid_argument("unknown topology '" + params.kind + "'");

  for (std::uint32_t r = 0; r < ranks && r < nic_scale.size(); ++r) {
    if (!(nic_scale[r] > 0)) throw std::invalid_argument("NIC scale factors must be positive");
    t->links_[t->nic_up_[r]].capacity *= nic_scale[r];
    t->links_[t->nic_down_[r]].capacity *= nic_scale[r];
  }
  if (params.degraded_fraction > 0) {
    if (!(params.degraded_factor > 0)) throw std::invalid_argument("degraded_factor must be positive");
    for (std::uint32_t l = 0; l < t->links_.size(); ++l) {
      Link& link = t->links_[l];
      if (link.kind == LinkKind::kIntraUp || link.kind == LinkKind::kIntraDown) continue;
      const double u = static_cast<double>(mix64(seed ^ mix64(l + 0x51ed27)) >> 11) * 0x1.0p-53;
      if (u < params.degraded_fraction) link.capacity *= params.degraded_factor;
    }
  }
  return t;
}

}  // namespace cta::sim
