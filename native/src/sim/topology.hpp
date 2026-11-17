// The flow model's network: directed links with a bandwidth and a latency.
//
// Ranks are vertices 0..R-1, each with a NIC link up to its switch and one
// down. With ranks_per_node > 1 and an intra-node bandwidth, ranks of a node
// also link to a node switch that carries the traffic between them. Fabrics:
//   switch     one crossbar
//   fat_tree   leaf-spine; leaf uplinks carry link_bandwidth * hosts_per_leaf /
//              (spines * oversubscription)
//   torus      routers on `dims`, node k on router k; ring is one dimension
//   dragonfly  all-to-all routers in a group, one global link per group pair
// Routes between switches are shortest paths; where several next hops are
// equally short, a hash of (src, dst) picks one (ECMP).
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "sim/params.hpp"
#include "sim/types.hpp"

namespace cta::sim {

enum class LinkKind : std::uint8_t { kNicUp, kNicDown, kIntraUp, kIntraDown, kFabric, kGlobal };

struct Link {
  double capacity;  // bytes per ns
  Time latency;
  LinkKind kind;
  std::uint32_t from, to;
};

class Topology {
 public:
  // Throws std::invalid_argument for an unknown kind or a shape too small.
  Topology(const SimParams& p, std::uint32_t ranks);

  const std::vector<Link>& links() const { return links_; }
  // Appends the links from rank src to rank dst to `out`.
  void route(std::uint32_t src, std::uint32_t dst, std::vector<std::uint32_t>& out);

 private:
  std::uint32_t add_vertices(std::uint32_t n);
  std::uint32_t add_link(double capacity, Time latency, LinkKind kind, std::uint32_t from, std::uint32_t to);
  const std::vector<std::uint32_t>& hops_to(std::uint32_t target);

  std::uint32_t per_node_;
  std::uint32_t vertices_ = 0;
  std::vector<Link> links_;
  std::vector<std::vector<std::uint32_t>> out_, in_;  // links leaving and entering each vertex
  std::vector<std::uint32_t> switch_of_, nic_up_, nic_down_, intra_up_, intra_down_;
  std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> hops_;  // target -> hops from each vertex
};

}  // namespace cta::sim
