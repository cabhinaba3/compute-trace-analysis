// Topologies for the flow model: directed links with a bandwidth and a
// latency, and the route of links between any two ranks.
//
// Every rank has a NIC link up into the fabric and one down from it. With
// ranks_per_node > 1 and an intra-node bandwidth, ranks of one node also have
// links to and from a node switch (NVSwitch-like), and traffic between them
// stays there. The fabric is one of:
//
//   switch     one crossbar: NIC up, NIC down.
//   fat_tree   leaf-spine: hosts_per_leaf ranks per leaf, every leaf linked to
//              every spine; the spine is chosen by hashing (src, dst) (ECMP).
//              Leaf-spine links carry link_bandwidth * hosts_per_leaf /
//              (spines * oversubscription), so oversubscription 1 is full
//              bisection.
//   torus      routers on a torus of `dims` (wrap-around links both ways);
//              the ranks of node k attach to router k; dimension-order routing
//              the shorter way round.
//   ring       a torus of one dimension (dims defaults to the node count).
//   dragonfly  hosts_per_router ranks per router, routers_per_group routers per
//              group, all-to-all local links in a group and one global link
//              between every pair of groups; minimal routing.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "sim/params.hpp"
#include "sim/types.hpp"

namespace cta::sim {

enum class LinkKind : std::uint8_t {
  kNicUp = 0,
  kNicDown = 1,
  kIntraUp = 2,
  kIntraDown = 3,
  kFabric = 4,  // switch-to-switch (leaf-spine, torus, dragonfly local)
  kGlobal = 5,  // dragonfly global
};

struct Link {
  double capacity;  // bytes per ns
  Time latency;
  LinkKind kind;
  std::uint32_t from, to;  // vertices: ranks are 0..R-1, switches follow
};

class Topology {
 public:
  virtual ~Topology() = default;

  // Throws std::invalid_argument for an unknown kind or a shape too small for
  // `ranks` ranks. nic_scale and degraded links scale the capacities.
  static std::unique_ptr<Topology> make(const TopologyParams& params, std::uint32_t ranks,
                                        std::uint32_t ranks_per_node, const std::vector<double>& nic_scale,
                                        std::uint64_t seed);

  const std::vector<Link>& links() const { return links_; }
  std::uint32_t vertices() const { return vertices_; }
  // Appends the links from rank src to rank dst (src != dst) to `out`.
  void route(std::uint32_t src, std::uint32_t dst, std::vector<std::uint32_t>& out) const;

 protected:
  Topology(std::uint32_t ranks, std::uint32_t ranks_per_node) : ranks_(ranks), per_node_(ranks_per_node) {}
  std::uint32_t add_link(double capacity, Time latency, LinkKind kind, std::uint32_t from, std::uint32_t to);
  std::uint32_t add_vertices(std::uint32_t n);
  // Builds the NIC and node-switch links; vertex of the rank's fabric switch per rank.
  void attach(const TopologyParams& p, const std::vector<std::uint32_t>& switch_of_rank);
  // The links between the fabric switches of two ranks on different nodes.
  virtual void fabric_route(std::uint32_t src, std::uint32_t dst, std::vector<std::uint32_t>& out) const = 0;

  std::uint32_t ranks_, per_node_;
  std::uint32_t vertices_ = 0;
  std::vector<Link> links_;
  std::vector<std::uint32_t> nic_up_, nic_down_, intra_up_, intra_down_;
  bool intra_ = false;
};

}  // namespace cta::sim
