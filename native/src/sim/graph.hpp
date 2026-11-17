// The graph the simulator runs: one row per op, all ranks together, and the
// dependency edges between ops. Python builds it from the GOAL or Chakra
// tables (cta.sim.graph); lowering builds a new one from it.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "sim/types.hpp"

namespace cta::sim {

// An owning graph, as lowering returns it.
struct GraphColumns {
  // Ops. `duration` is a calc's length or a collective's recorded length; `peer`
  // is a send's destination or a recv's source rank; `coll_group` indexes the
  // groups passed alongside and `coll_root` is a position in that group.
  std::vector<std::uint8_t> kind;
  std::vector<std::uint32_t> rank;
  std::vector<std::uint16_t> cpu, nic;
  std::vector<std::int64_t> duration;
  std::vector<std::uint64_t> bytes;
  std::vector<std::int32_t> peer;
  std::vector<std::uint64_t> tag;
  std::vector<std::int8_t> coll_type;
  std::vector<std::int32_t> coll_group, coll_root;

  // Edges: `edge_dst` waits for `edge_src` (see DepKind for `edge_kind`).
  std::vector<std::uint32_t> edge_src, edge_dst;
  std::vector<std::uint8_t> edge_kind;

  // Provenance of a lowered graph: the input op each op came from, its Role,
  // and the index of its action in the collective's schedule (kNone if none).
  std::vector<std::uint32_t> origin;
  std::vector<std::uint8_t> role;
  std::vector<std::uint32_t> action;

  void reserve(std::size_t ops, std::size_t edges);

  // Calls visit(name, member) for every column, under the names Python sees.
  template <typename Visit>
  void for_each_column(Visit&& visit) {
    visit("kind", kind);
    visit("rank", rank);
    visit("cpu", cpu);
    visit("nic", nic);
    visit("duration", duration);
    visit("bytes", bytes);
    visit("peer", peer);
    visit("tag", tag);
    visit("coll_type", coll_type);
    visit("coll_group", coll_group);
    visit("coll_root", coll_root);
    visit("edge_src", edge_src);
    visit("edge_dst", edge_dst);
    visit("edge_kind", edge_kind);
    visit("origin", origin);
    visit("role", role);
    visit("action", action);
  }
};

// A read-only view of a graph, over NumPy arrays or a GraphColumns.
struct GraphView {
  std::span<const std::uint8_t> kind;
  std::span<const std::uint32_t> rank;
  std::span<const std::uint16_t> cpu, nic;
  std::span<const std::int64_t> duration;
  std::span<const std::uint64_t> bytes;
  std::span<const std::int32_t> peer;
  std::span<const std::uint64_t> tag;
  std::span<const std::int8_t> coll_type;
  std::span<const std::int32_t> coll_group, coll_root;
  std::span<const std::uint32_t> edge_src, edge_dst;
  std::span<const std::uint8_t> edge_kind;

  static GraphView of(const GraphColumns& g);
  std::size_t size() const { return kind.size(); }
  std::size_t edges() const { return edge_src.size(); }
  OpKind op_kind(std::size_t i) const { return static_cast<OpKind>(kind[i]); }

  // Throws std::invalid_argument unless the columns have matching lengths, the
  // kinds and collective types are known, the edges point at ops, every
  // collective names a group, and the graph has at most 2^30 ops (the engine
  // keeps two flag bits in op ids).
  void check(std::size_t num_groups) const;
};

// Collective groups: group g is members[offsets[g] .. offsets[g + 1]), in the
// order its algorithms number them (position 0 is the first rank of a ring).
struct Groups {
  std::span<const std::uint32_t> offsets, members;

  std::size_t count() const { return offsets.empty() ? 0 : offsets.size() - 1; }
  std::span<const std::uint32_t> of(std::size_t g) const {
    return members.subspan(offsets[g], offsets[g + 1] - offsets[g]);
  }
  // Throws std::invalid_argument for offsets that are not a valid partition.
  void check() const;
};

// Each op's outgoing edges, grouped by source.
struct Successors {
  std::vector<std::uint64_t> offsets;  // ops + 1 entries
  std::vector<std::uint32_t> targets;
  std::vector<std::uint8_t> kinds;

  std::size_t begin(std::uint32_t op) const { return static_cast<std::size_t>(offsets[op]); }
  std::size_t end(std::uint32_t op) const { return static_cast<std::size_t>(offsets[op + 1]); }
};

Successors build_successors(const GraphView& g);

// The number of ops on or behind a dependency cycle: those Kahn's algorithm
// cannot order. 0 for an acyclic graph.
std::uint64_t count_unordered(const GraphView& g, const Successors& succ);

// The position of each op among the collectives of its (rank, group), so the
// k-th collective of a group on every member is instance k of that group.
// Also returns the first instance id of each group: instance = base[g] + k.
struct Instances {
  std::vector<std::uint32_t> seq;    // per op; kNone for non-collectives
  std::vector<std::uint64_t> base;   // per group, plus the total at the end
  std::uint64_t id(std::size_t op, const GraphView& g) const {
    return base[static_cast<std::size_t>(g.coll_group[op])] + seq[op];
  }
};
Instances number_instances(const GraphView& g, std::size_t num_groups);

}  // namespace cta::sim
