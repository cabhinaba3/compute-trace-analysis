// The graph the simulator runs: every rank's ops, in rank order, and the
// dependency edges between them.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "sim/types.hpp"

namespace cta::sim {

template <typename T>
using Owned = std::vector<T>;
template <typename T>
using View = std::span<const T>;

template <template <typename> class Column>
struct Graph {
  Column<std::uint8_t> kind;
  Column<std::uint32_t> rank;
  Column<std::uint16_t> cpu, nic;
  Column<std::int64_t> duration;  // a calc's length, or a collective's recorded one
  Column<std::uint64_t> bytes;
  Column<std::int32_t> peer;  // a send's destination or a recv's source
  Column<std::uint64_t> tag;
  Column<std::int8_t> coll_type;
  Column<std::int32_t> coll_group, coll_root;  // the root is a position in the group
  Column<std::uint32_t> edge_src, edge_dst;    // edge_dst waits for edge_src
  Column<std::uint8_t> edge_kind;
  // Lowered graphs only: the input op each op came from, its Role, and its
  // action in the collective's schedule.
  Column<std::uint32_t> origin;
  Column<std::uint8_t> role;
  Column<std::uint32_t> action;

  std::size_t size() const { return kind.size(); }
  std::size_t edges() const { return edge_src.size(); }
  OpKind op_kind(std::size_t i) const { return static_cast<OpKind>(kind[i]); }

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
using GraphColumns = Graph<Owned>;
using GraphView = Graph<View>;

// Collective groups: group g is members[offsets[g] .. offsets[g + 1]), in the
// order its algorithms number them.
struct Groups {
  View<std::uint32_t> offsets, members;

  std::size_t count() const { return offsets.empty() ? 0 : offsets.size() - 1; }
  View<std::uint32_t> of(std::size_t g) const { return members.subspan(offsets[g], offsets[g + 1] - offsets[g]); }
};

// Throws std::invalid_argument for mismatched columns, unknown kinds, edges or
// groups out of range, or more than 2^31 ops.
void check(const GraphView& g, const Groups& groups);

struct Successors {
  std::vector<std::uint64_t> offsets;
  std::vector<std::uint32_t> targets;
  std::vector<std::uint8_t> kinds;
};
Successors build_successors(const GraphView& g);

// The ops on or behind a dependency cycle.
std::uint64_t count_unordered(const GraphView& g, const Successors& succ);

// The k-th collective of a group on each member is instance base[group] + k.
struct Instances {
  std::vector<std::uint32_t> seq;  // per op; kNone for non-collectives
  std::vector<std::uint64_t> base;  // per group, then the total
  std::uint64_t id(std::size_t op, const GraphView& g) const {
    return base[static_cast<std::size_t>(g.coll_group[op])] + seq[op];
  }
};
Instances number_instances(const GraphView& g, std::size_t num_groups);

}  // namespace cta::sim
