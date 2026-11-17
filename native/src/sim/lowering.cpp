#include "sim/lowering.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace cta::sim {

std::uint32_t node_size_of(std::span<const std::uint32_t> members, std::uint32_t ranks_per_node) {
  if (members.empty()) return 0;
  if (ranks_per_node <= 1) return 1;
  std::uint32_t run = 0, size = 0;
  std::unordered_set<std::uint32_t> seen;
  for (std::size_t i = 0; i <= members.size(); ++i) {
    const bool same = i < members.size() && i > 0 && members[i] / ranks_per_node == members[i - 1] / ranks_per_node;
    if (i > 0 && !same) {  // a run ends at i
      if (size == 0) size = run;
      if (run != size || !seen.insert(members[i - 1] / ranks_per_node).second) return 0;
      run = 0;
    }
    ++run;
  }
  return size;
}

GraphColumns lower_collectives(const GraphView& g, const Groups& groups, const CollectiveParams& params,
                               std::uint32_t ranks_per_node, LoweringStats& stats) {
  groups.check();
  g.check(groups.count());
  const Instances instances = number_instances(g, groups.count());

  // Per group, lazily: rank -> position, and members per node.
  std::vector<std::unordered_map<std::uint32_t, std::uint32_t>> positions(groups.count());
  std::vector<std::uint32_t> node_sizes(groups.count(), kNone);
  auto position_of = [&](std::size_t group, std::uint32_t rank) {
    auto& map = positions[group];
    if (map.empty()) {
      const auto members = groups.of(group);
      for (std::uint32_t i = 0; i < members.size(); ++i) map.emplace(members[i], i);
      node_sizes[group] = node_size_of(members, ranks_per_node);
    }
    const auto it = map.find(rank);
    if (it == map.end())
      throw std::invalid_argument("rank " + std::to_string(rank) + " runs a collective of group " +
                                  std::to_string(group) + " but is not one of its members");
    return it->second;
  };

  GraphColumns out;
  out.reserve(g.size() * 2, g.edges() * 2);
  auto add_op = [&](OpKind kind, std::size_t from, std::int64_t duration, std::uint64_t bytes, std::int32_t peer,
                    std::uint64_t tag, Role role, std::uint32_t action) {
    out.kind.push_back(static_cast<std::uint8_t>(kind));
    out.rank.push_back(g.rank[from]);
    out.cpu.push_back(g.cpu[from]);
    out.nic.push_back(g.nic[from]);
    out.duration.push_back(duration);
    out.bytes.push_back(bytes);
    out.peer.push_back(peer);
    out.tag.push_back(tag);
    out.coll_type.push_back(-1);
    out.coll_group.push_back(-1);
    out.coll_root.push_back(-1);
    out.origin.push_back(static_cast<std::uint32_t>(from));
    out.role.push_back(static_cast<std::uint8_t>(role));
    out.action.push_back(action);
    return static_cast<std::uint32_t>(out.kind.size() - 1);
  };
  auto add_edge = [&](std::uint32_t src, std::uint32_t dst, DepKind kind) {
    out.edge_src.push_back(src);
    out.edge_dst.push_back(dst);
    out.edge_kind.push_back(static_cast<std::uint8_t>(kind));
  };

  // The new op standing for each input op's start and for its end.
  std::vector<std::uint32_t> starts(g.size()), ends(g.size());
  Schedule schedule;
  std::vector<std::uint32_t> placed;
  for (std::size_t i = 0; i < g.size(); ++i) {
    if (g.op_kind(i) != OpKind::kColl) {
      const std::uint32_t op = static_cast<std::uint32_t>(out.kind.size());
      out.kind.push_back(g.kind[i]);
      out.rank.push_back(g.rank[i]);
      out.cpu.push_back(g.cpu[i]);
      out.nic.push_back(g.nic[i]);
      out.duration.push_back(g.duration[i]);
      out.bytes.push_back(g.bytes[i]);
      out.peer.push_back(g.peer[i]);
      out.tag.push_back(g.tag[i]);
      out.coll_type.push_back(g.coll_type[i]);
      out.coll_group.push_back(g.coll_group[i]);
      out.coll_root.push_back(g.coll_root[i]);
      out.origin.push_back(static_cast<std::uint32_t>(i));
      out.role.push_back(static_cast<std::uint8_t>(Role::kOriginal));
      out.action.push_back(kNone);
      starts[i] = ends[i] = op;
      continue;
    }
    const auto group = static_cast<std::size_t>(g.coll_group[i]);
    const auto members = groups.of(group);
    const std::uint32_t me = position_of(group, g.rank[i]);
    const auto type = static_cast<CollType>(g.coll_type[i]);
    const std::uint32_t size = static_cast<std::uint32_t>(members.size());
    const std::uint32_t node_size = node_sizes[group];
    bool fell_back = false;
    CollSpec spec;
    spec.type = type;
    spec.algorithm = resolve_algorithm(type, params.algorithms[static_cast<std::size_t>(type)], size,
                                       node_size, &fell_back);
    spec.size = size;
    spec.bytes = g.bytes[i];
    spec.root = g.coll_root[i] > 0 ? static_cast<std::uint32_t>(g.coll_root[i]) % size : 0;
    spec.node_size = node_size == 0 ? 1 : node_size;
    spec.chunks = params.chunks;
    spec.reduce = params.reduce_ns_per_byte > 0;
    build_schedule(spec, me, schedule);
    ++stats.collectives;
    stats.fallbacks += fell_back ? 1 : 0;

    const std::uint64_t instance = instances.id(i, g);
    if (instance >= (std::uint64_t{1} << 43)) throw std::invalid_argument("too many collective instances");
    const std::uint32_t entry = add_op(OpKind::kNop, i, 0, 0, -1, 0, Role::kEntry, kNone);
    placed.resize(schedule.actions.size());
    for (std::uint32_t a = 0; a < schedule.actions.size(); ++a) {
      const Action& act = schedule.actions[a];
      if (act.seq >= (1u << 20)) throw std::invalid_argument("more than 2^20 messages between two members of a collective");
      const std::uint64_t tag = kCollectiveTagBit | (instance << 20) | act.seq;
      const auto peer = static_cast<std::int32_t>(members[act.peer]);
      switch (act.kind) {
        case ActionKind::kSend:
          placed[a] = add_op(OpKind::kSend, i, 0, act.bytes, peer, tag, Role::kStep, a);
          ++stats.messages;
          stats.bytes += act.bytes;
          break;
        case ActionKind::kRecv:
          placed[a] = add_op(OpKind::kRecv, i, 0, act.bytes, peer, tag, Role::kStep, a);
          break;
        case ActionKind::kReduce:
          placed[a] = add_op(OpKind::kCalc, i,
                             std::llround(static_cast<double>(act.bytes) * params.reduce_ns_per_byte), act.bytes,
                             -1, 0, Role::kStep, a);
          break;
      }
      const auto deps = schedule.deps_of(a);
      if (deps.empty()) add_edge(entry, placed[a], DepKind::kRequires);
      for (std::uint32_t d : deps) add_edge(placed[d], placed[a], DepKind::kRequires);
    }
    const std::uint32_t exit = add_op(OpKind::kNop, i, 0, 0, -1, 0, Role::kExit, kNone);
    const std::vector<std::uint32_t> last = params.data_completion ? schedule.outputs : schedule.sinks();
    if (last.empty()) add_edge(entry, exit, DepKind::kRequires);
    for (std::uint32_t a : last) add_edge(placed[a], exit, DepKind::kRequires);
    starts[i] = entry;
    ends[i] = exit;
  }

  for (std::size_t e = 0; e < g.edges(); ++e) {
    const auto kind = static_cast<DepKind>(g.edge_kind[e]);
    const std::uint32_t src = kind == DepKind::kRequires ? ends[g.edge_src[e]] : starts[g.edge_src[e]];
    add_edge(src, starts[g.edge_dst[e]], kind);
  }
  return out;
}

}  // namespace cta::sim
