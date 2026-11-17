#include "sim/lowering.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace cta::sim {

std::array<Algorithm, kNumCollTypes> algorithms_of(const SimParams& p) {
  std::array<Algorithm, kNumCollTypes> out{};
  for (int t = 0; t < kNumCollTypes; ++t) out[t] = parse_algorithm(p.algorithms[t]);
  return out;
}

std::uint32_t node_size_of(std::span<const std::uint32_t> members, std::uint32_t ranks_per_node) {
  if (ranks_per_node <= 1) return 1;
  std::uint32_t run = 0, size = 0;
  std::unordered_set<std::uint32_t> seen;
  for (std::size_t i = 0; i <= members.size(); ++i) {
    if (i > 0 && (i == members.size() || members[i] / ranks_per_node != members[i - 1] / ranks_per_node)) {
      if (size == 0) size = run;
      if (run != size || !seen.insert(members[i - 1] / ranks_per_node).second) return 0;
      run = 0;
    }
    ++run;
  }
  return size;
}

GraphColumns lower_collectives(const GraphView& g, const Groups& groups, const SimParams& params,
                               LoweringStats& stats) {
  check(g, groups);
  params.check();
  const auto algorithms = algorithms_of(params);
  const Instances instances = number_instances(g, groups.count());
  std::vector<std::unordered_map<std::uint32_t, std::uint32_t>> positions(groups.count());
  std::vector<std::uint32_t> node_sizes(groups.count(), 0);
  for (std::size_t grp = 0; grp < groups.count(); ++grp) {
    const auto members = groups.of(grp);
    for (std::uint32_t k = 0; k < members.size(); ++k) positions[grp].emplace(members[k], k);
    node_sizes[grp] = node_size_of(members, params.ranks_per_node);
  }

  GraphColumns out;
  auto add = [&](OpKind kind, std::size_t from, std::int64_t duration, std::uint64_t bytes, std::int32_t peer,
                 std::uint64_t tag, Role role, std::uint32_t action) {
    const bool original = role == Role::kOriginal;
    out.kind.push_back(static_cast<std::uint8_t>(kind));
    out.rank.push_back(g.rank[from]);
    out.cpu.push_back(g.cpu[from]);
    out.nic.push_back(g.nic[from]);
    out.duration.push_back(duration);
    out.bytes.push_back(bytes);
    out.peer.push_back(peer);
    out.tag.push_back(tag);
    out.coll_type.push_back(original ? g.coll_type[from] : -1);
    out.coll_group.push_back(original ? g.coll_group[from] : -1);
    out.coll_root.push_back(original ? g.coll_root[from] : -1);
    out.origin.push_back(static_cast<std::uint32_t>(from));
    out.role.push_back(static_cast<std::uint8_t>(role));
    out.action.push_back(action);
    return static_cast<std::uint32_t>(out.kind.size() - 1);
  };
  auto edge = [&](std::uint32_t src, std::uint32_t dst, DepKind kind) {
    out.edge_src.push_back(src);
    out.edge_dst.push_back(dst);
    out.edge_kind.push_back(static_cast<std::uint8_t>(kind));
  };

  // The new ops standing for each input op's start and end.
  std::vector<std::uint32_t> starts(g.size()), ends(g.size());
  Schedule schedule;
  std::vector<std::uint32_t> placed;
  for (std::size_t i = 0; i < g.size(); ++i) {
    if (g.op_kind(i) != OpKind::kColl) {
      starts[i] = ends[i] =
          add(g.op_kind(i), i, g.duration[i], g.bytes[i], g.peer[i], g.tag[i], Role::kOriginal, kNone);
      continue;
    }
    const auto grp = static_cast<std::size_t>(g.coll_group[i]);
    const auto members = groups.of(grp);
    const auto me = positions[grp].find(g.rank[i]);
    if (me == positions[grp].end())
      throw std::invalid_argument("rank " + std::to_string(g.rank[i]) + " is not a member of collective group " +
                                  std::to_string(grp));
    CollSpec spec;
    spec.type = static_cast<CollType>(g.coll_type[i]);
    spec.size = static_cast<std::uint32_t>(members.size());
    bool fell_back = false;
    spec.algorithm = resolve_algorithm(spec.type, algorithms[static_cast<std::size_t>(spec.type)], spec.size,
                                       node_sizes[grp], &fell_back);
    spec.bytes = g.bytes[i];
    spec.root = g.coll_root[i] > 0 ? static_cast<std::uint32_t>(g.coll_root[i]) % spec.size : 0;
    spec.node_size = std::max<std::uint32_t>(1, node_sizes[grp]);
    spec.chunks = std::max<std::uint32_t>(1, params.chunks);
    spec.reduce = params.reduce_ns_per_byte > 0;
    spec.all_to_all_per_peer = params.all_to_all_size == "per_peer";
    build_schedule(spec, me->second, schedule);
    ++stats.collectives;
    stats.fallbacks += fell_back;

    const std::uint64_t instance = instances.id(i, g);
    if (instance >= (std::uint64_t{1} << 43)) throw std::invalid_argument("too many collective instances");
    const std::uint32_t entry = add(OpKind::kNop, i, 0, 0, -1, 0, Role::kEntry, kNone);
    placed.resize(schedule.actions.size());
    for (std::uint32_t a = 0; a < schedule.actions.size(); ++a) {
      const Action& act = schedule.actions[a];
      if (act.seq >= (1u << 20)) throw std::invalid_argument("more than 2^20 messages between two members");
      const std::uint64_t tag = kCollectiveTagBit | (instance << 20) | act.seq;
      const auto peer = static_cast<std::int32_t>(members[act.peer]);
      if (act.kind == ActionKind::kReduce) {
        placed[a] = add(OpKind::kCalc, i, std::llround(static_cast<double>(act.bytes) * params.reduce_ns_per_byte),
                        act.bytes, -1, 0, Role::kStep, a);
      } else {
        const bool send = act.kind == ActionKind::kSend;
        placed[a] = add(send ? OpKind::kSend : OpKind::kRecv, i, 0, act.bytes, peer, tag, Role::kStep, a);
        stats.messages += send;
        stats.bytes += send ? act.bytes : 0;
      }
      const auto deps = schedule.deps_of(a);
      if (deps.empty()) edge(entry, placed[a], DepKind::kRequires);
      for (std::uint32_t d : deps) edge(placed[d], placed[a], DepKind::kRequires);
    }
    const std::uint32_t exit = add(OpKind::kNop, i, 0, 0, -1, 0, Role::kExit, kNone);
    const auto last = params.completion == "outputs" ? schedule.outputs : schedule.sinks();
    if (last.empty()) edge(entry, exit, DepKind::kRequires);
    for (std::uint32_t a : last) edge(placed[a], exit, DepKind::kRequires);
    starts[i] = entry;
    ends[i] = exit;
  }
  for (std::size_t e = 0; e < g.edges(); ++e) {
    const auto kind = static_cast<DepKind>(g.edge_kind[e]);
    edge(kind == DepKind::kRequires ? ends[g.edge_src[e]] : starts[g.edge_src[e]], starts[g.edge_dst[e]], kind);
  }
  return out;
}

}  // namespace cta::sim
