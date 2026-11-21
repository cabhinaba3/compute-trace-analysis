#include "sim/graph.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>

namespace cta::sim {

void check(const GraphView& g, const Groups& groups) {
  auto bad = [](const std::string& what) { throw std::invalid_argument("simulation graph: " + what); };
  const std::size_t n = g.size();
  if (n >= (std::size_t{1} << 31)) bad("more than 2^31 ops");
  for (std::size_t length : {g.rank.size(), g.cpu.size(), g.nic.size(), g.duration.size(), g.bytes.size(),
                             g.peer.size(), g.tag.size(), g.coll_type.size(), g.coll_group.size(), g.coll_root.size()})
    if (length != n) bad("op columns differ in length");
  if (g.edge_dst.size() != g.edges() || g.edge_kind.size() != g.edges()) bad("edge columns differ in length");
  if (!groups.offsets.empty() && (groups.offsets.front() != 0 || groups.offsets.back() != groups.members.size()))
    bad("group offsets must run from 0 to the number of members");
  for (std::size_t i = 0; i < n; ++i) {
    const auto kind = g.op_kind(i);
    if (g.kind[i] > static_cast<std::uint8_t>(OpKind::kNop)) bad("unknown op kind " + std::to_string(g.kind[i]));
    if (kind == OpKind::kColl &&
        (g.coll_type[i] < 0 || g.coll_type[i] >= kNumCollTypes || g.coll_group[i] < 0 ||
         static_cast<std::size_t>(g.coll_group[i]) >= groups.count()))
      bad("collective op " + std::to_string(i) + " has an unknown type or group");
    if ((kind == OpKind::kSend || kind == OpKind::kRecv) && g.peer[i] < 0)
      bad("send or recv op " + std::to_string(i) + " has no peer");
    if (g.duration[i] < 0) bad("negative duration");
  }
  for (std::size_t e = 0; e < g.edges(); ++e)
    if (g.edge_src[e] >= n || g.edge_dst[e] >= n || g.edge_kind[e] > 1) bad("edge out of range");
}

Successors build_successors(const GraphView& g) {
  Successors s;
  s.offsets.assign(g.size() + 1, 0);
  for (std::uint32_t src : g.edge_src) ++s.offsets[src + 1];
  for (std::size_t i = 0; i < g.size(); ++i) s.offsets[i + 1] += s.offsets[i];
  s.targets.resize(g.edges());
  s.kinds.resize(g.edges());
  std::vector<std::uint64_t> fill(s.offsets.begin(), s.offsets.end() - 1);
  for (std::size_t e = 0; e < g.edges(); ++e) {
    const std::uint64_t at = fill[g.edge_src[e]]++;
    s.targets[at] = g.edge_dst[e];
    s.kinds[at] = g.edge_kind[e];
  }
  return s;
}

std::uint64_t count_unordered(const GraphView& g, const Successors& succ) {
  std::vector<std::uint32_t> pending(g.size(), 0), ready;
  for (std::uint32_t dst : g.edge_dst) ++pending[dst];
  for (std::uint32_t i = 0; i < g.size(); ++i)
    if (pending[i] == 0) ready.push_back(i);
  std::uint64_t ordered = 0;
  while (!ready.empty()) {
    const std::uint32_t op = ready.back();
    ready.pop_back();
    ++ordered;
    for (std::uint64_t k = succ.offsets[op]; k < succ.offsets[op + 1]; ++k)
      if (--pending[succ.targets[k]] == 0) ready.push_back(succ.targets[k]);
  }
  return g.size() - ordered;
}

Instances number_instances(const GraphView& g, std::size_t num_groups) {
  Instances out;
  out.seq.assign(g.size(), kNone);
  std::unordered_map<std::uint64_t, std::uint32_t> next;  // (rank, group) -> collectives so far
  std::vector<std::uint64_t> most(num_groups, 0);
  for (std::size_t i = 0; i < g.size(); ++i) {
    if (g.op_kind(i) != OpKind::kColl) continue;
    const auto group = static_cast<std::uint32_t>(g.coll_group[i]);
    std::uint32_t& k = next[(std::uint64_t{g.rank[i]} << 32) | group];
    out.seq[i] = k++;
    most[group] = std::max<std::uint64_t>(most[group], k);
  }
  out.base.assign(num_groups + 1, 0);
  for (std::size_t grp = 0; grp < num_groups; ++grp) out.base[grp + 1] = out.base[grp] + most[grp];
  return out;
}

}  // namespace cta::sim
