#include "sim/graph.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace cta::sim {
namespace {

std::uint64_t pair_key(std::uint64_t a, std::uint64_t b) { return (a << 32) | b; }

}  // namespace

void GraphColumns::reserve(std::size_t ops, std::size_t edges) {
  kind.reserve(ops);
  rank.reserve(ops);
  cpu.reserve(ops);
  nic.reserve(ops);
  duration.reserve(ops);
  bytes.reserve(ops);
  peer.reserve(ops);
  tag.reserve(ops);
  coll_type.reserve(ops);
  coll_group.reserve(ops);
  coll_root.reserve(ops);
  origin.reserve(ops);
  role.reserve(ops);
  action.reserve(ops);
  edge_src.reserve(edges);
  edge_dst.reserve(edges);
  edge_kind.reserve(edges);
}

GraphView GraphView::of(const GraphColumns& g) {
  return {g.kind,      g.rank,       g.cpu,       g.nic,      g.duration,
          g.bytes,     g.peer,       g.tag,       g.coll_type, g.coll_group,
          g.coll_root, g.edge_src,   g.edge_dst,  g.edge_kind};
}

void GraphView::check(std::size_t num_groups) const {
  const std::size_t n = size();
  auto bad = [](const std::string& what) { throw std::invalid_argument("simulation graph: " + what); };
  if (rank.size() != n || cpu.size() != n || nic.size() != n || duration.size() != n ||
      bytes.size() != n || peer.size() != n || tag.size() != n || coll_type.size() != n ||
      coll_group.size() != n || coll_root.size() != n)
    bad("op columns differ in length");
  if (edge_dst.size() != edges() || edge_kind.size() != edges()) bad("edge columns differ in length");
  if (n >= (std::size_t{1} << 30)) bad("more than 2^30 ops");
  for (std::size_t i = 0; i < n; ++i) {
    if (kind[i] > static_cast<std::uint8_t>(OpKind::kNop)) bad("unknown op kind " + std::to_string(kind[i]));
    if (op_kind(i) == OpKind::kColl) {
      if (coll_type[i] < 0 || coll_type[i] >= kNumCollTypes)
        bad("unknown collective type " + std::to_string(coll_type[i]));
      if (coll_group[i] < 0 || static_cast<std::size_t>(coll_group[i]) >= num_groups)
        bad("collective op " + std::to_string(i) + " has no group");
      if (duration[i] < 0) bad("negative collective duration");
    }
    if (op_kind(i) == OpKind::kCalc && duration[i] < 0) bad("negative calc duration");
    if ((op_kind(i) == OpKind::kSend || op_kind(i) == OpKind::kRecv) && peer[i] < 0)
      bad("send or recv op " + std::to_string(i) + " has no peer");
  }
  for (std::size_t e = 0; e < edges(); ++e) {
    if (edge_src[e] >= n || edge_dst[e] >= n) bad("edge points past the last op");
    if (edge_kind[e] > static_cast<std::uint8_t>(DepKind::kIrequires)) bad("unknown edge kind");
  }
}

void Groups::check() const {
  if (offsets.empty()) return;
  if (offsets.front() != 0 || offsets.back() != members.size())
    throw std::invalid_argument("group offsets must run from 0 to the number of members");
  for (std::size_t g = 0; g + 1 < offsets.size(); ++g)
    if (offsets[g] > offsets[g + 1]) throw std::invalid_argument("group offsets must not decrease");
}

Successors build_successors(const GraphView& g) {
  const std::size_t n = g.size();
  Successors s;
  s.offsets.assign(n + 1, 0);
  for (std::size_t e = 0; e < g.edges(); ++e) ++s.offsets[g.edge_src[e] + 1];
  for (std::size_t i = 0; i < n; ++i) s.offsets[i + 1] += s.offsets[i];
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
  const std::size_t n = g.size();
  std::vector<std::uint32_t> pending(n, 0);
  for (std::size_t e = 0; e < g.edges(); ++e) ++pending[g.edge_dst[e]];
  std::vector<std::uint32_t> ready;
  for (std::uint32_t i = 0; i < n; ++i)
    if (pending[i] == 0) ready.push_back(i);
  std::uint64_t ordered = 0;
  while (!ready.empty()) {
    const std::uint32_t op = ready.back();
    ready.pop_back();
    ++ordered;
    for (std::size_t k = succ.begin(op); k < succ.end(op); ++k)
      if (--pending[succ.targets[k]] == 0) ready.push_back(succ.targets[k]);
  }
  return n - ordered;
}

Instances number_instances(const GraphView& g, std::size_t num_groups) {
  Instances out;
  out.seq.assign(g.size(), kNone);
  std::unordered_map<std::uint64_t, std::uint32_t> next;  // (rank, group) -> count so far
  std::vector<std::uint64_t> most(num_groups, 0);
  for (std::size_t i = 0; i < g.size(); ++i) {
    if (g.op_kind(i) != OpKind::kColl) continue;
    const auto group = static_cast<std::uint32_t>(g.coll_group[i]);
    std::uint32_t& k = next[pair_key(g.rank[i], group)];
    out.seq[i] = k++;
    if (k > most[group]) most[group] = k;
  }
  out.base.assign(num_groups + 1, 0);
  for (std::size_t grp = 0; grp < num_groups; ++grp) out.base[grp + 1] = out.base[grp] + most[grp];
  return out;
}

}  // namespace cta::sim
