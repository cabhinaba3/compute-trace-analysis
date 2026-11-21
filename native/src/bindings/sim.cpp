#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include "bindings/bindings.hpp"
#include "bindings/convert.hpp"
#include "sim/collectives.hpp"
#include "sim/engine.hpp"
#include "sim/lowering.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace cta::bindings {
namespace {

// A view of a NumPy array of exactly T (no conversion, so the view stays
// valid while the caller holds the array).
template <typename T>
std::span<const T> as_span(const nb::handle& value, const std::string& name) {
  nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu> array;
  if (!nb::try_cast(value, array, false))
    throw std::invalid_argument("'" + name + "' must be a contiguous 1-D NumPy array of " +
                                (std::is_signed_v<T> ? "int" : "uint") + std::to_string(8 * sizeof(T)));
  return {array.data(), array.shape(0)};
}

sim::GraphView view_of(const nb::dict& d) {
  sim::GraphView g;
  g.for_each_column([&](const char* name, auto& span) {
    using T = typename std::remove_reference_t<decltype(span)>::value_type;
    const std::string_view key(name);
    if (d.contains(name)) span = as_span<T>(d[name], name);
    else if (key != "origin" && key != "role" && key != "action")
      throw std::invalid_argument("graph has no column '" + std::string(key) + "'");
  });
  return g;
}

sim::Groups groups_of(const nb::handle& offsets, const nb::handle& members) {
  return {as_span<std::uint32_t>(offsets, "group_offsets"), as_span<std::uint32_t>(members, "group_members")};
}

sim::SimParams params_of(const nb::dict& d) {
  sim::SimParams p;
  for (auto [k, v] : d) {
    const auto key = nb::cast<std::string>(k);
    bool found = false;
    p.for_each_param([&](std::string_view name, auto& member) {
      if (found || name != key) return;
      found = true;
      member = nb::cast<std::remove_reference_t<decltype(member)>>(v);
    });
    if (!found) throw std::invalid_argument("unknown simulation parameter '" + key + "'");
  }
  p.check();
  return p;
}

sim::CollType coll_type_of(int t) {
  if (t < 0 || t >= sim::kNumCollTypes) throw std::invalid_argument("unknown collective type " + std::to_string(t));
  return static_cast<sim::CollType>(t);
}

// Every member's actions of one collective, and their dependencies.
struct ScheduleColumns {
  std::vector<std::uint32_t> member, action, peer, seq;
  std::vector<std::uint8_t> kind, output;
  std::vector<std::uint64_t> bytes;
  std::vector<std::uint32_t> dep_member, dep_action, dep_on;

  template <typename Visit>
  void for_each_column(Visit&& visit) {
    visit("member", member);
    visit("action", action);
    visit("kind", kind);
    visit("peer", peer);
    visit("bytes", bytes);
    visit("seq", seq);
    visit("output", output);
    visit("dep_member", dep_member);
    visit("dep_action", dep_action);
    visit("dep_on", dep_on);
  }
};

}  // namespace

void bind_sim(nb::module_& m) {
  m.def(
      "simulate",
      [](const nb::dict& graph, const nb::handle& offsets, const nb::handle& members, const nb::dict& params) {
        const sim::GraphView g = view_of(graph);
        const sim::Groups groups = groups_of(offsets, members);
        const sim::SimParams p = params_of(params);
        sim::SimColumns result;
        {
          nb::gil_scoped_release release;
          result = sim::simulate(g, groups, p);
        }
        return to_dict(std::move(result));
      },
      "graph"_a, "group_offsets"_a, "group_members"_a, "params"_a);

  m.def(
      "lower_collectives",
      [](const nb::dict& graph, const nb::handle& offsets, const nb::handle& members, const nb::dict& params) {
        const sim::GraphView g = view_of(graph);
        const sim::Groups groups = groups_of(offsets, members);
        const sim::SimParams p = params_of(params);
        sim::LoweringStats stats;
        sim::GraphColumns lowered;
        {
          nb::gil_scoped_release release;
          lowered = sim::lower_collectives(g, groups, p, stats);
        }
        return nb::make_tuple(to_dict(std::move(lowered)), to_dict(std::move(stats)));
      },
      "graph"_a, "group_offsets"_a, "group_members"_a, "params"_a);

  m.def(
      "collective_schedule",
      [](int coll_type, const std::string& algorithm, std::uint32_t size, std::uint64_t bytes, std::uint32_t root,
         std::uint32_t node_size, std::uint32_t chunks, bool reduce, bool all_to_all_per_peer) {
        const sim::CollSpec spec{coll_type_of(coll_type), sim::parse_algorithm(algorithm), size, bytes, root,
                                 node_size, chunks, reduce, all_to_all_per_peer};
        ScheduleColumns out;
        sim::Schedule s;
        for (std::uint32_t member = 0; member < size; ++member) {
          sim::build_schedule(spec, member, s);
          std::vector<std::uint8_t> is_output(s.actions.size(), 0);
          for (std::uint32_t a : s.outputs) is_output[a] = 1;
          for (std::uint32_t a = 0; a < s.actions.size(); ++a) {
            out.member.push_back(member);
            out.action.push_back(a);
            out.kind.push_back(static_cast<std::uint8_t>(s.actions[a].kind));
            out.peer.push_back(s.actions[a].peer);
            out.bytes.push_back(s.actions[a].bytes);
            out.seq.push_back(s.actions[a].seq);
            out.output.push_back(is_output[a]);
            for (std::uint32_t d : s.deps_of(a)) {
              out.dep_member.push_back(member);
              out.dep_action.push_back(a);
              out.dep_on.push_back(d);
            }
          }
        }
        return to_dict(std::move(out));
      },
      "coll_type"_a, "algorithm"_a, "size"_a, "bytes"_a, "root"_a = 0, "node_size"_a = 1, "chunks"_a = 1,
      "reduce"_a = false, "all_to_all_per_peer"_a = false);

  m.def(
      "collective_times",
      [](int coll_type, std::uint64_t bytes, const std::vector<std::uint32_t>& members, const nb::dict& params,
         std::uint32_t root) {
        sim::CollSpec spec;
        spec.type = coll_type_of(coll_type);
        spec.bytes = bytes;
        spec.root = members.empty() ? 0 : root % static_cast<std::uint32_t>(members.size());
        return sim::collective_finish_times(spec, members, params_of(params));
      },
      "coll_type"_a, "bytes"_a, "members"_a, "params"_a, "root"_a = 0);

  m.def(
      "algorithm_supported",
      [](int coll_type, const std::string& algorithm, std::uint32_t size, std::uint32_t node_size) {
        return sim::supports(coll_type_of(coll_type), sim::parse_algorithm(algorithm), size, node_size);
      },
      "coll_type"_a, "algorithm"_a, "size"_a, "node_size"_a = 1);
}

}  // namespace cta::bindings
