#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <optional>
#include <stdexcept>
#include <string>
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

using sim::Algorithm;
using sim::CollType;

template <typename T>
using Array = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <typename T>
constexpr const char* dtype_name() {
  if constexpr (std::is_same_v<T, std::uint8_t>) return "uint8";
  else if constexpr (std::is_same_v<T, std::int8_t>) return "int8";
  else if constexpr (std::is_same_v<T, std::uint16_t>) return "uint16";
  else if constexpr (std::is_same_v<T, std::int32_t>) return "int32";
  else if constexpr (std::is_same_v<T, std::uint32_t>) return "uint32";
  else if constexpr (std::is_same_v<T, std::int64_t>) return "int64";
  else return "uint64";
}

// A view of a NumPy array of exactly type T; no conversion, so the view stays
// valid while the caller holds the array.
template <typename T>
std::span<const T> as_span(const nb::handle& value, const std::string& name) {
  Array<T> array;
  if (!nb::try_cast(value, array, false))
    throw std::invalid_argument("'" + name + "' must be a contiguous 1-D NumPy array of " + dtype_name<T>());
  return {array.data(), array.shape(0)};
}

template <typename T>
std::span<const T> column(const nb::dict& d, const char* name) {
  if (!d.contains(name)) throw std::invalid_argument(std::string("graph has no column '") + name + "'");
  return as_span<T>(d[name], name);
}

sim::GraphView view_of(const nb::dict& d) {
  sim::GraphView g;
  g.kind = column<std::uint8_t>(d, "kind");
  g.rank = column<std::uint32_t>(d, "rank");
  g.cpu = column<std::uint16_t>(d, "cpu");
  g.nic = column<std::uint16_t>(d, "nic");
  g.duration = column<std::int64_t>(d, "duration");
  g.bytes = column<std::uint64_t>(d, "bytes");
  g.peer = column<std::int32_t>(d, "peer");
  g.tag = column<std::uint64_t>(d, "tag");
  g.coll_type = column<std::int8_t>(d, "coll_type");
  g.coll_group = column<std::int32_t>(d, "coll_group");
  g.coll_root = column<std::int32_t>(d, "coll_root");
  g.edge_src = column<std::uint32_t>(d, "edge_src");
  g.edge_dst = column<std::uint32_t>(d, "edge_dst");
  g.edge_kind = column<std::uint8_t>(d, "edge_kind");
  return g;
}

sim::NetworkModel parse_model(const std::string& s) {
  if (s == "loggops") return sim::NetworkModel::kLogGOPS;
  if (s == "alpha_beta") return sim::NetworkModel::kAlphaBeta;
  if (s == "flow") return sim::NetworkModel::kFlow;
  throw std::invalid_argument("unknown network model '" + s + "' (loggops, alpha_beta, flow)");
}

// The parameters from a flat dict whose keys are the names below. Intra-node
// LogGOPS and alpha-beta parameters default to the inter-node ones.
sim::SimParams params_of(const nb::dict& d) {
  sim::SimParams p;
  struct Intra {
    std::optional<double> L, o, g, G, O, alpha, beta;
  } intra;
  for (auto [k, v] : d) {
    const std::string key = nb::cast<std::string>(k);
    auto num = [&] { return nb::cast<double>(v); };
    auto count = [&] { return nb::cast<std::uint64_t>(v); };
    auto text = [&] { return nb::cast<std::string>(v); };
    if (key == "model") p.model = parse_model(text());
    else if (key == "world_size") p.world_size = static_cast<std::uint32_t>(count());
    else if (key == "ranks_per_node") p.ranks_per_node = static_cast<std::uint32_t>(count());
    else if (key == "L") p.loggops.L = num();
    else if (key == "o") p.loggops.o = num();
    else if (key == "g") p.loggops.g = num();
    else if (key == "G") p.loggops.G = num();
    else if (key == "O") p.loggops.O = num();
    else if (key == "intra_L") intra.L = num();
    else if (key == "intra_o") intra.o = num();
    else if (key == "intra_g") intra.g = num();
    else if (key == "intra_G") intra.G = num();
    else if (key == "intra_O") intra.O = num();
    else if (key == "S") p.S = count();
    else if (key == "rendezvous") {
      const std::string r = text();
      if (r == "sync") p.rendezvous = sim::Rendezvous::kSync;
      else if (r == "rts_cts") p.rendezvous = sim::Rendezvous::kRtsCts;
      else throw std::invalid_argument("unknown rendezvous protocol '" + r + "' (sync, rts_cts)");
    } else if (key == "alpha") p.alpha = num();
    else if (key == "beta") p.beta = num();
    else if (key == "intra_alpha") intra.alpha = num();
    else if (key == "intra_beta") intra.beta = num();
    else if (key == "topology") p.topology.kind = text();
    else if (key == "link_bandwidth") p.topology.link_bandwidth = num();
    else if (key == "link_latency") p.topology.link_latency = num();
    else if (key == "intra_bandwidth") p.topology.intra_bandwidth = num();
    else if (key == "intra_latency") p.topology.intra_latency = num();
    else if (key == "global_bandwidth") p.topology.global_bandwidth = num();
    else if (key == "dims") p.topology.dims = nb::cast<std::vector<std::uint32_t>>(v);
    else if (key == "hosts_per_leaf") p.topology.hosts_per_leaf = static_cast<std::uint32_t>(count());
    else if (key == "spines") p.topology.spines = static_cast<std::uint32_t>(count());
    else if (key == "oversubscription") p.topology.oversubscription = num();
    else if (key == "hosts_per_router") p.topology.hosts_per_router = static_cast<std::uint32_t>(count());
    else if (key == "routers_per_group") p.topology.routers_per_group = static_cast<std::uint32_t>(count());
    else if (key == "degraded_fraction") p.topology.degraded_fraction = num();
    else if (key == "degraded_factor") p.topology.degraded_factor = num();
    else if (key == "algorithms") {
      const auto names = nb::cast<std::vector<std::string>>(v);
      if (names.size() != sim::kNumCollTypes)
        throw std::invalid_argument("algorithms must name one algorithm per collective type");
      for (int t = 0; t < sim::kNumCollTypes; ++t) p.collectives.algorithms[t] = sim::parse_algorithm(names[t]);
    } else if (key == "reduce_ns_per_byte") p.collectives.reduce_ns_per_byte = num();
    else if (key == "chunks") p.collectives.chunks = static_cast<std::uint32_t>(std::max<std::uint64_t>(1, count()));
    else if (key == "data_completion") p.collectives.data_completion = nb::cast<bool>(v);
    else if (key == "all_to_all_per_peer") p.collectives.all_to_all_per_peer = nb::cast<bool>(v);
    else if (key == "collective_cost") {
      const std::string c = text();
      if (c == "model") p.collectives.cost = sim::CollectiveCost::kModel;
      else if (c == "recorded") p.collectives.cost = sim::CollectiveCost::kRecorded;
      else throw std::invalid_argument("unknown collective cost '" + c + "' (model, recorded)");
    } else if (key == "noise_frequency") p.noise.frequency = num();
    else if (key == "noise_duration") p.noise.duration = num();
    else if (key == "noise_exponential") p.noise.exponential = nb::cast<bool>(v);
    else if (key == "nic_scale") p.nic_scale = nb::cast<std::vector<double>>(v);
    else if (key == "seed") p.seed = count();
    else if (key == "max_events") p.max_events = count();
    else throw std::invalid_argument("unknown simulation parameter '" + key + "'");
  }
  if (intra.L || intra.o || intra.g || intra.G || intra.O) {
    p.has_intra_loggops = true;
    p.intra_loggops = {intra.L.value_or(p.loggops.L), intra.o.value_or(p.loggops.o), intra.g.value_or(p.loggops.g),
                       intra.G.value_or(p.loggops.G), intra.O.value_or(p.loggops.O)};
  }
  if (intra.alpha || intra.beta) {
    p.has_intra_alpha_beta = true;
    p.intra_alpha = intra.alpha.value_or(p.alpha);
    p.intra_beta = intra.beta.value_or(p.beta);
  }
  return p;
}

CollType coll_type_of(int t) {
  if (t < 0 || t >= sim::kNumCollTypes) throw std::invalid_argument("unknown collective type " + std::to_string(t));
  return static_cast<CollType>(t);
}

// Every member's schedule of one collective, one row per action.
struct ScheduleColumns {
  std::vector<std::uint32_t> member, action, peer, seq;
  std::vector<std::uint8_t> kind, output;
  std::vector<std::uint64_t> bytes;
  std::vector<std::uint32_t> dep_member, dep_action, dep_on;  // action (member, dep_action) waits for dep_on

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
      [](const nb::dict& graph, const nb::handle& group_offsets, const nb::handle& group_members,
         const nb::dict& params) {
        const sim::GraphView g = view_of(graph);
        const sim::Groups groups{as_span<std::uint32_t>(group_offsets, "group_offsets"),
                                 as_span<std::uint32_t>(group_members, "group_members")};
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
      [](const nb::dict& graph, const nb::handle& group_offsets, const nb::handle& group_members,
         const nb::dict& params) {
        const sim::GraphView g = view_of(graph);
        const sim::Groups groups{as_span<std::uint32_t>(group_offsets, "group_offsets"),
                                 as_span<std::uint32_t>(group_members, "group_members")};
        const sim::SimParams p = params_of(params);
        sim::LoweringStats stats;
        sim::GraphColumns lowered;
        {
          nb::gil_scoped_release release;
          lowered = sim::lower_collectives(g, groups, p.collectives, p.ranks_per_node, stats);
        }
        return nb::make_tuple(to_dict(std::move(lowered)), to_dict(std::move(stats)));
      },
      "graph"_a, "group_offsets"_a, "group_members"_a, "params"_a);

  m.def(
      "collective_schedule",
      [](int coll_type, const std::string& algorithm, std::uint32_t size, std::uint64_t bytes, std::uint32_t root,
         std::uint32_t node_size, std::uint32_t chunks, bool reduce, bool all_to_all_per_peer) {
        sim::CollSpec spec;
        spec.type = coll_type_of(coll_type);
        spec.algorithm = sim::parse_algorithm(algorithm);
        spec.size = size;
        spec.bytes = bytes;
        spec.root = root;
        spec.node_size = node_size;
        spec.chunks = chunks;
        spec.reduce = reduce;
        spec.all_to_all_per_peer = all_to_all_per_peer;
        ScheduleColumns out;
        sim::Schedule s;
        for (std::uint32_t member = 0; member < size; ++member) {
          sim::build_schedule(spec, member, s);
          std::vector<std::uint8_t> is_output(s.actions.size(), 0);
          for (std::uint32_t a : s.outputs) is_output[a] = 1;
          for (std::uint32_t a = 0; a < s.actions.size(); ++a) {
            const sim::Action& act = s.actions[a];
            out.member.push_back(member);
            out.action.push_back(a);
            out.kind.push_back(static_cast<std::uint8_t>(act.kind));
            out.peer.push_back(act.peer);
            out.bytes.push_back(act.bytes);
            out.seq.push_back(act.seq);
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
        spec.size = static_cast<std::uint32_t>(members.size());
        spec.bytes = bytes;
        spec.root = root;
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
