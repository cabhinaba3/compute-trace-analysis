// Settings of a run, under the dotted names of cta.sim.config. Times are ns,
// sizes bytes, bandwidths bytes per ns (GB/s), noise frequency per second.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "sim/types.hpp"

namespace cta::sim {

struct LogGOPS {
  double L = 2500, o = 1500, g = 1000, G = 6, O = 0;
};

struct SimParams {
  std::string model = "loggops";  // loggops, alpha_beta or flow
  std::uint32_t world_size = 0;   // 0: one more than the highest rank
  std::uint32_t ranks_per_node = 1;
  std::uint64_t seed = 1, max_events = 0;

  LogGOPS loggops;
  std::uint64_t eager_limit = 65535;
  std::optional<double> intra_L, intra_o, intra_g, intra_G, intra_O;  // same-node pairs
  double alpha = 5500, beta = 6;
  std::optional<double> intra_alpha, intra_beta;

  std::string topology = "switch";  // switch, fat_tree, torus, ring, dragonfly
  double link_bandwidth = 1.0 / 6, link_latency = 1250;
  double intra_bandwidth = 0, intra_latency = 0;  // 0: same-node traffic uses the fabric
  double global_bandwidth = 0;                    // dragonfly; 0: link_bandwidth
  double oversubscription = 1;
  std::vector<std::uint32_t> dims;
  std::uint32_t hosts_per_leaf = 16, spines = 8, hosts_per_router = 4, routers_per_group = 8;

  std::array<std::string, kNumCollTypes> algorithms = {
      "ring", "binomial_tree", "ring", "binomial_tree", "binomial_tree",
      "binomial_tree", "direct", "ring", "ring", "recursive_doubling"};
  std::string completion = "all";        // all, or outputs: done once it has its result
  std::string all_to_all_size = "total";  // total, or per_peer
  std::string collective_cost = "model";  // model, or recorded
  double reduce_ns_per_byte = 0;
  std::uint32_t chunks = 1;

  double noise_frequency = 0, noise_duration = 0;
  std::string noise_distribution = "fixed";  // fixed or exponential
  std::vector<double> nic_scale;             // per rank; missing ranks get 1
  double degraded_fraction = 0, degraded_factor = 1;

  template <typename Visit>
  void for_each_param(Visit&& v) {
    v("model", model);
    v("world_size", world_size);
    v("ranks_per_node", ranks_per_node);
    v("seed", seed);
    v("max_events", max_events);
    v("loggops.L", loggops.L);
    v("loggops.o", loggops.o);
    v("loggops.g", loggops.g);
    v("loggops.G", loggops.G);
    v("loggops.O", loggops.O);
    v("loggops.S", eager_limit);
    v("loggops.intra_L", intra_L);
    v("loggops.intra_o", intra_o);
    v("loggops.intra_g", intra_g);
    v("loggops.intra_G", intra_G);
    v("loggops.intra_O", intra_O);
    v("alpha_beta.alpha", alpha);
    v("alpha_beta.beta", beta);
    v("alpha_beta.intra_alpha", intra_alpha);
    v("alpha_beta.intra_beta", intra_beta);
    v("topology.kind", topology);
    v("topology.link_bandwidth", link_bandwidth);
    v("topology.link_latency", link_latency);
    v("topology.intra_bandwidth", intra_bandwidth);
    v("topology.intra_latency", intra_latency);
    v("topology.global_bandwidth", global_bandwidth);
    v("topology.oversubscription", oversubscription);
    v("topology.dims", dims);
    v("topology.hosts_per_leaf", hosts_per_leaf);
    v("topology.spines", spines);
    v("topology.hosts_per_router", hosts_per_router);
    v("topology.routers_per_group", routers_per_group);
    for (int t = 0; t < kNumCollTypes; ++t)
      v(std::string("collectives.algorithms.") + kCollTypeNames[t], algorithms[t]);
    v("collectives.completion", completion);
    v("collectives.all_to_all_size", all_to_all_size);
    v("collectives.cost", collective_cost);
    v("collectives.reduce_ns_per_byte", reduce_ns_per_byte);
    v("collectives.chunks", chunks);
    v("noise.frequency", noise_frequency);
    v("noise.duration", noise_duration);
    v("noise.distribution", noise_distribution);
    v("noise.nic_scale", nic_scale);
    v("noise.degraded_fraction", degraded_fraction);
    v("noise.degraded_factor", degraded_factor);
  }

  // Throws std::invalid_argument for an unknown choice or a negative cost.
  void check() const;
  bool has_intra_loggops() const { return intra_L || intra_o || intra_g || intra_G || intra_O; }
  LogGOPS intra_loggops() const {
    return {intra_L.value_or(loggops.L), intra_o.value_or(loggops.o), intra_g.value_or(loggops.g),
            intra_G.value_or(loggops.G), intra_O.value_or(loggops.O)};
  }
};

}  // namespace cta::sim
