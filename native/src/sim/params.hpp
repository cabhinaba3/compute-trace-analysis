// Settings of a simulation run. Times are nanoseconds, sizes bytes, bandwidths
// bytes per nanosecond (1 byte/ns = 1 GB/s). Python fills these from its
// configuration (cta.sim.config), whose documentation explains each one.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "sim/collectives.hpp"

namespace cta::sim {

enum class NetworkModel : std::uint8_t {
  kLogGOPS = 0,    // LogGOPSim: o, O on the CPUs, g, G on the NICs, L in flight, no congestion
  kAlphaBeta = 1,  // alpha + bytes * beta per message, nothing shared
  kFlow = 2,       // flows over the links of a topology, sharing bandwidth max-min fairly
};

enum class Rendezvous : std::uint8_t {
  kSync = 0,    // LogGOPSim: data leaves at once; the send completes when it is matched
  kRtsCts = 1,  // request-to-send, clear-to-send, then the data
};

enum class CollectiveCost : std::uint8_t {
  kModel = 0,     // from the algorithm's schedule and the network model
  kRecorded = 1,  // the shortest duration the trace recorded for the instance
};

struct CollectiveParams {
  std::array<Algorithm, kNumCollTypes> algorithms = {
      Algorithm::kRing,         Algorithm::kBinomialTree, Algorithm::kRing,   Algorithm::kBinomialTree,
      Algorithm::kBinomialTree, Algorithm::kBinomialTree, Algorithm::kDirect, Algorithm::kRing,
      Algorithm::kRing,         Algorithm::kRecursiveDoubling};
  double reduce_ns_per_byte = 0;  // > 0 adds reduction calcs (lowered) or time (analytical)
  std::uint32_t chunks = 1;
  // false: a member is done after every action of its schedule, its last
  // (rendezvous) sends included, as MPI and NCCL collectives are. true: once
  // it has its result (its outputs), as GOAL schedules that chain on a
  // collective's last recv behave.
  bool data_completion = false;
  bool all_to_all_per_peer = false;  // see CollSpec
  CollectiveCost cost = CollectiveCost::kModel;
};

// One set of LogGOPS parameters.
struct LogGOPS {
  double L = 2500, o = 1500, g = 1000, G = 6, O = 0;
};

struct TopologyParams {
  std::string kind = "switch";  // switch, fat_tree, torus, ring, dragonfly
  double link_bandwidth = 1.0 / 6, link_latency = 1250;
  double intra_bandwidth = 0, intra_latency = 0;  // 0 bandwidth: same-node pairs use the fabric
  double global_bandwidth = 0;                    // dragonfly global links; 0 = link_bandwidth
  std::vector<std::uint32_t> dims;                // torus; ring uses one dimension
  std::uint32_t hosts_per_leaf = 0, spines = 0;   // fat_tree (leaf-spine)
  double oversubscription = 1;
  std::uint32_t hosts_per_router = 0, routers_per_group = 0;  // dragonfly
  double degraded_fraction = 0, degraded_factor = 1;          // slow down this share of links
};

struct NoiseParams {
  double frequency = 0;  // detours per nanosecond per CPU
  double duration = 0;   // mean detour length
  bool exponential = false;  // exponential detour lengths instead of fixed ones
};

struct SimParams {
  NetworkModel model = NetworkModel::kLogGOPS;
  std::uint32_t world_size = 0;      // 0: one more than the highest rank seen
  std::uint32_t ranks_per_node = 1;  // ranks r and s share a node when r / n == s / n

  LogGOPS loggops;
  bool has_intra_loggops = false;
  LogGOPS intra_loggops;  // for same-node pairs
  std::uint64_t S = 65535;  // largest eager message (LogGOPS and flow)
  Rendezvous rendezvous = Rendezvous::kSync;

  double alpha = 5500, beta = 6;
  bool has_intra_alpha_beta = false;
  double intra_alpha = 0, intra_beta = 0;

  TopologyParams topology;  // flow; its o and O come from `loggops`

  CollectiveParams collectives;
  NoiseParams noise;
  std::vector<double> nic_scale;  // per rank bandwidth factor; missing ranks get 1
  std::uint64_t seed = 1;
  std::uint64_t max_events = 0;  // stop after this many events; 0 = no limit
};

}  // namespace cta::sim
