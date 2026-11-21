// Engine A: Single-pass O(P^2)-memory aggregate analyzer for all non-DAG
// distributed aspects (rank_skew, traffic_matrix, collectives, equivalence).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "streaming/batch.hpp"
#include "streaming/sources/source.hpp"

namespace cta::streaming::engines {

struct OnlineAggOptions {
  bool include_collectives_in_traffic = true;  // true for Chakra, false for GOAL
  std::uint32_t min_world_size = 1;
  std::string quantile_sketch = "ddsketch";  // "ddsketch" | "kll" | "gk"
  std::size_t quantile_exact_cap = 65536;
  double quantile_accuracy = 0.001;
  std::size_t chunk_nodes = 524288;
};

struct OnlineAggResult {
  // Basic graph counts.
  std::uint64_t num_nodes = 0;
  std::uint64_t num_raw_edges = 0;
  std::uint32_t world_size = 0;

  // Per-rank load table (sorted by rank_ids ascending).
  std::vector<std::uint32_t> rank_ids;
  std::vector<std::uint64_t> rank_compute_ns;
  std::vector<std::uint64_t> rank_comm_bytes;
  std::vector<std::uint64_t> rank_op_count;

  // Rank-skew summary scalars.
  double max_rank_compute_us = 0.0;
  double mean_rank_compute_us = 0.0;
  double compute_skew_ratio = 1.0;
  double max_rank_comm_mb = 0.0;
  double comm_jains_fairness = 1.0;

  // Collectives / comm characterization scalars & breakdown table.
  double total_comm_mb = 0.0;
  std::uint64_t comm_op_count = 0;
  double median_msg_kb = 0.0;
  double p95_msg_kb = 0.0;
  double max_msg_kb = 0.0;
  std::vector<std::string> comm_type_names;
  std::vector<std::uint64_t> comm_type_bytes;
  std::vector<std::uint64_t> comm_type_count;

  // Traffic matrix (sorted by src_rank, dst_rank) & scalars.
  std::vector<std::uint32_t> traffic_src;
  std::vector<std::uint32_t> traffic_dst;
  std::vector<std::uint64_t> traffic_msgs;
  std::vector<std::uint64_t> traffic_bytes;
  std::uint64_t active_rank_links = 0;
  double hotspot_link_mb = 0.0;
  double mean_peer_degree = 0.0;
  double total_p2p_mb = 0.0;

  // Equivalence conservation stats (Chakra <-> GOAL without materializing tables).
  std::uint64_t total_compute_ns = 0;
  std::uint64_t total_send_bytes = 0;
  std::uint64_t send_count = 0;
  std::uint64_t recv_count = 0;
  std::vector<std::string> coll_type_names;
  std::vector<std::uint64_t> coll_type_bytes;

  StreamProvenance provenance;

  template <typename Visit>
  void for_each_column(Visit&& visit) {
    visit("num_nodes", num_nodes);
    visit("num_raw_edges", num_raw_edges);
    visit("world_size", world_size);
    visit("rank_ids", rank_ids);
    visit("rank_compute_ns", rank_compute_ns);
    visit("rank_comm_bytes", rank_comm_bytes);
    visit("rank_op_count", rank_op_count);
    visit("max_rank_compute_us", max_rank_compute_us);
    visit("mean_rank_compute_us", mean_rank_compute_us);
    visit("compute_skew_ratio", compute_skew_ratio);
    visit("max_rank_comm_mb", max_rank_comm_mb);
    visit("comm_jains_fairness", comm_jains_fairness);
    visit("total_comm_mb", total_comm_mb);
    visit("comm_op_count", comm_op_count);
    visit("median_msg_kb", median_msg_kb);
    visit("p95_msg_kb", p95_msg_kb);
    visit("max_msg_kb", max_msg_kb);
    visit("comm_type_names", comm_type_names);
    visit("comm_type_bytes", comm_type_bytes);
    visit("comm_type_count", comm_type_count);
    visit("traffic_src", traffic_src);
    visit("traffic_dst", traffic_dst);
    visit("traffic_msgs", traffic_msgs);
    visit("traffic_bytes", traffic_bytes);
    visit("active_rank_links", active_rank_links);
    visit("hotspot_link_mb", hotspot_link_mb);
    visit("mean_peer_degree", mean_peer_degree);
    visit("total_p2p_mb", total_p2p_mb);
    visit("total_compute_ns", total_compute_ns);
    visit("total_send_bytes", total_send_bytes);
    visit("send_count", send_count);
    visit("recv_count", recv_count);
    visit("coll_type_names", coll_type_names);
    visit("coll_type_bytes", coll_type_bytes);
    visit("provenance", provenance);
  }
};

// Runs a single streaming pass over `source` and computes all aggregate aspects.
OnlineAggResult run_online_aggregates(GraphSource& source, const OnlineAggOptions& opts = {});

}  // namespace cta::streaming::engines
