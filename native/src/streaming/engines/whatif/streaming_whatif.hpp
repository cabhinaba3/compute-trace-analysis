// Engine D: Streaming multi-lane what-if & chunked interconnect repricing
// directly over a `.cta_graph` topological block cache.
//
// Avoids allocating the full O(12 * K * V) DP tables and full-graph DataFrames
// by streaming one topological block at a time with slot-recycled cross-block
// boundary state (and automatic paged disk fallback for adversarial Theta(V)
// cutwidth) plus a compact 1-byte/vertex/lane local-predecessor winner log
// truncated and reused after each lane batch.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "streaming/cache/graph_cache.hpp"

namespace cta::streaming::engines {

struct InterconnectSpec {
  std::int64_t gpus_per_node = 8;
  double intra_alpha_ns = 800.0;
  double intra_ns_per_byte = 0.02;
  double inter_alpha_ns = 800.0;
  double inter_ns_per_byte = 0.02;
  std::string algorithm = "flat";  // "flat" | "ring" | "tree" | "auto"
  std::unordered_map<std::string, std::int64_t> group_sizes;
};

struct RepriceStep {
  InterconnectSpec target;
  bool has_base = false;
  InterconnectSpec base;
};

struct RepriceAudit {
  bool has_recorded_comm_without_base = false;
  std::vector<std::string> defaulted_collectives;
};

struct StreamingWhatIfResult {
  std::vector<double> makespan;
  std::vector<std::int32_t> depth;
  std::vector<double> critical_compute;
  std::vector<double> critical_comm;
  std::vector<std::int64_t> critical_comm_bytes;
  std::vector<double> critical_by_group;  // num_scenarios * num_groups
  int lanes = 0;
  int workers = 0;
  std::uint64_t bytes_per_worker = 0;
  bool used_boundary_spill = false;
};

// Lightweight block-level vertex metadata (without edges) for streaming
// Python/Polars selector & group evaluation in O(block_vertices) RAM.
struct BlockMetaView {
  std::vector<std::uint32_t> rank;
  std::vector<std::uint8_t> op_class;
  std::vector<std::int16_t> comm_type_id;
  std::vector<std::int32_t> name_id;
  std::vector<std::string> local_names;  // empty when name_id[i] >= 0
  std::vector<double> weight_ns;         // after applying reprice_steps
  std::vector<std::uint8_t> is_comm;
  std::vector<std::uint8_t> is_compute;
  std::vector<std::uint8_t> is_virtual;
  std::vector<std::uint64_t> size_bytes;
};

// Computes the repriced weight of a single vertex after applying `steps`,
// matching `cta.analysis.interconnect.node_cost_ns` and `cta.analysis.whatif.reprice`.
double effective_vertex_weight_ns(const cache::TopoVertexMeta& vm, const std::vector<std::string>& comm_types,
                                  std::int64_t unique_ranks, const std::vector<RepriceStep>& steps);

// Inspects a `.cta_graph` file to report the exact warnings `whatif.reprice`
// emits (recorded comm without baseline interconnect, and collective types that
// defaulted to full world size).
RepriceAudit inspect_reprice_warnings(const cache::GraphCacheReader& reader, const RepriceStep& step);

// Reads only the vertex metadata of topological block `block_idx` (applying
// `reprice_steps` to compute effective weights) without allocating edge arrays.
BlockMetaView read_block_meta(const cache::GraphCacheReader& reader, std::uint32_t block_idx,
                              const std::vector<RepriceStep>& reprice_steps);

// Runs Engine D multi-lane streaming what-if over `cache_path`.
//
// `groups_path` points to a sequential binary file containing `num_total_vertices`
// uint32_t group IDs in topological position order (or empty if `inline_groups`
// is provided). `scale` is row-major `[num_scenarios * num_groups]`.
StreamingWhatIfResult solve_whatif_streaming(const std::string& cache_path, const std::string& groups_path,
                                             const std::uint32_t* inline_groups, std::size_t inline_groups_len,
                                             std::size_t num_groups, const double* scale, std::size_t num_scenarios,
                                             const std::vector<RepriceStep>& reprice_steps, int threads,
                                             std::uint64_t memory_budget_bytes, const std::string& scratch_dir);

}  // namespace cta::streaming::engines
