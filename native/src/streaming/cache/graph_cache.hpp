// Compact disk-backed topological block graph cache (.cta_graph) shared by
// Engine C (GridGraph exact DAG + Scalasca attribution) and Engine D
// (streaming what-if & interconnect repricing).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "streaming/batch.hpp"
#include "streaming/engines/engine.hpp"
#include "streaming/sources/source.hpp"

namespace cta::streaming::cache {

constexpr std::uint64_t kGraphCacheMagic = 0x4850415247415443ULL;  // "CTAGRAPH"
constexpr std::uint32_t kGraphCacheVersion = 1;
constexpr std::uint32_t kMaxInternedNames = 262144;

// Flags packed inside TopoVertexMeta::flags.
constexpr std::uint16_t kFlagIsComm = 1u << 0;
constexpr std::uint16_t kFlagIsCompute = 1u << 1;
constexpr std::uint16_t kFlagIsP2P = 1u << 2;
constexpr std::uint16_t kFlagIsVirtual = 1u << 3;
constexpr std::uint16_t kFlagIsSend = 1u << 4;
constexpr std::uint16_t kFlagIsRecv = 1u << 5;

// Fixed 64-byte POD metadata for one vertex at topological position `p`.
struct TopoVertexMeta {
  std::uint32_t orig_vid = 0;      // original arrival/row index (for lowest-id tie-breaking)
  std::uint32_t rank = 0;
  std::uint64_t node_id = 0;
  double weight_ns = 0.0;          // nominal or recorded weight
  std::uint64_t duration_ns = 0;   // raw recorded duration_ns (0 when unrecorded)
  std::uint64_t size_bytes = 0;
  std::uint32_t last_use_pos = 0;  // max topological position q consuming p (slot recycling!)
  std::int32_t name_id = -1;       // interned index (>= 0) or -1 if stored in per-block blob
  std::int16_t comm_type_id = -1;  // interned index in comm_types table
  std::uint16_t flags = 0;
  std::uint8_t op_class = 0;
  std::uint8_t pad[3]{};
  std::int32_t src_rank = -1;
  std::int32_t dst_rank = -1;
};
static_assert(sizeof(TopoVertexMeta) == 64, "TopoVertexMeta must remain 64 bytes");

struct BlockIndexEntry {
  std::uint64_t file_offset = 0;
  std::uint32_t pos_begin = 0;
  std::uint32_t pos_end = 0;
  std::uint64_t num_edges = 0;
  std::uint64_t local_names_bytes = 0;
};

struct GraphCacheHeader {
  std::uint64_t magic = kGraphCacheMagic;
  std::uint32_t version = kGraphCacheVersion;
  std::uint32_t vertex_width = 4;  // 4 bytes (uint32_t, up to 2^32 vertices)
  std::uint64_t num_real_vertices = 0;
  std::uint64_t num_total_vertices = 0;  // real + optional virtual barrier vertices
  std::uint64_t num_edges = 0;           // deduplicated distinct edges
  std::uint64_t unordered = 0;           // vertices held back by cycles
  std::uint32_t num_blocks = 0;
  std::uint32_t block_vertices = 0;
  std::uint32_t world_size = 0;
  std::uint8_t goal_semantics = 0;
  std::uint8_t collective_sync = 0;
  std::uint8_t has_recorded_comm_times = 0;
  std::uint8_t reserved = 0;
  std::uint64_t unmatched_p2p_sends = 0;
  std::uint64_t unmatched_p2p_recvs = 0;
  std::uint64_t max_cutwidth = 0;
  std::uint64_t rank_stats_offset = 0;
  std::uint64_t names_offset = 0;
  std::uint64_t comm_types_offset = 0;
  std::uint64_t witness_offset = 0;
  std::uint64_t block_index_offset = 0;
};

struct RankComputeEntry {
  std::uint32_t rank = 0;
  std::uint32_t op_count = 0;
  double compute_ns = 0.0;
};

// One topological block loaded into RAM on demand during forward/reverse sweeps.
struct TopoBlock {
  std::uint32_t pos_begin = 0;
  std::uint32_t pos_end = 0;
  std::vector<TopoVertexMeta> vertices;
  std::vector<std::uint64_t> pred_ptr;  // len + 1 entries (local to this block's preds)
  std::vector<std::uint32_t> preds;     // ascending topological positions u < p
  std::vector<std::uint8_t> pred_kind;  // EdgeKind per predecessor edge
  // Per-block name slices for vertices with name_id == -1 (unique GOAL labels).
  std::vector<std::uint32_t> local_name_end;
  std::vector<char> local_name_bytes;

  std::size_t size() const { return vertices.size(); }

  std::string_view name_of(std::size_t local_idx, const std::vector<std::string>& global_names) const {
    const std::int32_t nid = vertices[local_idx].name_id;
    if (nid >= 0 && static_cast<std::size_t>(nid) < global_names.size()) return global_names[static_cast<std::size_t>(nid)];
    if (local_idx < local_name_end.size()) {
      const std::uint32_t begin = local_idx == 0 ? 0 : local_name_end[local_idx - 1];
      const std::uint32_t end = local_name_end[local_idx];
      if (end >= begin && end <= local_name_bytes.size()) {
        return {local_name_bytes.data() + begin, end - begin};
      }
    }
    return "";
  }
};

// Read-only handle to a built `.cta_graph` binary file on disk.
class GraphCacheReader {
 public:
  explicit GraphCacheReader(const std::string& path);
  ~GraphCacheReader();
  GraphCacheReader(const GraphCacheReader&) = delete;
  GraphCacheReader& operator=(const GraphCacheReader&) = delete;

  const GraphCacheHeader& header() const { return header_; }
  const std::vector<std::string>& names() const { return names_; }
  const std::vector<std::string>& comm_types() const { return comm_types_; }
  const std::vector<RankComputeEntry>& rank_stats() const { return rank_stats_; }
  const std::vector<BlockIndexEntry>& block_index() const { return block_index_; }
  const std::vector<std::uint32_t>& cycle_witness_rank() const { return witness_rank_; }
  const std::vector<std::uint64_t>& cycle_witness_node_id() const { return witness_node_id_; }
  const std::string& path() const { return path_; }

  // Loads topological block `block_idx` sequentially from disk into `out`.
  void read_block(std::uint32_t block_idx, TopoBlock& out) const;

 private:
  void read_header_and_indexes(const std::string& path);

  std::string path_;
  std::FILE* fp_ = nullptr;
  GraphCacheHeader header_{};
  std::vector<std::string> names_;
  std::vector<std::string> comm_types_;
  std::vector<RankComputeEntry> rank_stats_;
  std::vector<BlockIndexEntry> block_index_;
  std::vector<std::uint32_t> witness_rank_;
  std::vector<std::uint64_t> witness_node_id_;
};

// Builds a `.cta_graph` binary cache from any GraphSource using GridGraph 2D
// edge-block partitioning (USENIX ATC '15) + semi-external fast-path within
// `opts.memory_budget_bytes`. Throws std::runtime_error if `fail_on_cycle` is
// true and a cycle is detected.
void build_graph_cache(GraphSource& source, const std::string& cache_path, const engines::DagSolveOptions& opts);

}  // namespace cta::streaming::cache
