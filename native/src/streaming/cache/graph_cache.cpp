#include "streaming/cache/graph_cache.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "streaming/scratch.hpp"
#include "util/interner.hpp"
#include "util/memory.hpp"
#include "util/util.hpp"

namespace cta::streaming::cache {
namespace {

constexpr std::uint32_t kNone32 = std::numeric_limits<std::uint32_t>::max();

// Closes a stdio file on scope exit, so an exception on the way does not leak the handle.
struct FileGuard {
  std::FILE* fp = nullptr;
  explicit FileGuard(std::FILE* f) : fp(f) {}
  FileGuard(const FileGuard&) = delete;
  FileGuard& operator=(const FileGuard&) = delete;
  ~FileGuard() { close(); }
  void close() {
    if (fp) std::fclose(fp);
    fp = nullptr;
  }
};

template <typename T>
void write_exact(std::FILE* fp, const T* data, std::size_t count) {
  if (count == 0) return;
  if (std::fwrite(data, sizeof(T), count, fp) != count) throw std::runtime_error("graph cache write failed");
}

template <typename T>
void read_exact(std::FILE* fp, T* data, std::size_t count) {
  if (count == 0) return;
  if (std::fread(data, sizeof(T), count, fp) != count) throw std::runtime_error("graph cache read failed");
}

void write_string_table(std::FILE* fp, const std::deque<std::string>& items) {
  const std::uint32_t count = static_cast<std::uint32_t>(items.size());
  write_exact(fp, &count, 1);
  for (const std::string& s : items) {
    const std::uint32_t len = static_cast<std::uint32_t>(s.size());
    write_exact(fp, &len, 1);
    write_exact(fp, s.data(), len);
  }
}

std::vector<std::string> read_string_table(std::FILE* fp) {
  std::uint32_t count = 0;
  read_exact(fp, &count, 1);
  std::vector<std::string> out;
  out.reserve(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    std::uint32_t len = 0;
    read_exact(fp, &len, 1);
    std::string s(len, '\0');
    read_exact(fp, s.data(), len);
    out.push_back(std::move(s));
  }
  return out;
}

// Raw vertex stored sequentially during Pass 0a before topological renumbering.
struct RawVertexRec {
  std::uint32_t rank = 0;
  std::uint64_t node_id = 0;
  double weight_ns = 0.0;
  std::uint64_t duration_ns = 0;
  std::uint64_t size_bytes = 0;
  std::int32_t name_id = -1;
  std::int16_t comm_type_id = -1;
  std::uint16_t flags = 0;
  std::uint8_t op_class = 0;
  std::uint8_t pad[3]{};
  std::int32_t src_rank = -1;
  std::int32_t dst_rank = -1;
  std::uint64_t spill_name_off = 0;
  std::uint32_t spill_name_len = 0;
  std::uint32_t pad2 = 0;
};

struct RawEdgeRec {
  std::uint32_t rank = 0;
  std::uint8_t kind = 0;
  std::uint8_t pad[3]{};
  std::uint64_t src_id = 0;
  std::uint64_t dst_id = 0;
};

struct ResolvedEdge {
  std::uint32_t u = 0;
  std::uint32_t v = 0;
  std::uint8_t kind = 0;
  std::uint8_t pad[3]{};
};

struct P2PRec {
  std::int32_t src_rank = -1;
  std::int32_t dst_rank = -1;
  std::uint32_t tag = 0;
  std::uint32_t seq = 0;
  std::uint32_t vid = 0;
  std::uint8_t is_send = 0;
  std::uint8_t pad[3]{};
};

struct DenseRun {
  std::uint32_t rank = 0;
  std::uint32_t first_vid = 0;
  std::uint32_t length = 0;
};

struct RankNodeVid {
  std::uint32_t rank = 0;
  std::uint32_t vid = 0;
  std::uint64_t node_id = 0;
};

struct ChannelKey {
  std::int32_t src = -1;
  std::int32_t dst = -1;
  std::uint32_t tag = 0;
  bool operator==(const ChannelKey& o) const { return src == o.src && dst == o.dst && tag == o.tag; }
};

struct ChannelKeyHash {
  std::size_t operator()(const ChannelKey& c) const {
    return static_cast<std::size_t>(mix64(static_cast<std::uint64_t>(static_cast<std::uint32_t>(c.src)) *
                                              0x9e3779b97f4a7c15ULL ^
                                          mix64(static_cast<std::uint64_t>(static_cast<std::uint32_t>(c.dst)) *
                                                    0xc2b2ae3d27d4eb4fULL ^
                                                static_cast<std::uint64_t>(c.tag))));
  }
};

// Priority for edge deduplication: finish-to-start dominates strict irequires.
int edge_kind_priority(std::uint8_t kind) {
  if (kind == static_cast<std::uint8_t>(EdgeKind::kIrequires)) return 1;
  return 0;
}

// Manages an array of scratch bucket files with small in-memory write buffers.
template <typename Rec>
class BucketWriter {
 public:
  BucketWriter(const ScratchDir& scratch, const std::string& prefix, std::size_t num_buckets)
      : paths_(num_buckets), bufs_(num_buckets), counts_(num_buckets, 0) {
    for (std::size_t b = 0; b < num_buckets; ++b) {
      paths_[b] = scratch.file(prefix + "_" + std::to_string(b) + ".bin");
    }
  }

  void push(std::size_t b, const Rec& rec) {
    auto& buf = bufs_[b];
    buf.push_back(rec);
    ++counts_[b];
    if (buf.size() >= kFlushCount) flush_bucket(b);
  }

  void flush_all() {
    for (std::size_t b = 0; b < bufs_.size(); ++b) flush_bucket(b);
  }

  std::size_t num_buckets() const { return paths_.size(); }
  std::uint64_t count(std::size_t b) const { return counts_[b]; }
  const std::string& path(std::size_t b) const { return paths_[b]; }

 private:
  static constexpr std::size_t kFlushCount = 4096;

  void flush_bucket(std::size_t b) {
    if (bufs_[b].empty()) return;
    std::FILE* fp = std::fopen(paths_[b].c_str(), "ab");
    if (!fp) throw std::runtime_error("cannot open scratch bucket " + paths_[b]);
    write_exact(fp, bufs_[b].data(), bufs_[b].size());
    std::fclose(fp);
    bufs_[b].clear();
  }

  std::vector<std::string> paths_;
  std::vector<std::vector<Rec>> bufs_;
  std::vector<std::uint64_t> counts_;
};

template <typename Rec>
std::vector<Rec> load_bucket(const std::string& path, std::uint64_t expected_count) {
  std::vector<Rec> out;
  if (expected_count == 0) return out;
  out.resize(static_cast<std::size_t>(expected_count));
  std::FILE* fp = std::fopen(path.c_str(), "rb");
  if (!fp) throw std::runtime_error("cannot open scratch bucket for read: " + path);
  read_exact(fp, out.data(), out.size());
  std::fclose(fp);
  return out;
}

std::uint64_t rank_node_hash(std::uint32_t r, std::uint64_t id) {
  return mix64(static_cast<std::uint64_t>(r) * 0x9e3779b97f4a7c15ULL ^ mix64(id));
}

}  // namespace

GraphCacheReader::GraphCacheReader(const std::string& path) : path_(path) {
  fp_ = std::fopen(path.c_str(), "rb");
  if (!fp_) throw std::runtime_error("cannot open graph cache " + path);
  // A throwing constructor never runs the destructor, so close the file here.
  try {
    read_header_and_indexes(path);
  } catch (...) {
    std::fclose(fp_);
    fp_ = nullptr;
    throw;
  }
}

void GraphCacheReader::read_header_and_indexes(const std::string& path) {
  read_exact(fp_, &header_, 1);
  if (header_.magic != kGraphCacheMagic || header_.version != kGraphCacheVersion) {
    throw std::invalid_argument("invalid or unsupported .cta_graph header in " + path);
  }
  // Every section the header points to must lie inside the file: a build cut short by a
  // crash leaves a header whose offsets run past the end.
  if (std::fseek(fp_, 0, SEEK_END) != 0) throw std::runtime_error("seek failed in graph cache");
  const long file_size = std::ftell(fp_);
  // A build that died early leaves the header with its offsets still zero, so they must
  // also lie past the header itself.
  const auto inside = [&](std::uint64_t off) {
    return file_size >= 0 && off >= sizeof(GraphCacheHeader) && off <= static_cast<std::uint64_t>(file_size);
  };
  if (!inside(header_.rank_stats_offset) || !inside(header_.names_offset) || !inside(header_.comm_types_offset) ||
      !inside(header_.witness_offset) || !inside(header_.block_index_offset) ||
      !inside(header_.block_index_offset + static_cast<std::uint64_t>(header_.num_blocks) * sizeof(BlockIndexEntry)))
    throw std::runtime_error("graph cache is truncated: " + path);

  if (std::fseek(fp_, static_cast<long>(header_.rank_stats_offset), SEEK_SET) != 0)
    throw std::runtime_error("seek failed in graph cache");
  std::uint32_t nr = 0;
  read_exact(fp_, &nr, 1);
  rank_stats_.resize(nr);
  read_exact(fp_, rank_stats_.data(), nr);

  if (std::fseek(fp_, static_cast<long>(header_.names_offset), SEEK_SET) != 0)
    throw std::runtime_error("seek failed in graph cache");
  names_ = read_string_table(fp_);

  if (std::fseek(fp_, static_cast<long>(header_.comm_types_offset), SEEK_SET) != 0)
    throw std::runtime_error("seek failed in graph cache");
  comm_types_ = read_string_table(fp_);

  if (std::fseek(fp_, static_cast<long>(header_.witness_offset), SEEK_SET) != 0)
    throw std::runtime_error("seek failed in graph cache");
  std::uint32_t nw = 0;
  read_exact(fp_, &nw, 1);
  witness_rank_.resize(nw);
  witness_node_id_.resize(nw);
  read_exact(fp_, witness_rank_.data(), nw);
  read_exact(fp_, witness_node_id_.data(), nw);

  if (std::fseek(fp_, static_cast<long>(header_.block_index_offset), SEEK_SET) != 0)
    throw std::runtime_error("seek failed in graph cache");
  block_index_.resize(header_.num_blocks);
  read_exact(fp_, block_index_.data(), header_.num_blocks);
}

GraphCacheReader::~GraphCacheReader() {
  if (fp_) std::fclose(fp_);
}

void GraphCacheReader::read_block(std::uint32_t block_idx, TopoBlock& out) const {
  if (block_idx >= block_index_.size()) throw std::out_of_range("graph cache block index out of range");
  const BlockIndexEntry& entry = block_index_[block_idx];
  const std::size_t len = static_cast<std::size_t>(entry.pos_end - entry.pos_begin);
  out.pos_begin = entry.pos_begin;
  out.pos_end = entry.pos_end;
  out.vertices.resize(len);
  out.pred_ptr.resize(len + 1);
  out.preds.resize(static_cast<std::size_t>(entry.num_edges));
  out.pred_kind.resize(static_cast<std::size_t>(entry.num_edges));
  out.local_name_end.resize(len);
  out.local_name_bytes.resize(static_cast<std::size_t>(entry.local_names_bytes));

  if (std::fseek(fp_, static_cast<long>(entry.file_offset), SEEK_SET) != 0)
    throw std::runtime_error("seek failed reading graph cache block");
  read_exact(fp_, out.vertices.data(), len);
  read_exact(fp_, out.pred_ptr.data(), len + 1);
  read_exact(fp_, out.preds.data(), out.preds.size());
  read_exact(fp_, out.pred_kind.data(), out.pred_kind.size());
  read_exact(fp_, out.local_name_end.data(), len);
  read_exact(fp_, out.local_name_bytes.data(), out.local_name_bytes.size());
}

void build_graph_cache(GraphSource& source, const std::string& cache_path, const engines::DagSolveOptions& opts) {
  const std::uint64_t budget = resolve_memory_budget(opts.memory_budget_bytes);
  ScratchDir scratch(opts.scratch_dir);

  // Partition counts sized to keep bucket reads well inside budget.
  const std::size_t kJoinBuckets = 16;
  const std::size_t kP2PBuckets = 16;

  // Pass 0a: Stream source into raw vertex file, raw edge buckets, and P2P buckets.
  const std::string raw_verts_path = scratch.file("raw_vertices.bin");
  const std::string spill_names_path = scratch.file("spill_names.bin");
  std::FILE* vfp = std::fopen(raw_verts_path.c_str(), "wb");
  std::FILE* nfp = std::fopen(spill_names_path.c_str(), "wb");
  FileGuard vguard(vfp), nguard(nfp);
  if (!vfp || !nfp) throw std::runtime_error("cannot open scratch vertex/name files");

  BucketWriter<RankNodeVid> v_join_buckets(scratch, "vjoin", kJoinBuckets);
  BucketWriter<RawEdgeRec> e_raw_buckets(scratch, "eraw", kJoinBuckets);
  BucketWriter<P2PRec> p2p_buckets(scratch, "p2p", kP2PBuckets);

  Interner name_interner, comm_type_interner;
  std::vector<DenseRun> dense_runs;
  bool still_dense = true;
  std::unordered_set<std::uint32_t> seen_dense_ranks;
  std::map<std::uint32_t, RankComputeEntry> rank_compute;
  std::unordered_map<ChannelKey, std::uint32_t, ChannelKeyHash> send_seq, recv_seq;

  struct CollParticipant {
    std::string group;
    std::string comm_type;
    std::uint32_t seq = 0;
    std::uint32_t rank = 0;
    std::uint32_t vid = 0;
  };
  std::vector<CollParticipant> coll_entries;
  std::unordered_map<std::uint32_t, std::unordered_map<std::string, std::uint32_t>> coll_seq_per_rank;

  std::uint64_t num_real = 0;
  std::uint64_t spill_name_offset = 0;
  bool has_recorded_comm = false;
  std::vector<RawVertexRec> vbuf;
  vbuf.reserve(4096);

  auto flush_vbuf = [&] {
    if (!vbuf.empty()) {
      write_exact(vfp, vbuf.data(), vbuf.size());
      vbuf.clear();
    }
  };

  GraphChunk chunk;
  const std::size_t read_step = opts.chunk_nodes > 0 ? opts.chunk_nodes : 262144;
  while (source.next_chunk(chunk, read_step)) {
    const NodeBatch& nb = chunk.nodes;
    for (std::size_t i = 0; i < nb.size(); ++i) {
      if (num_real >= std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("graph exceeds 2^32 vertices");
      const std::uint32_t vid = static_cast<std::uint32_t>(num_real++);
      const std::uint32_t r = nb.rank[i];
      const std::uint64_t nid = nb.node_id[i];
      const std::uint8_t oc = nb.op_class[i];
      const double w = engines::node_weight_ns(nb.duration_ns[i], oc, nb.size_bytes[i]);
      if (is_comm_class(oc) && nb.duration_ns[i] > 0) has_recorded_comm = true;

      // Dense rank-run check matching dag_edges.cpp try_dense().
      if (still_dense) {
        if (dense_runs.empty() || dense_runs.back().rank != r) {
          if (nid != 0 || !seen_dense_ranks.insert(r).second) {
            still_dense = false;
          } else {
            dense_runs.push_back({r, vid, 1});
          }
        } else {
          if (nid != dense_runs.back().length) {
            still_dense = false;
          } else {
            ++dense_runs.back().length;
          }
        }
      }

      auto& rce = rank_compute[r];
      rce.rank = r;
      ++rce.op_count;
      if (is_compute_class(oc)) rce.compute_ns += w;

      const std::string_view name = nb.name_at(i);
      std::int32_t name_id = -1;
      std::uint64_t sp_off = 0;
      std::uint32_t sp_len = 0;
      if (name_interner.size() < kMaxInternedNames) {
        name_id = name_interner.intern(name);
      } else {
        // Check if already interned; otherwise spill to per-vertex name stream.
        bool is_new = false;
        const std::int32_t existing = name_interner.intern(name, &is_new);
        if (!is_new) {
          name_id = existing;
        } else {
          sp_off = spill_name_offset;
          sp_len = static_cast<std::uint32_t>(name.size());
          write_exact(nfp, name.data(), name.size());
          spill_name_offset += name.size();
        }
      }
      const std::string_view ctype = nb.comm_type_at(i);
      const std::int16_t ct_id = static_cast<std::int16_t>(comm_type_interner.intern(ctype));

      std::uint16_t flags = 0;
      if (is_comm_class(oc)) flags |= kFlagIsComm;
      if (is_compute_class(oc)) flags |= kFlagIsCompute;
      if (is_p2p_class(oc)) flags |= kFlagIsP2P;
      if (oc == static_cast<std::uint8_t>(OpClass::kCommSend)) flags |= kFlagIsSend;
      if (oc == static_cast<std::uint8_t>(OpClass::kCommRecv)) flags |= kFlagIsRecv;

      RawVertexRec rec;
      rec.rank = r;
      rec.node_id = nid;
      rec.weight_ns = w;
      rec.duration_ns = nb.duration_ns[i];
      rec.size_bytes = nb.size_bytes[i];
      rec.name_id = name_id;
      rec.comm_type_id = ct_id;
      rec.flags = flags;
      rec.op_class = oc;
      rec.src_rank = nb.src_rank[i];
      rec.dst_rank = nb.dst_rank[i];
      rec.spill_name_off = sp_off;
      rec.spill_name_len = sp_len;
      vbuf.push_back(rec);
      if (vbuf.size() >= 4096) flush_vbuf();

      // Always record join tuple in case `still_dense` flips false later.
      v_join_buckets.push(static_cast<std::size_t>(r % kJoinBuckets), RankNodeVid{r, vid, nid});

      // Row-order P2P sequence numbering.
      if (oc == static_cast<std::uint8_t>(OpClass::kCommSend)) {
        const ChannelKey ck{nb.src_rank[i], nb.dst_rank[i], nb.message_tag[i]};
        const std::uint32_t seq = send_seq[ck]++;
        p2p_buckets.push(ChannelKeyHash{}(ck) % kP2PBuckets,
                         P2PRec{ck.src, ck.dst, ck.tag, seq, vid, 1, {}});
      } else if (oc == static_cast<std::uint8_t>(OpClass::kCommRecv)) {
        const ChannelKey ck{nb.src_rank[i], nb.dst_rank[i], nb.message_tag[i]};
        const std::uint32_t seq = recv_seq[ck]++;
        p2p_buckets.push(ChannelKeyHash{}(ck) % kP2PBuckets,
                         P2PRec{ck.src, ck.dst, ck.tag, seq, vid, 0, {}});
      } else if (opts.collective_sync == CollectiveSync::kBarrier &&
                 oc == static_cast<std::uint8_t>(OpClass::kCommColl)) {
        const std::string grp = nb.comm_group_at(i).empty() ? std::string("world") : std::string(nb.comm_group_at(i));
        const std::string ct_str(ctype);
        const std::uint32_t seq = coll_seq_per_rank[r][grp + ":" + ct_str]++;
        coll_entries.push_back({grp, ct_str, seq, r, vid});
      }
    }

    const EdgeBatch& eb = chunk.edges;
    for (std::size_t e = 0; e < eb.size(); ++e) {
      e_raw_buckets.push(static_cast<std::size_t>(eb.rank[e] % kJoinBuckets),
                         RawEdgeRec{eb.rank[e], eb.edge_kind[e], {}, eb.src_id[e], eb.dst_id[e]});
    }
  }
  flush_vbuf();
  v_join_buckets.flush_all();
  e_raw_buckets.flush_all();
  p2p_buckets.flush_all();
  send_seq.clear();
  recv_seq.clear();
  coll_seq_per_rank.clear();

  // Optional virtual 2-node collective barriers (B_in, B_out): weight 0, flagged kFlagIsVirtual.
  std::uint64_t num_total = num_real;
  struct BarrierGroup {
    std::uint32_t b_in = kNone32;
    std::uint32_t b_out = kNone32;
    std::vector<std::uint32_t> coll_vids;
  };
  std::vector<BarrierGroup> barrier_groups;
  std::unordered_map<std::uint32_t, std::uint32_t> coll_to_b_in;
  std::unordered_map<std::uint32_t, std::uint32_t> coll_to_b_out;

  const std::uint32_t world_size = static_cast<std::uint32_t>(rank_compute.size());
  if (opts.collective_sync == CollectiveSync::kBarrier && !coll_entries.empty()) {
    std::map<std::tuple<std::string, std::string, std::uint32_t>, std::vector<std::uint32_t>> grouped;
    for (const auto& ce : coll_entries) grouped[{ce.group, ce.comm_type, ce.seq}].push_back(ce.vid);
    for (auto& [key, vids] : grouped) {
      if (vids.size() <= 1) continue;
      const std::uint32_t b_in = static_cast<std::uint32_t>(num_total++);
      const std::uint32_t b_out = static_cast<std::uint32_t>(num_total++);
      RawVertexRec vrec{};
      vrec.flags = kFlagIsVirtual;
      vrec.op_class = static_cast<std::uint8_t>(OpClass::kOther);
      write_exact(vfp, &vrec, 1);  // b_in
      write_exact(vfp, &vrec, 1);  // b_out
      for (std::uint32_t cv : vids) {
        coll_to_b_in[cv] = b_in;
        coll_to_b_out[cv] = b_out;
      }
      barrier_groups.push_back({b_in, b_out, std::move(vids)});
    }
  }
  coll_entries.clear();
  vguard.close();
  nguard.close();

  // Empty graph fast path.
  if (num_total == 0) {
    std::FILE* out = std::fopen(cache_path.c_str(), "wb");
    if (!out) throw std::runtime_error("cannot open graph cache for write: " + cache_path);
    FileGuard out_guard(out);
    GraphCacheHeader hdr{};
    hdr.goal_semantics = static_cast<std::uint8_t>(opts.goal_semantics);
    hdr.collective_sync = static_cast<std::uint8_t>(opts.collective_sync);
    write_exact(out, &hdr, 1);
    hdr.rank_stats_offset = static_cast<std::uint64_t>(std::ftell(out));
    std::uint32_t zero = 0;
    write_exact(out, &zero, 1);
    hdr.names_offset = static_cast<std::uint64_t>(std::ftell(out));
    write_exact(out, &zero, 1);
    hdr.comm_types_offset = static_cast<std::uint64_t>(std::ftell(out));
    write_exact(out, &zero, 1);
    hdr.witness_offset = static_cast<std::uint64_t>(std::ftell(out));
    write_exact(out, &zero, 1);
    hdr.block_index_offset = static_cast<std::uint64_t>(std::ftell(out));
    std::fseek(out, 0, SEEK_SET);
    write_exact(out, &hdr, 1);
    out_guard.close();
    return;
  }

  // Choose GridGraph 1D vertex partition count P_v and block size strictly
  // respecting `budget` (so 2 chunks of vertex state fit easily in RAM).
  const std::uint64_t max_chunk_verts =
      std::max<std::uint64_t>(1024, std::min<std::uint64_t>(4u << 20, budget / 256));
  const std::uint32_t pv = static_cast<std::uint32_t>(
      std::max<std::uint64_t>(1, std::min<std::uint64_t>(64, (num_total + max_chunk_verts - 1) / max_chunk_verts)));
  const std::uint32_t chunk_span = static_cast<std::uint32_t>((num_total + pv - 1) / pv);

  // Pass 0b: Resolve all edges (intra-rank + P2P + optional barriers) into
  // GridGraph 2D edge blocks `grid[src_chunk * pv + dst_chunk]`.
  BucketWriter<ResolvedEdge> grid_blocks(scratch, "grid", static_cast<std::size_t>(pv) * pv);

  auto emit_resolved = [&](std::uint32_t u, std::uint32_t v, std::uint8_t kind) {
    // When collective barriers are active, edges entering a collective go to B_in
    // and edges leaving a collective come from B_out (2P total barrier edges).
    if (!coll_to_b_in.empty()) {
      auto oit = coll_to_b_out.find(u);
      if (oit != coll_to_b_out.end()) u = oit->second;
      auto iit = coll_to_b_in.find(v);
      if (iit != coll_to_b_in.end()) v = iit->second;
    }
    const std::size_t ci = u / chunk_span;
    const std::size_t cj = v / chunk_span;
    grid_blocks.push(ci * pv + cj, ResolvedEdge{u, v, kind, {}});
  };

  // Emit virtual barrier internal edges: B_in -> coll_k -> B_out (2P edges/barrier).
  for (const auto& bg : barrier_groups) {
    for (std::uint32_t cv : bg.coll_vids) {
      const std::size_t ci_in = bg.b_in / chunk_span;
      const std::size_t cj_c = cv / chunk_span;
      grid_blocks.push(ci_in * pv + cj_c, ResolvedEdge{bg.b_in, cv, static_cast<std::uint8_t>(EdgeKind::kCollBarrier), {}});
      const std::size_t ci_c = cv / chunk_span;
      const std::size_t cj_out = bg.b_out / chunk_span;
      grid_blocks.push(ci_c * pv + cj_out, ResolvedEdge{cv, bg.b_out, static_cast<std::uint8_t>(EdgeKind::kCollBarrier), {}});
    }
  }

  // Resolve intra-rank edges.
  if (still_dense) {
    std::sort(dense_runs.begin(), dense_runs.end(),
              [](const DenseRun& a, const DenseRun& b) { return a.rank < b.rank; });
    auto find_run = [&](std::uint32_t rank) -> const DenseRun* {
      auto it = std::lower_bound(dense_runs.begin(), dense_runs.end(), rank,
                                 [](const DenseRun& dr, std::uint32_t k) { return dr.rank < k; });
      if (it == dense_runs.end() || it->rank != rank) return nullptr;
      return &*it;
    };
    for (std::size_t b = 0; b < kJoinBuckets; ++b) {
      auto edges = load_bucket<RawEdgeRec>(e_raw_buckets.path(b), e_raw_buckets.count(b));
      for (const RawEdgeRec& e : edges) {
        const DenseRun* run = find_run(e.rank);
        if (!run || e.src_id >= run->length || e.dst_id >= run->length) continue;
        emit_resolved(run->first_vid + static_cast<std::uint32_t>(e.src_id),
                      run->first_vid + static_cast<std::uint32_t>(e.dst_id), e.kind);
      }
    }
  } else {
    // Partitioned hash join on (rank, node_id) supporting duplicate rows & sparse 64-bit IDs.
    for (std::size_t b = 0; b < kJoinBuckets; ++b) {
      if (e_raw_buckets.count(b) == 0 || v_join_buckets.count(b) == 0) continue;
      auto verts = load_bucket<RankNodeVid>(v_join_buckets.path(b), v_join_buckets.count(b));
      std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> index;
      index.reserve(verts.size() * 2 + 1);
      // Key by exact (rank, node_id) via bucketed vector check for collision safety.
      struct Entry {
        std::uint32_t rank;
        std::uint64_t node_id;
        std::vector<std::uint32_t> vids;
      };
      std::unordered_map<std::uint64_t, std::vector<Entry>> table;
      for (const auto& rv : verts) {
        auto& chain = table[rank_node_hash(rv.rank, rv.node_id)];
        bool found = false;
        for (auto& en : chain) {
          if (en.rank == rv.rank && en.node_id == rv.node_id) {
            en.vids.push_back(rv.vid);
            found = true;
            break;
          }
        }
        if (!found) chain.push_back({rv.rank, rv.node_id, {rv.vid}});
      }
      auto find_vids = [&](std::uint32_t r, std::uint64_t nid) -> const std::vector<std::uint32_t>* {
        auto it = table.find(rank_node_hash(r, nid));
        if (it == table.end()) return nullptr;
        for (const auto& en : it->second)
          if (en.rank == r && en.node_id == nid) return &en.vids;
        return nullptr;
      };
      auto edges = load_bucket<RawEdgeRec>(e_raw_buckets.path(b), e_raw_buckets.count(b));
      for (const RawEdgeRec& e : edges) {
        const auto* srcs = find_vids(e.rank, e.src_id);
        if (!srcs) continue;
        const auto* dsts = find_vids(e.rank, e.dst_id);
        if (!dsts) continue;
        for (std::uint32_t u : *srcs)
          for (std::uint32_t v : *dsts) emit_resolved(u, v, e.kind);
      }
    }
  }

  // Resolve P2P send -> recv buckets in exact row order.
  std::uint64_t unmatched_sends = 0, unmatched_recvs = 0;
  for (std::size_t b = 0; b < kP2PBuckets; ++b) {
    if (p2p_buckets.count(b) == 0) continue;
    auto recs = load_bucket<P2PRec>(p2p_buckets.path(b), p2p_buckets.count(b));
    std::unordered_map<ChannelKey, std::vector<std::pair<std::uint32_t, std::uint32_t>>, ChannelKeyHash> sv, rv;
    for (const P2PRec& p : recs) {
      const ChannelKey ck{p.src_rank, p.dst_rank, p.tag};
      if (p.is_send) {
        sv[ck].emplace_back(p.seq, p.vid);
      } else {
        rv[ck].emplace_back(p.seq, p.vid);
      }
    }
    for (auto& [ck, sends] : sv) {
      std::sort(sends.begin(), sends.end());
      auto rit = rv.find(ck);
      if (rit == rv.end()) {
        unmatched_sends += sends.size();
        continue;
      }
      auto& recvs = rit->second;
      std::sort(recvs.begin(), recvs.end());
      const std::size_t pairs = std::min(sends.size(), recvs.size());
      unmatched_sends += (sends.size() - pairs);
      for (std::size_t k = 0; k < pairs; ++k) {
        emit_resolved(sends[k].second, recvs[k].second, static_cast<std::uint8_t>(EdgeKind::kP2P));
      }
    }
    for (const auto& [ck, recvs] : rv) {
      auto sit = sv.find(ck);
      const std::size_t scnt = sit == sv.end() ? 0 : sit->second.size();
      if (recvs.size() > scnt) unmatched_recvs += (recvs.size() - scnt);
    }
  }
  grid_blocks.flush_all();

  // Pass 0c: Deduplicate each 2D GridGraph block (u, v) independently and
  // compute exact deduplicated in_degree[v] and num_edges.
  const bool semi_external = (num_total * sizeof(std::uint32_t) * 3 <= budget);
  std::vector<std::uint32_t> in_degree;
  std::vector<std::string> ext_indeg_paths;
  if (semi_external) {
    in_degree.assign(static_cast<std::size_t>(num_total), 0);
  } else {
    ext_indeg_paths.resize(pv);
    for (std::uint32_t c = 0; c < pv; ++c) {
      ext_indeg_paths[c] = scratch.file("indeg_" + std::to_string(c) + ".bin");
      const std::uint32_t c_begin = c * chunk_span;
      const std::uint32_t c_end = static_cast<std::uint32_t>(std::min<std::uint64_t>(num_total, c_begin + chunk_span));
      std::vector<std::uint32_t> zeros(c_end - c_begin, 0);
      std::FILE* fp = std::fopen(ext_indeg_paths[c].c_str(), "wb");
      write_exact(fp, zeros.data(), zeros.size());
      std::fclose(fp);
    }
  }

  std::vector<std::string> dedup_grid_paths(static_cast<std::size_t>(pv) * pv);
  std::vector<std::uint64_t> dedup_grid_counts(static_cast<std::size_t>(pv) * pv, 0);
  std::uint64_t total_dedup_edges = 0;
  bool natural_order_is_topo = true;

  for (std::uint32_t cj = 0; cj < pv; ++cj) {
    std::vector<std::uint32_t> local_indeg;
    const std::uint32_t j_begin = cj * chunk_span;
    const std::uint32_t j_end = static_cast<std::uint32_t>(std::min<std::uint64_t>(num_total, j_begin + chunk_span));
    if (!semi_external) {
      local_indeg = load_bucket<std::uint32_t>(ext_indeg_paths[cj], j_end - j_begin);
    }
    for (std::uint32_t ci = 0; ci < pv; ++ci) {
      const std::size_t gid = static_cast<std::size_t>(ci) * pv + cj;
      if (grid_blocks.count(gid) == 0) continue;
      auto edges = load_bucket<ResolvedEdge>(grid_blocks.path(gid), grid_blocks.count(gid));
      std::sort(edges.begin(), edges.end(), [](const ResolvedEdge& a, const ResolvedEdge& b) {
        if (a.u != b.u) return a.u < b.u;
        if (a.v != b.v) return a.v < b.v;
        return edge_kind_priority(a.kind) < edge_kind_priority(b.kind);
      });
      auto last = std::unique(edges.begin(), edges.end(),
                              [](const ResolvedEdge& a, const ResolvedEdge& b) { return a.u == b.u && a.v == b.v; });
      edges.erase(last, edges.end());
      for (const ResolvedEdge& e : edges) {
        if (e.u >= e.v) natural_order_is_topo = false;
        if (semi_external) {
          ++in_degree[e.v];
        } else {
          ++local_indeg[e.v - j_begin];
        }
      }
      total_dedup_edges += edges.size();
      dedup_grid_counts[gid] = edges.size();
      dedup_grid_paths[gid] = scratch.file("dgrid_" + std::to_string(ci) + "_" + std::to_string(cj) + ".bin");
      std::FILE* dfp = std::fopen(dedup_grid_paths[gid].c_str(), "wb");
      write_exact(dfp, edges.data(), edges.size());
      std::fclose(dfp);
    }
    if (!semi_external) {
      std::FILE* fp = std::fopen(ext_indeg_paths[cj].c_str(), "wb");
      write_exact(fp, local_indeg.data(), local_indeg.size());
      std::fclose(fp);
    }
  }

  // Pass 0d: GridGraph selective-scheduled topological sort (Kahn with diagonal
  // in-RAM collapsing + selective off-diagonal sweeps).
  std::vector<std::uint32_t> orig_of;
  orig_of.reserve(static_cast<std::size_t>(num_total));
  std::vector<bool> peeled(static_cast<std::size_t>(num_total), false);

  if (natural_order_is_topo) {
    // Fast path: natural vertex order is already provably acyclic & topological!
    for (std::uint32_t u = 0; u < num_total; ++u) {
      orig_of.push_back(u);
      peeled[u] = true;
    }
  } else {
    // GridGraph selective-scheduled peeling: active chunks have >=1 zero-in-degree unpeeled vertex.
    std::vector<bool> chunk_active(pv, true);
    bool progress = true;
    while (progress) {
      progress = false;
      for (std::uint32_t ci = 0; ci < pv; ++ci) {
        if (!chunk_active[ci]) continue;
        chunk_active[ci] = false;
        const std::uint32_t i_begin = ci * chunk_span;
        const std::uint32_t i_end =
            static_cast<std::uint32_t>(std::min<std::uint64_t>(num_total, i_begin + chunk_span));
        const std::uint32_t i_len = i_end - i_begin;

        std::vector<std::uint32_t> cur_indeg;
        std::uint32_t* indeg_ptr = nullptr;
        if (semi_external) {
          indeg_ptr = in_degree.data() + i_begin;
        } else {
          cur_indeg = load_bucket<std::uint32_t>(ext_indeg_paths[ci], i_len);
          indeg_ptr = cur_indeg.data();
        }

        // Seed diagonal Kahn with all unpeeled zero-in-degree vertices in chunk ci.
        std::vector<std::uint32_t> newly_peeled;
        for (std::uint32_t off = 0; off < i_len; ++off) {
          if (!peeled[i_begin + off] && indeg_ptr[off] == 0) newly_peeled.push_back(i_begin + off);
        }
        if (newly_peeled.empty()) continue;

        // Build local CSR of diagonal block grid[ci][ci] to collapse intra-chunk chains in 1 step!
        const std::size_t diag_id = static_cast<std::size_t>(ci) * pv + ci;
        std::vector<std::uint64_t> d_ptr;
        std::vector<std::uint32_t> d_succ;
        if (dedup_grid_counts[diag_id] > 0) {
          auto dedges = load_bucket<ResolvedEdge>(dedup_grid_paths[diag_id], dedup_grid_counts[diag_id]);
          d_ptr.assign(i_len + 1, 0);
          for (const auto& e : dedges) ++d_ptr[(e.u - i_begin) + 1];
          for (std::uint32_t k = 0; k < i_len; ++k) d_ptr[k + 1] += d_ptr[k];
          d_succ.resize(dedges.size());
          std::vector<std::uint64_t> cur(d_ptr.begin(), d_ptr.end() - 1);
          for (const auto& e : dedges) d_succ[cur[e.u - i_begin]++] = e.v;
        }

        for (std::size_t head = 0; head < newly_peeled.size(); ++head) {
          const std::uint32_t u = newly_peeled[head];
          if (peeled[u]) continue;
          peeled[u] = true;
          orig_of.push_back(u);
          if (!d_ptr.empty()) {
            const std::uint32_t lu = u - i_begin;
            for (std::uint64_t e = d_ptr[lu]; e < d_ptr[lu + 1]; ++e) {
              const std::uint32_t v = d_succ[e];
              if (!peeled[v] && --indeg_ptr[v - i_begin] == 0) newly_peeled.push_back(v);
            }
          }
        }
        if (!semi_external) {
          std::FILE* fp = std::fopen(ext_indeg_paths[ci].c_str(), "wb");
          write_exact(fp, cur_indeg.data(), cur_indeg.size());
          std::fclose(fp);
        }

        progress = true;
        std::unordered_set<std::uint32_t> batch_set(newly_peeled.begin(), newly_peeled.end());

        // Stream active off-diagonal blocks grid[ci][cj] sequentially to update chunk cj!
        for (std::uint32_t cj = 0; cj < pv; ++cj) {
          if (cj == ci) continue;
          const std::size_t gid = static_cast<std::size_t>(ci) * pv + cj;
          if (dedup_grid_counts[gid] == 0) continue;
          const std::uint32_t j_begin = cj * chunk_span;
          const std::uint32_t j_end =
              static_cast<std::uint32_t>(std::min<std::uint64_t>(num_total, j_begin + chunk_span));
          std::vector<std::uint32_t> j_indeg;
          std::uint32_t* j_ptr = semi_external ? (in_degree.data() + j_begin) : nullptr;
          if (!semi_external) {
            j_indeg = load_bucket<std::uint32_t>(ext_indeg_paths[cj], j_end - j_begin);
            j_ptr = j_indeg.data();
          }
          auto oedges = load_bucket<ResolvedEdge>(dedup_grid_paths[gid], dedup_grid_counts[gid]);
          bool activated = false;
          for (const auto& e : oedges) {
            if (!batch_set.count(e.u)) continue;
            if (!peeled[e.v] && --j_ptr[e.v - j_begin] == 0) activated = true;
          }
          if (!semi_external) {
            std::FILE* fp = std::fopen(ext_indeg_paths[cj].c_str(), "wb");
            write_exact(fp, j_indeg.data(), j_indeg.size());
            std::fclose(fp);
          }
          if (activated) chunk_active[cj] = true;
        }
      }
    }
  }

  // Handle cycles: extract sample witness + deterministic ascending fallback.
  const std::uint64_t unordered = num_total - orig_of.size();
  std::vector<std::uint32_t> witness_rank;
  std::vector<std::uint64_t> witness_node_id;
  if (unordered > 0) {
    // Build one unpeeled predecessor per unpeeled vertex across the grid.
    std::unordered_map<std::uint32_t, std::uint32_t> cyc_pred;
    for (std::uint32_t ci = 0; ci < pv; ++ci) {
      for (std::uint32_t cj = 0; cj < pv; ++cj) {
        const std::size_t gid = static_cast<std::size_t>(ci) * pv + cj;
        if (dedup_grid_counts[gid] == 0) continue;
        auto edges = load_bucket<ResolvedEdge>(dedup_grid_paths[gid], dedup_grid_counts[gid]);
        for (const auto& e : edges) {
          if (!peeled[e.u] && !peeled[e.v]) cyc_pred.try_emplace(e.v, e.u);
        }
      }
    }
    if (!cyc_pred.empty()) {
      // Load vertex rank/node_id lazily for the cycle witness only.
      std::FILE* rvfp = std::fopen(raw_verts_path.c_str(), "rb");
      auto lookup_raw = [&](std::uint32_t vid) -> RawVertexRec {
        RawVertexRec r{};
        std::fseek(rvfp, static_cast<long>(static_cast<std::uint64_t>(vid) * sizeof(RawVertexRec)), SEEK_SET);
        read_exact(rvfp, &r, 1);
        return r;
      };
      std::uint32_t cur = cyc_pred.begin()->first;
      std::unordered_map<std::uint32_t, std::size_t> seen_at;
      std::vector<std::uint32_t> trace;
      while (true) {
        auto it = seen_at.find(cur);
        if (it != seen_at.end()) {
          for (std::size_t k = trace.size(); k-- > it->second;) {
            const RawVertexRec rv = lookup_raw(trace[k]);
            if ((rv.flags & kFlagIsVirtual) == 0) {
              witness_rank.push_back(rv.rank);
              witness_node_id.push_back(rv.node_id);
            }
          }
          break;
        }
        seen_at[cur] = trace.size();
        trace.push_back(cur);
        auto pit = cyc_pred.find(cur);
        if (pit == cyc_pred.end()) break;
        cur = pit->second;
      }
      std::fclose(rvfp);
    }
    if (opts.fail_on_cycle) {
      throw std::runtime_error("cycle detected in execution graph (" + std::to_string(unordered) +
                               " unordered vertices)");
    }
    for (std::uint32_t u = 0; u < num_total; ++u) {
      if (!peeled[u]) orig_of.push_back(u);
    }
  }

  // Inverse permutation: pos_of[orig_vid] = topological position.
  std::vector<std::uint32_t> pos_of(static_cast<std::size_t>(num_total));
  for (std::uint32_t p = 0; p < num_total; ++p) pos_of[orig_of[p]] = p;

  // Pass 0e: Remap all deduplicated edges to topological positions (p < q),
  // compute exact last_use_pos[p] and partition by destination topological block!
  const std::uint32_t block_verts = static_cast<std::uint32_t>(
      std::max<std::uint64_t>(1024, std::min<std::uint64_t>(1u << 20, budget / 512)));
  const std::uint32_t num_blocks = static_cast<std::uint32_t>((num_total + block_verts - 1) / block_verts);
  BucketWriter<ResolvedEdge> dst_block_edges(scratch, "tblock", num_blocks);

  std::vector<std::uint32_t> last_use(static_cast<std::size_t>(num_total));
  for (std::uint32_t p = 0; p < num_total; ++p) last_use[p] = p;

  for (std::uint32_t ci = 0; ci < pv; ++ci) {
    for (std::uint32_t cj = 0; cj < pv; ++cj) {
      const std::size_t gid = static_cast<std::size_t>(ci) * pv + cj;
      if (dedup_grid_counts[gid] == 0) continue;
      auto edges = load_bucket<ResolvedEdge>(dedup_grid_paths[gid], dedup_grid_counts[gid]);
      for (const auto& e : edges) {
        const std::uint32_t p = pos_of[e.u];
        const std::uint32_t q = pos_of[e.v];
        if (p < q) {
          if (q > last_use[p]) last_use[p] = q;
          dst_block_edges.push(q / block_verts, ResolvedEdge{p, q, e.kind, {}});
        }
      }
    }
  }
  dst_block_edges.flush_all();

  // Compute max_cutwidth (number of live intervals [p, last_use[p]] crossing any point).
  std::vector<std::int64_t> cut_diff(static_cast<std::size_t>(num_total) + 1, 0);
  for (std::uint32_t p = 0; p < num_total; ++p) {
    if (last_use[p] > p) {
      ++cut_diff[p];
      --cut_diff[last_use[p]];
    }
  }
  std::uint64_t cur_cut = 0, max_cut = 0;
  for (std::uint32_t p = 0; p < num_total; ++p) {
    cur_cut = static_cast<std::uint64_t>(static_cast<std::int64_t>(cur_cut) + cut_diff[p]);
    if (cur_cut > max_cut) max_cut = cur_cut;
  }
  cut_diff.clear();
  cut_diff.shrink_to_fit();

  // Write the final `.cta_graph` container sequentially block by block!
  std::FILE* out = std::fopen(cache_path.c_str(), "wb");
  if (!out) throw std::runtime_error("cannot open graph cache for write: " + cache_path);
  FileGuard out_guard(out);
  GraphCacheHeader hdr{};
  hdr.num_real_vertices = num_real;
  hdr.num_total_vertices = num_total;
  hdr.num_edges = total_dedup_edges;
  hdr.unordered = unordered;
  hdr.num_blocks = num_blocks;
  hdr.block_vertices = block_verts;
  hdr.world_size = world_size;
  hdr.goal_semantics = static_cast<std::uint8_t>(opts.goal_semantics);
  hdr.collective_sync = static_cast<std::uint8_t>(opts.collective_sync);
  hdr.has_recorded_comm_times = has_recorded_comm ? 1 : 0;
  hdr.unmatched_p2p_sends = unmatched_sends;
  hdr.unmatched_p2p_recvs = unmatched_recvs;
  hdr.max_cutwidth = max_cut;
  write_exact(out, &hdr, 1);

  std::FILE* rvfp = std::fopen(raw_verts_path.c_str(), "rb");
  std::FILE* snfp = std::fopen(spill_names_path.c_str(), "rb");
  FileGuard rguard(rvfp), sguard(snfp);
  if (!rvfp || !snfp) throw std::runtime_error("cannot reopen scratch raw vertices/names");

  // Also load raw vertices into memory if they fit in budget/3 (fast path), else seek per vertex.
  const bool raw_verts_in_ram = (num_total * sizeof(RawVertexRec) <= budget / 3);
  std::vector<RawVertexRec> all_raw;
  if (raw_verts_in_ram) {
    all_raw.resize(static_cast<std::size_t>(num_total));
    read_exact(rvfp, all_raw.data(), all_raw.size());
  }

  std::vector<BlockIndexEntry> block_index(num_blocks);
  for (std::uint32_t b = 0; b < num_blocks; ++b) {
    const std::uint32_t pos_begin = b * block_verts;
    const std::uint32_t pos_end = static_cast<std::uint32_t>(std::min<std::uint64_t>(num_total, pos_begin + block_verts));
    const std::size_t len = pos_end - pos_begin;

    std::vector<TopoVertexMeta> b_verts(len);
    std::vector<std::uint32_t> local_name_end(len, 0);
    std::vector<char> local_name_bytes;

    for (std::size_t i = 0; i < len; ++i) {
      const std::uint32_t p = pos_begin + static_cast<std::uint32_t>(i);
      const std::uint32_t orig = orig_of[p];
      RawVertexRec raw{};
      if (raw_verts_in_ram) {
        raw = all_raw[orig];
      } else {
        std::fseek(rvfp, static_cast<long>(static_cast<std::uint64_t>(orig) * sizeof(RawVertexRec)), SEEK_SET);
        read_exact(rvfp, &raw, 1);
      }
      TopoVertexMeta m{};
      m.orig_vid = orig;
      m.rank = raw.rank;
      m.node_id = raw.node_id;
      m.weight_ns = raw.weight_ns;
      m.duration_ns = raw.duration_ns;
      m.size_bytes = raw.size_bytes;
      m.last_use_pos = last_use[p];
      m.name_id = raw.name_id;
      m.comm_type_id = raw.comm_type_id;
      m.flags = raw.flags;
      m.op_class = raw.op_class;
      m.src_rank = raw.src_rank;
      m.dst_rank = raw.dst_rank;
      b_verts[i] = m;

      if (raw.name_id < 0 && raw.spill_name_len > 0) {
        const std::size_t prev = local_name_bytes.size();
        local_name_bytes.resize(prev + raw.spill_name_len);
        std::fseek(snfp, static_cast<long>(raw.spill_name_off), SEEK_SET);
        read_exact(snfp, local_name_bytes.data() + prev, raw.spill_name_len);
      }
      local_name_end[i] = static_cast<std::uint32_t>(local_name_bytes.size());
    }

    auto bedges = load_bucket<ResolvedEdge>(dst_block_edges.path(b), dst_block_edges.count(b));
    std::vector<std::uint64_t> pred_ptr(len + 1, 0);
    for (const auto& e : bedges) ++pred_ptr[(e.v - pos_begin) + 1];
    for (std::size_t i = 0; i < len; ++i) pred_ptr[i + 1] += pred_ptr[i];
    std::vector<std::pair<std::uint32_t, std::uint8_t>> packed(bedges.size());
    std::vector<std::uint64_t> cur(pred_ptr.begin(), pred_ptr.end() - 1);
    for (const auto& e : bedges) packed[cur[e.v - pos_begin]++] = {e.u, e.kind};
    // Sort each vertex's predecessors ascending by topological position.
    for (std::size_t i = 0; i < len; ++i) {
      std::sort(packed.begin() + static_cast<std::ptrdiff_t>(pred_ptr[i]),
                packed.begin() + static_cast<std::ptrdiff_t>(pred_ptr[i + 1]),
                [](const auto& a, const auto& b) { return a.first < b.first; });
    }
    std::vector<std::uint32_t> preds(packed.size());
    std::vector<std::uint8_t> pred_kind(packed.size());
    for (std::size_t k = 0; k < packed.size(); ++k) {
      preds[k] = packed[k].first;
      pred_kind[k] = packed[k].second;
    }

    block_index[b].file_offset = static_cast<std::uint64_t>(std::ftell(out));
    block_index[b].pos_begin = pos_begin;
    block_index[b].pos_end = pos_end;
    block_index[b].num_edges = preds.size();
    block_index[b].local_names_bytes = local_name_bytes.size();

    write_exact(out, b_verts.data(), len);
    write_exact(out, pred_ptr.data(), len + 1);
    write_exact(out, preds.data(), preds.size());
    write_exact(out, pred_kind.data(), pred_kind.size());
    write_exact(out, local_name_end.data(), len);
    write_exact(out, local_name_bytes.data(), local_name_bytes.size());
  }
  rguard.close();
  sguard.close();

  // Append rank stats, string tables, cycle witness, and block index, then patch header.
  hdr.rank_stats_offset = static_cast<std::uint64_t>(std::ftell(out));
  const std::uint32_t nr = static_cast<std::uint32_t>(rank_compute.size());
  write_exact(out, &nr, 1);
  for (const auto& [r, rce] : rank_compute) write_exact(out, &rce, 1);

  hdr.names_offset = static_cast<std::uint64_t>(std::ftell(out));
  write_string_table(out, name_interner.items());

  hdr.comm_types_offset = static_cast<std::uint64_t>(std::ftell(out));
  write_string_table(out, comm_type_interner.items());

  hdr.witness_offset = static_cast<std::uint64_t>(std::ftell(out));
  const std::uint32_t nw = static_cast<std::uint32_t>(witness_rank.size());
  write_exact(out, &nw, 1);
  write_exact(out, witness_rank.data(), nw);
  write_exact(out, witness_node_id.data(), nw);

  hdr.block_index_offset = static_cast<std::uint64_t>(std::ftell(out));
  write_exact(out, block_index.data(), block_index.size());

  if (std::fseek(out, 0, SEEK_SET) != 0) throw std::runtime_error("seek failed patching .cta_graph header");
  write_exact(out, &hdr, 1);
  out_guard.close();
}

}  // namespace cta::streaming::cache
