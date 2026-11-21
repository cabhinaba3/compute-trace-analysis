#include "streaming/sources/source.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace cta::streaming {
namespace {

constexpr std::uint64_t kGoalMagicCookie = 4223;
constexpr std::uint64_t kGoalMagicInvalid = 2342;
constexpr std::int8_t kOpSend = 1;
constexpr std::int8_t kOpRecv = 2;
constexpr std::int8_t kOpCalc = 3;
constexpr std::size_t kRecordBytes = 39;

template <typename T>
T load_le(const std::uint8_t* p) {
  T v{};
  std::memcpy(&v, p, sizeof(T));
  return v;
}

GoalBinNode decode_record(const std::uint8_t* p) {
  GoalBinNode n;
  n.deps_cnt = load_le<std::uint32_t>(p + 0);
  n.op_type = load_le<std::int8_t>(p + 4);
  n.peer = load_le<std::uint32_t>(p + 5);
  n.size = load_le<std::uint64_t>(p + 9);
  n.tag = load_le<std::uint32_t>(p + 17);
  n.proc = p[21];
  n.nic = p[22];
  n.num_deps = load_le<std::uint32_t>(p + 23);
  n.deps_apdx = load_le<std::uint32_t>(p + 27);
  n.num_startdeps = load_le<std::uint32_t>(p + 31);
  n.startdeps_apdx = load_le<std::uint32_t>(p + 35);
  return n;
}

bool read_exact(int fd, std::uint8_t* dst, std::size_t n) {
  std::size_t got = 0;
  while (got < n) {
    const ssize_t r = ::read(fd, dst + got, n - got);
    if (r < 0) throw std::runtime_error("read failed on GOAL binary stream");
    if (r == 0) return false;
    got += static_cast<std::size_t>(r);
  }
  return true;
}

void push_goal_bin_node(const GoalBinNode& nr, std::uint32_t rank, std::uint64_t node_id, NodeBatch& out) {
  const bool is_send = nr.op_type == kOpSend;
  const bool is_recv = nr.op_type == kOpRecv;
  const bool is_calc = nr.op_type == kOpCalc;
  const OpClass oc = is_send ? OpClass::kCommSend : is_recv ? OpClass::kCommRecv : OpClass::kCompute;
  const std::string_view name = is_send ? "comm_send" : is_recv ? "comm_recv" : "compute";
  const std::string_view comm_type = is_send ? "SEND" : is_recv ? "RECV" : "";
  const std::uint64_t dur_ns = is_calc ? nr.size : 0;
  const std::uint64_t bytes = is_calc ? 0 : nr.size;
  const std::int32_t r_i32 = static_cast<std::int32_t>(rank);
  const std::int32_t peer_i32 = static_cast<std::int32_t>(nr.peer);
  const std::int32_t src_r = is_send ? r_i32 : is_recv ? peer_i32 : -1;
  const std::int32_t dst_r = is_send ? peer_i32 : is_recv ? r_i32 : -1;
  out.push(rank, node_id, name, oc, comm_type, dur_ns, bytes, 0, src_r, dst_r, nr.tag);
}

// Streaming source backed by an mmapped GoalBinFile (seekable, O(1) per rank).
class MappedGoalBinSource final : public GraphSource {
 public:
  explicit MappedGoalBinSource(const std::string& path) : file_(path) {}

  bool next_chunk(GraphChunk& out, std::size_t max_nodes) override {
    out.clear();
    if (max_nodes == 0) max_nodes = 1;
    const std::uint32_t nranks = file_.num_ranks();

    while (rank_idx_ < nranks && out.nodes.size() < max_nodes) {
      const GoalBinRankView& rv = file_.rank_view(rank_idx_);
      if (node_cursor_ >= rv.num_nodes) {
        ++rank_idx_;
        node_cursor_ = 0;
        continue;
      }
      const std::size_t room = max_nodes - out.nodes.size();
      const std::uint32_t avail = rv.num_nodes - node_cursor_;
      const std::uint32_t take = static_cast<std::uint32_t>(std::min<std::size_t>(room, avail));
      const std::uint32_t end_node = node_cursor_ + take;

      // Emit nodes first, then requires edges, then irequires edges for this slice
      // (preserving exact row order of _bin_edges_frames when a rank fits in one chunk).
      std::size_t req_edges = 0, ireq_edges = 0;
      for (std::uint32_t i = node_cursor_; i < end_node; ++i) {
        const GoalBinNode nr = rv.node(i);
        push_goal_bin_node(nr, rv.rank, i, out.nodes);
        req_edges += nr.num_deps;
        ireq_edges += nr.num_startdeps;
      }
      if (req_edges > 0) {
        out.edges.reserve(out.edges.size() + req_edges + ireq_edges);
        for (std::uint32_t i = node_cursor_; i < end_node; ++i) {
          const GoalBinNode nr = rv.node(i);
          if (nr.num_deps == 0) continue;
          const std::size_t end_apdx = static_cast<std::size_t>(nr.deps_apdx) + nr.num_deps;
          if (end_apdx > rv.appendix_len) throw std::invalid_argument("GOAL binary requires appendix out of bounds");
          for (std::uint32_t k = 0; k < nr.num_deps; ++k) {
            out.edges.push(rv.rank, i, rv.appendix(static_cast<std::size_t>(nr.deps_apdx) + k), EdgeKind::kRequires);
          }
        }
      }
      if (ireq_edges > 0) {
        for (std::uint32_t i = node_cursor_; i < end_node; ++i) {
          const GoalBinNode nr = rv.node(i);
          if (nr.num_startdeps == 0) continue;
          const std::size_t end_apdx = static_cast<std::size_t>(nr.startdeps_apdx) + nr.num_startdeps;
          if (end_apdx > rv.appendix_len) throw std::invalid_argument("GOAL binary irequires appendix out of bounds");
          for (std::uint32_t k = 0; k < nr.num_startdeps; ++k) {
            out.edges.push(rv.rank, i, rv.appendix(static_cast<std::size_t>(nr.startdeps_apdx) + k),
                           EdgeKind::kIrequires);
          }
        }
      }

      node_cursor_ = end_node;
      if (node_cursor_ >= rv.num_nodes) {
        ++rank_idx_;
        node_cursor_ = 0;
      }
    }
    return !out.empty();
  }

  bool is_seekable() const override { return true; }
  void reset() override {
    rank_idx_ = 0;
    node_cursor_ = 0;
  }
  std::string format_name() const override { return "goal_bin"; }
  std::uint64_t estimated_bytes() const override { return file_.file_size(); }

 private:
  GoalBinFile file_;
  std::uint32_t rank_idx_ = 0;
  std::uint32_t node_cursor_ = 0;
};

// Sequential streaming source for unseekable GOAL .bin pipes (reads one rank
// block at a time in order; suitable for Tier 3 aggregate streaming).
class PipeGoalBinSource final : public GraphSource {
 public:
  explicit PipeGoalBinSource(const std::string& path) {
    if (path == "-") {
      fd_ = ::dup(STDIN_FILENO);
      if (fd_ < 0) throw std::runtime_error("cannot dup stdin for GOAL binary stream");
    } else {
      fd_ = ::open(path.c_str(), O_RDONLY);
      if (fd_ < 0) throw std::runtime_error("cannot open " + path);
    }
    // A throwing constructor never runs the destructor, so close the descriptor here.
    try {
      read_header(path);
    } catch (...) {
      ::close(fd_);
      fd_ = -1;
      throw;
    }
  }

  ~PipeGoalBinSource() override {
    if (fd_ >= 0) ::close(fd_);
  }

  // A header that is cut short is an error, as it is for a mapped file: an empty
  // or truncated pipe is not an empty graph.
  void read_header(const std::string& path) {
    std::uint8_t head[14];
    if (!read_exact(fd_, head, 14)) throw std::invalid_argument("truncated GOAL binary header in " + path);
    const std::uint64_t cookie = load_le<std::uint64_t>(head);
    if (cookie != kGoalMagicCookie && cookie != kGoalMagicInvalid)
      throw std::invalid_argument("Invalid GOAL binary magic cookie in " + path);
    num_ranks_ = load_le<std::uint32_t>(head + 8);
    // Read the rank table in steps so a bogus rank count cannot allocate gigabytes
    // before the data is known to exist.
    constexpr std::size_t kStep = std::size_t{1} << 16;
    const std::size_t total = static_cast<std::size_t>(num_ranks_) * 2;
    for (std::size_t done = 0; done < total;) {
      const std::size_t take = std::min(kStep, total - done);
      bounds_.resize(done + take);
      if (!read_exact(fd_, reinterpret_cast<std::uint8_t*>(bounds_.data() + done), take * sizeof(std::uint64_t)))
        throw std::invalid_argument("truncated GOAL binary rank table in " + path);
      done += take;
    }
    stream_pos_ = 14 + bounds_.size() * sizeof(std::uint64_t);
  }

  bool next_chunk(GraphChunk& out, std::size_t max_nodes) override {
    out.clear();
    if (eof_) return false;
    if (max_nodes == 0) max_nodes = 1;

    while (out.nodes.size() < max_nodes) {
      if (cur_node_ >= cur_num_nodes_) {
        if (!load_next_rank()) {
          eof_ = true;
          break;
        }
        continue;
      }
      const std::uint32_t take =
          static_cast<std::uint32_t>(std::min<std::size_t>(max_nodes - out.nodes.size(), cur_num_nodes_ - cur_node_));
      const std::uint32_t end_node = cur_node_ + take;
      const std::uint8_t* rec_base = rank_buf_.data() + 8 + static_cast<std::size_t>(cur_num_roots_) * 4;
      const std::size_t apdx_off = 8 + static_cast<std::size_t>(cur_num_roots_) * 4 +
                                   static_cast<std::size_t>(cur_num_nodes_) * kRecordBytes;
      const std::uint8_t* apdx_base = rank_buf_.data() + apdx_off;
      const std::size_t apdx_len = (rank_buf_.size() - apdx_off) / 4;

      for (std::uint32_t i = cur_node_; i < end_node; ++i) {
        const GoalBinNode nr = decode_record(rec_base + static_cast<std::size_t>(i) * kRecordBytes);
        push_goal_bin_node(nr, cur_rank_, i, out.nodes);
      }
      for (std::uint32_t i = cur_node_; i < end_node; ++i) {
        const GoalBinNode nr = decode_record(rec_base + static_cast<std::size_t>(i) * kRecordBytes);
        if (nr.num_deps == 0) continue;
        if (static_cast<std::size_t>(nr.deps_apdx) + nr.num_deps > apdx_len)
          throw std::invalid_argument("GOAL binary requires appendix out of bounds");
        for (std::uint32_t k = 0; k < nr.num_deps; ++k) {
          out.edges.push(cur_rank_, i, load_le<std::uint32_t>(apdx_base + (static_cast<std::size_t>(nr.deps_apdx) + k) * 4), EdgeKind::kRequires);
        }
      }
      for (std::uint32_t i = cur_node_; i < end_node; ++i) {
        const GoalBinNode nr = decode_record(rec_base + static_cast<std::size_t>(i) * kRecordBytes);
        if (nr.num_startdeps == 0) continue;
        if (static_cast<std::size_t>(nr.startdeps_apdx) + nr.num_startdeps > apdx_len)
          throw std::invalid_argument("GOAL binary irequires appendix out of bounds");
        for (std::uint32_t k = 0; k < nr.num_startdeps; ++k) {
          out.edges.push(cur_rank_, i, load_le<std::uint32_t>(apdx_base + (static_cast<std::size_t>(nr.startdeps_apdx) + k) * 4),
                         EdgeKind::kIrequires);
        }
      }
      cur_node_ = end_node;
    }
    return !out.empty();
  }

  bool is_seekable() const override { return false; }
  void reset() override { throw std::runtime_error("cannot reset unseekable GOAL binary stream"); }
  std::string format_name() const override { return "goal_bin"; }
  std::uint64_t estimated_bytes() const override { return 0; }

 private:
  bool load_next_rank() {
    while (next_rank_ < num_ranks_) {
      const std::uint32_t r = next_rank_++;
      const std::uint64_t start_abs = 8 + bounds_[static_cast<std::size_t>(r) * 2];
      const std::uint64_t end_abs = 8 + bounds_[static_cast<std::size_t>(r) * 2 + 1];
      if (end_abs <= start_abs) continue;
      if (start_abs < stream_pos_) throw std::invalid_argument("GOAL binary pipe has out-of-order rank offsets");
      if (start_abs > stream_pos_) {
        std::vector<std::uint8_t> skip(static_cast<std::size_t>(start_abs - stream_pos_));
        if (!read_exact(fd_, skip.data(), skip.size())) throw std::invalid_argument("truncated GOAL binary stream");
        stream_pos_ = start_abs;
      }
      const std::size_t len = static_cast<std::size_t>(end_abs - start_abs);
      if (len < 8) throw std::invalid_argument("truncated GOAL binary rank header");
      rank_buf_.resize(len);
      if (!read_exact(fd_, rank_buf_.data(), len)) throw std::invalid_argument("truncated GOAL binary rank block");
      stream_pos_ = end_abs;
      cur_rank_ = r;
      cur_num_nodes_ = load_le<std::uint32_t>(rank_buf_.data());
      cur_num_roots_ = load_le<std::uint32_t>(rank_buf_.data() + 4);
      cur_node_ = 0;
      const std::size_t min_len = 8 + static_cast<std::size_t>(cur_num_roots_) * 4 +
                                  static_cast<std::size_t>(cur_num_nodes_) * kRecordBytes;
      if (min_len > len) throw std::invalid_argument("corrupt GOAL binary rank block length");
      if (cur_num_nodes_ > 0) return true;
    }
    return false;
  }

  int fd_ = -1;
  bool eof_ = false;
  std::uint32_t num_ranks_ = 0;
  std::vector<std::uint64_t> bounds_;
  std::uint64_t stream_pos_ = 0;
  std::uint32_t next_rank_ = 0;
  std::uint32_t cur_rank_ = 0;
  std::uint32_t cur_num_nodes_ = 0;
  std::uint32_t cur_num_roots_ = 0;
  std::uint32_t cur_node_ = 0;
  std::vector<std::uint8_t> rank_buf_;
};

}  // namespace

GoalBinNode GoalBinRankView::node(std::uint32_t idx) const {
  return decode_record(records_bytes + static_cast<std::size_t>(idx) * kRecordBytes);
}

std::uint32_t GoalBinRankView::root(std::uint32_t idx) const {
  return load_le<std::uint32_t>(roots_bytes + static_cast<std::size_t>(idx) * 4);
}

std::uint32_t GoalBinRankView::appendix(std::size_t idx) const {
  return load_le<std::uint32_t>(appendix_bytes + idx * 4);
}

GoalBinFile::GoalBinFile(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + path);
  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    throw std::runtime_error("cannot stat " + path);
  }
  size_ = static_cast<std::size_t>(st.st_size);
  if (size_ < 14) {
    ::close(fd);
    throw std::invalid_argument("GOAL binary file too small: " + path);
  }
  void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (p == MAP_FAILED) throw std::runtime_error("cannot map " + path);
  data_ = static_cast<const std::uint8_t*>(p);

  const std::uint64_t cookie = load_le<std::uint64_t>(data_);
  if (cookie != kGoalMagicCookie && cookie != kGoalMagicInvalid) {
    ::munmap(const_cast<std::uint8_t*>(data_), size_);
    data_ = nullptr;
    throw std::invalid_argument("Invalid GOAL binary magic cookie " + std::to_string(cookie) + " in " + path);
  }
  constexpr std::size_t kMapOffset = 8;
  const std::uint32_t num_ranks = load_le<std::uint32_t>(data_ + kMapOffset);
  const std::size_t table_end = kMapOffset + 6 + static_cast<std::size_t>(num_ranks) * 16;
  if (table_end > size_) {
    ::munmap(const_cast<std::uint8_t*>(data_), size_);
    data_ = nullptr;
    throw std::invalid_argument("truncated GOAL binary rank table in " + path);
  }

  ranks_.resize(num_ranks);
  for (std::uint32_t r = 0; r < num_ranks; ++r) {
    const std::size_t entry = kMapOffset + 6 + static_cast<std::size_t>(r) * 16;
    const std::uint64_t start_rel = load_le<std::uint64_t>(data_ + entry);
    const std::uint64_t end_rel = load_le<std::uint64_t>(data_ + entry + 8);
    GoalBinRankView& rv = ranks_[r];
    rv.rank = r;
    if (end_rel <= start_rel) continue;
    const std::uint64_t start_abs = kMapOffset + start_rel;
    const std::uint64_t end_abs = kMapOffset + end_rel;
    if (end_abs > size_ || end_abs - start_abs < 8) {
      ::munmap(const_cast<std::uint8_t*>(data_), size_);
      data_ = nullptr;
      throw std::invalid_argument("corrupt GOAL binary rank bounds in " + path);
    }
    const std::uint32_t num_nodes = load_le<std::uint32_t>(data_ + start_abs);
    const std::uint32_t num_roots = load_le<std::uint32_t>(data_ + start_abs + 4);
    const std::uint64_t roots_off = start_abs + 8;
    const std::uint64_t nodes_off = roots_off + static_cast<std::uint64_t>(num_roots) * 4;
    const std::uint64_t apdx_off = nodes_off + static_cast<std::uint64_t>(num_nodes) * kRecordBytes;
    if (apdx_off > end_abs) {
      ::munmap(const_cast<std::uint8_t*>(data_), size_);
      data_ = nullptr;
      throw std::invalid_argument("corrupt GOAL binary node section in " + path);
    }
    rv.num_nodes = num_nodes;
    rv.num_roots = num_roots;
    rv.roots_bytes = data_ + roots_off;
    rv.records_bytes = data_ + nodes_off;
    rv.appendix_bytes = data_ + apdx_off;
    rv.appendix_len = static_cast<std::size_t>((end_abs - apdx_off) / 4);
  }
}

GoalBinFile::~GoalBinFile() {
  if (data_) ::munmap(const_cast<std::uint8_t*>(data_), size_);
}

std::uint64_t GoalBinFile::total_nodes() const {
  std::uint64_t total = 0;
  for (const auto& rv : ranks_) total += rv.num_nodes;
  return total;
}

std::unique_ptr<GraphSource> open_goal_bin_stream(const std::string& path) {
  if (is_seekable_path(path)) return std::make_unique<MappedGoalBinSource>(path);
  return std::make_unique<PipeGoalBinSource>(path);
}

}  // namespace cta::streaming
