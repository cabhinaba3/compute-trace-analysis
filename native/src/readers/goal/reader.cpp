#include "readers/goal/reader.hpp"

#include <fcntl.h>
#include <simdjson.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "util/cpus.hpp"
#include "util/util.hpp"
#include "util/ordered_pool.hpp"
#include "readers/goal/syntax.hpp"

namespace cta::readers {
namespace {

using goal::Op;
using goal::OpKind;

// A read-only view of a file in memory.
class MappedFile {
 public:
  explicit MappedFile(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st {};
    if (::fstat(fd, &st) != 0) {
      ::close(fd);
      throw std::runtime_error("cannot stat " + path);
    }
    size_ = static_cast<std::size_t>(st.st_size);
    if (size_ > 0) {  // mmap rejects an empty file
      void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
      if (p == MAP_FAILED) {
        ::close(fd);
        throw std::runtime_error("cannot map " + path);
      }
      data_ = static_cast<const char*>(p);
    }
    ::close(fd);
  }
  ~MappedFile() {
    if (data_) ::munmap(const_cast<char*>(data_), size_);
  }
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  const char* data() const { return data_; }
  std::size_t size() const { return size_; }

 private:
  const char* data_ = nullptr;
  std::size_t size_ = 0;
};

// Copies of labels, so a label from a line rewritten by comment removal
// outlives that line. Freed together when a rank's labels are forgotten.
class Arena {
 public:
  std::string_view copy(std::string_view s) {
    if (s.size() > kBlock - used_ || blocks_.empty()) {
      if (s.size() > kBlock) {
        big_.emplace_back(new char[s.size()]);
        std::memcpy(big_.back().get(), s.data(), s.size());
        return {big_.back().get(), s.size()};
      }
      if (next_ == blocks_.size()) blocks_.emplace_back(new char[kBlock]);
      current_ = next_++;
      used_ = 0;
    }
    char* dst = blocks_[current_].get() + used_;
    std::memcpy(dst, s.data(), s.size());
    used_ += s.size();
    return {dst, s.size()};
  }
  void clear() {
    next_ = 0;
    used_ = kBlock;  // forces the next copy to take a block
    big_.clear();
  }

 private:
  static constexpr std::size_t kBlock = 1 << 16;
  std::vector<std::unique_ptr<char[]>> blocks_, big_;
  std::size_t next_ = 0, current_ = 0, used_ = kBlock;
};

std::uint32_t narrow_u32(std::uint64_t v) {
  if (v > std::numeric_limits<std::uint32_t>::max()) throw std::invalid_argument("number too large for a 32-bit column in GOAL text");
  return static_cast<std::uint32_t>(v);
}
std::int32_t narrow_i32(std::uint64_t v) {
  if (v > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
    throw std::invalid_argument("number too large for a 32-bit column in GOAL text");
  return static_cast<std::int32_t>(v);
}

// One piece of the file: [begin, end) holds whole lines and, except for the
// first piece, starts at a rank header.
struct Piece {
  std::size_t begin = 0, end = 0;
};

// Parses a piece into rows. The state at its start is the state after a rank
// header (or the start of the file), which is why pieces are independent.
std::unique_ptr<GoalColumns> parse_piece(const MappedFile& file, const Piece& piece) {
  const char* text = file.data();
  if (!simdjson::validate_utf8(text + piece.begin, piece.end - piece.begin))
    throw std::invalid_argument("GOAL text is not valid UTF-8");

  auto out = std::make_unique<GoalColumns>();
  GoalColumns& rows = *out;
  std::unordered_map<std::string_view, std::uint64_t> labels;
  Arena arena;
  std::string scratch;
  bool in_block = false;
  std::uint64_t rank = 0, next_id = 0;

  auto forget_labels = [&] {
    labels.clear();
    arena.clear();
  };

  std::size_t pos = piece.begin;
  while (pos < piece.end) {
    const std::string_view raw = goal::next_line(text, piece.end, pos);
    const std::string_view code = goal::code_of(raw, in_block, scratch);
    if (code.empty() || code == "{" || code.starts_with("num_ranks")) continue;
    if (code == "}") {
      forget_labels();
      continue;
    }
    std::uint64_t header_rank = 0;
    if (goal::parse_rank_header(code, header_rank)) {
      rank = header_rank;
      next_id = 0;
      forget_labels();
      continue;
    }
    std::string_view dst, src;
    bool irequires = false;
    if (goal::parse_dependency(code, dst, src, irequires)) {
      const auto s = labels.find(src);
      const auto d = labels.find(dst);
      if (s != labels.end() && d != labels.end()) {
        rows.edge_rank.push_back(narrow_u32(rank));
        rows.edge_src.push_back(s->second);
        rows.edge_dst.push_back(d->second);
        rows.edge_kind.push_back(irequires ? 1 : 0);
      }
      continue;
    }
    Op op;
    if (!goal::parse_op(code, op)) continue;

    if (!op.label.empty()) labels[arena.copy(op.label)] = next_id;
    const std::string_view name = !op.label.empty() ? op.label
                                  : op.kind == OpKind::kSend ? "comm_send"
                                  : op.kind == OpKind::kRecv ? "comm_recv"
                                                             : "compute";
    rows.rank.push_back(narrow_u32(rank));
    rows.node_id.push_back(next_id++);
    rows.name_bytes.insert(rows.name_bytes.end(), name.begin(), name.end());
    rows.name_end.push_back(rows.name_bytes.size());
    rows.op_class.push_back(static_cast<std::uint8_t>(op.kind));
    const bool is_compute = op.kind == OpKind::kCompute;
    rows.duration_ns.push_back(is_compute ? op.value : 0);
    rows.size_bytes.push_back(is_compute ? 0 : op.value);
    if (is_compute) {
      rows.src_rank.push_back(-1);
      rows.dst_rank.push_back(-1);
    } else {
      const std::int32_t me = narrow_i32(rank);
      const std::int32_t peer = narrow_i32(op.peer);
      rows.src_rank.push_back(op.kind == OpKind::kSend ? me : peer);
      rows.dst_rank.push_back(op.kind == OpKind::kSend ? peer : me);
    }
    rows.message_tag.push_back(narrow_u32(op.tag));
  }
  return out;
}

// Appends pieces' rows to the columns in the order they arrive.
class Merger {
 public:
  void add(std::unique_ptr<GoalColumns> piece) {
    GoalColumns& p = *piece;
    const std::uint64_t base = out_.name_bytes.size();
    for (std::uint64_t end : p.name_end) out_.name_end.push_back(base + end);
    append_vector(out_.name_bytes, p.name_bytes);
    append_vector(out_.rank, p.rank);
    append_vector(out_.node_id, p.node_id);
    append_vector(out_.op_class, p.op_class);
    append_vector(out_.duration_ns, p.duration_ns);
    append_vector(out_.size_bytes, p.size_bytes);
    append_vector(out_.src_rank, p.src_rank);
    append_vector(out_.dst_rank, p.dst_rank);
    append_vector(out_.message_tag, p.message_tag);
    append_vector(out_.edge_rank, p.edge_rank);
    append_vector(out_.edge_src, p.edge_src);
    append_vector(out_.edge_dst, p.edge_dst);
    append_vector(out_.edge_kind, p.edge_kind);
  }
  GoalColumns finish() { return std::move(out_); }

 private:
  GoalColumns out_;
};

}  // namespace

GoalColumns read_goal(const std::string& path, int threads, std::uint64_t chunk_bytes) {
  if (threads < 0) throw std::invalid_argument("threads must be >= 0");
  if (chunk_bytes < 1) throw std::invalid_argument("chunk_bytes must be at least 1");
  if (threads == 0) threads = usable_cpus();

  MappedFile file(path);  // declared before the pool, so it outlives the workers
  if (file.size() == 0) return {};

  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  double finish_seconds = 0;

  Merger merger;
  OrderedPool<Piece, std::unique_ptr<GoalColumns>> pool(
      threads,
      [&]() -> OrderedPool<Piece, std::unique_ptr<GoalColumns>>::Work {
        return [&](Piece& piece) { return parse_piece(file, piece); };
      },
      [&](std::unique_ptr<GoalColumns>&& rows) { merger.add(std::move(rows)); },
      static_cast<std::size_t>(threads) + 2);

  // Find the rank headers and cut before the first one past each chunk size.
  // Only a header reached with no "/*" comment open starts a new piece.
  const char* text = file.data();
  const std::size_t size = file.size();
  std::size_t piece_begin = 0, pos = 0;
  bool in_block = false;
  std::string scratch;
  while (pos < size) {
    const std::size_t line_begin = pos;
    const bool block_before = in_block;
    const std::string_view code = goal::code_of(goal::next_line(text, size, pos), in_block, scratch);
    std::uint64_t rank = 0;
    if (!block_before && line_begin - piece_begin >= chunk_bytes && goal::parse_rank_header(code, rank)) {
      pool.submit(Piece{piece_begin, line_begin});
      piece_begin = line_begin;
    }
  }
  pool.submit(Piece{piece_begin, size});
  const auto scanned = Clock::now();
  pool.finish();
  finish_seconds = duration_seconds(Clock::now() - scanned);

  GoalColumns columns = merger.finish();
  columns.timing.scan = duration_seconds(scanned - start);
  columns.timing.finish = finish_seconds;
  return columns;
}

}  // namespace cta::readers
