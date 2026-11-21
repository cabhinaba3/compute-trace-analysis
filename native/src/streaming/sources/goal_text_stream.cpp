#include "streaming/sources/source.hpp"

#include <fcntl.h>
#include <simdjson.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "readers/goal/syntax.hpp"

namespace cta::streaming {
namespace {

using cta::readers::goal::Op;
using cta::readers::goal::OpKind;

std::uint32_t narrow_u32(std::uint64_t v) {
  if (v > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("number too large for a 32-bit column in GOAL text");
  return static_cast<std::uint32_t>(v);
}

std::int32_t narrow_i32(std::uint64_t v) {
  if (v > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
    throw std::invalid_argument("number too large for a 32-bit column in GOAL text");
  return static_cast<std::int32_t>(v);
}

// Per-rank label arena, cleared when a rank block closes.
class LabelArena {
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
    used_ = kBlock;
    big_.clear();
  }

 private:
  static constexpr std::size_t kBlock = 1 << 16;
  std::vector<std::unique_ptr<char[]>> blocks_, big_;
  std::size_t next_ = 0, current_ = 0, used_ = kBlock;
};

struct PendingDep {
  std::string_view dst;
  std::string_view src;
  bool irequires = false;
};

// Bounded-RAM streaming reader for GOAL text (.goal), supporting both seekable
// files and unseekable pipes / stdin ("-").
class GoalTextSource final : public GraphSource {
 public:
  GoalTextSource(std::string path, GoalSemantics semantics)
      : path_(std::move(path)),
        semantics_(semantics),
        seekable_(is_seekable_path(path_)),
        est_bytes_(regular_file_size(path_)) {
    open_fd();
  }

  ~GoalTextSource() override { close_fd(); }

  bool next_chunk(GraphChunk& out, std::size_t max_nodes) override {
    out.clear();
    if (max_nodes == 0) max_nodes = 1;
    if (done_) return false;

    while (out.nodes.size() < max_nodes) {
      if (buf_pos_ < buf_valid_) {
        process_line(out);
        continue;
      }
      if (!refill()) {
        // Flush any pending strict-mode forward dependencies at EOF.
        flush_pending_deps(out);
        done_ = true;
        break;
      }
    }
    return !out.empty();
  }

  bool is_seekable() const override { return seekable_; }

  void reset() override {
    if (!seekable_) throw std::runtime_error("cannot reset unseekable GOAL text stream");
    close_fd();
    buf_.clear();
    carry_.clear();
    buf_pos_ = 0;
    buf_valid_ = 0;
    eof_read_ = false;
    done_ = false;
    in_block_ = false;
    rank_ = 0;
    next_id_ = 0;
    labels_.clear();
    pending_deps_.clear();
    arena_.clear();
    open_fd();
  }

  std::string format_name() const override { return "goal_text"; }
  std::uint64_t estimated_bytes() const override { return est_bytes_; }

 private:
  static constexpr std::size_t kReadBlock = 4u << 20;

  void open_fd() {
    if (path_ == "-") {
      fd_ = ::dup(STDIN_FILENO);
      if (fd_ < 0) throw std::runtime_error("cannot dup stdin for GOAL text stream");
    } else {
      fd_ = ::open(path_.c_str(), O_RDONLY);
      if (fd_ < 0) throw std::runtime_error("cannot open " + path_);
    }
  }

  void close_fd() {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  // Reads a block of whole lines (never cutting mid-line or mid-UTF-8 char)
  // and validates UTF-8 on the complete-lines slice.
  bool refill() {
    if (eof_read_) return false;
    buf_.clear();
    if (!carry_.empty()) {
      buf_.swap(carry_);
    }
    const std::size_t old_size = buf_.size();
    buf_.resize(old_size + kReadBlock);
    std::size_t total_read = 0;
    while (total_read == 0) {
      const ssize_t r = ::read(fd_, buf_.data() + old_size, kReadBlock);
      if (r < 0) throw std::runtime_error("read failed: " + path_);
      if (r == 0) {
        eof_read_ = true;
        buf_.resize(old_size);
        break;
      }
      total_read = static_cast<std::size_t>(r);
      buf_.resize(old_size + total_read);
      // If no line terminator exists yet and we haven't hit EOF, keep reading
      // until we find a line break or EOF so a single very long line stays intact.
      if (!std::memchr(buf_.data(), '\n', buf_.size()) && !std::memchr(buf_.data(), '\r', buf_.size())) {
        const std::size_t cur = buf_.size();
        buf_.resize(cur + kReadBlock);
        const ssize_t more = ::read(fd_, buf_.data() + cur, kReadBlock);
        if (more < 0) throw std::runtime_error("read failed: " + path_);
        if (more == 0) {
          eof_read_ = true;
          buf_.resize(cur);
          break;
        }
        buf_.resize(cur + static_cast<std::size_t>(more));
      }
    }
    if (buf_.empty()) return false;

    std::size_t cut = buf_.size();
    if (!eof_read_) {
      // Find the last '\n' or '\r' strictly before the final byte so "\r\n"
      // split across two reads stays together.
      cut = 0;
      for (std::size_t i = buf_.size(); i-- > 0;) {
        if (buf_[i] == '\n') {
          cut = i + 1;
          break;
        }
        if (buf_[i] == '\r' && i + 1 < buf_.size()) {
          cut = i + 1;
          break;
        }
      }
      if (cut == 0) {
        // One partial line longer than kReadBlock; accumulate into carry and keep reading.
        carry_.swap(buf_);
        return refill();
      }
      carry_.assign(buf_.data() + cut, buf_.size() - cut);
    }

    if (!simdjson::validate_utf8(buf_.data(), cut)) {
      throw std::invalid_argument("GOAL text is not valid UTF-8");
    }
    buf_pos_ = 0;
    buf_valid_ = cut;
    return true;
  }

  void flush_pending_deps(GraphChunk& out) {
    if (pending_deps_.empty()) return;
    const std::uint32_t r = narrow_u32(rank_);
    for (const PendingDep& pd : pending_deps_) {
      const auto s = labels_.find(pd.src);
      const auto d = labels_.find(pd.dst);
      if (s != labels_.end() && d != labels_.end()) {
        out.edges.push(r, s->second, d->second, pd.irequires ? EdgeKind::kIrequires : EdgeKind::kRequires);
      }
    }
    pending_deps_.clear();
  }

  void forget_labels(GraphChunk& out) {
    flush_pending_deps(out);
    labels_.clear();
    arena_.clear();
  }

  void process_line(GraphChunk& out) {
    const std::string_view raw = cta::readers::goal::next_line(buf_.data(), buf_valid_, buf_pos_);
    const std::string_view code = cta::readers::goal::code_of(raw, in_block_, scratch_);
    if (code.empty() || code == "{" || code.starts_with("num_ranks")) return;
    if (code == "}") {
      forget_labels(out);
      return;
    }
    std::uint64_t header_rank = 0;
    if (cta::readers::goal::parse_rank_header(code, header_rank)) {
      forget_labels(out);
      rank_ = header_rank;
      next_id_ = 0;
      return;
    }
    std::string_view dst, src;
    bool irequires = false;
    if (cta::readers::goal::parse_dependency(code, dst, src, irequires)) {
      const auto s = labels_.find(src);
      const auto d = labels_.find(dst);
      if (s != labels_.end() && d != labels_.end()) {
        out.edges.push(narrow_u32(rank_), s->second, d->second,
                       irequires ? EdgeKind::kIrequires : EdgeKind::kRequires);
      } else if (semantics_ == GoalSemantics::kStrict) {
        pending_deps_.push_back({arena_.copy(dst), arena_.copy(src), irequires});
      }
      return;
    }
    Op op;
    if (!cta::readers::goal::parse_op(code, op)) return;

    if (!op.label.empty()) labels_[arena_.copy(op.label)] = next_id_;
    const bool is_compute = op.kind == OpKind::kCompute;
    const bool is_send = op.kind == OpKind::kSend;
    const std::string_view name = !op.label.empty() ? op.label
                                  : is_send          ? "comm_send"
                                  : op.kind == OpKind::kRecv ? "comm_recv"
                                                             : "compute";
    const OpClass oc = is_compute ? OpClass::kCompute : is_send ? OpClass::kCommSend : OpClass::kCommRecv;
    const std::string_view comm_type = is_compute ? "" : is_send ? "SEND" : "RECV";
    const std::uint32_t r_u32 = narrow_u32(rank_);
    const std::uint64_t id = next_id_++;
    const std::uint64_t dur_ns = is_compute ? op.value : 0;
    const std::uint64_t bytes = is_compute ? 0 : op.value;
    std::int32_t sr = -1, dr = -1;
    if (!is_compute) {
      const std::int32_t me = narrow_i32(rank_);
      const std::int32_t peer = narrow_i32(op.peer);
      sr = is_send ? me : peer;
      dr = is_send ? peer : me;
    }
    out.nodes.push(r_u32, id, name, oc, comm_type, dur_ns, bytes, 0, sr, dr, narrow_u32(op.tag));
  }

  std::string path_;
  GoalSemantics semantics_;
  bool seekable_ = false;
  std::uint64_t est_bytes_ = 0;
  int fd_ = -1;
  std::string buf_, carry_, scratch_;
  std::size_t buf_pos_ = 0, buf_valid_ = 0;
  bool eof_read_ = false, done_ = false, in_block_ = false;
  std::uint64_t rank_ = 0, next_id_ = 0;
  std::unordered_map<std::string_view, std::uint64_t> labels_;
  std::vector<PendingDep> pending_deps_;
  LabelArena arena_;
};

}  // namespace

std::unique_ptr<GraphSource> open_goal_text_stream(const std::string& path, GoalSemantics semantics) {
  return std::make_unique<GoalTextSource>(path, semantics);
}

}  // namespace cta::streaming
