#include "streaming/sources/source.hpp"

#include <fcntl.h>
#include <simdjson.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "util/util.hpp"

namespace cta::streaming {
namespace {

bool contains_ci(std::string_view hay, std::string_view needle) {
  if (needle.size() > hay.size()) return false;
  for (std::size_t i = 0; i + needle.size() <= hay.size(); ++i) {
    bool ok = true;
    for (std::size_t j = 0; j < needle.size(); ++j) {
      char c = hay[i + j];
      if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
      if (c != needle[j]) {
        ok = false;
        break;
      }
    }
    if (ok) return true;
  }
  return false;
}

// Mirrors _HOST_OP_PATTERNS and _classify_host_op in src/cta/adapters/chakra/reader.py.
std::pair<OpClass, std::string_view> classify_host_op(std::string_view name) {
  if (contains_ci(name, "allreduce") || contains_ci(name, "all_reduce")) return {OpClass::kCommColl, "ALL_REDUCE"};
  if (contains_ci(name, "allgather") || contains_ci(name, "all_gather")) return {OpClass::kCommColl, "ALL_GATHER"};
  if (contains_ci(name, "reducescatter") || contains_ci(name, "reduce_scatter"))
    return {OpClass::kCommColl, "REDUCE_SCATTER"};
  if (contains_ci(name, "alltoall") || contains_ci(name, "all_to_all")) return {OpClass::kCommColl, "ALL_TO_ALL"};
  if (contains_ci(name, "send")) return {OpClass::kCommSend, "SEND"};
  if (contains_ci(name, "recv")) return {OpClass::kCommRecv, "RECV"};
  return {OpClass::kCompute, ""};
}

// Streaming splitter for Chakra host execution trace JSON: cuts the top-level
// "nodes" array (or a bare top-level array) into bounded JSON array chunks.
class HostNodesSplitter {
 public:
  HostNodesSplitter(std::size_t chunk_bytes, std::function<void(std::string&&)> emit)
      : chunk_bytes_(chunk_bytes), emit_(std::move(emit)) {}

  void feed(const char* base, std::size_t n) {
    const char* p = base;
    const char* const end = base + n;
    const char* seg = in_nodes_ ? base : nullptr;
    auto flush = [&](const char* upto) {
      if (seg && upto > seg) buf_.append(seg, static_cast<std::size_t>(upto - seg));
      seg = nullptr;
    };

    while (p < end) {
      char c = *p;
      if (in_string_) {
        if (escape_) {
          escape_ = false;
          if (capture_key_ && key_.size() < 256) key_.push_back(c);
          ++p;
          continue;
        }
        if (!capture_key_) {
          while (p < end && *p != '"' && *p != '\\') ++p;
          if (p == end) break;
          c = *p;
        }
        if (c == '\\') {
          escape_ = true;
          if (capture_key_ && key_.size() < 256) key_.push_back(c);
        } else if (c == '"') {
          in_string_ = false;
          capture_key_ = false;
        } else if (capture_key_ && key_.size() < 256) {
          key_.push_back(c);
        }
        ++p;
        continue;
      }
      if (in_fast_) {
        p = skip_node(p, end);
        continue;
      }

      switch (c) {
        case '"':
          in_string_ = true;
          if (stack_.size() == 1 && stack_[0] == '{' && expect_key_) {
            capture_key_ = true;
            key_.clear();
          }
          ++p;
          break;
        case '{':
        case '[': {
          if (in_nodes_ && stack_.size() == nodes_depth_) {
            in_fast_ = true;
            nested_ = 1;
            ++p;
            break;
          }
          if (stack_.empty()) {
            if (done_) throw std::invalid_argument("extra data after Chakra host JSON");
            if (!started_) started_ = true;
          }
          stack_.push_back(c);
          ++p;
          if (!nodes_seen_ && c == '[' &&
              ((stack_.size() == 1) || (stack_.size() == 2 && stack_[0] == '{' && key_ == "nodes"))) {
            in_nodes_ = true;
            nodes_seen_ = true;
            nodes_depth_ = stack_.size();
            buf_.clear();
            buf_.reserve(std::min<std::size_t>(chunk_bytes_ + chunk_bytes_ / 8, std::size_t{1} << 26) +
                         simdjson::SIMDJSON_PADDING);
            buf_.push_back('[');
            seg = p;
          } else if (stack_.size() == 1 && c == '{') {
            expect_key_ = true;
          }
          break;
        }
        case '}':
        case ']': {
          if (stack_.empty()) throw std::invalid_argument("unbalanced brackets in Chakra host JSON");
          if (in_nodes_ && stack_.size() == nodes_depth_) {
            flush(p);
            in_nodes_ = false;
            buf_.push_back(']');
            emit_(std::move(buf_));
            buf_.clear();
          }
          stack_.pop_back();
          ++p;
          if (stack_.empty()) done_ = true;
          break;
        }
        case ',':
          if (stack_.size() == 1 && stack_[0] == '{') {
            expect_key_ = true;
          } else if (in_nodes_ && stack_.size() == nodes_depth_) {
            const std::size_t pending = buf_.size() + (seg ? static_cast<std::size_t>(p - seg) : 0);
            if (pending >= chunk_bytes_) {
              flush(p);
              buf_.push_back(']');
              emit_(std::move(buf_));
              buf_.clear();
              buf_.push_back('[');
              ++p;
              seg = p;
              break;
            }
          }
          ++p;
          break;
        case ':':
          if (stack_.size() == 1 && stack_[0] == '{') expect_key_ = false;
          ++p;
          break;
        case ' ':
        case '\t':
        case '\n':
        case '\r':
          ++p;
          break;
        default:
          if (stack_.empty() && !started_)
            throw std::invalid_argument("Chakra host JSON must be an object or an array");
          ++p;
          break;
      }
    }
    if (seg) buf_.append(seg, static_cast<std::size_t>(end - seg));
  }

  void finish() {
    if (in_string_ || in_fast_ || !stack_.empty() || !started_)
      throw std::invalid_argument("Chakra host JSON ends early");
  }

 private:
  const char* skip_node(const char* p, const char* end) {
    int depth = nested_;
    while (p < end) {
      const char c = *p;
      if (c == '"') {
        ++p;
        const char* const content = p;
        while (true) {
          const void* hit = std::memchr(p, '"', static_cast<std::size_t>(end - p));
          if (!hit) {
            std::size_t slashes = 0;
            for (const char* r = end; r > content && r[-1] == '\\'; --r) ++slashes;
            in_string_ = true;
            escape_ = (slashes % 2) == 1;
            nested_ = depth;
            return end;
          }
          const char* q = static_cast<const char*>(hit);
          std::size_t slashes = 0;
          for (const char* r = q; r > content && r[-1] == '\\'; --r) ++slashes;
          p = q + 1;
          if (slashes % 2 == 0) break;
        }
        continue;
      }
      ++p;
      if (c == '{' || c == '[') {
        ++depth;
      } else if ((c == '}' || c == ']') && --depth == 0) {
        in_fast_ = false;
        nested_ = 0;
        return p;
      }
    }
    nested_ = depth;
    return p;
  }

  const std::size_t chunk_bytes_;
  std::function<void(std::string&&)> emit_;
  std::string buf_, key_;
  std::vector<char> stack_;
  bool in_string_ = false, escape_ = false;
  bool started_ = false, done_ = false;
  bool expect_key_ = false, capture_key_ = false;
  bool in_nodes_ = false, nodes_seen_ = false;
  bool in_fast_ = false;
  int nested_ = 0;
  std::size_t nodes_depth_ = 0;
};

double json_as_double(simdjson::ondemand::value& v, double fallback) {
  auto t = v.type();
  if (t.error()) return fallback;
  switch (t.value()) {
    case simdjson::ondemand::json_type::number: {
      double d = 0.0;
      if (!v.get_double().get(d)) return d;
      return fallback;
    }
    case simdjson::ondemand::json_type::string: {
      std::string_view sv;
      if (!v.get_string().get(sv) && !sv.empty()) {
        std::string s(sv);
        char* end = nullptr;
        const double d = std::strtod(s.c_str(), &end);
        if (end && *end == '\0') return d;
      }
      return fallback;
    }
    case simdjson::ondemand::json_type::boolean: {
      bool b = false;
      if (!v.get_bool().get(b)) return b ? 1.0 : 0.0;
      return fallback;
    }
    default:
      return fallback;
  }
}

std::int64_t json_as_i64(simdjson::ondemand::value& v, std::int64_t fallback) {
  auto t = v.type();
  if (t.error()) return fallback;
  switch (t.value()) {
    case simdjson::ondemand::json_type::number: {
      std::int64_t i = 0;
      if (!v.get_int64().get(i)) return i;
      double d = 0.0;
      if (!v.get_double().get(d)) return checked_i64(d, fallback);
      return fallback;
    }
    case simdjson::ondemand::json_type::string: {
      std::string_view sv;
      if (!v.get_string().get(sv) && !sv.empty()) {
        std::string s(sv);
        char* end = nullptr;
        const long long i = std::strtoll(s.c_str(), &end, 10);
        if (end && *end == '\0') return static_cast<std::int64_t>(i);
      }
      return fallback;
    }
    case simdjson::ondemand::json_type::boolean: {
      bool b = false;
      if (!v.get_bool().get(b)) return b ? 1 : 0;
      return fallback;
    }
    default:
      return fallback;
  }
}

void parse_host_json_chunk(const std::string& json_chunk, std::uint32_t rank, simdjson::ondemand::parser& parser,
                           GraphChunk& out) {
  simdjson::padded_string padded(json_chunk.data(), json_chunk.size());
  simdjson::ondemand::document doc;
  if (parser.iterate(padded).get(doc)) throw std::invalid_argument("invalid JSON chunk in Chakra host trace");
  simdjson::ondemand::array arr;
  if (doc.get_array().get(arr)) throw std::invalid_argument("expected JSON array chunk in Chakra host trace");

  std::vector<std::uint64_t> ctrl_deps, data_deps;
  for (auto elem : arr) {
    simdjson::ondemand::object obj;
    if (elem.get_object().get(obj)) continue;

    bool has_id = false;
    std::uint64_t node_id = 0;
    std::string name;
    bool has_dur_us = false, has_duration_micros = false;
    double dur_us = 1.0, duration_micros = 1.0;
    bool has_comm_bytes = false, has_tensor_bytes = false;
    std::int64_t comm_bytes = 0, tensor_bytes = 0;
    bool has_comm_src = false, has_src_rank = false;
    std::int64_t comm_src = 0, src_rank_val = 0;
    bool has_comm_dst = false, has_dst_rank = false;
    std::int64_t comm_dst = 0, dst_rank_val = 0;
    bool has_comm_tag = false, has_tag = false;
    std::int64_t comm_tag = 0, tag_val = 0;
    std::string comm_group;
    ctrl_deps.clear();
    data_deps.clear();

    for (auto field : obj) {
      std::string_view key;
      if (field.unescaped_key().get(key)) continue;
      simdjson::ondemand::value val = field.value();

      if (key == "id") {
        const std::int64_t v = json_as_i64(val, 0);
        node_id = v >= 0 ? static_cast<std::uint64_t>(v) : 0;
        has_id = true;
      } else if (key == "name") {
        std::string_view sv;
        if (!val.get_string().get(sv)) name.assign(sv.data(), sv.size());
      } else if (key == "dur_us") {
        has_dur_us = true;
        dur_us = json_as_double(val, 1.0);
      } else if (key == "duration_micros") {
        has_duration_micros = true;
        duration_micros = json_as_double(val, 1.0);
      } else if (key == "comm_bytes") {
        has_comm_bytes = true;
        comm_bytes = json_as_i64(val, 0);
      } else if (key == "tensor_bytes") {
        has_tensor_bytes = true;
        tensor_bytes = json_as_i64(val, 0);
      } else if (key == "comm_src") {
        has_comm_src = true;
        comm_src = json_as_i64(val, -1);
      } else if (key == "src_rank") {
        has_src_rank = true;
        src_rank_val = json_as_i64(val, -1);
      } else if (key == "comm_dst") {
        has_comm_dst = true;
        comm_dst = json_as_i64(val, -1);
      } else if (key == "dst_rank") {
        has_dst_rank = true;
        dst_rank_val = json_as_i64(val, -1);
      } else if (key == "comm_tag") {
        has_comm_tag = true;
        comm_tag = json_as_i64(val, 0);
      } else if (key == "tag") {
        has_tag = true;
        tag_val = json_as_i64(val, 0);
      } else if (key == "pg_name" || key == "comm_group") {
        std::string_view sv;
        if (!val.get_string().get(sv)) comm_group.assign(sv.data(), sv.size());
      } else if (key == "ctrl_deps") {
        simdjson::ondemand::array deps;
        if (!val.get_array().get(deps)) {
          for (auto d : deps) {
            simdjson::ondemand::value dv;
            if (!d.get(dv)) {
              const std::int64_t dep_id = json_as_i64(dv, -1);
              if (dep_id >= 0) ctrl_deps.push_back(static_cast<std::uint64_t>(dep_id));
            }
          }
        }
      } else if (key == "data_deps") {
        simdjson::ondemand::array deps;
        if (!val.get_array().get(deps)) {
          for (auto d : deps) {
            simdjson::ondemand::value dv;
            if (!d.get(dv)) {
              const std::int64_t dep_id = json_as_i64(dv, -1);
              if (dep_id >= 0) data_deps.push_back(static_cast<std::uint64_t>(dep_id));
            }
          }
        }
      }
    }

    if (!has_id) continue;
    const auto [oc, comm_type] = classify_host_op(name);
    const double chosen_us = has_dur_us ? dur_us : (has_duration_micros ? duration_micros : 1.0);
    const std::int64_t dur_ns_signed = checked_i64(chosen_us * 1000.0, 0);
    const std::uint64_t dur_ns = dur_ns_signed > 0 ? static_cast<std::uint64_t>(dur_ns_signed) : 0;
    const std::int64_t chosen_bytes = has_comm_bytes ? comm_bytes : (has_tensor_bytes ? tensor_bytes : 0);
    const std::uint64_t size_bytes = chosen_bytes > 0 ? static_cast<std::uint64_t>(chosen_bytes) : 0;

    const std::int64_t default_src = oc == OpClass::kCommSend ? static_cast<std::int64_t>(rank) : -1;
    const std::int64_t default_dst = oc == OpClass::kCommRecv ? static_cast<std::int64_t>(rank) : -1;
    const std::int64_t sr = has_comm_src ? comm_src : (has_src_rank ? src_rank_val : default_src);
    const std::int64_t dr = has_comm_dst ? comm_dst : (has_dst_rank ? dst_rank_val : default_dst);
    const std::int64_t tag_signed = has_comm_tag ? comm_tag : (has_tag ? tag_val : 0);
    const std::uint32_t tag = tag_signed > 0 ? static_cast<std::uint32_t>(tag_signed) : 0;

    out.nodes.push(rank, node_id, name, oc, comm_type, dur_ns, size_bytes, 0, static_cast<std::int32_t>(sr),
                   static_cast<std::int32_t>(dr), tag, comm_group);

    // Match ChakraHostAdapter.ingest order: ctrl_deps before data_deps.
    for (std::uint64_t dep : ctrl_deps) out.edges.push(rank, dep, node_id, EdgeKind::kCtrl);
    for (std::uint64_t dep : data_deps) out.edges.push(rank, dep, node_id, EdgeKind::kData);
  }
}

class ChakraHostSource final : public GraphSource {
 public:
  ChakraHostSource(std::vector<std::string> paths, std::vector<std::uint32_t> ranks)
      : paths_(std::move(paths)), ranks_(std::move(ranks)) {
    if (paths_.size() != ranks_.size()) throw std::invalid_argument("paths and ranks must have the same length");
    seekable_ = true;
    for (const auto& p : paths_) {
      if (!is_seekable_path(p)) seekable_ = false;
      est_bytes_ += regular_file_size(p);
    }
    read_buf_.resize(kReadBytes);
  }

  ~ChakraHostSource() override { close_current(); }

  bool next_chunk(GraphChunk& out, std::size_t max_nodes) override {
    out.clear();
    if (max_nodes == 0) max_nodes = 1;

    while (out.nodes.size() < max_nodes) {
      if (!pending_json_.empty()) {
        auto [r, chunk_str] = std::move(pending_json_.front());
        pending_json_.pop_front();
        parse_host_json_chunk(chunk_str, r, parser_, out);
        continue;
      }
      if (fd_ < 0) {
        if (!open_next_file(max_nodes)) break;
        continue;
      }
      const ssize_t got = ::read(fd_, read_buf_.data(), read_buf_.size());
      if (got < 0) throw std::runtime_error("read failed: " + current_path_);
      if (got == 0) {
        splitter_->finish();
        close_current();
        continue;
      }
      splitter_->feed(read_buf_.data(), static_cast<std::size_t>(got));
    }
    return !out.empty();
  }

  bool is_seekable() const override { return seekable_; }

  void reset() override {
    if (!seekable_) throw std::runtime_error("cannot reset unseekable Chakra host JSON stream");
    close_current();
    pending_json_.clear();
    file_idx_ = 0;
  }

  std::string format_name() const override { return "chakra_host"; }
  std::uint64_t estimated_bytes() const override { return est_bytes_; }

 private:
  static constexpr std::size_t kReadBytes = 4u << 20;

  void close_current() {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    splitter_.reset();
  }

  bool open_next_file(std::size_t max_nodes) {
    if (file_idx_ >= paths_.size()) return false;
    const std::size_t i = file_idx_++;
    current_path_ = paths_[i];
    current_rank_ = ranks_[i];
    if (current_path_ == "-") {
      fd_ = ::dup(STDIN_FILENO);
      if (fd_ < 0) throw std::runtime_error("cannot dup stdin for Chakra host JSON stream");
    } else {
      fd_ = ::open(current_path_.c_str(), O_RDONLY);
      if (fd_ < 0) throw std::runtime_error("cannot open " + current_path_);
    }
    // ~160 bytes per JSON node estimate keeps emitted JSON array chunks proportional to max_nodes.
    const std::size_t chunk_bytes = std::clamp<std::size_t>(max_nodes * 160, 64u << 10, 16u << 20);
    const std::uint32_t r = current_rank_;
    splitter_ = std::make_unique<HostNodesSplitter>(
        chunk_bytes, [this, r](std::string&& chunk) { pending_json_.emplace_back(r, std::move(chunk)); });
    return true;
  }

  std::vector<std::string> paths_;
  std::vector<std::uint32_t> ranks_;
  bool seekable_ = true;
  std::uint64_t est_bytes_ = 0;
  std::size_t file_idx_ = 0;
  int fd_ = -1;
  std::string current_path_;
  std::uint32_t current_rank_ = 0;
  std::vector<char> read_buf_;
  std::unique_ptr<HostNodesSplitter> splitter_;
  std::deque<std::pair<std::uint32_t, std::string>> pending_json_;
  simdjson::ondemand::parser parser_;
};

}  // namespace

std::unique_ptr<GraphSource> open_chakra_host_stream(const std::vector<std::string>& paths,
                                                     const std::vector<std::uint32_t>& ranks) {
  return std::make_unique<ChakraHostSource>(paths, ranks);
}

}  // namespace cta::streaming
