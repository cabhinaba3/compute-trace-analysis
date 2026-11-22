#include "readers/jax/json_reader.hpp"

#include <simdjson.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "readers/jax/common.hpp"
#include "util/interner.hpp"
#include "util/ordered_pool.hpp"
#include "util/util.hpp"

namespace cta::readers::jax {
namespace {

namespace od = simdjson::ondemand;

constexpr std::size_t kChunkPadding = 64;
static_assert(kChunkPadding >= simdjson::SIMDJSON_PADDING,
              "chunk padding is below what simdjson reads past the end");

// ---------------------------------------------------------------------------
// Streaming JSON splitter: cuts the top-level "traceEvents" array (or a bare
// top-level array) into chunks at commas between events, independent of block
// boundaries.

class JsonSplitter {
 public:
  JsonSplitter(std::size_t chunk_bytes, std::function<void(std::string&&)> emit)
      : chunk_bytes_(chunk_bytes), emit_(std::move(emit)) {}

  void feed(const char* base, std::size_t n) {
    const char* p = base;
    const char* const end = base + n;
    const char* seg = in_events_ ? base : nullptr;
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
        p = skip_event(p, end);
        continue;
      }

      if (c == '"' || c == '{' || c == '[' || c == '}' || c == ']' || c == ',' || c == ':' || c == ' ' ||
          c == '\t' || c == '\n' || c == '\r') {
        in_scalar_ = false;
      }
      switch (c) {
        case '"':
          if (stack_.empty()) check_start(c);
          if (in_events_ && stack_.size() == events_depth_) event_value();
          in_string_ = true;
          if (stack_.size() == 1 && stack_[0] == '{' && expect_key_) {
            capture_key_ = true;
            key_.clear();
          }
          ++p;
          break;
        case '{':
        case '[':
          if (in_events_ && stack_.size() == events_depth_) {
            event_value();
            in_fast_ = true;
            nested_ = 1;
            ++p;
            break;
          }
          check_start(c);
          if (events_seen_ && c == '[' && stack_.size() == 1 && stack_[0] == '{' && key_ == "traceEvents") {
            throw std::invalid_argument("trace JSON has more than one traceEvents array");
          }
          stack_.push_back(c);
          ++p;
          if (!events_seen_ && c == '[' &&
              ((stack_.size() == 1) || (stack_.size() == 2 && stack_[0] == '{' && key_ == "traceEvents"))) {
            in_events_ = true;
            events_seen_ = true;
            events_depth_ = stack_.size();
            slot_ = Slot::kStart;
            buf_.clear();
            reserve_chunk();
            buf_.push_back('[');
            seg = p;
          } else if (stack_.size() == 1 && c == '{') {
            expect_key_ = true;
          }
          break;
        case '}':
        case ']':
          if (stack_.empty()) throw std::invalid_argument("unbalanced brackets in trace JSON");
          if (in_events_ && stack_.size() == events_depth_) {
            if (slot_ == Slot::kAfterComma) throw std::invalid_argument("trailing comma in the trace events");
            flush(p);
            in_events_ = false;
            buf_.push_back(']');
            emit_(std::move(buf_));
            buf_.clear();
          }
          stack_.pop_back();
          ++p;
          if (stack_.empty()) done_ = true;
          break;
        case ',':
          if (stack_.empty()) check_start(c);
          if (stack_.size() == 1 && stack_[0] == '{') {
            expect_key_ = true;
          } else if (in_events_ && stack_.size() == events_depth_) {
            if (slot_ != Slot::kAfterValue) throw std::invalid_argument("unexpected comma in the trace events");
            slot_ = Slot::kAfterComma;
            const std::size_t pending = buf_.size() + (seg ? static_cast<std::size_t>(p - seg) : 0);
            if (pending >= chunk_bytes_) {
              flush(p);
              buf_.push_back(']');
              emit_(std::move(buf_));
              buf_.clear();
              reserve_chunk();
              buf_.push_back('[');
              ++p;
              seg = p;
              break;
            }
          }
          ++p;
          break;
        case ':':
          if (stack_.empty()) check_start(c);
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
          check_start(c);
          if (in_events_ && stack_.size() == events_depth_ && !in_scalar_) {
            event_value();
            in_scalar_ = true;
          }
          ++p;
          break;
      }
    }
    if (seg) buf_.append(seg, static_cast<std::size_t>(end - seg));
  }

  void finish() {
    if (in_string_ || in_fast_ || !stack_.empty() || !started_) {
      throw std::invalid_argument("trace JSON ends early");
    }
  }

 private:
  void reserve_chunk() {
    buf_.reserve(std::min<std::size_t>(chunk_bytes_ + chunk_bytes_ / 8, std::size_t{1} << 27) + kChunkPadding);
  }

  void check_start(char c) {
    if (stack_.empty()) {
      if (done_) throw std::invalid_argument("extra data after the trace JSON");
      if (!started_) {
        if (c != '{' && c != '[') throw std::invalid_argument("trace JSON must be an object or an array");
        started_ = true;
      }
    }
  }

  void event_value() {
    if (slot_ == Slot::kAfterValue) throw std::invalid_argument("missing comma between trace events");
    slot_ = Slot::kAfterValue;
  }

  const char* skip_event(const char* p, const char* end) {
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
  std::string buf_;
  std::vector<char> stack_;
  bool in_string_ = false, escape_ = false;
  bool started_ = false, done_ = false;
  bool expect_key_ = false, capture_key_ = false;
  bool in_events_ = false, events_seen_ = false;
  bool in_fast_ = false;
  int nested_ = 0;
  std::size_t events_depth_ = 0;
  enum class Slot : std::uint8_t { kStart, kAfterValue, kAfterComma };
  Slot slot_ = Slot::kStart;
  bool in_scalar_ = false;
  std::string key_;
};

// ---------------------------------------------------------------------------
// Compact chunk representation: interns distinct pids and (pid, tid) pairs
// locally so merge_chunks resolves metadata per distinct thread (O(threads))
// instead of per event (O(events)).

struct Tag {
  std::int8_t kind = 0;  // 0 = absent/null, 1 = int64, 2 = string (index into tags)
  int64_t v = 0;
  bool operator==(const Tag& o) const { return kind == o.kind && v == o.v; }
};

struct TagHash {
  std::size_t operator()(const Tag& t) const {
    return static_cast<std::size_t>(mix64(static_cast<uint64_t>(t.v)) ^
                                    (static_cast<uint64_t>(t.kind) * 0x9e3779b97f4a7c15ULL));
  }
};

struct ThreadKey {
  Tag pid, tid;
  bool operator==(const ThreadKey& o) const { return pid == o.pid && tid == o.tid; }
};

struct ThreadKeyHash {
  std::size_t operator()(const ThreadKey& k) const {
    uint64_t h = static_cast<uint64_t>(k.pid.v) * 0x9e3779b97f4a7c15ULL;
    h ^= static_cast<uint64_t>(k.tid.v) + 0x7f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= static_cast<uint64_t>(k.pid.kind) * 31ULL + static_cast<uint64_t>(k.tid.kind);
    return static_cast<std::size_t>(h);
  }
};

struct MetaEvent {
  bool is_thread = false;
  Tag pid;
  Tag tid;
  int32_t name_code = -1;
};

struct ExtraArgs {
  bool has_step_num = false;
  bool has_hlo_op = false;
  int64_t step_num = 0;
  int64_t correlation_id = -1;
  int64_t device_ordinal = 0;
  int32_t arg_name = -1;
  int32_t tf_op = -1;
  int32_t hlo_op = -1;
  int32_t hlo_module = -1;
  int32_t module = -1;
  int32_t memcpy_details = -1;
  int32_t memset_details = -1;
};

struct RawEvent {
  int32_t name_code = -1;
  int32_t extra_idx = -1;
  int32_t thread_idx = 0;
  int64_t start_ns = 0;
  int64_t end_ns = 0;
};

struct JsonChunk {
  Interner strings;
  Interner tags;
  std::vector<ThreadKey> threads;
  std::unordered_map<ThreadKey, int32_t, ThreadKeyHash> thread_index;
  std::vector<MetaEvent> metas;
  std::vector<RawEvent> events;
  std::vector<ExtraArgs> extras;

  int32_t thread_code(const Tag& pid, const Tag& tid) {
    const ThreadKey key{pid, tid};
    auto [it, inserted] = thread_index.emplace(key, static_cast<int32_t>(threads.size()));
    if (inserted) threads.push_back(key);
    return it->second;
  }
};

std::string encode_tag(const Tag& tag, const Interner& tags) {
  std::string s(1, static_cast<char>(tag.kind));
  if (tag.kind == 1) {
    s.append(reinterpret_cast<const char*>(&tag.v), sizeof tag.v);
  } else if (tag.kind == 2) {
    const std::string& text = tags.items()[static_cast<std::size_t>(tag.v)];
    const uint64_t len = text.size();
    s.append(reinterpret_cast<const char*>(&len), sizeof len);
    s.append(text);
  }
  return s;
}

int64_t tag_as_int(const Tag& tag, const Interner& tags) {
  if (tag.kind == 1) return tag.v;
  if (tag.kind == 2) {
    int64_t v = 0;
    if (parse_signed_digits(tags.items()[static_cast<std::size_t>(tag.v)], v)) return v;
  }
  return 0;
}

double number_as_double(od::number n) {
  if (n.is_int64()) return static_cast<double>(n.get_int64());
  if (n.is_uint64()) return static_cast<double>(n.get_uint64());
  return n.get_double();
}

double checked_time(double us) {
  if (!(std::fabs(us) < 4.0e15)) throw std::invalid_argument("ts and dur must be finite and below 4e15 us");
  return us;
}

double read_time(od::value val) {
  switch (val.type().value()) {
    case od::json_type::number:
      return checked_time(number_as_double(val.get_number().value()));
    case od::json_type::string: {
      std::string s(val.get_string().value());
      char* end = nullptr;
      const double d = std::strtod(s.c_str(), &end);
      if (!s.empty() && end && *end == '\0') return checked_time(d);
      break;
    }
    default:
      break;
  }
  throw std::invalid_argument("ts and dur must be numbers");
}

Tag read_tag(od::value val, Interner& tags) {
  Tag tag;
  switch (val.type().value()) {
    case od::json_type::number: {
      const od::number n = val.get_number().value();
      if (n.is_int64()) {
        tag = {1, n.get_int64()};
      } else if (n.is_uint64()) {
        tag = {1, static_cast<int64_t>(n.get_uint64())};
      } else {
        const double d = n.get_double();
        if (std::isfinite(d) && d == std::floor(d) && std::fabs(d) < 9.2e18) {
          tag = {1, static_cast<int64_t>(d)};
        } else {
          char tmp[40];
          std::snprintf(tmp, sizeof tmp, "%.17g", d);
          tag = {2, tags.intern(tmp)};
        }
      }
      break;
    }
    case od::json_type::string:
      tag = {2, tags.intern(val.get_string().value())};
      break;
    default:
      break;
  }
  return tag;
}

// Writes Python's `str(val or "")` into `out` without extra heap allocations.
void assign_scalar_text(od::value val, std::string& out) {
  out.clear();
  switch (val.type().value()) {
    case od::json_type::string:
      out.assign(val.get_string().value());
      return;
    case od::json_type::number: {
      const od::number n = val.get_number().value();
      if (n.is_int64()) {
        const int64_t v = n.get_int64();
        if (v != 0) out = std::to_string(v);
        return;
      }
      if (n.is_uint64()) {
        const uint64_t v = n.get_uint64();
        if (v != 0) out = std::to_string(v);
        return;
      }
      const double d = n.get_double();
      if (d != 0.0) {
        char tmp[40];
        std::snprintf(tmp, sizeof tmp, "%.17g", d);
        out.assign(tmp);
      }
      return;
    }
    case od::json_type::boolean:
      if (val.get_bool().value()) out.assign("True");
      return;
    default:
      return;
  }
}

bool read_nonneg_arg(od::value val, int64_t& out) {
  switch (val.type().value()) {
    case od::json_type::number: {
      const od::number n = val.get_number().value();
      if (n.is_int64()) {
        const int64_t v = n.get_int64();
        if (v >= 0) {
          out = v;
          return true;
        }
        return false;
      }
      if (n.is_uint64()) {
        const uint64_t v = n.get_uint64();
        if (v <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
          out = static_cast<int64_t>(v);
          return true;
        }
        return false;
      }
      return false;
    }
    case od::json_type::string:
      return parse_nonneg_digits(val.get_string().value(), out);
    default:
      return false;
  }
}

bool read_signed_arg(od::value val, int64_t& out) {
  switch (val.type().value()) {
    case od::json_type::number: {
      const od::number n = val.get_number().value();
      if (n.is_int64()) {
        out = n.get_int64();
        return true;
      }
      if (n.is_uint64()) {
        const uint64_t v = n.get_uint64();
        if (v <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
          out = static_cast<int64_t>(v);
          return true;
        }
        return false;
      }
      return false;
    }
    case od::json_type::string:
      return parse_signed_digits(val.get_string().value(), out);
    default:
      return false;
  }
}

struct ParsedEvent {
  char ph = 0;
  std::string name;
  double ts = 0.0, dur = 0.0;
  Tag pid, tid;
  bool has_args_of_interest = false;
  bool has_step_num = false;
  bool has_hlo_op = false;
  int64_t step_num = 0;
  int64_t correlation_id = -1;
  int64_t device_ordinal = 0;
  std::string arg_name;
  std::string tf_op;
  std::string hlo_op;
  std::string hlo_module;
  std::string module;
  std::string memcpy_details;
  std::string memset_details;

  void reset() {
    ph = 0;
    name.clear();
    ts = dur = 0.0;
    pid = tid = Tag{};
    has_args_of_interest = false;
    has_step_num = false;
    has_hlo_op = false;
    step_num = 0;
    correlation_id = -1;
    device_ordinal = 0;
    arg_name.clear();
    tf_op.clear();
    hlo_op.clear();
    hlo_module.clear();
    module.clear();
    memcpy_details.clear();
    memset_details.clear();
  }
};

void read_event_args(od::value val, ParsedEvent& e) {
  if (val.type().value() != od::json_type::object) return;
  od::object args = val.get_object().value();
  for (auto field : args) {
    const std::string_view key = field.unescaped_key().value();
    if (key.size() < 4 || key.size() > 14) continue;
    od::value v = field.value().value();
    if (key == "name") {
      assign_scalar_text(v, e.arg_name);
      if (!e.arg_name.empty()) e.has_args_of_interest = true;
    } else if (key == "correlation_id") {
      int64_t cid = -1;
      if (read_nonneg_arg(v, cid)) {
        e.correlation_id = cid;
        e.has_args_of_interest = true;
      }
    } else if (key == "step_num") {
      int64_t sn = 0;
      if (read_signed_arg(v, sn)) {
        e.has_step_num = true;
        e.step_num = sn;
        e.has_args_of_interest = true;
      }
    } else if (key == "hlo_op") {
      e.has_hlo_op = true;
      assign_scalar_text(v, e.hlo_op);
      e.has_args_of_interest = true;
    } else if (key == "hlo_module") {
      assign_scalar_text(v, e.hlo_module);
      if (!e.hlo_module.empty()) e.has_args_of_interest = true;
    } else if (key == "module") {
      assign_scalar_text(v, e.module);
      if (!e.module.empty()) e.has_args_of_interest = true;
    } else if (key == "tf_op") {
      assign_scalar_text(v, e.tf_op);
      if (!e.tf_op.empty()) e.has_args_of_interest = true;
    } else if (key == "memcpy_details") {
      assign_scalar_text(v, e.memcpy_details);
      if (!e.memcpy_details.empty()) e.has_args_of_interest = true;
    } else if (key == "Memset_details") {
      assign_scalar_text(v, e.memset_details);
      if (!e.memset_details.empty()) e.has_args_of_interest = true;
    } else if (key == "device_ordinal") {
      int64_t ord = 0;
      if (read_nonneg_arg(v, ord)) {
        e.device_ordinal = ord;
        e.has_args_of_interest = true;
      }
    }
  }
}

void parse_event_obj(od::value ev, JsonChunk& chunk, ParsedEvent& e) {
  e.reset();
  od::object obj = ev.get_object().value();
  for (auto field : obj) {
    const std::string_view key = field.unescaped_key().value();
    if (key.size() < 2 || key.size() > 4) continue;
    od::value val = field.value().value();
    if (key == "ph") {
      if (val.type().value() == od::json_type::string) {
        const std::string_view ph = val.get_string().value();
        if (ph == "X") {
          e.ph = 'X';
        } else if (ph == "M") {
          e.ph = 'M';
        }
      }
    } else if (key == "name") {
      if (val.type().value() == od::json_type::string) {
        e.name.assign(val.get_string().value());
      } else {
        assign_scalar_text(val, e.name);
      }
    } else if (key == "ts") {
      e.ts = read_time(val);
    } else if (key == "dur") {
      e.dur = read_time(val);
    } else if (key == "pid") {
      e.pid = read_tag(val, chunk.tags);
    } else if (key == "tid") {
      e.tid = read_tag(val, chunk.tags);
    } else if (key == "args") {
      read_event_args(val, e);
    }
  }

  if (e.ph == 'M') {
    if (e.name == "process_name") {
      chunk.metas.push_back({false, e.pid, Tag{}, chunk.strings.intern(e.arg_name)});
    } else if (e.name == "thread_name") {
      chunk.metas.push_back({true, e.pid, e.tid, chunk.strings.intern(e.arg_name)});
    }
    return;
  }
  if (e.ph != 'X') return;

  const int64_t duration_ns = us_to_ns(e.dur);
  if (duration_ns <= 0) return;
  const int64_t start_ns = us_to_ns(e.ts);
  const int64_t end_ns = start_ns + duration_ns;

  int32_t extra_idx = -1;
  if (e.has_args_of_interest) {
    extra_idx = static_cast<int32_t>(chunk.extras.size());
    ExtraArgs ex;
    ex.has_step_num = e.has_step_num;
    ex.has_hlo_op = e.has_hlo_op;
    ex.step_num = e.step_num;
    ex.correlation_id = e.correlation_id;
    ex.device_ordinal = e.device_ordinal;
    if (!e.arg_name.empty()) ex.arg_name = chunk.strings.intern(e.arg_name);
    if (!e.tf_op.empty()) ex.tf_op = chunk.strings.intern(e.tf_op);
    if (!e.hlo_op.empty()) ex.hlo_op = chunk.strings.intern(e.hlo_op);
    if (!e.hlo_module.empty()) ex.hlo_module = chunk.strings.intern(e.hlo_module);
    if (!e.module.empty()) ex.module = chunk.strings.intern(e.module);
    if (!e.memcpy_details.empty()) ex.memcpy_details = chunk.strings.intern(e.memcpy_details);
    if (!e.memset_details.empty()) ex.memset_details = chunk.strings.intern(e.memset_details);
    chunk.extras.push_back(ex);
  }

  const int32_t tidx = chunk.thread_code(e.pid, e.tid);
  chunk.events.push_back({chunk.strings.intern(e.name), extra_idx, tidx, start_ns, end_ns});
}

void parse_chunk(od::parser& parser, std::string& text, JsonChunk& chunk) {
  try {
    const std::size_t len = text.size();
    text.reserve(len + simdjson::SIMDJSON_PADDING);
    simdjson::padded_string_view view(text.data(), len, text.capacity());
    od::document doc = parser.iterate(view).value();
    od::array events = doc.get_array().value();
    ParsedEvent e;
    for (auto item : events) {
      od::value ev = item.value();
      if (ev.type().value() != od::json_type::object) continue;
      parse_event_obj(ev, chunk, e);
    }
  } catch (const simdjson::simdjson_error& err) {
    throw std::invalid_argument(std::string("invalid JSON: ") + err.what());
  }
}

std::string_view lookup_str(const Interner& strings, int32_t code) {
  if (code < 0) return {};
  return strings.items()[static_cast<std::size_t>(code)];
}

struct ResolvedLocalThread {
  bool is_device = false;
  bool line_allowed = true;
  int64_t device_ordinal = 0;
  int64_t stream = 0;
  int64_t host_thread_id = -1;  // assigned lazily on first host api call for file-order determinism
  std::string pt_key;
};

JaxColumns merge_chunks(const std::vector<std::unique_ptr<JsonChunk>>& chunks) {
  // Pass 1: resolve process_name -> device ordinals and thread_name -> line
  // prefixes across ALL chunks (later metadata event overrides earlier one).
  std::unordered_map<std::string, std::string> process_names;
  std::unordered_map<std::string, std::string> thread_names;
  std::unordered_map<std::string, std::vector<std::string>> pid_thread_keys;

  for (const auto& cp : chunks) {
    const JsonChunk& c = *cp;
    for (const MetaEvent& m : c.metas) {
      const std::string pid_key = encode_tag(m.pid, c.tags);
      const std::string_view mname = lookup_str(c.strings, m.name_code);
      if (!m.is_thread) {
        process_names[pid_key] = std::string(mname);
      } else {
        const std::string pt_key = pid_key + encode_tag(m.tid, c.tags);
        auto [it, inserted] = thread_names.emplace(pt_key, std::string(mname));
        if (!inserted) {
          it->second.assign(mname);
        } else {
          pid_thread_keys[pid_key].push_back(pt_key);
        }
      }
    }
  }

  std::unordered_map<std::string, int64_t> devices_by_pid;
  std::unordered_map<std::string, std::string_view> line_prefix_by_pid;
  for (const auto& [pid_key, pname] : process_names) {
    int64_t ord = 0;
    if (parse_device_ordinal(pname, ord)) {
      devices_by_pid[pid_key] = ord;
      std::vector<std::string_view> lnames;
      if (auto tit = pid_thread_keys.find(pid_key); tit != pid_thread_keys.end()) {
        lnames.reserve(tit->second.size());
        for (const std::string& pt : tit->second) lnames.emplace_back(thread_names[pt]);
      }
      line_prefix_by_pid[pid_key] = select_device_line_prefix(lnames);
    }
  }

  // Pass 2: pre-resolve each chunk's distinct (pid, tid) entries once, then
  // scan events in file order with zero per-event string allocations or hashes.
  JaxEventBuilder builder("json");
  std::unordered_map<std::string, int64_t> host_thread_ids;
  const ExtraArgs kEmptyExtra{};
  std::vector<ResolvedLocalThread> local_threads;

  for (const auto& cp : chunks) {
    const JsonChunk& c = *cp;
    local_threads.clear();
    local_threads.resize(c.threads.size());
    for (std::size_t ti = 0; ti < c.threads.size(); ++ti) {
      const ThreadKey& tk = c.threads[ti];
      ResolvedLocalThread& rt = local_threads[ti];
      const std::string pid_key = encode_tag(tk.pid, c.tags);
      rt.pt_key = pid_key + encode_tag(tk.tid, c.tags);
      rt.stream = euclidean_mod_stream(tag_as_int(tk.tid, c.tags));
      if (const auto dev_it = devices_by_pid.find(pid_key); dev_it != devices_by_pid.end()) {
        rt.is_device = true;
        rt.device_ordinal = dev_it->second;
        const std::string_view prefix = line_prefix_by_pid[pid_key];
        if (!prefix.empty()) {
          const auto tnit = thread_names.find(rt.pt_key);
          const std::string_view tname = tnit != thread_names.end() ? std::string_view(tnit->second) : "";
          rt.line_allowed = tname.starts_with(prefix);
        }
      }
    }

    for (const RawEvent& ev : c.events) {
      ResolvedLocalThread& rt = local_threads[static_cast<std::size_t>(ev.thread_idx)];
      if (!rt.is_device && ev.extra_idx < 0) continue;  // host event with no relevant args: fast skip!

      const ExtraArgs& ex = ev.extra_idx >= 0 ? c.extras[static_cast<std::size_t>(ev.extra_idx)] : kEmptyExtra;

      if (ex.has_step_num && !rt.is_device) {
        builder.add_step(ex.step_num, ev.start_ns, ev.end_ns);
        continue;
      }

      const std::string_view name = lookup_str(c.strings, ev.name_code);
      const std::string_view arg_name = lookup_str(c.strings, ex.arg_name);
      if (!rt.is_device && name.starts_with(kExecutePrefix) && !arg_name.empty()) {
        builder.add_execution(arg_name, ev.start_ns, ev.end_ns);
      }

      if (!rt.is_device && !ex.has_hlo_op) {
        if (ex.correlation_id >= 0) {
          if (rt.host_thread_id < 0) {
            auto [it, inserted] =
                host_thread_ids.emplace(rt.pt_key, static_cast<int64_t>(host_thread_ids.size()));
            rt.host_thread_id = it->second;
          }
          builder.add_api_call(name, ev.start_ns, ev.end_ns, rt.host_thread_id, ex.correlation_id);
        }
        continue;
      }

      if (rt.is_device && !rt.line_allowed) continue;

      const std::string_view hlo_op = lookup_str(c.strings, ex.hlo_op);
      const std::string_view hlo_mod = lookup_str(c.strings, ex.hlo_module);
      const std::string_view mod = hlo_mod.empty() ? lookup_str(c.strings, ex.module) : hlo_mod;
      const std::string_view tf_op = lookup_str(c.strings, ex.tf_op);
      const std::string_view memcpy_det = lookup_str(c.strings, ex.memcpy_details);
      const std::string_view details =
          memcpy_det.empty() ? lookup_str(c.strings, ex.memset_details) : memcpy_det;
      const int64_t device = rt.is_device ? rt.device_ordinal : ex.device_ordinal;

      builder.add_op(name, arg_name, tf_op, hlo_op, mod, details, ev.start_ns, ev.end_ns, rt.stream, device,
                     ex.correlation_id);
    }
  }
  return builder.finish();
}

}  // namespace

JaxColumns read_jax_json(const std::string& path, int threads, std::uint64_t chunk_bytes,
                         std::uint64_t block_bytes) {
  auto source = open_source(path);
  using Clock = std::chrono::steady_clock;
  double read_s = 0.0, split_s = 0.0, wait_s = 0.0;

  std::vector<std::unique_ptr<JsonChunk>> ordered_chunks;
  using Pool = OrderedPool<std::string, std::unique_ptr<JsonChunk>>;
  Pool pool(
      threads,
      []() -> Pool::Work {
        auto parser = std::make_shared<od::parser>();
        return [parser](std::string& text) {
          auto out = std::make_unique<JsonChunk>();
          parse_chunk(*parser, text, *out);
          return out;
        };
      },
      [&](std::unique_ptr<JsonChunk>&& out) { ordered_chunks.push_back(std::move(out)); },
      static_cast<std::size_t>(threads) + 4);

  JsonSplitter splitter(static_cast<std::size_t>(chunk_bytes), [&](std::string&& chunk) {
    const auto t0 = Clock::now();
    pool.submit(std::move(chunk));
    wait_s += duration_seconds(Clock::now() - t0);
  });

  std::vector<char> block(
      static_cast<std::size_t>(std::min<std::uint64_t>(block_bytes, std::uint64_t{1} << 30)));
  while (true) {
    const auto t0 = Clock::now();
    const std::size_t n = source->read(block.data(), block.size());
    const auto t1 = Clock::now();
    read_s += duration_seconds(t1 - t0);
    if (n == 0) break;
    splitter.feed(block.data(), n);
    split_s += duration_seconds(Clock::now() - t1);
  }
  splitter.finish();

  const auto t_finish = Clock::now();
  pool.finish();
  JaxColumns columns = merge_chunks(ordered_chunks);
  columns.timing.finish = duration_seconds(Clock::now() - t_finish);
  columns.timing.read = read_s;
  columns.timing.submit_wait = wait_s;
  columns.timing.split = split_s - wait_s;
  return columns;
}

}  // namespace cta::readers::jax
