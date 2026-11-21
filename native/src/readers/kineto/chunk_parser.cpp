#include "readers/kineto/chunk_parser.hpp"

#include <simdjson.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string_view>

#include "util/util.hpp"

namespace cta::readers::kineto {
namespace od = simdjson::ondemand;

static_assert(kChunkPadding >= simdjson::SIMDJSON_PADDING, "chunk padding is below what simdjson reads past the end");

int32_t ChunkOut::name_code(std::string_view s) {
  bool is_new = false;
  const int32_t code = names.intern(s, &is_new);
  if (is_new) nccl.push_back(contains_nccl(s) ? 1 : 0);
  return code;
}

int32_t ChunkOut::thread_code(const Tag& pid, const Tag& tid) {
  ThreadKey key{pid, tid};
  auto it = thread_index.find(key);
  if (it != thread_index.end()) return it->second;
  const int32_t id = static_cast<int32_t>(threads.size());
  threads.push_back(key);
  thread_index.emplace(key, id);
  return id;
}

namespace {

// ---------------------------------------------------------------------------
// Leftmost match of Python's r"profilerstep#\s*(\d+)" (case-insensitive).
bool find_step(std::string_view name, int64_t& id) {
  static constexpr std::string_view kKey = "profilerstep";
  std::size_t from = 0;
  while (true) {
    const std::size_t hash = name.find('#', from);
    if (hash == std::string_view::npos) return false;
    from = hash + 1;
    if (hash < kKey.size()) continue;
    bool match = true;
    for (std::size_t i = 0; i < kKey.size(); ++i) {
      if (static_cast<char>(name[hash - kKey.size() + i] | 0x20) != kKey[i]) {
        match = false;
        break;
      }
    }
    if (!match) continue;
    std::size_t q = hash + 1;
    while (q < name.size() && (name[q] == ' ' || (name[q] >= '\t' && name[q] <= '\r'))) ++q;
    const std::size_t first_digit = q;
    int64_t value = 0;
    while (q < name.size() && name[q] >= '0' && name[q] <= '9') {
      if (value < 100000000000000000LL) value = value * 10 + (name[q] - '0');
      ++q;
    }
    if (q == first_digit) continue;
    id = value;
    return true;
  }
}

// ---------------------------------------------------------------------------
// JSON values to numbers and tags. Wrong types become 0 or "absent" instead of
// failing, except ts and dur, which are required to be numbers.

double number_as_double(od::number n) {
  if (n.is_int64()) return static_cast<double>(n.get_int64());
  if (n.is_uint64()) return static_cast<double>(n.get_uint64());
  return n.get_double();
}

int64_t text_as_int(std::string_view s) {
  char tmp[40];
  if (s.empty() || s.size() >= sizeof tmp) return 0;
  std::memcpy(tmp, s.data(), s.size());
  tmp[s.size()] = '\0';
  char* end = nullptr;
  const long long v = std::strtoll(tmp, &end, 10);
  return (end && *end == '\0') ? static_cast<int64_t>(v) : 0;
}

// ts and dur are in microseconds and become nanoseconds in an int64, with room for start + duration.
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

// int(x) the way the Python reader took it: numbers truncate, numeric text
// parses, everything else (null, bool, containers) is 0.
int64_t read_int(od::value val, bool* is_integer = nullptr) {
  if (is_integer) *is_integer = false;
  switch (val.type().value()) {
    case od::json_type::number: {
      const od::number n = val.get_number().value();
      if (n.is_int64()) {
        if (is_integer) *is_integer = true;
        return n.get_int64();
      }
      if (n.is_uint64()) {
        if (is_integer) *is_integer = true;
        return static_cast<int64_t>(n.get_uint64());
      }
      const double d = n.get_double();
      return std::isfinite(d) && std::fabs(d) < 9.2e18 ? static_cast<int64_t>(d) : 0;
    }
    case od::json_type::string:
      return text_as_int(val.get_string().value());
    default:
      return 0;
  }
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
          tag = {1, static_cast<int64_t>(d)};  // 1.0 and 1 are the same Python dict key
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

int64_t tag_as_int(const Tag& tag, const Interner& tags) {
  if (tag.kind == 1) return tag.v;
  if (tag.kind == 2) return text_as_int(tags.items()[static_cast<std::size_t>(tag.v)]);
  return 0;
}

// ---------------------------------------------------------------------------
// One event's fields, as far as they are read.
struct Event {
  bool is_x = false;
  std::string cat;   // lower-cased
  std::string name;
  double ts = 0.0, dur = 0.0;
  Tag pid, tid;
  bool has_corr = false, has_stream = false, has_bytes = false, has_device = false;
  int64_t corr = 0, stream = 0, bytes = 0, device = 0;
  // Collective metadata PyTorch attaches to NCCL kernels and record_param_comms ops.
  bool has_coll = false;
  std::string coll, coll_dtype, coll_group;
  int64_t coll_in = 0, coll_out = 0, coll_size = 0;

  void reset() {
    is_x = false;
    cat.clear();
    name.clear();
    ts = dur = 0.0;
    pid = tid = Tag{};
    has_corr = has_stream = has_bytes = has_device = false;
    corr = stream = bytes = device = 0;
    has_coll = false;
    coll.clear();
    coll_dtype.clear();
    coll_group.clear();
    coll_in = coll_out = coll_size = 0;
  }
};

void read_args(od::value val, Event& e) {
  if (val.type().value() != od::json_type::object) return;
  od::object args = val.get_object().value();
  for (auto field : args) {
    std::string_view key = field.unescaped_key().value();
    if (key == "correlation") {
      e.has_corr = true;
      e.corr = read_int(field.value().value());
    } else if (key == "stream") {
      e.has_stream = true;
      e.stream = read_int(field.value().value());
    } else if (key == "bytes") {
      e.has_bytes = true;
      e.bytes = read_int(field.value().value());
    } else if (key == "device") {
      bool is_integer = false;
      const int64_t d = read_int(field.value().value(), &is_integer);
      if (is_integer) {
        e.has_device = true;
        e.device = d;
      }
    } else if (key == "Collective name") {
      if (field.value().value().type().value() == od::json_type::string) {
        e.has_coll = true;
        e.coll.assign(field.value().value().get_string().value());
      }
    } else if (key == "In msg nelems") {
      e.coll_in = read_int(field.value().value());
    } else if (key == "Out msg nelems") {
      e.coll_out = read_int(field.value().value());
    } else if (key == "Group size") {
      e.coll_size = read_int(field.value().value());
    } else if (key == "dtype") {
      if (field.value().value().type().value() == od::json_type::string)
        e.coll_dtype.assign(field.value().value().get_string().value());
    } else if (key == "Process Group Name") {
      if (field.value().value().type().value() == od::json_type::string)
        e.coll_group.assign(field.value().value().get_string().value());
    }
  }
}

bool is_cpu_op(const std::string& c) { return c == "cpu_op" || c == "operator" || c == "user_annotation"; }
bool is_runtime(const std::string& c) { return c == "cuda_runtime" || c == "cuda_driver" || c == "runtime"; }
bool is_gpu(const std::string& c) {
  return c == "kernel" || c == "gpu_memcpy" || c == "gpu_memset" || c == "memcpy" || c == "memset";
}

// Python's int(round(x * 1000.0)): round half to even.
int64_t to_ns(double us) { return static_cast<int64_t>(std::nearbyint(us * 1000.0)); }

void add_event(od::value ev, ChunkOut& out, Event& e) {
  e.reset();
  od::object obj = ev.get_object().value();
  for (auto field : obj) {
    std::string_view key = field.unescaped_key().value();
    if (key.size() < 2 || key.size() > 4) continue;
    od::value val = field.value().value();
    if (key == "ph") {
      if (val.type().value() == od::json_type::string) e.is_x = (val.get_string().value() == "X");
    } else if (key == "cat") {
      if (val.type().value() == od::json_type::string) {
        std::string_view s = val.get_string().value();
        e.cat.resize(s.size());
        for (std::size_t i = 0; i < s.size(); ++i) {
          const char c = s[i];
          e.cat[i] = (c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : c;
        }
      }
    } else if (key == "name") {
      if (val.type().value() == od::json_type::string) e.name.assign(val.get_string().value());
    } else if (key == "ts") {
      e.ts = read_time(val);
    } else if (key == "dur") {
      e.dur = read_time(val);
    } else if (key == "pid") {
      e.pid = read_tag(val, out.tags);
    } else if (key == "tid") {
      e.tid = read_tag(val, out.tags);
    } else if (key == "args") {
      read_args(val, e);
    }
  }
  if (!e.is_x) return;
  const int64_t start = to_ns(e.ts);
  const int64_t duration = to_ns(e.dur);
  if (duration <= 0) return;
  const int64_t end = start + duration;

  int64_t step = 0;
  const bool is_step = find_step(e.name, step);
  if (is_step) {
    // PyTorch also emits each step as a gpu_user_annotation, the GPU-side copy
    // of the same step; counting it would add a second step with the same id.
    if (e.cat != "gpu_user_annotation") {
      out.s_id.push_back(step);
      out.s_start.push_back(start);
      out.s_end.push_back(end);
    }
  } else if (is_cpu_op(e.cat)) {
    out.o_start.push_back(start);
    out.o_end.push_back(end);
    out.o_name.push_back(out.name_code(e.name));
    out.o_thread.push_back(out.thread_code(e.pid, e.tid));
  } else if (is_runtime(e.cat)) {
    const int32_t code = out.name_code(e.name);
    if (out.r_slot.size() <= static_cast<std::size_t>(code)) out.r_slot.resize(out.names.size(), -1);
    int32_t& slot = out.r_slot[static_cast<std::size_t>(code)];
    if (slot < 0) {
      slot = static_cast<int32_t>(out.r_name.size());
      out.r_name.push_back(code);
      out.r_dur.push_back(0);
      out.r_calls.push_back(0);
    }
    out.r_dur[static_cast<std::size_t>(slot)] += duration;
    out.r_calls[static_cast<std::size_t>(slot)] += 1;
    out.rc_name.push_back(code);
    out.rc_start.push_back(start);
    out.rc_end.push_back(end);
    out.rc_thread.push_back(out.thread_code(e.pid, e.tid));
    out.rc_corr.push_back(e.has_corr ? e.corr : -1);
    if (e.has_corr) {
      out.l_corr.push_back(e.corr);
      out.l_start.push_back(start);
      out.l_thread.push_back(out.thread_code(e.pid, e.tid));
    }
  }

  // A step-named kernel is still a GPU row: the Python reader's second pass
  // never looked at step names.
  if (is_gpu(e.cat)) {
    const int32_t code = out.name_code(e.name);
    std::int8_t category = 0;
    if (e.cat.find("memcpy") != std::string::npos || e.cat.find("memset") != std::string::npos) {
      category = 2;
    } else if (out.nccl[static_cast<std::size_t>(code)]) {
      category = 1;
    }
    out.g_name.push_back(code);
    out.g_cat.push_back(category);
    out.g_start.push_back(start);
    out.g_end.push_back(end);
    out.g_bytes.push_back(e.has_bytes ? e.bytes : 0);
    out.g_stream.push_back(e.has_stream ? e.stream : tag_as_int(e.tid, out.tags));
    out.g_device.push_back(e.has_device ? e.device : tag_as_int(e.pid, out.tags));
    out.g_corr.push_back(e.has_corr ? e.corr : -1);
    if (e.has_coll) {
      out.c_row.push_back(static_cast<int64_t>(out.g_name.size()) - 1);
      out.c_op.push_back(out.name_code(e.coll));
      out.c_dtype.push_back(out.name_code(e.coll_dtype));
      out.c_group.push_back(out.name_code(e.coll_group));
      out.c_in.push_back(e.coll_in);
      out.c_out.push_back(e.coll_out);
      out.c_size.push_back(e.coll_size);
    }
  }
}

}  // namespace

struct ChunkParser::Impl {
  od::parser parser;
};

ChunkParser::ChunkParser() : impl_(std::make_unique<Impl>()) {}
ChunkParser::~ChunkParser() = default;

void ChunkParser::parse(std::string& text, ChunkOut& out) {
  try {
    const std::size_t len = text.size();
    text.reserve(len + simdjson::SIMDJSON_PADDING);
    simdjson::padded_string_view view(text.data(), len, text.capacity());
    od::document doc = impl_->parser.iterate(view).value();
    od::array events = doc.get_array().value();
    Event e;
    for (auto item : events) {
      od::value ev = item.value();
      if (ev.type().value() != od::json_type::object) continue;
      add_event(ev, out, e);
    }
  } catch (const simdjson::simdjson_error& e) {
    throw std::invalid_argument(std::string("invalid JSON: ") + e.what());
  }
}

std::optional<std::int64_t> parse_rank(const std::string& info) {
  if (info.empty()) return std::nullopt;
  try {
    simdjson::padded_string padded(info);
    od::parser parser;
    od::document doc = parser.iterate(padded).value();
    od::object obj = doc.get_object().value();
    for (auto field : obj) {
      if (field.unescaped_key().value() == "rank") {
        od::value v = field.value().value();
        if (v.type().value() != od::json_type::number) return std::nullopt;
        const od::number n = v.get_number().value();
        if (n.is_int64()) return n.get_int64();
        return std::nullopt;
      }
    }
  } catch (const simdjson::simdjson_error&) {
  }
  return std::nullopt;
}

}  // namespace cta::readers::kineto
