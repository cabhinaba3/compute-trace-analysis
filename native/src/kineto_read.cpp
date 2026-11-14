#include "kineto_read.hpp"

#include <simdjson.h>
#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_map>

#if defined(__linux__)
#include <sched.h>
#endif

namespace cta {
namespace {

namespace od = simdjson::ondemand;
using std::int32_t;
using std::int64_t;

constexpr std::size_t kInfoLimit = 4u << 20;  // largest distributedInfo kept

int usable_cpus() {
#if defined(__linux__)
  cpu_set_t set;
  if (sched_getaffinity(0, sizeof set, &set) == 0) return std::max(1, CPU_COUNT(&set));
#endif
  return static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
}

// ---------------------------------------------------------------------------
// Distinct strings numbered in order of first appearance.
class Interner {
 public:
  int32_t intern(std::string_view s, bool* is_new = nullptr) {
    auto it = index_.find(s);
    if (it != index_.end()) {
      if (is_new) *is_new = false;
      return it->second;
    }
    const int32_t id = static_cast<int32_t>(items_.size());
    items_.emplace_back(s);  // a deque never moves its elements, so the views stay valid
    index_.emplace(std::string_view(items_.back()), id);
    if (is_new) *is_new = true;
    return id;
  }
  const std::deque<std::string>& items() const { return items_; }
  std::size_t size() const { return items_.size(); }

 private:
  std::deque<std::string> items_;
  std::unordered_map<std::string_view, int32_t> index_;
};

// ---------------------------------------------------------------------------
// A pid or tid: absent/null, an integer, or text (v = id in the tag interner).
struct Tag {
  std::int8_t kind = 0;
  int64_t v = 0;
  bool operator==(const Tag& o) const { return kind == o.kind && v == o.v; }
};
struct ThreadKey {
  Tag pid, tid;
  bool operator==(const ThreadKey& o) const { return pid == o.pid && tid == o.tid; }
};
struct ThreadKeyHash {
  std::size_t operator()(const ThreadKey& k) const {
    std::uint64_t h = static_cast<std::uint64_t>(k.pid.v) * 0x9e3779b97f4a7c15ULL;
    h ^= static_cast<std::uint64_t>(k.tid.v) + 0x7f4a7c15 + (h << 6) + (h >> 2);
    h ^= static_cast<std::uint64_t>(k.pid.kind) * 31 + static_cast<std::uint64_t>(k.tid.kind);
    return static_cast<std::size_t>(h);
  }
};

// ---------------------------------------------------------------------------
// What one parsed chunk holds. Strings and thread keys are local to the chunk
// and renumbered when the chunk is merged.
struct ChunkOut {
  Interner names;
  std::vector<std::uint8_t> nccl;  // per local name: contains "nccl", any case
  Interner tags;                   // text pids and tids
  std::vector<ThreadKey> threads;
  std::unordered_map<ThreadKey, int32_t, ThreadKeyHash> thread_index;

  std::vector<int32_t> g_name;
  std::vector<std::int8_t> g_cat;
  std::vector<int64_t> g_start, g_end, g_bytes, g_stream, g_device, g_corr;

  std::vector<int64_t> o_start, o_end;
  std::vector<int32_t> o_name, o_thread;

  std::vector<int64_t> s_id, s_start, s_end;

  std::vector<int32_t> r_name;  // runtime names in first-seen order
  std::vector<int64_t> r_dur, r_calls;
  std::vector<int32_t> r_slot;  // local name code -> index into r_*, or -1

  std::vector<int64_t> l_corr, l_start;
  std::vector<int32_t> l_thread;

  int32_t name_code(std::string_view s);
  int32_t thread_code(const Tag& pid, const Tag& tid);
};

bool contains_nccl(std::string_view s) {
  for (std::size_t i = 0; i + 4 <= s.size(); ++i) {
    auto low = [&](std::size_t j) { return static_cast<char>(s[i + j] | 0x20); };
    if (low(0) == 'n' && low(1) == 'c' && low(2) == 'c' && low(3) == 'l') return true;
  }
  return false;
}

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

double read_time(od::value val) {
  switch (val.type().value()) {
    case od::json_type::number:
      return number_as_double(val.get_number().value());
    case od::json_type::string: {
      std::string s(val.get_string().value());
      char* end = nullptr;
      const double d = std::strtod(s.c_str(), &end);
      if (!s.empty() && end && *end == '\0') return d;
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

  void reset() {
    is_x = false;
    cat.clear();
    name.clear();
    ts = dur = 0.0;
    pid = tid = Tag{};
    has_corr = has_stream = has_bytes = has_device = false;
    corr = stream = bytes = device = 0;
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
  }
}

// ---------------------------------------------------------------------------
struct Chunk {
  std::size_t seq = 0;
  std::string data;  // "[" + events separated by commas + "]"
};

void parse_chunk(Chunk& chunk, od::parser& parser, ChunkOut& out) {
  const std::size_t len = chunk.data.size();
  chunk.data.reserve(len + simdjson::SIMDJSON_PADDING);
  simdjson::padded_string_view view(chunk.data.data(), len, chunk.data.capacity());
  od::document doc = parser.iterate(view).value();
  od::array events = doc.get_array().value();
  Event e;
  for (auto item : events) {
    od::value ev = item.value();
    if (ev.type().value() != od::json_type::object) continue;
    add_event(ev, out, e);
  }
}

// ---------------------------------------------------------------------------
// Open-addressing map from a correlation id to the CUDA runtime call that
// carried it; a later call with the same id replaces the earlier one.
class LaunchMap {
 public:
  void put(int64_t key, int64_t start, int32_t thread) {
    if (key == kEmpty) return;
    if ((size_ + 1) * 2 > slots_.size()) grow();
    Slot& s = slots_[probe(key)];
    if (s.key == kEmpty) {
      s.key = key;
      ++size_;
    }
    s.start = start;
    s.thread = thread;
  }
  struct Hit {
    int64_t start;
    int32_t thread;
  };
  bool find(int64_t key, Hit& hit) const {
    if (slots_.empty() || key == kEmpty) return false;
    const Slot& s = slots_[probe(key)];
    if (s.key == kEmpty) return false;
    hit = {s.start, s.thread};
    return true;
  }

 private:
  static constexpr int64_t kEmpty = std::numeric_limits<int64_t>::min();
  struct Slot {
    int64_t key = kEmpty;
    int64_t start = 0;
    int32_t thread = 0;
  };
  static std::uint64_t mix(std::uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
  }
  std::size_t probe(int64_t key) const {
    const std::size_t mask = slots_.size() - 1;
    std::size_t i = mix(static_cast<std::uint64_t>(key)) & mask;
    while (slots_[i].key != kEmpty && slots_[i].key != key) i = (i + 1) & mask;
    return i;
  }
  void grow() {
    std::vector<Slot> old = std::move(slots_);
    slots_.assign(old.empty() ? 1024 : old.size() * 2, Slot{});
    size_ = 0;
    for (const Slot& s : old) {
      if (s.key != kEmpty) {
        slots_[probe(s.key)] = s;
        ++size_;
      }
    }
  }
  std::vector<Slot> slots_;
  std::size_t size_ = 0;
};

// Folds parsed chunks, in file order, into one set of columns.
class Merger {
 public:
  void add(ChunkOut& c) {
    std::vector<int32_t> name_map;
    name_map.reserve(c.names.size());
    for (const std::string& s : c.names.items()) name_map.push_back(names_.intern(s));

    std::vector<int32_t> thread_map;
    thread_map.reserve(c.threads.size());
    for (const ThreadKey& key : c.threads) {
      const std::string text = encode(key.pid, c.tags) + encode(key.tid, c.tags);
      auto it = thread_ids_.try_emplace(text, static_cast<int32_t>(thread_ids_.size())).first;
      thread_map.push_back(it->second);
    }

    for (int32_t n : c.g_name) out_.gpu_name.push_back(name_map[static_cast<std::size_t>(n)]);
    append(out_.gpu_category, c.g_cat);
    append(out_.gpu_start, c.g_start);
    append(out_.gpu_end, c.g_end);
    append(out_.gpu_bytes, c.g_bytes);
    append(out_.gpu_stream, c.g_stream);
    append(out_.gpu_device, c.g_device);
    append(gpu_corr_, c.g_corr);

    append(out_.op_start, c.o_start);
    append(out_.op_end, c.o_end);
    for (int32_t n : c.o_name) out_.op_name.push_back(name_map[static_cast<std::size_t>(n)]);
    for (int32_t t : c.o_thread) out_.op_thread.push_back(thread_map[static_cast<std::size_t>(t)]);

    append(out_.step_id, c.s_id);
    append(out_.step_start, c.s_start);
    append(out_.step_end, c.s_end);

    for (std::size_t j = 0; j < c.r_name.size(); ++j) {
      const std::size_t code = static_cast<std::size_t>(name_map[static_cast<std::size_t>(c.r_name[j])]);
      if (runtime_slot_.size() <= code) runtime_slot_.resize(names_.size(), -1);
      if (runtime_slot_[code] < 0) {
        runtime_slot_[code] = static_cast<int32_t>(out_.runtime_name.size());
        out_.runtime_name.push_back(static_cast<int32_t>(code));
        out_.runtime_duration.push_back(0);
        out_.runtime_calls.push_back(0);
      }
      const std::size_t slot = static_cast<std::size_t>(runtime_slot_[code]);
      out_.runtime_duration[slot] += c.r_dur[j];
      out_.runtime_calls[slot] += c.r_calls[j];
    }

    for (std::size_t j = 0; j < c.l_corr.size(); ++j)
      launches_.put(c.l_corr[j], c.l_start[j], thread_map[static_cast<std::size_t>(c.l_thread[j])]);
  }

  KinetoColumns finish() {
    // A kernel can come before its launch in the file, so launches are
    // matched only now that every chunk is in.
    const std::size_t n = gpu_corr_.size();
    out_.gpu_launch_ns.resize(n);
    out_.gpu_launch_thread.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      LaunchMap::Hit hit{};
      if (launches_.find(gpu_corr_[i], hit)) {
        out_.gpu_launch_ns[i] = static_cast<double>(hit.start);
        out_.gpu_launch_thread[i] = hit.thread;
      } else {
        out_.gpu_launch_ns[i] = std::numeric_limits<double>::quiet_NaN();
        out_.gpu_launch_thread[i] = -1;
      }
    }
    out_.names.assign(names_.items().begin(), names_.items().end());
    return std::move(out_);
  }

 private:
  template <typename T>
  static void append(std::vector<T>& to, const std::vector<T>& from) {
    to.insert(to.end(), from.begin(), from.end());
  }
  static std::string encode(const Tag& tag, const Interner& tags) {
    std::string s(1, static_cast<char>(tag.kind));
    if (tag.kind == 1) {
      s.append(reinterpret_cast<const char*>(&tag.v), sizeof tag.v);
    } else if (tag.kind == 2) {
      const std::string& text = tags.items()[static_cast<std::size_t>(tag.v)];
      const std::uint64_t len = text.size();
      s.append(reinterpret_cast<const char*>(&len), sizeof len);
      s.append(text);
    }
    return s;
  }

  KinetoColumns out_;
  Interner names_;
  std::unordered_map<std::string, int32_t> thread_ids_;
  std::vector<int32_t> runtime_slot_;
  std::vector<int64_t> gpu_corr_;
  LaunchMap launches_;
};

// ---------------------------------------------------------------------------
// Parses chunks on worker threads and merges them in order on one more thread.
// At most `max_in_flight` chunks (queued, parsing or waiting to merge) exist at
// a time, which bounds memory however fast the splitter runs.
class Pipeline {
 public:
  explicit Pipeline(int workers) : max_in_flight_(static_cast<std::size_t>(workers) + 4) {
    for (int i = 0; i < workers; ++i) workers_.emplace_back([this] { work(); });
    merger_thread_ = std::thread([this] { merge(); });
  }
  ~Pipeline() { stop(); }
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;

  void submit(std::string&& data) {
    std::unique_lock<std::mutex> lock(m_);
    room_.wait(lock, [&] { return aborted_ || next_submit_ - next_merge_ < max_in_flight_; });
    if (aborted_) {
      if (error_) std::rethrow_exception(error_);
      throw std::runtime_error("trace reader stopped");
    }
    queue_.push_back(Chunk{next_submit_++, std::move(data)});
    work_.notify_one();
  }

  KinetoColumns finish() {
    {
      std::lock_guard<std::mutex> lock(m_);
      closed_ = true;
    }
    work_.notify_all();
    result_.notify_all();
    for (auto& t : workers_) t.join();
    merger_thread_.join();
    if (error_) std::rethrow_exception(error_);
    return merger_.finish();
  }

 private:
  void fail(std::exception_ptr e) {
    {
      std::lock_guard<std::mutex> lock(m_);
      if (!error_) error_ = e;
      aborted_ = true;
    }
    work_.notify_all();
    room_.notify_all();
    result_.notify_all();
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(m_);
      aborted_ = true;
      closed_ = true;
    }
    work_.notify_all();
    room_.notify_all();
    result_.notify_all();
    for (auto& t : workers_)
      if (t.joinable()) t.join();
    if (merger_thread_.joinable()) merger_thread_.join();
  }

  void work() {
    od::parser parser;
    while (true) {
      Chunk chunk;
      {
        std::unique_lock<std::mutex> lock(m_);
        work_.wait(lock, [&] { return aborted_ || !queue_.empty() || closed_; });
        if (aborted_ || queue_.empty()) return;
        chunk = std::move(queue_.front());
        queue_.pop_front();
      }
      try {
        auto out = std::make_unique<ChunkOut>();
        parse_chunk(chunk, parser, *out);
        chunk.data = std::string();
        {
          std::lock_guard<std::mutex> lock(m_);
          done_.emplace(chunk.seq, std::move(out));
        }
        result_.notify_all();
      } catch (const simdjson::simdjson_error& e) {
        fail(std::make_exception_ptr(std::invalid_argument(std::string("invalid JSON: ") + e.what())));
        return;
      } catch (...) {
        fail(std::current_exception());
        return;
      }
    }
  }

  void merge() {
    while (true) {
      std::unique_ptr<ChunkOut> out;
      {
        std::unique_lock<std::mutex> lock(m_);
        result_.wait(lock, [&] {
          return aborted_ || done_.count(next_merge_) > 0 || (closed_ && next_merge_ == next_submit_);
        });
        if (aborted_) return;
        auto it = done_.find(next_merge_);
        if (it == done_.end()) return;  // closed and everything merged
        out = std::move(it->second);
        done_.erase(it);
        ++next_merge_;
      }
      room_.notify_all();
      try {
        merger_.add(*out);
      } catch (...) {
        fail(std::current_exception());
        return;
      }
    }
  }

  const std::size_t max_in_flight_;
  std::mutex m_;
  std::condition_variable work_, room_, result_;
  std::deque<Chunk> queue_;
  std::map<std::size_t, std::unique_ptr<ChunkOut>> done_;
  std::size_t next_submit_ = 0, next_merge_ = 0;
  bool closed_ = false, aborted_ = false;
  std::exception_ptr error_;
  std::vector<std::thread> workers_;
  std::thread merger_thread_;
  Merger merger_;
};

// ---------------------------------------------------------------------------
// Walks the JSON text block by block and cuts the trace-event array into
// chunks at the commas between events. It tracks string and escape state, so
// braces and quotes inside names never confuse it, and it keeps the value of
// the top-level "distributedInfo" key wherever that sits in the file.
class Splitter {
 public:
  Splitter(std::size_t chunk_bytes, std::function<void(std::string&&)> emit)
      : chunk_bytes_(chunk_bytes), emit_(std::move(emit)) {}

  void feed(const char* base, std::size_t n) {
    const char* p = base;
    const char* const end = base + n;
    const char* seg = in_events_ ? base : nullptr;  // start of event bytes not yet copied
    auto flush = [&](const char* upto) {
      if (seg && upto > seg) buf_.append(seg, static_cast<std::size_t>(upto - seg));
      seg = nullptr;
    };

    while (p < end) {
      char c = *p;
      if (in_string_) {
        const bool capturing = capture_key_ || capture_info_;
        if (escape_) {
          escape_ = false;
          if (capturing) note(c);
          ++p;
          continue;
        }
        if (!capturing) {
          while (p < end && *p != '"' && *p != '\\') ++p;
          if (p == end) break;
          c = *p;
        }
        if (c == '\\') {
          escape_ = true;
          if (capturing) note(c);
        } else if (c == '"') {
          in_string_ = false;
          capture_key_ = false;
          if (capture_info_) note(c);
        } else if (capturing) {
          note(c);
        }
        ++p;
        continue;
      }

      if (in_fast_) {
        p = skip_event(p, end);
        continue;
      }

      const bool was_capturing_info = capture_info_;
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
          if (in_events_ && stack_.size() == events_depth_) {
            // An event: only its brackets and strings matter until it closes.
            in_fast_ = true;
            nested_ = 1;
            ++p;
            break;
          }
          check_start(c);
          stack_.push_back(c);
          ++p;
          if (!events_seen_ && c == '[' &&
              ((stack_.size() == 1) || (stack_.size() == 2 && stack_[0] == '{' && key_ == "traceEvents"))) {
            in_events_ = true;
            events_seen_ = true;
            events_depth_ = stack_.size();
            buf_.clear();
            reserve_chunk();
            buf_.push_back('[');
            seg = p;
          } else if (stack_.size() == 2 && stack_[0] == '{' && c == '{' && key_ == "distributedInfo") {
            capture_info_ = true;
            info_.clear();
          } else if (stack_.size() == 1 && c == '{') {
            expect_key_ = true;
          }
          break;
        }
        case '}':
        case ']': {
          if (stack_.empty()) throw std::invalid_argument("unbalanced brackets in trace JSON");
          if (in_events_ && stack_.size() == events_depth_) {
            flush(p);
            in_events_ = false;
            buf_.push_back(']');
            emit_(std::move(buf_));
            buf_.clear();
          }
          stack_.pop_back();
          if (capture_info_ && stack_.size() == 1) capture_info_ = false;
          ++p;
          if (stack_.empty()) done_ = true;
          break;
        }
        case ',':
          if (stack_.size() == 1 && stack_[0] == '{') {
            expect_key_ = true;
          } else if (in_events_ && stack_.size() == events_depth_) {
            const std::size_t pending = buf_.size() + (seg ? static_cast<std::size_t>(p - seg) : 0);
            if (pending >= chunk_bytes_) {
              flush(p);
              buf_.push_back(']');
              emit_(std::move(buf_));
              buf_.clear();
              reserve_chunk();
              buf_.push_back('[');
              ++p;
              seg = p;  // the comma itself is dropped
              break;
            }
          }
          ++p;
          break;
        case ':':
          if (stack_.size() == 1 && stack_[0] == '{') {
            expect_key_ = false;
          }
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
          ++p;
          break;
      }
      if (was_capturing_info || capture_info_) info_push(c);
    }
    if (seg) buf_.append(seg, static_cast<std::size_t>(end - seg));
  }

  void finish() {
    if (in_string_ || in_fast_ || !stack_.empty() || !started_) throw std::invalid_argument("trace JSON ends early");
  }
  const std::string& info() const { return info_; }

 private:
  void reserve_chunk() {
    // A chunk size far above the file size must not reserve that much memory.
    buf_.reserve(std::min<std::size_t>(chunk_bytes_ + chunk_bytes_ / 8, std::size_t{1} << 27) +
                 simdjson::SIMDJSON_PADDING);
  }

  // Skips the rest of one event (a bracket was just opened; nested_ counts the
  // open ones). Strings are jumped over with memchr, honouring backslash
  // escapes. Stops after the closing bracket, or at the end of the block with
  // in_string_ / escape_ set when a string runs on into the next block.
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
          if (slashes % 2 == 0) break;  // an escaped quote does not end the string
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

  void check_start(char c) {
    if (stack_.empty()) {
      if (done_) throw std::invalid_argument("extra data after the trace JSON");
      if (!started_) {
        if (c != '{' && c != '[') throw std::invalid_argument("trace JSON must be an object or an array");
        started_ = true;
      }
    }
  }
  void info_push(char c) {
    if (info_.size() < kInfoLimit) info_.push_back(c);
  }
  void note(char c) {
    if (capture_key_ && key_.size() < 256) key_.push_back(c);
    if (capture_info_ && info_.size() < kInfoLimit) info_.push_back(c);
  }

  const std::size_t chunk_bytes_;
  std::function<void(std::string&&)> emit_;
  std::string buf_;
  std::vector<char> stack_;
  bool in_string_ = false, escape_ = false;
  bool started_ = false, done_ = false;
  bool expect_key_ = false, capture_key_ = false;
  bool in_events_ = false, events_seen_ = false;
  bool in_fast_ = false;  // inside an event, tracked by nested_ instead of stack_
  int nested_ = 0;
  std::size_t events_depth_ = 0;
  bool capture_info_ = false;
  std::string key_, info_;
};

// ---------------------------------------------------------------------------
class Source {
 public:
  virtual ~Source() = default;
  virtual std::size_t read(char* buf, std::size_t cap) = 0;
};

class PlainSource : public Source {
 public:
  explicit PlainSource(const std::string& path) : f_(std::fopen(path.c_str(), "rb")) {
    if (!f_) throw std::runtime_error("cannot open " + path);
  }
  ~PlainSource() override { std::fclose(f_); }
  std::size_t read(char* buf, std::size_t cap) override {
    const std::size_t n = std::fread(buf, 1, cap, f_);
    if (n < cap && std::ferror(f_)) throw std::runtime_error("read failed");
    return n;
  }

 private:
  std::FILE* f_;
};

class GzSource : public Source {
 public:
  explicit GzSource(const std::string& path) : gz_(gzopen(path.c_str(), "rb")) {
    if (!gz_) throw std::runtime_error("cannot open " + path);
    gzbuffer(gz_, 1u << 20);
  }
  ~GzSource() override { gzclose(gz_); }
  std::size_t read(char* buf, std::size_t cap) override {
    const int n = gzread(gz_, buf, static_cast<unsigned>(std::min<std::size_t>(cap, 1u << 30)));
    if (n < 0) {
      int code = 0;
      throw std::runtime_error(std::string("gzip read failed: ") + gzerror(gz_, &code));
    }
    return static_cast<std::size_t>(n);
  }

 private:
  gzFile gz_;
};

std::optional<int64_t> rank_of(const std::string& info) {
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

bool ends_with_gz(const std::string& path) {
  return path.size() >= 3 && (path.compare(path.size() - 3, 3, ".gz") == 0 || path.compare(path.size() - 3, 3, ".GZ") == 0);
}

}  // namespace

KinetoColumns read_kineto_columns(const std::string& path, int threads, std::uint64_t chunk_bytes,
                                  std::uint64_t block_bytes) {
  if (threads < 0) throw std::invalid_argument("threads must be >= 0");
  if (block_bytes < 1) throw std::invalid_argument("block_bytes must be at least 1");
  if (chunk_bytes < 64) throw std::invalid_argument("chunk_bytes must be at least 64");
  if (threads == 0) threads = usable_cpus();

  std::unique_ptr<Source> source;
  if (ends_with_gz(path)) {
    source = std::make_unique<GzSource>(path);
  } else {
    source = std::make_unique<PlainSource>(path);
  }

  using Clock = std::chrono::steady_clock;
  auto seconds = [](Clock::duration d) { return std::chrono::duration<double>(d).count(); };
  double read_s = 0, split_s = 0, wait_s = 0;

  Pipeline pipeline(threads);
  Splitter splitter(static_cast<std::size_t>(chunk_bytes), [&](std::string&& chunk) {
    const auto t0 = Clock::now();
    pipeline.submit(std::move(chunk));
    wait_s += seconds(Clock::now() - t0);
  });
  std::vector<char> block(static_cast<std::size_t>(std::min<std::uint64_t>(block_bytes, std::uint64_t{1} << 30)));
  while (true) {
    const auto t0 = Clock::now();
    const std::size_t n = source->read(block.data(), block.size());
    const auto t1 = Clock::now();
    read_s += seconds(t1 - t0);
    if (n == 0) break;
    splitter.feed(block.data(), n);
    split_s += seconds(Clock::now() - t1);
  }
  splitter.finish();
  const auto t_finish = Clock::now();
  KinetoColumns columns = pipeline.finish();
  columns.finish_seconds = seconds(Clock::now() - t_finish);
  columns.read_seconds = read_s;
  columns.submit_wait_seconds = wait_s;
  columns.split_seconds = split_s - wait_s;
  columns.rank = rank_of(splitter.info());
  return columns;
}

}  // namespace cta
