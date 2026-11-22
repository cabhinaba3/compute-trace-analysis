// Helpers shared by the JAX Chrome-trace JSON, XPlane protobuf, and Perfetto
// protobuf readers.
#pragma once

#include <zlib.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "readers/jax/reader.hpp"
#include "util/interner.hpp"
#include "util/launch_map.hpp"

namespace cta::readers::jax {

using std::int32_t;
using std::int64_t;
using std::uint64_t;

inline constexpr std::string_view kExecutePrefix = "CommonPjRtLoadedExecutable::Execute (";
inline constexpr int64_t kStreamMod = int64_t{1} << 31;

// ---------------------------------------------------------------------------
// Plain or gzip byte stream (self-contained within readers/jax).

class Source {
 public:
  virtual ~Source() = default;
  virtual std::size_t read(char* buf, std::size_t cap) = 0;
};

namespace detail {

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
    int code = Z_OK;
    const char* msg = gzerror(gz_, &code);
    if (code != Z_OK && code != Z_STREAM_END)
      throw std::runtime_error(std::string("gzip read failed: ") + (msg ? msg : "truncated file"));
    return static_cast<std::size_t>(n);
  }

 private:
  gzFile gz_;
};

inline bool has_gz_suffix(const std::string& path) {
  return path.ends_with(".gz") || path.ends_with(".GZ");
}

inline bool has_gz_magic(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path);
  unsigned char hdr[2] = {0, 0};
  const std::size_t n = std::fread(hdr, 1, 2, f);
  std::fclose(f);
  return n == 2 && hdr[0] == 0x1f && hdr[1] == 0x8b;
}

}  // namespace detail

inline std::unique_ptr<Source> open_source(const std::string& path) {
  if (detail::has_gz_suffix(path) || detail::has_gz_magic(path)) {
    return std::make_unique<detail::GzSource>(path);
  }
  return std::make_unique<detail::PlainSource>(path);
}

// ---------------------------------------------------------------------------
// Buffered reader for streaming top-level protobuf length-delimited messages
// (field `target_field`, wire type 2) without hitting libprotobuf's 2 GiB
// single-CodedInputStream limit.

class ProtoFrameReader {
 public:
  explicit ProtoFrameReader(std::unique_ptr<Source> source, std::size_t buf_cap = 1u << 20)
      : source_(std::move(source)), buf_(std::max<std::size_t>(buf_cap, 4096)) {}

  double read_seconds() const { return read_s_; }

  // Reads the next length-delimited payload of `target_field` into `out`.
  // Unknown top-level fields with valid wire types are skipped.
  // Returns false at clean EOF; throws std::invalid_argument on malformed wire framing.
  bool next_message(std::uint32_t target_field, std::string& out) {
    while (true) {
      std::uint64_t tag = 0;
      if (!read_varint(tag, /*allow_eof=*/true)) return false;
      const std::uint32_t field = static_cast<std::uint32_t>(tag >> 3);
      const std::uint32_t wire = static_cast<std::uint32_t>(tag & 7u);
      if (field == 0) throw std::invalid_argument("invalid protobuf tag (field 0)");
      if (field == target_field && wire == 2u) {
        std::uint64_t len = 0;
        if (!read_varint(len, /*allow_eof=*/false))
          throw std::invalid_argument("truncated protobuf message length");
        if (len > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
          throw std::invalid_argument("protobuf submessage exceeds 2 GiB");
        out.resize(static_cast<std::size_t>(len));
        read_exact(out.data(), static_cast<std::size_t>(len));
        return true;
      }
      skip_field(wire);
    }
  }

 private:
  bool refill() {
    pos_ = 0;
    const auto t0 = std::chrono::steady_clock::now();
    end_ = source_->read(buf_.data(), buf_.size());
    read_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return end_ > 0;
  }

  bool read_byte(unsigned char& b) {
    if (pos_ == end_ && !refill()) return false;
    b = static_cast<unsigned char>(buf_[pos_++]);
    return true;
  }

  bool read_varint(std::uint64_t& value, bool allow_eof) {
    value = 0;
    for (int shift = 0; shift < 64; shift += 7) {
      unsigned char b = 0;
      if (!read_byte(b)) {
        if (shift == 0 && allow_eof) return false;
        throw std::invalid_argument("truncated protobuf varint");
      }
      if (shift == 63 && (b & 0xfe) != 0) throw std::invalid_argument("protobuf varint overflow");
      value |= static_cast<std::uint64_t>(b & 0x7fu) << shift;
      if ((b & 0x80u) == 0) return true;
    }
    throw std::invalid_argument("malformed protobuf varint");
  }

  void read_exact(char* dst, std::size_t n) {
    std::size_t done = 0;
    while (done < n) {
      if (pos_ == end_ && !refill()) throw std::invalid_argument("truncated protobuf payload");
      const std::size_t take = std::min(n - done, end_ - pos_);
      std::memcpy(dst + done, buf_.data() + pos_, take);
      pos_ += take;
      done += take;
    }
  }

  void skip_exact(std::uint64_t n) {
    while (n > 0) {
      if (pos_ == end_ && !refill()) throw std::invalid_argument("truncated protobuf field");
      const std::size_t avail = end_ - pos_;
      const std::size_t take = static_cast<std::size_t>(std::min<std::uint64_t>(n, avail));
      pos_ += take;
      n -= take;
    }
  }

  void skip_field(std::uint32_t wire) {
    switch (wire) {
      case 0u: {
        std::uint64_t dummy = 0;
        read_varint(dummy, /*allow_eof=*/false);
        return;
      }
      case 1u:
        skip_exact(8);
        return;
      case 2u: {
        std::uint64_t len = 0;
        read_varint(len, /*allow_eof=*/false);
        skip_exact(len);
        return;
      }
      case 5u:
        skip_exact(4);
        return;
      default:
        throw std::invalid_argument("unsupported protobuf wire type");
    }
  }

  std::unique_ptr<Source> source_;
  std::vector<char> buf_;
  std::size_t pos_ = 0;
  std::size_t end_ = 0;
  double read_s_ = 0.0;
};

// ---------------------------------------------------------------------------
// Time, integer, regex-free classification, and scope helpers.

// Python's int(round(us * 1000.0)): round half to even.
inline int64_t us_to_ns(double us) { return static_cast<int64_t>(std::nearbyint(us * 1000.0)); }

// Exact 64-bit picoseconds to nanoseconds, round half to even.
inline int64_t ps_to_ns(int64_t ps) {
  int64_t q = ps / 1000;
  const int64_t r = ps % 1000;
  if (r > 500 || (r == 500 && (q & 1))) {
    ++q;
  } else if (r < -500 || (r == -500 && (q & 1))) {
    --q;
  }
  return q;
}

// Python's `int(tid) % 2**31` (non-negative Euclidean modulo).
inline int64_t euclidean_mod_stream(int64_t tid) {
  int64_t m = tid % kStreamMod;
  if (m < 0) m += kStreamMod;
  return m;
}

inline char ascii_lower(char c) {
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : c;
}

inline bool is_ascii_alpha(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

inline bool is_ascii_digit(char c) { return c >= '0' && c <= '9'; }

// Matches Python `str(v).isdigit()` + `int(v)` for non-negative ASCII integers.
inline bool parse_nonneg_digits(std::string_view s, int64_t& out) {
  if (s.empty()) return false;
  uint64_t v = 0;
  constexpr uint64_t kMax = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  for (char c : s) {
    if (!is_ascii_digit(c)) return false;
    const uint64_t d = static_cast<uint64_t>(c - '0');
    if (v > (kMax - d) / 10) return false;
    v = v * 10 + d;
  }
  out = static_cast<int64_t>(v);
  return true;
}

// Matches Python `str(v).lstrip("-").isdigit()` + `int(v)` for signed ASCII integers.
inline bool parse_signed_digits(std::string_view s, int64_t& out) {
  if (s.empty()) return false;
  bool neg = false;
  std::size_t i = 0;
  if (s[0] == '-') {
    neg = true;
    i = 1;
  }
  if (i >= s.size()) return false;
  uint64_t v = 0;
  const uint64_t limit = neg ? (static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1u)
                             : static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  for (; i < s.size(); ++i) {
    if (!is_ascii_digit(s[i])) return false;
    const uint64_t d = static_cast<uint64_t>(s[i] - '0');
    if (v > (limit - d) / 10) return false;
    v = v * 10 + d;
  }
  if (neg) {
    out = (v == limit) ? std::numeric_limits<int64_t>::min() : -static_cast<int64_t>(v);
  } else {
    out = static_cast<int64_t>(v);
  }
  return true;
}

// Exact equivalent of Python's `re.compile(r"/device:(?:[A-Za-z]+):(\d+)")`.
inline bool parse_device_ordinal(std::string_view process_name, int64_t& ordinal) {
  static constexpr std::string_view kPrefix = "/device:";
  std::size_t pos = 0;
  while ((pos = process_name.find(kPrefix, pos)) != std::string_view::npos) {
    std::size_t i = pos + kPrefix.size();
    const std::size_t kind_start = i;
    while (i < process_name.size() && is_ascii_alpha(process_name[i])) ++i;
    if (i > kind_start && i < process_name.size() && process_name[i] == ':') {
      ++i;
      const std::size_t dig_start = i;
      while (i < process_name.size() && is_ascii_digit(process_name[i])) ++i;
      if (i > dig_start && parse_nonneg_digits(process_name.substr(dig_start, i - dig_start), ordinal)) {
        return true;
      }
    }
    ++pos;
  }
  return false;
}

// Device line prefix selection: prefer raw "Stream #", else "XLA Ops", else "".
inline std::string_view select_device_line_prefix(const std::vector<std::string_view>& line_names) {
  bool has_xla_ops = false;
  for (std::string_view name : line_names) {
    if (name.starts_with("Stream #")) return "Stream #";
    if (name == "XLA Ops") has_xla_ops = true;
  }
  return has_xla_ops ? std::string_view("XLA Ops") : std::string_view("");
}

inline bool match_ci_at(std::string_view s, std::size_t pos, std::string_view lower_lit) {
  if (pos + lower_lit.size() > s.size()) return false;
  for (std::size_t i = 0; i < lower_lit.size(); ++i) {
    if (ascii_lower(s[pos + i]) != lower_lit[i]) return false;
  }
  return true;
}

inline std::size_t skip_sep(std::string_view s, std::size_t pos) {
  if (pos < s.size() && (s[pos] == '-' || s[pos] == '_')) return pos + 1;
  return pos;
}

// Exact equivalent of:
// r"all[-_]?reduce|all[-_]?gather|reduce[-_]?scatter|all[-_]?to[-_]?all"
// r"|collective[-_]?permute|nccl|^(send|recv)[-.\d]" (case-insensitive).
inline bool is_comm_text(std::string_view s) {
  if (s.size() >= 5 && (match_ci_at(s, 0, "send") || match_ci_at(s, 0, "recv"))) {
    const char next = s[4];
    if (next == '-' || next == '.' || is_ascii_digit(next)) return true;
  }
  for (std::size_t i = 0; i < s.size(); ++i) {
    const char c = ascii_lower(s[i]);
    if (c == 'n' && match_ci_at(s, i, "nccl")) return true;
    if (c == 'a' && match_ci_at(s, i, "all")) {
      const std::size_t j = skip_sep(s, i + 3);
      if (match_ci_at(s, j, "reduce") || match_ci_at(s, j, "gather")) return true;
      if (match_ci_at(s, j, "to")) {
        const std::size_t k = skip_sep(s, j + 2);
        if (match_ci_at(s, k, "all")) return true;
      }
    } else if (c == 'r' && match_ci_at(s, i, "reduce")) {
      const std::size_t j = skip_sep(s, i + 6);
      if (match_ci_at(s, j, "scatter")) return true;
    } else if (c == 'c' && match_ci_at(s, i, "collective")) {
      const std::size_t j = skip_sep(s, i + 10);
      if (match_ci_at(s, j, "permute")) return true;
    }
  }
  return false;
}

// Exact equivalent of:
// r"memcpy|memset|copy[-_]?(start|done)|infeed|outfeed" (case-insensitive).
inline bool is_copy_text(std::string_view s) {
  for (std::size_t i = 0; i < s.size(); ++i) {
    const char c = ascii_lower(s[i]);
    if (c == 'm') {
      if (match_ci_at(s, i, "memcpy") || match_ci_at(s, i, "memset")) return true;
    } else if (c == 'c' && match_ci_at(s, i, "copy")) {
      const std::size_t j = skip_sep(s, i + 4);
      if (match_ci_at(s, j, "start") || match_ci_at(s, j, "done")) return true;
    } else if (c == 'i' && match_ci_at(s, i, "infeed")) {
      return true;
    } else if (c == 'o' && match_ci_at(s, i, "outfeed")) {
      return true;
    }
  }
  return false;
}

// 0 = compute, 1 = comm, 2 = memcpy.
inline std::int8_t classify_op(std::string_view name, std::string_view hlo_op_raw) {
  const std::string_view hlo_op = hlo_op_raw.empty() ? name : hlo_op_raw;
  if (is_comm_text(name) || is_comm_text(hlo_op)) return 1;
  if (is_copy_text(name)) return 2;
  return 0;
}

// Exact equivalent of `re.compile(r"(?:size|num_bytes):(\d+)")`.
inline int64_t parse_transfer_bytes(std::string_view details) {
  for (std::size_t i = 0; i < details.size(); ++i) {
    std::size_t digits = std::string_view::npos;
    if (details[i] == 's' && details.substr(i).starts_with("size:")) {
      digits = i + 5;
    } else if (details[i] == 'n' && details.substr(i).starts_with("num_bytes:")) {
      digits = i + 10;
    }
    if (digits != std::string_view::npos && digits < details.size() && is_ascii_digit(details[digits])) {
      std::size_t end = digits;
      while (end < details.size() && is_ascii_digit(details[end])) ++end;
      int64_t value = 0;
      if (parse_nonneg_digits(details.substr(digits, end - digits), value)) return value;
    }
  }
  return 0;
}

// Exact equivalent of Python's `_scope_path(args, module)`.
inline void compute_scope(std::string_view arg_name, std::string_view tf_op, std::string_view module,
                          std::string& scope, std::string& scope_path) {
  scope.clear();
  scope_path.clear();
  for (std::string_view v : {arg_name, tf_op}) {
    if (v.find('/') != std::string_view::npos || v.starts_with("jit(")) {
      std::size_t pos = 0;
      while (pos < v.size()) {
        const std::size_t slash = v.find('/', pos);
        const std::size_t seg_end = (slash == std::string_view::npos) ? v.size() : slash;
        if (seg_end > pos) {
          const std::string_view seg = v.substr(pos, seg_end - pos);
          if (!scope_path.empty()) scope_path.append(" > ");
          scope_path.append(seg);
          scope.assign(seg);
        }
        if (slash == std::string_view::npos) break;
        pos = slash + 1;
      }
      return;
    }
  }
  if (module.starts_with("jit_") || module.starts_with("pjit_") || module.starts_with("pmap_")) {
    const std::size_t us = module.find('_');
    scope.assign(module.substr(0, us));
    scope.push_back('(');
    scope.append(module.substr(us + 1));
    scope.push_back(')');
    scope_path = scope;
    return;
  }
  if (!module.empty()) {
    scope.assign(module);
    scope_path.assign(module);
  }
}

// ---------------------------------------------------------------------------
// Unified column builder shared by JSON, XPlane, and Perfetto readers.

class JaxEventBuilder {
 public:
  explicit JaxEventBuilder(std::string format) { out_.format = std::move(format); }

  void add_step(int64_t step_num, int64_t start_ns, int64_t end_ns) {
    out_.step_id.push_back(step_num);
    out_.step_start.push_back(start_ns);
    out_.step_end.push_back(end_ns);
  }

  void add_execution(std::string_view module, int64_t start_ns, int64_t end_ns) {
    out_.exec_module.push_back(names_.intern(module));
    out_.exec_start.push_back(start_ns);
    out_.exec_end.push_back(end_ns);
  }

  void add_api_call(std::string_view name, int64_t start_ns, int64_t end_ns, int64_t thread_id,
                    int64_t correlation_id) {
    // First host launch for a correlation id wins (matches Python dict.setdefault).
    launches_.put_if_absent(correlation_id, start_ns, thread_id);
    out_.rcall_name.push_back(names_.intern(name));
    out_.rcall_start.push_back(start_ns);
    out_.rcall_end.push_back(end_ns);
    out_.rcall_thread.push_back(thread_id);
    out_.rcall_correlation.push_back(correlation_id);
  }

  void add_op(std::string_view name, std::string_view arg_name, std::string_view tf_op,
              std::string_view hlo_op_raw, std::string_view module, std::string_view transfer_details,
              int64_t start_ns, int64_t end_ns, int64_t stream, int64_t device, int64_t correlation_id) {
    compute_scope(arg_name, tf_op, module, scope_scratch_, scope_path_scratch_);
    out_.op_name.push_back(names_.intern(name));
    out_.op_scope.push_back(names_.intern(scope_scratch_));
    out_.op_scope_path.push_back(names_.intern(scope_path_scratch_));
    out_.op_hlo_op.push_back(names_.intern(hlo_op_raw));
    out_.op_hlo_module.push_back(names_.intern(module));
    out_.op_category.push_back(classify_op(name, hlo_op_raw));
    out_.op_start.push_back(start_ns);
    out_.op_end.push_back(end_ns);
    out_.op_bytes.push_back(parse_transfer_bytes(transfer_details));
    out_.op_stream.push_back(stream);
    out_.op_device.push_back(device);
    out_.op_correlation.push_back(correlation_id);
  }

  JaxColumns finish() {
    const std::size_t n = out_.op_correlation.size();
    out_.op_launch_ns.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      const int64_t corr = out_.op_correlation[i];
      LaunchMap::Hit hit{};
      if (corr >= 0 && launches_.find(corr, hit)) {
        out_.op_launch_ns[i] = static_cast<double>(hit.start);
      } else {
        out_.op_launch_ns[i] = std::numeric_limits<double>::quiet_NaN();
      }
    }
    out_.names.assign(names_.items().begin(), names_.items().end());
    return std::move(out_);
  }

 private:
  Interner names_;
  LaunchMap launches_;
  JaxColumns out_;
  std::string scope_scratch_;
  std::string scope_path_scratch_;
};

}  // namespace cta::readers::jax
