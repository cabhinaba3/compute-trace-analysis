// Shared helpers for the synthetic-trace generators: a buffered (optionally
// gzip-compressed) output sink, number formatting that matches Python's
// json.dumps, JSON string escaping, and a deterministic RNG.
#pragma once

#include <zlib.h>

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cta {

// Buffered writer to a plain file or, for a ".gz" path, a gzip stream. Never
// holds more than one buffer of output, so file size is not limited by RAM.
class Sink {
 public:
  explicit Sink(const std::string& path) {
    gz_ = path.size() >= 3 && path.compare(path.size() - 3, 3, ".gz") == 0;
    if (gz_) {
      gzf_ = gzopen(path.c_str(), "wb6");
      if (!gzf_) throw std::runtime_error("cannot open " + path);
    } else {
      f_ = std::fopen(path.c_str(), "wb");
      if (!f_) throw std::runtime_error("cannot open " + path);
    }
    buf_.reserve(kBuf);
  }
  Sink(const Sink&) = delete;
  Sink& operator=(const Sink&) = delete;
  ~Sink() {
    try {
      close();
    } catch (...) {
    }
  }

  void put(std::string_view s) {
    bytes_ += s.size();
    if (buf_.size() + s.size() > kBuf) flush();
    if (s.size() >= kBuf) {
      raw(s.data(), s.size());
    } else {
      buf_.append(s);
    }
  }
  void put(char c) { put(std::string_view(&c, 1)); }

  // Uncompressed bytes written so far.
  std::uint64_t bytes() const { return bytes_; }

  void close() {
    if (!f_ && !gzf_) return;
    flush();
    int rc = 0;
    if (gzf_) {
      rc = gzclose(gzf_);
      gzf_ = nullptr;
    } else {
      rc = std::fclose(f_);
      f_ = nullptr;
    }
    if (rc != 0 && rc != Z_OK) throw std::runtime_error("close failed");
  }

 private:
  static constexpr std::size_t kBuf = 1 << 20;
  void flush() {
    if (!buf_.empty()) raw(buf_.data(), buf_.size());
    buf_.clear();
  }
  void raw(const char* p, std::size_t n) {
    if (gz_) {
      if (gzwrite(gzf_, p, static_cast<unsigned>(n)) != static_cast<int>(n))
        throw std::runtime_error("gzwrite failed");
    } else if (std::fwrite(p, 1, n, f_) != n) {
      throw std::runtime_error("fwrite failed");
    }
  }
  bool gz_ = false;
  std::FILE* f_ = nullptr;
  gzFile gzf_ = nullptr;
  std::string buf_;
  std::uint64_t bytes_ = 0;
};

// Appends a double exactly as Python's repr does: the shortest round-trip
// digits, in fixed notation for decimal exponents -4..15 (with ".0" added to
// whole numbers) and in scientific notation ("1e+16", "1.5e-05") outside it.
// std::to_chars alone picks whichever form is shorter, which differs from repr.
inline void append_double(std::string& out, double v) {
  if (v == 0.0) {
    out.append(std::signbit(v) ? "-0.0" : "0.0");
    return;
  }
  char tmp[48];
  auto r = std::to_chars(tmp, tmp + sizeof tmp, v, std::chars_format::scientific);
  std::string_view sci(tmp, static_cast<std::size_t>(r.ptr - tmp));  // "-d.ddde+XX"
  if (sci.front() == '-') {
    out.push_back('-');
    sci.remove_prefix(1);
  }
  const std::size_t epos = sci.find('e');
  std::string digits;  // mantissa digits without the point
  for (char c : sci.substr(0, epos))
    if (c != '.') digits.push_back(c);
  const int exp10 = std::atoi(std::string(sci.substr(epos + 1)).c_str());
  const int n = static_cast<int>(digits.size());
  if (exp10 >= -4 && exp10 < 16) {
    if (exp10 >= 0) {
      if (n <= exp10 + 1) {
        out.append(digits).append(static_cast<std::size_t>(exp10 + 1 - n), '0').append(".0");
      } else {
        out.append(digits, 0, static_cast<std::size_t>(exp10 + 1)).push_back('.');
        out.append(digits, static_cast<std::size_t>(exp10 + 1), std::string::npos);
      }
    } else {
      out.append("0.").append(static_cast<std::size_t>(-exp10 - 1), '0').append(digits);
    }
  } else {
    out.push_back(digits[0]);
    if (n > 1) out.push_back('.'), out.append(digits, 1, std::string::npos);
    out.push_back('e');
    out.push_back(exp10 < 0 ? '-' : '+');
    const int ae = exp10 < 0 ? -exp10 : exp10;
    if (ae < 10) out.push_back('0');
    out.append(std::to_string(ae));
  }
}

inline void append_int(std::string& out, std::int64_t v) {
  char tmp[24];
  auto r = std::to_chars(tmp, tmp + sizeof tmp, v);
  out.append(tmp, static_cast<std::size_t>(r.ptr - tmp));
}

// Appends s as a JSON string literal (with quotes), escaping like json.dumps
// with ensure_ascii=False for ASCII input.
inline void append_json_string(std::string& out, std::string_view s) {
  out.push_back('"');
  for (unsigned char c : s) {
    switch (c) {
      case '"': out.append("\\\""); break;
      case '\\': out.append("\\\\"); break;
      case '\n': out.append("\\n"); break;
      case '\r': out.append("\\r"); break;
      case '\t': out.append("\\t"); break;
      default:
        if (c < 0x20) {
          char u[8];
          std::snprintf(u, sizeof u, "\\u%04x", c);
          out.append(u);
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  out.push_back('"');
}

// xoshiro256** seeded through splitmix64. Range mapping is done by hand
// (std::*_distribution output differs between standard libraries), so a seed
// gives the same bytes on every platform.
class Rng {
 public:
  explicit Rng(std::uint64_t seed) {
    for (auto& w : s_) {
      seed += 0x9e3779b97f4a7c15ULL;
      std::uint64_t z = seed;
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
      w = z ^ (z >> 31);
    }
  }
  std::uint64_t next() {
    const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;
    const std::uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45);
    return result;
  }
  // Uniform integer in [lo, hi].
  std::int64_t range(std::int64_t lo, std::int64_t hi) {
    return lo + static_cast<std::int64_t>(next() % static_cast<std::uint64_t>(hi - lo + 1));
  }
  // Uniform double in [0, 1).
  double unit() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }

 private:
  static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  std::uint64_t s_[4];
};

}  // namespace cta
