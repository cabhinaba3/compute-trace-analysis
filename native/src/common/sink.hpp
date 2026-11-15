// Buffered output to a plain file or a gzip stream.
#pragma once

#include <zlib.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cta {

// Buffered writer to a plain file or, for a ".gz" path, a gzip stream. Never
// holds more than one buffer of output, so file size is not limited by RAM.
class Sink {
 public:
  explicit Sink(const std::string& path) {
    gz_ = path.ends_with(".gz");
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

}  // namespace cta
