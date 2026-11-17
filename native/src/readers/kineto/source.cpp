#include "readers/kineto/source.hpp"

#include <zlib.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace cta::readers::kineto {
namespace {

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

bool ends_with_gz(const std::string& path) {
  return path.ends_with(".gz") || path.ends_with(".GZ");
}

}  // namespace

std::unique_ptr<Source> open_source(const std::string& path) {
  if (ends_with_gz(path)) return std::make_unique<GzSource>(path);
  return std::make_unique<PlainSource>(path);
}

}  // namespace cta::readers::kineto
