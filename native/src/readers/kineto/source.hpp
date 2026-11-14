#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace cta::kineto_reader {

// A byte stream: a plain file, or a gzip file decompressed as it is read.
class Source {
 public:
  virtual ~Source() = default;

  // Reads up to `cap` bytes; returns 0 at the end. Throws std::runtime_error.
  virtual std::size_t read(char* buf, std::size_t cap) = 0;
};

// Opens `path`; a ".gz" suffix means gzip.
std::unique_ptr<Source> open_source(const std::string& path);

}  // namespace cta::kineto_reader
