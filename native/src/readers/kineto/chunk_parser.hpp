// Parses one chunk of trace events into columns.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "readers/kineto/chunk.hpp"

namespace cta::kineto_reader {

// Owns the JSON parser, which is reused from chunk to chunk; one per thread.
class ChunkParser {
 public:
  ChunkParser();
  ~ChunkParser();
  ChunkParser(const ChunkParser&) = delete;
  ChunkParser& operator=(const ChunkParser&) = delete;

  // Throws std::invalid_argument for malformed JSON or non-numeric ts and dur.
  void parse(Chunk& chunk, ChunkOut& out);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// The integer "rank" in the text of a distributedInfo object, if there is one.
std::optional<std::int64_t> parse_rank(const std::string& distributed_info);

}  // namespace cta::kineto_reader
