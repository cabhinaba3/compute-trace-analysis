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

  // Parses "[" + events separated by commas + "]". `text` gains the padding
  // the JSON parser reads past its end. Throws std::invalid_argument for
  // malformed JSON or non-numeric ts and dur.
  void parse(std::string& text, ChunkOut& out);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// The integer "rank" in the text of a distributedInfo object, if there is one.
std::optional<std::int64_t> parse_rank(const std::string& distributed_info);

}  // namespace cta::kineto_reader
