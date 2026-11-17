#pragma once

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace cta::readers::kineto {

// Walks the JSON text block by block and cuts the trace-event array into
// chunks at the commas between events. It tracks string and escape state, so
// braces and quotes inside names never confuse it, and it keeps the value of
// the top-level "distributedInfo" key wherever that sits in the file. A bare
// array at the top level is read as the events.
//
// The result does not depend on how the text is cut into blocks.
class Splitter {
 public:
  // `emit` receives each chunk: "[" + events separated by commas + "]".
  Splitter(std::size_t chunk_bytes, std::function<void(std::string&&)> emit)
      : chunk_bytes_(chunk_bytes), emit_(std::move(emit)) {}

  // Feeds the next block of text. Throws std::invalid_argument for structure
  // that cannot be JSON.
  void feed(const char* base, std::size_t n);

  // Call after the last block; throws std::invalid_argument if the text ended
  // inside a value.
  void finish() {
    if (in_string_ || in_fast_ || !stack_.empty() || !started_) throw std::invalid_argument("trace JSON ends early");
  }

  // Text of the top-level "distributedInfo" object, or empty.
  const std::string& info() const { return info_; }

 private:
  void reserve_chunk();
  const char* skip_event(const char* p, const char* end);
  void check_start(char c);
  void info_push(char c);
  void note(char c);

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

}  // namespace cta::readers::kineto
