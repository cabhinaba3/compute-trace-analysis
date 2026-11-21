#include "readers/kineto/splitter.hpp"

#include <algorithm>
#include <cstring>

#include "readers/kineto/chunk.hpp"

namespace cta::readers::kineto {
namespace {

constexpr std::size_t kInfoLimit = 4u << 20;  // largest distributedInfo kept

}  // namespace

void Splitter::reserve_chunk() {
  // A chunk size far above the file size must not reserve that much memory.
  buf_.reserve(std::min<std::size_t>(chunk_bytes_ + chunk_bytes_ / 8, std::size_t{1} << 27) +
               kChunkPadding);
}

void Splitter::check_start(char c) {
  if (stack_.empty()) {
    if (done_) throw std::invalid_argument("extra data after the trace JSON");
    if (!started_) {
      if (c != '{' && c != '[') throw std::invalid_argument("trace JSON must be an object or an array");
      started_ = true;
    }
  }
}

void Splitter::event_value() {
  if (slot_ == Slot::kAfterValue) throw std::invalid_argument("missing comma between trace events");
  slot_ = Slot::kAfterValue;
}

void Splitter::info_push(char c) {
  if (info_.size() < kInfoLimit) info_.push_back(c);
}

void Splitter::note(char c) {
  if (capture_key_ && key_.size() < 256) key_.push_back(c);
  if (capture_info_ && info_.size() < kInfoLimit) info_.push_back(c);
}

// Skips the rest of one event (a bracket was just opened; nested_ counts the
// open ones). Strings are jumped over with memchr, honouring backslash
// escapes. Stops after the closing bracket, or at the end of the block with
// in_string_ / escape_ set when a string runs on into the next block.
const char* Splitter::skip_event(const char* p, const char* end) {
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

void Splitter::feed(const char* base, std::size_t n) {
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
    if (c == '"' || c == '{' || c == '[' || c == '}' || c == ']' || c == ',' || c == ':' || c == ' ' || c == '\t' ||
        c == '\n' || c == '\r')
      in_scalar_ = false;
    switch (c) {
      case '"':
        if (stack_.empty()) check_start(c);
        if (in_events_ && stack_.size() == events_depth_) event_value();
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
          event_value();
          in_fast_ = true;
          nested_ = 1;
          ++p;
          break;
        }
        check_start(c);
        if (events_seen_ && c == '[' && stack_.size() == 1 && stack_[0] == '{' && key_ == "traceEvents")
          throw std::invalid_argument("trace JSON has more than one traceEvents array");
        stack_.push_back(c);
        ++p;
        if (!events_seen_ && c == '[' &&
            ((stack_.size() == 1) || (stack_.size() == 2 && stack_[0] == '{' && key_ == "traceEvents"))) {
          in_events_ = true;
          events_seen_ = true;
          events_depth_ = stack_.size();
          slot_ = Slot::kStart;
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
          if (slot_ == Slot::kAfterComma) throw std::invalid_argument("trailing comma in the trace events");
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
        if (stack_.empty()) check_start(c);
        if (stack_.size() == 1 && stack_[0] == '{') {
          expect_key_ = true;
        } else if (in_events_ && stack_.size() == events_depth_) {
          if (slot_ != Slot::kAfterValue) throw std::invalid_argument("unexpected comma in the trace events");
          slot_ = Slot::kAfterComma;
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
        if (stack_.empty()) check_start(c);
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
        if (in_events_ && stack_.size() == events_depth_ && !in_scalar_) {
          event_value();
          in_scalar_ = true;
        }
        ++p;
        break;
    }
    if (was_capturing_info || capture_info_) info_push(c);
  }
  if (seg) buf_.append(seg, static_cast<std::size_t>(end - seg));
}

}  // namespace cta::readers::kineto
