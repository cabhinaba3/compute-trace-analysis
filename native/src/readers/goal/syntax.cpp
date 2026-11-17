#include "readers/goal/syntax.hpp"

#include <stdexcept>
#include <vector>

namespace cta::readers::goal {

std::string_view code_of_slow(std::string_view raw, bool& in_block, std::string& scratch) {
  std::string line(strip(raw));
  if (in_block) {
    const std::size_t close = line.find("*/");
    if (close == std::string::npos) return {};
    line = std::string(strip(std::string_view(line).substr(close + 2)));
    in_block = false;
  }
  const std::size_t open = line.find("/*");
  if (open != std::string::npos) {
    const std::string before = line.substr(0, open);
    const std::string after = line.substr(open + 2);
    const std::size_t close = after.find("*/");
    if (close != std::string::npos) {
      line = std::string(strip(before + " " + after.substr(close + 2)));
    } else {
      line = std::string(strip(before));
      in_block = true;
    }
  }
  for (const char* token : {"//", "#"}) {
    const std::size_t at = line.find(token);
    if (at != std::string::npos) line = std::string(strip(std::string_view(line).substr(0, at)));
  }
  scratch = std::move(line);
  return scratch;
}

namespace {

bool is_digit(char c) { return c >= '0' && c <= '9'; }
char lower(char c) { return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : c; }

// A cursor over one line, with the few matching steps the formats need. A
// failed step leaves the cursor where it was.
class Cursor {
 public:
  explicit Cursor(std::string_view s) : s_(s) {}
  bool at_end() const { return i_ == s_.size(); }
  std::size_t position() const { return i_; }
  void reset(std::size_t i) { i_ = i; }
  std::string_view rest() const { return s_.substr(i_); }

  void skip_space() {
    while (i_ < s_.size() && is_space(s_[i_])) ++i_;
  }
  // One or more spaces.
  bool space() {
    if (i_ >= s_.size() || !is_space(s_[i_])) return false;
    skip_space();
    return true;
  }
  // A word, in any letter case.
  bool word(std::string_view w) {
    if (s_.size() - i_ < w.size()) return false;
    for (std::size_t k = 0; k < w.size(); ++k)
      if (lower(s_[i_ + k]) != w[k]) return false;
    i_ += w.size();
    return true;
  }
  // One or more digits; throws if the number is over 2^64 - 1.
  bool number(std::uint64_t& value) {
    if (i_ >= s_.size() || !is_digit(s_[i_])) return false;
    value = 0;
    while (i_ < s_.size() && is_digit(s_[i_])) {
      const std::uint64_t digit = static_cast<std::uint64_t>(s_[i_] - '0');
      if (value > (UINT64_MAX - digit) / 10) throw std::invalid_argument("number does not fit in 64 bits in GOAL text");
      value = value * 10 + digit;
      ++i_;
    }
    return true;
  }
  // "<space> <word> <space> <number>", or nothing (the cursor stays).
  bool option(std::string_view w, std::uint64_t& value) {
    const std::size_t start = i_;
    if (space() && word(w) && space() && number(value)) return true;
    i_ = start;
    return false;
  }
  // "<space> <word> <space> <number>" must be present.
  bool required(std::string_view w, std::uint64_t& value) { return option(w, value); }

 private:
  std::string_view s_;
  std::size_t i_ = 0;
};

// The size multiplier of a unit suffix; "tb" is not known and counts as 1.
std::uint64_t unit_bytes(char prefix) {
  switch (lower(prefix)) {
    case 'k': return 1024;
    case 'm': return 1024ull * 1024;
    case 'g': return 1024ull * 1024 * 1024;
    default: return 1;
  }
}

}  // namespace

bool parse_rank_header(std::string_view code, std::uint64_t& rank) {
  Cursor c(code);
  std::uint64_t value = 0;
  if (code.compare(0, 4, "rank") != 0) return false;
  c.reset(4);
  if (!c.space() || !c.number(value)) return false;
  rank = value;
  return true;
}

bool parse_dependency(std::string_view code, std::string_view& dst, std::string_view& src, bool& irequires) {
  std::string_view words[3];
  std::size_t count = 0, i = 0;
  while (i < code.size()) {
    while (i < code.size() && is_space(code[i])) ++i;
    if (i >= code.size()) break;
    const std::size_t begin = i;
    while (i < code.size() && !is_space(code[i])) ++i;
    if (count == 3) return false;
    words[count++] = code.substr(begin, i - begin);
  }
  if (count != 3) return false;
  if (words[1] != "requires" && words[1] != "irequires") return false;
  dst = words[0];
  src = words[2];
  irequires = words[1] == "irequires";
  return true;
}

bool parse_op(std::string_view code, Op& op) {
  // An optional "label:" first: no colon or space in the label, at least one
  // character of it, and optional space after the colon.
  std::size_t at = 0;
  std::string_view label;
  while (at < code.size() && code[at] != ':' && !is_space(code[at])) ++at;
  Cursor c(code);
  if (at > 0 && at < code.size() && code[at] == ':') {
    label = code.substr(0, at);
    c.reset(at + 1);
    c.skip_space();
  }

  std::uint64_t tag = 0, ignored = 0;
  if (c.word("send") || c.word("recv")) {
    const bool is_send = lower(code[c.position() - 4]) == 's';
    std::uint64_t size = 0;
    if (!c.space() || !c.number(size)) return false;
    // An optional unit, "b" or one of k, m, g, t before it.
    std::uint64_t multiplier = 1;
    const std::string_view rest = c.rest();
    if (!rest.empty() && lower(rest[0]) == 'b') {
      c.reset(c.position() + 1);
    } else if (rest.size() >= 2 && std::string_view("kmgt").find(lower(rest[0])) != std::string_view::npos &&
               lower(rest[1]) == 'b') {
      multiplier = unit_bytes(rest[0]);
      c.reset(c.position() + 2);
    }
    std::uint64_t peer = 0;
    if (!c.space() || !c.word(is_send ? "to" : "from") || !c.space() || !c.number(peer)) return false;
    c.option("tag", tag);
    c.option("cpu", ignored);
    c.option("nic", ignored);
    if (!c.at_end()) return false;
    if (multiplier != 1 && size > UINT64_MAX / multiplier)
      throw std::invalid_argument("number does not fit in 64 bits in GOAL text");
    op = {is_send ? OpKind::kSend : OpKind::kRecv, label, size * multiplier, peer, tag};
    return true;
  }
  if (c.word("calc")) {
    std::uint64_t duration = 0;
    if (!c.space() || !c.number(duration)) return false;
    c.option("cpu", ignored);
    c.option("nic", ignored);
    if (!c.at_end()) return false;
    op = {OpKind::kCompute, label, duration, 0, 0};
    return true;
  }
  return false;
}

}  // namespace cta::readers::goal
