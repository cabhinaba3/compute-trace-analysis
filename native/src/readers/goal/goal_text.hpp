// The lines of a GOAL text schedule: how they end, how comments come off, and
// which of them are rank headers, dependencies or ops. The rank-boundary scan
// and the parser both use these, so they agree on every line.
//
// The rules are those of the Python reader this replaces (cta.adapters.goal),
// down to its quirks: see the comments on each function.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

namespace cta::goal_text {

// Python's \s and str.strip() for ASCII: space, \t \n \v \f \r, \x1c..\x1f.
inline bool is_space(char c) { return c == ' ' || (c >= '\t' && c <= '\r') || (c >= '\x1c' && c <= '\x1f'); }

inline std::string_view strip(std::string_view s) {
  while (!s.empty() && is_space(s.front())) s.remove_prefix(1);
  while (!s.empty() && is_space(s.back())) s.remove_suffix(1);
  return s;
}

// Reads the line at `pos` (no terminator) and moves `pos` past it. A line ends
// at "\n", "\r\n" or a lone "\r", or at the end of the text. Call while
// pos < size.
inline std::string_view next_line(const char* data, std::size_t size, std::size_t& pos) {
  const char* base = data + pos;
  const std::size_t remaining = size - pos;
  const char* nl = static_cast<const char*>(std::memchr(base, '\n', remaining));
  const std::size_t scan = nl ? static_cast<std::size_t>(nl - base) : remaining;
  if (const char* cr = static_cast<const char*>(std::memchr(base, '\r', scan))) {
    const std::size_t len = static_cast<std::size_t>(cr - base);
    pos += len + 1;
    if (pos < size && data[pos] == '\n') ++pos;
    return {base, len};
  }
  pos += nl ? scan + 1 : remaining;
  return {base, scan};
}

// The code on a line, with surrounding space and comments taken off. `in_block`
// says whether a "/*" comment is open before the line and is updated for the
// next one. The result is a view into `raw` or into `scratch`.
//
// Comments come off in one pass per line, as the Python reader did: a "/* */"
// pair first (only the first on the line), then everything after "//", then
// everything after "#".
std::string_view code_of_slow(std::string_view raw, bool& in_block, std::string& scratch);

inline std::string_view code_of(std::string_view raw, bool& in_block, std::string& scratch) {
  if (!in_block && !std::memchr(raw.data(), '/', raw.size()) && !std::memchr(raw.data(), '#', raw.size()))
    return strip(raw);
  return code_of_slow(raw, in_block, scratch);
}

// "rank N" followed by anything: a prefix match, case-sensitive.
bool parse_rank_header(std::string_view code, std::uint64_t& rank);

// "<dst> requires <src>" or "<dst> irequires <src>": three words, the middle
// one exactly "requires" or "irequires" (case-sensitive).
bool parse_dependency(std::string_view code, std::string_view& dst, std::string_view& src, bool& irequires);

enum class OpKind : std::uint8_t { kCompute = 0, kSend = 1, kRecv = 2 };

struct Op {
  OpKind kind = OpKind::kCompute;
  std::string_view label;  // empty when the line has none
  std::uint64_t value = 0;  // a calc's duration, or a send's or recv's bytes
  std::uint64_t peer = 0;
  std::uint64_t tag = 0;
};

// "[label:] send <size>[kb|mb|gb|tb|b] to <peer> [tag N] [cpu N] [nic N]", the
// same with "recv ... from", and "[label:] calc <ns> [cpu N] [nic N]", in any
// letter case and with the options in that order only. "tb" counts as bytes.
// Throws std::invalid_argument for a number that does not fit in 64 bits.
bool parse_op(std::string_view code, Op& op);

}  // namespace cta::goal_text
