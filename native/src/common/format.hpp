// Number and string formatting that matches Python's json.dumps.
#pragma once

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

namespace cta {

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

}  // namespace cta
