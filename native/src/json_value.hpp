// Minimal ordered JSON value and a writer whose output matches Python's
// json.dumps (indent=None: ", " and ": " separators; indent=N: one item per
// line). Only what the small synthetic generators need.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "common.hpp"

namespace cta {

struct Json;
using JsonArray = std::vector<Json>;
using JsonObject = std::vector<std::pair<std::string, Json>>;

struct Json {
  std::variant<bool, std::int64_t, double, std::string, JsonArray, JsonObject> v;
  Json(bool x) : v(x) {}
  Json(int x) : v(static_cast<std::int64_t>(x)) {}
  Json(std::int64_t x) : v(x) {}
  Json(double x) : v(x) {}
  Json(const char* x) : v(std::string(x)) {}
  Json(std::string x) : v(std::move(x)) {}
  Json(JsonArray x) : v(std::move(x)) {}
  Json(JsonObject x) : v(std::move(x)) {}
};

inline void dump_json(std::string& out, const Json& j, int indent, int level = 0) {
  auto newline = [&](int lvl) {
    out.push_back('\n');
    out.append(static_cast<std::size_t>(indent * lvl), ' ');
  };
  struct Visitor {
    std::string& out;
    int indent, level;
    void operator()(bool b) const { out.append(b ? "true" : "false"); }
    void operator()(std::int64_t i) const { append_int(out, i); }
    void operator()(double d) const { append_double(out, d); }
    void operator()(const std::string& s) const { append_json_string(out, s); }
    void operator()(const JsonArray& a) const {
      if (a.empty()) { out.append("[]"); return; }
      out.push_back('[');
      for (std::size_t i = 0; i < a.size(); ++i) {
        if (i) out.append(indent >= 0 ? "," : ", ");
        if (indent >= 0) nl(level + 1);
        dump_json(out, a[i], indent, level + 1);
      }
      if (indent >= 0) nl(level);
      out.push_back(']');
    }
    void operator()(const JsonObject& o) const {
      if (o.empty()) { out.append("{}"); return; }
      out.push_back('{');
      for (std::size_t i = 0; i < o.size(); ++i) {
        if (i) out.append(indent >= 0 ? "," : ", ");
        if (indent >= 0) nl(level + 1);
        append_json_string(out, o[i].first);
        out.append(": ");
        dump_json(out, o[i].second, indent, level + 1);
      }
      if (indent >= 0) nl(level);
      out.push_back('}');
    }
    void nl(int lvl) const {
      out.push_back('\n');
      out.append(static_cast<std::size_t>(indent * lvl), ' ');
    }
  };
  (void)newline;
  std::visit(Visitor{out, indent, level}, j.v);
}

inline void write_json_file(const std::string& path, const Json& j, int indent) {
  std::string text;
  dump_json(text, j, indent);
  Sink sink(path);
  sink.put(text);
  sink.close();
}

}  // namespace cta
